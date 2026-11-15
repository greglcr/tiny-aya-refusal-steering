"""Pinned public benchmark snapshots and conservative lexical leakage filtering."""
import hashlib
import io
import json
from pathlib import Path
import re
import unicodedata
import urllib.request

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

XSTEST_SHA = 'd7bb5bd738c1fcbc36edd83d5e7d1b71a3e2d84d'
JBB_SHA = '909e68c01d94222b8ad2e397a017e2e12e2adb73'
CALIBRATION = Path('artifacts/global_refusal_benchmark_batched_seed42/calibration')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def canonical(text):
    return ' '.join(re.findall(r'\w+',unicodedata.normalize('NFKC',text).casefold()))


def snapshot(url,path):
    path = Path(path)
    if not path.exists():
        path.write_bytes(urllib.request.urlopen(url,timeout=120).read())
    return path.read_bytes()


def filter_leakage(frame, anchors, threshold=.9):
    """Exclude normalized equality/containment and high character similarity.

    This is a lexical screen, not a guarantee against semantic contamination.
    Match the underlying harmful goal as well as the adversarially wrapped input.
    """
    anchor_texts = anchors.prompt.tolist()
    normalized = [canonical(t) for t in anchor_texts]
    texts = frame.prompt.tolist()+frame.goal.tolist()
    vectorizer = TfidfVectorizer(analyzer='char_wb',ngram_range=(3,5),min_df=1)
    matrix = vectorizer.fit_transform(anchor_texts+texts)
    similarity = (matrix[len(anchor_texts):] @ matrix[:len(anchor_texts)].T).toarray()
    reasons = []
    for i,row in frame.reset_index(drop=True).iterrows():
        candidates = [canonical(row.prompt), canonical(row.goal)]
        reason = ''; anchor_index = None
        for j,a in enumerate(normalized):
            if any(t == a for t in candidates):reason='normalized_exact';anchor_index=j;break
            if any(min(len(a),len(t))>=40 and (a in t or t in a) for t in candidates):
                reason='normalized_containment';anchor_index=j;break
        scores = similarity[[i,i+len(frame)]].max(axis=0)
        best = int(scores.argmax())
        if not reason and scores[best]>=threshold:reason='char_tfidf_ge_0.9';anchor_index=best
        reasons.append({'excluded':bool(reason),'reason':reason,'max_char_similarity':float(scores[best]),
                        'anchor_example_id':str(anchors.iloc[anchor_index].get('example_id',anchor_index)) if anchor_index is not None else ''})
    audit = pd.concat([frame.reset_index(drop=True),pd.DataFrame(reasons)],axis=1)
    return audit.loc[~audit.excluded,frame.columns].reset_index(drop=True),audit


def prepare(out):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    snapshots=out/'sources';snapshots.mkdir(exist_ok=True)
    manifest=json.loads((CALIBRATION/'manifest.json').read_text())
    if sha(CALIBRATION/'splits.csv') != manifest['split_sha256']:raise ValueError('Changed calibration splits.')
    csv_url=f'https://raw.githubusercontent.com/paul-rottger/xstest/{XSTEST_SHA}/xstest_prompts.csv'
    raw=snapshot(csv_url,snapshots/'xstest.csv')
    xs=pd.read_csv(io.BytesIO(raw),keep_default_na=False)
    print('XSTest source labels:',xs.label.value_counts().to_dict(),flush=True)
    rows=[]
    for r in xs.to_dict('records'):
        label='harmful' if str(r['type']).startswith('contrast_') else 'safe'
        rows.append({'example_id':f"xstest_{r['id']}",'benchmark':'xstest','family':str(r['type']),
                     'prompt':r['prompt'],'goal':r['prompt'],'label':label,'category':str(r['focus']),
                     'source_label':str(r['label']),'source_model':'','source_url':csv_url})
    source_info=[{'url':csv_url,'sha256':sha(snapshots/'xstest.csv'),'revision':XSTEST_SHA,
                  'license':'CC-BY-4.0; cite Rottger et al., NAACL 2024'}]
    for family,path in [('GCG','attack-artifacts/GCG/white_box/llama-2-7b-chat-hf.json'),
                        ('PAIR','attack-artifacts/PAIR/black_box/vicuna-13b-v1.5.json')]:
        url=f'https://raw.githubusercontent.com/JailbreakBench/artifacts/{JBB_SHA}/{path}'
        file=snapshots/f'jbb_{family}_{Path(path).stem}.json';data=json.loads(snapshot(url,file))
        missing=0
        for r in data['jailbreaks']:
            if not isinstance(r.get('prompt'),str) or not r['prompt'].strip():missing+=1;continue
            rows.append({'example_id':f"jbb_{family}_{r['index']}",'benchmark':'jbb_transfer','family':family,
                         'prompt':r['prompt'],'goal':r['goal'],'label':'harmful','category':r['category'],
                         'source_label':'harmful','source_model':Path(path).stem,'source_url':url})
        source_info.append({'url':url,'sha256':sha(file),'revision':JBB_SHA,'missing_prompts':missing,
                            'note':'Transferred released artifacts; source success flags are NOT Tiny Aya outcomes. Check upstream licensing before redistribution.'})
    frame=pd.DataFrame(rows)
    # Keep the first duplicate input only; preserve a duplicate audit.
    duplicates=frame.loc[frame.prompt.map(canonical).duplicated(keep=False)]
    duplicates.to_csv(out/'duplicate_inputs.csv',index=False)
    frame=frame.loc[~frame.prompt.map(canonical).duplicated()].reset_index(drop=True)
    splits=pd.read_csv(CALIBRATION/'splits.csv',keep_default_na=False)
    # Screen against every original prompt, including previously inspected test.
    clean,audit=filter_leakage(frame,splits)
    audit.to_csv(out/'leakage_audit.csv',index=False)
    clean['cluster_id']=clean.goal.map(lambda s:hashlib.sha256(canonical(s).encode()).hexdigest()[:20])
    clean=clean.sort_values(['benchmark','family','example_id']).reset_index(drop=True)
    clean.to_csv(out/'benchmark_prompts.csv',index=False)
    metadata={'sources':source_info,'calibration_sha256':sha(CALIBRATION/'manifest.json'),
              'source_candidates':len(frame),'excluded_leakage':int(audit.excluded.sum()),
              'duplicates':len(duplicates),'remaining':len(clean),
              'counts':clean.groupby(['benchmark','label']).size().to_dict().__repr__(),
              'screen':'normalized equality/containment plus char_wb TF-IDF >= .9 versus all original prompts; goal and attack both checked',
              'limitations':'Not semantic decontamination; JBB goals may overlap across attack families, clustered in analysis. PAIR null prompts omitted, not imputed.'}
    (out/'data_manifest.json').write_text(json.dumps(metadata,indent=2)+'\n')
    print(clean.groupby(['benchmark','family','label']).size().to_string(),flush=True)
    print('Excluded lexical overlaps:',metadata['excluded_leakage'],flush=True)
    return clean


if __name__=='__main__':
    import sys
    prepare(sys.argv[1] if len(sys.argv)>1 else 'artifacts/safety_campaign_v1')
