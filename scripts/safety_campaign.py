"""Reproducible SafeTune -> AuditKIT benchmark campaign. No external LLM API.

Thirteen preregistered conditions: baseline, 3 variants x 3 budgets and
3 random shared directions at the middle budget. No tuning on these benchmarks.
"""
from contextlib import nullcontext
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import time

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from safetune.interventions.steer.caa import CAAModel

from scripts.campaign_data import CALIBRATION, sha
from scripts.layer_steering import VARIANTS, steering_wrapper, unit_vectors
from scripts.global_refusal import check_model, generate_responses, resource_report
from scripts.auditkit_bridge import evaluate_batch


def conditions():
    return [{'name':'baseline','variant':'baseline','factor':0.,'seed':None}] + [
        {'name':f'{v}_f{f:g}','variant':v,'factor':f,'seed':None}
        for f in (.25,.5,1.) for v in VARIANTS] + [
        {'name':f'random_shared_s{seed}_f0.5','variant':'random_shared','factor':.5,'seed':seed}
        for seed in (0,1,2)]


def context_for(model,raw,layer,condition):
    budget=float(raw[layer].float().norm())*condition['factor']
    if condition['variant']=='baseline':return nullcontext()
    if condition['variant']=='random_shared':
        d=torch.randn(raw[layer].shape,generator=torch.Generator().manual_seed(condition['seed']))
        d=d/d.norm()
        return CAAModel(model,{l:d*budget/len(raw) for l in raw},strength=1.)
    return steering_wrapper(model,raw,layer,condition['variant'],budget)


def load_jsonl(path):
    path=Path(path)
    if not path.exists():return []
    records=[]
    for i,line in enumerate(path.read_text().splitlines()):
        try:records.append(json.loads(line))
        except json.JSONDecodeError as exc:raise ValueError(f'Broken JSONL line {i+1} in {path}') from exc
    return records


def append_jsonl(path,records):
    import os
    with Path(path).open('a',encoding='utf-8') as handle:
        for row in records:handle.write(json.dumps(row,ensure_ascii=False)+'\n')
        handle.flush();os.fsync(handle.fileno())


def write_status(out,stage,**values):
    doc={'stage':stage,'updated_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),**values}
    temp=Path(out)/'status.tmp';temp.write_text(json.dumps(doc,indent=2)+'\n')
    temp.replace(Path(out)/'status.json')


def generate(out,stop_epoch=None,limit=None):
    out=Path(out);data=pd.read_csv(out/'benchmark_prompts.csv',keep_default_na=False)
    if limit is not None:data=data.iloc[:limit].copy()
    source=json.loads((CALIBRATION/'manifest.json').read_text())
    raw=torch.load(CALIBRATION/'raw_directions.pt',map_location='cpu',weights_only=True)
    layer=int(source['selection']['layer']);unit_vectors(raw)
    if sha(CALIBRATION/'selected_direction.pt') != source['direction_sha256']:raise ValueError('Direction changed.')
    config={'model':source['model'],'max_new_tokens':512,'batch_size':8,'conditions':conditions(),
            'budget':'sum_layer_l2_norms; per-layer budget B/4; B=factor*norm(raw_layer_14)',
            'dataset_sha256':sha(out/'benchmark_prompts.csv'),'limit':limit,
            'raw_directions_sha256':sha(CALIBRATION/'raw_directions.pt'),
            'data_manifest_sha256':sha(out/'data_manifest.json'),
            'versions':{p:importlib.metadata.version(p) for p in ('safetune','auditkit','torch','transformers')},
            'code_sha256':{p:sha(Path('scripts')/p) for p in ('safety_campaign.py','auditkit_bridge.py','global_refusal.py','layer_steering.py','campaign_data.py','campaign_judge.py','campaign_report.py','run_campaign.py')},
            'interpretation':'Fixed external benchmark curves; not used to select hyperparameters. Random controls at factor .5 only.'}
    manifest_path=out/'campaign_manifest.json'
    if manifest_path.exists() and json.loads(manifest_path.read_text()) != config:raise ValueError('Campaign fingerprint mismatch; use new output directory.')
    manifest_path.write_text(json.dumps(config,indent=2,ensure_ascii=False)+'\n')
    if not torch.cuda.is_available():raise RuntimeError('CUDA is required.')
    if torch.cuda.mem_get_info()[0]<10*1024**3:raise RuntimeError('Need 10 GiB free VRAM.')
    torch.set_num_threads(4)
    torch.manual_seed(42)
    tokenizer=AutoTokenizer.from_pretrained(config['model']['model_id'],revision=config['model']['revision'],local_files_only=True)
    if tokenizer.chat_template!=config['model']['chat_template']:raise ValueError('Template mismatch.')
    tokenizer.pad_token_id=tokenizer.eos_token_id
    # Determine all exclusions before the first generation; never truncate attacks.
    lengths=[len(tokenizer.apply_chat_template([{'role':'user','content':p}],tokenize=True,add_generation_prompt=True)) for p in data.prompt]
    data=data.assign(input_tokens=lengths)
    data.loc[data.input_tokens>4096].to_csv(out/'length_exclusions.csv',index=False)
    data=data.loc[data.input_tokens<=4096].reset_index(drop=True)
    # Stable length order improves padding efficiency; identical batch boundaries for every condition.
    data=data.sort_values(['input_tokens','example_id']).reset_index(drop=True)
    data.to_csv(out/'evaluation_order.csv',index=False)
    model=AutoModelForCausalLM.from_pretrained(config['model']['model_id'],revision=config['model']['revision'],
                                             dtype='auto',device_map={'':'cuda:0'},local_files_only=True).eval()
    print('Resources:',resource_report(model),flush=True)
    responses_path=out/'responses.jsonl'
    saved=load_jsonl(responses_path)
    lookup={(r['condition'],r['example_id']):r for r in saved}
    if len(lookup)!=len(saved):raise ValueError('Duplicate saved response key.')
    start=time.time();total=len(data)*len(conditions())
    for condition in conditions():
        for offset in range(0,len(data),8):
            if stop_epoch and time.time()>stop_epoch:
                write_status(out,'generation_budget_stop',completed=len(lookup),planned=total)
                return
            batch=data.iloc[offset:offset+8];rows=batch.to_dict('records')
            keys=[(condition['name'],r['example_id']) for r in rows]
            hits=[k in lookup for k in keys]
            if all(hits):continue
            if any(hits):raise ValueError('Partial batch in saved results; preserve batch boundaries before resuming.')
            generated=[]
            def callback(prompts):
                if prompts!=batch.prompt.tolist():raise ValueError('AuditKIT changed request order.')
                before=time.perf_counter()
                with context_for(model,raw,layer,condition):
                    answers=generate_responses(model,tokenizer,batch,512,8)
                check_model(model)
                elapsed=time.perf_counter()-before
                for record in answers.to_dict('records'):
                    record.update(condition=condition['name'],variant=condition['variant'],factor=condition['factor'],
                                  random_seed=condition['seed'],batch_seconds=elapsed,batch_offset=offset,
                                  total_budget=float(raw[layer].float().norm())*condition['factor'])
                    generated.append(record)
                return answers.response.tolist()
            identity=hashlib.sha256(json.dumps(config,sort_keys=True).encode()).hexdigest()+':'+condition['name']
            result=evaluate_batch(rows,callback,identity,max_tokens=512)
            if len(generated)!=len(rows):raise RuntimeError('Missing generator metadata.')
            for record,prediction in zip(generated,result.predictions):
                if prediction.raw_output!=record['response']:raise RuntimeError('AuditKIT output mismatch.')
                for score in prediction.metadata['scores']:
                    record[score['name']]=None if score.get('metadata',{}).get('unknown') else score['value']
                record['auditkit_run_id']=result.run_id;record['auditkit_fingerprint']=result.fingerprint
            batchdir=out/'auditkit'/condition['name']/f'batch_{offset:05d}';batchdir.mkdir(parents=True,exist_ok=True)
            result.save(str(batchdir/'run.json'))
            append_jsonl(responses_path,generated)
            lookup.update({(r['condition'],r['example_id']):r for r in generated})
            write_status(out,'generation',condition=condition['name'],completed=len(lookup),planned=total,
                         elapsed_seconds=time.time()-start)
            print(f"GEN {condition['name']} {min(offset+8,len(data))}/{len(data)}; all {len(lookup)}/{total}",flush=True)
    write_status(out,'generation_complete',completed=len(lookup),planned=total)
    (out/'GENERATION_COMPLETE.json').write_text(json.dumps({'responses':len(lookup),'prompts':len(data),'conditions':len(conditions())},indent=2)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',default='artifacts/safety_campaign_v1')
    p.add_argument('--stop-epoch',type=float);p.add_argument('--limit',type=int)
    a=p.parse_args();generate(a.out,a.stop_epoch,a.limit)
