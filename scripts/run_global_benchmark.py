"""Calibrate SafeTune and re-evaluate the exact cached Tiny Aya benchmark.

Run from the project root: .venv/bin/python -m scripts.run_global_benchmark
Uses locally cached model files only. Does not touch the notebook's kernel.
"""
import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from scripts.global_refusal import (
    GlobalConfig, split_prompts, fit_and_select, resource_report,
    generate_responses, temporary_weight_orthogonalization,
)
from scripts.refusal_eval import evaluate_refusals


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', default='artifacts/global_refusal_benchmark_batched_seed42')
    args = parser.parse_args()
    out = Path(args.run_dir)
    out.mkdir(parents=True, exist_ok=True)
    cache = pd.read_csv('data/tiny_aya_global_responses_cache.csv', keep_default_na=False)
    if cache.config_json.nunique() != 1 or cache.prompt.duplicated().any():
        raise ValueError('Expected one original benchmark configuration with unique prompts.')
    original = json.loads(cache.config_json.iloc[0])
    if original['intervention'] != 'none' or original['do_sample']:
        raise ValueError('Expected an unmodified, greedily decoded original benchmark.')
    if not torch.cuda.is_available():
        raise RuntimeError('A CUDA GPU is required for this benchmark run.')
    torch.set_num_threads(4)
    cfg = GlobalConfig(max_new_tokens=original['max_new_tokens'], generation_batch_size=8)
    prompts = cache[[c for c in ('example_id', 'prompt', 'label', 'source') if c in cache]].copy()
    splits = split_prompts(prompts, cfg)
    if (out/'benchmark_comparison.csv').exists():
        raise FileExistsError('Benchmark already completed in this directory.')
    print('GPU:', torch.cuda.get_device_name(), 'free GiB:', torch.cuda.mem_get_info()[0]/2**30, flush=True)
    tokenizer = AutoTokenizer.from_pretrained(original['model_id'], revision=original['revision'], local_files_only=True)
    if tokenizer.chat_template != original['chat_template']:
        raise RuntimeError('Tokenizer template differs from the original benchmark.')
    # Match the original notebook's explicit generation padding token.
    tokenizer.pad_token_id = tokenizer.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(
        original['model_id'], revision=original['revision'], dtype='auto',
        device_map={'': 'cuda:0'}, local_files_only=True,
    ).eval()
    print('Resources:', resource_report(model), flush=True)
    calibration = out/'calibration'
    manifest_path = calibration/'manifest.json'
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if 'selection' not in manifest:
            raise RuntimeError('An incomplete calibration exists; inspect it or use a new run directory.')
        if manifest['model']['revision'] != original['revision'] or manifest['config'] != json.loads(json.dumps(cfg.__dict__)):
            raise ValueError('Calibration configuration mismatch.')
        if hashlib.sha256((calibration/'selected_direction.pt').read_bytes()).hexdigest() != manifest['direction_sha256']:
            raise ValueError('Direction file mismatch.')
        if hashlib.sha256(splits.to_csv(index=False).encode()).hexdigest() != manifest['split_sha256']:
            raise ValueError('Split mismatch.')
        direction = torch.load(calibration/'selected_direction.pt', map_location='cpu', weights_only=True)
    else:
        print('Starting train/validation calibration; test prompts remain excluded.', flush=True)
        direction, _ = fit_and_select(model, tokenizer, splits, calibration, cfg,
                                     model_metadata=original)
    print('Calibration complete. Recomputing matched-batch baseline.', flush=True)
    matched_baseline_path = out/'benchmark_baseline_batched.csv'
    if matched_baseline_path.exists():
        matched_baseline = pd.read_csv(matched_baseline_path, keep_default_na=False)
    else:
        matched_baseline = generate_responses(model, tokenizer, prompts, cfg.max_new_tokens, cfg.generation_batch_size)
        matched_baseline.to_csv(matched_baseline_path, index=False)
    print('Starting 256-prompt weight-orthogonalized evaluation.', flush=True)
    rows = []
    partial = out/'benchmark_orthogonalized.csv'
    if partial.exists():
        previous = pd.read_csv(partial, keep_default_na=False)
        for row in previous.to_dict('records'):
            match = prompts.loc[prompts.prompt.eq(row['prompt'])]
            if len(match) != 1 or match.iloc[0]['label'] != row['label']:
                raise ValueError('Partial benchmark mismatch.')
        if previous.prompt.duplicated().any():
            raise ValueError('Duplicate partial responses.')
        rows = previous.to_dict('records')
    completed = {row['prompt'] for row in rows}
    with temporary_weight_orthogonalization(model, direction):
        remaining = prompts.loc[~prompts.prompt.isin(completed)]
        for offset in range(0, len(remaining), cfg.generation_batch_size):
            batch = remaining.iloc[offset:offset+cfg.generation_batch_size]
            answer = generate_responses(model, tokenizer, batch, cfg.max_new_tokens, cfg.generation_batch_size)
            rows.extend(answer.to_dict('records'))
            pd.DataFrame(rows).to_csv(partial, index=False)
            print(f'Benchmark saved {len(rows)}/{len(prompts)}', flush=True)
    print('Original weights restored.', flush=True)
    changed = pd.DataFrame(rows)
    # Align on prompt instead of relying on generation/cache order.
    changed = prompts[['prompt']].merge(changed, on='prompt', validate='one_to_one')
    changed = changed.merge(splits[['prompt', 'split']], on='prompt', validate='one_to_one')
    baseline = matched_baseline.merge(splits[['prompt', 'split']], on='prompt', validate='one_to_one')
    cached_baseline = cache.merge(splits[['prompt', 'split']], on='prompt', validate='one_to_one')
    summaries = []
    for scope in ('all_exploratory', 'test'):
        for condition, frame in (('original_cached_baseline', cached_baseline), ('baseline', baseline), ('weight_orthogonalized', changed)):
            subset = frame if scope == 'all_exploratory' else frame.loc[frame.split.eq('test')]
            annotated, summary = evaluate_refusals(subset)
            annotated.to_csv(out/f'{scope}_{condition}_annotations.csv', index=False)
            summaries.append(summary.assign(scope=scope, condition=condition))
    comparison = pd.concat(summaries, ignore_index=True)
    comparison.to_csv(out/'benchmark_comparison.csv', index=False)
    refusal_changes = baseline[['prompt','label','response']].rename(columns={'response':'baseline_response'}).merge(
        changed[['prompt','response','split','generated_tokens','hit_token_limit']].rename(columns={'response':'modified_response'}),
        on='prompt', validate='one_to_one')
    refusal_changes.to_csv(out/'paired_responses.csv', index=False)
    (out/'benchmark_manifest.json').write_text(json.dumps({
        'baseline_config': original, 'generation_batch_size': cfg.generation_batch_size, 'baseline_cache_sha256': hashlib.sha256(Path('data/tiny_aya_global_responses_cache.csv').read_bytes()).hexdigest(),
        'calibration_directory': str(calibration), 'changed_prompt_count': len(changed),
        'note': 'Full benchmark exploratory: includes train/validation. Report test separately. Original cached baseline has no truncation metadata. Primary baseline is regenerated with the same batch size as the modified model.',
    }, ensure_ascii=False, indent=2)+'\n')
    print(comparison.to_string(index=False), flush=True)
    print('Modified responses hitting token limit:', int(changed.hit_token_limit.eq(True).sum()), flush=True)


if __name__ == '__main__':
    main()
