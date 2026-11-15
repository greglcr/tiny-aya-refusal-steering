"""Small validation experiment: global versus layer-specific additive steering.

Reuses train-fitted SafeTune vectors; never modifies weights. All variants use
block-output addition at every token. Budget B means sum_l ||delta_l||_2 = B,
not equality of the resulting residual changes or of model behavior.
"""
from dataclasses import asdict, dataclass
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from safetune.interventions.steer.caa import CAAModel
from safetune.interventions.steer.refusal_direction import RefusalDirectionModel

from scripts.global_refusal import check_model, generate_responses, resource_report
from scripts.refusal_eval import evaluate_refusals, JUDGE_VERSION

VARIANTS = ('single_layer', 'shared_direction', 'layer_specific')


@dataclass(frozen=True)
class LayerConfig:
    seed: int = 42
    validation_per_class: int = 8
    budget_factors: tuple = (0.25, 0.5, 1.0)
    max_new_tokens: int = 256
    batch_size: int = 8


def unit_vectors(raw):
    if not raw:
        raise ValueError('No direction vectors.')
    result = {}
    for layer, vector in sorted(raw.items()):
        vector = torch.as_tensor(vector).detach().float().cpu()
        if vector.ndim != 1 or not torch.isfinite(vector).all() or vector.norm() <= 1e-8:
            raise ValueError(f'Invalid vector at layer {layer}.')
        result[int(layer)] = vector/vector.norm()
    if len({v.shape for v in result.values()}) != 1:
        raise ValueError('Vector dimensions differ.')
    return result


def geometry(raw):
    units = unit_vectors(raw)
    layers = list(units)
    matrix = torch.stack(list(units.values()))
    cosine = pd.DataFrame((matrix @ matrix.T).numpy(), index=layers, columns=layers)
    cosine.index.name = 'layer'
    norms = pd.DataFrame([{'layer': l, 'raw_norm': float(torch.as_tensor(raw[l]).float().norm())}
                          for l in layers])
    return cosine, norms


def intervention_vectors(raw, selected_layer, variant, budget):
    """Return actual per-layer additive perturbations with an L1 sum of L2 norms."""
    units = unit_vectors(raw)
    if selected_layer not in units or variant not in VARIANTS:
        raise ValueError('Unknown selected layer or variant.')
    if not torch.isfinite(torch.tensor(budget)) or budget < 0:
        raise ValueError('Budget must be finite and nonnegative.')
    if variant == 'single_layer':
        return {selected_layer: budget*units[selected_layer]}
    count = len(units)
    return {layer: (budget/count)*(units[selected_layer] if variant == 'shared_direction' else unit)
            for layer, unit in units.items()}


def steering_wrapper(model, raw, selected_layer, variant, budget):
    vectors = intervention_vectors(raw, selected_layer, variant, budget)
    if variant == 'single_layer':
        # RefusalDirectionModel normalizes its vector; strength supplies its norm.
        return RefusalDirectionModel(model, unit_vectors(raw)[selected_layer],
                                     mode='steer', layers=[selected_layer], strength=budget)
    # CAAModel preserves per-layer vector magnitudes. Strength 1 applies the
    # already-budgeted vectors, preventing an extra normalization/multiplier.
    return CAAModel(model, vectors, strength=1.0)


def validation_subset(splits, config):
    if splits.prompt.duplicated().any():
        raise ValueError('Prompt leakage in saved splits.')
    if config.validation_per_class < 1:
        raise ValueError('Positive validation size required.')
    val = splits.loc[splits.split.eq('validation')]
    groups = []
    for label in ('safe', 'harmful'):
        group = val.loc[val.label.eq(label)]
        if len(group) < config.validation_per_class:
            raise ValueError('Not enough validation examples.')
        groups.append(group.sample(n=config.validation_per_class, random_state=config.seed))
    return pd.concat(groups, ignore_index=True).sample(frac=1, random_state=config.seed).reset_index(drop=True)


def summarize(answers, condition, factor, budget):
    annotated, summary = evaluate_refusals(answers)
    truncation = answers.groupby('label')['hit_token_limit'].sum()
    summary['hit_token_limit'] = summary.label.map(truncation).astype(int)
    summary['condition'] = condition
    summary['budget_factor'] = factor
    summary['total_budget'] = budget
    return annotated, summary


def plot_results(cosine, summary, out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    out = Path(out)
    fig, ax = plt.subplots(figsize=(5.3, 4.4))
    im = ax.imshow(cosine.values, vmin=-1, vmax=1, cmap='coolwarm')
    labels = [str(l) for l in cosine.index]
    ax.set_xticks(range(len(labels)), labels); ax.set_yticks(range(len(labels)), labels)
    ax.set_xlabel('Decoder block output (zero-based)'); ax.set_ylabel('Decoder block output (zero-based)')
    ax.set_title('Cosine similarity of train-fitted directions')
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, f'{cosine.iloc[i,j]:.2f}', ha='center', va='center', color='black')
    fig.colorbar(im, ax=ax, label='Cosine similarity')
    fig.tight_layout(); fig.savefig(out/'direction_cosines.png', dpi=160); plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    names = {'single_layer':'Single layer', 'shared_direction':'Shared direction', 'layer_specific':'Layer-specific directions'}
    for ax, label, title in zip(axes, ('safe','harmful'), ('Legitimate prompts','Harmful prompts')):
        baseline = summary.loc[summary.condition.eq('baseline') & summary.label.eq(label), 'refusal_rate'].iloc[0]
        ax.axhline(baseline*100, color='gray', linestyle='--', label='Unmodified baseline')
        for variant in VARIANTS:
            frame = summary.loc[summary.condition.eq(variant) & summary.label.eq(label)].sort_values('budget_factor')
            ax.plot(frame.budget_factor, frame.refusal_rate*100, marker='o', label=names[variant])
        ax.set_xlabel('Total budget / norm of raw selected direction')
        ax.set_title(title); ax.set_ylim(-3,103); ax.grid(alpha=.2)
    axes[0].set_ylabel('Lexical refusal rate (%)')
    axes[1].legend(fontsize=8, loc='best')
    fig.suptitle('Validation pilot: 8 prompts per class; matched sum of injected norms')
    fig.tight_layout(); fig.savefig(out/'refusal_by_budget.png', dpi=160); plt.close(fig)


def run(calibration_dir, output_dir, config=LayerConfig()):
    calibration = Path(calibration_dir); out = Path(output_dir)
    original_manifest = json.loads((calibration/'manifest.json').read_text())
    model_config = original_manifest['model']
    selected_layer = int(original_manifest['selection']['layer'])
    raw = torch.load(calibration/'raw_directions.pt', map_location='cpu', weights_only=True)
    units = unit_vectors(raw)
    selected = torch.load(calibration/'selected_direction.pt', map_location='cpu', weights_only=True)
    torch.testing.assert_close(units[selected_layer], selected.float(), atol=1e-6, rtol=1e-5)
    if hashlib.sha256((calibration/'selected_direction.pt').read_bytes()).hexdigest() != original_manifest['direction_sha256']:
        raise ValueError('Selected direction fingerprint mismatch.')
    if hashlib.sha256((calibration/'splits.csv').read_bytes()).hexdigest() != original_manifest['split_sha256']:
        raise ValueError('Split fingerprint mismatch.')
    splits = pd.read_csv(calibration/'splits.csv', keep_default_na=False)
    prompts = validation_subset(splits, config)
    scale = float(raw[selected_layer].float().norm())
    metadata = {'config':asdict(config), 'selected_layer':selected_layer, 'layers':list(units),
                'model_id':model_config['model_id'], 'revision':model_config['revision'],
                'raw_directions_sha256':hashlib.sha256((calibration/'raw_directions.pt').read_bytes()).hexdigest(),
                'calibration_manifest_sha256':hashlib.sha256((calibration/'manifest.json').read_bytes()).hexdigest(),
                'subset_sha256':hashlib.sha256(prompts.to_csv(index=False).encode()).hexdigest(),
                'reference_raw_norm':scale, 'budget_convention':'sum_layer_l2_norms',
                'apply_at':'decoder_block_output_all_tokens', 'judge':JUDGE_VERSION,
                'versions':{p:importlib.metadata.version(p) for p in ('safetune','transformers','torch')},
                'code_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                'scope':'validation_only_exploratory; selected layer previously chosen by ablation'}
    metadata = json.loads(json.dumps(metadata))
    out.mkdir(parents=True, exist_ok=True)
    manifest_path = out/'manifest.json'
    if manifest_path.exists() and json.loads(manifest_path.read_text()) != metadata:
        raise ValueError('Existing experiment has different provenance. Use a new output directory.')
    manifest_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False)+'\n')
    prompts.to_csv(out/'validation_subset.csv', index=False)
    cosine, norms = geometry(raw)
    cosine.to_csv(out/'direction_cosines.csv'); norms.to_csv(out/'direction_norms.csv', index=False)
    print('Cosines:\n', cosine.to_string(), flush=True)
    print('Reference norm:', scale, 'budgets:', [f*scale for f in config.budget_factors], flush=True)
    if (out/'COMPLETE.json').exists():
        summary = pd.read_csv(out/'summary.csv')
        if len(summary) != 2*(1+len(VARIANTS)*len(config.budget_factors)):
            raise ValueError('Incomplete saved summary.')
        plot_results(cosine, summary, out)
        return summary
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required for the Tiny Aya experiment; geometry is already saved.')
    free, _ = torch.cuda.mem_get_info()
    # Model weights + conservative generation reserve, before allocating weights.
    if free < 10*1024**3:
        raise RuntimeError('Less than 10 GiB free VRAM; wait before loading another model.')
    torch.set_num_threads(4)
    tokenizer = AutoTokenizer.from_pretrained(model_config['model_id'], revision=model_config['revision'], local_files_only=True)
    if tokenizer.chat_template != model_config['chat_template']:
        raise ValueError('Chat template mismatch.')
    tokenizer.pad_token_id = tokenizer.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(
        model_config['model_id'], revision=model_config['revision'], dtype='auto',
        device_map={'':'cuda:0'}, local_files_only=True,
    ).eval()
    print('Resources:',resource_report(model),flush=True)
    conditions = [('baseline',0.,0.)] + [(v,f,f*scale) for f in config.budget_factors for v in VARIANTS]
    summaries = []
    for condition, factor, budget in conditions:
        stem = 'baseline' if condition == 'baseline' else f'{condition}_{factor:g}'
        path = out/f'{stem}_responses.csv'
        print('Condition:',condition,'factor:',factor,'budget:',budget,flush=True)
        if path.exists():
            answers = pd.read_csv(path, keep_default_na=False)
            if answers.prompt.tolist() != prompts.prompt.tolist() or answers.label.tolist() != prompts.label.tolist():
                raise ValueError('Cached response subset mismatch.')
        elif condition == 'baseline':
            answers = generate_responses(model,tokenizer,prompts,config.max_new_tokens,config.batch_size)
        else:
            check_model(model)
            with steering_wrapper(model,raw,selected_layer,condition,budget):
                answers = generate_responses(model,tokenizer,prompts,config.max_new_tokens,config.batch_size)
            check_model(model)  # no intervention hooks may survive the context
        answers.to_csv(path,index=False)
        annotated, summary = summarize(answers,condition,factor,budget)
        annotated.to_csv(out/f'{stem}_annotations.csv',index=False)
        summaries.append(summary)
        pd.concat(summaries,ignore_index=True).to_csv(out/'summary.csv',index=False)
        print(summary.to_string(index=False),flush=True)
    summary = pd.concat(summaries,ignore_index=True)
    plot_results(cosine,summary,out)
    (out/'COMPLETE.json').write_text(json.dumps({'conditions':len(conditions),'responses':len(prompts)*len(conditions)},indent=2)+'\n')
    print('COMPLETE:',out,flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--calibration-dir',default='artifacts/global_refusal_benchmark_batched_seed42/calibration')
    parser.add_argument('--output-dir',default='artifacts/layer_steering_seed42')
    args = parser.parse_args()
    run(args.calibration_dir,args.output_dir)


if __name__ == '__main__': main()
