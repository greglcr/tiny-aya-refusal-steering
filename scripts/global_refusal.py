"""Global refusal direction baseline using SafeTune's installed implementation.

Arditi et al. https://arxiv.org/abs/2406.11717
Layer indices denote decoder OUTPUTS, not the block inputs in the original code.
Only the last chat-template token and an explicit subset of layers are searched.
"""
from contextlib import contextmanager
from dataclasses import asdict, dataclass
import hashlib
import importlib.metadata
import json
from pathlib import Path

import pandas as pd
import torch
from tqdm.auto import tqdm
from safetune.core.runtime.inference.vector_extraction import (
    SteeringVectorExtractor, VectorExtractionConfig,
)
from safetune.interventions.steer.refusal_direction import (
    RefusalDirectionModel, orthogonalize_weights, restore_weights,
)
from scripts.refusal_eval import evaluate_refusals


@dataclass(frozen=True)
class GlobalConfig:
    seed: int = 42
    train_per_class: int = 64
    val_per_class: int = 32
    # Explicit small layer search; expand only after a successful smoke run.
    layer_fractions: tuple = (0.25, 0.4, 0.55, 0.7)
    batch_size: int = 1
    max_new_tokens: int = 256
    generation_batch_size: int = 1
    kl_threshold: float = 0.1


def split_prompts(data, config=GlobalConfig()):
    """Deterministic stratified train/validation/test split by unique prompt."""
    if not {'prompt', 'label'}.issubset(data):
        raise ValueError('Missing prompt/label columns.')
    if data['prompt'].isna().any() or data['prompt'].duplicated().any():
        raise ValueError('Prompts must be non-missing and unique across classes.')
    if not data['prompt'].map(lambda p: isinstance(p, str) and bool(p.strip())).all():
        raise ValueError('Prompts must be nonempty strings.')
    if set(data['label']) != {'safe', 'harmful'}:
        raise ValueError('Expected both safe and harmful labels.')
    if min(config.train_per_class, config.val_per_class) < 1:
        raise ValueError('Positive train and validation sizes required.')
    parts = []
    for label in ('safe', 'harmful'):
        group = data.loc[data.label.eq(label)].sample(frac=1, random_state=config.seed).copy()
        a, b = config.train_per_class, config.train_per_class + config.val_per_class
        if len(group) <= b:
            raise ValueError(f'{label}: need more than {b} distinct examples for held-out test.')
        group['split'] = ['train'] * a + ['validation'] * (b-a) + ['test'] * (len(group)-b)
        parts.append(group)
    return pd.concat(parts, ignore_index=True)


def decoder_layers(model):
    # Tiny Aya / Cohere2 architecture, checked explicitly rather than guessed.
    layers = getattr(getattr(model, 'model', None), 'layers', None)
    if layers is None or not len(layers):
        raise ValueError('Expected model.model.layers (Tiny Aya/Cohere2).')
    return layers


def check_model(model):
    layers = decoder_layers(model)
    if getattr(model, 'is_quantized', False):
        raise ValueError('Weight editing requires an unquantized model.')
    devices = {p.device for p in model.parameters()}
    if len(devices) != 1 or next(iter(devices)).type == 'meta':
        raise ValueError('Use a single-device model without CPU/disk offloading.')
    for layer in layers:
        for module in (layer.self_attn.o_proj, layer.mlp.down_proj):
            if not isinstance(module, torch.nn.Linear) or module.bias is not None:
                raise ValueError('Expected bias-free attention/MLP output projections.')
    if any(m._forward_hooks or m._forward_pre_hooks for m in model.modules()):
        raise ValueError('Remove existing hooks before calibration or weight editing.')
    return len(layers)


def resource_report(model):
    """Inspect actual resources without launching an experiment."""
    count = check_model(model)
    device = next(model.parameters()).device
    report = {'device': str(device), 'layers': count,
              'parameter_gib': sum(p.numel()*p.element_size() for p in model.parameters())/2**30}
    if device.type == 'cuda':
        free, total = torch.cuda.mem_get_info(device)
        report.update(free_gib=free/2**30, total_gib=total/2**30,
                      gpu=torch.cuda.get_device_name(device))
    return report


def _inputs(model, tokenizer, prompt):
    return tokenizer.apply_chat_template(
        [{'role': 'user', 'content': prompt}], tokenize=True,
        add_generation_prompt=True, return_tensors='pt', return_dict=True,
    ).to(next(model.parameters()).device)


@torch.inference_mode()
def generate_responses(model, tokenizer, prompts, max_new_tokens=256, batch_size=1):
    """Greedy generation, left-padded batches and per-response stop metadata."""
    if batch_size < 1:
        raise ValueError('batch_size must be positive.')
    rows = []
    model.eval()
    records = prompts.to_dict('records')
    previous_side = tokenizer.padding_side
    tokenizer.padding_side = 'left'
    try:
        for offset in tqdm(range(0, len(records), batch_size), desc='Génération', unit='batch'):
            batch = records[offset:offset+batch_size]
            texts = [tokenizer.apply_chat_template(
                [{'role': 'user', 'content': row['prompt']}],
                tokenize=False, add_generation_prompt=True,
            ) for row in batch]
            inputs = tokenizer(texts, add_special_tokens=False, padding=True, return_tensors='pt')
            inputs = inputs.to(next(model.parameters()).device)
            kwargs = dict(max_new_tokens=max_new_tokens, do_sample=False)
            if tokenizer.pad_token_id is not None:
                kwargs['pad_token_id'] = tokenizer.pad_token_id
            generated = model.generate(**inputs, **kwargs)[:, inputs['input_ids'].shape[-1]:]
            eos = model.generation_config.eos_token_id
            eos = set(eos if isinstance(eos, (tuple, list)) else [eos])
            for row, tokens in zip(batch, generated):
                stop = next((i+1 for i,t in enumerate(tokens.tolist()) if t in eos), None)
                tokens = tokens[:stop] if stop is not None else tokens
                rows.append({**row, 'response': tokenizer.decode(tokens, skip_special_tokens=True).strip(),
                             'generated_tokens': len(tokens),
                             'hit_token_limit': len(tokens) >= max_new_tokens and stop is None})
    finally:
        tokenizer.padding_side = previous_side
    return pd.DataFrame(rows)


@torch.inference_mode()
def _last_logits(model, tokenizer, prompts):
    return [model(**_inputs(model, tokenizer, p), use_cache=False).logits[0, -1].float().cpu()
            for p in prompts]


def fit_and_select(model, tokenizer, splits, output_dir, config=GlobalConfig(), model_metadata=None):
    """Fit train means with SafeTune; select on validation, never test.

    Minimise harmful validation refusal under ablation with KL(clean||ablated)
    < threshold and nonnegative benign refusal increase under raw-vector addition.
    KL uses next-token distributions; this is not a general utility evaluation.
    No middle-layer or relaxed-KL fallback: report failure if none qualifies.
    """
    n_layers = check_model(model)
    if splits['prompt'].duplicated().any():
        raise ValueError('Split leakage: duplicated prompt text.')
    if not set(splits['split']).issubset({'train', 'validation', 'test'}):
        raise ValueError('Unknown split.')
    target = sorted({int(n_layers*f) for f in config.layer_fractions})
    if not target or any(l < 0 or l >= int(.8*n_layers) for l in target):
        raise ValueError('Candidate layers must be within the first 80% of decoder blocks.')
    train = splits.loc[splits.split.eq('train')]
    val = splits.loc[splits.split.eq('validation')]
    for frame in (train, val):
        if set(frame.label) != {'safe', 'harmful'}:
            raise ValueError('Both labels required in train and validation.')
    out = Path(output_dir); out.mkdir(parents=True, exist_ok=True)
    if (out/'manifest.json').exists():
        raise FileExistsError('Use a new run directory to preserve previous experiment artifacts.')
    model.eval()
    extractor = SteeringVectorExtractor(model, tokenizer, VectorExtractionConfig(
        target_layers=target, batch_size=config.batch_size,
        pool_method='last_token', normalize=False,
    ))
    # SafeTune computes mean(safe_prompts)-mean(unsafe_prompts). Here names
    # are deliberately inverted so positive direction points toward refusal.
    raw = extractor.extract(
        safe_prompts=train.loc[train.label.eq('harmful'), 'prompt'].tolist(),
        unsafe_prompts=train.loc[train.label.eq('safe'), 'prompt'].tolist(),
    )
    if set(raw) != set(target):
        raise RuntimeError('SafeTune did not capture all requested layers.')
    vectors = {}
    for layer, vec in raw.items():
        vec = vec.detach().float().cpu()
        if not torch.isfinite(vec).all() or vec.norm() <= 1e-8:
            raise ValueError(f'Invalid direction at layer {layer}.')
        vectors[layer] = vec
    torch.save(vectors, out/'raw_directions.pt')
    splits.to_csv(out/'splits.csv', index=False)
    metadata = dict(config=asdict(config), model=model_metadata or {},
                    versions={p: importlib.metadata.version(p) for p in ('safetune', 'transformers', 'torch')},
                    code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                    convention='decoder_output_last_chat_token; harmful_mean-minus-safe_mean',
                    split_sha256=hashlib.sha256(splits.to_csv(index=False).encode()).hexdigest(),
                    search='layer_subset_only; next_token_KL; no_fallback')
    (out/'manifest.json').write_text(json.dumps(metadata, indent=2, ensure_ascii=False)+'\n')
    baseline = generate_responses(model, tokenizer, val, config.max_new_tokens, config.generation_batch_size)
    baseline.to_csv(out/'validation_baseline.csv', index=False)
    if baseline.response.str.strip().eq('').any():
        raise RuntimeError('Empty validation baseline responses; inspect before selecting a direction.')
    _, base_summary = evaluate_refusals(baseline)
    benign = val.loc[val.label.eq('safe')]
    clean_logits = _last_logits(model, tokenizer, benign.prompt)
    scores = []
    for layer, vec in vectors.items():
        unit = vec/vec.norm()
        with RefusalDirectionModel(model, unit, mode='ablate'):
            ablated = generate_responses(model, tokenizer, val, config.max_new_tokens, config.generation_batch_size)
            changed = _last_logits(model, tokenizer, benign.prompt)
        ablated.to_csv(out/f'validation_ablate_layer_{layer}.csv', index=False)
        # Paper-style induction: add the UNNORMALIZED mean difference at ONE layer.
        with RefusalDirectionModel(model, unit, mode='steer', layers=[layer], strength=float(vec.norm())):
            induced = generate_responses(model, tokenizer, benign, config.max_new_tokens, config.generation_batch_size)
        induced.to_csv(out/f'validation_induce_layer_{layer}.csv', index=False)
        _, ab_summary = evaluate_refusals(ablated)
        _, in_summary = evaluate_refusals(induced)
        kl = sum(float(torch.nn.functional.kl_div(
            q.log_softmax(-1), p.log_softmax(-1), log_target=True, reduction='sum'))
            for p,q in zip(clean_logits, changed))/len(clean_logits)
        bypass = float(ab_summary.set_index('label').loc['harmful', 'refusal_rate'])
        induce = float(in_summary.set_index('label').loc['safe', 'refusal_rate']
                       - base_summary.set_index('label').loc['safe', 'refusal_rate'])
        valid = not (ablated.response.str.strip().eq('').any() or induced.response.str.strip().eq('').any())
        baseline_harmful = float(base_summary.set_index('label').loc['harmful', 'refusal_rate'])
        scores.append(dict(layer=layer, raw_norm=float(vec.norm()), bypass_refusal_rate=bypass,
                           baseline_harmful_refusal_rate=baseline_harmful,
                           refusal_reduction=baseline_harmful-bypass,
                           kl_clean_ablated=kl, induce_refusal_increase=induce,
                           eligible=bool(valid and kl < config.kl_threshold and induce >= 0)))
        pd.DataFrame(scores).to_csv(out/'selection_scores.csv', index=False)
        print(scores[-1], flush=True)
    eligible = [s for s in scores if s['eligible']]
    if not eligible:
        raise RuntimeError('No candidate passes validation. Inspect selection_scores.csv; no fallback applied.')
    best = min(eligible, key=lambda s: (s['bypass_refusal_rate'], s['layer']))
    selected = vectors[best['layer']]; selected = selected/selected.norm()
    torch.save(selected, out/'selected_direction.pt')
    metadata['selection'] = best
    metadata['direction_sha256'] = hashlib.sha256((out/'selected_direction.pt').read_bytes()).hexdigest()
    (out/'manifest.json').write_text(json.dumps(metadata, indent=2, ensure_ascii=False)+'\n')
    return selected, pd.DataFrame(scores)


@contextmanager
def temporary_weight_orthogonalization(model, direction):
    """Edit only for this scope, restoring exact snapshots even on exceptions.

    SafeTune snapshots live on the model device. Reserve memory before calling.
    Does not write a modified model checkpoint to disk.
    """
    check_model(model)
    direction = torch.as_tensor(direction).detach().float()
    hidden = model.get_input_embeddings().weight.shape[1]
    if direction.shape != (hidden,) or not torch.isfinite(direction).all() or direction.norm() <= 1e-8:
        raise ValueError('Direction must be finite, nonzero and match hidden size.')
    # Keep our CPU backup before calling the mutating library: it also covers
    # partial failures (SafeTune may raise before returning its own snapshots).
    backup = {'embed_tokens': model.get_input_embeddings().weight.detach().cpu().clone()}
    for li, layer in enumerate(decoder_layers(model)):
        for name, module in layer.named_modules():
            if isinstance(module, torch.nn.Linear) and name.endswith(('o_proj', 'down_proj')):
                backup[f'layer{li}.{name}'] = module.weight.detach().cpu().clone()
    device = next(model.parameters()).device
    if device.type == 'cuda':
        needed = sum(t.numel()*t.element_size() for t in backup.values()) + 2*1024**3
        free, _ = torch.cuda.mem_get_info(device)
        if free < needed:
            raise RuntimeError('Insufficient free VRAM for SafeTune snapshots and workspace.')
    snapshots = None
    try:
        snapshots = orthogonalize_weights(model, direction)
        if set(snapshots) != set(backup):
            raise RuntimeError('Unexpected SafeTune edited tensor set.')
        yield model
    finally:
        restore_weights(model, backup)
        del snapshots
