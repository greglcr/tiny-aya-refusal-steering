# Reproducing the analysis

## Saved results and figures

The aggregate tables and two figures are included in this Git repository. The companion [Hugging Face results repository](https://huggingface.co/datasets/GLECO/tiny-aya-refusal-steering) also contains all 8,060 per-example evaluation scores and source/split indexes.

From the repository root, with Matplotlib installed:

```bash
python scripts/plot_tradeoff.py
```

This regenerates `figures/safety_overrefusal.png` from `results/summary.csv`, without model loading or a GPU.

`summary.csv` includes event counts, measured denominators, unknown counts and Wilson 95% intervals. `paired_differences.csv` includes paired changes from baseline with 2,000 underlying-request-group bootstrap resamples, seed 42. These intervals are exploratory and uncorrected for multiple comparisons. `per_family.csv` breaks the results down by benchmark family.

## Experiment implementation

The experiment modules are in `scripts/`:

- `global_refusal.py`: extraction, validation-based selection and ablation utilities.
- `layer_steering.py`: normalized directions and equal-total-budget interventions.
- `auditkit_bridge.py`: AuditKIT callable generation and scores.
- `safety_campaign.py`: conditions, generation and resumption.
- `campaign_judge.py`: Qwen3Guard judgments and precomputed AuditKIT evaluation.
- `campaign_report.py`: metrics and original report generation.
- `campaign_data.py`: pinned benchmark download and lexical overlap filtering.
- `run_campaign.py`: sequential execution of the campaign stages.

`requirements.lock` records the original Python 3.12 experiment environment. This includes SafeTune 0.1.19, AuditKIT 1.1.12, Transformers 5.21.1 and PyTorch 2.15.0. Existing tests run with `python -m unittest discover -s tests` in that environment.

## Requirements for full regeneration

The original modules expect calibration manifests and `splits.csv`, benchmark prompt text, model caches, and the original notebook for the reporting step. Exact regeneration requires these additional inputs. The published calibration index records the split assignments and source names, but its project-specific IDs do not reconstruct prompt text. The published source URLs and evaluated-example index identify the external benchmarks.

To rerun the GPU pipeline exactly, supply those original inputs in the expected paths and obtain model access. To independently re-judge answers, the raw generated answers are also required. The supplied tables support score analysis and figure reproduction.

## Model artifacts

The four original train-fitted activation vectors and scientific configuration are in [the separate model-artifact repository](https://huggingface.co/GLECO/tiny-aya-refusal-steering-vectors). Load the tensors with `torch.load(..., map_location="cpu", weights_only=True)`. Block indices are zero-based. Shared and layer-specific interventions divide the total budget equally across blocks 9, 14, 19 and 25; the selected global vector comes from block 14.

The vector repository is public and ungated. Base-model weights must be obtained from Cohere's repository under its access conditions and terms. Neither this Git repository nor the vector artifact is a replacement checkpoint.
