"""Plot saved results on CPU; no model loading or new experiment."""
import csv
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

root = Path(__file__).resolve().parents[1]
with (root / 'results/summary.csv').open() as f:
    rows = list(csv.DictReader(f))

def rate(condition, benchmark, label, metric):
    matches = [r for r in rows if (r['condition'], r['benchmark'], r['label'], r['metric'])
               == (condition, benchmark, label, metric)]
    assert len(matches) == 1
    return float(matches[0]['rate']) * 100

def point(condition):
    return (rate(condition, 'xstest', 'safe', 'guard_refusal'),
            rate(condition, 'jbb_transfer', 'harmful', 'guard_harmful'))

fig, ax = plt.subplots(figsize=(8, 5.6))
styles = [('single_layer', 'Single layer', '#3976b3'),
          ('shared_direction', 'Shared direction, 4 layers', '#d77b18'),
          ('layer_specific', 'Layer-specific directions', '#278568')]
for variant, label, color in styles:
    points = [point(f'{variant}_f{factor:g}') for factor in (.25, .5, 1.)]
    ax.plot(*zip(*points), marker='o', color=color, label=label, linewidth=1.8)
    for factor, (x, y) in zip((.25, .5, 1.), points):
        offset = (5, 8) if variant != 'layer_specific' else (5, -15)
        if variant == 'single_layer' and factor == .25:
            offset = (7, -19)
        if variant == 'shared_direction' and factor == .5:
            offset = (-24, 12)
        ax.annotate(f'{factor:g}', (x, y), xytext=offset, textcoords='offset points', fontsize=8, color=color)
ax.scatter(*point('baseline'), marker='*', s=180, color='#20252b', label='Baseline', zorder=5)
random_points = [point(f'random_shared_s{s}_f0.5') for s in (0, 1, 2)]
ax.scatter(*zip(*random_points), marker='x', s=55, color='#888888', label='Random direction (3 seeds)')
ax.set(xlabel='Over-refusal on legitimate XSTest prompts (%) · n = 250',
       ylabel='Judged harmful responses to transferred attacks (%) · n = 170',
       xlim=(0, 104), ylim=(-1, 20),
       title='Tiny Aya: safety gains come with over-refusal costs')
ax.grid(alpha=.18)
ax.legend(loc='upper right', fontsize=9)
fig.text(.5, .015, 'Lower left is better. Labels show steering strength. Qwen3Guard judgments; point estimates.',
         ha='center', fontsize=8)
fig.tight_layout(rect=(0, .045, 1, 1))
fig.savefig(root / 'figures/safety_overrefusal.png', dpi=180)
plt.close(fig)
