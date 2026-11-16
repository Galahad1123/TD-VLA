"""Rebuild formal multistream training curves from the archived JSONL logs."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
train = [json.loads(s) for s in (ROOT/'logs/metrics.jsonl').read_text().splitlines()]
val = [json.loads(s) for s in (ROOT/'logs/validation.jsonl').read_text().splitlines()]
fig, axes = plt.subplots(2, 3, figsize=(14, 7.5))
for i, (loss, metric, label) in enumerate([
    ('action', 'action_l1', 'Action L1'),
    ('track', 'track_mae', 'Track MAE (normalized coordinates)'),
    ('stage', 'stage_accuracy', 'Stage accuracy'),
]):
    x = np.array([r['step'] for r in train])
    y = np.array([r['losses'][loss] for r in train])
    smooth = np.array([y[max(0,j-19):j+1].mean() for j in range(len(y))])
    axes[0,i].plot(x/1000, y, color='#93c5fd', alpha=.5, lw=.7)
    axes[0,i].plot(x/1000, smooth, color='#2563eb', lw=1.7)
    axes[0,i].set_yscale('log')
    axes[0,i].set_title('Train: '+{'action':'Action L1','track':'Track SmoothL1','stage':'Stage CE'}[loss])
    vx = [r['step']/1000 for r in val]
    vy = [np.mean([t[metric] for t in r['tasks'].values()]) for r in val]
    axes[1,i].plot(vx, vy, color='#ea580c', marker='.', lw=1.7)
    axes[1,i].set_title('Validation: '+label)
    if metric=='stage_accuracy':
        axes[1,i].set_ylim(.97,1.002)
    for ax in axes[:,i]:
        ax.set_xlabel('Optimizer steps (thousands)')
        ax.grid(alpha=.2)
        ax.spines[['top','right']].set_visible(False)
fig.suptitle('TD-VLA · Formal 45,000-step run', fontsize=18)
fig.tight_layout(rect=(0,.065,1,.95))
fig.text(.025,.025,'Blue: raw train batches + trailing 20-point mean. Orange: fixed 240-frame validation, six-task macro mean, online weights.',fontsize=10)
for ext in ['png','svg']:
    fig.savefig(ROOT/f'figures/multistream-curves.{ext}',dpi=180,facecolor='white')
plt.close(fig)
