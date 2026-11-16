"""Plot logged Action L1 and task-macro fixed-subset validation; no new evaluation."""
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter
    import numpy as np

    train = [json.loads(line) for line in (args.run / 'metrics.jsonl').read_text().splitlines()]
    validation = [json.loads(line) for line in (args.run / 'validation.jsonl').read_text().splitlines()]
    x = np.array([r['step'] for r in train])
    y = np.array([r['losses']['action'] for r in train])
    vx = np.array([r['step'] for r in validation])
    vy = np.array([np.mean([t['action_l1'] for t in r['tasks'].values()]) for r in validation])
    smooth = np.array([y[max(0, i-19):i+1].mean() for i in range(len(y))])
    assert len(x) == 901 and len(vx) == 45 and x[-1] == vx[-1] == 45000
    assert np.isfinite(y).all() and np.isfinite(vy).all() and (y > 0).all() and (vy > 0).all()

    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11,
                         'axes.spines.top': False, 'axes.spines.right': False,
                         'axes.labelcolor': '#334155', 'text.color': '#0f172a',
                         'xtick.color': '#475569', 'ytick.color': '#475569',
                         'svg.fonttype': 'none'})
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.6))
    fig.subplots_adjust(left=.075, right=.965, bottom=.23, top=.76, wspace=.25)
    fig.suptitle('TD-VLA · Action loss', fontsize=21, fontweight='bold', x=.075, ha='left', y=.96)
    fig.text(.075, .88, 'Clean · 6 tasks · global batch 128 · 45,000 optimizer steps',
             fontsize=12, color='#64748b')
    for ax in axes:
        ax.plot(x, y, color='#93c5fd', alpha=.6, lw=.8, label='Train: logged batches')
        ax.plot(x, smooth, color='#2563eb', lw=2, label='Train: trailing 20-point mean')
        ax.plot(vx, vy, color='#ea580c', marker='o', markersize=3.1, lw=2,
                label='Validation: six-task macro mean')
        ax.grid(axis='y', alpha=.18)
        ax.set_xlabel('Optimizer step')
        ax.set_ylabel('Action L1 loss')
        ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f'{value/1000:g}k'))
    axes[0].set_yscale('log')
    axes[0].set_xlim(0, 47000)
    axes[0].set_title('Full run · logarithmic loss axis', loc='left', fontsize=12, pad=12)
    axes[1].set_xlim(10000, 47000)
    axes[1].set_ylim(0, max(y[x >= 10000].max(), vy[vx >= 10000].max()) * 1.17)
    axes[1].set_title('After 10k steps · linear loss axis', loc='left', fontsize=12, pad=12)
    axes[1].annotate(f'Validation  {vy[-1]:.5f}', (vx[-1], vy[-1]),
                     xytext=(-8, 12), textcoords='offset points', ha='right', color='#c2410c', fontsize=10)
    axes[1].annotate(f'Train mean  {smooth[-1]:.5f}', (x[-1], smooth[-1]),
                     xytext=(-8, 12), textcoords='offset points', ha='right', color='#1d4ed8', fontsize=10)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower left', bbox_to_anchor=(.065, .085),
               ncol=3, frameon=False, fontsize=10)
    fig.text(.075, .055, 'Validation: 240 fixed frames from 30 held-out episodes; online weights, not EMA.',
             fontsize=9, color='#64748b')
    fig.text(.075, .025, 'Train: global valid-coordinate mean per logged batch. Validation: per-task valid-coordinate mean, then task macro average.',
             fontsize=9, color='#64748b')
    args.output.mkdir(parents=True, exist_ok=True)
    for extension in ['png', 'svg']:
        fig.savefig(args.output / f'action-loss.{extension}', dpi=190, facecolor='white')
    plt.close(fig)
    print(json.dumps({'training_points': len(x), 'validation_points': len(vx),
                      'last_train_action': float(y[-1]), 'last_train_mean': float(smooth[-1]),
                      'last_validation_action_macro': float(vy[-1]),
                      'output': str(args.output)}))


if __name__ == '__main__':
    main()
