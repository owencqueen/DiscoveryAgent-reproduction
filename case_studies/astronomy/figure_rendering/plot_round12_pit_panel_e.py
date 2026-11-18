#!/usr/bin/env python3
'Plot Figure 4e from the 20-bin PIT densities grouped by true redshift.'
from pathlib import Path
import argparse
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
MODEL_COLORS = {'Low_z': '#E24A33', 'High_z': '#348ABD'}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--output-dir', type=Path, required=True)
    ap.add_argument('--histogram-json', type=Path,
                    default=ROOT/'data/round12_pit_histograms.json')
    ap.add_argument('--width', type=float, default=2.0)
    ap.add_argument('--height', type=float, default=2.7)
    ap.add_argument('--font-size', type=float, default=7.5)
    args = ap.parse_args()
    data = json.loads(args.histogram_json.read_text())
    plt.rcdefaults()
    plt.rcParams['pdf.fonttype'] = 42
    fig, ax = plt.subplots(figsize=(args.width, args.height), dpi=300)
    for key, label in [('Low_z', 'z < 0.03'), ('High_z', 'z >= 0.03')]:
        g = data['groups'][key]
        edges = data['histogram_edges']
        centers = [(a+b)/2 for a,b in zip(edges[:-1], edges[1:])]
        ax.hist(centers, bins=edges, weights=g['counts'], density=True,
                histtype='step', color=MODEL_COLORS[key], linewidth=1.25,
                label=f"{label}\n(N={g['n']:,}, KS={g['KS_stat']:.3f})")
    ax.axhline(1, color='k', linestyle='--', linewidth=.8, alpha=.6,
               label='Uniform (density=1)')
    ax.set_xlabel('PIT value', fontsize=args.font_size)
    ax.set_ylabel('Density', fontsize=args.font_size)
    ax.set_title('Low-z vs High-z', fontsize=args.font_size, fontweight='bold')
    ax.legend(fontsize=args.font_size-2, loc='upper right', handlelength=1.2,
              handletextpad=.4, borderpad=.4, labelspacing=.5)
    ax.set_xlim(0,1)
    ax.set_ylim(bottom=0)
    ax.tick_params(labelsize=args.font_size-1, pad=2, length=2.5, width=.65)
    for spine in ax.spines.values():
        spine.set_linewidth(.65)
    ax.set_xticks([0,.5,1])
    ax.set_yticks([0,1,2,3])
    fig.tight_layout(pad=.45)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for extension in ['pdf','png']:
        fig.savefig(args.output_dir/f'round12_pit_panel_e.{extension}', dpi=400)
    plt.close(fig)


if __name__ == '__main__':
    main()
