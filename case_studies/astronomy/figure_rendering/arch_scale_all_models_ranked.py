'Plot Figure 4f from architecture evaluation summaries.\n\nResNet18 and PhotoFusion-128 use the fixed reference values from the analysis.'

import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from matplotlib.patches import Patch

HERE = os.path.dirname(os.path.abspath(__file__))
TRACE = os.path.dirname(HERE)
OUT_DIR = os.path.join(TRACE, "data")
PDF_DIR = os.environ.get("REPRO_OUTPUT_DIR", os.path.join(TRACE, "outputs"))

FIGSIZE = (3.4, 4.2)
YTICK_SIZE = 7.5
XLABEL_SIZE = 8.5
XTICK_SIZE = 8
VALUE_SIZE = 5
LEGEND_SIZE = 6.5

# Two-line y-tick labels so the label column is narrower
STACKED = {
    "ResNet18-FD (baseline)": "ResNet18-FD\n(baseline)",
    "PhotoFusion-128 (ref)": "PhotoFusion-128\n(ref)",
    "ConvNeXt-Tiny lr=1e-4": "ConvNeXt-Tiny\nlr=1e-4",
    "ConvNeXt-Tiny lr=1e-3": "ConvNeXt-Tiny\nlr=1e-3",
    "ResNet101 lr=5e-4": "ResNet101\nlr=5e-4",
    "ResNet101+PhotoFusion": "ResNet101\n+PhotoFusion",
    "ConvNeXt-T+PhotoFusion": "ConvNeXt-T\n+PhotoFusion",
}


def load_models():
    with open(f"{OUT_DIR}/arch_scale_experiment_results.json") as f:
        v1 = json.load(f)
    with open(f"{OUT_DIR}/arch_scale_v2_experiment_results.json") as f:
        v2 = json.load(f)
    with open(f"{OUT_DIR}/arch_photofusion_experiment_results.json") as f:
        pf = json.load(f)

    def from_v(d, key, name):
        sr = d[key]["subgroup_results"]
        return dict(name=name, full=100 * sr["Full"]["p95c"], dwarf=100 * sr["Dwarfs"]["p95c"])

    def from_pf(key, name):
        sm = pf["models"][key]["subgroup_metrics"]
        return dict(name=name, full=sm["Full"]["P_at_95C"], dwarf=sm["Dwarfs"]["P_at_95C"])

    models = [
        dict(name="ResNet18-FD (baseline)", full=11.69, dwarf=22.11),
        dict(name="PhotoFusion-128 (ref)", full=13.11, dwarf=22.28),
        from_v(v1, "convnext_tiny", "ConvNeXt-Tiny"),
        from_v(v1, "resnet50", "ResNet50"),
        from_v(v1, "vit_small", "ViT-Small"),
        from_v(v2, "convnext_small", "ConvNeXt-Small"),
        from_v(v2, "resnet101", "ResNet101"),
        from_v(v2, "convnext_tiny_lr1e4", "ConvNeXt-Tiny lr=1e-4"),
        from_v(v2, "convnext_tiny_lr1e3", "ConvNeXt-Tiny lr=1e-3"),
        from_pf("ResNet101_lr5e4", "ResNet101 lr=5e-4"),
        from_pf("ResNet101_PhotoFusion", "ResNet101+PhotoFusion"),
        from_pf("ConvNeXtTiny_PhotoFusion", "ConvNeXt-T+PhotoFusion"),
    ]
    # Same order as the original: sorted by full-sample P@95C, best at the bottom
    models.sort(key=lambda m: -m["full"])
    return models


def plot(models, out_path):
    fig, ax = plt.subplots(figsize=FIGSIZE)
    names = [m["name"] for m in models]
    full = [m["full"] for m in models]
    dwarf = [m["dwarf"] for m in models]
    y_pos = np.arange(len(names))
    w = 0.35

    bars1 = ax.barh(y_pos - w/2, full, w, color="steelblue", edgecolor="black", linewidth=0.5)
    bars2 = ax.barh(y_pos + w/2, dwarf, w, color="coral", edgecolor="black", linewidth=0.5)
    for bars, vals in ((bars1, full), (bars2, dwarf)):
        for bar, val in zip(bars, vals):
            ax.text(bar.get_width() + 0.2, bar.get_y() + bar.get_height()/2,
                    f"{val:.2f}", ha="left", va="center", fontsize=VALUE_SIZE)

    ax.set_yticks(y_pos)
    ax.set_yticklabels([STACKED.get(n, n) for n in names], fontsize=YTICK_SIZE,
                       linespacing=0.95)
    ax.set_ylim(-0.6, len(names) - 0.4)
    ax.set_xlim(0, 26.5)
    ax.tick_params(axis="x", labelsize=XTICK_SIZE)
    ax.set_xlabel("Purity at 95% Completeness (%)", fontsize=XLABEL_SIZE)
    ax.grid(axis="x", alpha=0.3)

    handles = [Patch(facecolor="steelblue", edgecolor="black", linewidth=0.5),
               Patch(facecolor="coral", edgecolor="black", linewidth=0.5)]
    labels = ["Full Sample", "Dwarfs"]
    ax.legend(handles, labels, fontsize=LEGEND_SIZE, ncol=2, loc="lower center",
              bbox_to_anchor=(0.5, 1.0), frameon=False, columnspacing=1.0,
              handlelength=1.6)

    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    os.makedirs(PDF_DIR, exist_ok=True)
    plot(load_models(), os.path.join(PDF_DIR, "arch_scale_all_models_ranked.pdf"))
