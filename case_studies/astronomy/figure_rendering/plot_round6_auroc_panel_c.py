'Plot Figure 4c: four validation AUROC curves over 25 epochs.'

import argparse
import json
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
SOURCE = HERE.parent / "data/prior_experiment_histories.json"
REFERENCE_WIDTH = 430.28350830078125
REFERENCE_HEIGHT = 466.4077453613281
REFERENCE_AXES = (56.82500076293945, 26.12188720703125,
                  419.5249938964844, 414.2018737792969)
PLACED_AXES_RATIO = 101.8895 / 80.9857

# Preserve the agent-produced AUROC plot's original condition colors.
MODEL_COLORS = {
    "photo_only": "orange", "ssl_partial": "green",
    "imagenet_partial": "blue", "ssl_full": "purple",
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--width", type=float, default=6.5)
    parser.add_argument("--font-size", type=float, default=10)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    histories = json.loads(SOURCE.read_text())
    x0, y0, x1, y1 = REFERENCE_AXES
    bounds = [x0 / REFERENCE_WIDTH, (REFERENCE_HEIGHT - y1) / REFERENCE_HEIGHT,
              (x1 - x0) / REFERENCE_WIDTH, (y1 - y0) / REFERENCE_HEIGHT]
    height = args.width * bounds[2] / (PLACED_AXES_RATIO * bounds[3])

    # Reuse the defaults of the agent's script rather than applying a new theme.
    plt.rcdefaults()
    plt.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans"],
        "font.size": args.font_size, "axes.labelsize": args.font_size,
        "axes.titlesize": args.font_size * 1.2,
        "xtick.labelsize": args.font_size, "ytick.labelsize": args.font_size,
        "pdf.fonttype": 42, "ps.fonttype": 42,
    })
    fig, ax = plt.subplots(figsize=(args.width, height), dpi=300)
    rows = []
    for key in MODEL_COLORS:
        values = histories[key]["val_auc_roc"]
        assert len(values) == 25 and all(0 <= v <= 1 for v in values)
        rows.extend({"condition": key, "epoch": epoch, "validation_auroc": value}
                    for epoch, value in enumerate(values, 1))
        ax.plot(range(1, 26), values, color=MODEL_COLORS[key],
                label=key, linestyle="-", linewidth=1.5, marker="o", markersize=2)

    ax.set(xlabel="Epoch", ylabel="AUC-ROC", title="Validation AUC-ROC")
    ax.legend(fontsize=args.font_size * 0.7)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    # Restore measured bounds so the saved page matches the placed 4c shape.
    ax.set_position(bounds)
    fig.canvas.draw()
    page_box = fig.bbox
    visible_ticks = [t for t in ax.get_xticklabels()
                     if ax.get_xlim()[0] <= t.get_position()[0] <= ax.get_xlim()[1]]
    visible_ticks += [t for t in ax.get_yticklabels()
                      if ax.get_ylim()[0] <= t.get_position()[1] <= ax.get_ylim()[1]]
    for text in [ax.title, ax.xaxis.label, ax.yaxis.label, *visible_ticks]:
        b = text.get_window_extent(fig.canvas.get_renderer())
        assert page_box.contains(b.x0, b.y0) and page_box.contains(b.x1, b.y1), text.get_text()
    stem = args.output_dir / "A6_validation_auroc_4c"
    fig.savefig(stem.with_suffix(".pdf"), dpi=300, facecolor="white")
    fig.savefig(stem.with_suffix(".png"), dpi=300, facecolor="white")
    plt.close(fig)
    with stem.with_suffix(".csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["condition", "epoch", "validation_auroc"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved {stem}.pdf; page ratio={args.width / height:.4f}; axes ratio={PLACED_AXES_RATIO:.4f}")


if __name__ == "__main__":
    main()
