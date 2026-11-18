#!/usr/bin/env python3
'Plot Figure 6c degree-bin GWAS hit rates. Out-degree bins preserve ties.'

from __future__ import annotations

import argparse
import os
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from plot_outdegree_tie_preserving import (
    INPUT_CSV,
    assert_expected_population,
    build_summary as build_outdegree_summary,
    wilson_interval,
)


HERE = Path(__file__).resolve().parent
REGENERATED_DIR = Path(os.environ["REPRO_OUTPUT_DIR"])
PDF_PATH = REGENERATED_DIR / "gwas_by_degree_deciles_combined_tie_preserving.pdf"
PNG_PATH = REGENERATED_DIR / "gwas_by_degree_deciles_combined_tie_preserving.png"
LARGE_TEXT_PDF_PATH = (
    REGENERATED_DIR
    / "gwas_by_degree_deciles_combined_tie_preserving_large_text.pdf"
)
LARGE_TEXT_PNG_PATH = (
    REGENERATED_DIR
    / "gwas_by_degree_deciles_combined_tie_preserving_large_text.png"
)
EXTRA_LARGE_TEXT_PDF_PATH = (
    REGENERATED_DIR
    / "gwas_by_degree_deciles_combined_tie_preserving_extra_large_text.pdf"
)
EXTRA_LARGE_TEXT_PNG_PATH = (
    REGENERATED_DIR
    / "gwas_by_degree_deciles_combined_tie_preserving_extra_large_text.png"
)

GWAS_COLOR = "#2E86AB"
OVERALL_COLOR = "#E63946"


def build_indegree_summary(data: pd.DataFrame) -> pd.DataFrame:
    """Reproduce the agent's original equal-sized in-degree rank deciles."""
    working = data.copy()
    working["degree_rank"] = working["in_degree"].rank(method="first")
    working["decile"] = pd.qcut(working["degree_rank"], q=10, labels=False) + 1

    rows = []
    for decile in range(1, 11):
        group = working.loc[working["decile"].eq(decile)]
        hits = int(group["GWAS_hit"].sum())
        total = len(group)
        ci_lower, ci_upper = wilson_interval(hits, total)
        rows.append(
            {
                "bin": decile,
                "degree_min": int(group["in_degree"].min()),
                "degree_max": int(group["in_degree"].max()),
                "n_genes": total,
                "n_gwas_hits": hits,
                "hit_rate": hits / total,
                "ci_lower": ci_lower,
                "ci_upper": ci_upper,
            }
        )

    summary = pd.DataFrame(rows)
    # These archived values guarantee that the upper panel is unchanged.
    expected_min = [0, 24, 33, 41, 48, 56, 65, 76, 89, 111]
    expected_max = [24, 33, 40, 48, 56, 65, 76, 89, 111, 237]
    expected_hits = [29, 39, 37, 45, 44, 42, 56, 56, 71, 97]
    if summary["degree_min"].tolist() != expected_min:
        raise ValueError("In-degree bin minima differ from the archived panel")
    if summary["degree_max"].tolist() != expected_max:
        raise ValueError("In-degree bin maxima differ from the archived panel")
    if summary["n_gwas_hits"].tolist() != expected_hits:
        raise ValueError("In-degree GWAS-hit counts differ from the archived panel")
    return summary


def plot_panel(
    ax: plt.Axes,
    summary: pd.DataFrame,
    labels: list[str],
    title: str,
    xlabel: str,
    overall_rate: float,
    large_text: bool,
    extra_large_text: bool,
) -> None:
    """Set typography for the stacked degree plots."""
    x = np.arange(1, len(summary) + 1)
    y = summary["hit_rate"].to_numpy()
    errors = np.vstack(
        [
            y - summary["ci_lower"].to_numpy(),
            summary["ci_upper"].to_numpy() - y,
        ]
    )

    overall_artist = ax.axhline(
        overall_rate,
        color=OVERALL_COLOR,
        linestyle="--",
        linewidth=1.5,
        label=f"Overall ({overall_rate:.3f})",
    )
    hit_rate_artist = ax.errorbar(
        x,
        y,
        yerr=errors,
        fmt="o-",
        linewidth=2,
        markersize=8,
        capsize=4,
        capthick=1.5,
        color=GWAS_COLOR,
        ecolor=GWAS_COLOR,
        label="GWAS Hit Rate",
    )

    if extra_large_text:
        font_sizes = {
            "title": 18,
            "axis_label": 18,
            "x_tick": 14,
            "y_tick": 14,
            "legend": 14,
        }
    elif large_text:
        font_sizes = {
            "title": 16,
            "axis_label": 15,
            "x_tick": 11,
            "y_tick": 12,
            "legend": 12,
        }
    else:
        font_sizes = {
            "title": 14,
            "axis_label": 13,
            "x_tick": 9,
            "y_tick": 10,
            "legend": 10,
        }

    ax.set_title(title, fontsize=font_sizes["title"], fontweight="bold")
    ax.set_xlabel(xlabel, fontsize=font_sizes["axis_label"], fontweight="bold")
    ax.set_ylabel(
        "GWAS Hit Rate", fontsize=font_sizes["axis_label"], fontweight="bold"
    )
    ax.set_xticks(x)
    ax.set_xticklabels(
        labels, rotation=45, ha="right", fontsize=font_sizes["x_tick"]
    )
    ax.tick_params(axis="y", labelsize=font_sizes["y_tick"])
    ax.legend(
        [overall_artist, hit_rate_artist],
        [f"Overall ({overall_rate:.3f})", "GWAS Hit Rate"],
        loc="best",
        fontsize=font_sizes["legend"],
    )
    ax.grid(True, alpha=0.3)
    ymax = max(float(summary["ci_upper"].max()) * 1.1, overall_rate * 1.5)
    ax.set_ylim(0, ymax)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Explicitly allow replacement of existing output files.",
    )
    parser.add_argument(
        "--large-text",
        action="store_true",
        help="Increase typography relative to the unchanged figure dimensions.",
    )
    parser.add_argument(
        "--extra-large-text",
        action="store_true",
        help="Increase typography substantially relative to the figure dimensions.",
    )
    args = parser.parse_args()

    if args.large_text and args.extra_large_text:
        parser.error("Choose at most one text-size variant")
    if args.extra_large_text:
        pdf_path = EXTRA_LARGE_TEXT_PDF_PATH
        png_path = EXTRA_LARGE_TEXT_PNG_PATH
    elif args.large_text:
        pdf_path = LARGE_TEXT_PDF_PATH
        png_path = LARGE_TEXT_PNG_PATH
    else:
        pdf_path = PDF_PATH
        png_path = PNG_PATH
    outputs = [pdf_path, png_path]
    existing = [path for path in outputs if path.exists()]
    if existing and not args.overwrite:
        joined = ", ".join(str(path) for path in existing)
        raise FileExistsError(f"Refusing to overwrite existing outputs: {joined}")

    source = pd.read_csv(INPUT_CSV)
    data = source.loc[source["condition"].eq("Stim8hr")].copy()
    assert_expected_population(data)
    overall_rate = float(data["GWAS_hit"].mean())

    indegree = build_indegree_summary(data)
    outdegree = build_outdegree_summary(data)

    indegree_labels = [
        f"{int(row.degree_min)}-{int(row.degree_max)}"
        for row in indegree.itertuples()
    ]
    outdegree_labels = outdegree["outdegree_bin"].astype(str).tolist()
    condition_suffix = "" if args.extra_large_text else " (Stim8hr)"

    fig, axes = plt.subplots(2, 1, figsize=(6, 8.25))
    plot_panel(
        axes[0],
        indegree,
        indegree_labels,
        f"GWAS Hit Rate by In-Degree Decile{condition_suffix}",
        "In-Degree Decile",
        overall_rate,
        args.large_text,
        args.extra_large_text,
    )
    plot_panel(
        axes[1],
        outdegree,
        outdegree_labels,
        f"GWAS Hit Rate by Out-Degree Bin{condition_suffix}",
        "Out-Degree Bin",
        overall_rate,
        args.large_text,
        args.extra_large_text,
    )
    fig.tight_layout(h_pad=1.25)
    fig.savefig(pdf_path, dpi=300, bbox_inches="tight")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Source: {INPUT_CSV}")
    print("In-degree panel: rank deciles")
    print("Out-degree panel: ties grouped together")
    if args.extra_large_text:
        typography = "extra-large-text variant"
    elif args.large_text:
        typography = "large-text variant"
    else:
        typography = "default scale"
    print(f"Typography: {typography}")
    print(f"Saved: {pdf_path}")
    print(f"Saved: {png_path}")


if __name__ == "__main__":
    main()
