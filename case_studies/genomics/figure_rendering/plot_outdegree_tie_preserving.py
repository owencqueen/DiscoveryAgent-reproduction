#!/usr/bin/env python3
'Compute degree-bin GWAS hit rates with Wilson 95% intervals.\n\nIdentical out-degrees remain in the same bin; the lowest degrees form separate bins.'

from __future__ import annotations

import argparse
import os
from pathlib import Path
from statistics import NormalDist

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
INPUT_CSV = Path(os.environ["REPRO_DATA_ROOT"]) / "data/processed/gene_topology_with_gwas.csv"

PDF_PATH = Path(os.environ["REPRO_OUTPUT_DIR"]) / "gwas_hit_rate_by_outdegree_tie_preserving.pdf"
PNG_PATH = Path(os.environ["REPRO_OUTPUT_DIR"]) / "gwas_hit_rate_by_outdegree_tie_preserving.png"
CSV_PATH = Path(os.environ["REPRO_OUTPUT_DIR"]) / "gwas_hit_rate_by_outdegree_tie_preserving.csv"

# Plot colors.
GWAS_COLOR = "#2E86AB"
OVERALL_COLOR = "#E63946"


def wilson_interval(successes: int, total: int, confidence: float = 0.95) -> tuple[float, float]:
    """Return a two-sided Wilson score interval for a binomial proportion."""
    z = NormalDist().inv_cdf(1 - (1 - confidence) / 2)
    proportion = successes / total
    denominator = 1 + z**2 / total
    center = (proportion + z**2 / (2 * total)) / denominator
    margin = (
        z
        * np.sqrt((proportion * (1 - proportion) + z**2 / (4 * total)) / total)
        / denominator
    )
    return max(0.0, center - margin), min(1.0, center + margin)


def build_summary(data: pd.DataFrame) -> pd.DataFrame:
    """Create outcome-blind bins that preserve every tied integer degree."""
    # Values 0, 1, and 2 are large point masses and are retained as separate
    # groups. The remaining cut points are the original empirical 60th, 70th,
    # 80th, and 90th percentile boundaries, rounded down to integer degrees.
    bin_edges = [-np.inf, 0, 1, 2, 4, 8, 27, 136, np.inf]
    bin_labels = ["0", "1", "2", "3-4", "5-8", "9-27", "28-136", ">=138"]

    working = data.copy()
    working["outdegree_bin"] = pd.cut(
        working["out_degree"],
        bins=bin_edges,
        labels=bin_labels,
        right=True,
        ordered=True,
    )
    if working["outdegree_bin"].isna().any():
        raise ValueError("At least one out-degree was not assigned to a bin")

    summary = (
        working.groupby("outdegree_bin", observed=True, sort=True)
        .agg(
            degree_min=("out_degree", "min"),
            degree_max=("out_degree", "max"),
            n_genes=("GWAS_hit", "size"),
            n_gwas_hits=("GWAS_hit", "sum"),
            hit_rate=("GWAS_hit", "mean"),
        )
        .reset_index()
    )

    intervals = [
        wilson_interval(int(row.n_gwas_hits), int(row.n_genes))
        for row in summary.itertuples()
    ]
    summary["ci_lower"] = [interval[0] for interval in intervals]
    summary["ci_upper"] = [interval[1] for interval in intervals]
    return summary


def assert_expected_population(data: pd.DataFrame) -> None:
    """Fail loudly if the archived agent input no longer matches its report."""
    required = {"condition", "out_degree", "GWAS_hit"}
    missing = required - set(data.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    if len(data) != 6_279:
        raise ValueError(f"Expected 6,279 Stim8hr genes, found {len(data):,}")
    if int(data["GWAS_hit"].sum()) != 516:
        raise ValueError(f"Expected 516 GWAS hits, found {int(data['GWAS_hit'].sum())}")
    if data[["out_degree", "GWAS_hit"]].isna().any().any():
        raise ValueError("Missing out-degree or GWAS-hit values")
    if not set(data["GWAS_hit"].unique()).issubset({0, 1}):
        raise ValueError("GWAS_hit must be binary")


def plot_summary(summary: pd.DataFrame, overall_rate: float) -> plt.Figure:
    x = np.arange(1, len(summary) + 1)
    y = summary["hit_rate"].to_numpy()
    errors = np.vstack(
        [
            y - summary["ci_lower"].to_numpy(),
            summary["ci_upper"].to_numpy() - y,
        ]
    )

    fig, ax = plt.subplots(figsize=(10, 6))
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
    overall_artist = ax.axhline(
        overall_rate,
        color=OVERALL_COLOR,
        linestyle="--",
        linewidth=1.5,
        label=f"Overall Rate ({overall_rate:.4f})",
    )

    plot_labels = ["0", "1", "2", "3-4", "5-8", "9-27", "28-136", ">=138"]
    ax.set_xticks(x)
    ax.set_xticklabels(plot_labels, rotation=45, ha="right", fontsize=9)
    ax.set_xlabel("Out-Degree Bin", fontsize=12)
    ax.set_ylabel("GWAS Hit Rate (Proportion)", fontsize=12)
    ax.set_title(
        "GWAS Hit Rate by Out-Degree Bin (Stim8hr)",
        fontsize=14,
        fontweight="bold",
    )
    ax.legend(
        [overall_artist, hit_rate_artist],
        [f"Overall Rate ({overall_rate:.4f})", "GWAS Hit Rate"],
        loc="best",
        fontsize=10,
    )
    ax.grid(True, alpha=0.3)
    ymax = max(float(summary["ci_upper"].max()) * 1.1, overall_rate * 1.5)
    ax.set_ylim(0, ymax)
    fig.tight_layout()
    return fig


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Explicitly allow replacement of this script's new output files.",
    )
    args = parser.parse_args()

    outputs = [PDF_PATH, PNG_PATH, CSV_PATH]
    existing = [path for path in outputs if path.exists()]
    if existing and not args.overwrite:
        joined = ", ".join(str(path) for path in existing)
        raise FileExistsError(f"Refusing to overwrite existing outputs: {joined}")

    source = pd.read_csv(INPUT_CSV)
    data = source.loc[source["condition"].eq("Stim8hr")].copy()
    assert_expected_population(data)

    summary = build_summary(data)
    if int(summary["n_genes"].sum()) != len(data):
        raise ValueError("Binned gene counts do not recover the original population")
    if int(summary["n_gwas_hits"].sum()) != int(data["GWAS_hit"].sum()):
        raise ValueError("Binned GWAS hits do not recover the original total")

    overall_rate = float(data["GWAS_hit"].mean())
    summary.to_csv(CSV_PATH, index=False)
    figure = plot_summary(summary, overall_rate)
    figure.savefig(PDF_PATH, dpi=400, bbox_inches="tight")
    figure.savefig(PNG_PATH, dpi=300, bbox_inches="tight")
    plt.close(figure)

    print(f"Source: {INPUT_CSV}")
    print(f"Population: {len(data):,} genes; {int(data['GWAS_hit'].sum())} GWAS hits")
    print(summary.to_string(index=False))
    print(f"Saved: {PDF_PATH}")
    print(f"Saved: {PNG_PATH}")
    print(f"Saved: {CSV_PATH}")


if __name__ == "__main__":
    main()
