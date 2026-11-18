#!/usr/bin/env python3
'Plot Figure 6d from perturbation summaries, feedback counts and GWAS labels.'

import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


TRACE_ROOT = Path(os.environ["REPRO_DATA_ROOT"])
PERTURBATION_PATH = TRACE_ROOT / "data/processed/perturbation_summary.csv"
FEEDBACK_PATH = TRACE_ROOT / "data/processed/stim8hr_feedback_topology_gwas.csv"
GWAS_PATH = (
    TRACE_ROOT
    / "data/downloads/autoimmune_l2g_predictions_score_gt_0.5.csv"
)
OUTPUT_PATH = Path(os.environ["REPRO_OUTPUT_DIR"]) / "feedback_dai_panel_b.pdf"

FEEDBACK_BIN_ORDER = ["0", "1-5", "6-20", "20+"]
GWAS_COLORS = {"Non-GWAS": "#1f77b4", "GWAS": "#d62728"}


def assign_feedback_bin(count: int) -> str:
    """Assign feedback counts to the four displayed bins."""
    if count == 0:
        return "0"
    if count <= 5:
        return "1-5"
    if count <= 20:
        return "6-20"
    return "20+"


def load_panel_data() -> pd.DataFrame:
    """Construct the gene-level DAI and feedback table."""
    perturbations = pd.read_csv(PERTURBATION_PATH)
    perturbations = perturbations.loc[
        (perturbations["culture_condition"] == "Stim8hr")
        & perturbations["ontarget_significant"]
        & ~perturbations["offtarget_flag"]
        & (perturbations["n_total_de_genes"] > 0)
    ].copy()

    perturbations["DAI"] = (
        perturbations["n_down_genes"] - perturbations["n_up_genes"]
    ) / perturbations["n_total_de_genes"]

    # Average DAI across QC-filtered rows for each gene.
    dai_by_gene = (
        perturbations.groupby("target_contrast_gene_name", as_index=False)["DAI"]
        .mean()
        .rename(columns={"target_contrast_gene_name": "gene"})
    )

    feedback = pd.read_csv(FEEDBACK_PATH, usecols=["gene", "feedback_count"])
    gwas_genes = set(pd.read_csv(GWAS_PATH, usecols=["gene_symbol"])["gene_symbol"])

    plot_data = dai_by_gene.merge(feedback, on="gene", how="inner")
    plot_data["feedback_bin"] = pd.Categorical(
        plot_data["feedback_count"].map(assign_feedback_bin),
        categories=FEEDBACK_BIN_ORDER,
        ordered=True,
    )
    plot_data["GWAS_status"] = plot_data["gene"].isin(gwas_genes).map(
        {False: "Non-GWAS", True: "GWAS"}
    )
    return plot_data


def main() -> None:
    plot_data = load_panel_data()

    fig, ax = plt.subplots(figsize=(7, 3))

    sns.boxplot(
        data=plot_data,
        x="feedback_bin",
        y="DAI",
        hue="GWAS_status",
        ax=ax,
        palette=GWAS_COLORS,
    )
    ax.axhline(y=0, color="gray", linestyle="--", linewidth=1, alpha=0.7)
    ax.set_xlabel("Feedback Count Bin", fontsize=12)
    ax.set_ylabel("Directional Asymmetry Index", fontsize=10, labelpad=3)
    ax.set_title(
        "B) Directional Asymmetry Index by Feedback Bin and GWAS Status",
        fontsize=11,
    )
    ax.legend(title="")

    plt.tight_layout()
    fig.savefig(OUTPUT_PATH, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
