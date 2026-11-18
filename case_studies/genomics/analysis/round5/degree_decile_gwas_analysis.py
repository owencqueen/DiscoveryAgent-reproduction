import os
"""
Degree Decile GWAS Analysis Script
===================================
This script performs Analysis C (in-degree decile analysis) and Analysis D (out-degree decile analysis)
for the Stim8hr condition, examining the relationship between gene network connectivity and GWAS hit rates.

Analyses include:
- Decile binning by degree rank
- GWAS hit rate calculation per decile
- Spearman correlation for monotonic trend testing
- Chi-squared trend test (Cochran-Armitage approximation)
- Threshold analysis comparing top X% vs bottom (100-X)%
- Line plots with 95% binomial confidence intervals
"""

import os
import sys
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import spearmanr, chi2_contingency, fisher_exact
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================

DATA_PATH = (os.environ["REPRO_DATA_ROOT"] + '/data/processed/gene_topology_with_gwas.csv')
OUTPUT_DIR = (os.environ["REPRO_DATA_ROOT"] + '/reports/analysis_outputs')
FIGURES_DIR = (os.environ["REPRO_DATA_ROOT"] + '/reports/figures')
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "degree_decile_gwas_results.txt")

# Ensure output directories exist
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def binomial_ci(successes, total, confidence=0.95):
    """
    Calculate binomial proportion confidence interval using Wilson score interval.
    More accurate than normal approximation for small samples or extreme proportions.
    """
    if total == 0:
        return 0, 0, 0

    p = successes / total
    z = stats.norm.ppf(1 - (1 - confidence) / 2)

    # Wilson score interval
    denominator = 1 + z**2 / total
    center = (p + z**2 / (2 * total)) / denominator
    margin = z * np.sqrt((p * (1 - p) + z**2 / (4 * total)) / total) / denominator

    lower = max(0, center - margin)
    upper = min(1, center + margin)

    return p, lower, upper


def cochran_armitage_trend_test(counts_table):
    """
    Cochran-Armitage trend test for proportions across ordered groups.

    Parameters:
    -----------
    counts_table : DataFrame with columns ['successes', 'total'] indexed by decile

    Returns:
    --------
    z_statistic, p_value
    """
    n_groups = len(counts_table)
    scores = np.arange(1, n_groups + 1)  # Linear scores for deciles

    successes = counts_table['successes'].values
    totals = counts_table['total'].values

    N = totals.sum()
    R = successes.sum()
    p_bar = R / N

    # Numerator
    numerator = np.sum(scores * successes) - (R * np.sum(scores * totals) / N)

    # Denominator
    var_term = p_bar * (1 - p_bar) * (N * np.sum(scores**2 * totals) - np.sum(scores * totals)**2) / (N**2)

    if var_term <= 0:
        return np.nan, np.nan

    z = numerator / np.sqrt(var_term)
    p_value = 2 * (1 - stats.norm.cdf(abs(z)))  # Two-tailed

    return z, p_value


def threshold_analysis(df, degree_col, threshold_pct, output_lines):
    """
    Compare top X% vs bottom (100-X)% by degree using Fisher's exact test.

    Parameters:
    -----------
    df : DataFrame with degree and GWAS_hit columns
    degree_col : column name for degree
    threshold_pct : percentage threshold (e.g., 10 for top 10%)
    output_lines : list to append output text

    Returns:
    --------
    odds_ratio, p_value, contingency_table
    """
    threshold_rank = np.percentile(df[degree_col], 100 - threshold_pct)

    top_group = df[df[degree_col] >= threshold_rank]
    bottom_group = df[df[degree_col] < threshold_rank]

    # Contingency table: [[top_hits, top_non_hits], [bottom_hits, bottom_non_hits]]
    top_hits = top_group['GWAS_hit'].sum()
    top_total = len(top_group)
    bottom_hits = bottom_group['GWAS_hit'].sum()
    bottom_total = len(bottom_group)

    table = [[top_hits, top_total - top_hits],
             [bottom_hits, bottom_total - bottom_hits]]

    odds_ratio, p_value = fisher_exact(table)

    top_rate = top_hits / top_total if top_total > 0 else 0
    bottom_rate = bottom_hits / bottom_total if bottom_total > 0 else 0

    result = (f"  Top {threshold_pct}% vs Bottom {100-threshold_pct}%:\n"
              f"    Top group: {top_hits}/{top_total} hits ({top_rate:.3f})\n"
              f"    Bottom group: {bottom_hits}/{bottom_total} hits ({bottom_rate:.3f})\n"
              f"    Odds ratio: {odds_ratio:.3f}\n"
              f"    Fisher's exact p-value: {p_value:.4e}\n")

    output_lines.append(result)
    print(result)

    return odds_ratio, p_value, table


def analyze_degree_deciles(df, degree_col, condition_name, output_lines):
    """
    Perform comprehensive decile analysis for a given degree column.

    Parameters:
    -----------
    df : DataFrame filtered for specific condition
    degree_col : 'in_degree' or 'out_degree'
    condition_name : string for labeling
    output_lines : list to append output text

    Returns:
    --------
    decile_stats : DataFrame with decile statistics
    """
    degree_label = "In-Degree" if degree_col == "in_degree" else "Out-Degree"

    header = f"\n{'='*80}\n{degree_label.upper()} DECILE ANALYSIS - {condition_name}\n{'='*80}\n"
    output_lines.append(header)
    print(header)

    # Create rank-based deciles (10 equal-sized bins by rank)
    df = df.copy()
    df['degree_rank'] = df[degree_col].rank(method='first')

    # Use qcut on ranks to get 10 equal-sized bins
    df['decile'] = pd.qcut(df['degree_rank'], q=10, labels=False) + 1  # 1-10

    # Calculate statistics per decile
    decile_stats = []

    for decile in range(1, 11):
        decile_data = df[df['decile'] == decile]

        degree_min = decile_data[degree_col].min()
        degree_max = decile_data[degree_col].max()
        n_genes = len(decile_data)
        n_hits = decile_data['GWAS_hit'].sum()
        hit_rate, ci_lower, ci_upper = binomial_ci(n_hits, n_genes)

        decile_stats.append({
            'decile': decile,
            'degree_min': degree_min,
            'degree_max': degree_max,
            'n_genes': n_genes,
            'n_hits': n_hits,
            'hit_rate': hit_rate,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper
        })

    decile_df = pd.DataFrame(decile_stats)

    # Print decile statistics
    output_lines.append(f"\n{degree_label} Decile Statistics:\n")
    output_lines.append("-" * 100 + "\n")
    output_lines.append(f"{'Decile':>8} | {'Range':>15} | {'N Genes':>10} | {'N GWAS Hits':>12} | {'Hit Rate':>10} | {'95% CI':>20}\n")
    output_lines.append("-" * 100 + "\n")

    print(f"\n{degree_label} Decile Statistics:")
    print("-" * 100)
    print(f"{'Decile':>8} | {'Range':>15} | {'N Genes':>10} | {'N GWAS Hits':>12} | {'Hit Rate':>10} | {'95% CI':>20}")
    print("-" * 100)

    for _, row in decile_df.iterrows():
        line = (f"{int(row['decile']):>8} | "
                f"{int(row['degree_min']):>6}-{int(row['degree_max']):<7} | "
                f"{int(row['n_genes']):>10} | "
                f"{int(row['n_hits']):>12} | "
                f"{row['hit_rate']:>10.4f} | "
                f"[{row['ci_lower']:.4f}, {row['ci_upper']:.4f}]\n")
        output_lines.append(line)
        print(line, end='')

    print("-" * 100)
    output_lines.append("-" * 100 + "\n")

    # Overall statistics
    overall_rate = df['GWAS_hit'].sum() / len(df)
    output_lines.append(f"\nOverall GWAS hit rate: {df['GWAS_hit'].sum()}/{len(df)} ({overall_rate:.4f})\n")
    print(f"\nOverall GWAS hit rate: {df['GWAS_hit'].sum()}/{len(df)} ({overall_rate:.4f})")

    # ========================================================================
    # STATISTICAL TESTS
    # ========================================================================

    output_lines.append(f"\n{'-'*40}\nStatistical Tests:\n{'-'*40}\n")
    print(f"\n{'-'*40}\nStatistical Tests:\n{'-'*40}")

    # 1. Spearman correlation for monotonic trend
    spearman_r, spearman_p = spearmanr(decile_df['decile'], decile_df['hit_rate'])
    spearman_result = (f"Spearman correlation (decile vs hit rate):\n"
                       f"  rho = {spearman_r:.4f}\n"
                       f"  p-value = {spearman_p:.4e}\n")
    output_lines.append(spearman_result)
    print(spearman_result)

    # 2. Cochran-Armitage trend test
    counts_for_trend = pd.DataFrame({
        'successes': decile_df['n_hits'].values,
        'total': decile_df['n_genes'].values
    })
    z_stat, ca_p = cochran_armitage_trend_test(counts_for_trend)
    ca_result = (f"Cochran-Armitage Trend Test:\n"
                 f"  Z-statistic = {z_stat:.4f}\n"
                 f"  p-value = {ca_p:.4e}\n")
    output_lines.append(ca_result)
    print(ca_result)

    # ========================================================================
    # THRESHOLD ANALYSIS
    # ========================================================================

    output_lines.append(f"\n{'-'*40}\nThreshold Analysis (Fisher's Exact Test):\n{'-'*40}\n")
    print(f"\n{'-'*40}\nThreshold Analysis (Fisher's Exact Test):\n{'-'*40}")

    threshold_results = {}
    for thresh in [10, 20, 30]:
        or_val, p_val, table = threshold_analysis(df, degree_col, thresh, output_lines)
        threshold_results[thresh] = {'odds_ratio': or_val, 'p_value': p_val}

    return decile_df, overall_rate, spearman_r, spearman_p, z_stat, ca_p, threshold_results


def create_decile_plot(decile_df, overall_rate, degree_col, title, save_path):
    """
    Create a line plot of GWAS hit rate by degree decile with error bars.
    """
    degree_label = "In-Degree" if degree_col == "in_degree" else "Out-Degree"

    fig, ax = plt.subplots(figsize=(10, 6))

    # Create x-axis labels with degree ranges
    x_labels = [f"{int(row['degree_min'])}-{int(row['degree_max'])}"
                for _, row in decile_df.iterrows()]
    x = np.arange(1, 11)

    # Calculate error bars (asymmetric)
    yerr_lower = decile_df['hit_rate'] - decile_df['ci_lower']
    yerr_upper = decile_df['ci_upper'] - decile_df['hit_rate']

    # Plot line with error bars
    ax.errorbar(x, decile_df['hit_rate'],
                yerr=[yerr_lower, yerr_upper],
                fmt='o-',
                linewidth=2,
                markersize=8,
                capsize=4,
                capthick=1.5,
                color='#2E86AB',
                ecolor='#2E86AB',
                label='GWAS Hit Rate')

    # Add horizontal dashed line for overall rate
    ax.axhline(y=overall_rate, color='#E63946', linestyle='--',
               linewidth=1.5, label=f'Overall Rate ({overall_rate:.4f})')

    # Formatting
    ax.set_xlabel(f'{degree_label} Decile', fontsize=12)
    ax.set_ylabel('GWAS Hit Rate (Proportion)', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')

    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, rotation=45, ha='right', fontsize=9)

    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)

    # Set y-axis to start at 0
    ymax = max(decile_df['ci_upper'].max() * 1.1, overall_rate * 1.5)
    ax.set_ylim(0, ymax)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Saved figure: {save_path}")


def create_combined_plot(indegree_df, outdegree_df, indegree_overall, outdegree_overall, save_path):
    """
    Create a side-by-side figure comparing in-degree and out-degree decile analysis.
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    for idx, (decile_df, overall_rate, degree_col, title) in enumerate([
        (indegree_df, indegree_overall, 'in_degree', 'GWAS Hit Rate by In-Degree Decile (Stim8hr)'),
        (outdegree_df, outdegree_overall, 'out_degree', 'GWAS Hit Rate by Out-Degree Decile (Stim8hr)')
    ]):
        ax = axes[idx]
        degree_label = "In-Degree" if degree_col == "in_degree" else "Out-Degree"

        # Create x-axis labels with degree ranges
        x_labels = [f"{int(row['degree_min'])}-{int(row['degree_max'])}"
                    for _, row in decile_df.iterrows()]
        x = np.arange(1, 11)

        # Calculate error bars (asymmetric)
        yerr_lower = decile_df['hit_rate'] - decile_df['ci_lower']
        yerr_upper = decile_df['ci_upper'] - decile_df['hit_rate']

        # Plot line with error bars
        ax.errorbar(x, decile_df['hit_rate'],
                    yerr=[yerr_lower, yerr_upper],
                    fmt='o-',
                    linewidth=2,
                    markersize=8,
                    capsize=4,
                    capthick=1.5,
                    color='#2E86AB',
                    ecolor='#2E86AB',
                    label='GWAS Hit Rate')

        # Add horizontal dashed line for overall rate
        ax.axhline(y=overall_rate, color='#E63946', linestyle='--',
                   linewidth=1.5, label=f'Overall Rate ({overall_rate:.4f})')

        # Formatting
        ax.set_xlabel(f'{degree_label} Decile', fontsize=12)
        ax.set_ylabel('GWAS Hit Rate (Proportion)', fontsize=12)
        ax.set_title(title, fontsize=12, fontweight='bold')

        ax.set_xticks(x)
        ax.set_xticklabels(x_labels, rotation=45, ha='right', fontsize=8)

        ax.legend(loc='best', fontsize=9)
        ax.grid(True, alpha=0.3)

        # Set y-axis to start at 0
        ymax = max(decile_df['ci_upper'].max() * 1.1, overall_rate * 1.5)
        ax.set_ylim(0, ymax)

    plt.suptitle('Degree Decile Analysis: Relationship Between Network Connectivity and GWAS Hit Rate',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Saved combined figure: {save_path}")


# ============================================================================
# MAIN ANALYSIS
# ============================================================================

def main():
    output_lines = []

    # Header
    header = """
================================================================================
DEGREE DECILE GWAS ANALYSIS
================================================================================
Analysis C: In-Degree Decile Analysis
Analysis D: Out-Degree Decile Analysis
Condition: Stim8hr
================================================================================
"""
    output_lines.append(header)
    print(header)

    # Load data
    print("Loading data...")
    df = pd.read_csv(DATA_PATH)

    output_lines.append(f"Data loaded: {len(df)} rows\n")
    print(f"Data loaded: {len(df)} rows")

    # Filter for Stim8hr condition
    df_stim8hr = df[df['condition'] == 'Stim8hr'].copy()

    output_lines.append(f"Stim8hr condition: {len(df_stim8hr)} genes\n")
    output_lines.append(f"Total GWAS hits in Stim8hr: {df_stim8hr['GWAS_hit'].sum()}\n")
    print(f"Stim8hr condition: {len(df_stim8hr)} genes")
    print(f"Total GWAS hits in Stim8hr: {df_stim8hr['GWAS_hit'].sum()}")

    # ========================================================================
    # ANALYSIS C: IN-DEGREE DECILE ANALYSIS
    # ========================================================================

    indegree_df, indegree_overall, in_spearman_r, in_spearman_p, in_z, in_ca_p, in_thresh = \
        analyze_degree_deciles(df_stim8hr, 'in_degree', 'Stim8hr', output_lines)

    # Create in-degree plot
    create_decile_plot(
        indegree_df,
        indegree_overall,
        'in_degree',
        'GWAS Hit Rate by In-Degree Decile (Stim8hr)',
        os.path.join(FIGURES_DIR, 'gwas_by_indegree_decile.png')
    )

    # ========================================================================
    # ANALYSIS D: OUT-DEGREE DECILE ANALYSIS
    # ========================================================================

    outdegree_df, outdegree_overall, out_spearman_r, out_spearman_p, out_z, out_ca_p, out_thresh = \
        analyze_degree_deciles(df_stim8hr, 'out_degree', 'Stim8hr', output_lines)

    # Create out-degree plot
    create_decile_plot(
        outdegree_df,
        outdegree_overall,
        'out_degree',
        'GWAS Hit Rate by Out-Degree Decile (Stim8hr)',
        os.path.join(FIGURES_DIR, 'gwas_by_outdegree_decile.png')
    )

    # ========================================================================
    # COMBINED FIGURE
    # ========================================================================

    create_combined_plot(
        indegree_df,
        outdegree_df,
        indegree_overall,
        outdegree_overall,
        os.path.join(FIGURES_DIR, 'gwas_by_degree_deciles_combined.png')
    )

    # ========================================================================
    # SUMMARY
    # ========================================================================

    summary = f"""
================================================================================
ANALYSIS SUMMARY
================================================================================

IN-DEGREE ANALYSIS:
  - Spearman correlation: rho = {in_spearman_r:.4f}, p = {in_spearman_p:.4e}
  - Cochran-Armitage trend: Z = {in_z:.4f}, p = {in_ca_p:.4e}
  - Top 10% vs Bottom 90%: OR = {in_thresh[10]['odds_ratio']:.3f}, p = {in_thresh[10]['p_value']:.4e}
  - Top 20% vs Bottom 80%: OR = {in_thresh[20]['odds_ratio']:.3f}, p = {in_thresh[20]['p_value']:.4e}
  - Top 30% vs Bottom 70%: OR = {in_thresh[30]['odds_ratio']:.3f}, p = {in_thresh[30]['p_value']:.4e}

OUT-DEGREE ANALYSIS:
  - Spearman correlation: rho = {out_spearman_r:.4f}, p = {out_spearman_p:.4e}
  - Cochran-Armitage trend: Z = {out_z:.4f}, p = {out_ca_p:.4e}
  - Top 10% vs Bottom 90%: OR = {out_thresh[10]['odds_ratio']:.3f}, p = {out_thresh[10]['p_value']:.4e}
  - Top 20% vs Bottom 80%: OR = {out_thresh[20]['odds_ratio']:.3f}, p = {out_thresh[20]['p_value']:.4e}
  - Top 30% vs Bottom 70%: OR = {out_thresh[30]['odds_ratio']:.3f}, p = {out_thresh[30]['p_value']:.4e}

INTERPRETATION:
"""

    # Add interpretation based on results
    interpretation = ""

    # In-degree interpretation
    if in_spearman_p < 0.05 and in_spearman_r > 0:
        interpretation += "  - IN-DEGREE: Significant POSITIVE trend - higher in-degree associated with higher GWAS hit rate\n"
    elif in_spearman_p < 0.05 and in_spearman_r < 0:
        interpretation += "  - IN-DEGREE: Significant NEGATIVE trend - lower in-degree associated with higher GWAS hit rate\n"
    else:
        interpretation += "  - IN-DEGREE: No significant monotonic trend detected\n"

    # Out-degree interpretation
    if out_spearman_p < 0.05 and out_spearman_r > 0:
        interpretation += "  - OUT-DEGREE: Significant POSITIVE trend - higher out-degree associated with higher GWAS hit rate\n"
    elif out_spearman_p < 0.05 and out_spearman_r < 0:
        interpretation += "  - OUT-DEGREE: Significant NEGATIVE trend - lower out-degree associated with higher GWAS hit rate\n"
    else:
        interpretation += "  - OUT-DEGREE: No significant monotonic trend detected\n"

    summary += interpretation
    summary += "\n================================================================================\n"

    output_lines.append(summary)
    print(summary)

    # Save output to file
    with open(OUTPUT_FILE, 'w') as f:
        f.writelines(output_lines)

    print(f"\nResults saved to: {OUTPUT_FILE}")
    print(f"Figures saved to: {FIGURES_DIR}/")


if __name__ == "__main__":
    main()
