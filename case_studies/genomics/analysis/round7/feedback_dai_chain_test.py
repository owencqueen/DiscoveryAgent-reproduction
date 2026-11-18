import os
"""
Feedback-DAI Chain Test: Testing whether "fewer feedback loops -> more unidirectional -> more repressive" holds.

This script tests the logical chain by examining the relationship between:
- Feedback loop count (regulatory complexity)
- Directional Asymmetry Index (DAI): (n_down - n_up) / n_total_de

Hypothesis: Genes with fewer feedback loops should have higher DAI (more repressive)
"""

import numpy as np
import pandas as pd
import scanpy as sc
from scipy import stats
from scipy.sparse import issparse
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# PATHS
# =============================================================================
DATA_PATH = (os.environ["REPRO_DATA_ROOT"] + '/data/GWCD4i.DE_stats.h5ad')
GWAS_PATH = (os.environ["REPRO_DATA_ROOT"] + '/data/downloads/autoimmune_l2g_predictions_score_gt_0.5.csv')
EXISTING_FEEDBACK_PATH = (os.environ["REPRO_DATA_ROOT"] + '/data/processed/stim8hr_feedback_topology_gwas.csv')
OUTPUT_FIG_PATH = (os.environ["REPRO_DATA_ROOT"] + '/reports/figures/feedback_dai_chain_test.png')

# =============================================================================
# STEP 1: Load and filter data, compute DAI
# =============================================================================
print("=" * 80)
print("STEP 1: Loading data and computing DAI")
print("=" * 80)

# Load h5ad
print("\nLoading h5ad file...")
adata = sc.read_h5ad(DATA_PATH)
print(f"Full dataset shape: {adata.shape}")
print(f"Layers available: {list(adata.layers.keys())}")

# Filter to Stim8hr
stim8hr_mask = adata.obs['culture_condition'] == 'Stim8hr'
adata_stim = adata[stim8hr_mask].copy()
print(f"\nStim8hr subset shape: {adata_stim.shape}")

# Quality filter: ontarget_significant=True, offtarget_flag=False
quality_mask = (adata_stim.obs['ontarget_significant'] == True) & (adata_stim.obs['offtarget_flag'] == False)
adata_qc = adata_stim[quality_mask].copy()
print(f"After quality filtering: {adata_qc.shape}")

# Get perturbation gene names
perturb_genes = adata_qc.obs['target_contrast_gene_name'].values
unique_perturb_genes = adata_qc.obs['target_contrast_gene_name'].unique()
print(f"Unique perturbation genes: {len(unique_perturb_genes)}")

# Compute DAI for each perturbation
# DAI = (n_down_genes - n_up_genes) / n_total_de_genes
n_down = adata_qc.obs['n_down_genes'].values
n_up = adata_qc.obs['n_up_genes'].values
n_total = adata_qc.obs['n_total_de_genes'].values

# Only keep genes with n_total > 0
valid_mask = n_total > 0
print(f"Perturbations with n_total_de > 0: {valid_mask.sum()} / {len(valid_mask)}")

# Create dataframe with DAI
dai_df = pd.DataFrame({
    'gene': perturb_genes[valid_mask],
    'n_down': n_down[valid_mask],
    'n_up': n_up[valid_mask],
    'n_total_de': n_total[valid_mask],
    'DAI': (n_down[valid_mask] - n_up[valid_mask]) / n_total[valid_mask]
})

print(f"\nDAI statistics (all perturbations):")
print(f"  Mean DAI: {dai_df['DAI'].mean():.4f}")
print(f"  Median DAI: {dai_df['DAI'].median():.4f}")
print(f"  Std DAI: {dai_df['DAI'].std():.4f}")
print(f"  Range: [{dai_df['DAI'].min():.4f}, {dai_df['DAI'].max():.4f}]")

# =============================================================================
# STEP 2: Compute feedback counts from scratch
# =============================================================================
print("\n" + "=" * 80)
print("STEP 2: Computing feedback counts from scratch")
print("=" * 80)

# Get gene names in var (measured genes)
if 'gene_name' in adata_qc.var.columns:
    measured_genes = set(adata_qc.var['gene_name'].values)
else:
    measured_genes = set(adata_qc.var_names)

print(f"Measured genes in var: {len(measured_genes)}")

# Find genes that are BOTH perturbed AND measured
perturbed_set = set(unique_perturb_genes)
both_perturb_and_measured = perturbed_set.intersection(measured_genes)
print(f"Genes that are both perturbed AND measured: {len(both_perturb_and_measured)}")

# Create mapping from gene name to obs index (row) and var index (column)
gene_to_obs_idx = {}
for i, gene in enumerate(adata_qc.obs['target_contrast_gene_name']):
    if gene not in gene_to_obs_idx:
        gene_to_obs_idx[gene] = i  # First occurrence

if 'gene_name' in adata_qc.var.columns:
    gene_to_var_idx = {gene: i for i, gene in enumerate(adata_qc.var['gene_name'])}
else:
    gene_to_var_idx = {gene: i for i, gene in enumerate(adata_qc.var_names)}

# Get adj_p_value layer
adj_pval = adata_qc.layers['adj_p_value']
print(f"\nadj_p_value layer type: {type(adj_pval)}")
print(f"adj_p_value shape: {adj_pval.shape}")

# Convert to dense if sparse for easier indexing
if issparse(adj_pval):
    print("Converting sparse adj_p_value to dense (this may take a moment)...")
    adj_pval = adj_pval.toarray()

# Compute feedback count for each gene that is both perturbed and measured
print("\nComputing feedback counts...")
feedback_counts = {}

genes_to_process = list(both_perturb_and_measured)
for i, gene_A in enumerate(genes_to_process):
    if i % 100 == 0:
        print(f"  Processing gene {i}/{len(genes_to_process)}...")

    feedback_partners = 0
    obs_idx_A = gene_to_obs_idx.get(gene_A)
    var_idx_A = gene_to_var_idx.get(gene_A)

    if obs_idx_A is None or var_idx_A is None:
        continue

    for gene_B in genes_to_process:
        if gene_A == gene_B:
            continue

        obs_idx_B = gene_to_obs_idx.get(gene_B)
        var_idx_B = gene_to_var_idx.get(gene_B)

        if obs_idx_B is None or var_idx_B is None:
            continue

        # Check if A affects B (A's row, B's column)
        pval_A_affects_B = adj_pval[obs_idx_A, var_idx_B]
        # Check if B affects A (B's row, A's column)
        pval_B_affects_A = adj_pval[obs_idx_B, var_idx_A]

        # Handle NaN as non-significant
        if np.isnan(pval_A_affects_B) or np.isnan(pval_B_affects_A):
            continue

        # Feedback if both are significant (adj_p < 0.10)
        if pval_A_affects_B < 0.10 and pval_B_affects_A < 0.10:
            feedback_partners += 1

    feedback_counts[gene_A] = feedback_partners

print(f"\nComputed feedback counts for {len(feedback_counts)} genes")

# Create feedback dataframe
feedback_df = pd.DataFrame({
    'gene': list(feedback_counts.keys()),
    'feedback_count': list(feedback_counts.values())
})

print(f"\nFeedback count statistics:")
print(f"  Mean: {feedback_df['feedback_count'].mean():.2f}")
print(f"  Median: {feedback_df['feedback_count'].median():.2f}")
print(f"  Max: {feedback_df['feedback_count'].max()}")
print(f"  Genes with feedback > 0: {(feedback_df['feedback_count'] > 0).sum()}")

# =============================================================================
# STEP 3: Load GWAS genes
# =============================================================================
print("\n" + "=" * 80)
print("STEP 3: Loading GWAS genes")
print("=" * 80)

gwas_df = pd.read_csv(GWAS_PATH)
gwas_genes = set(gwas_df['gene_symbol'].unique())
print(f"Total GWAS genes: {len(gwas_genes)}")

# =============================================================================
# STEP 4: Merge feedback and DAI data
# =============================================================================
print("\n" + "=" * 80)
print("STEP 4: Merging feedback and DAI data")
print("=" * 80)

# Aggregate DAI by gene (take mean if multiple perturbations)
dai_by_gene = dai_df.groupby('gene').agg({
    'DAI': 'mean',
    'n_down': 'mean',
    'n_up': 'mean',
    'n_total_de': 'mean'
}).reset_index()

# Merge with feedback counts
merged_df = dai_by_gene.merge(feedback_df, on='gene', how='inner')
merged_df['GWAS_hit'] = merged_df['gene'].isin(gwas_genes).astype(int)

print(f"Merged dataset: {len(merged_df)} genes")
print(f"  GWAS genes in merged: {merged_df['GWAS_hit'].sum()}")
print(f"  Non-GWAS genes: {(merged_df['GWAS_hit'] == 0).sum()}")

# Create feedback bins
def assign_feedback_bin(count):
    if count == 0:
        return '0'
    elif count <= 5:
        return '1-5'
    elif count <= 20:
        return '6-20'
    else:
        return '20+'

merged_df['feedback_bin'] = merged_df['feedback_count'].apply(assign_feedback_bin)
merged_df['feedback_bin'] = pd.Categorical(merged_df['feedback_bin'],
                                            categories=['0', '1-5', '6-20', '20+'],
                                            ordered=True)

# Also load existing feedback data for comparison
existing_feedback = pd.read_csv(EXISTING_FEEDBACK_PATH)
print(f"\nExisting feedback file has {len(existing_feedback)} genes")
print(f"Correlation with our computed feedback counts:")
common_genes = set(merged_df['gene']).intersection(set(existing_feedback['gene']))
if len(common_genes) > 10:
    compare_df = merged_df[merged_df['gene'].isin(common_genes)].merge(
        existing_feedback[['gene', 'feedback_count']],
        on='gene',
        suffixes=('_new', '_existing')
    )
    corr = compare_df['feedback_count_new'].corr(compare_df['feedback_count_existing'])
    print(f"  Pearson correlation: {corr:.4f}")

# =============================================================================
# STEP 5: Statistical tests
# =============================================================================
print("\n" + "=" * 80)
print("STEP 5: Statistical tests")
print("=" * 80)

# 5a) Spearman correlation between feedback_count and DAI (all genes)
print("\n--- 5a) Spearman correlation: feedback_count vs DAI (ALL genes) ---")
spearman_all = stats.spearmanr(merged_df['feedback_count'], merged_df['DAI'])
print(f"  N = {len(merged_df)}")
print(f"  Spearman rho = {spearman_all.correlation:.4f}")
print(f"  p-value = {spearman_all.pvalue:.2e}")

# 5b) Mann-Whitney U: DAI for feedback > 0 vs feedback == 0
print("\n--- 5b) Mann-Whitney U: DAI for feedback > 0 vs feedback == 0 ---")
dai_fb_zero = merged_df[merged_df['feedback_count'] == 0]['DAI']
dai_fb_pos = merged_df[merged_df['feedback_count'] > 0]['DAI']
mwu_all = stats.mannwhitneyu(dai_fb_zero, dai_fb_pos, alternative='two-sided')
print(f"  N (feedback = 0): {len(dai_fb_zero)}")
print(f"  N (feedback > 0): {len(dai_fb_pos)}")
print(f"  Median DAI (feedback = 0): {dai_fb_zero.median():.4f}")
print(f"  Median DAI (feedback > 0): {dai_fb_pos.median():.4f}")
print(f"  Mann-Whitney U = {mwu_all.statistic:.1f}")
print(f"  p-value = {mwu_all.pvalue:.2e}")

# 5c) Among GWAS genes: Spearman correlation
print("\n--- 5c) Among GWAS genes: Spearman correlation ---")
gwas_subset = merged_df[merged_df['GWAS_hit'] == 1]
if len(gwas_subset) > 5:
    spearman_gwas = stats.spearmanr(gwas_subset['feedback_count'], gwas_subset['DAI'])
    print(f"  N = {len(gwas_subset)}")
    print(f"  Spearman rho = {spearman_gwas.correlation:.4f}")
    print(f"  p-value = {spearman_gwas.pvalue:.2e}")
else:
    print(f"  Insufficient GWAS genes: {len(gwas_subset)}")
    spearman_gwas = None

# 5d) Among GWAS genes: Mann-Whitney U
print("\n--- 5d) Among GWAS genes: Mann-Whitney U (feedback > 0 vs == 0) ---")
gwas_fb_zero = gwas_subset[gwas_subset['feedback_count'] == 0]['DAI']
gwas_fb_pos = gwas_subset[gwas_subset['feedback_count'] > 0]['DAI']
if len(gwas_fb_zero) > 2 and len(gwas_fb_pos) > 2:
    mwu_gwas = stats.mannwhitneyu(gwas_fb_zero, gwas_fb_pos, alternative='two-sided')
    print(f"  N (feedback = 0): {len(gwas_fb_zero)}")
    print(f"  N (feedback > 0): {len(gwas_fb_pos)}")
    print(f"  Median DAI (feedback = 0): {gwas_fb_zero.median():.4f}")
    print(f"  Median DAI (feedback > 0): {gwas_fb_pos.median():.4f}")
    print(f"  Mann-Whitney U = {mwu_gwas.statistic:.1f}")
    print(f"  p-value = {mwu_gwas.pvalue:.2e}")
else:
    print(f"  Insufficient samples in one group")
    mwu_gwas = None

# 5e) Mean/Median DAI across feedback count bins
print("\n--- 5e) DAI across feedback count bins ---")
bin_stats = merged_df.groupby('feedback_bin')['DAI'].agg(['mean', 'median', 'std', 'count'])
print(bin_stats.to_string())

# Kruskal-Wallis test across bins
groups = [group['DAI'].values for name, group in merged_df.groupby('feedback_bin')]
kruskal = stats.kruskal(*groups)
print(f"\nKruskal-Wallis test across bins:")
print(f"  H-statistic = {kruskal.statistic:.2f}")
print(f"  p-value = {kruskal.pvalue:.2e}")

# Linear trend test (Jonckheere-Terpstra alternative: simple regression)
merged_df['feedback_bin_numeric'] = merged_df['feedback_bin'].cat.codes
trend_spearman = stats.spearmanr(merged_df['feedback_bin_numeric'], merged_df['DAI'])
print(f"\nTrend test (Spearman with ordered bins):")
print(f"  Spearman rho = {trend_spearman.correlation:.4f}")
print(f"  p-value = {trend_spearman.pvalue:.2e}")

# =============================================================================
# STEP 6: Visualization
# =============================================================================
print("\n" + "=" * 80)
print("STEP 6: Creating visualizations")
print("=" * 80)

fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# Panel A: Scatter plot of feedback_count vs DAI, colored by GWAS status
ax = axes[0, 0]
# Add jitter to feedback_count for visibility
np.random.seed(42)
jitter_x = merged_df['feedback_count'] + np.random.normal(0, 0.3, len(merged_df))

colors = ['#1f77b4' if g == 0 else '#d62728' for g in merged_df['GWAS_hit']]
scatter = ax.scatter(jitter_x, merged_df['DAI'], c=colors, alpha=0.5, s=20)
ax.axhline(y=0, color='gray', linestyle='--', linewidth=1, alpha=0.7)
ax.set_xlabel('Feedback Count (with jitter)', fontsize=12)
ax.set_ylabel('DAI (Directional Asymmetry Index)', fontsize=12)
ax.set_title(f'A) Feedback Count vs DAI\n(Spearman rho={spearman_all.correlation:.3f}, p={spearman_all.pvalue:.2e})', fontsize=12)

# Legend
from matplotlib.patches import Patch
legend_elements = [Patch(facecolor='#1f77b4', label='Non-GWAS', alpha=0.5),
                   Patch(facecolor='#d62728', label='GWAS', alpha=0.5)]
ax.legend(handles=legend_elements, loc='upper right')

# Panel B: Box plot of DAI across feedback count bins, split by GWAS/non-GWAS
ax = axes[0, 1]
# Prepare data for grouped boxplot
plot_data = merged_df[['feedback_bin', 'DAI', 'GWAS_hit']].copy()
plot_data['GWAS_status'] = plot_data['GWAS_hit'].map({0: 'Non-GWAS', 1: 'GWAS'})

sns.boxplot(data=plot_data, x='feedback_bin', y='DAI', hue='GWAS_status',
            ax=ax, palette={'Non-GWAS': '#1f77b4', 'GWAS': '#d62728'})
ax.axhline(y=0, color='gray', linestyle='--', linewidth=1, alpha=0.7)
ax.set_xlabel('Feedback Count Bin', fontsize=12)
ax.set_ylabel('DAI', fontsize=12)
ax.set_title('B) DAI by Feedback Bin and GWAS Status', fontsize=12)
ax.legend(title='')

# Panel C: Violin plot of DAI for feedback=0 vs feedback>0
ax = axes[1, 0]
merged_df['has_feedback'] = (merged_df['feedback_count'] > 0).map({False: 'No Feedback\n(n={})'.format(len(dai_fb_zero)),
                                                                    True: 'Has Feedback\n(n={})'.format(len(dai_fb_pos))})
sns.violinplot(data=merged_df, x='has_feedback', y='DAI', ax=ax, palette=['#2ca02c', '#9467bd'])
ax.axhline(y=0, color='gray', linestyle='--', linewidth=1, alpha=0.7)
ax.set_xlabel('', fontsize=12)
ax.set_ylabel('DAI', fontsize=12)
ax.set_title(f'C) DAI: No Feedback vs Has Feedback\n(Mann-Whitney p={mwu_all.pvalue:.2e})', fontsize=12)

# Panel D: Among GWAS genes only: scatter of feedback_count vs DAI with trend line
ax = axes[1, 1]
if len(gwas_subset) > 5:
    jitter_gwas = gwas_subset['feedback_count'] + np.random.normal(0, 0.3, len(gwas_subset))
    ax.scatter(jitter_gwas, gwas_subset['DAI'], c='#d62728', alpha=0.6, s=40, label='GWAS genes')

    # Add trend line
    z = np.polyfit(gwas_subset['feedback_count'], gwas_subset['DAI'], 1)
    p = np.poly1d(z)
    x_range = np.linspace(gwas_subset['feedback_count'].min(), gwas_subset['feedback_count'].max(), 100)
    ax.plot(x_range, p(x_range), "k--", alpha=0.7, linewidth=2, label='Trend line')

    ax.axhline(y=0, color='gray', linestyle='--', linewidth=1, alpha=0.7)
    ax.set_xlabel('Feedback Count (with jitter)', fontsize=12)
    ax.set_ylabel('DAI', fontsize=12)
    if spearman_gwas:
        ax.set_title(f'D) GWAS Genes Only: Feedback vs DAI\n(Spearman rho={spearman_gwas.correlation:.3f}, p={spearman_gwas.pvalue:.2e})', fontsize=12)
    else:
        ax.set_title('D) GWAS Genes Only: Feedback vs DAI', fontsize=12)
    ax.legend(loc='upper right')
else:
    ax.text(0.5, 0.5, 'Insufficient GWAS genes\nfor visualization',
            ha='center', va='center', transform=ax.transAxes, fontsize=14)
    ax.set_title('D) GWAS Genes Only: Feedback vs DAI', fontsize=12)

plt.tight_layout()
plt.savefig(OUTPUT_FIG_PATH, dpi=150, bbox_inches='tight')
print(f"\nFigure saved to: {OUTPUT_FIG_PATH}")

# =============================================================================
# STEP 7: Comprehensive Summary
# =============================================================================
print("\n" + "=" * 80)
print("STEP 7: COMPREHENSIVE SUMMARY")
print("=" * 80)

print("""
HYPOTHESIS: "Fewer feedback loops -> more unidirectional -> more repressive"

If true, we expect:
  - NEGATIVE correlation between feedback_count and DAI
  - Genes with feedback=0 should have HIGHER DAI than genes with feedback>0
""")

print("\n" + "-" * 60)
print("SAMPLE SIZES:")
print("-" * 60)
print(f"Total genes analyzed: {len(merged_df)}")
print(f"  GWAS genes: {merged_df['GWAS_hit'].sum()}")
print(f"  Non-GWAS genes: {(merged_df['GWAS_hit'] == 0).sum()}")
print(f"Genes with feedback = 0: {len(dai_fb_zero)}")
print(f"Genes with feedback > 0: {len(dai_fb_pos)}")
print(f"GWAS genes with feedback = 0: {len(gwas_fb_zero)}")
print(f"GWAS genes with feedback > 0: {len(gwas_fb_pos)}")

print("\n" + "-" * 60)
print("CORRELATION RESULTS:")
print("-" * 60)
print(f"ALL GENES:")
print(f"  Spearman rho = {spearman_all.correlation:.4f}")
print(f"  p-value = {spearman_all.pvalue:.2e}")
print(f"  Interpretation: {'SUPPORTS' if spearman_all.correlation < 0 and spearman_all.pvalue < 0.05 else 'DOES NOT SUPPORT'} hypothesis")

if spearman_gwas:
    print(f"\nGWAS GENES ONLY:")
    print(f"  Spearman rho = {spearman_gwas.correlation:.4f}")
    print(f"  p-value = {spearman_gwas.pvalue:.2e}")
    print(f"  Interpretation: {'SUPPORTS' if spearman_gwas.correlation < 0 and spearman_gwas.pvalue < 0.05 else 'DOES NOT SUPPORT'} hypothesis")

print("\n" + "-" * 60)
print("MANN-WHITNEY U RESULTS (DAI comparison):")
print("-" * 60)
print(f"ALL GENES:")
print(f"  Median DAI (feedback = 0): {dai_fb_zero.median():.4f}")
print(f"  Median DAI (feedback > 0): {dai_fb_pos.median():.4f}")
print(f"  Difference: {dai_fb_zero.median() - dai_fb_pos.median():.4f}")
print(f"  Mann-Whitney U = {mwu_all.statistic:.1f}")
print(f"  p-value = {mwu_all.pvalue:.2e}")
direction = "higher" if dai_fb_zero.median() > dai_fb_pos.median() else "lower"
significant = mwu_all.pvalue < 0.05
supports = direction == "higher" and significant
print(f"  Interpretation: Genes without feedback have {direction} DAI ({'significantly' if significant else 'not significantly'}) -> {'SUPPORTS' if supports else 'DOES NOT SUPPORT'} hypothesis")

if mwu_gwas:
    print(f"\nGWAS GENES ONLY:")
    print(f"  Median DAI (feedback = 0): {gwas_fb_zero.median():.4f}")
    print(f"  Median DAI (feedback > 0): {gwas_fb_pos.median():.4f}")
    print(f"  Difference: {gwas_fb_zero.median() - gwas_fb_pos.median():.4f}")
    print(f"  Mann-Whitney U = {mwu_gwas.statistic:.1f}")
    print(f"  p-value = {mwu_gwas.pvalue:.2e}")
    direction_gwas = "higher" if gwas_fb_zero.median() > gwas_fb_pos.median() else "lower"
    significant_gwas = mwu_gwas.pvalue < 0.05
    supports_gwas = direction_gwas == "higher" and significant_gwas
    print(f"  Interpretation: GWAS genes without feedback have {direction_gwas} DAI ({'significantly' if significant_gwas else 'not significantly'}) -> {'SUPPORTS' if supports_gwas else 'DOES NOT SUPPORT'} hypothesis")

print("\n" + "-" * 60)
print("DAI ACROSS FEEDBACK BINS:")
print("-" * 60)
print(bin_stats.to_string())
print(f"\nKruskal-Wallis H = {kruskal.statistic:.2f}, p = {kruskal.pvalue:.2e}")
print(f"Trend test (Spearman): rho = {trend_spearman.correlation:.4f}, p = {trend_spearman.pvalue:.2e}")

# Check if there's a decreasing trend
bin_medians = [merged_df[merged_df['feedback_bin'] == b]['DAI'].median() for b in ['0', '1-5', '6-20', '20+']]
print(f"\nMedian DAI by bin: {bin_medians}")
decreasing_trend = all(bin_medians[i] >= bin_medians[i+1] for i in range(len(bin_medians)-1))
print(f"Monotonically decreasing? {decreasing_trend}")

print("\n" + "=" * 80)
print("FINAL VERDICT")
print("=" * 80)

# Assess overall evidence
evidence_points = []
if spearman_all.correlation < 0 and spearman_all.pvalue < 0.05:
    evidence_points.append("+ Significant negative correlation (all genes)")
elif spearman_all.correlation < 0:
    evidence_points.append("~ Negative correlation trend (all genes, not significant)")
else:
    evidence_points.append("- No negative correlation (all genes)")

if dai_fb_zero.median() > dai_fb_pos.median() and mwu_all.pvalue < 0.05:
    evidence_points.append("+ Genes without feedback have significantly higher DAI")
elif dai_fb_zero.median() > dai_fb_pos.median():
    evidence_points.append("~ Genes without feedback have higher DAI (not significant)")
else:
    evidence_points.append("- Genes without feedback do NOT have higher DAI")

if decreasing_trend:
    evidence_points.append("+ Monotonically decreasing DAI with increasing feedback")
else:
    evidence_points.append("~ No clear monotonic trend across bins")

if spearman_gwas and spearman_gwas.correlation < 0 and spearman_gwas.pvalue < 0.05:
    evidence_points.append("+ Significant negative correlation in GWAS genes")
elif spearman_gwas and spearman_gwas.correlation < 0:
    evidence_points.append("~ Negative correlation trend in GWAS genes (not significant)")
else:
    evidence_points.append("- No clear pattern in GWAS genes")

print("\nEvidence assessment:")
for pt in evidence_points:
    print(f"  {pt}")

n_support = sum(1 for pt in evidence_points if pt.startswith('+'))
n_weak = sum(1 for pt in evidence_points if pt.startswith('~'))
n_against = sum(1 for pt in evidence_points if pt.startswith('-'))

print(f"\nSummary: {n_support} strong support, {n_weak} weak/trend, {n_against} against")

if n_support >= 3:
    verdict = "STRONG SUPPORT: The chain 'fewer feedback -> more unidirectional -> more repressive' HOLDS"
elif n_support >= 2 or (n_support >= 1 and n_weak >= 2):
    verdict = "MODERATE SUPPORT: The chain shows partial evidence, with some trends in expected direction"
elif n_support >= 1 or n_weak >= 2:
    verdict = "WEAK SUPPORT: There are hints of the expected pattern, but not statistically robust"
else:
    verdict = "NO SUPPORT: The data does not support the hypothesized chain"

print(f"\n*** {verdict} ***")

print("\n" + "=" * 80)
print("Analysis complete!")
print("=" * 80)

# Save the computed inputs used by the manuscript renderer.
processed_dir = os.path.dirname(EXISTING_FEEDBACK_PATH)
os.makedirs(processed_dir, exist_ok=True)
feedback_df.to_csv(EXISTING_FEEDBACK_PATH, index=False)
perturbation_summary_path = os.path.join(processed_dir, 'perturbation_summary.csv')
adata.obs.to_csv(perturbation_summary_path, index_label='index')
print(f"Saved: {EXISTING_FEEDBACK_PATH}")
print(f"Saved: {perturbation_summary_path}")
