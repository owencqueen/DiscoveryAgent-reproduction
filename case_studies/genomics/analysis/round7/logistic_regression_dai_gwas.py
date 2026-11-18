import os
"""
Logistic regression analysis of DAI (Directional Asymmetry Index) as a predictor
of GWAS hit status, controlling for network topology and expression level.

Steps:
1. Compute network topology metrics (out-degree, in-degree, relay status, DAI,
   mean_signed_zscore) for Stim8hr quality-filtered perturbations.
2. Fit nested logistic regression models predicting GWAS_hit from topology,
   DAI, expression, and mean signed z-score.
3. Within in-degree decile analysis of DAI differences by GWAS status.
4. Within expression tertile analysis of DAI differences by GWAS status.
5. Model comparison and forest plot figures.
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import anndata as ad
import statsmodels.api as sm
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# PATHS
# ============================================================================
BASE_DIR = (os.environ["REPRO_DATA_ROOT"] + '')
DATA_PATH = f'{BASE_DIR}/data/GWCD4i.DE_stats.h5ad'
GWAS_PATH = f'{BASE_DIR}/data/downloads/autoimmune_l2g_predictions_score_gt_0.5.csv'
FIG_DIR = f'{BASE_DIR}/reports/figures'

# ============================================================================
# DATA LOADING
# ============================================================================
print("=" * 80)
print("LOADING DATA")
print("=" * 80)

adata = ad.read_h5ad(DATA_PATH)
print(f"Loaded adata: {adata.shape[0]} perturbations x {adata.shape[1]} genes")

gwas_df = pd.read_csv(GWAS_PATH)
print(f"GWAS L2G file columns: {list(gwas_df.columns)}")
gwas_genes = set(gwas_df['gene_symbol'].unique())
print(f"Unique GWAS L2G genes (score > 0.5): {len(gwas_genes)}")

# ============================================================================
# STEP 1: COMPUTE NETWORK TOPOLOGY METRICS (Stim8hr only)
# ============================================================================
print("\n" + "=" * 80)
print("STEP 1: COMPUTE NETWORK TOPOLOGY METRICS (Stim8hr)")
print("=" * 80)

# Filter to Stim8hr quality-filtered set
mask = ((adata.obs['culture_condition'] == 'Stim8hr') &
        (adata.obs['ontarget_significant'] == True) &
        (adata.obs['offtarget_flag'] == False))
stim8 = adata[mask].copy()
print(f"Stim8hr quality-filtered perturbations: {stim8.shape[0]}")

# Build gene_name -> var index mapping
var_gene_names = stim8.var['gene_name'].values
gene_name_to_var_idx = {}
for idx, gn in enumerate(var_gene_names):
    gene_name_to_var_idx[gn] = idx

# Get perturbation gene names
perturb_gene_names = stim8.obs['target_contrast_gene_name'].values

# Extract the adj_p_value matrix (perturbations x genes)
# This is potentially large, so we work with it as a dense array
print("Extracting adj_p_value matrix...")
adj_pval_matrix = stim8.layers['adj_p_value']
if hasattr(adj_pval_matrix, 'toarray'):
    adj_pval_matrix = adj_pval_matrix.toarray()
else:
    adj_pval_matrix = np.array(adj_pval_matrix)
print(f"adj_p_value matrix shape: {adj_pval_matrix.shape}")

# Extract z-score matrix for mean_signed_zscore computation
print("Extracting z-score matrix...")
zscore_matrix = stim8.layers['zscore']
if hasattr(zscore_matrix, 'toarray'):
    zscore_matrix = zscore_matrix.toarray()
else:
    zscore_matrix = np.array(zscore_matrix)

# ---- Out-degree ----
out_degree = stim8.obs['n_downstream'].values.astype(float)

# ---- In-degree ----
# For each measured gene (column j), count how many perturbation rows have
# adj_p_value < 0.10, EXCLUDING the self-perturbation row.
print("Computing in-degree for each gene...")
sig_mask = adj_pval_matrix < 0.10  # boolean matrix (n_perturb x n_genes)

# For each perturbation, find its column index (if it exists as a measured gene)
# and mask out the self-perturbation
perturb_col_indices = []
for gn in perturb_gene_names:
    if gn in gene_name_to_var_idx:
        perturb_col_indices.append(gene_name_to_var_idx[gn])
    else:
        perturb_col_indices.append(-1)  # not in var
perturb_col_indices = np.array(perturb_col_indices)

# Create self-exclusion mask: for each row i, set column perturb_col_indices[i] to False
self_exclusion = sig_mask.copy()
for i in range(self_exclusion.shape[0]):
    col_idx = perturb_col_indices[i]
    if col_idx >= 0:
        self_exclusion[i, col_idx] = False

# In-degree for each gene column = sum over perturbation rows (excluding self)
in_degree_per_col = self_exclusion.sum(axis=0)  # shape: (n_genes,)

# Map gene_name -> in_degree
gene_in_degree = {}
for j, gn in enumerate(var_gene_names):
    gene_in_degree[gn] = in_degree_per_col[j]

# Map in-degree back to perturbation genes
in_degree = np.array([gene_in_degree.get(gn, 0) for gn in perturb_gene_names])

# ---- is_relay ----
in_deg_75 = np.percentile(in_degree, 75)
out_deg_75 = np.percentile(out_degree, 75)
is_relay = ((in_degree >= in_deg_75) & (out_degree >= out_deg_75)).astype(int)
print(f"In-degree 75th percentile: {in_deg_75}")
print(f"Out-degree 75th percentile: {out_deg_75}")
print(f"Number of relay genes: {is_relay.sum()}")

# ---- DAI ----
n_down = stim8.obs['n_down_genes'].values.astype(float)
n_up = stim8.obs['n_up_genes'].values.astype(float)
total = n_down + n_up
dai = np.where(total > 0, (n_down - n_up) / total, np.nan)

# ---- GWAS_hit ----
gwas_hit = np.array([1 if gn in gwas_genes else 0 for gn in perturb_gene_names])

# ---- baseMean ----
base_mean = stim8.obs['target_baseMean'].values.astype(float)

# ---- mean_signed_zscore ----
# For each perturbation row, compute the mean z-score across genes that are
# significantly DE (adj_p_value < 0.10), excluding the self-target gene.
print("Computing mean signed z-score for each perturbation...")
mean_signed_zscore = np.full(stim8.shape[0], np.nan)
for i in range(stim8.shape[0]):
    # Significant genes mask for this perturbation (excluding self)
    sig_genes_mask = self_exclusion[i, :]
    if sig_genes_mask.sum() > 0:
        mean_signed_zscore[i] = zscore_matrix[i, sig_genes_mask].mean()

# ---- Build DataFrame ----
topo_df = pd.DataFrame({
    'gene_name': perturb_gene_names,
    'out_degree': out_degree,
    'in_degree': in_degree,
    'is_relay': is_relay,
    'DAI': dai,
    'GWAS_hit': gwas_hit,
    'baseMean': base_mean,
    'mean_signed_zscore': mean_signed_zscore,
})

print(f"\nTopology DataFrame shape: {topo_df.shape}")
print(f"\nSummary statistics:")
print(topo_df.describe().to_string())
print(f"\nGWAS hit distribution:")
print(topo_df['GWAS_hit'].value_counts().to_string())
print(f"\nRelay distribution:")
print(topo_df['is_relay'].value_counts().to_string())
print(f"\nDAI NaN count: {topo_df['DAI'].isna().sum()}")
print(f"mean_signed_zscore NaN count: {topo_df['mean_signed_zscore'].isna().sum()}")
print(f"baseMean NaN count: {topo_df['baseMean'].isna().sum()}")

# ============================================================================
# STEP 2: FIT NESTED LOGISTIC REGRESSION MODELS
# ============================================================================
print("\n" + "=" * 80)
print("STEP 2: NESTED LOGISTIC REGRESSION MODELS")
print("=" * 80)

# Prepare data: only include perturbations where DAI is not NaN
model_df = topo_df.dropna(subset=['DAI']).copy()
model_df['log1p_in_degree'] = np.log1p(model_df['in_degree'])
model_df['log1p_baseMean'] = np.log1p(model_df['baseMean'])
print(f"Model DataFrame (DAI not NaN): {model_df.shape[0]} rows")
print(f"GWAS hits in model data: {model_df['GWAS_hit'].sum()}")

# For M5: further restrict to non-NaN mean_signed_zscore
m5_df = model_df.dropna(subset=['mean_signed_zscore']).copy()
print(f"M5 DataFrame (mean_signed_zscore not NaN): {m5_df.shape[0]} rows")
print(f"GWAS hits in M5 data: {m5_df['GWAS_hit'].sum()}")


def fit_logit(df, formula_rhs_cols, target='GWAS_hit'):
    """Fit a logistic regression model using statsmodels GLM with Binomial."""
    X = sm.add_constant(df[formula_rhs_cols])
    y = df[target]
    model = sm.GLM(y, X, family=sm.families.Binomial())
    result = model.fit()
    return result


def print_model_results(name, result):
    """Print formatted model results with OR and 95% CI."""
    print(f"\n--- {name} ---")
    print(f"AIC: {result.aic:.4f}")
    print(f"BIC: {result.bic_deviance:.4f}")
    print(f"Log-likelihood: {result.llf:.4f}")
    print(f"N observations: {result.nobs:.0f}")
    print()

    ci = result.conf_int()
    coef = result.params
    pvals = result.pvalues
    or_vals = np.exp(coef)
    or_ci_lo = np.exp(ci[0])
    or_ci_hi = np.exp(ci[1])

    header = f"{'Variable':<25} {'Coef':>10} {'OR':>10} {'95% CI Low':>12} {'95% CI High':>12} {'p-value':>12}"
    print(header)
    print("-" * len(header))
    for var_name in coef.index:
        print(f"{var_name:<25} {coef[var_name]:>10.4f} {or_vals[var_name]:>10.4f} "
              f"{or_ci_lo[var_name]:>12.4f} {or_ci_hi[var_name]:>12.4f} {pvals[var_name]:>12.4e}")
    return result


# M1: GWAS_hit ~ log1p(in_degree)
m1 = fit_logit(model_df, ['log1p_in_degree'])
print_model_results("M1: GWAS_hit ~ log1p(in_degree)", m1)

# M2: GWAS_hit ~ log1p(in_degree) + is_relay
m2 = fit_logit(model_df, ['log1p_in_degree', 'is_relay'])
print_model_results("M2: GWAS_hit ~ log1p(in_degree) + is_relay", m2)

# M3: GWAS_hit ~ log1p(in_degree) + is_relay + DAI
m3 = fit_logit(model_df, ['log1p_in_degree', 'is_relay', 'DAI'])
print_model_results("M3: GWAS_hit ~ log1p(in_degree) + is_relay + DAI", m3)

# M4: GWAS_hit ~ log1p(in_degree) + is_relay + DAI + log1p(baseMean)
# Need to drop NaN baseMean for M4
m4_df = model_df.dropna(subset=['baseMean']).copy()
print(f"\nM4 DataFrame (baseMean not NaN): {m4_df.shape[0]} rows")
m4 = fit_logit(m4_df, ['log1p_in_degree', 'is_relay', 'DAI', 'log1p_baseMean'])
print_model_results("M4: GWAS_hit ~ log1p(in_degree) + is_relay + DAI + log1p(baseMean)", m4)

# For M2 comparison on M4 subset, refit M2 on same data
m2_on_m4 = fit_logit(m4_df, ['log1p_in_degree', 'is_relay'])
print_model_results("M2 (refitted on M4 subset)", m2_on_m4)

# M5: GWAS_hit ~ log1p(in_degree) + is_relay + mean_signed_zscore
m5 = fit_logit(m5_df, ['log1p_in_degree', 'is_relay', 'mean_signed_zscore'])
print_model_results("M5: GWAS_hit ~ log1p(in_degree) + is_relay + mean_signed_zscore", m5)

# Refit M2 on M5's subset for fair comparison
m2_on_m5 = fit_logit(m5_df, ['log1p_in_degree', 'is_relay'])
print_model_results("M2 (refitted on M5 subset)", m2_on_m5)

# ---- Likelihood Ratio Tests ----
print("\n" + "=" * 80)
print("LIKELIHOOD RATIO TESTS")
print("=" * 80)


def lrt(reduced, full, label):
    """Likelihood ratio test between nested models."""
    lr_stat = -2 * (reduced.llf - full.llf)
    df_diff = full.df_model - reduced.df_model
    p_value = stats.chi2.sf(lr_stat, df_diff)
    print(f"\n{label}")
    print(f"  LR statistic: {lr_stat:.4f}")
    print(f"  Degrees of freedom: {df_diff}")
    print(f"  p-value: {p_value:.4e}")
    return lr_stat, df_diff, p_value


# M3 vs M2: Does DAI add to topology?
lrt(m2, m3, "M3 vs M2: Does DAI add predictive value beyond topology?")

# M4 vs M2: Does DAI + expression add? (on M4 subset)
lrt(m2_on_m4, m4, "M4 vs M2 (on M4 subset): Does DAI + baseMean add beyond topology?")

# M5 vs M2: Does mean z-score add? (on M5 subset)
lrt(m2_on_m5, m5, "M5 vs M2 (on M5 subset): Does mean_signed_zscore add beyond topology?")

# ---- AIC / BIC Summary Table ----
print("\n" + "=" * 80)
print("MODEL COMPARISON SUMMARY")
print("=" * 80)
models_info = {
    'M1': m1,
    'M2': m2,
    'M3': m3,
    'M4': m4,
    'M5': m5,
}
header = f"{'Model':<8} {'N':>6} {'AIC':>12} {'BIC':>12} {'LogLik':>12} {'df_model':>10}"
print(header)
print("-" * len(header))
for name, res in models_info.items():
    print(f"{name:<8} {res.nobs:>6.0f} {res.aic:>12.4f} {res.bic_deviance:>12.4f} "
          f"{res.llf:>12.4f} {res.df_model:>10.0f}")

# ============================================================================
# STEP 3: WITHIN IN-DEGREE DECILE ANALYSIS
# ============================================================================
print("\n" + "=" * 80)
print("STEP 3: WITHIN IN-DEGREE DECILE ANALYSIS")
print("=" * 80)

decile_df = model_df.copy()
decile_df['in_degree_decile'] = pd.qcut(decile_df['in_degree'], q=10, duplicates='drop')

# Sort the decile labels
decile_labels = sorted(decile_df['in_degree_decile'].dropna().unique(), key=lambda x: x.left)

print(f"\nNumber of in-degree decile bins: {len(decile_labels)}")
print(f"\n{'Decile':<30} {'N_GWAS':>8} {'N_nonGWAS':>10} {'Med_DAI_GWAS':>14} {'Med_DAI_nonGWAS':>16} {'U_stat':>10} {'p-value':>12}")
print("-" * 110)

decile_results = []
for dec in decile_labels:
    subset = decile_df[decile_df['in_degree_decile'] == dec]
    gwas = subset[subset['GWAS_hit'] == 1]['DAI']
    non_gwas = subset[subset['GWAS_hit'] == 0]['DAI']

    med_gwas = gwas.median() if len(gwas) > 0 else np.nan
    med_non_gwas = non_gwas.median() if len(non_gwas) > 0 else np.nan

    if len(gwas) > 0 and len(non_gwas) > 0:
        u_stat, p_val = stats.mannwhitneyu(gwas.dropna(), non_gwas.dropna(), alternative='two-sided')
    else:
        u_stat, p_val = np.nan, np.nan

    decile_results.append({
        'decile': str(dec),
        'n_gwas': len(gwas),
        'n_non_gwas': len(non_gwas),
        'med_dai_gwas': med_gwas,
        'med_dai_non_gwas': med_non_gwas,
        'dai_diff': med_gwas - med_non_gwas if not np.isnan(med_gwas) and not np.isnan(med_non_gwas) else np.nan,
        'u_stat': u_stat,
        'p_value': p_val,
    })
    print(f"{str(dec):<30} {len(gwas):>8} {len(non_gwas):>10} {med_gwas:>14.4f} {med_non_gwas:>16.4f} "
          f"{u_stat:>10.1f} {p_val:>12.4e}")

decile_res_df = pd.DataFrame(decile_results)
decile_res_df['decile_rank'] = range(1, len(decile_res_df) + 1)

# Spearman correlation: decile rank vs DAI difference
valid_deciles = decile_res_df.dropna(subset=['dai_diff'])
if len(valid_deciles) >= 3:
    spearman_r, spearman_p = stats.spearmanr(valid_deciles['decile_rank'], valid_deciles['dai_diff'])
else:
    spearman_r, spearman_p = np.nan, np.nan
print(f"\nSpearman correlation (decile rank vs DAI diff): r={spearman_r:.4f}, p={spearman_p:.4e}")

# ---- Figure: In-degree decile analysis ----
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Subplot 1: Grouped bar chart
x = np.arange(len(decile_res_df))
width = 0.35
bars1 = axes[0].bar(x - width / 2, decile_res_df['med_dai_gwas'], width, label='GWAS', color='#e74c3c', alpha=0.8)
bars2 = axes[0].bar(x + width / 2, decile_res_df['med_dai_non_gwas'], width, label='Non-GWAS', color='#3498db', alpha=0.8)
axes[0].set_xlabel('In-Degree Decile', fontsize=12)
axes[0].set_ylabel('Median DAI', fontsize=12)
axes[0].set_title('Median DAI by GWAS Status\nacross In-Degree Deciles', fontsize=13)
axes[0].set_xticks(x)
# Create shorter labels
short_labels = [f"D{i+1}" for i in range(len(decile_res_df))]
axes[0].set_xticklabels(short_labels, fontsize=10)
axes[0].legend(fontsize=11)
axes[0].axhline(y=0, color='gray', linestyle='--', linewidth=0.8)

# Add significance stars
for i, row in decile_res_df.iterrows():
    if not np.isnan(row['p_value']) and row['p_value'] < 0.05:
        max_y = max(row['med_dai_gwas'], row['med_dai_non_gwas'])
        axes[0].text(i, max_y + 0.02, '*', ha='center', fontsize=14, color='black')

# Subplot 2: Scatter plot of decile rank vs DAI difference
axes[1].scatter(valid_deciles['decile_rank'], valid_deciles['dai_diff'],
                s=80, color='#2c3e50', zorder=5)
# Add trend line
if len(valid_deciles) >= 2:
    z = np.polyfit(valid_deciles['decile_rank'], valid_deciles['dai_diff'], 1)
    p_line = np.poly1d(z)
    x_line = np.linspace(valid_deciles['decile_rank'].min(), valid_deciles['decile_rank'].max(), 100)
    axes[1].plot(x_line, p_line(x_line), 'r--', alpha=0.7, linewidth=2)

axes[1].set_xlabel('In-Degree Decile Rank', fontsize=12)
axes[1].set_ylabel('DAI Difference (GWAS - non-GWAS)', fontsize=12)
axes[1].set_title('DAI Difference vs In-Degree Decile', fontsize=13)
axes[1].axhline(y=0, color='gray', linestyle='--', linewidth=0.8)
axes[1].annotate(f"Spearman r = {spearman_r:.3f}\np = {spearman_p:.3e}",
                 xy=(0.05, 0.95), xycoords='axes fraction',
                 fontsize=11, verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
plt.savefig(f'{FIG_DIR}/dai_gwas_indegree_deciles.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"\nSaved: {FIG_DIR}/dai_gwas_indegree_deciles.png")

# ============================================================================
# STEP 4: WITHIN EXPRESSION TERTILE ANALYSIS
# ============================================================================
print("\n" + "=" * 80)
print("STEP 4: WITHIN EXPRESSION TERTILE ANALYSIS")
print("=" * 80)

tertile_df = model_df.dropna(subset=['baseMean']).copy()
tertile_df['expr_tertile'] = pd.qcut(tertile_df['baseMean'], q=3, labels=['Low', 'Medium', 'High'])

print(f"\nExpression tertile bin counts:")
print(tertile_df['expr_tertile'].value_counts().sort_index().to_string())

tertile_labels = ['Low', 'Medium', 'High']

print(f"\n{'Tertile':<12} {'N_GWAS':>8} {'N_nonGWAS':>10} {'Med_DAI_GWAS':>14} {'Med_DAI_nonGWAS':>16} {'U_stat':>10} {'p-value':>12}")
print("-" * 90)

tertile_results = []
for tert in tertile_labels:
    subset = tertile_df[tertile_df['expr_tertile'] == tert]
    gwas = subset[subset['GWAS_hit'] == 1]['DAI']
    non_gwas = subset[subset['GWAS_hit'] == 0]['DAI']

    med_gwas = gwas.median() if len(gwas) > 0 else np.nan
    med_non_gwas = non_gwas.median() if len(non_gwas) > 0 else np.nan

    if len(gwas) > 0 and len(non_gwas) > 0:
        u_stat, p_val = stats.mannwhitneyu(gwas.dropna(), non_gwas.dropna(), alternative='two-sided')
    else:
        u_stat, p_val = np.nan, np.nan

    tertile_results.append({
        'tertile': tert,
        'n_gwas': len(gwas),
        'n_non_gwas': len(non_gwas),
        'med_dai_gwas': med_gwas,
        'med_dai_non_gwas': med_non_gwas,
        'u_stat': u_stat,
        'p_value': p_val,
    })
    p_str = f"{p_val:.4e}" if not np.isnan(p_val) else "N/A"
    u_str = f"{u_stat:.1f}" if not np.isnan(u_stat) else "N/A"
    print(f"{tert:<12} {len(gwas):>8} {len(non_gwas):>10} {med_gwas:>14.4f} {med_non_gwas:>16.4f} "
          f"{u_str:>10} {p_str:>12}")

tertile_res_df = pd.DataFrame(tertile_results)

# ---- Figure: Expression tertile analysis ----
fig, ax = plt.subplots(figsize=(8, 6))
x = np.arange(len(tertile_res_df))
width = 0.35
ax.bar(x - width / 2, tertile_res_df['med_dai_gwas'], width, label='GWAS', color='#e74c3c', alpha=0.8)
ax.bar(x + width / 2, tertile_res_df['med_dai_non_gwas'], width, label='Non-GWAS', color='#3498db', alpha=0.8)
ax.set_xlabel('Expression Tertile (target_baseMean)', fontsize=12)
ax.set_ylabel('Median DAI', fontsize=12)
ax.set_title('Median DAI by GWAS Status\nacross Expression Tertiles', fontsize=13)
ax.set_xticks(x)
ax.set_xticklabels(tertile_res_df['tertile'], fontsize=11)
ax.legend(fontsize=11)
ax.axhline(y=0, color='gray', linestyle='--', linewidth=0.8)

# Add significance stars
for i, row in tertile_res_df.iterrows():
    if not np.isnan(row['p_value']) and row['p_value'] < 0.05:
        max_y = max(row['med_dai_gwas'], row['med_dai_non_gwas'])
        ax.text(i, max_y + 0.02, '*', ha='center', fontsize=14, color='black')

plt.tight_layout()
plt.savefig(f'{FIG_DIR}/dai_gwas_expression_tertiles.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"\nSaved: {FIG_DIR}/dai_gwas_expression_tertiles.png")

# ============================================================================
# ADDITIONAL FIGURE: MODEL COMPARISON
# ============================================================================
print("\n" + "=" * 80)
print("CREATING MODEL COMPARISON FIGURE")
print("=" * 80)

fig, axes = plt.subplots(1, 2, figsize=(16, 7))

# Subplot 1: AIC comparison bar chart
model_names = ['M1', 'M2', 'M3', 'M4', 'M5']
aic_vals = [models_info[m].aic for m in model_names]
colors = ['#95a5a6', '#3498db', '#e74c3c', '#2ecc71', '#9b59b6']
bars = axes[0].bar(model_names, aic_vals, color=colors, alpha=0.85, edgecolor='black', linewidth=0.5)
axes[0].set_xlabel('Model', fontsize=12)
axes[0].set_ylabel('AIC', fontsize=12)
axes[0].set_title('AIC Comparison Across Models', fontsize=13)
# Add value labels on bars
for bar, val in zip(bars, aic_vals):
    axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                 f'{val:.1f}', ha='center', va='bottom', fontsize=10)
# Adjust y-axis to show differences
min_aic = min(aic_vals)
max_aic = max(aic_vals)
y_range = max_aic - min_aic
axes[0].set_ylim(min_aic - y_range * 0.3, max_aic + y_range * 0.3)

# Add model descriptions as annotations
model_desc = {
    'M1': 'log1p(in_deg)',
    'M2': '+ is_relay',
    'M3': '+ DAI',
    'M4': '+ DAI + baseMean',
    'M5': '+ mean_zscore',
}
for i, (name, desc) in enumerate(model_desc.items()):
    axes[0].text(i, min_aic - y_range * 0.15, desc, ha='center', va='top',
                 fontsize=8, rotation=0, style='italic')

# Subplot 2: Forest plot for M3 and M4
# Exclude intercept for readability
def get_or_data(result, label):
    """Extract OR and CI data from a model result."""
    coef = result.params
    ci = result.conf_int()
    data = []
    for var_name in coef.index:
        if var_name == 'const':
            continue
        or_val = np.exp(coef[var_name])
        ci_lo = np.exp(ci[0][var_name])
        ci_hi = np.exp(ci[1][var_name])
        pval = result.pvalues[var_name]
        data.append({
            'variable': var_name,
            'model': label,
            'OR': or_val,
            'CI_lo': ci_lo,
            'CI_hi': ci_hi,
            'p_value': pval,
        })
    return data

forest_data = get_or_data(m3, 'M3') + get_or_data(m4, 'M4')
forest_df = pd.DataFrame(forest_data)

# Create forest plot
variables = forest_df['variable'].unique()
model_colors = {'M3': '#e74c3c', 'M4': '#2ecc71'}
y_positions = {}
y_offset = 0
y_labels = []
y_ticks = []

for var in variables:
    var_data = forest_df[forest_df['variable'] == var]
    for j, (_, row) in enumerate(var_data.iterrows()):
        y_pos = y_offset + j * 0.35
        y_positions[(var, row['model'])] = y_pos
        axes[1].plot(row['OR'], y_pos, 'o', color=model_colors[row['model']],
                     markersize=8, zorder=5)
        axes[1].plot([row['CI_lo'], row['CI_hi']], [y_pos, y_pos], '-',
                     color=model_colors[row['model']], linewidth=2, zorder=4)
        # Add p-value annotation
        sig_marker = ''
        if row['p_value'] < 0.001:
            sig_marker = '***'
        elif row['p_value'] < 0.01:
            sig_marker = '**'
        elif row['p_value'] < 0.05:
            sig_marker = '*'
        if sig_marker:
            axes[1].text(row['CI_hi'] + 0.01 * (axes[1].get_xlim()[1] - axes[1].get_xlim()[0] if axes[1].get_xlim()[1] != axes[1].get_xlim()[0] else 1),
                         y_pos, sig_marker, va='center', fontsize=10, color=model_colors[row['model']])

    y_labels.append(var)
    y_ticks.append(y_offset + 0.175)
    y_offset += 1.0

axes[1].axvline(x=1, color='gray', linestyle='--', linewidth=1, zorder=1)
axes[1].set_yticks(y_ticks)
axes[1].set_yticklabels(y_labels, fontsize=10)
axes[1].set_xlabel('Odds Ratio (95% CI)', fontsize=12)
axes[1].set_title('Forest Plot: Odds Ratios for M3 and M4', fontsize=13)

# Legend for forest plot
from matplotlib.lines import Line2D
legend_elements = [Line2D([0], [0], marker='o', color='w', markerfacecolor='#e74c3c',
                           markersize=8, label='M3'),
                   Line2D([0], [0], marker='o', color='w', markerfacecolor='#2ecc71',
                           markersize=8, label='M4')]
axes[1].legend(handles=legend_elements, loc='lower right', fontsize=11)

plt.tight_layout()
plt.savefig(f'{FIG_DIR}/logistic_regression_model_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved: {FIG_DIR}/logistic_regression_model_comparison.png")

print("\n" + "=" * 80)
print("ANALYSIS COMPLETE")
print("=" * 80)
