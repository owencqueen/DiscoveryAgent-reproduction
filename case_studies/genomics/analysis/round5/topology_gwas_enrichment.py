import os
#!/usr/bin/env python3
"""
Topology-GWAS Enrichment Analysis
Tests whether relay nodes (high in-degree AND high out-degree) are enriched for autoimmune GWAS variants.
"""

import pandas as pd
import numpy as np
import anndata as ad
from scipy.stats import fisher_exact
from statsmodels.stats.multitest import multipletests
import statsmodels.api as sm
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
import warnings
import os
warnings.filterwarnings('ignore')

BASE = (os.environ["REPRO_DATA_ROOT"] + '')
os.makedirs(f'{BASE}/reports/figures', exist_ok=True)
os.makedirs(f'{BASE}/reports/analysis_outputs', exist_ok=True)

results_lines = []
def log(msg):
    results_lines.append(msg)
    print(msg)

# ======================================================================
# STEP 1: Create GWAS gene set
# ======================================================================
log("=" * 80)
log("STEP 1: Create GWAS Gene Set")
log("=" * 80)

gwas = pd.read_csv(f'{BASE}/data/downloads/autoimmune_l2g_predictions.csv')
log(f"Total GWAS L2G predictions: {len(gwas)}")
log(f"Unique diseases: {gwas['disease_name'].nunique()}")
log(f"Diseases: {sorted(gwas['disease_name'].unique())}")

# Filter for L2G score > 0.5
gwas_filtered = gwas[gwas['l2g_score'] > 0.5]
log(f"\nAfter L2G > 0.5 filter: {len(gwas_filtered)} predictions")
log(f"Unique genes with L2G > 0.5: {gwas_filtered['gene_symbol'].nunique()}")

# Binary: GWAS_hit if gene appears with score > 0.5 for ANY autoimmune disease
gwas_hits = set(gwas_filtered['gene_symbol'].unique())
log(f"Number of GWAS hit genes: {len(gwas_hits)}")

# Count diseases per gene
disease_counts = gwas_filtered.groupby('gene_symbol')['disease_id'].nunique().reset_index()
disease_counts.columns = ['gene_name', 'n_diseases']
log(f"\nDisease count distribution:")
log(f"  Mean: {disease_counts['n_diseases'].mean():.2f}")
log(f"  Median: {disease_counts['n_diseases'].median():.0f}")
log(f"  Max: {disease_counts['n_diseases'].max()} ({disease_counts.loc[disease_counts['n_diseases'].idxmax(), 'gene_name']})")

# ======================================================================
# STEP 2: Classify genes into topological categories
# ======================================================================
log("\n" + "=" * 80)
log("STEP 2: Classify Genes into Topological Categories")
log("=" * 80)

topo = pd.read_csv(f'{BASE}/reports/gene_topology.csv')
log(f"Total gene-condition pairs: {len(topo)}")
log(f"Conditions: {topo['condition'].unique()}")
log(f"Unique genes: {topo['gene_name'].nunique()}")

# Add GWAS hit info
topo['GWAS_hit'] = topo['gene_name'].isin(gwas_hits).astype(int)
topo = topo.merge(disease_counts, on='gene_name', how='left')
topo['n_diseases'] = topo['n_diseases'].fillna(0).astype(int)

log(f"\nGWAS hits in topology data: {topo[topo['GWAS_hit']==1]['gene_name'].nunique()} unique genes")
log(f"GWAS hits NOT in topology data: {len(gwas_hits - set(topo['gene_name'].unique()))}")

# --- Median split classification ---
for cond in ['Rest', 'Stim8hr', 'Stim48hr']:
    mask = topo['condition'] == cond
    sub = topo.loc[mask].copy()

    in_med = sub['in_degree'].median()
    out_med = sub['out_degree'].median()

    log(f"\n--- {cond} ---")
    log(f"  Median in_degree: {in_med}, Median out_degree: {out_med}")
    log(f"  Mean in_degree: {sub['in_degree'].mean():.1f}, Mean out_degree: {sub['out_degree'].mean():.1f}")

    # Median split categories
    conditions_list = [
        (sub['out_degree'] > out_med) & (sub['in_degree'] <= in_med),
        (sub['out_degree'] > out_med) & (sub['in_degree'] > in_med),
        (sub['out_degree'] <= out_med) & (sub['in_degree'] > in_med),
        (sub['out_degree'] <= out_med) & (sub['in_degree'] <= in_med)
    ]
    choices = ['Upstream controller', 'Relay/amplifier', 'Terminal effector', 'Peripheral']
    topo.loc[mask, 'median_category'] = np.select(conditions_list, choices, default='Peripheral')

    # Quartile split categories
    in_q75 = sub['in_degree'].quantile(0.75)
    out_q75 = sub['out_degree'].quantile(0.75)

    log(f"  75th pct in_degree: {in_q75}, 75th pct out_degree: {out_q75}")

    q_conditions = [
        (sub['out_degree'] >= out_q75) & (sub['in_degree'] >= in_q75),
        (sub['out_degree'] >= out_q75) & (sub['in_degree'] < in_q75),
        (sub['out_degree'] < out_q75) & (sub['in_degree'] >= in_q75),
        (sub['out_degree'] < out_q75) & (sub['in_degree'] < in_q75)
    ]
    q_choices = ['Strong relay', 'Strong upstream', 'Strong terminal', 'Other']
    topo.loc[mask, 'quartile_category'] = np.select(q_conditions, q_choices, default='Other')

# Report category counts
log("\n--- Category Counts (Median Split) ---")
for cond in ['Rest', 'Stim8hr', 'Stim48hr']:
    sub = topo[topo['condition'] == cond]
    log(f"\n{cond}:")
    vc = sub['median_category'].value_counts()
    for cat in ['Upstream controller', 'Relay/amplifier', 'Terminal effector', 'Peripheral']:
        cnt = vc.get(cat, 0)
        gwas_cnt = sub[(sub['median_category'] == cat) & (sub['GWAS_hit'] == 1)].shape[0]
        log(f"  {cat}: {cnt} genes ({gwas_cnt} GWAS hits, {gwas_cnt/cnt*100:.1f}%)")

log("\n--- Category Counts (Quartile Split) ---")
for cond in ['Rest', 'Stim8hr', 'Stim48hr']:
    sub = topo[topo['condition'] == cond]
    log(f"\n{cond}:")
    vc = sub['quartile_category'].value_counts()
    for cat in ['Strong relay', 'Strong upstream', 'Strong terminal', 'Other']:
        cnt = vc.get(cat, 0)
        gwas_cnt = sub[(sub['quartile_category'] == cat) & (sub['GWAS_hit'] == 1)].shape[0]
        log(f"  {cat}: {cnt} genes ({gwas_cnt} GWAS hits, {gwas_cnt/cnt*100:.1f}%)")

# ======================================================================
# STEP 3: Fisher's exact tests for GWAS enrichment
# ======================================================================
log("\n" + "=" * 80)
log("STEP 3: GWAS Enrichment per Topological Category (Fisher's Exact Tests)")
log("=" * 80)

def fishers_test(cat_gwas, cat_total, bg_gwas, bg_total):
    """2x2 Fisher's exact: category vs background."""
    cat_no_gwas = cat_total - cat_gwas
    bg_no_gwas = bg_total - bg_gwas
    table = [[cat_gwas, cat_no_gwas], [bg_gwas, bg_no_gwas]]
    odds, pval = fisher_exact(table, alternative='two-sided')
    return odds, pval

# Store all Fisher results for FDR
all_fisher_results = []

for scheme, cat_col, categories in [
    ('Median split', 'median_category', ['Upstream controller', 'Relay/amplifier', 'Terminal effector', 'Peripheral']),
    ('Quartile split', 'quartile_category', ['Strong relay', 'Strong upstream', 'Strong terminal', 'Other'])
]:
    log(f"\n--- {scheme} ---")
    for cond in ['Rest', 'Stim8hr', 'Stim48hr']:
        sub = topo[topo['condition'] == cond]
        total_gwas = sub['GWAS_hit'].sum()
        total_n = len(sub)

        log(f"\n  {cond} (total: {total_n}, GWAS hits: {total_gwas}, rate: {total_gwas/total_n*100:.1f}%):")

        # Each category vs ALL OTHER
        for cat in categories:
            cat_mask = sub[cat_col] == cat
            cat_gwas = sub.loc[cat_mask, 'GWAS_hit'].sum()
            cat_n = cat_mask.sum()
            bg_gwas = total_gwas - cat_gwas
            bg_n = total_n - cat_n

            odds, pval = fishers_test(cat_gwas, cat_n, bg_gwas, bg_n)
            frac = cat_gwas / cat_n if cat_n > 0 else 0

            all_fisher_results.append({
                'scheme': scheme, 'condition': cond, 'comparison': f'{cat} vs rest',
                'category': cat, 'cat_gwas': cat_gwas, 'cat_n': cat_n,
                'bg_gwas': bg_gwas, 'bg_n': bg_n,
                'odds_ratio': odds, 'p_value': pval, 'fraction': frac
            })
            log(f"    {cat}: {cat_gwas}/{cat_n} ({frac*100:.1f}%), OR={odds:.3f}, p={pval:.2e}")

        # Pairwise: relay vs each other category
        relay_cat = 'Relay/amplifier' if scheme == 'Median split' else 'Strong relay'
        relay_mask = sub[cat_col] == relay_cat
        relay_gwas = sub.loc[relay_mask, 'GWAS_hit'].sum()
        relay_n = relay_mask.sum()

        log(f"\n    Pairwise comparisons (relay vs others):")
        for other_cat in categories:
            if other_cat == relay_cat:
                continue
            other_mask = sub[cat_col] == other_cat
            other_gwas = sub.loc[other_mask, 'GWAS_hit'].sum()
            other_n = other_mask.sum()

            odds, pval = fishers_test(relay_gwas, relay_n, other_gwas, other_n)
            all_fisher_results.append({
                'scheme': scheme, 'condition': cond, 'comparison': f'{relay_cat} vs {other_cat}',
                'category': relay_cat, 'cat_gwas': relay_gwas, 'cat_n': relay_n,
                'bg_gwas': other_gwas, 'bg_n': other_n,
                'odds_ratio': odds, 'p_value': pval,
                'fraction': relay_gwas/relay_n if relay_n > 0 else 0
            })
            log(f"      {relay_cat} vs {other_cat}: OR={odds:.3f}, p={pval:.2e}")

# Apply FDR correction per condition
fisher_df = pd.DataFrame(all_fisher_results)
for cond in ['Rest', 'Stim8hr', 'Stim48hr']:
    cond_mask = fisher_df['condition'] == cond
    pvals = fisher_df.loc[cond_mask, 'p_value'].values
    if len(pvals) > 0:
        _, fdr_vals, _, _ = multipletests(pvals, method='fdr_bh')
        fisher_df.loc[cond_mask, 'fdr'] = fdr_vals

log("\n--- FDR-corrected results (category vs rest comparisons only) ---")
for scheme in ['Median split', 'Quartile split']:
    log(f"\n{scheme}:")
    sub_df = fisher_df[(fisher_df['scheme'] == scheme) & (fisher_df['comparison'].str.contains('vs rest'))]
    for _, row in sub_df.iterrows():
        log(f"  {row['condition']} | {row['category']}: OR={row['odds_ratio']:.3f}, p={row['p_value']:.2e}, FDR={row['fdr']:.2e}")

log("\n--- FDR-corrected results (pairwise relay comparisons) ---")
for scheme in ['Median split', 'Quartile split']:
    log(f"\n{scheme}:")
    sub_df = fisher_df[(fisher_df['scheme'] == scheme) & (~fisher_df['comparison'].str.contains('vs rest'))]
    for _, row in sub_df.iterrows():
        log(f"  {row['condition']} | {row['comparison']}: OR={row['odds_ratio']:.3f}, p={row['p_value']:.2e}, FDR={row['fdr']:.2e}")

fisher_df.to_csv(f'{BASE}/reports/analysis_outputs/fisher_test_results.csv', index=False)

# ======================================================================
# STEP 4: Logistic regression (Stim48hr)
# ======================================================================
log("\n" + "=" * 80)
log("STEP 4: Logistic Regression (Stim48hr)")
log("=" * 80)

stim48 = topo[topo['condition'] == 'Stim48hr'].copy()
log(f"Stim48hr genes: {len(stim48)}, GWAS hits: {stim48['GWAS_hit'].sum()}")

# Load AnnData for baseMean
log("Loading AnnData for baseMean...")
adata = ad.read_h5ad((os.environ["REPRO_DATA_ROOT"] + '/data/GWCD4i.DE_stats.h5ad'))

# Get mean baseMean per measured gene in Stim48hr
stim48_mask = adata.obs['culture_condition'] == 'Stim48hr'
adata_s48 = adata[stim48_mask]
basemean_matrix = adata_s48.layers['baseMean']
if hasattr(basemean_matrix, 'toarray'):
    basemean_matrix = basemean_matrix.toarray()
mean_basemean = pd.DataFrame({
    'gene_name': adata_s48.var['gene_name'].values,
    'mean_baseMean': np.nanmean(basemean_matrix, axis=0)
})
log(f"Computed mean baseMean for {len(mean_basemean)} genes")

stim48 = stim48.merge(mean_basemean, on='gene_name', how='left')
stim48['log_baseMean'] = np.log1p(stim48['mean_baseMean'])
log(f"Genes with baseMean data: {stim48['mean_baseMean'].notna().sum()}")

# Prepare variables
stim48['log_in_degree'] = np.log1p(stim48['in_degree'])
stim48['log_out_degree'] = np.log1p(stim48['out_degree'])
stim48['interaction'] = stim48['log_in_degree'] * stim48['log_out_degree']

# Drop rows with missing baseMean for model fitting
stim48_clean = stim48.dropna(subset=['log_baseMean']).copy()
log(f"Genes for regression (after dropping NaN baseMean): {len(stim48_clean)}")

y = stim48_clean['GWAS_hit'].values

# Model 1: out_degree only
log("\n--- Model 1: GWAS_hit ~ out_degree ---")
X1 = sm.add_constant(stim48_clean[['log_out_degree']])
m1 = sm.Logit(y, X1).fit(disp=0)
log(m1.summary2().tables[1].to_string())
log(f"AIC: {m1.aic:.2f}, Pseudo-R²: {m1.prsquared:.6f}")

# Model 2: in_degree + out_degree
log("\n--- Model 2: GWAS_hit ~ in_degree + out_degree ---")
X2 = sm.add_constant(stim48_clean[['log_in_degree', 'log_out_degree']])
m2 = sm.Logit(y, X2).fit(disp=0)
log(m2.summary2().tables[1].to_string())
log(f"AIC: {m2.aic:.2f}, Pseudo-R²: {m2.prsquared:.6f}")

# Model 3: in_degree + out_degree + interaction + baseMean
log("\n--- Model 3: GWAS_hit ~ in_degree + out_degree + interaction + baseMean ---")
X3 = sm.add_constant(stim48_clean[['log_in_degree', 'log_out_degree', 'interaction', 'log_baseMean']])
m3 = sm.Logit(y, X3).fit(disp=0)
log(m3.summary2().tables[1].to_string())
log(f"AIC: {m3.aic:.2f}, Pseudo-R²: {m3.prsquared:.6f}")

# Model 4: topological category (categorical)
log("\n--- Model 4: GWAS_hit ~ topological_category ---")
cat_dummies = pd.get_dummies(stim48_clean['median_category'], drop_first=False).astype(float)
# Use Peripheral as reference
ref = 'Peripheral'
other_cats = [c for c in cat_dummies.columns if c != ref]
X4 = sm.add_constant(cat_dummies[other_cats].astype(float))
m4 = sm.Logit(y, X4).fit(disp=0)
log(m4.summary2().tables[1].to_string())
log(f"AIC: {m4.aic:.2f}, Pseudo-R²: {m4.prsquared:.6f}")

# Cross-validated ROC-AUC
log("\n--- 5-Fold Cross-Validated ROC-AUC ---")
scaler = StandardScaler()

for name, X_data in [('Model 1 (out_degree)', stim48_clean[['log_out_degree']]),
                      ('Model 2 (in+out degree)', stim48_clean[['log_in_degree', 'log_out_degree']]),
                      ('Model 3 (in+out+interaction+baseMean)', stim48_clean[['log_in_degree', 'log_out_degree', 'interaction', 'log_baseMean']]),
                      ('Model 4 (category)', cat_dummies[other_cats])]:
    X_scaled = scaler.fit_transform(X_data)
    clf = LogisticRegression(max_iter=1000, solver='lbfgs')
    scores = cross_val_score(clf, X_scaled, y, cv=5, scoring='roc_auc')
    log(f"  {name}: AUC = {scores.mean():.4f} ± {scores.std():.4f}")

# ======================================================================
# STEP 5: Visualizations
# ======================================================================
log("\n" + "=" * 80)
log("STEP 5: Visualizations")
log("=" * 80)

# --- Figure 1: 2D scatter of topology colored by GWAS hit ---
fig, ax = plt.subplots(1, 1, figsize=(10, 8))
s48 = stim48.copy()
s48['log_in'] = np.log10(s48['in_degree'] + 1)
s48['log_out'] = np.log10(s48['out_degree'] + 1)

# Plot non-GWAS first, then GWAS on top
non_gwas = s48[s48['GWAS_hit'] == 0]
gwas_pts = s48[s48['GWAS_hit'] == 1]

ax.scatter(non_gwas['log_out'], non_gwas['log_in'], c='#cccccc', alpha=0.3, s=10, label=f'Non-GWAS (n={len(non_gwas)})', rasterized=True)
ax.scatter(gwas_pts['log_out'], gwas_pts['log_in'], c='#e74c3c', alpha=0.6, s=15, label=f'GWAS hit (n={len(gwas_pts)})', zorder=5, rasterized=True)

# Overlay median split lines
in_med = s48['in_degree'].median()
out_med = s48['out_degree'].median()
ax.axhline(y=np.log10(in_med + 1), color='blue', linestyle='--', alpha=0.7, label=f'Median in-degree ({in_med:.0f})')
ax.axvline(x=np.log10(out_med + 1), color='green', linestyle='--', alpha=0.7, label=f'Median out-degree ({out_med:.0f})')

# Add quadrant labels
xlim = ax.get_xlim()
ylim = ax.get_ylim()
ax.text(np.log10(out_med+1)/2, ylim[1]*0.95, 'Terminal\neffector', ha='center', va='top', fontsize=9, color='purple', fontweight='bold')
ax.text((np.log10(out_med+1) + xlim[1])/2, ylim[1]*0.95, 'Relay/\namplifier', ha='center', va='top', fontsize=9, color='darkred', fontweight='bold')
ax.text(np.log10(out_med+1)/2, ylim[0] + 0.05, 'Peripheral', ha='center', va='bottom', fontsize=9, color='gray', fontweight='bold')
ax.text((np.log10(out_med+1) + xlim[1])/2, ylim[0] + 0.05, 'Upstream\ncontroller', ha='center', va='bottom', fontsize=9, color='darkblue', fontweight='bold')

ax.set_xlabel('log10(out_degree + 1)', fontsize=12)
ax.set_ylabel('log10(in_degree + 1)', fontsize=12)
ax.set_title('Stim48hr: Gene Topology vs GWAS Hit Status', fontsize=14)
ax.legend(fontsize=10, loc='upper right')
plt.tight_layout()
plt.savefig(f'{BASE}/reports/figures/topology_gwas_scatter.png', dpi=200, bbox_inches='tight')
plt.close()
log("Saved: topology_gwas_scatter.png")

# --- Figure 2: Bar plot of GWAS enrichment per category ---
fig, axes = plt.subplots(1, 3, figsize=(16, 6), sharey=True)
colors = {'Upstream controller': '#3498db', 'Relay/amplifier': '#e74c3c', 'Terminal effector': '#9b59b6', 'Peripheral': '#95a5a6'}
categories_order = ['Peripheral', 'Upstream controller', 'Terminal effector', 'Relay/amplifier']

for idx, cond in enumerate(['Rest', 'Stim8hr', 'Stim48hr']):
    ax = axes[idx]
    sub = topo[topo['condition'] == cond]
    fracs = []
    ci_low = []
    ci_high = []

    for cat in categories_order:
        cat_sub = sub[sub['median_category'] == cat]
        n = len(cat_sub)
        k = cat_sub['GWAS_hit'].sum()
        p = k / n if n > 0 else 0
        # Wilson score interval
        z = 1.96
        denom = 1 + z**2 / n
        center = (p + z**2 / (2*n)) / denom
        margin = z * np.sqrt((p*(1-p) + z**2/(4*n)) / n) / denom
        fracs.append(p)
        ci_low.append(max(0, center - margin))
        ci_high.append(min(1, center + margin))

    errs = [[f - cl for f, cl in zip(fracs, ci_low)], [ch - f for f, ch in zip(fracs, ci_high)]]
    bars = ax.bar(range(len(categories_order)), fracs, color=[colors[c] for c in categories_order],
                  yerr=errs, capsize=5, edgecolor='black', linewidth=0.5)
    ax.set_xticks(range(len(categories_order)))
    ax.set_xticklabels([c.replace('/', '/\n') for c in categories_order], fontsize=9, rotation=0)
    ax.set_title(cond, fontsize=13, fontweight='bold')
    if idx == 0:
        ax.set_ylabel('Fraction GWAS Hits', fontsize=12)

    # Add count labels
    for i, cat in enumerate(categories_order):
        cat_sub = sub[sub['median_category'] == cat]
        n = len(cat_sub)
        k = cat_sub['GWAS_hit'].sum()
        ax.text(i, fracs[i] + errs[1][i] + 0.005, f'{k}/{n}', ha='center', va='bottom', fontsize=7)

plt.suptitle('GWAS Enrichment by Topological Category (Median Split)', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{BASE}/reports/figures/topology_gwas_enrichment.png', dpi=200, bbox_inches='tight')
plt.close()
log("Saved: topology_gwas_enrichment.png")

# --- Figure 3: Heatmap of odds ratios ---
# Use category vs rest comparisons
or_data = fisher_df[fisher_df['comparison'].str.contains('vs rest')].copy()

fig, axes = plt.subplots(1, 2, figsize=(16, 5))

for ax_idx, (scheme, cat_col) in enumerate([
    ('Median split', ['Peripheral', 'Upstream controller', 'Terminal effector', 'Relay/amplifier']),
    ('Quartile split', ['Other', 'Strong upstream', 'Strong terminal', 'Strong relay'])
]):
    ax = axes[ax_idx]
    sub_or = or_data[or_data['scheme'] == scheme]

    # Create matrix
    matrix = pd.DataFrame(index=['Rest', 'Stim8hr', 'Stim48hr'], columns=cat_col, dtype=float)
    pval_matrix = pd.DataFrame(index=['Rest', 'Stim8hr', 'Stim48hr'], columns=cat_col, dtype=float)

    for _, row in sub_or.iterrows():
        matrix.loc[row['condition'], row['category']] = row['odds_ratio']
        pval_matrix.loc[row['condition'], row['category']] = row['fdr']

    # Create annotation with significance stars
    annot = matrix.copy().astype(str)
    for r in matrix.index:
        for c in matrix.columns:
            or_val = matrix.loc[r, c]
            fdr_val = pval_matrix.loc[r, c]
            stars = ''
            if fdr_val < 0.001:
                stars = '***'
            elif fdr_val < 0.01:
                stars = '**'
            elif fdr_val < 0.05:
                stars = '*'
            annot.loc[r, c] = f'{or_val:.2f}{stars}'

    sns.heatmap(matrix.astype(float), annot=annot, fmt='', cmap='RdBu_r', center=1.0,
                vmin=0.5, vmax=2.0, ax=ax, linewidths=0.5, cbar_kws={'label': 'Odds Ratio'})
    ax.set_title(f'{scheme}', fontsize=12, fontweight='bold')
    ax.set_ylabel('')

plt.suptitle('GWAS Enrichment Odds Ratios by Topological Category\n(* FDR<0.05, ** FDR<0.01, *** FDR<0.001)', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{BASE}/reports/figures/topology_gwas_odds_ratios.png', dpi=200, bbox_inches='tight')
plt.close()
log("Saved: topology_gwas_odds_ratios.png")

# ======================================================================
# Save all results
# ======================================================================
with open(f'{BASE}/reports/analysis_outputs/topology_gwas_results.txt', 'w') as f:
    f.write('\n'.join(results_lines))
log(f"\nAll results saved to reports/analysis_outputs/topology_gwas_results.txt")

# Save topology with categories
topo.to_csv(f'{BASE}/data/processed/gene_topology_with_gwas.csv', index=False)
log("Saved: data/processed/gene_topology_with_gwas.csv")

print("\n\nDONE - All analysis complete!")
