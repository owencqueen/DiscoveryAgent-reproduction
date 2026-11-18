import os
#!/usr/bin/env python3
"""
Comprehensive constraint-network analyses for Stim8hr condition.
Analyses 1-6: Decile tests, quartile enrichment, topological categories,
matched-pair relay vs terminal, PID vs GWAS, logistic regression.
"""

import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import mannwhitneyu, kruskal, wilcoxon, spearmanr, fisher_exact
from statsmodels.stats.multitest import multipletests
import statsmodels.api as sm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import os
import json

warnings.filterwarnings('ignore')

# Paths
BASE = (os.environ["REPRO_DATA_ROOT"] + '')
DATA_PATH = os.path.join(BASE, 'data/processed/merged_constraint_network.csv')
L2G_PATH = os.path.join(BASE, 'data/downloads/autoimmune_l2g_unique_genes.csv')
OUT_DIR = os.path.join(BASE, 'reports/analysis_outputs')
FIG_DIR = os.path.join(BASE, 'reports/figures')
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

# Load and filter
df_all = pd.read_csv(DATA_PATH)
print(f"Full dataset: {len(df_all)} rows, conditions: {df_all['condition'].unique()}")
df = df_all[df_all['condition'] == 'Stim8hr'].copy()
print(f"Stim8hr subset: {len(df)} rows")
print(f"Columns: {list(df.columns)}")
print(f"s_het non-null: {df['s_het'].notna().sum()}")
print(f"pLI non-null: {df['pLI'].notna().sum()}")
print(f"LOEUF non-null: {df['LOEUF'].notna().sum()}")

# Drop rows with missing s_het for constraint analyses
df_constraint = df.dropna(subset=['s_het']).copy()
print(f"Rows with s_het: {len(df_constraint)}")

results_text = []
results_text.append("=" * 80)
results_text.append("COMPREHENSIVE CONSTRAINT-NETWORK ANALYSES - Stim8hr Condition")
results_text.append("=" * 80)
results_text.append(f"Total Stim8hr genes: {len(df)}")
results_text.append(f"Genes with s_het: {len(df_constraint)}")
results_text.append("")

# ============================================================================
# ANALYSIS 1: Directional decile-based tests
# ============================================================================
results_text.append("=" * 80)
results_text.append("ANALYSIS 1: Directional Decile-Based Tests")
results_text.append("=" * 80)

for degree_type in ['in_degree', 'out_degree']:
    results_text.append(f"\n--- {degree_type} ---")

    # Create deciles
    df_constraint[f'{degree_type}_decile'] = pd.qcut(
        df_constraint[degree_type].rank(method='first'),
        q=10, labels=False
    ) + 1  # 1-10

    decile_stats = df_constraint.groupby(f'{degree_type}_decile').agg(
        n_genes=('s_het', 'count'),
        median_s_het=('s_het', 'median'),
        mean_s_het=('s_het', 'mean'),
        median_pLI=('pLI', 'median'),
        median_LOEUF=('LOEUF', 'median'),
        median_degree=(degree_type, 'median')
    ).reset_index()

    results_text.append(f"\nDecile statistics for {degree_type}:")
    results_text.append(decile_stats.to_string(index=False))

    # Spearman correlation of decile rank vs median s_het
    rho, p_spearman = spearmanr(decile_stats[f'{degree_type}_decile'], decile_stats['median_s_het'])
    results_text.append(f"\nSpearman correlation (decile rank vs median s_het): rho={rho:.4f}, p={p_spearman:.6f}")

    # One-sided Mann-Whitney: top decile vs bottom 9
    top_decile = df_constraint[df_constraint[f'{degree_type}_decile'] == 10]['s_het']
    bottom_9 = df_constraint[df_constraint[f'{degree_type}_decile'] < 10]['s_het']
    stat_mw, p_mw = mannwhitneyu(top_decile, bottom_9, alternative='greater')
    results_text.append(f"Mann-Whitney (top decile vs bottom 9, one-sided greater): U={stat_mw:.0f}, p={p_mw:.6f}")

    d1 = decile_stats[decile_stats[f'{degree_type}_decile'] == 1]['median_s_het'].values[0]
    d10 = decile_stats[decile_stats[f'{degree_type}_decile'] == 10]['median_s_het'].values[0]
    results_text.append(f"Decile 1 (lowest {degree_type}) median s_het: {d1:.6f}")
    results_text.append(f"Decile 10 (highest {degree_type}) median s_het: {d10:.6f}")
    results_text.append(f"Fold change (D10/D1): {d10/d1:.4f}" if d1 > 0 else f"D10-D1 diff: {d10-d1:.6f}")

    # Also do Spearman for pLI and LOEUF
    rho_pli, p_pli = spearmanr(decile_stats[f'{degree_type}_decile'], decile_stats['median_pLI'])
    rho_loeuf, p_loeuf = spearmanr(decile_stats[f'{degree_type}_decile'], decile_stats['median_LOEUF'])
    results_text.append(f"Spearman (decile rank vs median pLI): rho={rho_pli:.4f}, p={p_pli:.6f}")
    results_text.append(f"Spearman (decile rank vs median LOEUF): rho={rho_loeuf:.4f}, p={p_loeuf:.6f}")

# Plot decile trends
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
for i, degree_type in enumerate(['in_degree', 'out_degree']):
    decile_stats = df_constraint.groupby(f'{degree_type}_decile').agg(
        median_s_het=('s_het', 'median'),
        median_pLI=('pLI', 'median'),
        median_LOEUF=('LOEUF', 'median')
    ).reset_index()

    for j, (metric, label) in enumerate([('median_s_het', 's_het'), ('median_pLI', 'pLI'), ('median_LOEUF', 'LOEUF')]):
        ax = axes[i, j]
        ax.bar(decile_stats[f'{degree_type}_decile'], decile_stats[metric], color='steelblue', edgecolor='black')
        ax.set_xlabel(f'{degree_type} Decile')
        ax.set_ylabel(f'Median {label}')
        ax.set_title(f'{degree_type} vs {label}')
        ax.set_xticks(range(1, 11))

plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'analysis1_decile_trends.png'), dpi=150, bbox_inches='tight')
plt.close()
print("Analysis 1 done.")

# ============================================================================
# ANALYSIS 2: Top quartile enrichment (one-sided tests)
# ============================================================================
results_text.append("\n" + "=" * 80)
results_text.append("ANALYSIS 2: Top Quartile Enrichment (One-Sided Tests)")
results_text.append("=" * 80)

# Define thresholds
s_het_75 = df_constraint['s_het'].quantile(0.75)
in_deg_75 = df_constraint['in_degree'].quantile(0.75)
out_deg_75 = df_constraint['out_degree'].quantile(0.75)

df_constraint['high_s_het'] = df_constraint['s_het'] >= s_het_75
df_constraint['high_in_degree'] = df_constraint['in_degree'] >= in_deg_75
df_constraint['high_out_degree'] = df_constraint['out_degree'] >= out_deg_75
df_constraint['high_pLI'] = df_constraint['pLI'] >= 0.9

results_text.append(f"\nThresholds: s_het 75th={s_het_75:.6f}, in_degree 75th={in_deg_75:.0f}, out_degree 75th={out_deg_75:.0f}")
results_text.append(f"N high s_het: {df_constraint['high_s_het'].sum()}")
results_text.append(f"N high in-degree: {df_constraint['high_in_degree'].sum()}")
results_text.append(f"N high out-degree: {df_constraint['high_out_degree'].sum()}")
results_text.append(f"N high pLI (>=0.9): {df_constraint['high_pLI'].sum()}")

# Fisher's exact tests
for degree_type in ['in_degree', 'out_degree']:
    high_deg_col = f'high_{degree_type}'

    # s_het enrichment
    ct = pd.crosstab(df_constraint[high_deg_col], df_constraint['high_s_het'])
    # Table: [[~high_deg & ~high_shet, ~high_deg & high_shet], [high_deg & ~high_shet, high_deg & high_shet]]
    table = [[ct.iloc[0, 0], ct.iloc[0, 1]], [ct.iloc[1, 0], ct.iloc[1, 1]]]
    odds_ratio, p_fisher = fisher_exact(table, alternative='greater')
    results_text.append(f"\nFisher's exact ({degree_type} vs high s_het, one-sided greater):")
    results_text.append(f"  Contingency table: {table}")
    results_text.append(f"  Odds ratio: {odds_ratio:.4f}, p-value: {p_fisher:.6f}")

    # pLI enrichment
    ct2 = pd.crosstab(df_constraint[high_deg_col], df_constraint['high_pLI'])
    table2 = [[ct2.iloc[0, 0], ct2.iloc[0, 1]], [ct2.iloc[1, 0], ct2.iloc[1, 1]]]
    odds_ratio2, p_fisher2 = fisher_exact(table2, alternative='greater')
    results_text.append(f"\nFisher's exact ({degree_type} vs high pLI>=0.9, one-sided greater):")
    results_text.append(f"  Contingency table: {table2}")
    results_text.append(f"  Odds ratio: {odds_ratio2:.4f}, p-value: {p_fisher2:.6f}")

# Mann-Whitney: top quartile vs bottom 3 quartiles
for degree_type in ['in_degree', 'out_degree']:
    high_deg_col = f'high_{degree_type}'
    top_q = df_constraint[df_constraint[high_deg_col]]['s_het']
    bottom_3q = df_constraint[~df_constraint[high_deg_col]]['s_het']
    stat, p_val = mannwhitneyu(top_q, bottom_3q, alternative='greater')
    results_text.append(f"\nMann-Whitney (top quartile {degree_type} vs bottom 3 quartiles, s_het):")
    results_text.append(f"  U={stat:.0f}, p={p_val:.6f}")
    results_text.append(f"  Top Q median s_het: {top_q.median():.6f}, Bottom 3Q median: {bottom_3q.median():.6f}")

print("Analysis 2 done.")

# ============================================================================
# ANALYSIS 3: Topological category analysis
# ============================================================================
results_text.append("\n" + "=" * 80)
results_text.append("ANALYSIS 3: Topological Category Analysis")
results_text.append("=" * 80)

def classify_topology(row):
    if row['high_in_degree'] and row['high_out_degree']:
        return 'Strong relay'
    elif row['high_in_degree'] and not row['high_out_degree']:
        return 'Strong terminal'
    elif not row['high_in_degree'] and row['high_out_degree']:
        return 'Strong upstream'
    else:
        return 'Other'

df_constraint['topology_category'] = df_constraint.apply(classify_topology, axis=1)

cat_order = ['Strong relay', 'Strong terminal', 'Strong upstream', 'Other']
cat_stats = df_constraint.groupby('topology_category').agg(
    n_genes=('s_het', 'count'),
    median_s_het=('s_het', 'median'),
    mean_s_het=('s_het', 'mean'),
    std_s_het=('s_het', 'std'),
    median_pLI=('pLI', 'median'),
    median_LOEUF=('LOEUF', 'median')
).reindex(cat_order)

results_text.append("\nCategory statistics:")
results_text.append(cat_stats.to_string())

# Kruskal-Wallis
groups = [df_constraint[df_constraint['topology_category'] == c]['s_het'].values for c in cat_order]
kw_stat, kw_p = kruskal(*groups)
results_text.append(f"\nKruskal-Wallis test (s_het across 4 categories): H={kw_stat:.4f}, p={kw_p:.6f}")

# Pairwise Mann-Whitney with FDR correction
pairs = []
pair_labels = []
for i in range(len(cat_order)):
    for j in range(i+1, len(cat_order)):
        u, p = mannwhitneyu(groups[i], groups[j], alternative='two-sided')
        pairs.append(p)
        pair_labels.append(f"{cat_order[i]} vs {cat_order[j]}")

reject, pvals_corrected, _, _ = multipletests(pairs, method='fdr_bh')
results_text.append("\nPairwise Mann-Whitney with FDR correction:")
for label, p_raw, p_fdr, rej in zip(pair_labels, pairs, pvals_corrected, reject):
    results_text.append(f"  {label}: raw p={p_raw:.6f}, FDR q={p_fdr:.6f}, significant={rej}")

# Specific one-sided tests
relay = df_constraint[df_constraint['topology_category'] == 'Strong relay']['s_het']
terminal = df_constraint[df_constraint['topology_category'] == 'Strong terminal']['s_het']
other = df_constraint[df_constraint['topology_category'] == 'Other']['s_het']

u_rt, p_rt = mannwhitneyu(relay, terminal, alternative='greater')
results_text.append(f"\nRelay vs Terminal (one-sided: relay higher?): U={u_rt:.0f}, p={p_rt:.6f}")
results_text.append(f"  Relay median s_het: {relay.median():.6f}, Terminal median: {terminal.median():.6f}")

u_ro, p_ro = mannwhitneyu(relay, other, alternative='greater')
results_text.append(f"Relay vs Other (one-sided: relay higher?): U={u_ro:.0f}, p={p_ro:.6f}")
results_text.append(f"  Relay median s_het: {relay.median():.6f}, Other median: {other.median():.6f}")

# Boxplot
fig, ax = plt.subplots(figsize=(10, 6))
plot_data = df_constraint[df_constraint['topology_category'].isin(cat_order)]
order = cat_order
colors = ['#e74c3c', '#3498db', '#2ecc71', '#95a5a6']
bp = sns.boxplot(data=plot_data, x='topology_category', y='s_het', order=order,
                 palette=colors, ax=ax, showfliers=False)
sns.stripplot(data=plot_data, x='topology_category', y='s_het', order=order,
              color='black', alpha=0.05, size=1, ax=ax)
ax.set_xlabel('Topological Category')
ax.set_ylabel('s_het')
ax.set_title('Selective Constraint (s_het) by Topological Category - Stim8hr')
for i, cat in enumerate(order):
    n = cat_stats.loc[cat, 'n_genes']
    ax.text(i, ax.get_ylim()[1] * 0.95, f'n={n}', ha='center', fontsize=9)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'analysis3_topology_boxplot.png'), dpi=150, bbox_inches='tight')
plt.close()
print("Analysis 3 done.")

# ============================================================================
# ANALYSIS 4: Matched-pair relay vs terminal
# ============================================================================
results_text.append("\n" + "=" * 80)
results_text.append("ANALYSIS 4: Matched-Pair Relay vs Terminal")
results_text.append("=" * 80)

relay_df = df_constraint[df_constraint['topology_category'] == 'Strong relay'][['gene_name', 'in_degree', 's_het']].copy()
terminal_df = df_constraint[df_constraint['topology_category'] == 'Strong terminal'][['gene_name', 'in_degree', 's_het']].copy()

relay_df = relay_df.sort_values('in_degree').reset_index(drop=True)
terminal_df = terminal_df.sort_values('in_degree').reset_index(drop=True)

# Nearest-neighbor matching without replacement
matched_relay = []
matched_terminal = []
available_terminal = terminal_df.copy()

for _, relay_row in relay_df.iterrows():
    if len(available_terminal) == 0:
        break
    diffs = (available_terminal['in_degree'] - relay_row['in_degree']).abs()
    best_idx = diffs.idxmin()
    matched_relay.append(relay_row)
    matched_terminal.append(available_terminal.loc[best_idx])
    available_terminal = available_terminal.drop(best_idx)

matched_relay_df = pd.DataFrame(matched_relay)
matched_terminal_df = pd.DataFrame(matched_terminal)

results_text.append(f"\nMatched pairs: {len(matched_relay_df)}")
results_text.append(f"Relay genes available: {len(relay_df)}, Terminal genes available: {len(terminal_df)}")
results_text.append(f"\nMatched Relay: mean s_het={matched_relay_df['s_het'].mean():.6f}, median={matched_relay_df['s_het'].median():.6f}")
results_text.append(f"Matched Terminal: mean s_het={matched_terminal_df['s_het'].mean():.6f}, median={matched_terminal_df['s_het'].median():.6f}")
results_text.append(f"Mean in_degree Relay: {matched_relay_df['in_degree'].mean():.2f}, Terminal: {matched_terminal_df['in_degree'].mean():.2f}")

# Wilcoxon signed-rank test (one-sided: relay higher?)
diff = matched_relay_df['s_het'].values - matched_terminal_df['s_het'].values
# Remove zeros for Wilcoxon
nonzero_diff = diff[diff != 0]
if len(nonzero_diff) > 10:
    w_stat, p_wilcox = wilcoxon(nonzero_diff, alternative='greater')
    results_text.append(f"\nWilcoxon signed-rank (one-sided: relay > terminal): W={w_stat:.0f}, p={p_wilcox:.6f}")
else:
    results_text.append(f"\nInsufficient non-zero differences for Wilcoxon test (n={len(nonzero_diff)})")

# Also do paired t-test for reference
t_stat, p_ttest = stats.ttest_rel(matched_relay_df['s_het'].values, matched_terminal_df['s_het'].values)
p_ttest_onesided = p_ttest / 2 if t_stat > 0 else 1 - p_ttest / 2
results_text.append(f"Paired t-test (one-sided): t={t_stat:.4f}, p={p_ttest_onesided:.6f}")

print("Analysis 4 done.")

# ============================================================================
# ANALYSIS 5: PID vs GWAS gene group comparison
# ============================================================================
results_text.append("\n" + "=" * 80)
results_text.append("ANALYSIS 5: PID vs GWAS Gene Group Comparison")
results_text.append("=" * 80)

# PID gene list (curated)
pid_genes_raw = [
    'FOXP3', 'CTLA4', 'STAT5B', 'IL2RA', 'LRBA', 'DOCK8', 'CARD11', 'MAGT1',
    'STIM1', 'ORAI1', 'ITK', 'CD3D', 'CD3E', 'CD247', 'ZAP70', 'LCK', 'LAT',
    'WAS', 'RAC2', 'RHOH', 'CORO1A', 'STK4', 'DOCK2', 'IKBKB', 'NFKB2',
    'RELA', 'RELB', 'NFKB1', 'TRAF3IP2', 'BCL10', 'MALT1', 'CARD11',
    'PIK3CD', 'PIK3R1', 'PTEN', 'ADA', 'PNP', 'DCLRE1C', 'LIG4', 'NHEJ1',
    'BIRC4', 'CASP8', 'CASP10', 'FAS', 'FASLG', 'UNC13D', 'STX11', 'STXBP2',
    'LYST', 'AP3B1', 'RAB27A', 'PRF1', 'GZMB', 'SH2D1A', 'TNFRSF13B',
    'TNFRSF13C', 'AICDA', 'UNG', 'MSH6', 'MSH2', 'PMS2'
]
pid_genes = set(pid_genes_raw)

# Load GWAS genes (L2G predictions)
try:
    l2g_df = pd.read_csv(L2G_PATH)
    gwas_genes = set(l2g_df.iloc[:, 0].values)  # First column should be gene symbols
    results_text.append(f"Loaded {len(gwas_genes)} GWAS genes from L2G predictions")
except:
    results_text.append("Could not load L2G genes, using empty set")
    gwas_genes = set()

all_genes_in_data = set(df_constraint['gene_name'].values)

pid_in_data = pid_genes.intersection(all_genes_in_data)
gwas_in_data = gwas_genes.intersection(all_genes_in_data)
pid_gwas_overlap = pid_in_data.intersection(gwas_in_data)

results_text.append(f"\nPID genes provided: {len(pid_genes)}")
results_text.append(f"PID genes in dataset: {len(pid_in_data)}")
results_text.append(f"PID genes found: {sorted(pid_in_data)}")
results_text.append(f"PID genes NOT found: {sorted(pid_genes - all_genes_in_data)}")
results_text.append(f"\nGWAS genes in dataset: {len(gwas_in_data)}")
results_text.append(f"PID ∩ GWAS overlap: {len(pid_gwas_overlap)}")
results_text.append(f"Overlap genes: {sorted(pid_gwas_overlap)}")

# Partition genes
pid_only = pid_in_data - gwas_in_data
gwas_only = gwas_in_data - pid_in_data
overlap = pid_gwas_overlap
background_genes = all_genes_in_data - pid_in_data - gwas_in_data

def assign_group(gene):
    if gene in overlap:
        return 'PID∩GWAS'
    elif gene in pid_only:
        return 'PID-only'
    elif gene in gwas_only:
        return 'GWAS-only'
    else:
        return 'Background'

df_constraint['gene_group'] = df_constraint['gene_name'].apply(assign_group)

group_order = ['PID-only', 'PID∩GWAS', 'GWAS-only', 'Background']
group_stats = df_constraint.groupby('gene_group').agg(
    n_genes=('s_het', 'count'),
    median_s_het=('s_het', 'median'),
    mean_s_het=('s_het', 'mean'),
    std_s_het=('s_het', 'std'),
    median_pLI=('pLI', 'median'),
    median_LOEUF=('LOEUF', 'median')
).reindex(group_order)

results_text.append(f"\nGene group statistics:")
results_text.append(group_stats.to_string())

# Kruskal-Wallis
group_data = [df_constraint[df_constraint['gene_group'] == g]['s_het'].dropna().values for g in group_order]
# Only test groups with > 0 members
valid_groups = [(g, d) for g, d in zip(group_order, group_data) if len(d) > 0]
valid_labels = [g for g, d in valid_groups]
valid_data = [d for g, d in valid_groups]

if len(valid_data) >= 2:
    kw_stat, kw_p = kruskal(*valid_data)
    results_text.append(f"\nKruskal-Wallis (s_het across groups): H={kw_stat:.4f}, p={kw_p:.6f}")

    # Pairwise Mann-Whitney with FDR
    pair_p = []
    pair_labels2 = []
    for i in range(len(valid_labels)):
        for j in range(i+1, len(valid_labels)):
            if len(valid_data[i]) > 0 and len(valid_data[j]) > 0:
                u, p = mannwhitneyu(valid_data[i], valid_data[j], alternative='two-sided')
                pair_p.append(p)
                pair_labels2.append(f"{valid_labels[i]} vs {valid_labels[j]}")

    if len(pair_p) > 0:
        reject2, pvals_corr2, _, _ = multipletests(pair_p, method='fdr_bh')
        results_text.append("\nPairwise Mann-Whitney with FDR correction:")
        for label, p_raw, p_fdr, rej in zip(pair_labels2, pair_p, pvals_corr2, reject2):
            results_text.append(f"  {label}: raw p={p_raw:.6f}, FDR q={p_fdr:.6f}, significant={rej}")

# Trend test: PID > overlap > GWAS > background
# Assign ordinal scores (4=PID, 3=overlap, 2=GWAS, 1=background)
group_rank_map = {'PID-only': 4, 'PID∩GWAS': 3, 'GWAS-only': 2, 'Background': 1}
df_constraint['group_rank'] = df_constraint['gene_group'].map(group_rank_map)
rho_trend, p_trend = spearmanr(df_constraint['group_rank'], df_constraint['s_het'])
results_text.append(f"\nSpearman trend test (PID > overlap > GWAS > background):")
results_text.append(f"  rho={rho_trend:.4f}, p={p_trend:.6f}")

# Boxplot for gene groups
fig, ax = plt.subplots(figsize=(10, 6))
plot_groups = df_constraint[df_constraint['gene_group'].isin(group_order)]
colors_grp = ['#e74c3c', '#f39c12', '#3498db', '#95a5a6']
sns.boxplot(data=plot_groups, x='gene_group', y='s_het', order=group_order,
            palette=colors_grp, ax=ax, showfliers=False)
sns.stripplot(data=plot_groups, x='gene_group', y='s_het', order=group_order,
              color='black', alpha=0.1, size=1, ax=ax)
ax.set_xlabel('Gene Group')
ax.set_ylabel('s_het')
ax.set_title('Selective Constraint by PID/GWAS Classification - Stim8hr')
for i, cat in enumerate(group_order):
    if cat in group_stats.index:
        n = group_stats.loc[cat, 'n_genes']
        ax.text(i, ax.get_ylim()[1] * 0.95, f'n={n}', ha='center', fontsize=9)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'analysis5_pid_gwas_boxplot.png'), dpi=150, bbox_inches='tight')
plt.close()
print("Analysis 5 done.")

# ============================================================================
# ANALYSIS 6: Logistic regression controlling for expression
# ============================================================================
results_text.append("\n" + "=" * 80)
results_text.append("ANALYSIS 6: Logistic Regression Controlling for Expression")
results_text.append("=" * 80)

# Prepare variables
df_reg = df_constraint.dropna(subset=['s_het', 'in_degree', 'out_degree', 'target_baseMean']).copy()
df_reg['high_s_het_binary'] = (df_reg['s_het'] >= s_het_75).astype(int)
df_reg['log_in_degree'] = np.log1p(df_reg['in_degree'])
df_reg['log_out_degree'] = np.log1p(df_reg['out_degree'])
df_reg['is_relay'] = (df_reg['topology_category'] == 'Strong relay').astype(int)
df_reg['log_baseMean'] = np.log1p(df_reg['target_baseMean'])

results_text.append(f"\nRegression sample size: {len(df_reg)}")

model_specs = {
    'Model 1': ['log_in_degree'],
    'Model 2': ['log_in_degree', 'log_out_degree'],
    'Model 3': ['log_in_degree', 'is_relay'],
    'Model 4': ['log_in_degree', 'is_relay', 'log_baseMean'],
}

model_results = {}
for name, predictors in model_specs.items():
    X = sm.add_constant(df_reg[predictors])
    y = df_reg['high_s_het_binary']
    try:
        model = sm.Logit(y, X).fit(disp=0)
        model_results[name] = model

        results_text.append(f"\n--- {name}: {' + '.join(predictors)} ---")
        results_text.append(f"AIC: {model.aic:.2f}")
        results_text.append(f"Log-likelihood: {model.llf:.2f}")
        results_text.append(f"Pseudo R²: {model.prsquared:.4f}")

        coef_df = pd.DataFrame({
            'Coefficient': model.params,
            'Std Error': model.bse,
            'Odds Ratio': np.exp(model.params),
            'z-value': model.tvalues,
            'p-value': model.pvalues,
            'OR 95% CI Lower': np.exp(model.conf_int()[0]),
            'OR 95% CI Upper': np.exp(model.conf_int()[1])
        })
        results_text.append(coef_df.to_string())
    except Exception as e:
        results_text.append(f"\n--- {name}: FAILED - {str(e)} ---")

# Likelihood ratio tests for nested models
results_text.append("\n--- Likelihood Ratio Tests (Nested Models) ---")
comparisons = [
    ('Model 1', 'Model 2', 'Adding out_degree'),
    ('Model 1', 'Model 3', 'Adding relay status'),
    ('Model 3', 'Model 4', 'Adding expression level'),
]

for m1_name, m2_name, description in comparisons:
    if m1_name in model_results and m2_name in model_results:
        m1 = model_results[m1_name]
        m2 = model_results[m2_name]
        lr_stat = 2 * (m2.llf - m1.llf)
        df_diff = m2.df_model - m1.df_model
        if df_diff > 0:
            p_lr = stats.chi2.sf(lr_stat, df_diff)
            results_text.append(f"\n{m1_name} vs {m2_name} ({description}):")
            results_text.append(f"  LR statistic: {lr_stat:.4f}, df={df_diff}, p={p_lr:.6f}")
            results_text.append(f"  AIC improvement: {m1.aic - m2.aic:.2f}")

# Key question summary
results_text.append("\n--- Key Question: Does relay status predict s_het beyond in-degree and expression? ---")
if 'Model 4' in model_results:
    m4 = model_results['Model 4']
    if 'is_relay' in m4.params.index:
        relay_coef = m4.params['is_relay']
        relay_p = m4.pvalues['is_relay']
        relay_or = np.exp(relay_coef)
        results_text.append(f"In Model 4 (controlling for in_degree + baseMean):")
        results_text.append(f"  Relay coefficient: {relay_coef:.4f}, OR: {relay_or:.4f}, p={relay_p:.6f}")
        results_text.append(f"  {'YES' if relay_p < 0.05 else 'NO'} - relay status {'significantly' if relay_p < 0.05 else 'does not significantly'} predict high s_het after controlling for in-degree and expression")

print("Analysis 6 done.")

# ============================================================================
# Save all results
# ============================================================================
# Save text results
results_file = os.path.join(OUT_DIR, 'constraint_network_analyses_stim8hr.txt')
with open(results_file, 'w') as f:
    f.write('\n'.join(results_text))
print(f"\nAll results saved to: {results_file}")

# Save summary tables
cat_stats.to_csv(os.path.join(OUT_DIR, 'analysis3_topology_category_stats.csv'))
group_stats.to_csv(os.path.join(OUT_DIR, 'analysis5_gene_group_stats.csv'))

# Save the enriched dataframe
df_constraint.to_csv(os.path.join(OUT_DIR, 'stim8hr_constraint_annotated.csv'), index=False)

# Print full results
print('\n'.join(results_text))
print("\n\nDONE - All 6 analyses completed successfully.")
