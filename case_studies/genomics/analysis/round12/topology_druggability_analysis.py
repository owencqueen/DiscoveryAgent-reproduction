import os
#!/usr/bin/env python3
"""
Topology-Druggability Analysis
==============================
Merges gene topology classification (relay/terminal/upstream/other) with
OpenTargets druggability data and tests whether network topology categories
differ in druggability/tractability.

Inputs:
  - results/gene_topology_classification_stim8hr.csv
  - results/opentargets_druggability.csv
  - data/GWCD4i.DE_stats.h5ad (for expression confound check)

Outputs:
  - reports/analysis_outputs/topology_druggability_results.txt
  - data/processed/topology_druggability_summary.csv
  - reports/figures/druggability_*.png
"""

import sys
import os
import warnings
warnings.filterwarnings('ignore')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from scipy import stats
from itertools import combinations

# ── Paths ──────────────────────────────────────────────────────────────────
BASE = (os.environ["REPRO_DATA_ROOT"] + '')
TOPO_PATH = os.path.join(BASE, 'results/gene_topology_classification_stim8hr.csv')
DRUG_PATH = os.path.join(BASE, 'results/opentargets_druggability.csv')
H5AD_PATH = os.path.join(BASE, 'data/GWCD4i.DE_stats.h5ad')

OUT_TXT = os.path.join(BASE, 'reports/analysis_outputs/topology_druggability_results.txt')
OUT_CSV = os.path.join(BASE, 'data/processed/topology_druggability_summary.csv')
FIG_DIR = os.path.join(BASE, 'reports/figures')

# ── Colors (Wong palette) ─────────────────────────────────────────────────
CAT_COLORS = {
    'relay':    '#E69F00',
    'terminal': '#56B4E9',
    'upstream': '#009E73',
    'other':    '#999999',
}
CAT_ORDER = ['relay', 'terminal', 'upstream', 'other']

# ── Tee-style output ──────────────────────────────────────────────────────
class TeeOutput:
    def __init__(self, filepath):
        self.terminal = sys.stdout
        self.file = open(filepath, 'w')
    def write(self, msg):
        self.terminal.write(msg)
        self.file.write(msg)
    def flush(self):
        self.terminal.flush()
        self.file.flush()
    def close(self):
        self.file.close()

tee = TeeOutput(OUT_TXT)
sys.stdout = tee

# ── Helper functions ──────────────────────────────────────────────────────

def binomial_ci(k, n, z=1.96):
    """Wilson score interval for binomial proportion."""
    if n == 0:
        return 0.0, 0.0, 0.0
    p = k / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    margin = z * np.sqrt((p * (1 - p) + z**2 / (4 * n)) / n) / denom
    return p, max(0, centre - margin), min(1, centre + margin)


def fisher_or_ci(table, haldane=True):
    """
    Fisher's exact test with OR and 95% CI.
    table: 2x2 array [[a, b], [c, d]]
    Returns: OR, p_value, ci_low, ci_high
    """
    a, b, c, d = table[0, 0], table[0, 1], table[1, 0], table[1, 1]
    if haldane and (a == 0 or b == 0 or c == 0 or d == 0):
        a, b, c, d = a + 0.5, b + 0.5, c + 0.5, d + 0.5
    or_val, p_val = stats.fisher_exact(table, alternative='two-sided')
    log_or = np.log(a * d / (b * c)) if (b * c) > 0 else np.nan
    se = np.sqrt(1/a + 1/b + 1/c + 1/d)
    ci_low = np.exp(log_or - 1.96 * se)
    ci_high = np.exp(log_or + 1.96 * se)
    return or_val, p_val, ci_low, ci_high


def significance_stars(p):
    if p < 0.001:
        return '***'
    elif p < 0.01:
        return '**'
    elif p < 0.05:
        return '*'
    else:
        return 'ns'


# ══════════════════════════════════════════════════════════════════════════
# DATA LOADING AND MERGE
# ══════════════════════════════════════════════════════════════════════════
print("=" * 80)
print("TOPOLOGY-DRUGGABILITY ANALYSIS")
print("=" * 80)
print()

topo = pd.read_csv(TOPO_PATH)
drug = pd.read_csv(DRUG_PATH)

print(f"Topology file: {topo.shape[0]} genes, categories: {topo['category'].value_counts().to_dict()}")
print(f"Druggability file: {drug.shape[0]} genes")
print()

# Exclude measured_only
topo_filt = topo[topo['category'] != 'measured_only'].copy()
print(f"After excluding measured_only: {topo_filt.shape[0]} genes")

# Merge — left join so topology genes not in druggability get NaN
merged = topo_filt.merge(drug, left_on='gene_name', right_on='approvedSymbol', how='left')
print(f"Merged: {merged.shape[0]} rows")
n_matched = merged['approvedSymbol'].notna().sum()
n_unmatched = merged['approvedSymbol'].isna().sum()
print(f"  Matched to druggability: {n_matched}")
print(f"  Not in druggability DB:  {n_unmatched} (treated as NOT tractable)")
print()

# Fill unmatched genes: not tractable
tractable_cols = ['Small_Molecule_tractable', 'Antibody_tractable',
                  'PROTAC_tractable', 'Other_Clinical_Modalities_tractable']
for col in tractable_cols:
    merged[col] = merged[col].fillna(False).astype(bool)
merged['n_modalities_tractable'] = merged['n_modalities_tractable'].fillna(0).astype(int)
merged['tractable_any'] = (merged['n_modalities_tractable'] > 0)

# Per-category counts
print("Category gene counts after merge:")
for cat in CAT_ORDER:
    sub = merged[merged['category'] == cat]
    print(f"  {cat:12s}: {len(sub):5d} genes, {sub['tractable_any'].sum():4d} tractable, "
          f"{(sub['approvedSymbol'].notna()).sum():4d} in druggability DB")
print()

# ══════════════════════════════════════════════════════════════════════════
# ANALYSIS A: OVERALL TRACTABILITY
# ══════════════════════════════════════════════════════════════════════════
print("=" * 80)
print("A. OVERALL TRACTABILITY BY TOPOLOGY CATEGORY")
print("=" * 80)
print()

# A1. Fraction tractable per category
print("A1. Fraction of genes tractable in >= 1 modality:")
frac_data = {}
for cat in CAT_ORDER:
    sub = merged[merged['category'] == cat]
    k = sub['tractable_any'].sum()
    n = len(sub)
    p, lo, hi = binomial_ci(k, n)
    frac_data[cat] = {'n': n, 'k': k, 'frac': p, 'ci_lo': lo, 'ci_hi': hi}
    print(f"  {cat:12s}: {k:4d}/{n:4d} = {p:.3f}  (95% CI: [{lo:.3f}, {hi:.3f}])")
print()

# A2. Mean n_modalities
print("A2. Mean number of tractable modalities:")
for cat in CAT_ORDER:
    sub = merged[merged['category'] == cat]
    m = sub['n_modalities_tractable'].mean()
    s = sub['n_modalities_tractable'].std()
    print(f"  {cat:12s}: mean = {m:.3f} +/- {s:.3f}")
print()

# A3. Chi-squared test
print("A3. Chi-squared test (tractable_any x category):")
ct = pd.crosstab(merged['category'], merged['tractable_any'])
ct = ct.reindex(CAT_ORDER)
chi2, p_chi, dof, expected = stats.chi2_contingency(ct)
print(f"  Chi2 = {chi2:.2f}, df = {dof}, p = {p_chi:.2e}")
print()

# A4-A5. Pairwise Fisher's exact: relay vs each other
print("A4-A5. Pairwise Fisher's exact tests (relay vs each other category):")
print(f"  {'Comparison':<25s} {'OR':>8s} {'p-value':>12s} {'95% CI':>20s} {'Sig':>5s}")
print("  " + "-" * 75)
fisher_results_overall = {}
for other_cat in ['terminal', 'upstream', 'other']:
    relay_sub = merged[merged['category'] == 'relay']
    other_sub = merged[merged['category'] == other_cat]
    table = np.array([
        [relay_sub['tractable_any'].sum(), (~relay_sub['tractable_any']).sum()],
        [other_sub['tractable_any'].sum(), (~other_sub['tractable_any']).sum()]
    ])
    or_val, p_val, ci_lo, ci_hi = fisher_or_ci(table)
    sig = significance_stars(p_val)
    fisher_results_overall[f'relay_vs_{other_cat}'] = {
        'OR': or_val, 'p': p_val, 'ci_lo': ci_lo, 'ci_hi': ci_hi
    }
    print(f"  relay vs {other_cat:<12s}  {or_val:8.3f}  {p_val:12.2e}  [{ci_lo:.3f}, {ci_hi:.3f}]  {sig:>5s}")
print()

# ══════════════════════════════════════════════════════════════════════════
# ANALYSIS B: PER-MODALITY TRACTABILITY
# ══════════════════════════════════════════════════════════════════════════
print("=" * 80)
print("B. PER-MODALITY TRACTABILITY RATES")
print("=" * 80)
print()

modality_labels = {
    'Small_Molecule_tractable': 'Small Molecule',
    'Antibody_tractable': 'Antibody',
    'PROTAC_tractable': 'PROTAC',
    'Other_Clinical_Modalities_tractable': 'Other Clinical'
}

modality_frac = {}  # {modality: {cat: frac}}
modality_fisher = {}  # {modality: {comparison: {OR, p, ci}}}

for col, label in modality_labels.items():
    print(f"--- {label} ---")
    modality_frac[label] = {}
    modality_fisher[label] = {}

    # Fraction per category
    for cat in CAT_ORDER:
        sub = merged[merged['category'] == cat]
        k = sub[col].sum()
        n = len(sub)
        p, lo, hi = binomial_ci(k, n)
        modality_frac[label][cat] = {'frac': p, 'ci_lo': lo, 'ci_hi': hi, 'k': k, 'n': n}
        print(f"  {cat:12s}: {k:4d}/{n:4d} = {p:.3f}  (95% CI: [{lo:.3f}, {hi:.3f}])")

    # Fisher's exact: relay vs others
    print(f"  {'Comparison':<25s} {'OR':>8s} {'p-value':>12s} {'95% CI':>20s}")
    for other_cat in ['terminal', 'upstream', 'other']:
        relay_sub = merged[merged['category'] == 'relay']
        other_sub = merged[merged['category'] == other_cat]
        table = np.array([
            [relay_sub[col].sum(), (~relay_sub[col]).sum()],
            [other_sub[col].sum(), (~other_sub[col]).sum()]
        ])
        or_val, p_val, ci_lo, ci_hi = fisher_or_ci(table)
        modality_fisher[label][f'relay_vs_{other_cat}'] = {
            'OR': or_val, 'p': p_val, 'ci_lo': ci_lo, 'ci_hi': ci_hi
        }
        sig = significance_stars(p_val)
        print(f"  relay vs {other_cat:<12s}  {or_val:8.3f}  {p_val:12.2e}  [{ci_lo:.3f}, {ci_hi:.3f}] {sig}")
    print()

# ══════════════════════════════════════════════════════════════════════════
# ANALYSIS C: CLINICAL PRECEDENCE
# ══════════════════════════════════════════════════════════════════════════
print("=" * 80)
print("C. CLINICAL PRECEDENCE (best_bucket_rank <= 3)")
print("=" * 80)
print()

merged['clinical_precedence'] = merged['best_bucket_rank'].apply(
    lambda x: True if pd.notna(x) and x <= 3 else False
)

print("C1. Clinical precedence rate per category:")
clin_data = {}
for cat in CAT_ORDER:
    sub = merged[merged['category'] == cat]
    k = sub['clinical_precedence'].sum()
    n = len(sub)
    p, lo, hi = binomial_ci(k, n)
    clin_data[cat] = {'frac': p, 'ci_lo': lo, 'ci_hi': hi, 'k': k, 'n': n}
    print(f"  {cat:12s}: {k:4d}/{n:4d} = {p:.3f}  (95% CI: [{lo:.3f}, {hi:.3f}])")
print()

print("C2. Fisher's exact tests (relay vs each other):")
for other_cat in ['terminal', 'upstream', 'other']:
    relay_sub = merged[merged['category'] == 'relay']
    other_sub = merged[merged['category'] == other_cat]
    table = np.array([
        [relay_sub['clinical_precedence'].sum(), (~relay_sub['clinical_precedence']).sum()],
        [other_sub['clinical_precedence'].sum(), (~other_sub['clinical_precedence']).sum()]
    ])
    or_val, p_val, ci_lo, ci_hi = fisher_or_ci(table)
    sig = significance_stars(p_val)
    print(f"  relay vs {other_cat:<12s}: OR={or_val:.3f}, p={p_val:.2e}, "
          f"CI=[{ci_lo:.3f}, {ci_hi:.3f}] {sig}")
print()

print("C3. Cross-tabulation: best_bucket_category x topology category")
merged_with_bucket = merged[merged['best_bucket_category'].notna()].copy()
ct_bucket = pd.crosstab(merged_with_bucket['best_bucket_category'],
                         merged_with_bucket['category'])
ct_bucket = ct_bucket[CAT_ORDER]
print(ct_bucket.to_string())
print()

# Normalized
ct_bucket_norm = pd.crosstab(merged_with_bucket['best_bucket_category'],
                              merged_with_bucket['category'], normalize='columns')
ct_bucket_norm = ct_bucket_norm[CAT_ORDER]
print("Normalized by column (fraction within each category):")
print(ct_bucket_norm.round(3).to_string())
print()

# ══════════════════════════════════════════════════════════════════════════
# ANALYSIS D: BEST BUCKET RANK DISTRIBUTION
# ══════════════════════════════════════════════════════════════════════════
print("=" * 80)
print("D. BEST BUCKET RANK DISTRIBUTION")
print("=" * 80)
print()

# Only genes with a bucket rank
has_rank = merged[merged['best_bucket_rank'].notna()].copy()
print(f"Genes with best_bucket_rank: {len(has_rank)}")

# D1. KW test
groups_rank = [has_rank[has_rank['category'] == cat]['best_bucket_rank'].values
               for cat in CAT_ORDER]
kw_stat, kw_p = stats.kruskal(*groups_rank)
print(f"D1. Kruskal-Wallis test: H = {kw_stat:.2f}, p = {kw_p:.2e}")
print()

# D2. Median and mean per category
print("D2. Best bucket rank summary (lower = more druggable):")
for cat in CAT_ORDER:
    sub = has_rank[has_rank['category'] == cat]['best_bucket_rank']
    print(f"  {cat:12s}: n={len(sub):4d}, median={sub.median():.1f}, mean={sub.mean():.2f}, std={sub.std():.2f}")
print()

# D3. Pairwise Mann-Whitney U
print("D3. Pairwise Mann-Whitney U tests (relay vs each other):")
relay_rank = has_rank[has_rank['category'] == 'relay']['best_bucket_rank']
for other_cat in ['terminal', 'upstream', 'other']:
    other_rank = has_rank[has_rank['category'] == other_cat]['best_bucket_rank']
    u_stat, u_p = stats.mannwhitneyu(relay_rank, other_rank, alternative='two-sided')
    sig = significance_stars(u_p)
    print(f"  relay vs {other_cat:<12s}: U={u_stat:.0f}, p={u_p:.2e} {sig}")
print()

# ══════════════════════════════════════════════════════════════════════════
# ANALYSIS E: APPROVED DRUG TARGETS
# ══════════════════════════════════════════════════════════════════════════
print("=" * 80)
print("E. APPROVED DRUG TARGETS (best_bucket_rank == 1)")
print("=" * 80)
print()

merged['approved_drug'] = merged['best_bucket_rank'].apply(
    lambda x: True if pd.notna(x) and x == 1 else False
)

print("E1. Approved drug target rate per category:")
approved_data = {}
for cat in CAT_ORDER:
    sub = merged[merged['category'] == cat]
    k = sub['approved_drug'].sum()
    n = len(sub)
    p, lo, hi = binomial_ci(k, n)
    approved_data[cat] = {'frac': p, 'ci_lo': lo, 'ci_hi': hi, 'k': k, 'n': n}
    print(f"  {cat:12s}: {k:3d}/{n:4d} = {p:.3f}  (95% CI: [{lo:.3f}, {hi:.3f}])")
print()

print("E2. Fisher's exact tests (relay vs each other):")
for other_cat in ['terminal', 'upstream', 'other']:
    relay_sub = merged[merged['category'] == 'relay']
    other_sub = merged[merged['category'] == other_cat]
    table = np.array([
        [relay_sub['approved_drug'].sum(), (~relay_sub['approved_drug']).sum()],
        [other_sub['approved_drug'].sum(), (~other_sub['approved_drug']).sum()]
    ])
    or_val, p_val, ci_lo, ci_hi = fisher_or_ci(table)
    sig = significance_stars(p_val)
    print(f"  relay vs {other_cat:<12s}: OR={or_val:.3f}, p={p_val:.2e}, "
          f"CI=[{ci_lo:.3f}, {ci_hi:.3f}] {sig}")
print()

# E3. List relay genes that are approved drug targets
relay_approved = merged[(merged['category'] == 'relay') & (merged['approved_drug'] == True)]
print(f"E3. Relay genes that are approved drug targets ({len(relay_approved)}):")
for _, row in relay_approved.sort_values('in_degree', ascending=False).iterrows():
    print(f"  {row['gene_name']:12s}  in_degree={row['in_degree']:.0f}  "
          f"out_degree={row['out_degree']:.0f}  "
          f"best_modality={row['best_modality']}")
print()

# ══════════════════════════════════════════════════════════════════════════
# ANALYSIS F: EXPRESSION CONFOUND CHECK
# ══════════════════════════════════════════════════════════════════════════
print("=" * 80)
print("F. EXPRESSION CONFOUND CHECK")
print("=" * 80)
print()

import anndata as ad
adata = ad.read_h5ad(H5AD_PATH)
# Quality filter, Stim8hr
mask = ((adata.obs['ontarget_significant'] == True) &
        (adata.obs['offtarget_flag'] == False) &
        (adata.obs['culture_condition'] == 'Stim8hr'))
stim8 = adata[mask].copy()
print(f"Stim8hr QC-filtered perturbations: {stim8.shape[0]}")

# Median target_baseMean per gene
expr_df = stim8.obs[['target_contrast_gene_name', 'target_baseMean']].copy()
expr_df = expr_df.dropna(subset=['target_baseMean'])
gene_expr = expr_df.groupby('target_contrast_gene_name')['target_baseMean'].median().reset_index()
gene_expr.columns = ['gene_name', 'median_baseMean']
print(f"Genes with expression data: {gene_expr.shape[0]}")

# Merge with topology+druggability
merged_expr = merged.merge(gene_expr, on='gene_name', how='left')
has_expr = merged_expr[merged_expr['median_baseMean'].notna()].copy()
print(f"Genes with expression + topology: {has_expr.shape[0]}")

# Expression tertiles
has_expr['expr_tertile'] = pd.qcut(has_expr['median_baseMean'], 3,
                                    labels=['Low', 'Mid', 'High'])
print(f"Expression tertile counts: {has_expr['expr_tertile'].value_counts().to_dict()}")
print()

# F1. Within each tertile, tractability fraction per category
print("F1. Tractability fraction by expression tertile x topology category:")
expr_confound = {}
for tertile in ['Low', 'Mid', 'High']:
    sub_tert = has_expr[has_expr['expr_tertile'] == tertile]
    print(f"  --- {tertile} expression tertile ---")
    expr_confound[tertile] = {}
    for cat in CAT_ORDER:
        sub_cat = sub_tert[sub_tert['category'] == cat]
        k = sub_cat['tractable_any'].sum()
        n = len(sub_cat)
        p, lo, hi = binomial_ci(k, n)
        expr_confound[tertile][cat] = {'frac': p, 'ci_lo': lo, 'ci_hi': hi, 'k': k, 'n': n}
        print(f"    {cat:12s}: {k:3d}/{n:4d} = {p:.3f}  (95% CI: [{lo:.3f}, {hi:.3f}])")
    print()

# F2. Fisher's exact: relay vs all others combined within each tertile
print("F2. Fisher's exact (relay vs all non-relay) within each tertile:")
expr_fisher_results = {}
for tertile in ['Low', 'Mid', 'High']:
    sub_tert = has_expr[has_expr['expr_tertile'] == tertile]
    relay_sub = sub_tert[sub_tert['category'] == 'relay']
    nonrelay_sub = sub_tert[sub_tert['category'] != 'relay']
    table = np.array([
        [relay_sub['tractable_any'].sum(), (~relay_sub['tractable_any']).sum()],
        [nonrelay_sub['tractable_any'].sum(), (~nonrelay_sub['tractable_any']).sum()]
    ])
    or_val, p_val, ci_lo, ci_hi = fisher_or_ci(table)
    sig = significance_stars(p_val)
    expr_fisher_results[tertile] = {'OR': or_val, 'p': p_val, 'ci_lo': ci_lo, 'ci_hi': ci_hi}
    print(f"  {tertile:4s} tertile: relay n={len(relay_sub)}, non-relay n={len(nonrelay_sub)}, "
          f"OR={or_val:.3f}, p={p_val:.2e}, CI=[{ci_lo:.3f}, {ci_hi:.3f}] {sig}")
print()

# Also do relay vs each specific category within each tertile
print("F3. Fisher's exact (relay vs each category) within each tertile:")
for tertile in ['Low', 'Mid', 'High']:
    sub_tert = has_expr[has_expr['expr_tertile'] == tertile]
    print(f"  --- {tertile} expression tertile ---")
    for other_cat in ['terminal', 'upstream', 'other']:
        relay_sub = sub_tert[sub_tert['category'] == 'relay']
        other_sub = sub_tert[sub_tert['category'] == other_cat]
        if len(relay_sub) == 0 or len(other_sub) == 0:
            print(f"    relay vs {other_cat:<12s}: skipped (empty group)")
            continue
        table = np.array([
            [relay_sub['tractable_any'].sum(), (~relay_sub['tractable_any']).sum()],
            [other_sub['tractable_any'].sum(), (~other_sub['tractable_any']).sum()]
        ])
        or_val, p_val, ci_lo, ci_hi = fisher_or_ci(table)
        sig = significance_stars(p_val)
        print(f"    relay vs {other_cat:<12s}: OR={or_val:.3f}, p={p_val:.2e}, "
              f"CI=[{ci_lo:.3f}, {ci_hi:.3f}] {sig}")
    print()

# ══════════════════════════════════════════════════════════════════════════
# SUMMARY CSV
# ══════════════════════════════════════════════════════════════════════════
print("=" * 80)
print("SUMMARY TABLE")
print("=" * 80)
print()

summary_rows = []
for cat in CAT_ORDER:
    sub = merged[merged['category'] == cat]
    sub_rank = sub[sub['best_bucket_rank'].notna()]
    row = {
        'category': cat,
        'n_genes': len(sub),
        'n_druggability_matched': sub['approvedSymbol'].notna().sum(),
        'frac_tractable_any': sub['tractable_any'].mean(),
        'mean_n_modalities': sub['n_modalities_tractable'].mean(),
        'frac_small_molecule': sub['Small_Molecule_tractable'].mean(),
        'frac_antibody': sub['Antibody_tractable'].mean(),
        'frac_protac': sub['PROTAC_tractable'].mean(),
        'frac_other_clinical': sub['Other_Clinical_Modalities_tractable'].mean(),
        'frac_clinical_precedence': sub['clinical_precedence'].mean(),
        'frac_approved_drug': sub['approved_drug'].mean(),
        'median_best_bucket_rank': sub_rank['best_bucket_rank'].median() if len(sub_rank) > 0 else np.nan,
        'mean_best_bucket_rank': sub_rank['best_bucket_rank'].mean() if len(sub_rank) > 0 else np.nan,
    }
    summary_rows.append(row)

summary_df = pd.DataFrame(summary_rows)
summary_df.to_csv(OUT_CSV, index=False)
print(summary_df.to_string(index=False))
print(f"\nSaved to: {OUT_CSV}")
print()

# ══════════════════════════════════════════════════════════════════════════
# FIGURE 1: OVERALL TRACTABILITY
# ══════════════════════════════════════════════════════════════════════════

fig, ax = plt.subplots(figsize=(10, 5))

# Group: Overall + 4 modalities
group_labels = ['Overall', 'Small Molecule', 'Antibody', 'PROTAC', 'Other Clinical']
x = np.arange(len(group_labels))
width = 0.18

for i, cat in enumerate(CAT_ORDER):
    fracs = []
    ci_errs = []

    # Overall
    fd = frac_data[cat]
    fracs.append(fd['frac'])
    ci_errs.append([fd['frac'] - fd['ci_lo'], fd['ci_hi'] - fd['frac']])

    # Per-modality
    for label in ['Small Molecule', 'Antibody', 'PROTAC', 'Other Clinical']:
        mf = modality_frac[label][cat]
        fracs.append(mf['frac'])
        ci_errs.append([mf['frac'] - mf['ci_lo'], mf['ci_hi'] - mf['frac']])

    ci_errs = np.array(ci_errs).T
    bars = ax.bar(x + (i - 1.5) * width, fracs, width, label=cat,
                  color=CAT_COLORS[cat], yerr=ci_errs, capsize=2,
                  edgecolor='black', linewidth=0.5)

# Add significance stars (relay vs other) for overall
for j, grp in enumerate(group_labels):
    if grp == 'Overall':
        p_val = fisher_results_overall['relay_vs_other']['p']
    else:
        p_val = modality_fisher[grp]['relay_vs_other']['p']
    star = significance_stars(p_val)
    if star != 'ns':
        # Position above relay bar
        relay_frac_val = frac_data[cat]['frac'] if grp == 'Overall' else modality_frac[grp]['relay']['frac']
        max_y = max(
            frac_data['relay']['frac'] if grp == 'Overall' else modality_frac[grp]['relay']['frac'],
            frac_data['other']['frac'] if grp == 'Overall' else modality_frac[grp]['other']['frac']
        )
        ax.annotate(star, xy=(j, max_y + 0.04), ha='center', fontsize=9, color='red')

ax.set_xticks(x)
ax.set_xticklabels(group_labels, fontsize=10)
ax.set_ylabel('Fraction tractable', fontsize=12)
ax.set_title('Druggability by Network Topology Category', fontsize=13, fontweight='bold')
ax.legend(title='Topology', fontsize=9, title_fontsize=10, loc='upper right')
ax.set_ylim(0, ax.get_ylim()[1] * 1.15)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()
fig.savefig(os.path.join(FIG_DIR, 'druggability_overall_tractability.png'), dpi=150,
            bbox_inches='tight')
plt.close(fig)
print("Saved: druggability_overall_tractability.png")

# ══════════════════════════════════════════════════════════════════════════
# FIGURE 2: BUCKET DISTRIBUTION
# ══════════════════════════════════════════════════════════════════════════

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Panel A: Stacked bar of best_bucket_category
bucket_cats_ordered = ['Clinical_Precedence', 'Discovery_Precedence', 'Literature_Evidence',
                       'Predicted_Tractable_High', 'Predicted_Tractable_Med', 'Predicted_Tractable']
bucket_colors = plt.cm.RdYlGn(np.linspace(0.8, 0.2, len(bucket_cats_ordered)))

ax = axes[0]
# Fraction within each topology category
bottom = np.zeros(len(CAT_ORDER))
for j, bucket in enumerate(bucket_cats_ordered):
    vals = []
    for cat in CAT_ORDER:
        sub = merged[merged['category'] == cat]
        n_total = len(sub)
        n_bucket = ((sub['best_bucket_category'] == bucket).sum())
        vals.append(n_bucket / n_total if n_total > 0 else 0)
    ax.bar(CAT_ORDER, vals, bottom=bottom, label=bucket.replace('_', ' '),
           color=bucket_colors[j], edgecolor='white', linewidth=0.3)
    bottom += vals

# Add "Not tractable" bar
not_tract = []
for cat in CAT_ORDER:
    sub = merged[merged['category'] == cat]
    not_tract.append(1 - sub['tractable_any'].mean())
ax.bar(CAT_ORDER, not_tract, bottom=bottom, label='Not tractable',
       color='#DDDDDD', edgecolor='white', linewidth=0.3)
ax.set_ylabel('Fraction of genes', fontsize=11)
ax.set_title('Bucket category distribution', fontsize=12, fontweight='bold')
ax.legend(fontsize=7, loc='upper right', ncol=1)
ax.set_xticklabels([c.capitalize() for c in CAT_ORDER], fontsize=10)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Panel B: Box/violin of best_bucket_rank
ax = axes[1]
data_for_violin = []
positions = []
for i, cat in enumerate(CAT_ORDER):
    sub = has_rank[has_rank['category'] == cat]['best_bucket_rank']
    data_for_violin.append(sub.values)
    positions.append(i)

vp = ax.violinplot(data_for_violin, positions=positions, showmeans=True,
                   showmedians=True, showextrema=False)
for i, body in enumerate(vp['bodies']):
    body.set_facecolor(CAT_COLORS[CAT_ORDER[i]])
    body.set_alpha(0.7)
vp['cmeans'].set_color('black')
vp['cmedians'].set_color('red')

ax.set_xticks(positions)
ax.set_xticklabels([c.capitalize() for c in CAT_ORDER], fontsize=10)
ax.set_ylabel('Best bucket rank (lower = more druggable)', fontsize=11)
ax.set_title('Best bucket rank distribution', fontsize=12, fontweight='bold')
ax.invert_yaxis()
ax.set_yticks(range(1, 10))
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Add median annotation
for i, cat in enumerate(CAT_ORDER):
    sub = has_rank[has_rank['category'] == cat]['best_bucket_rank']
    ax.text(i, sub.median() + 0.3, f'med={sub.median():.1f}', ha='center', fontsize=8)

plt.tight_layout()
fig.savefig(os.path.join(FIG_DIR, 'druggability_bucket_distribution.png'), dpi=150,
            bbox_inches='tight')
plt.close(fig)
print("Saved: druggability_bucket_distribution.png")

# ══════════════════════════════════════════════════════════════════════════
# FIGURE 3: SUMMARY 2x2 FIGURE
# ══════════════════════════════════════════════════════════════════════════

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Panel A: Overall tractability fraction by category
ax = axes[0, 0]
fracs_overall = [frac_data[c]['frac'] for c in CAT_ORDER]
ci_lo_overall = [frac_data[c]['frac'] - frac_data[c]['ci_lo'] for c in CAT_ORDER]
ci_hi_overall = [frac_data[c]['ci_hi'] - frac_data[c]['frac'] for c in CAT_ORDER]
bars = ax.bar(CAT_ORDER, fracs_overall,
              yerr=[ci_lo_overall, ci_hi_overall], capsize=4,
              color=[CAT_COLORS[c] for c in CAT_ORDER],
              edgecolor='black', linewidth=0.5)
ax.set_ylabel('Fraction tractable (any modality)', fontsize=10)
ax.set_title('A. Overall tractability', fontsize=12, fontweight='bold')
ax.set_xticklabels([c.capitalize() for c in CAT_ORDER], fontsize=10)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
# Add counts inside bars
for i, cat in enumerate(CAT_ORDER):
    ax.text(i, fracs_overall[i] / 2, f"n={frac_data[cat]['n']}",
            ha='center', va='center', fontsize=9, color='white', fontweight='bold')
# Significance bracket: relay vs other
p_ro = fisher_results_overall['relay_vs_other']['p']
star_ro = significance_stars(p_ro)
if star_ro != 'ns':
    y_max = max(fracs_overall) + max(ci_hi_overall) + 0.03
    ax.plot([0, 0, 3, 3], [y_max, y_max + 0.01, y_max + 0.01, y_max],
            color='black', linewidth=1)
    ax.text(1.5, y_max + 0.015, f'{star_ro} p={p_ro:.1e}', ha='center', fontsize=9)

# Panel B: Per-modality tractability rates
ax = axes[0, 1]
mod_names = ['Small\nMolecule', 'Antibody', 'PROTAC', 'Other\nClinical']
mod_keys = ['Small Molecule', 'Antibody', 'PROTAC', 'Other Clinical']
x = np.arange(len(mod_names))
width = 0.2
for i, cat in enumerate(CAT_ORDER):
    fracs_mod = [modality_frac[mk][cat]['frac'] for mk in mod_keys]
    ax.bar(x + (i - 1.5) * width, fracs_mod, width,
           color=CAT_COLORS[cat], label=cat.capitalize(),
           edgecolor='black', linewidth=0.3)
ax.set_xticks(x)
ax.set_xticklabels(mod_names, fontsize=9)
ax.set_ylabel('Fraction tractable', fontsize=10)
ax.set_title('B. Per-modality tractability', fontsize=12, fontweight='bold')
ax.legend(fontsize=8, title='Topology', title_fontsize=9)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Panel C: Clinical precedence rate
ax = axes[1, 0]
clin_fracs = [clin_data[c]['frac'] for c in CAT_ORDER]
clin_lo = [clin_data[c]['frac'] - clin_data[c]['ci_lo'] for c in CAT_ORDER]
clin_hi = [clin_data[c]['ci_hi'] - clin_data[c]['frac'] for c in CAT_ORDER]
ax.bar(CAT_ORDER, clin_fracs, yerr=[clin_lo, clin_hi], capsize=4,
       color=[CAT_COLORS[c] for c in CAT_ORDER],
       edgecolor='black', linewidth=0.5)
ax.set_ylabel('Fraction with clinical precedence', fontsize=10)
ax.set_title('C. Clinical precedence (rank <= 3)', fontsize=12, fontweight='bold')
ax.set_xticklabels([c.capitalize() for c in CAT_ORDER], fontsize=10)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
for i, cat in enumerate(CAT_ORDER):
    ax.text(i, clin_fracs[i] + clin_hi[i] + 0.005,
            f"{clin_data[cat]['k']}/{clin_data[cat]['n']}",
            ha='center', fontsize=8)

# Panel D: Expression confound check
ax = axes[1, 1]
tertile_labels = ['Low', 'Mid', 'High']
x = np.arange(len(tertile_labels))
width = 0.2
for i, cat in enumerate(CAT_ORDER):
    fracs_expr = [expr_confound[t][cat]['frac'] for t in tertile_labels]
    ax.bar(x + (i - 1.5) * width, fracs_expr, width,
           color=CAT_COLORS[cat], label=cat.capitalize(),
           edgecolor='black', linewidth=0.3)
ax.set_xticks(x)
ax.set_xticklabels([f'{t}\nexpr' for t in tertile_labels], fontsize=10)
ax.set_ylabel('Fraction tractable', fontsize=10)
ax.set_title('D. Expression confound check', fontsize=12, fontweight='bold')
ax.legend(fontsize=8, title='Topology', title_fontsize=9)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
# Add Fisher results per tertile
for j, tert in enumerate(tertile_labels):
    res = expr_fisher_results[tert]
    star = significance_stars(res['p'])
    if star != 'ns':
        y_max = max(expr_confound[tert][c]['frac'] for c in CAT_ORDER) + 0.03
        ax.text(j, y_max, f'{star}\nOR={res["OR"]:.2f}', ha='center', fontsize=7,
                color='red')

plt.suptitle('Topology-Druggability Analysis: CD4+ T Cell CRISPRi Network',
             fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
fig.savefig(os.path.join(FIG_DIR, 'druggability_summary_figure.png'), dpi=150,
            bbox_inches='tight')
plt.close(fig)
print("Saved: druggability_summary_figure.png")

# ══════════════════════════════════════════════════════════════════════════
# FIGURE 4: EXPRESSION CONFOUND DETAIL
# ══════════════════════════════════════════════════════════════════════════

fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)

for j, tert in enumerate(tertile_labels):
    ax = axes[j]
    fracs_tert = [expr_confound[tert][c]['frac'] for c in CAT_ORDER]
    ci_lo_tert = [expr_confound[tert][c]['frac'] - expr_confound[tert][c]['ci_lo'] for c in CAT_ORDER]
    ci_hi_tert = [expr_confound[tert][c]['ci_hi'] - expr_confound[tert][c]['frac'] for c in CAT_ORDER]
    ax.bar(CAT_ORDER, fracs_tert,
           yerr=[ci_lo_tert, ci_hi_tert], capsize=4,
           color=[CAT_COLORS[c] for c in CAT_ORDER],
           edgecolor='black', linewidth=0.5)
    ax.set_title(f'{tert} expression tertile', fontsize=12, fontweight='bold')
    ax.set_xticklabels([c.capitalize() for c in CAT_ORDER], fontsize=9, rotation=30)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Annotate counts
    for i, cat in enumerate(CAT_ORDER):
        info = expr_confound[tert][cat]
        ax.text(i, fracs_tert[i] + ci_hi_tert[i] + 0.01,
                f"{info['k']}/{info['n']}", ha='center', fontsize=7)

    # Fisher result
    res = expr_fisher_results[tert]
    star = significance_stars(res['p'])
    ax.text(0.5, 0.95, f"relay vs rest: OR={res['OR']:.2f}, p={res['p']:.2e} {star}",
            transform=ax.transAxes, fontsize=8, ha='center', va='top',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', alpha=0.8))

axes[0].set_ylabel('Fraction tractable (any modality)', fontsize=11)
plt.suptitle('Druggability by Topology, Stratified by Expression Level',
             fontsize=13, fontweight='bold')
plt.tight_layout()
fig.savefig(os.path.join(FIG_DIR, 'druggability_expression_confound.png'), dpi=150,
            bbox_inches='tight')
plt.close(fig)
print("Saved: druggability_expression_confound.png")

# ══════════════════════════════════════════════════════════════════════════
# FINAL SUMMARY
# ══════════════════════════════════════════════════════════════════════════
print()
print("=" * 80)
print("KEY FINDINGS SUMMARY")
print("=" * 80)
print()
print(f"1. Relay genes show {'HIGHER' if frac_data['relay']['frac'] > frac_data['other']['frac'] else 'LOWER'} "
      f"overall tractability ({frac_data['relay']['frac']:.1%}) vs other ({frac_data['other']['frac']:.1%})")
print(f"   Fisher relay vs other: OR={fisher_results_overall['relay_vs_other']['OR']:.3f}, "
      f"p={fisher_results_overall['relay_vs_other']['p']:.2e}")
print(f"2. Chi-squared test (tractability x category): chi2={chi2:.2f}, p={p_chi:.2e}")
print(f"3. Kruskal-Wallis on best_bucket_rank: H={kw_stat:.2f}, p={kw_p:.2e}")
print(f"4. Relay approved drug targets: {approved_data['relay']['k']}/{approved_data['relay']['n']} "
      f"({approved_data['relay']['frac']:.1%})")
print(f"5. Expression confound: relay vs rest within tertiles:")
for tert in tertile_labels:
    res = expr_fisher_results[tert]
    print(f"   {tert}: OR={res['OR']:.3f}, p={res['p']:.2e}")
print()
print("=" * 80)
print("ALL OUTPUTS SAVED SUCCESSFULLY")
print("=" * 80)

sys.stdout = tee.terminal
tee.close()
print(f"\nResults saved to: {OUT_TXT}")
print(f"Summary CSV: {OUT_CSV}")
print(f"Figures: {FIG_DIR}/druggability_*.png")
