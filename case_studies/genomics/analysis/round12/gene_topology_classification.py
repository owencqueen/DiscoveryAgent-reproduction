import os
"""
gene_topology_classification.py

Compute in-degree and out-degree for each gene in the Stim8hr condition of
the CD4+ T cell CRISPRi perturb-seq dataset, then classify genes into four
topological categories (Relay, Terminal, Upstream, Other) using 75th percentile
thresholds.

Definitions:
  - In-degree: number of QC-filtered Stim8hr perturbations that significantly
    affect the gene (adj_p_value < 0.10), excluding self-perturbation.
  - Out-degree: n_downstream from .obs for QC-filtered Stim8hr perturbations.
    For genes perturbed multiple times, we take the value from .obs (should be
    one QC-pass row per gene per condition).
  - Classification thresholds: 75th percentile of in-degree and out-degree
    computed on genes that have BOTH metrics defined.

Outputs:
  results/gene_topology_classification_stim8hr.csv  -- full gene table
  results/genes_relay.txt
  results/genes_terminal.txt
  results/genes_upstream.txt
  results/genes_other.txt
  results/genes_measured_only.txt
"""

import matplotlib
matplotlib.use('Agg')

import anndata as ad
import pandas as pd
import numpy as np
from scipy import sparse
import warnings
import os

warnings.filterwarnings('ignore')

# ============================================================================
# Configuration
# ============================================================================
BASE = (os.environ["REPRO_DATA_ROOT"] + '')
DATA_PATH = os.path.join(BASE, 'data', 'GWCD4i.DE_stats.h5ad')
RESULTS_DIR = os.path.join(BASE, 'results')
P_THRESHOLD = 0.10
CONDITION = 'Stim8hr'

os.makedirs(RESULTS_DIR, exist_ok=True)

# ============================================================================
# 1. Load data
# ============================================================================
print("=" * 70)
print("STEP 1: Loading data")
print("=" * 70)
adata = ad.read_h5ad(DATA_PATH)
print(f"Full dataset shape: {adata.shape}")
print(f"Conditions: {adata.obs['culture_condition'].value_counts().to_dict()}")

# ============================================================================
# 2. Filter to Stim8hr condition
# ============================================================================
print("\n" + "=" * 70)
print("STEP 2: Filtering to Stim8hr condition")
print("=" * 70)

stim8_mask = adata.obs['culture_condition'] == CONDITION
adata_stim8 = adata[stim8_mask].copy()
print(f"Stim8hr rows: {adata_stim8.shape[0]}")

# Apply quality filter: ontarget_significant == True AND offtarget_flag == False
qc_mask = (
    (adata_stim8.obs['ontarget_significant'] == True) &
    (adata_stim8.obs['offtarget_flag'] == False)
)
adata_qc = adata_stim8[qc_mask].copy()
print(f"QC-filtered Stim8hr rows: {adata_qc.shape[0]}")

n_obs, n_genes = adata_qc.shape
print(f"Shape after QC filter: {n_obs} perturbations x {n_genes} genes")

# ============================================================================
# 3. Build gene name mappings
# ============================================================================
print("\n" + "=" * 70)
print("STEP 3: Building gene name mappings")
print("=" * 70)

gene_names_var = adata_qc.var['gene_name'].values  # measured gene names
perturbed_genes = adata_qc.obs['target_contrast_gene_name'].values  # perturbed gene names

# Map gene_name -> column indices (some gene names may map to multiple ENSG IDs)
gene_name_to_cols = {}
for j, gn in enumerate(gene_names_var):
    gene_name_to_cols.setdefault(gn, []).append(j)

unique_measured = set(gene_name_to_cols.keys())
unique_perturbed = set(perturbed_genes)
overlap = unique_measured & unique_perturbed

print(f"Unique measured gene names: {len(unique_measured)}")
print(f"Unique perturbed gene names (QC Stim8hr): {len(unique_perturbed)}")
print(f"Overlap (both measured and perturbed): {len(overlap)}")

# ============================================================================
# 4. Compute IN-DEGREE
# ============================================================================
print("\n" + "=" * 70)
print("STEP 4: Computing IN-DEGREE for Stim8hr")
print("=" * 70)

# Extract adj_p_value matrix
adj_p_raw = adata_qc.layers['adj_p_value']
if sparse.issparse(adj_p_raw):
    adj_p_raw = adj_p_raw.toarray()
adj_p = np.array(adj_p_raw, dtype=np.float64)

# Significance mask (NaN comparisons yield False, which is correct)
sig_mask = adj_p < P_THRESHOLD
print(f"Total significant entries (before self-exclusion): {np.sum(sig_mask):,}")
print(f"NaN entries in adj_p: {np.sum(np.isnan(adj_p)):,}")

# Build self-effect exclusion mask
self_mask = np.zeros((n_obs, n_genes), dtype=bool)
for gn, cols in gene_name_to_cols.items():
    row_indices = np.where(perturbed_genes == gn)[0]
    if len(row_indices) > 0 and len(cols) > 0:
        self_mask[np.ix_(row_indices, cols)] = True

n_self_excluded = np.sum(sig_mask & self_mask)
print(f"Self-effect entries excluded: {n_self_excluded:,}")

# Final significance mask with self-exclusion
sig_no_self = sig_mask & ~self_mask
print(f"Total significant entries (after self-exclusion): {np.sum(sig_no_self):,}")

# In-degree per column (measured gene): count of perturbations affecting it
in_degree_per_col = np.sum(sig_no_self, axis=0)  # shape (n_genes,)

# Build in-degree dataframe and aggregate by gene name (sum duplicates)
in_degree_df = pd.DataFrame({
    'gene_name': gene_names_var,
    'in_degree': in_degree_per_col
})
in_degree_agg = in_degree_df.groupby('gene_name')['in_degree'].sum().reset_index()

print(f"\nIn-degree summary ({len(in_degree_agg)} genes):")
print(f"  Mean:   {in_degree_agg['in_degree'].mean():.2f}")
print(f"  Median: {in_degree_agg['in_degree'].median():.1f}")
print(f"  Min:    {in_degree_agg['in_degree'].min()}")
print(f"  Max:    {in_degree_agg['in_degree'].max()}")

# ============================================================================
# 5. Compute OUT-DEGREE
# ============================================================================
print("\n" + "=" * 70)
print("STEP 5: Computing OUT-DEGREE for Stim8hr")
print("=" * 70)

# Out-degree comes from .obs['n_downstream'] for QC-filtered Stim8hr perturbations.
# Map perturbation target gene name -> n_downstream.
# If a gene has multiple QC-pass rows at Stim8hr (unlikely but possible), take max.
out_degree_df = pd.DataFrame({
    'gene_name': perturbed_genes,
    'n_downstream': adata_qc.obs['n_downstream'].values
})
out_degree_agg = out_degree_df.groupby('gene_name')['n_downstream'].max().reset_index()
out_degree_agg.columns = ['gene_name', 'out_degree']

print(f"Out-degree summary ({len(out_degree_agg)} genes):")
print(f"  Mean:   {out_degree_agg['out_degree'].mean():.2f}")
print(f"  Median: {out_degree_agg['out_degree'].median():.1f}")
print(f"  Min:    {out_degree_agg['out_degree'].min()}")
print(f"  Max:    {out_degree_agg['out_degree'].max()}")

# ============================================================================
# 6. Merge into gene-level table
# ============================================================================
print("\n" + "=" * 70)
print("STEP 6: Merging in-degree and out-degree")
print("=" * 70)

# Start with all measured genes (in-degree defined for all)
gene_df = in_degree_agg.copy()

# Merge out-degree (left join: genes not perturbed will have NaN out-degree)
gene_df = gene_df.merge(out_degree_agg, on='gene_name', how='left')

# Flag genes that have both metrics vs measured-only
gene_df['has_out_degree'] = gene_df['out_degree'].notna()

n_both = gene_df['has_out_degree'].sum()
n_measured_only = (~gene_df['has_out_degree']).sum()
print(f"Genes with both in-degree and out-degree: {n_both}")
print(f"Genes with in-degree only (measured_only): {n_measured_only}")

# ============================================================================
# 7. Classify genes using Q75 thresholds
# ============================================================================
print("\n" + "=" * 70)
print("STEP 7: Classifying genes into topological categories")
print("=" * 70)

# Compute Q75 thresholds on genes that have BOTH metrics
both_mask = gene_df['has_out_degree']
q75_in = gene_df.loc[both_mask, 'in_degree'].quantile(0.75)
q75_out = gene_df.loc[both_mask, 'out_degree'].quantile(0.75)

print(f"Q75 thresholds (computed on {n_both} genes with both metrics):")
print(f"  In-degree  Q75 = {q75_in}")
print(f"  Out-degree Q75 = {q75_out}")

# Classify
def classify_gene(row):
    if not row['has_out_degree']:
        return 'measured_only'
    high_in = row['in_degree'] >= q75_in
    high_out = row['out_degree'] >= q75_out
    if high_in and high_out:
        return 'relay'
    elif high_in and not high_out:
        return 'terminal'
    elif not high_in and high_out:
        return 'upstream'
    else:
        return 'other'

gene_df['category'] = gene_df.apply(classify_gene, axis=1)

# Print category counts
print("\nCategory counts:")
category_counts = gene_df['category'].value_counts()
for cat in ['relay', 'terminal', 'upstream', 'other', 'measured_only']:
    if cat in category_counts.index:
        count = category_counts[cat]
        pct = 100 * count / len(gene_df)
        print(f"  {cat:15s}: {count:6d}  ({pct:.1f}%)")
print(f"  {'TOTAL':15s}: {len(gene_df):6d}")

# Among classified genes only (excluding measured_only)
classified = gene_df[gene_df['category'] != 'measured_only']
print(f"\nAmong {len(classified)} classified genes (with both metrics):")
for cat in ['relay', 'terminal', 'upstream', 'other']:
    if cat in category_counts.index:
        count = category_counts[cat]
        pct = 100 * count / len(classified)
        print(f"  {cat:15s}: {count:6d}  ({pct:.1f}%)")

# ============================================================================
# 8. Summary statistics per category
# ============================================================================
print("\n" + "=" * 70)
print("STEP 8: Summary statistics per category")
print("=" * 70)

for cat in ['relay', 'terminal', 'upstream', 'other']:
    subset = gene_df[gene_df['category'] == cat]
    if len(subset) > 0:
        print(f"\n  {cat.upper()} (n={len(subset)}):")
        print(f"    In-degree:  mean={subset['in_degree'].mean():.1f}, "
              f"median={subset['in_degree'].median():.1f}, "
              f"range=[{subset['in_degree'].min()}, {subset['in_degree'].max()}]")
        print(f"    Out-degree: mean={subset['out_degree'].mean():.1f}, "
              f"median={subset['out_degree'].median():.1f}, "
              f"range=[{int(subset['out_degree'].min())}, {int(subset['out_degree'].max())}]")

# ============================================================================
# 9. Save outputs
# ============================================================================
print("\n" + "=" * 70)
print("STEP 9: Saving outputs")
print("=" * 70)

# Sort by category then by in_degree descending for readability
gene_df_sorted = gene_df.sort_values(
    ['category', 'in_degree'], ascending=[True, False]
).reset_index(drop=True)

# Save full classification table
output_csv = os.path.join(RESULTS_DIR, 'gene_topology_classification_stim8hr.csv')
gene_df_sorted.to_csv(output_csv, index=False)
print(f"Saved: {output_csv} ({len(gene_df_sorted)} genes)")

# Save per-category gene name lists
for cat in ['relay', 'terminal', 'upstream', 'other', 'measured_only']:
    genes_in_cat = gene_df_sorted.loc[
        gene_df_sorted['category'] == cat, 'gene_name'
    ].tolist()
    list_path = os.path.join(RESULTS_DIR, f'genes_{cat}.txt')
    with open(list_path, 'w') as f:
        for g in genes_in_cat:
            f.write(g + '\n')
    print(f"Saved: {list_path} ({len(genes_in_cat)} genes)")

# ============================================================================
# 10. Print top genes per category
# ============================================================================
print("\n" + "=" * 70)
print("STEP 10: Top 10 genes per category")
print("=" * 70)

for cat in ['relay', 'terminal', 'upstream', 'other']:
    subset = gene_df[gene_df['category'] == cat].copy()
    if cat == 'relay':
        # Sort by sum of in+out degree
        subset['total_degree'] = subset['in_degree'] + subset['out_degree']
        subset = subset.sort_values('total_degree', ascending=False)
    elif cat == 'terminal':
        subset = subset.sort_values('in_degree', ascending=False)
    elif cat == 'upstream':
        subset = subset.sort_values('out_degree', ascending=False)
    else:
        subset = subset.sort_values('in_degree', ascending=False)

    print(f"\n  Top 10 {cat.upper()} genes:")
    for i, (_, row) in enumerate(subset.head(10).iterrows()):
        out_str = f"{int(row['out_degree'])}" if pd.notna(row['out_degree']) else 'N/A'
        print(f"    {i+1:2d}. {row['gene_name']:12s}  in_deg={int(row['in_degree']):5d}  out_deg={out_str:>5s}")

print("\nDone.")
