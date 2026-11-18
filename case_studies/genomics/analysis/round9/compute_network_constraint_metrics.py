import os
"""
compute_network_constraint_metrics.py

Computes network metrics (in-degree, out-degree) from perturb-seq data,
merges with genetic constraint metrics (s_het, pLI, LOEUF), and computes
Spearman correlations across conditions.

Outputs:
  - data/processed/network_metrics_by_condition.csv
  - data/processed/merged_constraint_network.csv
  - data/processed/spearman_correlations_stim8hr.csv
  - data/processed/spearman_correlations_all_conditions.csv
"""

import anndata as ad
import pandas as pd
import numpy as np
from scipy import stats, sparse
import warnings
warnings.filterwarnings('ignore')

BASE = (os.environ["REPRO_DATA_ROOT"] + '')

# =============================================================================
# STEP 1: Load perturb-seq data and compute network metrics
# =============================================================================
print("=" * 80)
print("STEP 1: Load perturb-seq data and compute network metrics")
print("=" * 80)

# Load AnnData
print("Loading AnnData...")
adata = ad.read_h5ad(f"{BASE}/data/GWCD4i.DE_stats.h5ad")
print(f"Shape: {adata.shape}")
print(f"Culture conditions: {adata.obs['culture_condition'].value_counts().to_dict()}")

# Get gene name mapping from var
var_df = adata.var.copy()
gene_names = var_df['gene_name'].values  # measured genes (columns)

conditions = ['Rest', 'Stim8hr', 'Stim48hr']
all_condition_dfs = {}

for cond in conditions:
    print(f"\n=== Processing {cond} ===")

    # Filter to this condition
    mask_cond = adata.obs['culture_condition'] == cond
    adata_cond = adata[mask_cond]
    print(f"  Total perturbations in {cond}: {adata_cond.shape[0]}")

    # Filter to high-quality perturbations
    mask_hq = (adata_cond.obs['ontarget_significant'] == True) & (adata_cond.obs['offtarget_flag'] == False)
    adata_hq = adata_cond[mask_hq]
    print(f"  High-quality perturbations: {adata_hq.shape[0]}")

    # OUT-DEGREE: use n_downstream from .obs
    out_degree = adata_hq.obs[['target_contrast_gene_name', 'n_downstream']].copy()
    out_degree.columns = ['gene_name', 'out_degree']
    out_degree = out_degree.reset_index(drop=True)

    # Also get baseMean for the target gene
    basemean_col = adata_hq.obs[['target_contrast_gene_name', 'target_baseMean']].copy()
    basemean_col.columns = ['gene_name', 'target_baseMean']
    basemean_col = basemean_col.reset_index(drop=True)

    # IN-DEGREE: For each measured gene (column), count how many HQ perturbations
    # significantly affect it (adj_p_value < 0.10), excluding self-perturbation
    print("  Computing in-degree (this may take a while)...")
    adj_p = adata_hq.layers['adj_p_value']
    target_genes = adata_hq.obs['target_contrast_gene_name'].values

    # Convert to dense if sparse (needed for element-wise operations)
    if sparse.issparse(adj_p):
        # Process in chunks to manage memory
        n_obs, n_vars = adj_p.shape
        in_degree_counts = np.zeros(n_vars, dtype=int)

        chunk_size = 1000
        for start in range(0, n_obs, chunk_size):
            end = min(start + chunk_size, n_obs)
            chunk = adj_p[start:end]
            if sparse.issparse(chunk):
                chunk = chunk.toarray()

            # Significant hits (adj_p < 0.10)
            sig_mask = chunk < 0.10

            # Create self-perturbation mask for this chunk
            chunk_targets = target_genes[start:end]
            for i, tgt in enumerate(chunk_targets):
                # Find if target gene is in measured genes (columns)
                col_idx = np.where(gene_names == tgt)[0]
                if len(col_idx) > 0:
                    sig_mask[i, col_idx[0]] = False  # Exclude self-perturbation

            in_degree_counts += sig_mask.sum(axis=0)

    else:
        sig_mask = adj_p < 0.10
        # Exclude self-perturbation
        for i, tgt in enumerate(target_genes):
            col_idx = np.where(gene_names == tgt)[0]
            if len(col_idx) > 0:
                sig_mask[i, col_idx[0]] = False
        in_degree_counts = sig_mask.sum(axis=0)

    # Create in-degree DataFrame
    in_degree_df = pd.DataFrame({
        'gene_name': gene_names,
        'in_degree': in_degree_counts
    })

    print(f"  In-degree stats: mean={in_degree_counts.mean():.1f}, max={in_degree_counts.max()}")
    print(f"  Out-degree stats: mean={out_degree['out_degree'].mean():.1f}, max={out_degree['out_degree'].max()}")

    # Merge out-degree and baseMean (per perturbed gene) with in-degree (per measured gene)
    # Some genes appear as both perturbed and measured
    merged = pd.merge(out_degree, in_degree_df, on='gene_name', how='outer')
    merged = pd.merge(merged, basemean_col, on='gene_name', how='outer')

    # If a gene was perturbed multiple times (shouldn't happen within same condition), take first
    merged = merged.groupby('gene_name').first().reset_index()

    merged['condition'] = cond
    all_condition_dfs[cond] = merged
    print(f"  Merged gene count: {len(merged)}")

# Combine all conditions
network_df = pd.concat(all_condition_dfs.values(), ignore_index=True)
print(f"\nTotal network entries (gene x condition): {len(network_df)}")

# Save network metrics
network_df.to_csv(f"{BASE}/data/processed/network_metrics_by_condition.csv", index=False)
print(f"Saved network metrics to data/processed/network_metrics_by_condition.csv")


# =============================================================================
# STEP 2: Load and process s_het estimates
# =============================================================================
print("\n" + "=" * 80)
print("STEP 2: Load and process s_het estimates")
print("=" * 80)

# Load s_het
shet_df = pd.read_csv(f"{BASE}/data/external/s_het_estimates.genebayes.tsv", sep='\t')
print(f"\ns_het data: {shet_df.shape}")
print(shet_df.head())

# The s_het file has 'ensg' (Ensembl) and 'hgnc' (HGNC ID like HGNC:24141)
# We need to map to gene symbols. The h5ad var has gene_ids (Ensembl) and gene_name (symbol)
# Let's use Ensembl IDs for the join

# Create a gene name lookup from the h5ad var table
gene_lookup = adata.var[['gene_ids', 'gene_name']].copy().reset_index(drop=True)
# Strip version numbers from Ensembl IDs if present
gene_lookup['ensg_clean'] = gene_lookup['gene_ids'].str.split('.').str[0]
shet_df['ensg_clean'] = shet_df['ensg'].str.split('.').str[0]

shet_mapped = pd.merge(shet_df, gene_lookup[['ensg_clean', 'gene_name']], on='ensg_clean', how='inner')
print(f"s_het genes mapped to h5ad gene names: {len(shet_mapped)}")

# Keep relevant columns
shet_final = shet_mapped[['gene_name', 'ensg', 'post_mean', 'post_lower_95', 'post_upper_95', 'obs_lof', 'exp_lof']].copy()
shet_final.rename(columns={'post_mean': 's_het'}, inplace=True)
print(f"s_het final: {shet_final.shape}")


# =============================================================================
# STEP 3: Load and process gnomAD constraint
# =============================================================================
print("\n" + "=" * 80)
print("STEP 3: Load and process gnomAD constraint")
print("=" * 80)

# Load gnomAD - filter to canonical transcripts for one row per gene
print("\nLoading gnomAD constraint data...")
gnomad_df = pd.read_csv(f"{BASE}/data/external/gnomad_constraint.tsv", sep='\t', low_memory=False)
print(f"gnomAD raw: {gnomad_df.shape}")

# Filter to canonical transcripts (or MANE select if available)
# Try mane_select first, fall back to canonical
if 'mane_select' in gnomad_df.columns:
    gnomad_canon = gnomad_df[gnomad_df['mane_select'] == True].copy()
    print(f"MANE select transcripts: {len(gnomad_canon)}")
    if len(gnomad_canon) < 5000:
        gnomad_canon = gnomad_df[gnomad_df['canonical'] == True].copy()
        print(f"Falling back to canonical transcripts: {len(gnomad_canon)}")
else:
    gnomad_canon = gnomad_df[gnomad_df['canonical'] == True].copy()
    print(f"Canonical transcripts: {len(gnomad_canon)}")

# If still too few, just deduplicate by gene name keeping first
if len(gnomad_canon) < 5000:
    gnomad_canon = gnomad_df.drop_duplicates(subset='gene', keep='first').copy()
    print(f"Deduplicated by gene: {len(gnomad_canon)}")

# Select relevant columns - handle possible column name variations
loeuf_col = 'lof.oe_ci.upper' if 'lof.oe_ci.upper' in gnomad_canon.columns else None
pli_col = 'lof.pLI' if 'lof.pLI' in gnomad_canon.columns else None

print(f"LOEUF column: {loeuf_col}")
print(f"pLI column: {pli_col}")

cols_to_keep = ['gene']
if loeuf_col:
    cols_to_keep.append(loeuf_col)
if pli_col:
    cols_to_keep.append(pli_col)

gnomad_final = gnomad_canon[cols_to_keep].copy()
gnomad_final.rename(columns={'gene': 'gene_name'}, inplace=True)
if loeuf_col:
    gnomad_final.rename(columns={loeuf_col: 'LOEUF'}, inplace=True)
if pli_col:
    gnomad_final.rename(columns={pli_col: 'pLI'}, inplace=True)

# Convert to numeric
for col in ['LOEUF', 'pLI']:
    if col in gnomad_final.columns:
        gnomad_final[col] = pd.to_numeric(gnomad_final[col], errors='coerce')

# Deduplicate by gene name (take first occurrence)
gnomad_final = gnomad_final.drop_duplicates(subset='gene_name', keep='first')
print(f"gnomAD final: {gnomad_final.shape}")
print(gnomad_final.head())


# =============================================================================
# STEP 4: Merge everything
# =============================================================================
print("\n" + "=" * 80)
print("STEP 4: Merge everything")
print("=" * 80)

# Merge network metrics with s_het and gnomAD for each condition
print("\n=== MERGING ALL DATA ===")

merged_all = pd.merge(network_df, shet_final, on='gene_name', how='left')
merged_all = pd.merge(merged_all, gnomad_final, on='gene_name', how='left')

print(f"Total merged entries: {len(merged_all)}")
for cond in conditions:
    cond_data = merged_all[merged_all['condition'] == cond]
    n_total = len(cond_data)
    n_shet = cond_data['s_het'].notna().sum()
    n_pli = cond_data['pLI'].notna().sum() if 'pLI' in cond_data.columns else 0
    n_loeuf = cond_data['LOEUF'].notna().sum() if 'LOEUF' in cond_data.columns else 0
    n_outdeg = cond_data['out_degree'].notna().sum()
    n_indeg = cond_data['in_degree'].notna().sum()
    n_complete = cond_data.dropna(subset=['s_het', 'out_degree', 'in_degree']).shape[0]
    if 'pLI' in cond_data.columns:
        n_complete_all = cond_data.dropna(subset=['s_het', 'pLI', 'LOEUF', 'out_degree', 'in_degree']).shape[0]
    else:
        n_complete_all = n_complete

    print(f"\n  {cond}: {n_total} genes")
    print(f"    s_het available: {n_shet}")
    print(f"    pLI available: {n_pli}")
    print(f"    LOEUF available: {n_loeuf}")
    print(f"    out_degree available: {n_outdeg}")
    print(f"    in_degree available: {n_indeg}")
    print(f"    Complete (s_het + out + in): {n_complete}")
    print(f"    Complete (all 5 metrics): {n_complete_all}")

# Save merged data
merged_all.to_csv(f"{BASE}/data/processed/merged_constraint_network.csv", index=False)
print(f"\nSaved merged data to data/processed/merged_constraint_network.csv")


# =============================================================================
# STEP 5: Descriptive statistics (Stim8hr focus)
# =============================================================================
print("\n" + "=" * 80)
print("STEP 5: DESCRIPTIVE STATISTICS (Stim8hr)")
print("=" * 80)

stim8 = merged_all[merged_all['condition'] == 'Stim8hr'].copy()

# Basic s_het stats
print(f"\n--- s_het statistics ---")
shet_data = stim8['s_het'].dropna()
print(f"N genes with s_het: {len(shet_data)}")
print(f"Mean: {shet_data.mean():.6f}")
print(f"Median: {shet_data.median():.6f}")
print(f"Std: {shet_data.std():.6f}")
print(f"Min: {shet_data.min():.6f}")
print(f"Max: {shet_data.max():.6f}")
print(f"25th percentile: {shet_data.quantile(0.25):.6f}")
print(f"75th percentile: {shet_data.quantile(0.75):.6f}")

# Compute Spearman correlations
print(f"\n--- Spearman correlations (Stim8hr) ---")
metrics_to_correlate = ['in_degree', 'out_degree', 'target_baseMean']
log_metrics = {
    'log1p_in_degree': np.log1p(stim8['in_degree']),
    'log1p_out_degree': np.log1p(stim8['out_degree']),
}
stim8['log1p_in_degree'] = log_metrics['log1p_in_degree']
stim8['log1p_out_degree'] = log_metrics['log1p_out_degree']
stim8['log1p_baseMean'] = np.log1p(stim8['target_baseMean'])

all_metrics = ['in_degree', 'out_degree', 'log1p_in_degree', 'log1p_out_degree', 'target_baseMean']
constraint_measures = ['s_het']
if 'pLI' in stim8.columns:
    constraint_measures.append('pLI')
if 'LOEUF' in stim8.columns:
    constraint_measures.append('LOEUF')

results = []
for constraint in constraint_measures:
    for metric in all_metrics:
        valid = stim8[[constraint, metric]].dropna()
        if len(valid) > 10:
            rho, pval = stats.spearmanr(valid[constraint], valid[metric])
            results.append({
                'constraint_metric': constraint,
                'network_metric': metric,
                'spearman_rho': rho,
                'p_value': pval,
                'n_genes': len(valid)
            })
            print(f"  {constraint} vs {metric}: rho={rho:.4f}, p={pval:.2e}, n={len(valid)}")

results_df = pd.DataFrame(results)
results_df.to_csv(f"{BASE}/data/processed/spearman_correlations_stim8hr.csv", index=False)
print(f"\nSaved Spearman correlations to data/processed/spearman_correlations_stim8hr.csv")

# Also compute for all conditions
print("\n--- Spearman correlations (all conditions) ---")
all_results = []
for cond in conditions:
    cond_data = merged_all[merged_all['condition'] == cond].copy()
    cond_data['log1p_in_degree'] = np.log1p(cond_data['in_degree'])
    cond_data['log1p_out_degree'] = np.log1p(cond_data['out_degree'])

    for constraint in constraint_measures:
        for metric in all_metrics:
            valid = cond_data[[constraint, metric]].dropna()
            if len(valid) > 10:
                rho, pval = stats.spearmanr(valid[constraint], valid[metric])
                all_results.append({
                    'condition': cond,
                    'constraint_metric': constraint,
                    'network_metric': metric,
                    'spearman_rho': rho,
                    'p_value': pval,
                    'n_genes': len(valid)
                })
                print(f"  [{cond}] {constraint} vs {metric}: rho={rho:.4f}, p={pval:.2e}, n={len(valid)}")

all_results_df = pd.DataFrame(all_results)
all_results_df.to_csv(f"{BASE}/data/processed/spearman_correlations_all_conditions.csv", index=False)
print(f"\nSaved all-conditions correlations to data/processed/spearman_correlations_all_conditions.csv")

print("\n=== ANALYSIS COMPLETE ===")
