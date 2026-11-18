import os
import anndata as ad
import pandas as pd
import numpy as np
from scipy import stats
import os
import warnings
warnings.filterwarnings('ignore')

BASE = (os.environ["REPRO_DATA_ROOT"] + '')
DATA_PATH = (os.environ["REPRO_DATA_ROOT"] + '/data/GWCD4i.DE_stats.h5ad')

# Load data
print("Loading data...")
adata = ad.read_h5ad(DATA_PATH)
print(f"Full dataset: {adata.shape}")

# Print varm info for each condition
for cond in ['Rest', 'Stim8hr', 'Stim48hr']:
    key = f'measured_genes_stats_{cond}'
    df = pd.DataFrame(adata.varm[key], index=adata.var_names)
    print(f"\n=== {key} ===")
    print(f"Columns: {df.columns.tolist()}")
    print(df.head(5).to_string())

# Step 2: Compute in-degree per gene per condition
# Filter to high-quality perturbations
quality_mask = (adata.obs['ontarget_significant'] == True) & (adata.obs['offtarget_flag'] == False)
adata_hq = adata[quality_mask].copy()
print(f"\nHigh-quality perturbations: {adata_hq.shape[0]} (from {adata.shape[0]})")

conditions = ['Rest', 'Stim8hr', 'Stim48hr']
gene_names = adata.var['gene_name'].values if 'gene_name' in adata.var.columns else adata.var_names.values
var_names = adata.var_names.values

all_results = []

for cond in conditions:
    print(f"\n--- Processing {cond} ---")

    # Subset to this condition
    cond_mask = adata_hq.obs['culture_condition'] == cond
    adata_cond = adata_hq[cond_mask]
    print(f"  Perturbations in {cond}: {adata_cond.shape[0]}")

    # Get adj_p_value matrix for this condition
    adj_p = adata_cond.layers['adj_p_value']

    # Get perturbed gene names for this condition
    perturbed_genes = adata_cond.obs['target_contrast_gene_name'].values

    # Create a boolean matrix of significant effects (adj_p < 0.10)
    # But we need to exclude self-perturbation (where perturbed gene == measured gene)
    sig_matrix = np.array(adj_p < 0.10)  # ensure dense

    # For each perturbation (row), find if the perturbed gene matches any measured gene (column)
    # and set that entry to False (exclude self-effects)
    # Build a mapping from gene_name to column indices
    gene_name_to_col = {}
    for j, gn in enumerate(gene_names):
        if gn not in gene_name_to_col:
            gene_name_to_col[gn] = []
        gene_name_to_col[gn].append(j)

    # Zero out self-perturbation effects
    n_self_excluded = 0
    for i, pg in enumerate(perturbed_genes):
        if pg in gene_name_to_col:
            for j in gene_name_to_col[pg]:
                if sig_matrix[i, j]:
                    n_self_excluded += 1
                sig_matrix[i, j] = False

    print(f"  Self-perturbation effects excluded: {n_self_excluded}")

    # In-degree = number of perturbations that significantly regulate each gene (column sums)
    in_degree = sig_matrix.sum(axis=0)
    print(f"  In-degree range: {in_degree.min()} - {in_degree.max()}")
    print(f"  In-degree mean: {in_degree.mean():.2f}, median: {np.median(in_degree):.1f}")

    # Step 3: Extract out-degree (n_downstream) for this condition
    # Only for genes that are both perturbed AND measured
    obs_cond = adata_cond.obs.copy()
    out_degree_df = obs_cond[['target_contrast_gene_name', 'n_downstream']].copy()
    out_degree_df.columns = ['gene_name', 'out_degree']
    out_degree_df['out_degree'] = pd.to_numeric(out_degree_df['out_degree'], errors='coerce')

    # Build in-degree dataframe
    in_degree_df = pd.DataFrame({
        'gene_name': gene_names,
        'in_degree': in_degree
    })

    # Merge
    merged = pd.merge(out_degree_df, in_degree_df, on='gene_name', how='inner')
    merged['condition'] = cond

    print(f"  Genes with both in-degree and out-degree: {len(merged)}")
    print(f"  In-degree summary:")
    print(f"    {merged['in_degree'].describe().to_string()}")
    print(f"  Out-degree summary:")
    print(f"    {merged['out_degree'].describe().to_string()}")

    # Spearman correlation
    rho, pval = stats.spearmanr(merged['in_degree'], merged['out_degree'])
    print(f"  Spearman correlation: rho={rho:.4f}, p={pval:.2e}")

    all_results.append(merged)

# Combine all conditions
result_df = pd.concat(all_results, ignore_index=True)
result_df = result_df[['gene_name', 'condition', 'in_degree', 'out_degree']]

# Save to CSV
os.makedirs(f'{BASE}/reports', exist_ok=True)
os.makedirs(f'{BASE}/reports/analysis_outputs', exist_ok=True)
result_df.to_csv(f'{BASE}/reports/gene_topology.csv', index=False)
print(f"\nSaved gene_topology.csv with {len(result_df)} rows")
print(f"Columns: {result_df.columns.tolist()}")
print(f"Conditions: {result_df['condition'].value_counts().to_dict()}")

# Step 5: Top 20 genes by in-degree and out-degree for Stim48hr
stim48 = result_df[result_df['condition'] == 'Stim48hr'].copy()

print("\n" + "="*60)
print("TOP 20 GENES BY IN-DEGREE (Stim48hr)")
print("="*60)
top_in = stim48.nlargest(20, 'in_degree')[['gene_name', 'in_degree', 'out_degree']]
print(top_in.to_string(index=False))

print("\n" + "="*60)
print("TOP 20 GENES BY OUT-DEGREE (Stim48hr)")
print("="*60)
top_out = stim48.nlargest(20, 'out_degree')[['gene_name', 'out_degree', 'in_degree']]
print(top_out.to_string(index=False))

# Also save the top gene tables
top_in.to_csv(f'{BASE}/reports/analysis_outputs/top20_indegree_stim48hr.csv', index=False)
top_out.to_csv(f'{BASE}/reports/analysis_outputs/top20_outdegree_stim48hr.csv', index=False)
print(f"\nSaved top gene tables to reports/analysis_outputs/")
