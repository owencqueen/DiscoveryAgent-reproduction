import os

'Compute Florida premium distributions for 2018 and 2023.\n\nPUMA boundaries differ between the two years; statewide distributions are compared.'

import warnings
warnings.filterwarnings('ignore')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd

from io import StringIO

# ============================================================================
# CONFIGURATION
# ============================================================================
BASE_DIR = (os.environ["REPRO_DATA_ROOT"] + '')

DATA_2018 = f"{BASE_DIR}/data/downloads/pums_housing_2018/psam_h12.csv"
DATA_2023 = f"{BASE_DIR}/data/downloads/pums_housing_2023/psam_h12.csv"
PUMA_MEASURES_2023 = f"{BASE_DIR}/data/processed/puma_three_insurance_measures.csv"

FIG_DIR = f"{BASE_DIR}/reports/figures"
OUT_DIR = f"{BASE_DIR}/reports/analysis_outputs"
PROCESSED_DIR = f"{BASE_DIR}/data/processed"

# Collect all text output in a buffer
output_buffer = StringIO()

def tee_print(*args, **kwargs):
    """Print to both stdout and the output buffer."""
    import builtins
    text = " ".join(str(a) for a in args)
    builtins.print(text, **kwargs)
    print(text, file=output_buffer, **kwargs)


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def weighted_median(values, weights):
    """Compute weighted median using cumulative weight approach."""
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    mask = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    values, weights = values[mask], weights[mask]
    if len(values) == 0:
        return np.nan
    sorted_indices = np.argsort(values)
    sorted_values = values[sorted_indices]
    sorted_weights = weights[sorted_indices]
    cumulative_weight = np.cumsum(sorted_weights)
    cutoff = sorted_weights.sum() / 2.0
    idx = np.searchsorted(cumulative_weight, cutoff)
    return sorted_values[min(idx, len(sorted_values) - 1)]


def weighted_percentile(values, weights, percentiles):
    """
    Compute weighted percentiles using np.interp on the cumulative weight distribution.
    percentiles: list of values in [0, 100].
    Returns array of percentile values.
    """
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    mask = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    values, weights = values[mask], weights[mask]
    if len(values) == 0:
        return [np.nan] * len(percentiles)
    sorted_indices = np.argsort(values)
    sorted_values = values[sorted_indices]
    sorted_weights = weights[sorted_indices]
    cumulative_weight = np.cumsum(sorted_weights)
    # Normalize to [0, 100]
    cumulative_pct = 100.0 * cumulative_weight / cumulative_weight[-1]
    results = np.interp(percentiles, cumulative_pct, sorted_values)
    return results


def weighted_mean(values, weights):
    """Compute weighted mean."""
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    mask = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    return np.average(values[mask], weights=weights[mask])


# ============================================================================
# PART 1: Compute 2018 FL PUMA-level log_median_premium
# ============================================================================
tee_print("=" * 80)
tee_print("PART 1: Computing 2018 FL PUMA-level Insurance Measures")
tee_print("=" * 80)

# Load 2018 data -- only columns we need to save memory
cols_2018 = ['ST', 'PUMA', 'WGTP', 'TEN', 'INSP', 'VALP', 'HINCP', 'MV']
df_2018_raw = pd.read_csv(DATA_2018, usecols=cols_2018)
tee_print(f"2018 FL Housing PUMS: {len(df_2018_raw):,} total records")
tee_print(f"  State (ST) values: {sorted(df_2018_raw['ST'].dropna().unique())}")

# Filter to owner-occupied with valid insurance, value, income, weight
df_2018_owners = df_2018_raw[
    (df_2018_raw['TEN'].isin([1, 2])) &
    (df_2018_raw['INSP'] > 0) &
    (df_2018_raw['VALP'] > 0) &
    (df_2018_raw['HINCP'] > 0) &
    (df_2018_raw['WGTP'] > 0)
].copy()
tee_print(f"  After filtering (owner-occupied, INSP>0, VALP>0, HINCP>0, WGTP>0): {len(df_2018_owners):,}")

# Compute PUMA-level measures for 2018
puma_2018_list = []
for puma, grp in df_2018_owners.groupby('PUMA'):
    med_insp = weighted_median(grp['INSP'].values, grp['WGTP'].values)
    med_hincp = weighted_median(grp['HINCP'].values, grp['WGTP'].values)
    med_valp = weighted_median(grp['VALP'].values, grp['WGTP'].values)
    total_weight = grp['WGTP'].sum()
    puma_2018_list.append({
        'puma': puma,
        'weighted_median_premium': med_insp,
        'log_median_premium_2018': np.log(med_insp) if med_insp > 0 else np.nan,
        'weighted_median_income': med_hincp,
        'log_median_income_2018': np.log(med_hincp) if med_hincp > 0 else np.nan,
        'weighted_median_home_value': med_valp,
        'log_median_home_value_2018': np.log(med_valp) if med_valp > 0 else np.nan,
        'weighted_owner_units': total_weight,
        'n_records': len(grp)
    })

puma_2018 = pd.DataFrame(puma_2018_list)
tee_print(f"  Number of 2018 PUMAs (2010-vintage): {len(puma_2018)}")
tee_print(f"\n  2018 PUMA-level log_median_premium distribution:")
tee_print(f"    Mean:   {puma_2018['log_median_premium_2018'].mean():.4f}")
tee_print(f"    Std:    {puma_2018['log_median_premium_2018'].std():.4f}")
tee_print(f"    Min:    {puma_2018['log_median_premium_2018'].min():.4f}")
tee_print(f"    Q1:     {puma_2018['log_median_premium_2018'].quantile(0.25):.4f}")
tee_print(f"    Median: {puma_2018['log_median_premium_2018'].median():.4f}")
tee_print(f"    Q3:     {puma_2018['log_median_premium_2018'].quantile(0.75):.4f}")
tee_print(f"    Max:    {puma_2018['log_median_premium_2018'].max():.4f}")

# Save 2018 PUMA measures
puma_2018_out = puma_2018.copy()
puma_2018_out.insert(0, 'state_fips', 12)
puma_2018_out.insert(1, 'state_abbr', 'FL')
puma_2018_out.to_csv(f"{PROCESSED_DIR}/florida_puma_insurance_2018.csv", index=False)
tee_print(f"\n  Saved 2018 PUMA measures to: data/processed/florida_puma_insurance_2018.csv")


# ============================================================================
# PART 2: Compare State-Wide Distributions (2018 vs 2023)
# ============================================================================
tee_print("\n" + "=" * 80)
tee_print("PART 2: State-Wide Distribution Comparison (2018 vs 2023)")
tee_print("=" * 80)

# --- 2018 individual-level weighted percentiles ---
pctiles = [10, 25, 50, 75, 90]
insp_pctiles_2018 = weighted_percentile(
    df_2018_owners['INSP'].values,
    df_2018_owners['WGTP'].values,
    pctiles
)
mean_insp_2018 = weighted_mean(df_2018_owners['INSP'].values, df_2018_owners['WGTP'].values)

tee_print(f"\n  2018 FL State-Wide INSP (Annual Insurance Premium, nominal $):")
tee_print(f"    Weighted mean: ${mean_insp_2018:,.0f}")
for p, v in zip(pctiles, insp_pctiles_2018):
    tee_print(f"    P{p:02d}: ${v:,.0f}")

# --- 2023 individual-level (load raw data, same filters) ---
cols_2023 = ['STATE', 'PUMA', 'WGTP', 'TEN', 'INSP', 'VALP', 'HINCP', 'MV']
df_2023_raw = pd.read_csv(DATA_2023, usecols=cols_2023)
tee_print(f"\n  2023 FL Housing PUMS: {len(df_2023_raw):,} total records")

df_2023_owners = df_2023_raw[
    (df_2023_raw['TEN'].isin([1, 2])) &
    (df_2023_raw['INSP'] > 0) &
    (df_2023_raw['VALP'] > 0) &
    (df_2023_raw['HINCP'] > 0) &
    (df_2023_raw['WGTP'] > 0)
].copy()
tee_print(f"  After filtering (owner-occupied, INSP>0, VALP>0, HINCP>0, WGTP>0): {len(df_2023_owners):,}")

insp_pctiles_2023 = weighted_percentile(
    df_2023_owners['INSP'].values,
    df_2023_owners['WGTP'].values,
    pctiles
)
mean_insp_2023 = weighted_mean(df_2023_owners['INSP'].values, df_2023_owners['WGTP'].values)

tee_print(f"\n  2023 FL State-Wide INSP (Annual Insurance Premium, nominal $):")
tee_print(f"    Weighted mean: ${mean_insp_2023:,.0f}")
for p, v in zip(pctiles, insp_pctiles_2023):
    tee_print(f"    P{p:02d}: ${v:,.0f}")

# --- Changes ---
tee_print(f"\n  Change in INSP Distribution (2018 -> 2023, nominal $):")
tee_print(f"    {'Statistic':<12} {'2018':>10} {'2023':>10} {'Change':>10} {'% Change':>10}")
tee_print(f"    {'-'*52}")
tee_print(f"    {'Mean':<12} {mean_insp_2018:>10,.0f} {mean_insp_2023:>10,.0f} "
          f"{mean_insp_2023 - mean_insp_2018:>+10,.0f} "
          f"{100 * (mean_insp_2023 - mean_insp_2018) / mean_insp_2018:>+9.1f}%")
for p, v18, v23 in zip(pctiles, insp_pctiles_2018, insp_pctiles_2023):
    chg = v23 - v18
    pct_chg = 100 * chg / v18 if v18 != 0 else np.nan
    tee_print(f"    {'P' + str(p):<12} {v18:>10,.0f} {v23:>10,.0f} {chg:>+10,.0f} {pct_chg:>+9.1f}%")

# --- PUMA-level log_median_premium distributions ---
# 2023 PUMA-level: compute from raw data
puma_2023_list = []
for puma, grp in df_2023_owners.groupby('PUMA'):
    med_insp = weighted_median(grp['INSP'].values, grp['WGTP'].values)
    puma_2023_list.append({
        'puma': puma,
        'log_median_premium_2023': np.log(med_insp) if med_insp > 0 else np.nan,
        'weighted_owner_units': grp['WGTP'].sum()
    })
puma_2023_computed = pd.DataFrame(puma_2023_list)

tee_print(f"\n  PUMA-level log_median_premium distribution comparison:")
tee_print(f"    {'Statistic':<12} {'2018 (N=' + str(len(puma_2018)) + ')':>18} "
          f"{'2023 (N=' + str(len(puma_2023_computed)) + ')':>18}")
tee_print(f"    {'-'*50}")
for stat_name, stat_func in [
    ('Mean', lambda s: s.mean()),
    ('Std', lambda s: s.std()),
    ('Min', lambda s: s.min()),
    ('Q1', lambda s: s.quantile(0.25)),
    ('Median', lambda s: s.median()),
    ('Q3', lambda s: s.quantile(0.75)),
    ('Max', lambda s: s.max()),
]:
    v18 = stat_func(puma_2018['log_median_premium_2018'])
    v23 = stat_func(puma_2023_computed['log_median_premium_2023'])
    tee_print(f"    {stat_name:<12} {v18:>18.4f} {v23:>18.4f}")

# State-wide weighted mean of PUMA-level log_median_premium
wt_mean_lmp_2018 = np.average(
    puma_2018['log_median_premium_2018'],
    weights=puma_2018['weighted_owner_units']
)
wt_mean_lmp_2023 = np.average(
    puma_2023_computed['log_median_premium_2023'],
    weights=puma_2023_computed['weighted_owner_units']
)
# Weighted median of PUMA-level log_median_premium
wt_med_lmp_2018 = weighted_median(
    puma_2018['log_median_premium_2018'].values,
    puma_2018['weighted_owner_units'].values
)
wt_med_lmp_2023 = weighted_median(
    puma_2023_computed['log_median_premium_2023'].values,
    puma_2023_computed['weighted_owner_units'].values
)

tee_print(f"\n  State-wide weighted aggregation of PUMA-level log_median_premium:")
tee_print(f"    Weighted mean:  2018={wt_mean_lmp_2018:.4f}  2023={wt_mean_lmp_2023:.4f}  "
          f"Change={wt_mean_lmp_2023 - wt_mean_lmp_2018:+.4f} "
          f"({100 * (wt_mean_lmp_2023 - wt_mean_lmp_2018) / wt_mean_lmp_2018:+.2f}%)")
tee_print(f"    Weighted median: 2018={wt_med_lmp_2018:.4f}  2023={wt_med_lmp_2023:.4f}  "
          f"Change={wt_med_lmp_2023 - wt_med_lmp_2018:+.4f} "
          f"({100 * (wt_med_lmp_2023 - wt_med_lmp_2018) / wt_med_lmp_2018:+.2f}%)")

# Convert log difference to approximate dollar change
med_2018_dollars = np.exp(wt_med_lmp_2018)
med_2023_dollars = np.exp(wt_med_lmp_2023)
tee_print(f"\n    In dollar terms (exp of weighted median):")
tee_print(f"      2018: ${med_2018_dollars:,.0f}/year")
tee_print(f"      2023: ${med_2023_dollars:,.0f}/year")
tee_print(f"      Change: ${med_2023_dollars - med_2018_dollars:+,.0f} "
          f"({100 * (med_2023_dollars - med_2018_dollars) / med_2018_dollars:+.1f}%)")



# --------------------------------------------------------------------------
# Figure 1: Side-by-side histograms of INSP (weighted) for 2018 and 2023
# --------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=False)

# 2018
ax = axes[0]
insp_2018 = df_2018_owners['INSP'].values
wgtp_2018 = df_2018_owners['WGTP'].values
log_bins = np.logspace(np.log10(max(insp_2018.min(), 1)), np.log10(insp_2018.max()), 50)
ax.hist(insp_2018, bins=log_bins, weights=wgtp_2018, color='steelblue', alpha=0.7,
        edgecolor='white', linewidth=0.5)
med_2018 = weighted_median(insp_2018, wgtp_2018)
ax.axvline(med_2018, color='red', linewidth=2, linestyle='--', label=f'Median: ${med_2018:,.0f}')
ax.set_xscale('log')
ax.set_xlabel('Annual Insurance Premium (INSP, $)', fontsize=11)
ax.set_ylabel('Weighted Count', fontsize=11)
ax.set_title('2018 FL Insurance Premium Distribution', fontsize=13, fontweight='bold')
ax.legend(fontsize=10)
ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: f'${x:,.0f}'))
ax.tick_params(axis='x', rotation=30)

# 2023
ax = axes[1]
insp_2023 = df_2023_owners['INSP'].values
wgtp_2023 = df_2023_owners['WGTP'].values
log_bins_23 = np.logspace(np.log10(max(insp_2023.min(), 1)), np.log10(insp_2023.max()), 50)
ax.hist(insp_2023, bins=log_bins_23, weights=wgtp_2023, color='darkorange', alpha=0.7,
        edgecolor='white', linewidth=0.5)
med_2023 = weighted_median(insp_2023, wgtp_2023)
ax.axvline(med_2023, color='red', linewidth=2, linestyle='--', label=f'Median: ${med_2023:,.0f}')
ax.set_xscale('log')
ax.set_xlabel('Annual Insurance Premium (INSP, $)', fontsize=11)
ax.set_ylabel('Weighted Count', fontsize=11)
ax.set_title('2023 FL Insurance Premium Distribution', fontsize=13, fontweight='bold')
ax.legend(fontsize=10)
ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: f'${x:,.0f}'))
ax.tick_params(axis='x', rotation=30)

fig.suptitle(f'FL Insurance Premium Distributions: 2018 vs 2023\n'
             f'Weighted median rose from \\${med_2018:,.0f} to \\${med_2023:,.0f} '
             f'(+{100*(med_2023-med_2018)/med_2018:.1f}%)',
             fontsize=14, fontweight='bold', y=0.98)
fig.tight_layout()
fig.savefig(f"{FIG_DIR}/temporal_premium_distribution_2018_vs_2023.png",
            dpi=150, bbox_inches='tight')
plt.close(fig)
tee_print(f"  Saved: reports/figures/temporal_premium_distribution_2018_vs_2023.png")
