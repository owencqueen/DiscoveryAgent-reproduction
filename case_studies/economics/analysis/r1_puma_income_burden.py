import os

#!/usr/bin/env python3
'Compute PUMA housing-insurance aggregates for six states from ACS PUMS.'

import os
import pandas as pd
import numpy as np
from pathlib import Path

# ============================================================================
# Configuration
# ============================================================================

BASE_DIR = Path((os.environ["REPRO_DATA_ROOT"] + ''))

# Input file paths
DATA_FILES = {
    6: BASE_DIR / "data/downloads/pums_housing_2023/psam_h06.csv",   # California
    12: BASE_DIR / "data/downloads/pums_housing_2023/psam_h12.csv",  # Florida
    22: BASE_DIR / "data/downloads/pums_housing_2023/psam_h22.csv",  # Louisiana
    27: BASE_DIR / "data/downloads/pums_housing_2023/psam_h27.csv",  # Minnesota
    39: BASE_DIR / "data/downloads/pums_housing_2023/psam_h39.csv",  # Ohio
    48: BASE_DIR / "data/downloads/pums_housing_2023/psam_h48.csv",  # Texas
}

# Output paths
OUTPUT_CSV = BASE_DIR / "data/processed/puma_insurance_burden_6states.csv"
OUTPUT_STATS = BASE_DIR / "reports/analysis_outputs/insurance_burden_descriptive_stats.txt"

# State name mapping (FIPS codes)
STATE_NAMES = {
    6: 'California',
    12: 'Florida',
    22: 'Louisiana',
    27: 'Minnesota',
    39: 'Ohio',
    48: 'Texas'
}

# Key columns to load (for efficiency)
KEY_COLUMNS = ['STATE', 'PUMA', 'WGTP', 'TEN', 'INSP', 'VALP', 'HINCP', 'SMOCP']


# ============================================================================
# Utility Functions
# ============================================================================

def weighted_median(values, weights):
    """
    Compute weighted median of values.

    Parameters:
    -----------
    values : array-like
        Values to compute weighted median for
    weights : array-like
        Weights corresponding to values

    Returns:
    --------
    float : weighted median value
    """
    # Remove NaN values
    mask = ~np.isnan(values) & ~np.isnan(weights)
    values = np.array(values)[mask]
    weights = np.array(weights)[mask]

    if len(values) == 0:
        return np.nan

    # Sort by values
    sorted_indices = np.argsort(values)
    sorted_values = values[sorted_indices]
    sorted_weights = weights[sorted_indices]

    # Compute cumulative weights
    cumsum = np.cumsum(sorted_weights)
    total_weight = cumsum[-1]

    # Find the value where cumulative weight crosses 50%
    threshold = total_weight / 2
    idx = np.searchsorted(cumsum, threshold)

    # If exact 50%, average with next value
    if idx < len(sorted_values) - 1 and cumsum[idx] == threshold:
        return (sorted_values[idx] + sorted_values[idx + 1]) / 2

    return sorted_values[min(idx, len(sorted_values) - 1)]


def weighted_mean(values, weights):
    """
    Compute weighted mean of values.

    Parameters:
    -----------
    values : array-like
        Values to compute weighted mean for
    weights : array-like
        Weights corresponding to values

    Returns:
    --------
    float : weighted mean value
    """
    # Remove NaN values
    mask = ~np.isnan(values) & ~np.isnan(weights)
    values = np.array(values)[mask]
    weights = np.array(weights)[mask]

    if len(values) == 0 or np.sum(weights) == 0:
        return np.nan

    return np.average(values, weights=weights)


# ============================================================================
# Step 1: Load and Inspect Data
# ============================================================================

def load_and_inspect_data():
    """Load all 6 CSV files and concatenate into one DataFrame."""

    print("=" * 80)
    print("STEP 1: LOAD AND INSPECT DATA")
    print("=" * 80)

    dfs = []
    for state_code, filepath in DATA_FILES.items():
        print(f"\nLoading {STATE_NAMES[state_code]} (ST={state_code}): {filepath}")
        df = pd.read_csv(filepath, usecols=KEY_COLUMNS, low_memory=False)
        print(f"  Records loaded: {len(df):,}")
        dfs.append(df)

    # Concatenate all DataFrames
    combined_df = pd.concat(dfs, ignore_index=True)
    print(f"\n{'='*40}")
    print(f"TOTAL RECORDS LOADED: {len(combined_df):,}")
    print(f"{'='*40}")

    # Print column names and basic info
    print("\n--- Column Names and Types ---")
    print(combined_df.dtypes)

    print("\n--- Basic Info ---")
    print(combined_df.info())

    print("\n--- First 10 rows of key columns ---")
    print(combined_df[['STATE', 'PUMA', 'WGTP', 'TEN', 'INSP', 'VALP', 'HINCP', 'SMOCP']].head(10).to_string())

    # Summary statistics for key numeric columns
    print("\n--- Summary Statistics for Key Variables ---")
    print(combined_df[['WGTP', 'INSP', 'VALP', 'HINCP', 'SMOCP']].describe().to_string())

    # Distribution of TEN (tenure) values
    print("\n--- Distribution of TEN (Tenure) Values ---")
    print(combined_df['TEN'].value_counts(dropna=False).sort_index())

    return combined_df


# ============================================================================
# Step 2: Filter Data
# ============================================================================

def filter_data(df):
    """
    Filter to owner-occupied units with insurance.
    TEN in (1, 2): owner-occupied
    INSP > 0: has insurance
    """

    print("\n" + "=" * 80)
    print("STEP 2: FILTER DATA")
    print("=" * 80)

    initial_count = len(df)
    print(f"\nInitial record count: {initial_count:,}")

    # Filter for owner-occupied (TEN in 1, 2)
    df_owners = df[df['TEN'].isin([1, 2])].copy()
    after_tenure_filter = len(df_owners)
    print(f"After filtering for owner-occupied (TEN in 1,2): {after_tenure_filter:,}")
    print(f"  Removed: {initial_count - after_tenure_filter:,} records")

    # Filter for has insurance (INSP > 0)
    df_filtered = df_owners[df_owners['INSP'] > 0].copy()
    final_count = len(df_filtered)
    print(f"After filtering for has insurance (INSP > 0): {final_count:,}")
    print(f"  Removed: {after_tenure_filter - final_count:,} records")

    print(f"\n{'='*40}")
    print(f"FINAL FILTERED RECORDS: {final_count:,}")
    print(f"{'='*40}")

    # Show distribution by state
    print("\n--- Filtered Records by State ---")
    state_counts = df_filtered.groupby('STATE')['WGTP'].agg(['count', 'sum'])
    state_counts.columns = ['n_records', 'weighted_count']
    state_counts['state_name'] = state_counts.index.map(STATE_NAMES)
    print(state_counts.to_string())

    return df_filtered


# ============================================================================
# Step 3: Aggregate to PUMA Level
# ============================================================================

def aggregate_to_puma_level(df):
    """
    Aggregate data to (ST, PUMA) level with weighted statistics.
    """

    print("\n" + "=" * 80)
    print("STEP 3: AGGREGATE TO PUMA LEVEL")
    print("=" * 80)

    # Rename STATE to ST for consistency with the request
    df = df.rename(columns={'STATE': 'ST'})

    # Group by (ST, PUMA)
    groups = df.groupby(['ST', 'PUMA'])

    results = []

    print("\nAggregating data for each PUMA...")
    total_pumas = groups.ngroups
    processed = 0

    for (st, puma), group in groups:
        processed += 1
        if processed % 200 == 0:
            print(f"  Processed {processed}/{total_pumas} PUMAs...")

        # n_owners_insured: sum of WGTP (weighted count)
        n_owners_insured = group['WGTP'].sum()

        # INSP statistics (all records have INSP > 0 after filtering)
        median_insp = weighted_median(group['INSP'].values, group['WGTP'].values)
        mean_insp = weighted_mean(group['INSP'].values, group['WGTP'].values)

        # VALP statistics (only where VALP > 0)
        valp_mask = group['VALP'] > 0
        if valp_mask.any():
            median_valp = weighted_median(group.loc[valp_mask, 'VALP'].values,
                                         group.loc[valp_mask, 'WGTP'].values)
            mean_valp = weighted_mean(group.loc[valp_mask, 'VALP'].values,
                                     group.loc[valp_mask, 'WGTP'].values)
        else:
            median_valp = np.nan
            mean_valp = np.nan

        # HINCP statistics (only where HINCP is not null)
        hincp_mask = group['HINCP'].notna()
        if hincp_mask.any():
            median_hincp = weighted_median(group.loc[hincp_mask, 'HINCP'].values,
                                          group.loc[hincp_mask, 'WGTP'].values)
            mean_hincp = weighted_mean(group.loc[hincp_mask, 'HINCP'].values,
                                      group.loc[hincp_mask, 'WGTP'].values)
        else:
            median_hincp = np.nan
            mean_hincp = np.nan

        # SMOCP statistics (only where SMOCP > 0)
        smocp_mask = group['SMOCP'] > 0
        if smocp_mask.any():
            median_smocp = weighted_median(group.loc[smocp_mask, 'SMOCP'].values,
                                          group.loc[smocp_mask, 'WGTP'].values)
            mean_smocp = weighted_mean(group.loc[smocp_mask, 'SMOCP'].values,
                                      group.loc[smocp_mask, 'WGTP'].values)
        else:
            median_smocp = np.nan
            mean_smocp = np.nan

        results.append({
            'ST': st,
            'PUMA': puma,
            'n_owners_insured': n_owners_insured,
            'median_insp': median_insp,
            'mean_insp': mean_insp,
            'median_valp': median_valp,
            'mean_valp': mean_valp,
            'median_hincp': median_hincp,
            'mean_hincp': mean_hincp,
            'median_smocp': median_smocp,
            'mean_smocp': mean_smocp
        })

    # Create PUMA-level DataFrame
    puma_df = pd.DataFrame(results)

    print(f"\nTotal PUMAs aggregated: {len(puma_df):,}")

    # Compute derived ratios
    print("\nComputing derived ratios...")

    # ins_to_income_ratio: mean_insp / mean_hincp
    puma_df['ins_to_income_ratio'] = puma_df['mean_insp'] / puma_df['mean_hincp']

    # ins_to_value_ratio_per1000: (mean_insp / mean_valp) * 1000
    puma_df['ins_to_value_ratio_per1000'] = (puma_df['mean_insp'] / puma_df['mean_valp']) * 1000

    # ins_share_owner_costs: (mean_insp / 12) / mean_smocp
    puma_df['ins_share_owner_costs'] = (puma_df['mean_insp'] / 12) / puma_df['mean_smocp']

    # Add state_name column
    puma_df['state_name'] = puma_df['ST'].map(STATE_NAMES)

    print("\n--- Sample of aggregated PUMA data ---")
    print(puma_df.head(10).to_string())

    print("\n--- Summary of derived ratios ---")
    print(puma_df[['ins_to_income_ratio', 'ins_to_value_ratio_per1000', 'ins_share_owner_costs']].describe().to_string())

    return puma_df


# ============================================================================
# Step 4: Descriptive Statistics
# ============================================================================

def compute_descriptive_statistics(puma_df):
    """
    Compute and print descriptive statistics.
    Returns a list of strings to write to file.
    """

    print("\n" + "=" * 80)
    print("STEP 4: DESCRIPTIVE STATISTICS")
    print("=" * 80)

    output_lines = []
    output_lines.append("=" * 80)
    output_lines.append("INSURANCE BURDEN DESCRIPTIVE STATISTICS")
    output_lines.append("Analysis of PUMS Housing Data - 6 States")
    output_lines.append("=" * 80)
    output_lines.append("")

    # a) Distribution of ins_to_value_ratio_per1000 across PUMAs within each state
    print("\n--- a) Distribution of ins_to_value_ratio_per1000 by State ---")
    output_lines.append("-" * 80)
    output_lines.append("a) Distribution of ins_to_value_ratio_per1000 by State")
    output_lines.append("   (Insurance cost per $1000 of home value)")
    output_lines.append("-" * 80)

    value_ratio_stats = puma_df.groupby('state_name')['ins_to_value_ratio_per1000'].agg([
        ('min', 'min'),
        ('p25', lambda x: x.quantile(0.25)),
        ('median', 'median'),
        ('p75', lambda x: x.quantile(0.75)),
        ('max', 'max'),
        ('std', 'std')
    ])

    print(value_ratio_stats.to_string())
    output_lines.append(value_ratio_stats.to_string())
    output_lines.append("")

    # b) Distribution of ins_to_income_ratio across PUMAs within each state
    print("\n--- b) Distribution of ins_to_income_ratio by State ---")
    output_lines.append("-" * 80)
    output_lines.append("b) Distribution of ins_to_income_ratio by State")
    output_lines.append("   (Annual insurance cost as fraction of household income)")
    output_lines.append("-" * 80)

    income_ratio_stats = puma_df.groupby('state_name')['ins_to_income_ratio'].agg([
        ('min', 'min'),
        ('p25', lambda x: x.quantile(0.25)),
        ('median', 'median'),
        ('p75', lambda x: x.quantile(0.75)),
        ('max', 'max'),
        ('std', 'std')
    ])

    print(income_ratio_stats.to_string())
    output_lines.append(income_ratio_stats.to_string())
    output_lines.append("")

    # c) Top 10 and bottom 10 PUMAs by ins_to_value_ratio_per1000 within FLORIDA
    print("\n--- c) Florida: Top 10 PUMAs by ins_to_value_ratio_per1000 ---")
    output_lines.append("-" * 80)
    output_lines.append("c) Florida: Top 10 PUMAs by ins_to_value_ratio_per1000")
    output_lines.append("   (Highest insurance cost per $1000 of home value)")
    output_lines.append("-" * 80)

    florida_df = puma_df[puma_df['state_name'] == 'Florida'].copy()
    florida_top10 = florida_df.nlargest(10, 'ins_to_value_ratio_per1000')[
        ['ST', 'PUMA', 'ins_to_value_ratio_per1000', 'mean_insp', 'mean_valp']
    ]

    print(florida_top10.to_string(index=False))
    output_lines.append(florida_top10.to_string(index=False))
    output_lines.append("")

    print("\n--- c) Florida: Bottom 10 PUMAs by ins_to_value_ratio_per1000 ---")
    output_lines.append("-" * 80)
    output_lines.append("c) Florida: Bottom 10 PUMAs by ins_to_value_ratio_per1000")
    output_lines.append("   (Lowest insurance cost per $1000 of home value)")
    output_lines.append("-" * 80)

    florida_bottom10 = florida_df.nsmallest(10, 'ins_to_value_ratio_per1000')[
        ['ST', 'PUMA', 'ins_to_value_ratio_per1000', 'mean_insp', 'mean_valp']
    ]

    print(florida_bottom10.to_string(index=False))
    output_lines.append(florida_bottom10.to_string(index=False))
    output_lines.append("")

    # d) Within-state coefficient of variation for ins_to_value_ratio_per1000
    print("\n--- d) Coefficient of Variation for ins_to_value_ratio_per1000 by State ---")
    output_lines.append("-" * 80)
    output_lines.append("d) Coefficient of Variation for ins_to_value_ratio_per1000 by State")
    output_lines.append("   (CV = std / mean, measures relative variability)")
    output_lines.append("-" * 80)

    cv_stats = puma_df.groupby('state_name')['ins_to_value_ratio_per1000'].agg(['mean', 'std'])
    cv_stats['cv'] = cv_stats['std'] / cv_stats['mean']
    cv_stats = cv_stats.sort_values('cv', ascending=False)

    print(cv_stats.to_string())
    output_lines.append(cv_stats.to_string())
    output_lines.append("")

    output_lines.append("=" * 80)
    output_lines.append("END OF REPORT")
    output_lines.append("=" * 80)

    return output_lines


# ============================================================================
# Step 5: Save Results
# ============================================================================

def save_results(puma_df, output_lines):
    """
    Save the PUMA-level aggregated DataFrame and statistics to files.
    """

    print("\n" + "=" * 80)
    print("STEP 5: SAVE RESULTS")
    print("=" * 80)

    # Ensure output directories exist
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_STATS.parent.mkdir(parents=True, exist_ok=True)

    # Save PUMA-level DataFrame
    print(f"\nSaving PUMA-level data to: {OUTPUT_CSV}")
    puma_df.to_csv(OUTPUT_CSV, index=False)
    print(f"  Shape: {puma_df.shape}")
    print(f"  File size: {OUTPUT_CSV.stat().st_size:,} bytes")

    # Verify saved file
    print(f"\n--- First 10 rows of saved file ---")
    saved_df = pd.read_csv(OUTPUT_CSV)
    print(saved_df.head(10).to_string())

    # Save descriptive statistics
    print(f"\nSaving descriptive statistics to: {OUTPUT_STATS}")
    with open(OUTPUT_STATS, 'w') as f:
        f.write('\n'.join(output_lines))
    print(f"  File size: {OUTPUT_STATS.stat().st_size:,} bytes")

    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)


# ============================================================================
# Main Execution
# ============================================================================

def main():
    """Main execution function."""

    print("\n" + "#" * 80)
    print("# PUMS HOUSING INSURANCE ANALYSIS")
    print("# Analyzing insurance burden across 6 states")
    print("#" * 80 + "\n")

    # Step 1: Load and inspect data
    df = load_and_inspect_data()

    # Step 2: Filter data
    df_filtered = filter_data(df)

    # Step 3: Aggregate to PUMA level
    puma_df = aggregate_to_puma_level(df_filtered)

    # Step 4: Compute descriptive statistics
    output_lines = compute_descriptive_statistics(puma_df)

    # Step 5: Save results
    save_results(puma_df, output_lines)

    return puma_df


if __name__ == "__main__":
    puma_df = main()
