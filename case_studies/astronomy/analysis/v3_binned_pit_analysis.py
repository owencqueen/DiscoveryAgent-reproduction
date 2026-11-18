import os
"""
PIT (Probability Integral Transform) analysis for V3 Binned model.

Evaluates the calibration of the predicted redshift PDFs by computing
the PIT value for each galaxy and checking for uniformity.

PIT_i = CDF(z_true_i) = sum of bin_probs[i, j] for all bins j
        where the bin's upper edge <= z_true_i, plus a fractional
        contribution from the bin containing z_true_i.

Well-calibrated PDFs produce PIT ~ Uniform(0, 1).

Outputs:
  - reports/figures/v3_binned_pit_histogram.png
  - reports/figures/v3_binned_pit_histogram_subgroups.png
  - reports/analysis_outputs/v3_binned_pit_stats.json
  - reports/analysis_outputs/round12_pit_histograms.json
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import json

BASE = (os.environ["REPRO_DATA_ROOT"] + '')
PRED_PATH = f"{BASE}/reports/analysis_outputs/v3_binned_val_predictions.npz"
FIG_DIR = f"{BASE}/reports/figures"
OUT_DIR = f"{BASE}/reports/analysis_outputs"

# ──────────────────────────────────────────────────────────────────────
# 1. Load data
# ──────────────────────────────────────────────────────────────────────
data = np.load(PRED_PATH)
bin_probs = data["bin_probs"]        # (12130, 100) softmax probabilities
bin_centers = data["bin_centers"]    # (100,) bin centers
z_true = data["z_true"]             # (12130,)
mag_r = data["mag_r"]
mag_g = data["mag_g"]
is_dwarf = data["is_dwarf"]
N, K = bin_probs.shape
print(f"Loaded {N} galaxies, {K} bins")

# ──────────────────────────────────────────────────────────────────────
# 2. Reconstruct bin edges from bin centers
#    V3 uses K=100 non-uniform bins: 30 bins in [0, 0.05), 70 in [0.05, 0.6).
#    Edges are midpoints between consecutive centers, with edge[0]=0 and
#    edge[K] extrapolated symmetrically from the last center.
# ──────────────────────────────────────────────────────────────────────
bin_edges = np.zeros(K + 1)
bin_edges[0] = 0.0
for i in range(1, K):
    bin_edges[i] = 0.5 * (bin_centers[i - 1] + bin_centers[i])
bin_edges[K] = 2.0 * bin_centers[K - 1] - bin_edges[K - 1]

print(f"Bin edges: [{bin_edges[0]:.4f}, ..., {bin_edges[-1]:.4f}]")
print(f"z_true range: [{z_true.min():.4f}, {z_true.max():.4f}]")
print(f"Galaxies with z_true > last bin edge: {(z_true > bin_edges[-1]).sum()}")

# ──────────────────────────────────────────────────────────────────────
# 3. Compute PIT values (vectorized)
#    PIT_i = CDF(z_true_i)
#          = sum of bin_probs for fully-covered bins
#          + fractional probability for the bin containing z_true_i
#    Assumes uniform density within each bin for the fractional part.
#    Galaxies beyond the last bin edge get PIT = 1.0.
# ──────────────────────────────────────────────────────────────────────
def compute_pit_vectorized(bin_probs, bin_edges, z_true):
    """Vectorized PIT computation with linear interpolation within bins."""
    N_gal, K_bins = bin_probs.shape
    rows = np.arange(N_gal)

    # Find bin index j such that bin_edges[j] <= z_true < bin_edges[j+1]
    j = np.searchsorted(bin_edges, z_true, side="right") - 1
    j = np.clip(j, 0, K_bins - 1)

    # Cumulative sum along bin axis
    cumsum = np.cumsum(bin_probs, axis=1)  # (N, K)

    # CDF at the left edge of bin j
    cdf_left = np.zeros(N_gal)
    mask_gt0 = j > 0
    cdf_left[mask_gt0] = cumsum[rows[mask_gt0], j[mask_gt0] - 1]

    # Fractional contribution from containing bin
    bin_widths = bin_edges[j + 1] - bin_edges[j]
    frac = np.where(bin_widths > 0, (z_true - bin_edges[j]) / bin_widths, 0.0)
    frac = np.clip(frac, 0.0, 1.0)

    pit = cdf_left + frac * bin_probs[rows, j]

    # Handle galaxies beyond the last bin edge
    pit[z_true >= bin_edges[-1]] = 1.0
    pit[z_true <= bin_edges[0]] = 0.0

    return pit


def compute_pit_loop(bin_probs, bin_edges, z_true):
    """Loop-based PIT for validation."""
    N_gal, K_bins = bin_probs.shape
    pit = np.zeros(N_gal)
    for i in range(N_gal):
        z = z_true[i]
        if z >= bin_edges[-1]:
            pit[i] = 1.0
            continue
        if z <= bin_edges[0]:
            pit[i] = 0.0
            continue
        j = np.searchsorted(bin_edges, z, side="right") - 1
        j = min(j, K_bins - 1)
        if j > 0:
            pit[i] = bin_probs[i, :j].sum()
        bw = bin_edges[j + 1] - bin_edges[j]
        if bw > 0:
            pit[i] += ((z - bin_edges[j]) / bw) * bin_probs[i, j]
    return pit


print("Computing PIT values...")
pit_values = compute_pit_vectorized(bin_probs, bin_edges, z_true)

# Sanity check on a small subset
pit_check = compute_pit_loop(bin_probs[:20], bin_edges, z_true[:20])
assert np.allclose(pit_values[:20], pit_check, atol=1e-6), \
    "Vectorized and loop PIT disagree!"
print("Sanity check passed (vectorized matches loop).")

print(f"PIT: mean={pit_values.mean():.4f}, std={pit_values.std():.4f}, "
      f"min={pit_values.min():.4f}, max={pit_values.max():.4f}")

# ──────────────────────────────────────────────────────────────────────
# 4. Define subgroups and compute statistics
# ──────────────────────────────────────────────────────────────────────
color_gr = mag_g - mag_r

subgroups = {
    "All":       np.ones(N, dtype=bool),
    "Dwarf":     is_dwarf.astype(bool),
    "Non-dwarf": ~is_dwarf.astype(bool),
    "Bin_A":     mag_r < 19.0,
    "Bin_B":     (mag_r >= 19.0) & (mag_r < 20.5),
    "Bin_C":     (mag_r >= 20.5) & (mag_r < 21.2),
    "Low_z":     z_true < 0.03,
    "High_z":    z_true >= 0.03,
    "Blue":      color_gr < 0.3,
    "Red":       color_gr >= 0.3,
}


def pit_stats(pit_vals, label=""):
    """Compute PIT summary statistics and uniformity tests."""
    n = len(pit_vals)
    if n < 10:
        return {"label": label, "N": n, "mean": float("nan"), "std": float("nan"),
                "KS_stat": float("nan"), "KS_pvalue": float("nan"),
                "chi2_stat": float("nan"), "chi2_pvalue": float("nan")}

    ks_s, ks_p = stats.kstest(pit_vals, "uniform", args=(0, 1))

    # Chi-squared test on 20-bin histogram
    n_bins = 20
    observed, _ = np.histogram(pit_vals, bins=n_bins, range=(0, 1))
    expected = n / n_bins
    chi2_s = float(np.sum((observed - expected) ** 2 / expected))
    chi2_p = float(1.0 - stats.chi2.cdf(chi2_s, df=n_bins - 1))

    return {
        "label": label,
        "N": int(n),
        "mean": float(np.mean(pit_vals)),
        "std": float(np.std(pit_vals)),
        "median": float(np.median(pit_vals)),
        "KS_stat": float(ks_s),
        "KS_pvalue": float(ks_p),
        "chi2_stat": chi2_s,
        "chi2_pvalue": chi2_p,
    }


all_stats = {}
print("\nSubgroup PIT statistics:")
print(f"  {'Subgroup':12s}  {'N':>5s}  {'Mean':>6s}  {'Std':>6s}  {'KS':>6s}  {'KS p':>10s}  {'chi2':>7s}  {'chi2 p':>10s}")
print("  " + "-" * 75)
for label, mask in subgroups.items():
    s = pit_stats(pit_values[mask], label=label)
    all_stats[label] = s
    print(f"  {label:12s}  {s['N']:5d}  {s['mean']:.4f}  {s['std']:.4f}  "
          f"{s['KS_stat']:.4f}  {s['KS_pvalue']:10.2e}  {s['chi2_stat']:7.1f}  {s['chi2_pvalue']:10.2e}")

# Also compute Cramer-von Mises for the overall sample
try:
    cvm_result = stats.cramervonmises(pit_values, "uniform", args=(0, 1))
    cvm_stat, cvm_pval = float(cvm_result.statistic), float(cvm_result.pvalue)
    print(f"\n  Cramer-von Mises (All): stat={cvm_stat:.5f}, p={cvm_pval:.3e}")
except Exception as e:
    print(f"\n  CvM test unavailable: {e}")
    cvm_stat, cvm_pval = None, None

# ──────────────────────────────────────────────────────────────────────
# 5. Plot 1: Overall PIT histogram
# ──────────────────────────────────────────────────────────────────────
n_hist_bins = 20
expected_count = N / n_hist_bins

fig, ax = plt.subplots(1, 1, figsize=(7, 5))

counts, edges_h, patches = ax.hist(
    pit_values, bins=n_hist_bins, range=(0, 1),
    color="#4C72B0", edgecolor="white", linewidth=0.8, alpha=0.85,
    label="PIT histogram"
)

ax.axhline(expected_count, color="k", linestyle="--", linewidth=1.5,
           label=f"Uniform expectation ({expected_count:.0f})")

# Poisson +/- 1 sigma band
sigma_poisson = np.sqrt(expected_count)
ax.axhspan(expected_count - sigma_poisson, expected_count + sigma_poisson,
           color="gray", alpha=0.15, label=r"$\pm 1\sigma$ Poisson")

s = all_stats["All"]
stats_text = (
    f"N = {N:,}\n"
    f"Mean PIT = {s['mean']:.4f}  (ideal 0.500)\n"
    f"Std PIT = {s['std']:.4f}  (ideal 0.289)\n"
    f"KS stat = {s['KS_stat']:.4f}\n"
    f"KS p-value = {s['KS_pvalue']:.2e}\n"
    f"$\\chi^2$ = {s['chi2_stat']:.1f} (dof=19), p = {s['chi2_pvalue']:.2e}"
)
ax.text(
    0.97, 0.95, stats_text,
    transform=ax.transAxes, fontsize=9.5,
    verticalalignment="top", horizontalalignment="right",
    bbox=dict(boxstyle="round,pad=0.4", facecolor="wheat", alpha=0.75)
)

ax.set_xlabel("PIT value", fontsize=12)
ax.set_ylabel("Count", fontsize=12)
ax.set_title("V3 Binned Model: PIT Histogram (All Galaxies)", fontsize=13, fontweight="bold")
ax.legend(loc="upper left", fontsize=9.5)
ax.set_xlim(0, 1)
ax.set_ylim(bottom=0)
ax.tick_params(labelsize=10)

fig.tight_layout()
fig.savefig(f"{FIG_DIR}/v3_binned_pit_histogram.png", dpi=200, bbox_inches="tight")
plt.close(fig)
print(f"\nSaved: {FIG_DIR}/v3_binned_pit_histogram.png")

# ──────────────────────────────────────────────────────────────────────
# 6. Plot 2: Subgroup PIT histograms (2x2)
# ──────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(12, 9))
hist_edges = np.linspace(0, 1, n_hist_bins + 1)


def plot_subgroup_panel(ax, group_list, title):
    """Plot overlaid density-normalized step histograms.

    group_list: list of (label, mask, color) tuples
    """
    for label, mask, color in group_list:
        pit_sub = pit_values[mask]
        n_sub = mask.sum()
        if n_sub < 10:
            continue
        ks_s, _ = stats.kstest(pit_sub, "uniform", args=(0, 1))
        ax.hist(
            pit_sub, bins=hist_edges, density=True,
            histtype="step", linewidth=2.0, color=color,
            label=f"{label} (N={n_sub:,}, KS={ks_s:.3f})"
        )

    ax.axhline(1.0, color="k", linestyle="--", linewidth=1.2, alpha=0.6,
               label="Uniform (density=1)")
    ax.set_xlabel("PIT value", fontsize=11)
    ax.set_ylabel("Density", fontsize=11)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.legend(fontsize=8.5, loc="best")
    ax.set_xlim(0, 1)
    ax.set_ylim(bottom=0)
    ax.tick_params(labelsize=9)


# (a) All + Dwarf vs Non-dwarf
plot_subgroup_panel(axes[0, 0], [
    ("All", subgroups["All"], "#888888"),
    ("Dwarf", subgroups["Dwarf"], "#E24A33"),
    ("Non-dwarf", subgroups["Non-dwarf"], "#348ABD"),
], "(a) All / Dwarf / Non-dwarf")

# (b) Magnitude bins
plot_subgroup_panel(axes[0, 1], [
    ("Bin A (r<19)", subgroups["Bin_A"], "#E24A33"),
    ("Bin B (19-20.5)", subgroups["Bin_B"], "#348ABD"),
    ("Bin C (20.5-21.2)", subgroups["Bin_C"], "#988ED5"),
], "(b) Magnitude Bins (r-band)")

# (c) Low-z vs High-z
plot_subgroup_panel(axes[1, 0], [
    ("z < 0.03", subgroups["Low_z"], "#E24A33"),
    ("z >= 0.03", subgroups["High_z"], "#348ABD"),
], r"(c) Low-z ($z < 0.03$) vs High-z ($z \geq 0.03$)")

# (d) Blue vs Red
plot_subgroup_panel(axes[1, 1], [
    ("Blue (g-r < 0.3)", subgroups["Blue"], "#348ABD"),
    ("Red (g-r >= 0.3)", subgroups["Red"], "#E24A33"),
], r"(d) Blue ($g{-}r < 0.3$) vs Red ($g{-}r \geq 0.3$)")

fig.suptitle("V3 Binned Model: PIT Histograms by Subgroup",
             fontsize=14, fontweight="bold", y=1.01)
fig.tight_layout()
fig.savefig(f"{FIG_DIR}/v3_binned_pit_histogram_subgroups.png",
            dpi=200, bbox_inches="tight")
plt.close(fig)
print(f"Saved: {FIG_DIR}/v3_binned_pit_histogram_subgroups.png")

# ──────────────────────────────────────────────────────────────────────
# 7. Save statistics to JSON
# ──────────────────────────────────────────────────────────────────────
ideal_std = round(1.0 / np.sqrt(12), 4)  # 0.2887

output_json = {
    "description": "PIT statistics for V3 Binned model (validation set)",
    "n_galaxies": int(N),
    "n_bins": int(K),
    "bin_edge_range": [float(bin_edges[0]), float(bin_edges[-1])],
    "n_beyond_last_edge": int((z_true > bin_edges[-1]).sum()),
    "overall": {
        "mean": float(pit_values.mean()),
        "std": float(pit_values.std()),
        "median": float(np.median(pit_values)),
        "ideal_mean": 0.5,
        "ideal_std": ideal_std,
        "KS_stat": float(all_stats["All"]["KS_stat"]),
        "KS_pvalue": float(all_stats["All"]["KS_pvalue"]),
        "chi2_stat": float(all_stats["All"]["chi2_stat"]),
        "chi2_pvalue": float(all_stats["All"]["chi2_pvalue"]),
    },
    "per_subgroup": all_stats,
    "histogram_counts_20bins": [int(c) for c in counts],
    "expected_count_per_bin": float(expected_count),
}

if cvm_stat is not None:
    output_json["overall"]["CvM_stat"] = cvm_stat
    output_json["overall"]["CvM_pvalue"] = cvm_pval

out_path = f"{OUT_DIR}/v3_binned_pit_stats.json"
with open(out_path, "w") as f:
    json.dump(output_json, f, indent=2)
print(f"Saved: {out_path}")

# Export the same subgroup histograms used above for the manuscript renderer.
histogram_groups = {}
for label, mask in subgroups.items():
    group_counts, _ = np.histogram(pit_values[mask], bins=hist_edges)
    histogram_groups[label] = {
        "n": int(all_stats[label]["N"]),
        "KS_stat": float(all_stats[label]["KS_stat"]),
        "counts": group_counts.tolist(),
    }

histogram_path = f"{OUT_DIR}/round12_pit_histograms.json"
with open(histogram_path, "w") as f:
    json.dump({"histogram_edges": hist_edges.tolist(),
               "groups": histogram_groups}, f, indent=2)
print(f"Saved: {histogram_path}")

# ──────────────────────────────────────────────────────────────────────
# 8. Interpretive summary
# ──────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("PIT ANALYSIS SUMMARY")
print("=" * 70)

s_all = all_stats["All"]
print(f"\nOverall PIT:")
print(f"  Mean:  {s_all['mean']:.4f}  (ideal: 0.5000, delta: {s_all['mean'] - 0.5:+.4f})")
print(f"  Std:   {s_all['std']:.4f}  (ideal: {ideal_std}, delta: {s_all['std'] - ideal_std:+.4f})")
print(f"  KS:    {s_all['KS_stat']:.4f}  (p = {s_all['KS_pvalue']:.2e})")
print(f"  chi2:  {s_all['chi2_stat']:.1f}  (p = {s_all['chi2_pvalue']:.2e})")

if s_all["mean"] > 0.52:
    print("  --> PIT mean > 0.5: PDFs tend to be biased LOW (under-predict z)")
elif s_all["mean"] < 0.48:
    print("  --> PIT mean < 0.5: PDFs tend to be biased HIGH (over-predict z)")
else:
    print("  --> PIT mean near 0.5: minimal overall bias")

if s_all["std"] > ideal_std + 0.02:
    print("  --> PIT std > ideal: PDFs are OVER-CONFIDENT (too narrow)")
elif s_all["std"] < ideal_std - 0.02:
    print("  --> PIT std < ideal: PDFs are UNDER-CONFIDENT (too wide)")
else:
    print("  --> PIT std near ideal: reasonable width calibration")

# Tail fractions
pit_low = (pit_values < 0.1).mean()
pit_high = (pit_values > 0.9).mean()
print(f"\n  Frac PIT < 0.1: {pit_low:.4f}  (ideal: 0.10)")
print(f"  Frac PIT > 0.9: {pit_high:.4f}  (ideal: 0.10)")
if pit_low > 0.15:
    print("  --> Excess low PIT: model over-predicts z (CDF reaches z_true slowly)")
if pit_high > 0.15:
    print("  --> Excess high PIT: model under-predicts z (CDF reaches z_true quickly)")

print(f"\nSubgroup highlights:")
for label in ["Dwarf", "Non-dwarf", "Low_z", "High_z", "Blue", "Red",
              "Bin_A", "Bin_B", "Bin_C"]:
    s = all_stats[label]
    flags = []
    if s["mean"] < 0.40:
        flags.append("HIGH BIAS (over-predicts z)")
    elif s["mean"] > 0.60:
        flags.append("LOW BIAS (under-predicts z)")
    if s["std"] > 0.35:
        flags.append("OVER-CONFIDENT")
    elif s["std"] < 0.24:
        flags.append("UNDER-CONFIDENT")
    flag_str = " ** " + ", ".join(flags) if flags else ""
    print(f"  {label:12s}: mean={s['mean']:.3f}  std={s['std']:.3f}  "
          f"KS={s['KS_stat']:.3f} (p={s['KS_pvalue']:.2e}){flag_str}")

print("\nDone.")
