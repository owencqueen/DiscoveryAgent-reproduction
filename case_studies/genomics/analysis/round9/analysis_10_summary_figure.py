import os
"""
analysis_10_summary_figure.py
=============================
Creates a publication-quality 2x2 multi-panel summary figure linking
selective constraint (s_het) to network topology and gene-group membership
in the Stim8hr CD4 perturbation dataset.

Panels:
  A) Scatter: s_het vs log1p(out_degree), relay genes highlighted
  B) Box plot: s_het by topological category
  C) Box plot: s_het by gene group
  D) Forest / horizontal bar of odds ratios from logistic regression Model 4:
       high_s_het ~ log1p(in_degree) + is_relay + log(baseMean)
"""

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

# ── Paths ────────────────────────────────────────────────────────────────
BASE = (os.environ["REPRO_DATA_ROOT"] + '')
INPUT_CSV = f"{BASE}/reports/analysis_outputs/stim8hr_constraint_annotated.csv"
OUTPUT_FIG = f"{BASE}/reports/figures/summary_constraint_topology.png"
OUTPUT_TXT = f"{BASE}/reports/analysis_outputs/summary_figure_stats.txt"

# ── 1. Load data ─────────────────────────────────────────────────────────
df = pd.read_csv(INPUT_CSV)
print(f"Loaded {len(df)} rows from annotated Stim8hr dataset.")

# ── 2. Derived columns ──────────────────────────────────────────────────
# Simplified topology label
def topo_label(cat):
    cat_lower = str(cat).lower()
    if "relay" in cat_lower:
        return "Relay"
    if "terminal" in cat_lower:
        return "Terminal"
    if "upstream" in cat_lower:
        return "Upstream"
    return "Other"

df["topo_simple"] = df["topology_category"].apply(topo_label)
df["is_relay"] = (df["topo_simple"] == "Relay").astype(int)

# Rename gene_group values for consistent display
group_map = {
    "PID-only": "PID only",
    "PID∩GWAS": "PID \u2229 GWAS",
    "GWAS-only": "GWAS only",
    "Background": "Background",
}
df["gene_group_label"] = df["gene_group"].map(group_map).fillna(df["gene_group"])

# ── 3. Logistic regression (Model 4) ────────────────────────────────────
# Filter to rows with valid target_baseMean > 0 and non-NaN s_het
reg_df = df.dropna(subset=["target_baseMean", "s_het"]).copy()
reg_df = reg_df[reg_df["target_baseMean"] > 0].copy()

# Define high_s_het (80th percentile of the FULL dataset, not just the subset)
s_het_80 = df["s_het"].quantile(0.80)
reg_df["high_s_het_bool"] = (reg_df["s_het"] > s_het_80).astype(int)

# Predictors
reg_df["log1p_in_degree"] = np.log1p(reg_df["in_degree"])
reg_df["log_baseMean"] = np.log(reg_df["target_baseMean"])

X = reg_df[["log1p_in_degree", "is_relay", "log_baseMean"]].copy()
X = sm.add_constant(X)
y = reg_df["high_s_het_bool"]

model = sm.GLM(y, X, family=sm.families.Binomial())
result = model.fit()
print(result.summary())

# Odds ratios and 95% CIs
or_vals = np.exp(result.params)
ci_raw = result.conf_int()
or_lower = np.exp(ci_raw[0])
or_upper = np.exp(ci_raw[1])

predictors = result.params.index.tolist()
predictor_labels = {
    "const": "Intercept",
    "log1p_in_degree": "log1p(in_degree)",
    "is_relay": "Is Relay",
    "log_baseMean": "log(baseMean)",
}

# ── 4. Summary statistics for text output ────────────────────────────────
stats_lines = []
stats_lines.append("Summary Figure Statistics")
stats_lines.append("=" * 60)

# Topology category medians
stats_lines.append("\n--- s_het medians by topology category ---")
for cat in ["Relay", "Terminal", "Upstream", "Other"]:
    sub = df[df["topo_simple"] == cat]["s_het"]
    stats_lines.append(f"  {cat:12s}: median={sub.median():.6f}  n={len(sub)}")

# Gene group medians
stats_lines.append("\n--- s_het medians by gene group ---")
for grp in ["PID only", "PID \u2229 GWAS", "GWAS only", "Background"]:
    sub = df[df["gene_group_label"] == grp]["s_het"]
    stats_lines.append(f"  {grp:16s}: median={sub.median():.6f}  n={len(sub)}")

# Mann-Whitney U: relay vs non-relay
relay_s = df[df["is_relay"] == 1]["s_het"]
nonrelay_s = df[df["is_relay"] == 0]["s_het"]
u_stat, u_p = stats.mannwhitneyu(relay_s, nonrelay_s, alternative="greater")
stats_lines.append(f"\nMann-Whitney U (relay > non-relay): U={u_stat:.0f}, p={u_p:.2e}")

# Logistic regression
stats_lines.append(f"\n--- Logistic Regression Model 4 ---")
stats_lines.append(f"  s_het 80th percentile threshold: {s_het_80:.6f}")
stats_lines.append(f"  N observations: {result.nobs:.0f}")
stats_lines.append(f"  Pseudo R-squared (McFadden): "
                    f"{1 - result.llf / result.llnull:.4f}")
for p in predictors:
    stats_lines.append(
        f"  {predictor_labels.get(p, p):22s}: "
        f"OR={or_vals[p]:.3f}  "
        f"95%CI=[{or_lower[p]:.3f}, {or_upper[p]:.3f}]  "
        f"p={result.pvalues[p]:.2e}"
    )

stats_text = "\n".join(stats_lines)
print(stats_text)
with open(OUTPUT_TXT, "w") as f:
    f.write(stats_text + "\n")
print(f"\nStats saved to {OUTPUT_TXT}")

# ── 5. Create figure ─────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(14, 12))
fig.patch.set_facecolor("white")

# Colour palettes
TOPO_COLORS = {
    "Relay":    "#E05555",
    "Terminal": "#5B8BD6",
    "Upstream": "#6BBF6B",
    "Other":    "#BBBBBB",
}
GROUP_COLORS = {
    "PID only":       "#D45B90",
    "PID \u2229 GWAS": "#8B5CF6",
    "GWAS only":      "#F59E0B",
    "Background":     "#9CA3AF",
}

# ═══════════════════════════════════════════════════════════════════════════
# Panel A – Scatter: s_het vs log1p(out_degree)
# ═══════════════════════════════════════════════════════════════════════════
ax = axes[0, 0]

# Subset to rows with valid out_degree (non-NaN)
plot_df = df.dropna(subset=["out_degree", "s_het"]).copy()
plot_df["log1p_out"] = np.log1p(plot_df["out_degree"])

non_relay = plot_df[plot_df["is_relay"] == 0]
relay = plot_df[plot_df["is_relay"] == 1]

ax.scatter(non_relay["log1p_out"], non_relay["s_het"],
           c="#A0C4E8", alpha=0.25, s=8, edgecolors="none",
           label=f"Non-relay (n={len(non_relay)})", rasterized=True)
ax.scatter(relay["log1p_out"], relay["s_het"],
           c="#E05555", alpha=0.65, s=18, edgecolors="none",
           label=f"Relay (n={len(relay)})", zorder=3)

# Label top relay genes by s_het, stagger offsets to reduce overlap
top_relays = relay.nlargest(6, "s_het")
offsets = [(8, 8), (-45, 12), (8, -14), (-50, -10), (8, 10), (-45, -14)]
for idx, (_, row) in enumerate(top_relays.iterrows()):
    ofs = offsets[idx % len(offsets)]
    ax.annotate(
        row["gene_name"],
        xy=(row["log1p_out"], row["s_het"]),
        xytext=ofs, textcoords="offset points",
        fontsize=8, fontstyle="italic", color="#B03030",
        arrowprops=dict(arrowstyle="-", color="#B03030", lw=0.5),
    )

ax.set_yscale("log")
ax.set_xlabel("log1p(out_degree)", fontsize=12)
ax.set_ylabel("$s_{het}$ (log scale)", fontsize=12)
ax.set_title("A) $s_{het}$ vs Out-degree (Stim8hr)", fontsize=14, fontweight="bold")
ax.legend(fontsize=9, loc="lower right", framealpha=0.9)
ax.tick_params(labelsize=10)

# ═══════════════════════════════════════════════════════════════════════════
# Panel B – Box plots by topological category
# ═══════════════════════════════════════════════════════════════════════════
ax = axes[0, 1]

topo_order = ["Relay", "Terminal", "Upstream", "Other"]
topo_data = [df.loc[df["topo_simple"] == cat, "s_het"].dropna() for cat in topo_order]

bp = ax.boxplot(
    topo_data, positions=range(len(topo_order)), widths=0.5,
    patch_artist=True, showfliers=False, zorder=2,
    medianprops=dict(color="black", linewidth=1.5),
)
for patch, cat in zip(bp["boxes"], topo_order):
    patch.set_facecolor(TOPO_COLORS[cat])
    patch.set_alpha(0.7)

# Jittered points
rng = np.random.default_rng(42)
for i, (cat, vals) in enumerate(zip(topo_order, topo_data)):
    jitter = rng.normal(0, 0.08, size=len(vals))
    ax.scatter(
        i + jitter, vals, c=TOPO_COLORS[cat],
        s=6, alpha=0.15, edgecolors="none", rasterized=True, zorder=1,
    )

ax.set_yscale("log")
ax.set_xticks(range(len(topo_order)))
ax.set_xticklabels(topo_order, fontsize=10)
ax.set_ylabel("$s_{het}$ (log scale)", fontsize=12)
ax.set_title("B) $s_{het}$ by Topological Category", fontsize=14, fontweight="bold")
ax.tick_params(labelsize=10)
ax.yaxis.grid(True, alpha=0.3, linestyle="--")
ax.set_axisbelow(True)

# Significance annotation: relay vs other using Mann-Whitney U
u_ro, p_ro = stats.mannwhitneyu(topo_data[0], topo_data[3], alternative="greater")
sig_label = "***" if p_ro < 0.001 else ("**" if p_ro < 0.01 else ("*" if p_ro < 0.05 else "n.s."))
y_max = max(topo_data[0].max(), topo_data[3].max()) * 1.3
ax.plot([0, 0, 3, 3], [y_max, y_max * 1.05, y_max * 1.05, y_max],
        lw=1.0, color="black")
ax.text(1.5, y_max * 1.08, f"{sig_label} (p={p_ro:.1e})", ha="center",
        fontsize=9, color="black")

# Annotate medians and sample sizes below x-axis labels
for i, (cat, vals) in enumerate(zip(topo_order, topo_data)):
    med = vals.median()
    n = len(vals)
    ax.annotate(
        f"n={n}\nmed={med:.4f}",
        xy=(i, 0), xycoords=("data", "axes fraction"),
        xytext=(0, -22), textcoords="offset points",
        ha="center", va="top", fontsize=7.5,
        color="#444444", fontstyle="italic",
        annotation_clip=False,
    )

# ═══════════════════════════════════════════════════════════════════════════
# Panel C – Box plots by gene group
# ═══════════════════════════════════════════════════════════════════════════
ax = axes[1, 0]

group_order = ["PID only", "PID \u2229 GWAS", "GWAS only", "Background"]
group_data = [df.loc[df["gene_group_label"] == g, "s_het"].dropna() for g in group_order]

bp2 = ax.boxplot(
    group_data, positions=range(len(group_order)), widths=0.5,
    patch_artist=True, showfliers=False, zorder=2,
    medianprops=dict(color="black", linewidth=1.5),
)
for patch, grp in zip(bp2["boxes"], group_order):
    patch.set_facecolor(GROUP_COLORS[grp])
    patch.set_alpha(0.7)

for i, (grp, vals) in enumerate(zip(group_order, group_data)):
    jitter = rng.normal(0, 0.08, size=len(vals))
    ax.scatter(
        i + jitter, vals, c=GROUP_COLORS[grp],
        s=6, alpha=0.15, edgecolors="none", rasterized=True, zorder=1,
    )

ax.set_yscale("log")
ax.set_xticks(range(len(group_order)))
# Use shorter labels for display
display_labels = ["PID only", "PID\u2229GWAS", "GWAS only", "Background"]
ax.set_xticklabels(display_labels, fontsize=10)
ax.set_ylabel("$s_{het}$ (log scale)", fontsize=12)
ax.set_title("C) $s_{het}$ by Gene Group", fontsize=14, fontweight="bold")
ax.tick_params(labelsize=10)
ax.yaxis.grid(True, alpha=0.3, linestyle="--")
ax.set_axisbelow(True)

# Annotate medians and sample sizes below x-axis labels
for i, (grp, vals) in enumerate(zip(group_order, group_data)):
    med = vals.median()
    n = len(vals)
    ax.annotate(
        f"n={n}\nmed={med:.4f}",
        xy=(i, 0), xycoords=("data", "axes fraction"),
        xytext=(0, -22), textcoords="offset points",
        ha="center", va="top", fontsize=7.5,
        color="#444444", fontstyle="italic",
        annotation_clip=False,
    )

# ═══════════════════════════════════════════════════════════════════════════
# Panel D – Forest plot of logistic regression odds ratios
# ═══════════════════════════════════════════════════════════════════════════
ax = axes[1, 1]

# Exclude intercept from the forest plot
pred_no_const = [p for p in predictors if p != "const"]
y_pos = np.arange(len(pred_no_const))
ors = [or_vals[p] for p in pred_no_const]
ci_lo = [or_lower[p] for p in pred_no_const]
ci_hi = [or_upper[p] for p in pred_no_const]
pvals = [result.pvalues[p] for p in pred_no_const]
labels = [predictor_labels.get(p, p) for p in pred_no_const]

# Error bars: distance from OR to lower/upper
xerr_lo = [ors[i] - ci_lo[i] for i in range(len(ors))]
xerr_hi = [ci_hi[i] - ors[i] for i in range(len(ors))]

colors_d = ["#5B8BD6", "#E05555", "#6BBF6B"]
ax.axvline(x=1.0, color="gray", linestyle="--", linewidth=1, zorder=1)

ax.barh(
    y_pos, ors, xerr=[xerr_lo, xerr_hi],
    height=0.45, color=colors_d[:len(pred_no_const)],
    edgecolor="black", linewidth=0.6, alpha=0.8,
    error_kw=dict(ecolor="black", lw=1.5, capsize=4, capthick=1.2),
    zorder=2,
)

# Add OR and p-value text
for i, (or_v, p_v) in enumerate(zip(ors, pvals)):
    sig = "***" if p_v < 0.001 else ("**" if p_v < 0.01 else ("*" if p_v < 0.05 else ""))
    ax.text(
        max(ci_hi[i], or_v) + 0.05, i,
        f"OR={or_v:.2f} {sig}\n(p={p_v:.2e})",
        va="center", fontsize=9, color="black",
    )

ax.set_yticks(y_pos)
ax.set_yticklabels(labels, fontsize=11)
ax.set_xlabel("Odds Ratio (exp(coef))", fontsize=12)
ax.set_title("D) Logistic Regression Odds Ratios (Model 4)", fontsize=14, fontweight="bold")
ax.tick_params(labelsize=10)
ax.xaxis.grid(True, alpha=0.3, linestyle="--")
ax.set_axisbelow(True)

# Ensure x-axis includes OR=1 line comfortably
x_max = max(ci_hi) * 1.5
ax.set_xlim(0, x_max)

# ═══════════════════════════════════════════════════════════════════════════
# Finalise and save
# ═══════════════════════════════════════════════════════════════════════════
plt.tight_layout(pad=2.5, h_pad=4.0, w_pad=3.0)
fig.savefig(OUTPUT_FIG, dpi=300, bbox_inches="tight", facecolor="white")
plt.close(fig)
print(f"\nFigure saved to {OUTPUT_FIG}")
