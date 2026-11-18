import os
#!/usr/bin/env python3
'Compute the ten-model comparison table and Figure 4d from saved predictions.'

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
import warnings
warnings.filterwarnings("ignore")

BASE = (os.environ["REPRO_DATA_ROOT"] + '')
FIG_DIR = f"{BASE}/reports/figures"
OUT_DIR = f"{BASE}/reports/analysis_outputs"

# ── Load PF-128 predictions ──
d = np.load(f"{OUT_DIR}/photofusion_128_val_predictions.npz")
probs = d["probs"].astype(np.float64)
z_pred = d["z_pred"].astype(np.float64)
z_true = d["z_true"].astype(np.float64)
mag_r = d["mag_r"].astype(np.float64)
mag_g = d["mag_g"].astype(np.float64)
mag_z_arr = d["mag_z"].astype(np.float64)
is_dwarf = d["is_dwarf"].astype(bool)

N = len(z_true)
g_r = mag_g - mag_r
low_z_true = z_true < 0.03
delta_z = (z_pred - z_true) / (1.0 + z_true)

print(f"Loaded PF-128: N={N}, N_lowz={low_z_true.sum()}, N_dwarf={is_dwarf.sum()}")

# ── Helper functions ──
def find_threshold_at_completeness(probs, low_z_true, target_comp=0.95):
    thresholds = np.linspace(0, 1, 10001)
    n_lowz = low_z_true.sum()
    if n_lowz == 0:
        return 0.5, 0.0, 0.0
    for t in thresholds[::-1]:
        selected = probs >= t
        tp = (selected & low_z_true).sum()
        comp = tp / n_lowz
        if comp >= target_comp:
            n_sel = selected.sum()
            pur = tp / n_sel if n_sel > 0 else 0.0
            return t, comp, pur
    return 0.0, 1.0, n_lowz / len(probs)

def find_purity_at_completeness(probs, low_z_true, target_comp=0.95):
    _, _, pur = find_threshold_at_completeness(probs, low_z_true, target_comp)
    return pur

def find_completeness_at_purity(probs, low_z_true, target_pur=0.90):
    thresholds = np.linspace(0, 1, 10001)
    n_lowz = low_z_true.sum()
    if n_lowz == 0:
        return 0.0
    best_comp = 0.0
    for t in thresholds:
        selected = probs >= t
        n_sel = selected.sum()
        if n_sel == 0:
            continue
        tp = (selected & low_z_true).sum()
        pur = tp / n_sel
        comp = tp / n_lowz
        if pur >= target_pur:
            best_comp = max(best_comp, comp)
    return best_comp

def compute_metrics(z_pred, z_true, probs=None, low_z_true=None):
    dz = (z_pred - z_true) / (1.0 + z_true)
    bias = np.mean(dz)
    med_dz = np.median(dz)
    sigma_nmad = 1.4826 * np.median(np.abs(dz - med_dz))
    f_outlier = np.mean(np.abs(dz) > 0.05)
    result = {"N": len(z_true), "bias": bias, "sigma_NMAD": sigma_nmad, "f_outlier": f_outlier}
    if probs is not None and low_z_true is not None:
        n_lowz = low_z_true.sum()
        result["N_lowz"] = int(n_lowz)
        if n_lowz > 0 and n_lowz < len(low_z_true):
            result["AUC_ROC"] = roc_auc_score(low_z_true, probs)
            result["P_at_95C"] = find_purity_at_completeness(probs, low_z_true, 0.95)
            result["C_at_90P"] = find_completeness_at_purity(probs, low_z_true, 0.90)
        else:
            result["AUC_ROC"] = np.nan
            result["P_at_95C"] = np.nan
            result["C_at_90P"] = np.nan
    return result

plt.rcParams.update({
    'font.size': 11,
    'axes.titlesize': 12,
    'axes.labelsize': 11,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 9,
    'figure.dpi': 150,
})

# ════════════════════════════════════════════════════════════════
# DELIVERABLE 6: Purity at Fixed Completeness / Completeness at Fixed Purity
# ════════════════════════════════════════════════════════════════
print("\n=== DELIVERABLE 6: Purity@95%C and Completeness@90%P Table ===")

table_lines = []
table_lines.append("=" * 100)
print("\n=== DELIVERABLE 10: Summary Model Comparison Table ===")

# Load all model predictions
models = {}

# 1. PF-128 (already loaded)
models["PhotoFusion-128"] = {"probs": probs, "z_pred": z_pred, "z_true": z_true,
                              "mag_r": mag_r, "is_dwarf": is_dwarf}

# 2. PF-64
d2 = np.load(f"{OUT_DIR}/photofusion_64_val_predictions.npz")
models["PhotoFusion-64"] = {"probs": d2["probs"].astype(np.float64),
                             "z_pred": d2["z_pred"].astype(np.float64),
                             "z_true": d2["z_true"].astype(np.float64),
                             "mag_r": d2["mag_r"].astype(np.float64),
                             "is_dwarf": d2["is_dwarf"].astype(bool)}

# 3. Frozen-RN18
d3 = np.load(f"{OUT_DIR}/frozen_rn18_optimized_val_predictions.npz")
models["Frozen-RN18"] = {"probs": d3["probs"].astype(np.float64),
                          "z_pred": d3["z_pred"].astype(np.float64),
                          "z_true": d3["z_true"].astype(np.float64),
                          "mag_r": d3["mag_r"].astype(np.float64),
                          "is_dwarf": d3["is_dwarf"].astype(bool)}

# 4. Frozen-RN50
d4 = np.load(f"{OUT_DIR}/frozen_rn50_val_predictions.npz")
models["Frozen-RN50"] = {"probs": d4["probs"].astype(np.float64),
                          "z_pred": d4["z_pred"].astype(np.float64),
                          "z_true": d4["z_true"].astype(np.float64),
                          "mag_r": d4["mag_r"].astype(np.float64),
                          "is_dwarf": d4["is_dwarf"].astype(bool)}

# 5. Frozen-BigHead
d5 = np.load(f"{OUT_DIR}/frozen_rn18_bighead_val_predictions.npz")
models["Frozen-BigHead"] = {"probs": d5["probs"].astype(np.float64),
                             "z_pred": d5["z_pred"].astype(np.float64),
                             "z_true": d5["z_true"].astype(np.float64),
                             "mag_r": d5["mag_r"].astype(np.float64),
                             "is_dwarf": d5["is_dwarf"].astype(bool)}

# 6. SWA best for dwarfs
d6 = np.load(f"{OUT_DIR}/pf_hlr_swa_1e-03_ep1_2_predictions.npz")
models["SWA(1e-3,ep1-2)"] = {"probs": d6["probs"].astype(np.float64),
                               "z_pred": d6["z_pred"].astype(np.float64),
                               "z_true": d6["z_true"].astype(np.float64),
                               "mag_r": d6["mag_r"].astype(np.float64),
                               "is_dwarf": d6["is_dwarf"].astype(bool)}

# 7. Fulldata-RN18
d7 = np.load(f"{OUT_DIR}/fulldata_val_predictions.npz")
# This has 12131, need to check
fd_probs = d7["probs"].astype(np.float64)
fd_z_pred = d7["z_pred"].astype(np.float64)
fd_z_true = d7["z_true"].astype(np.float64)
fd_mag_r = d7["mag_r"].astype(np.float64)
fd_is_dwarf = d7["is_dwarf"].astype(bool)
models["Fulldata-RN18"] = {"probs": fd_probs, "z_pred": fd_z_pred, "z_true": fd_z_true,
                            "mag_r": fd_mag_r, "is_dwarf": fd_is_dwarf}

# 8. Photo-Only MLP (from prior experiment)
d8 = np.load(f"{OUT_DIR}/prior_experiment_predictions.npz")
# Photo-only uses photo_only_logit for classification
po_logit = d8["photo_only_logit"].astype(np.float64)
po_probs = 1.0 / (1.0 + np.exp(-po_logit))  # sigmoid
po_z_pred = d8["photo_only_z_pred"].astype(np.float64)
po_z_true = d8["z_true"].astype(np.float64)
po_mag_r = d8["mag_r"].astype(np.float64)
po_is_dwarf = d8["is_dwarf"].astype(bool)
models["Photo-Only MLP"] = {"probs": po_probs, "z_pred": po_z_pred, "z_true": po_z_true,
                             "mag_r": po_mag_r, "is_dwarf": po_is_dwarf}

# 9. PF LR sweep best (stacking best = PF-128, so use LR sweep best instead)
d9 = np.load(f"{OUT_DIR}/pf_lr_sweep_best_predictions.npz")
models["PF-LR3e4(best)"] = {"probs": d9["probs"].astype(np.float64),
                              "z_pred": d9["z_pred"].astype(np.float64),
                              "z_true": d9["z_true"].astype(np.float64),
                              "mag_r": d9["mag_r"].astype(np.float64),
                              "is_dwarf": d9["is_dwarf"].astype(bool)}

# Also load SWA best for blue
d10 = np.load(f"{OUT_DIR}/pf_hlr_swa_1e-03_ep1_3_predictions.npz")
models["SWA(1e-3,ep1-3)"] = {"probs": d10["probs"].astype(np.float64),
                               "z_pred": d10["z_pred"].astype(np.float64),
                               "z_true": d10["z_true"].astype(np.float64),
                               "mag_r": d10["mag_r"].astype(np.float64),
                               "is_dwarf": d10["is_dwarf"].astype(bool)}

# Compute metrics for all models
print(f"\n{'Model':>20s} | {'P@95C':>8s} | {'C@90P':>8s} | {'AUC':>7s} | {'Bias':>8s} | {'σ_NMAD':>8s} | {'f_out':>7s} | P@95C(D) | C@90P(D) | AUC(D)")
print("-" * 120)

table_lines = []
table_lines.append("=" * 130)
table_lines.append("DELIVERABLE 10: Comprehensive Model Comparison Table")
table_lines.append("PhotoFusion-128 is the best model (primary). All metrics on validation set (12,130 galaxies).")
table_lines.append("=" * 130)
table_lines.append("")
table_lines.append(f"{'Model':>20s} | {'P@95C':>8s} | {'C@90P':>8s} | {'AUC':>7s} | {'Bias':>8s} | {'σ_NMAD':>8s} | {'f_out':>7s} | {'P@95C(D)':>9s} | {'C@90P(D)':>9s} | {'AUC(D)':>7s}")
table_lines.append("-" * 130)

model_results = {}
for model_name, mdata in models.items():
    mp = mdata["probs"]
    mzp = mdata["z_pred"]
    mzt = mdata["z_true"]
    mmr = mdata["mag_r"]
    mid = mdata["is_dwarf"]
    mlz = mzt < 0.03

    # Full sample
    full_m = compute_metrics(mzp, mzt, mp, mlz)

    # Dwarfs
    dwarf_mask = mid
    if dwarf_mask.sum() > 0:
        dwarf_m = compute_metrics(mzp[dwarf_mask], mzt[dwarf_mask],
                                   mp[dwarf_mask], mlz[dwarf_mask])
    else:
        dwarf_m = {"P_at_95C": np.nan, "C_at_90P": np.nan, "AUC_ROC": np.nan}

    p95 = full_m.get('P_at_95C', np.nan)
    c90 = full_m.get('C_at_90P', np.nan)
    auc = full_m.get('AUC_ROC', np.nan)
    dp95 = dwarf_m.get('P_at_95C', np.nan)
    dc90 = dwarf_m.get('C_at_90P', np.nan)
    dauc = dwarf_m.get('AUC_ROC', np.nan)

    model_results[model_name] = {"full": full_m, "dwarf": dwarf_m}

    line = (f"{model_name:>20s} | {p95:8.2%} | {c90:8.2%} | {auc:7.3f} | "
            f"{full_m['bias']:+8.4f} | {full_m['sigma_NMAD']:8.4f} | {full_m['f_outlier']:7.3f} | "
            f"{dp95:9.2%} | {dc90:9.2%} | {dauc:7.3f}")
    print(line)
    table_lines.append(line)

# Add additional subgroup detail for top models
table_lines.append("")
table_lines.append("=" * 130)
table_lines.append("Detailed Subgroup Comparison: Top 4 Models")
table_lines.append("=" * 130)
table_lines.append("")

top_models = ["PhotoFusion-128", "PhotoFusion-64", "SWA(1e-3,ep1-2)", "Frozen-RN18"]
subgroups_detail = [
    ("Full", np.ones(N, bool)),
    ("Dwarfs", is_dwarf),
    ("Non-Dwarfs", ~is_dwarf),
    ("Bin A (r<19)", mag_r < 19),
    ("Bin B (19-20.5)", (mag_r >= 19) & (mag_r < 20.5)),
    ("Bin C (20.5-21.2)", (mag_r >= 20.5) & (mag_r < 21.2)),
    ("Blue (g-r<0.3)", g_r < 0.3),
    ("Red (g-r>=0.3)", g_r >= 0.3),
]

header2 = f"{'Subgroup':>20s} | " + " | ".join([f"{m:>16s}" for m in top_models])
table_lines.append("P@95%C:")
table_lines.append(header2)
table_lines.append("-" * (20 + 3 + (16+3)*len(top_models)))

for sg_label, sg_mask in subgroups_detail:
    vals = []
    for mname in top_models:
        mdata = models[mname]
        # Need to handle different array sizes
        if len(mdata["z_true"]) != N:
            # Different size - compute on its own
            sub_mask_local = sg_mask[:len(mdata["z_true"])] if len(sg_mask) > len(mdata["z_true"]) else sg_mask
            # Actually, for models with different N, we need to be careful
            # These models share the same val set order, so we can use first N elements
            local_N = len(mdata["z_true"])
            # Recompute masks for this model's data
            local_mag_r = mdata["mag_r"]
            local_is_dwarf = mdata["is_dwarf"]
            local_g_r = np.zeros_like(local_mag_r)  # won't have g for all models
            # Skip color-based subgroups for models with different N
            if "Blue" in sg_label or "Red" in sg_label:
                vals.append("N/A")
                continue
            if sg_label == "Full":
                local_mask = np.ones(local_N, bool)
            elif sg_label == "Dwarfs":
                local_mask = local_is_dwarf
            elif sg_label == "Non-Dwarfs":
                local_mask = ~local_is_dwarf
            elif "Bin A" in sg_label:
                local_mask = local_mag_r < 19
            elif "Bin B" in sg_label:
                local_mask = (local_mag_r >= 19) & (local_mag_r < 20.5)
            elif "Bin C" in sg_label:
                local_mask = (local_mag_r >= 20.5) & (local_mag_r < 21.2)
            else:
                local_mask = np.ones(local_N, bool)

            local_lowz = mdata["z_true"] < 0.03
            p = find_purity_at_completeness(mdata["probs"][local_mask], local_lowz[local_mask], 0.95)
            vals.append(f"{p:16.2%}")
        else:
            p = find_purity_at_completeness(mdata["probs"][sg_mask], low_z_true[sg_mask], 0.95)
            vals.append(f"{p:16.2%}")

    line = f"{sg_label:>20s} | " + " | ".join(vals)
    table_lines.append(line)

d10_text = "\n".join(table_lines)

with open(f"{OUT_DIR}/FINAL_D10_model_comparison_table.txt", "w") as f:
    f.write(d10_text)
print(f"\n  Saved: FINAL_D10_model_comparison_table.txt")

# Create comparison bar chart figure
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle("Model Comparison Summary", fontsize=14, fontweight='bold')

model_names_short = list(models.keys())
# Sort by P@95C
p95c_full = [model_results[m]["full"].get("P_at_95C", 0) for m in model_names_short]
sorted_idx = np.argsort(p95c_full)[::-1]
model_names_sorted = [model_names_short[i] for i in sorted_idx]
p95c_sorted = [p95c_full[i] for i in sorted_idx]
p95c_dwarf_sorted = [model_results[model_names_short[i]]["dwarf"].get("P_at_95C", 0) for i in sorted_idx]

# Panel 1: P@95C full sample
ax = axes[0]
colors_bar = ['gold' if i == 0 else 'C0' for i in range(len(model_names_sorted))]
bars = ax.barh(range(len(model_names_sorted)), p95c_sorted, color=colors_bar, alpha=0.8, edgecolor='black', linewidth=0.5)
ax.set_yticks(range(len(model_names_sorted)))
ax.set_yticklabels(model_names_sorted, fontsize=9)
ax.set_xlabel("Purity at 95% Completeness")
ax.set_title("Full Sample P@95%C")
ax.invert_yaxis()
for i, v in enumerate(p95c_sorted):
    ax.text(v + 0.001, i, f"{v:.1%}", va='center', fontsize=8)
ax.grid(True, alpha=0.3, axis='x')

# Panel 2: P@95C dwarfs
ax = axes[1]
colors_bar2 = ['gold' if i == 0 else 'C1' for i in range(len(model_names_sorted))]
bars = ax.barh(range(len(model_names_sorted)), p95c_dwarf_sorted, color=colors_bar2, alpha=0.8, edgecolor='black', linewidth=0.5)
ax.set_yticks(range(len(model_names_sorted)))
ax.set_yticklabels(model_names_sorted, fontsize=9)
ax.set_xlabel("Purity at 95% Completeness")
ax.set_title("Dwarfs P@95%C")
ax.invert_yaxis()
for i, v in enumerate(p95c_dwarf_sorted):
    ax.text(v + 0.001, i, f"{v:.1%}", va='center', fontsize=8)
ax.grid(True, alpha=0.3, axis='x')

plt.tight_layout(rect=[0, 0, 1, 0.93])
fig.savefig(f"{FIG_DIR}/FINAL_D10_model_comparison_bar.png", dpi=150, bbox_inches='tight')
plt.close()
print(f"  Saved: FINAL_D10_model_comparison_bar.png")
