#!/usr/bin/env python3
'Plot Figure 4d: full-sample and dwarf P@95C for ten models.'
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
import warnings
warnings.filterwarnings("ignore")

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(os.environ["REPRO_DATA_ROOT"], "reports", "analysis_outputs")
PDF_DIR = os.environ.get("REPRO_OUTPUT_DIR", os.path.join(HERE, "..", "outputs"))
os.makedirs(PDF_DIR, exist_ok=True)


# ── Helper functions (verbatim from final_deliverables_part2.py) ──
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
    result = {"N": len(z_true), "bias": bias, "sigma_NMAD": sigma_nmad,
              "f_outlier": f_outlier}
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


def load(fname, prob_key="probs"):
    d = np.load(os.path.join(OUT_DIR, fname))
    if prob_key == "photo_only_logit":
        p = 1.0 / (1.0 + np.exp(-d["photo_only_logit"].astype(np.float64)))
        zp = d["photo_only_z_pred"].astype(np.float64)
    else:
        p = d[prob_key].astype(np.float64)
        zp = d["z_pred"].astype(np.float64)
    return {"probs": p, "z_pred": zp,
            "z_true": d["z_true"].astype(np.float64),
            "mag_r": d["mag_r"].astype(np.float64),
            "is_dwarf": d["is_dwarf"].astype(bool)}


# ── Load all 10 models (same set/order as the original D10) ──
models = {
    "PhotoFusion-128":  load("photofusion_128_val_predictions.npz"),
    "PhotoFusion-64":   load("photofusion_64_val_predictions.npz"),
    "Frozen-RN18":      load("frozen_rn18_optimized_val_predictions.npz"),
    "Frozen-RN50":      load("frozen_rn50_val_predictions.npz"),
    "Frozen-BigHead":   load("frozen_rn18_bighead_val_predictions.npz"),
    "SWA(1e-3,ep1-2)":  load("pf_hlr_swa_1e-03_ep1_2_predictions.npz"),
    "Fulldata-RN18":    load("fulldata_val_predictions.npz"),
    "Photo-Only MLP":   load("prior_experiment_predictions.npz", "photo_only_logit"),
    "PF-LR3e4(best)":   load("pf_lr_sweep_best_predictions.npz"),
    "SWA(1e-3,ep1-3)":  load("pf_hlr_swa_1e-03_ep1_3_predictions.npz"),
}

model_results = {}
for name, m in models.items():
    lz = m["z_true"] < 0.03
    full_m = compute_metrics(m["z_pred"], m["z_true"], m["probs"], lz)
    dm = m["is_dwarf"]
    dwarf_m = compute_metrics(m["z_pred"][dm], m["z_true"][dm],
                              m["probs"][dm], lz[dm]) if dm.sum() > 0 else \
        {"P_at_95C": np.nan}
    model_results[name] = {"full": full_m, "dwarf": dwarf_m}
    print(f"{name:>20s}  P@95C={full_m['P_at_95C']:.2%}  "
          f"dwarf={dwarf_m.get('P_at_95C', np.nan):.2%}")

# ── Bar chart (verbatim plotting logic from the original) ──
plt.rcParams.update({
    'font.size': 11, 'axes.titlesize': 12, 'axes.labelsize': 11,
    'xtick.labelsize': 10, 'ytick.labelsize': 10, 'legend.fontsize': 9,
    'figure.dpi': 150,
})

fig, ax = plt.subplots(figsize=(2.8, 4.5), layout="constrained")

model_names_short = list(models.keys())
p95c_full = [model_results[m]["full"].get("P_at_95C", 0) for m in model_names_short]
sorted_idx = np.argsort(p95c_full)[::-1]
model_names_sorted = [model_names_short[i] for i in sorted_idx]
p95c_sorted = [p95c_full[i] for i in sorted_idx]

colors_bar = ['gold' if i == 0 else 'C0' for i in range(len(model_names_sorted))]
ax.barh(range(len(model_names_sorted)), p95c_sorted, color=colors_bar,
        alpha=0.8, edgecolor='black', linewidth=0.5)
ax.set_yticks(range(len(model_names_sorted)))
ax.set_yticklabels(model_names_sorted, fontsize=9)
ax.set_xlabel("")
fig.supxlabel("Purity at 95% Completeness", fontsize=10)
ax.set_title("Full Sample P@95%C")
ax.invert_yaxis()
# headroom so the value labels sit inside the axes, not over the spine
ax.set_xlim(0, max(p95c_sorted) * 1.18)
for i, v in enumerate(p95c_sorted):
    ax.text(v + max(p95c_sorted) * 0.012, i, f"{v:.1%}", va='center', fontsize=8)
ax.grid(True, alpha=0.3, axis='x')
ax.spines[['top', 'right']].set_visible(False)

out = os.path.join(PDF_DIR, "FINAL_D10_model_comparison_bar.pdf")
fig.savefig(out, bbox_inches='tight', pad_inches=0.05)
plt.close()
print(f"\nSaved: {out}")
