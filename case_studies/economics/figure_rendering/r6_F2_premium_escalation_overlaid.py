#!/usr/bin/env python3
'Plot Figure 5d: survey-weighted Florida premium distributions with shared log bins.'
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.environ["REPRO_DATA_ROOT"]
FIG_OUT = os.environ["REPRO_OUTPUT_DIR"]
os.makedirs(FIG_OUT, exist_ok=True)

DATA_2018 = os.path.join(BASE, "data/downloads/pums_housing_2018/psam_h12.csv")
DATA_2023 = os.path.join(BASE, "data/downloads/pums_housing_2023/psam_h12.csv")


def weighted_median(values, weights):
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    mask = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    values, weights = values[mask], weights[mask]
    if len(values) == 0:
        return np.nan
    order = np.argsort(values)
    v, w = values[order], weights[order]
    cum = np.cumsum(w)
    idx = np.searchsorted(cum, w.sum() / 2.0)
    return v[min(idx, len(v) - 1)]


d18 = pd.read_csv(DATA_2018, usecols=["ST", "PUMA", "WGTP", "TEN", "INSP", "VALP", "HINCP", "MV"])
d18 = d18[(d18["TEN"].isin([1, 2])) & (d18["INSP"] > 0) & (d18["VALP"] > 0) &
          (d18["HINCP"] > 0) & (d18["WGTP"] > 0)].copy()

d23 = pd.read_csv(DATA_2023, usecols=["STATE", "PUMA", "WGTP", "TEN", "INSP", "VALP", "HINCP", "MV"])
d23 = d23[(d23["TEN"].isin([1, 2])) & (d23["INSP"] > 0) & (d23["VALP"] > 0) &
          (d23["HINCP"] > 0) & (d23["WGTP"] > 0)].copy()

insp_18, wgtp_18 = d18["INSP"].values, d18["WGTP"].values
insp_23, wgtp_23 = d23["INSP"].values, d23["WGTP"].values

# Shared log bins spanning both years so the overlaid histograms are comparable
lo = max(min(insp_18.min(), insp_23.min()), 1)
hi = max(insp_18.max(), insp_23.max())
bins = np.logspace(np.log10(lo), np.log10(hi), 50)

med_18 = weighted_median(insp_18, wgtp_18)
med_23 = weighted_median(insp_23, wgtp_23)

fig, ax = plt.subplots(figsize=(8, 5.5))
ax.hist(insp_18, bins=bins, weights=wgtp_18, color="steelblue", alpha=0.5,
        edgecolor="white", linewidth=0.5, label="2018")
ax.hist(insp_23, bins=bins, weights=wgtp_23, color="darkorange", alpha=0.5,
        edgecolor="white", linewidth=0.5, label="2023")
ax.axvline(med_18, color="steelblue", linewidth=2, linestyle="--",
           label=f"2018 median: ${med_18:,.0f}")
ax.axvline(med_23, color="darkorange", linewidth=2, linestyle="--",
           label=f"2023 median: ${med_23:,.0f}")

ax.set_xscale("log")
ax.set_xlabel("Annual Insurance Premium (INSP, $)", fontsize=14)
ax.set_ylabel("Weighted Count", fontsize=14)
ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: f"${x:,.0f}"))
ax.tick_params(axis="x", rotation=30, labelsize=12)
ax.tick_params(axis="y", labelsize=12)
ax.legend(fontsize=12)
ax.set_title(f"FL Insurance Premium Distribution: 2018 vs 2023\n"
             f"Weighted median rose from \\${med_18:,.0f} to \\${med_23:,.0f} "
             f"(+{100*(med_23-med_18)/med_18:.1f}%)",
             fontsize=15, fontweight="bold")

fig.tight_layout()
out = os.path.join(FIG_OUT, "r6_F2_premium_escalation_2018_2023.pdf")
fig.savefig(out, bbox_inches="tight")
plt.close(fig)
print(f"Saved {out}")
