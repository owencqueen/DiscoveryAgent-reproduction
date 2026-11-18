#!/usr/bin/env python3
'Plot Figure 5c: PUMA insurance-to-income ratios by state.'
import os
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.environ["REPRO_DATA_ROOT"]
FIG_OUT = os.environ["REPRO_OUTPUT_DIR"]
os.makedirs(FIG_OUT, exist_ok=True)

insurance_df = pd.read_csv(os.path.join(BASE, "data/processed/puma_insurance_burden_6states.csv"))

state_order_income = (insurance_df.groupby("state_name")["ins_to_income_ratio"]
                      .mean().sort_values(ascending=False).index.tolist())

fig, ax = plt.subplots(figsize=(6.5, 5.5))
sns.boxplot(data=insurance_df, x="state_name", y="ins_to_income_ratio",
            order=state_order_income, ax=ax, palette="plasma")
ax.set_xlabel("State", fontsize=15)
ax.set_ylabel("Insurance-to-Income Ratio", fontsize=15)
ax.set_title("Within-State Variation:\nInsurance-to-Income Ratio", fontsize=17)
ax.tick_params(axis="x", rotation=45, labelsize=12)
ax.tick_params(axis="y", labelsize=12)

plt.tight_layout()
out = os.path.join(FIG_OUT, "r1_insurance_burden_by_state.pdf")
plt.savefig(out, bbox_inches="tight")
plt.close()
print(f"Saved {out}")
