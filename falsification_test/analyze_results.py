"""Per-bucket breakdown + scatter plot + outlier inspection on results.csv."""
import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, os.path.join(ROOT, "api"))
from services.prediction import hcpcs_code_to_bucket, HCPCS_BUCKET_NAMES  # noqa: E402

df = pd.read_csv(os.path.join(ROOT, "falsification_test", "results.csv"))
df["bucket"] = df["hcpcs"].apply(hcpcs_code_to_bucket)
df["bucket_name"] = df["bucket"].map(HCPCS_BUCKET_NAMES)
df["err_mod_emp"] = df["model_pred"] - df["empirical"]
df["err_mod_off"] = df["model_pred"] - df["official_pfs"]
df["err_emp_off"] = df["empirical"] - df["official_pfs"]
df["pct_emp_of_off"] = df["empirical"] / df["official_pfs"]

print(f"\nN = {len(df)}")
print("\n=== PER-BUCKET BREAKDOWN ===\n")
print(f"{'bucket':<18} {'n':>3} {'MAE(mod,emp)':>12} {'MAE(mod,off)':>12} {'MAE(emp,off)':>12} {'mean(emp/off)':>13}")
for b, sub in df.groupby("bucket_name"):
    print(f"{b:<18} {len(sub):>3} "
          f"{sub['err_mod_emp'].abs().mean():>12.2f} "
          f"{sub['err_mod_off'].abs().mean():>12.2f} "
          f"{sub['err_emp_off'].abs().mean():>12.2f} "
          f"{sub['pct_emp_of_off'].mean():>13.3f}")

# Highlight the rows where empirical << official by > 50% (TC/PC suspects)
print("\n=== Rows where empirical < 60% of official (likely TC/PC split or modifier discount) ===\n")
suspects = df[df["pct_emp_of_off"] < 0.6].sort_values("pct_emp_of_off")
print(suspects[["year", "state", "hcpcs", "specialty", "pos",
                "empirical", "official_pfs", "model_pred", "pct_emp_of_off"]].to_string(index=False))

print(f"\n  -> {len(suspects)}/{len(df)} rows show >40% gap. "
      f"In those rows, model tracks empirical: MAE={suspects['err_mod_emp'].abs().mean():.2f} "
      f"vs MAE(model,official)={suspects['err_mod_off'].abs().mean():.2f}")

# Same for empirical > 90% of official (where the lookup nearly suffices)
near = df[df["pct_emp_of_off"] > 0.9]
print(f"\n=== Rows where empirical >= 90% of official (n={len(near)}) ===")
print(f"  MAE(model, empirical) = ${near['err_mod_emp'].abs().mean():.2f}")
print(f"  MAE(model, official)  = ${near['err_mod_off'].abs().mean():.2f}")
print(f"  -> When the fee schedule already captures the truth, "
      f"the model's edge over the lookup shrinks to ${near['err_mod_off'].abs().mean() - near['err_mod_emp'].abs().mean():+.2f}.")

# Scatter plot
fig, axes = plt.subplots(1, 2, figsize=(13, 6))
mx = max(df["empirical"].max(), df["official_pfs"].max(), df["model_pred"].max()) * 1.05
buckets = sorted(df["bucket_name"].unique())
colors = plt.cm.tab10(np.linspace(0, 1, len(buckets)))
cmap = {b: c for b, c in zip(buckets, colors)}

ax = axes[0]
for b, sub in df.groupby("bucket_name"):
    ax.scatter(sub["empirical"], sub["model_pred"], s=44, label=b, color=cmap[b], alpha=0.85, edgecolor="k", linewidth=0.5)
ax.plot([0, mx], [0, mx], "k--", lw=1, alpha=0.5, label="y=x")
ax.set_xlabel("Empirical Avg_Mdcr_Alowd_Amt ($)")
ax.set_ylabel("Model prediction ($)")
ax.set_title(f"Model vs empirical (Pearson r = {df['model_pred'].corr(df['empirical']):.3f})")
ax.set_xlim(0, mx); ax.set_ylim(0, mx)
ax.legend(fontsize=8, loc="upper left")
ax.grid(alpha=0.3)

ax = axes[1]
for b, sub in df.groupby("bucket_name"):
    ax.scatter(sub["official_pfs"], sub["model_pred"], s=44, label=b, color=cmap[b], alpha=0.85, edgecolor="k", linewidth=0.5)
ax.plot([0, mx], [0, mx], "k--", lw=1, alpha=0.5, label="y=x")
ax.set_xlabel("Official PFS, state-area-weighted ($)")
ax.set_ylabel("Model prediction ($)")
ax.set_title(f"Model vs official (Pearson r = {df['model_pred'].corr(df['official_pfs']):.3f})")
ax.set_xlim(0, mx); ax.set_ylim(0, mx)
ax.legend(fontsize=8, loc="upper left")
ax.grid(alpha=0.3)

plt.tight_layout()
out_png = os.path.join(ROOT, "falsification_test", "scatter.png")
plt.savefig(out_png, dpi=110, bbox_inches="tight")
print(f"\nScatter saved to: {out_png}")
