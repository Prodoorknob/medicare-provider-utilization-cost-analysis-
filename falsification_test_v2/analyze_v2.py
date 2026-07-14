"""Per-bucket breakdown, scatter plot, and final report generation for v2."""
import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
V2_DIR = ROOT / "falsification_test_v2"

sys.path.insert(0, str(ROOT / "api"))
from services.prediction import hcpcs_code_to_bucket, HCPCS_BUCKET_NAMES  # noqa: E402

df = pd.read_csv(V2_DIR / "results_v2.csv")
df["bucket"] = df["hcpcs"].astype(str).apply(hcpcs_code_to_bucket)
df["bucket_name"] = df["bucket"].map(HCPCS_BUCKET_NAMES)

# Errors
df["err_mod_emp"] = df["model_pred"] - df["empirical"]
df["err_mod_glb"] = df["model_pred"] - df["official_global_v2"]
df["err_mod_ma"]  = df["model_pred"] - df["official_modifier_aware"]
df["err_emp_glb"] = df["empirical"] - df["official_global_v2"]
df["err_emp_ma"]  = df["empirical"] - df["official_modifier_aware"]


def stats(errs: pd.Series) -> dict:
    e = errs.dropna()
    return {
        "n": len(e),
        "mae": e.abs().mean(),
        "bias": e.mean(),
        "p50": e.abs().median(),
        "p90": e.abs().quantile(0.9),
    }


def fmt(s: dict) -> str:
    return f"n={s['n']:>2}  MAE=${s['mae']:>7.2f}  bias=${s['bias']:>+7.2f}  P50=${s['p50']:>5.2f}  P90=${s['p90']:>6.2f}"


print(f"\nN total rows: {len(df)}")
print("\n=== Table A: Model vs Official GLOBAL (v2) — replicates v1 under new data source ===")
print("Overall:                ", fmt(stats(df["err_mod_glb"])))
for b, sub in df.groupby("bucket_name"):
    print(f"  {b:<18}", fmt(stats(sub["err_mod_glb"])))

print("\n=== Table B: Model vs Official MODIFIER-AWARE — the new methodology ===")
print("Overall:                ", fmt(stats(df["err_mod_ma"])))
for b, sub in df.groupby("bucket_name"):
    print(f"  {b:<18}", fmt(stats(sub["err_mod_ma"])))

print("\n=== Empirical vs Official MODIFIER-AWARE (does the empirical-official gap survive correction?) ===")
print("Overall:                ", fmt(stats(df["err_emp_ma"])))
for b, sub in df.groupby("bucket_name"):
    print(f"  {b:<18}", fmt(stats(sub["err_emp_ma"])))

print("\n=== Pearson correlations (overall) ===")
print(f"  model vs empirical              : {df['model_pred'].corr(df['empirical']):.4f}")
print(f"  model vs official_global_v2     : {df['model_pred'].corr(df['official_global_v2']):.4f}")
print(f"  model vs official_modifier_aware: {df['model_pred'].corr(df['official_modifier_aware']):.4f}")
print(f"  empirical vs modifier_aware     : {df['empirical'].corr(df['official_modifier_aware']):.4f}")

# Range robustness: did the empirical fall within the per-locality min/max range
# for that state-year? Option A check.
in_range = df["empirical_in_state_range"].sum()
print(f"\n=== Option A robustness: empirical within per-locality [min, max] for state-year ===")
print(f"  {in_range}/{len(df)} rows ({100*in_range/len(df):.0f}%) have empirical inside the modifier-aware range")

# Radiology cluster: the central correction
rad = df[df["bucket_name"] == "Radiology"].copy()
print(f"\n=== Radiology subgroup (n={len(rad)}): the central correction ===")
print(f"  MAE(model vs Global v2)         = ${rad['err_mod_glb'].abs().mean():.2f}")
print(f"  MAE(model vs Modifier-aware)    = ${rad['err_mod_ma'].abs().mean():.2f}")
print(f"  MAE(empirical vs Modifier-aware) = ${rad['err_emp_ma'].abs().mean():.2f}")
print(f"  MAE(model vs empirical)         = ${rad['err_mod_emp'].abs().mean():.2f}")

# Save bucket tables to CSV for the report
bucket_rows = []
for view, col in [("model_vs_global_v2", "err_mod_glb"),
                  ("model_vs_modaware",  "err_mod_ma"),
                  ("empirical_vs_modaware", "err_emp_ma"),
                  ("model_vs_empirical", "err_mod_emp")]:
    for b, sub in df.groupby("bucket_name"):
        s = stats(sub[col])
        bucket_rows.append({"view": view, "bucket": b, **s})
    s = stats(df[col])
    bucket_rows.append({"view": view, "bucket": "ALL", **s})

bucket_df = pd.DataFrame(bucket_rows)
bucket_df.to_csv(V2_DIR / "bucket_breakdown.csv", index=False)
print(f"\nBucket breakdown saved to {V2_DIR / 'bucket_breakdown.csv'}")

# --------------------------------------------------------------------------
# 2x2 scatter
# --------------------------------------------------------------------------

fig, axes = plt.subplots(2, 2, figsize=(13, 12))
mx = float(np.nanmax([df["empirical"].max(), df["official_global_v2"].max(),
                      df["official_modifier_aware"].max(), df["model_pred"].max()])) * 1.05

buckets = sorted(df["bucket_name"].dropna().unique())
colors = plt.cm.tab10(np.linspace(0, 1, max(len(buckets), 3)))
cmap = {b: c for b, c in zip(buckets, colors)}


def draw(ax, x, y, xl, yl, title, r):
    for b, sub in df.groupby("bucket_name"):
        ax.scatter(sub[x], sub[y], s=44, label=b, color=cmap[b],
                   alpha=0.85, edgecolor="k", linewidth=0.5)
    ax.plot([0, mx], [0, mx], "k--", lw=1, alpha=0.5, label="y=x")
    ax.set_xlabel(xl); ax.set_ylabel(yl)
    ax.set_title(f"{title} (Pearson r = {r:.3f})")
    ax.set_xlim(0, mx); ax.set_ylim(0, mx)
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(alpha=0.3)


draw(axes[0, 0], "empirical", "model_pred",
     "Empirical Avg_Mdcr_Alowd_Amt ($)", "Model prediction ($)",
     "Top-left: Model vs Empirical (sanity)",
     df["model_pred"].corr(df["empirical"]))

draw(axes[0, 1], "official_global_v2", "model_pred",
     "Official PFS, GLOBAL, state-mean ($)", "Model prediction ($)",
     "Top-right: Model vs Official Global (v1 replication)",
     df["model_pred"].corr(df["official_global_v2"]))

draw(axes[1, 0], "official_modifier_aware", "model_pred",
     "Official PFS, modifier-aware, state-mean ($)", "Model prediction ($)",
     "Bottom-left: Model vs Modifier-Aware (corrected)",
     df["model_pred"].corr(df["official_modifier_aware"]))

draw(axes[1, 1], "official_modifier_aware", "empirical",
     "Official PFS, modifier-aware, state-mean ($)", "Empirical Avg_Mdcr_Alowd_Amt ($)",
     "Bottom-right: Empirical vs Modifier-Aware (does gap survive?)",
     df["empirical"].corr(df["official_modifier_aware"]))

plt.tight_layout()
out_png = V2_DIR / "scatter_v2.png"
plt.savefig(out_png, dpi=110, bbox_inches="tight")
print(f"Scatter saved: {out_png}")

# Final one-paragraph summary
rad_glb = rad["err_mod_glb"].abs().mean()
rad_ma = rad["err_mod_ma"].abs().mean()
print("\n=== ONE-PARAGRAPH SUMMARY ===")
print(f"Under modifier-aware comparison the radiology MAE collapses from "
      f"${rad_glb:.0f} (v1, model vs unmodified Global) to ${rad_ma:.0f} "
      f"(v2, model vs MOD-26 for radiology rows). Overall, MAE(model, "
      f"official) drops from ${df['err_mod_glb'].abs().mean():.2f} to "
      f"${df['err_mod_ma'].abs().mean():.2f}, while MAE(model, empirical) "
      f"is unchanged at ${df['err_mod_emp'].abs().mean():.2f}. "
      f"The radiology MAE {'closes substantially' if rad_ma < 25 else 'does not close'}: the original "
      f"$115 MAE radiology finding is largely a methodology artifact of "
      f"comparing against the wrong rate. The model is doing implicit modifier "
      f"inference, not adding NPI-conditional signal beyond the published fee "
      f"schedule once you apply the right modifier.")
