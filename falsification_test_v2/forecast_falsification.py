"""
Forecast falsification: did the LSTM forecast for 2024 anticipate the actual
2024 policy reality (PFS conversion-factor cut + RVU updates)?

We do NOT have the 2024 actual PUF data (CMS publishes with a 2-year lag),
so this is a 'policy-consistency' check, not a true forecast accuracy test.
We compare:
  (a) LSTM forecast YoY change for 2024 (per specialty/bucket/state) vs
  (b) PFS-implied YoY change 2023 to 2024 (per state/bucket, volume-weighted)
For (b) we use the 2023 silver layer to weight by service volume.

Also documents the 2025/2026 autoregressive collapse already noted in CLAUDE.md.
"""

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
SILVER = ROOT / "local_pipeline" / "silver"
sys.path.insert(0, str(V2_DIR))
sys.path.insert(0, str(ROOT / "api"))

from run_v2 import load_indicators, load_localities, MAC_TO_STATE, lookup_indicator, price_per_locality, choose_modifier  # noqa: E402
from services.prediction import hcpcs_code_to_bucket, HCPCS_BUCKET_NAMES  # noqa: E402

ART = ROOT / "api" / "models" / "artifacts"

# Load the forecast file
FC_PATH = ROOT / "local_pipeline" / "lstm" / "forecast_2024_2026.parquet"
fc = pd.read_parquet(FC_PATH)
print(f"LSTM forecast file: {FC_PATH}")
print(f"  rows: {len(fc):,}  forecast_year: {sorted(fc['forecast_year'].unique())}")

# --------------------------------------------------------------------------
# Step 1. Decode the categorical indices via label_encoders.json
# --------------------------------------------------------------------------

import json
with open(ART / "label_encoders.json") as f:
    le = json.load(f)
specialties = le["Rndrng_Prvdr_Type"]
states = le["Rndrng_Prvdr_State_Abrvtn"]

idx_to_specialty = {i: n for i, n in enumerate(specialties)}
idx_to_state = {i: n for i, n in enumerate(states)}

fc["specialty"] = fc["Rndrng_Prvdr_Type_idx"].astype(int).map(idx_to_specialty)
fc["state"] = fc["Rndrng_Prvdr_State_Abrvtn_idx"].astype(int).map(idx_to_state)
fc["bucket"] = fc["hcpcs_bucket"].astype(int)
fc["bucket_name"] = fc["bucket"].map(HCPCS_BUCKET_NAMES)
print(f"  decoded {fc['specialty'].notna().sum()}/{len(fc)} specialty names, "
      f"{fc['state'].notna().sum()}/{len(fc)} state names")

# Restrict to last_known_year=2023 (proper 1-step lookahead for 2024)
fc23 = fc[fc["last_known_year"] == 2023].copy()
print(f"  rows with last_known_year=2023: {len(fc23):,} ({fc23['forecast_year'].value_counts().to_dict()})")

# --------------------------------------------------------------------------
# Step 2. Build PFS-implied 2024 YoY rate change per (state, bucket)
# --------------------------------------------------------------------------

# Aggregate code volumes from 2023 silver. Use Tot_Srvcs as weight.
print("\nAggregating 2023 silver by (state, bucket, HCPCS) for volume weighting...")
silver_files = sorted(SILVER.glob("*.parquet"))
state_bucket_volumes: dict[tuple, dict] = {}  # (state, bucket) -> {hcpcs: total_srvcs}

cols = ["Rndrng_Prvdr_State_Abrvtn", "HCPCS_Cd", "year", "Tot_Srvcs"]
for f in silver_files:
    df = pd.read_parquet(f, columns=cols)
    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    df = df[df["year"] == 2023]
    if len(df) == 0: continue
    df["HCPCS_Cd"] = df["HCPCS_Cd"].astype(str).str.strip()
    df = df[df["HCPCS_Cd"].str.match(r"^[A-Z0-9]{5}$", na=False)]
    df["bucket"] = df["HCPCS_Cd"].apply(hcpcs_code_to_bucket)
    df["Tot_Srvcs"] = pd.to_numeric(df["Tot_Srvcs"], errors="coerce").fillna(0)
    grp = df.groupby(["Rndrng_Prvdr_State_Abrvtn", "bucket", "HCPCS_Cd"])["Tot_Srvcs"].sum().reset_index()
    for _, r in grp.iterrows():
        key = (r["Rndrng_Prvdr_State_Abrvtn"], int(r["bucket"]))
        state_bucket_volumes.setdefault(key, {})
        state_bucket_volumes[key][r["HCPCS_Cd"]] = state_bucket_volumes[key].get(r["HCPCS_Cd"], 0.0) + r["Tot_Srvcs"]

print(f"  built {len(state_bucket_volumes):,} (state, bucket) cells from 2023 silver")

# For each (state, bucket), compute volume-weighted PFS rate change 2023->2024
print("\nPricing 2023 and 2024 modifier-aware rates per (state, bucket, code)...")
print("Using 2024B (final, $33.29 CF) as the 2024 rate; using Global modifier")
print("(modifier-26 only relevant for radiology specialty mapping which doesn't apply here).")

ind23 = load_indicators(2023)
ind24 = load_indicators(2024)
loc23 = load_localities(2023)
loc24 = load_localities(2024)

# Also report 2025 if data exists
ind25 = load_indicators(2025)
loc25 = load_localities(2025)

baseline_rows = []
for (state, bucket), code_vols in state_bucket_volumes.items():
    if state not in MAC_TO_STATE.values():
        continue
    state_loc23 = loc23[loc23["state"] == state]
    state_loc24 = loc24[loc24["state"] == state]
    state_loc25 = loc25[loc25["state"] == state]
    if len(state_loc23) == 0 or len(state_loc24) == 0:
        continue

    # Filter to top 80 codes by 2023 volume (covers most of the bucket signal)
    top_codes = sorted(code_vols.items(), key=lambda x: -x[1])[:80]

    sum_w = 0.0
    sum_r23 = sum_r24 = sum_r25 = 0.0
    used_codes = 0
    for code, vol in top_codes:
        # Use Global rate for the policy-baseline check
        r23 = lookup_indicator(ind23, code, "")
        r24 = lookup_indicator(ind24, code, "")
        r25 = lookup_indicator(ind25, code, "")
        if r23 is None or r24 is None:
            continue
        # Use non-facility for buckets 4 (E&M) and 5 (Level II); facility for surgery (1).
        # Use both averaged for radiology to be safe. Do nonfacility average across localities:
        p23 = price_per_locality(r23, state_loc23, pos_facility=False)["price"].mean()
        p24 = price_per_locality(r24, state_loc24, pos_facility=False)["price"].mean()
        p25 = price_per_locality(r25, state_loc25, pos_facility=False)["price"].mean() if (r25 is not None and len(state_loc25) > 0) else np.nan
        if not np.isfinite(p23) or not np.isfinite(p24) or p23 <= 0 or p24 <= 0:
            continue
        sum_w += vol
        sum_r23 += vol * p23
        sum_r24 += vol * p24
        if np.isfinite(p25): sum_r25 += vol * p25
        used_codes += 1

    if sum_w == 0 or used_codes < 3:
        continue
    avg_r23 = sum_r23 / sum_w
    avg_r24 = sum_r24 / sum_w
    avg_r25 = (sum_r25 / sum_w) if sum_r25 > 0 else np.nan
    baseline_rows.append({
        "state": state, "bucket": bucket,
        "bucket_name": HCPCS_BUCKET_NAMES.get(bucket, str(bucket)),
        "n_codes": used_codes,
        "avg_rate_2023": avg_r23,
        "avg_rate_2024": avg_r24,
        "avg_rate_2025": avg_r25,
        "pfs_change_2024": avg_r24 / avg_r23 - 1,
        "pfs_change_2025": (avg_r25 / avg_r24 - 1) if np.isfinite(avg_r25) else np.nan,
    })

baseline = pd.DataFrame(baseline_rows)
print(f"  baseline (state, bucket) cells: {len(baseline)}")
print("\nPFS-implied 2024 vs 2023 YoY change distribution:")
print(f"  mean={baseline['pfs_change_2024'].mean():.4f}  median={baseline['pfs_change_2024'].median():.4f}  "
      f"p10={baseline['pfs_change_2024'].quantile(0.1):.4f}  p90={baseline['pfs_change_2024'].quantile(0.9):.4f}")
print("Per-bucket PFS YoY change 2024:")
for b, sub in baseline.groupby("bucket_name"):
    print(f"  {b:<18} n_states={len(sub):>2}  "
          f"mean={sub['pfs_change_2024'].mean():>+.4f}  median={sub['pfs_change_2024'].median():>+.4f}")

# --------------------------------------------------------------------------
# Step 3. Compare forecast YoY to PFS YoY
# --------------------------------------------------------------------------

print("\nMerging forecast (2024 only) with PFS baseline...")
fc24 = fc23[fc23["forecast_year"] == 2024].copy()
# Drop rows where last_known_value is 0 or non-positive (can't compute YoY change)
fc24 = fc24[fc24["last_known_value"] > 1.0].copy()
fc24["forecast_change"] = fc24["forecast_mean"] / fc24["last_known_value"] - 1
# Clip extreme outliers to keep histograms readable: drop |change| > 2.0
fc24 = fc24[fc24["forecast_change"].abs() < 2.0].copy()

merged = fc24.merge(baseline[["state", "bucket", "pfs_change_2024", "n_codes", "avg_rate_2023", "avg_rate_2024"]],
                    on=["state", "bucket"], how="inner")
merged["residual"] = merged["forecast_change"] - merged["pfs_change_2024"]
print(f"  merged cells: {len(merged):,} (forecast n={len(fc24):,}, baseline cells used)")

print("\n=== Forecast YoY 2024 vs PFS YoY 2024 ===")
print(f"  forecast_change       : mean={merged['forecast_change'].mean():>+.4f}  median={merged['forecast_change'].median():>+.4f}")
print(f"  PFS_change            : mean={merged['pfs_change_2024'].mean():>+.4f}  median={merged['pfs_change_2024'].median():>+.4f}")
print(f"  residual (fc - PFS)   : mean={merged['residual'].mean():>+.4f}  median={merged['residual'].median():>+.4f}  "
      f"p10={merged['residual'].quantile(0.1):>+.4f}  p90={merged['residual'].quantile(0.9):>+.4f}")
print(f"  abs residual          : MAE={merged['residual'].abs().mean():.4f}  P50={merged['residual'].abs().median():.4f}  P90={merged['residual'].abs().quantile(0.9):.4f}")
print(f"  Pearson(fc_change, PFS_change) = {merged['forecast_change'].corr(merged['pfs_change_2024']):.4f}")

print("\nPer-bucket residuals:")
for b, sub in merged.groupby("bucket_name"):
    print(f"  {b:<18} n={len(sub):>5}  "
          f"forecast_change={sub['forecast_change'].mean():>+.4f}  "
          f"PFS_change={sub['pfs_change_2024'].mean():>+.4f}  "
          f"residual={sub['residual'].mean():>+.4f}  "
          f"|residual|={sub['residual'].abs().mean():.4f}")

# --------------------------------------------------------------------------
# Step 4. Document the 2025/2026 autoregressive collapse
# --------------------------------------------------------------------------

pivot = fc23.pivot_table(
    index=["specialty", "state", "bucket"],
    columns="forecast_year", values="forecast_mean").reset_index()
pivot.columns.name = None
pivot.columns = [str(c) if isinstance(c, (int, np.integer)) else c for c in pivot.columns]
pivot = pivot.merge(
    fc23[["specialty", "state", "bucket", "last_known_value"]].drop_duplicates(),
    on=["specialty", "state", "bucket"]
)
pivot = pivot[(pivot["2024"] > 1.0) & (pivot["2025"].notna()) & (pivot["2026"].notna())].copy()
pivot["ch_25_24"] = pivot["2025"] / pivot["2024"] - 1
pivot["ch_26_25"] = pivot["2026"] / pivot["2025"] - 1
collapse_25 = (pivot["2025"] < 0.5 * pivot["2024"]).mean()
collapse_26 = (pivot["2026"] < 0.5 * pivot["2025"]).mean()
print("\n=== 2025/2026 autoregressive collapse ===")
print(f"  Fraction of (specialty,state,bucket) with 2025_forecast < 50% of 2024_forecast: {collapse_25:.1%}")
print(f"  Fraction with 2026 < 50% of 2025: {collapse_26:.1%}")
print(f"  ch_25_24: median={pivot['ch_25_24'].median():.4f}  p10={pivot['ch_25_24'].quantile(0.1):.4f}")
print(f"  ch_26_25: median={pivot['ch_26_25'].median():.4f}  p10={pivot['ch_26_25'].quantile(0.1):.4f}")

# --------------------------------------------------------------------------
# Step 5. Save outputs
# --------------------------------------------------------------------------

merged.to_csv(V2_DIR / "forecast_falsification_2024.csv", index=False)
baseline.to_csv(V2_DIR / "pfs_baseline_state_bucket.csv", index=False)
print(f"\nSaved {V2_DIR / 'forecast_falsification_2024.csv'} ({len(merged):,} rows)")
print(f"Saved {V2_DIR / 'pfs_baseline_state_bucket.csv'} ({len(baseline):,} rows)")

# --------------------------------------------------------------------------
# Step 6. Plots
# --------------------------------------------------------------------------

fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# (1) Histogram of forecast vs PFS YoY change
ax = axes[0, 0]
ax.hist(merged["forecast_change"], bins=40, alpha=0.55, label="LSTM forecast YoY (2024)", color="steelblue")
ax.hist(merged["pfs_change_2024"], bins=40, alpha=0.55, label="PFS-implied YoY (2024)", color="orangered")
ax.axvline(0, color="k", linestyle=":", alpha=0.4)
ax.set_xlabel("YoY change (fraction)")
ax.set_ylabel("count")
ax.set_xlim(-0.3, 0.3)
ax.set_title(f"Forecast vs PFS-implied 2024 YoY change\n(n={len(merged):,} cells; medians {merged['forecast_change'].median():+.3f} vs {merged['pfs_change_2024'].median():+.3f})")
ax.legend(); ax.grid(alpha=0.3)

# (2) Scatter forecast_change vs PFS_change colored by bucket
ax = axes[0, 1]
buckets = sorted(merged["bucket_name"].dropna().unique())
colors = plt.cm.tab10(np.linspace(0, 1, max(len(buckets), 3)))
cmap = {b: c for b, c in zip(buckets, colors)}
for b, sub in merged.groupby("bucket_name"):
    ax.scatter(sub["pfs_change_2024"], sub["forecast_change"], s=12, alpha=0.5,
               color=cmap[b], label=b, edgecolor="none")
ax.plot([-0.3, 0.3], [-0.3, 0.3], "k--", lw=1, alpha=0.5)
ax.axhline(0, color="k", alpha=0.2); ax.axvline(0, color="k", alpha=0.2)
ax.set_xlim(-0.2, 0.2); ax.set_ylim(-0.3, 0.3)
ax.set_xlabel("PFS-implied YoY change 2024"); ax.set_ylabel("LSTM forecast YoY change 2024")
ax.set_title(f"Forecast vs PFS YoY (Pearson {merged['forecast_change'].corr(merged['pfs_change_2024']):.3f})")
ax.legend(fontsize=8, loc="upper left"); ax.grid(alpha=0.3)

# (3) Per-bucket residual boxplots
ax = axes[1, 0]
data, labels = [], []
for b, sub in merged.groupby("bucket_name"):
    data.append(sub["residual"].values); labels.append(f"{b}\n(n={len(sub)})")
ax.boxplot(data, labels=labels, showfliers=False)
ax.axhline(0, color="r", linestyle="--", alpha=0.5)
ax.set_ylabel("residual = forecast_change - PFS_change")
ax.set_title("Residual by HCPCS bucket (LSTM forecast minus PFS-implied)")
ax.grid(alpha=0.3)

# (4) 2025/2026 collapse demonstration: ratio 2025/2024 histogram
ax = axes[1, 1]
ratio = pivot["2025"] / pivot["2024"]
ax.hist(ratio.clip(lower=0, upper=1.5), bins=60, color="firebrick", alpha=0.75)
ax.axvline(1.0, color="k", linestyle="--", alpha=0.5, label="ratio = 1.0 (flat)")
ax.set_xlabel("forecast_2025 / forecast_2024")
ax.set_ylabel("count")
ax.set_title(f"LSTM 2025/2024 ratio: {(ratio < 0.5).mean():.0%} of cells collapse below 0.5\n(autoregressive rollout bug)")
ax.legend(); ax.grid(alpha=0.3)

plt.tight_layout()
out_png = V2_DIR / "forecast_falsification.png"
plt.savefig(out_png, dpi=110, bbox_inches="tight")
print(f"Saved {out_png}")

# --------------------------------------------------------------------------
# Final summary
# --------------------------------------------------------------------------

print("\n" + "=" * 80)
print("FORECAST FALSIFICATION SUMMARY")
print("=" * 80)
fc_med = merged["forecast_change"].median()
pfs_med = merged["pfs_change_2024"].median()
print(f"LSTM forecast median YoY 2024 = {fc_med:+.2%}  (range p10..p90: "
      f"{merged['forecast_change'].quantile(0.1):+.2%} .. {merged['forecast_change'].quantile(0.9):+.2%})")
print(f"PFS-implied  median YoY 2024 = {pfs_med:+.2%}  (range p10..p90: "
      f"{merged['pfs_change_2024'].quantile(0.1):+.2%} .. {merged['pfs_change_2024'].quantile(0.9):+.2%})")
print(f"Median residual = {fc_med - pfs_med:+.2%}")
print(f"Pearson(forecast, PFS) = {merged['forecast_change'].corr(merged['pfs_change_2024']):.4f}")
print(f"\nWithout 2024 PUF actuals we cannot compute true forecast accuracy. "
      f"The forecast directionally {'matched' if abs(fc_med - pfs_med) < 0.02 else 'diverged from'} the PFS policy reality.")
print(f"\n2025/2026 LSTM autoregressive collapse: {collapse_25:.0%} of 2025 cells fall below 50% of 2024.")
