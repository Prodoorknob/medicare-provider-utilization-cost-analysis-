"""
detect_outliers.py -- Phase B, spec section 4

Runs three anomaly detection methods over npi_profiles.parquet and produces
a unified long-format flags table for downstream investigation.

Methods:
    A. Z-score within (specialty, state, year) groups; falls back to
       (specialty, year) when state-level group size < min_group_size.
    B. Isolation Forest per specialty (pooled across years),
       contamination = 0.01.
    C. Temporal rules from precomputed yoy_* columns.

Output: local_pipeline/anomaly/flags.parquet  (long format, one row per flag)
    columns: npi, year, specialty, state, flag_type, flag_metric,
             flag_reason, severity, value, benchmark_mean, benchmark_std

Usage:
    python anomaly/detect_outliers.py
    python anomaly/detect_outliers.py --z-threshold 3.0 --if-contamination 0.01
    python anomaly/detect_outliers.py --methods zscore,temporal
"""

import os
import argparse
import time
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

_PROJECT_ROOT    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_OUT_DIR  = os.path.join(_PROJECT_ROOT, "local_pipeline", "anomaly")
DEFAULT_PROFILES = os.path.join(DEFAULT_OUT_DIR, "npi_profiles.parquet")
DEFAULT_FLAGS    = os.path.join(DEFAULT_OUT_DIR, "flags.parquet")

# Metrics flagged by z-score (spec 4.1 Method A).
# Heavy-tailed metrics are log1p-transformed before z-scoring so that the
# 3-sigma tail matches a ~1% rate rather than over-firing on lognormal data.
Z_METRICS = [
    "total_services",
    "srvcs_per_bene",
    "avg_allowed",
    "charge_to_allowed_ratio",
    "n_unique_hcpcs",
    "herfindahl_index",
]
Z_METRICS_LOG = {
    "total_services":          True,
    "srvcs_per_bene":          True,
    "avg_allowed":             True,
    "charge_to_allowed_ratio": True,
    "n_unique_hcpcs":          True,
    "herfindahl_index":        False,   # already bounded in [0, 1]
}

# Features for Isolation Forest (spec 4.1 Method B)
IF_FEATURES = [
    "total_services",
    "total_beneficiaries",
    "srvcs_per_bene",
    "avg_charge",
    "avg_allowed",
    "charge_to_allowed_ratio",
    "n_unique_hcpcs",
    "herfindahl_index",
    "facility_pct",
    "bucket_0_pct", "bucket_1_pct", "bucket_2_pct",
    "bucket_3_pct", "bucket_4_pct", "bucket_5_pct",
    "risk_score",
]


# ---------------------------------------------------------------------------
# Method A: Z-score
# ---------------------------------------------------------------------------

def _group_stats(profiles: pd.DataFrame, keys: list[str], metric: str) -> pd.DataFrame:
    """Compute mean, std, count for `metric` within group `keys`."""
    g = profiles.groupby(keys, observed=True)[metric]
    stats = g.agg(mean="mean", std="std", count="count").reset_index()
    return stats


def detect_zscore(
    profiles: pd.DataFrame,
    threshold: float = 3.0,
    min_group_size: int = 30,
) -> pd.DataFrame:
    """Flag NPI-years where any Z_METRIC z-score exceeds threshold.

    Primary benchmark: (specialty, state, year).
    Fallback:         (specialty, year) when primary group size < min_group_size.
    """
    flags_list = []
    # Light columns only for joins
    base = profiles[["Rndrng_NPI", "year", "specialty", "state"] + Z_METRICS].copy()

    for metric in Z_METRICS:
        # Drop NaNs for this metric
        df = base[["Rndrng_NPI", "year", "specialty", "state", metric]].dropna(subset=[metric])
        if df.empty:
            continue

        # Heavy-tailed metrics: compute stats on log1p scale (clip >=0 first)
        use_log = Z_METRICS_LOG.get(metric, False)
        work_col = f"_{metric}_z"
        if use_log:
            df[work_col] = np.log1p(df[metric].clip(lower=0))
        else:
            df[work_col] = df[metric]

        # Primary: state-specialty-year stats
        sss = _group_stats(df, ["specialty", "state", "year"], work_col).rename(
            columns={"mean": "mean_sss", "std": "std_sss", "count": "n_sss"}
        )
        # Fallback: specialty-year stats
        sy  = _group_stats(df, ["specialty", "year"], work_col).rename(
            columns={"mean": "mean_sy", "std": "std_sy", "count": "n_sy"}
        )

        df = df.merge(sss, on=["specialty", "state", "year"], how="left")
        df = df.merge(sy,  on=["specialty", "year"],          how="left")

        # Choose benchmark: use state-specialty-year when group is dense enough, else specialty-year
        use_sss = df["n_sss"].fillna(0) >= min_group_size
        df["bench_mean"] = np.where(use_sss, df["mean_sss"], df["mean_sy"])
        df["bench_std"]  = np.where(use_sss, df["std_sss"],  df["std_sy"])
        df["bench_scope"] = np.where(use_sss, "specialty_state_year", "specialty_year")

        # Z-score computed on the transformed column (log or linear as selected above)
        std_safe = df["bench_std"].replace(0, np.nan)
        df["z"] = (df[work_col] - df["bench_mean"]) / std_safe
        hits = df[df["z"].abs() > threshold].copy()
        if hits.empty:
            continue

        hits["flag_type"]      = "z_score"
        hits["flag_metric"]    = metric
        hits["value"]          = hits[metric]
        hits["benchmark_mean"] = hits["bench_mean"]
        hits["benchmark_std"]  = hits["bench_std"]
        hits["severity"]       = np.minimum(hits["z"].abs() / 10.0, 1.0)
        scale = "log1p" if use_log else "linear"
        hits["flag_reason"]    = hits.apply(
            lambda r: f"{metric} z={r['z']:+.2f} [{scale}] "
                      f"(value={r[metric]:.2f}, {r['bench_scope']} mean={r['bench_mean']:.2f}, "
                      f"std={r['bench_std']:.2f})",
            axis=1,
        )

        flags_list.append(hits[[
            "Rndrng_NPI", "year", "specialty", "state",
            "flag_type", "flag_metric", "flag_reason",
            "severity", "value", "benchmark_mean", "benchmark_std",
        ]])
        print(f"  z-score {metric}: {len(hits):,} flags "
              f"(thin groups fell back to specialty-year: {(~use_sss[df['z'].abs() > threshold]).sum():,})")

    if not flags_list:
        return pd.DataFrame()
    return pd.concat(flags_list, ignore_index=True)


# ---------------------------------------------------------------------------
# Method B: Isolation Forest
# ---------------------------------------------------------------------------

def detect_isolation_forest(
    profiles: pd.DataFrame,
    contamination: float = 0.01,
    min_providers: int = 200,
    random_state: int = 42,
) -> pd.DataFrame:
    """Per-specialty Isolation Forest; flag rows with score below threshold."""
    flags_list = []
    specialty_sizes = profiles.groupby("specialty", observed=True).size()
    eligible = specialty_sizes[specialty_sizes >= min_providers].index.tolist()
    print(f"  specialties evaluated: {len(eligible)} "
          f"(skipped {len(specialty_sizes) - len(eligible)} with <{min_providers} rows)")

    for specialty in eligible:
        sub = profiles[profiles["specialty"] == specialty].copy()
        X = sub[IF_FEATURES].replace([np.inf, -np.inf], np.nan).fillna(0.0).values
        if len(X) < min_providers:
            continue
        model = IsolationForest(
            n_estimators=100,
            contamination=contamination,
            random_state=random_state,
            n_jobs=-1,
        )
        model.fit(X)
        scores = model.decision_function(X)      # higher = more normal
        preds  = model.predict(X)                # -1 = outlier
        is_outlier = preds == -1
        if not is_outlier.any():
            continue

        hits = sub[is_outlier].copy()
        hits["score"] = scores[is_outlier]
        # Normalize severity: more negative score = more anomalous -> higher severity
        min_s, max_s = scores.min(), scores.max()
        denom = max_s - min_s if max_s > min_s else 1.0
        hits["severity"] = ((max_s - hits["score"]) / denom).clip(0.0, 1.0)
        hits["flag_type"]      = "isolation_forest"
        hits["flag_metric"]    = "multivariate"
        hits["flag_reason"]    = hits["score"].apply(
            lambda s: f"iforest_score={s:+.4f} (specialty='{specialty}', contamination={contamination})"
        )
        hits["value"]          = hits["score"]
        hits["benchmark_mean"] = float(np.mean(scores))
        hits["benchmark_std"]  = float(np.std(scores))

        flags_list.append(hits[[
            "Rndrng_NPI", "year", "specialty", "state",
            "flag_type", "flag_metric", "flag_reason",
            "severity", "value", "benchmark_mean", "benchmark_std",
        ]])

    if not flags_list:
        return pd.DataFrame()
    flags = pd.concat(flags_list, ignore_index=True)
    print(f"  isolation forest flags: {len(flags):,}")
    return flags


# ---------------------------------------------------------------------------
# Method C: Temporal rules (spec 4.1 Method C)
# ---------------------------------------------------------------------------

def detect_temporal(
    profiles: pd.DataFrame,
    exclude_years: tuple[int, ...] = (2021,),
    min_abs_services: float = 1000.0,
    yoy_volume_thresh: float = 2.0,
    yoy_billing_thresh: float = 2.5,
    spike_volume_thresh: float = 2.0,
    spike_bene_thresh: float = 0.5,
) -> pd.DataFrame:
    """Rule-based YoY jumps with absolute-volume gates.

    Absolute gate filters out provider ramp-ups (e.g. 10 -> 30 services = +200%
    but not suspicious).  exclude_years drops the COVID recovery year(s) where
    everyone's YoY reset artificially.  Only right-tail spikes (value > thresh)
    are flagged, not downward swings.
    """

    flags_list = []
    base = profiles[[
        "Rndrng_NPI", "year", "specialty", "state",
        "yoy_volume_change", "yoy_billing_change", "yoy_bene_change",
        "total_services",
    ]].copy()

    # Filter: exclude COVID recovery years and low absolute-volume providers
    eligible = base[
        (~base["year"].isin(list(exclude_years))) &
        (base["total_services"] >= min_abs_services)
    ]

    rules = [
        ("yoy_volume_change",  yoy_volume_thresh,  "yoy_volume >{thr:.0%} ({val:+.0%}), current services={svcs:.0f}"),
        ("yoy_billing_change", yoy_billing_thresh, "yoy_billing >{thr:.0%} ({val:+.0%}), current services={svcs:.0f}"),
    ]

    for metric, thresh, reason_tmpl in rules:
        sub = eligible[eligible[metric] > thresh].copy()
        if sub.empty:
            continue
        sub["flag_type"]      = "temporal"
        sub["flag_metric"]    = metric
        sub["value"]          = sub[metric]
        sub["benchmark_mean"] = thresh
        sub["benchmark_std"]  = np.nan
        sub["severity"]       = np.minimum(sub[metric] / (thresh * 5.0), 1.0)
        sub["flag_reason"]    = sub.apply(
            lambda r: reason_tmpl.format(thr=thresh, val=r[metric], svcs=r["total_services"]),
            axis=1,
        )
        flags_list.append(sub[[
            "Rndrng_NPI", "year", "specialty", "state",
            "flag_type", "flag_metric", "flag_reason",
            "severity", "value", "benchmark_mean", "benchmark_std",
        ]])
        print(f"  temporal {metric}>{thresh}: {len(sub):,} flags")

    # Volume spike with flat beneficiaries (upcoding/unbundling pattern)
    spike_mask = (
        (eligible["yoy_volume_change"] > spike_volume_thresh) &
        (eligible["yoy_bene_change"]   < spike_bene_thresh)
    )
    spike = eligible[spike_mask].copy()
    if not spike.empty:
        spike["flag_type"]      = "temporal"
        spike["flag_metric"]    = "volume_spike_flat_benes"
        spike["value"]          = spike["yoy_volume_change"]
        spike["benchmark_mean"] = spike_volume_thresh
        spike["benchmark_std"]  = np.nan
        spike["severity"]       = np.minimum(spike["yoy_volume_change"] / 10.0, 1.0)
        spike["flag_reason"]    = spike.apply(
            lambda r: f"volume {r['yoy_volume_change']:+.0%} with benes only "
                      f"{r['yoy_bene_change']:+.0%} (possible upcoding/unbundling), "
                      f"current services={r['total_services']:.0f}",
            axis=1,
        )
        flags_list.append(spike[[
            "Rndrng_NPI", "year", "specialty", "state",
            "flag_type", "flag_metric", "flag_reason",
            "severity", "value", "benchmark_mean", "benchmark_std",
        ]])
        print(f"  temporal volume_spike_flat_benes: {len(spike):,} flags")

    if not flags_list:
        return pd.DataFrame()
    return pd.concat(flags_list, ignore_index=True)


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--profiles", default=DEFAULT_PROFILES)
    ap.add_argument("--output",   default=DEFAULT_FLAGS)
    ap.add_argument("--methods",  default="zscore,isolation_forest,temporal",
                    help="Comma-separated subset of {zscore, isolation_forest, temporal}")
    ap.add_argument("--z-threshold",      type=float, default=3.0)
    ap.add_argument("--z-min-group-size", type=int,   default=30)
    ap.add_argument("--if-contamination", type=float, default=0.01)
    ap.add_argument("--if-min-providers", type=int,   default=200)
    args = ap.parse_args()

    methods = {m.strip() for m in args.methods.split(",")}

    print(f"Loading profiles: {args.profiles}")
    profiles = pd.read_parquet(args.profiles)
    print(f"  {len(profiles):,} NPI-year rows")

    all_flags = []

    if "zscore" in methods:
        print(f"\n[A] Z-score (threshold={args.z_threshold}, "
              f"min_group_size={args.z_min_group_size})...")
        t0 = time.time()
        f_z = detect_zscore(profiles, args.z_threshold, args.z_min_group_size)
        print(f"  -> {len(f_z):,} z-score flags in {time.time()-t0:.1f}s")
        if not f_z.empty:
            all_flags.append(f_z)

    if "isolation_forest" in methods:
        print(f"\n[B] Isolation Forest (contamination={args.if_contamination}, "
              f"min_providers={args.if_min_providers})...")
        t0 = time.time()
        f_i = detect_isolation_forest(profiles, args.if_contamination, args.if_min_providers)
        print(f"  -> {len(f_i):,} isolation-forest flags in {time.time()-t0:.1f}s")
        if not f_i.empty:
            all_flags.append(f_i)

    if "temporal" in methods:
        print("\n[C] Temporal rules...")
        t0 = time.time()
        f_t = detect_temporal(profiles)
        print(f"  -> {len(f_t):,} temporal flags in {time.time()-t0:.1f}s")
        if not f_t.empty:
            all_flags.append(f_t)

    if not all_flags:
        print("No flags produced.")
        return

    flags = pd.concat(all_flags, ignore_index=True)
    flags.to_parquet(args.output, index=False, compression="snappy")

    print(f"\nWrote {len(flags):,} flag rows -> {args.output}")

    # Summary
    print("\n-- Flag summary --")
    print(f"  Total flag rows:       {len(flags):,}")
    print(f"  Unique (NPI, year):    {flags[['Rndrng_NPI','year']].drop_duplicates().shape[0]:,}")
    print(f"  Flag rate vs profiles: "
          f"{flags[['Rndrng_NPI','year']].drop_duplicates().shape[0] / len(profiles) * 100:.2f}%")
    print("\n  Flags by type:")
    print(flags.groupby("flag_type").size().rename("count").to_string())
    print("\n  Flags by metric (top 10):")
    print(flags.groupby("flag_metric").size().nlargest(10).rename("count").to_string())
    print("\n  Flag rate by specialty (top 10, minimum 500 profile rows):")
    sp_profile_n = profiles.groupby("specialty", observed=True).size().rename("n_profile")
    sp_flag_n    = flags.drop_duplicates(subset=["Rndrng_NPI", "year", "specialty"])\
                        .groupby("specialty", observed=True).size().rename("n_flag")
    rate = pd.concat([sp_profile_n, sp_flag_n], axis=1).fillna(0)
    rate = rate[rate["n_profile"] >= 500]
    rate["flag_pct"] = rate["n_flag"] / rate["n_profile"] * 100
    print(rate.sort_values("flag_pct", ascending=False).head(10).round(2).to_string())


if __name__ == "__main__":
    main()
