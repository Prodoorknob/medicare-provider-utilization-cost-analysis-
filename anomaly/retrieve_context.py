"""
retrieve_context.py -- Phase C, spec section 5

Build a ProviderContext for a given (NPI, year) by stitching together:
    - npi_profiles.parquet (this NPI's metrics + history)
    - specialty_benchmarks.parquet (national specialty stats)
    - state_specialty_benchmarks.parquet (state-level stats)
    - silver/{state}.parquet (top HCPCS codes for the NPI)

Designed to be called many times: ContextRetriever caches parquets and
silver-per-state reads so a batch of N NPIs amortizes loading costs.
"""

import os
import numpy as np
import pandas as pd

from schemas import ProviderContext


_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_ANOMALY_DIR = os.path.join(_PROJECT_ROOT, "local_pipeline", "anomaly")
DEFAULT_SILVER_DIR  = os.path.join(_PROJECT_ROOT, "local_pipeline", "silver")

# Metrics to compute percentile ranks against peers for
PCT_RANK_METRICS = [
    "total_services",
    "total_beneficiaries",
    "total_billing",
    "srvcs_per_bene",
    "avg_charge",
    "avg_allowed",
    "charge_to_allowed_ratio",
    "n_unique_hcpcs",
    "herfindahl_index",
    "facility_pct",
]

# Metrics surfaced in benchmark comparison tables
BENCH_METRICS = [
    "total_services",
    "total_billing",
    "srvcs_per_bene",
    "avg_allowed",
    "charge_to_allowed_ratio",
    "n_unique_hcpcs",
    "herfindahl_index",
    "facility_pct",
    "risk_score",
]

BUCKET_NAMES = {
    0: "Anesthesia",
    1: "Surgery",
    2: "Radiology",
    3: "Lab/Pathology",
    4: "Medicine/E&M",
    5: "HCPCS Level II",
}


class ContextRetriever:
    """Loads profiles + benchmarks once, caches silver reads per state."""

    def __init__(
        self,
        anomaly_dir: str = DEFAULT_ANOMALY_DIR,
        silver_dir:  str = DEFAULT_SILVER_DIR,
    ):
        self.anomaly_dir = anomaly_dir
        self.silver_dir  = silver_dir

        print("Loading profiles + benchmarks...")
        self.profiles = pd.read_parquet(os.path.join(anomaly_dir, "npi_profiles.parquet"))
        self.sp_bench = pd.read_parquet(os.path.join(anomaly_dir, "specialty_benchmarks.parquet"))
        self.ss_bench = pd.read_parquet(os.path.join(anomaly_dir, "state_specialty_benchmarks.parquet"))

        # Optional: specialty-HCPCS scope table. If absent, out-of-specialty
        # evaluation stays disabled (rule reports NOT EVALUABLE, same as before).
        scope_path = os.path.join(anomaly_dir, "specialty_scopes.parquet")
        self.scopes: dict[str, set[str]] | None = None
        self.scope_meta: dict | None = None
        if os.path.exists(scope_path):
            sc = pd.read_parquet(scope_path, columns=["specialty", "HCPCS_Cd", "in_scope"])
            self.scopes = {
                spec: set(grp.loc[grp["in_scope"], "HCPCS_Cd"].astype(str).tolist())
                for spec, grp in sc.groupby("specialty", observed=True)
            }
            summary_path = os.path.join(anomaly_dir, "specialty_scopes_summary.json")
            if os.path.exists(summary_path):
                import json
                with open(summary_path, encoding="utf-8") as fh:
                    self.scope_meta = json.load(fh).get("thresholds")
            print(f"  scopes:   {len(self.scopes):,} specialties loaded")

        # Index for fast NPI lookup
        self.profiles_by_npi = self.profiles.set_index("Rndrng_NPI", drop=False).sort_index()

        # Quick lookup for benchmark rows
        self.sp_bench_idx = self.sp_bench.set_index(["specialty", "year"], drop=False)
        self.ss_bench_idx = self.ss_bench.set_index(["specialty", "state", "year"], drop=False)

        # Lazy silver cache
        self._silver_cache: dict[str, pd.DataFrame] = {}

        print(f"  profiles: {len(self.profiles):,} rows, "
              f"{self.profiles['Rndrng_NPI'].nunique():,} unique NPIs")

    # --- silver cache ------------------------------------------------------

    def _load_silver_state(self, state: str) -> pd.DataFrame | None:
        if state in self._silver_cache:
            return self._silver_cache[state]
        path = os.path.join(self.silver_dir, f"{state}.parquet")
        if not os.path.exists(path):
            self._silver_cache[state] = None
            return None
        cols = ["Rndrng_NPI", "HCPCS_Cd", "HCPCS_Desc", "Tot_Srvcs", "year"]
        df = pd.read_parquet(path, columns=cols)
        df["Rndrng_NPI"] = df["Rndrng_NPI"].astype(str).str.strip()
        df["year"]       = pd.to_numeric(df["year"], errors="coerce").astype("Int16")
        self._silver_cache[state] = df
        return df

    # --- component helpers -------------------------------------------------

    def _bench_row_to_dict(self, row: pd.Series) -> dict[str, dict[str, float]]:
        """Extract {metric: {mean, p5, p25, p50, p75, p95}} from a benchmark row."""
        out: dict[str, dict[str, float]] = {}
        for m in BENCH_METRICS:
            stats = {}
            for stat in ["mean", "p5", "p25", "p50", "p75", "p95"]:
                col = f"{m}_{stat}"
                if col in row.index and pd.notna(row[col]):
                    stats[stat] = float(row[col])
            if stats:
                out[m] = stats
        out["_n_providers"] = {"n": float(row.get("n_providers", np.nan))}
        return out

    def _percentile_ranks(self, specialty: str, year: int, npi_row: pd.Series) -> dict[str, float]:
        """NPI's percentile rank (0-100) within specialty-year peer group per metric."""
        peers = self.profiles[
            (self.profiles["specialty"] == specialty) &
            (self.profiles["year"]      == year)
        ]
        out: dict[str, float] = {}
        for m in PCT_RANK_METRICS:
            if m not in peers.columns or m not in npi_row.index:
                continue
            v = peers[m].dropna()
            if v.empty or pd.isna(npi_row[m]):
                continue
            out[m] = float((v <= npi_row[m]).mean() * 100)
        return out

    def _npi_hcpcs_frame(self, npi: str, state: str, year: int) -> pd.DataFrame | None:
        """All (HCPCS_Cd, HCPCS_Desc, total Tot_Srvcs) for one (NPI, year)."""
        silver = self._load_silver_state(state)
        if silver is None:
            return None
        sub = silver[(silver["Rndrng_NPI"] == npi) & (silver["year"] == year)]
        if sub.empty:
            return None
        return (
            sub.groupby(["HCPCS_Cd", "HCPCS_Desc"], observed=True, sort=False)
               ["Tot_Srvcs"].sum()
               .reset_index()
               .rename(columns={"Tot_Srvcs": "count"})
               .sort_values("count", ascending=False)
        )

    def _top_hcpcs(self, all_codes: pd.DataFrame | None, top_n: int = 10) -> list[dict]:
        if all_codes is None or all_codes.empty:
            return []
        total = all_codes["count"].sum()
        out = all_codes.head(top_n).copy()
        out["pct_of_total"] = (out["count"] / total * 100).round(1) if total > 0 else 0.0
        return out.to_dict(orient="records")

    def _out_of_specialty(
        self, all_codes: pd.DataFrame | None, specialty: str,
    ) -> tuple[float | None, list[str]]:
        """Return (pct of services on out-of-scope codes, up-to-10 out-of-scope codes)."""
        if self.scopes is None or all_codes is None or all_codes.empty:
            return None, []
        whitelist = self.scopes.get(specialty)
        if whitelist is None:
            return None, []
        total = float(all_codes["count"].sum())
        if total <= 0:
            return None, []
        mask = ~all_codes["HCPCS_Cd"].astype(str).isin(whitelist)
        oos = all_codes[mask]
        pct = float(oos["count"].sum() / total)
        top_out = oos.head(10)["HCPCS_Cd"].astype(str).tolist()
        return pct, top_out

    def _trend_direction(self, history: list[dict]) -> str:
        """Classify volume trajectory from history (chronological)."""
        if len(history) < 2:
            return "insufficient_history"
        yoys = [h.get("yoy_volume_change") for h in history if h.get("yoy_volume_change") is not None]
        if not yoys:
            return "insufficient_history"
        last = yoys[-1]
        if last is not None and last > 1.0:
            return "spike"
        recent = [y for y in yoys[-3:] if y is not None]
        if recent and np.mean(recent) > 0.1:
            return "increasing"
        if recent and np.mean(recent) < -0.1:
            return "decreasing"
        return "stable"

    # --- main API ---------------------------------------------------------

    def get_context(self, npi: str, year: int) -> ProviderContext | None:
        npi = str(npi).strip()
        try:
            rows = self.profiles_by_npi.loc[[npi]]
        except KeyError:
            return None
        current = rows[rows["year"] == year]
        if current.empty:
            return None
        r = current.iloc[0]
        specialty = str(r["specialty"])
        state     = str(r["state"])

        # History -- chronological
        hist_cols = [
            "year", "total_services", "total_beneficiaries", "total_billing",
            "srvcs_per_bene", "avg_allowed", "charge_to_allowed_ratio",
            "n_unique_hcpcs", "herfindahl_index", "facility_pct",
            "yoy_volume_change", "yoy_billing_change", "risk_score",
        ]
        history = (
            rows.sort_values("year")[hist_cols]
                .to_dict(orient="records")
        )
        # Cast types for JSON-friendliness
        for h in history:
            for k, v in list(h.items()):
                if isinstance(v, (np.floating, np.integer)):
                    h[k] = float(v) if pd.notna(v) else None

        # Benchmarks
        try:
            nat_row = self.sp_bench_idx.loc[(specialty, year)]
            if isinstance(nat_row, pd.DataFrame):
                nat_row = nat_row.iloc[0]
            specialty_national = self._bench_row_to_dict(nat_row)
        except KeyError:
            specialty_national = {}

        try:
            st_row = self.ss_bench_idx.loc[(specialty, state, year)]
            if isinstance(st_row, pd.DataFrame):
                st_row = st_row.iloc[0]
            specialty_state = self._bench_row_to_dict(st_row)
        except KeyError:
            specialty_state = {}

        # Specialty median risk score (from national benchmark row)
        risk_median = None
        if "risk_score" in specialty_national:
            risk_median = specialty_national["risk_score"].get("p50")

        # Core metrics snapshot (current year)
        metrics_snapshot = {}
        for m in PCT_RANK_METRICS + ["total_allowed", "risk_score"]:
            if m in r.index and pd.notna(r[m]):
                metrics_snapshot[m] = float(r[m])

        # Bucket distribution
        bucket_dist = {}
        for b in range(6):
            col = f"bucket_{b}_pct"
            if col in r.index and pd.notna(r[col]):
                bucket_dist[BUCKET_NAMES[b]] = float(r[col])

        # Percentile ranks
        pct_ranks = self._percentile_ranks(specialty, year, r)

        # HCPCS breakdown (requires silver)
        all_codes = self._npi_hcpcs_frame(npi, state, year)
        top_hcpcs = self._top_hcpcs(all_codes)
        oos_pct, oos_codes = self._out_of_specialty(all_codes, specialty)
        if oos_pct is not None:
            metrics_snapshot["out_of_specialty_pct"] = oos_pct

        return ProviderContext(
            npi=npi,
            year=int(year),
            specialty=specialty,
            state=state,
            years_active=int(len(rows)),
            risk_score=float(r["risk_score"]) if pd.notna(r.get("risk_score")) else None,
            risk_score_specialty_median=risk_median,
            metrics=metrics_snapshot,
            specialty_national=specialty_national,
            specialty_state=specialty_state,
            percentile_ranks=pct_ranks,
            history=history,
            trend_direction=self._trend_direction(history),
            top_hcpcs=top_hcpcs,
            bucket_distribution=bucket_dist,
            out_of_specialty_codes=oos_codes,
            data_available={
                "metrics": True,
                "history": True,
                "national_benchmark": bool(specialty_national),
                "state_benchmark":    bool(specialty_state),
                "top_hcpcs":          bool(top_hcpcs),
                "out_of_specialty":   oos_pct is not None,
                "rural_geocontext":   False,
                "beneficiary_linkage": False,
                "date_of_service":    False,
                "diagnosis_codes":    False,
            },
        )


if __name__ == "__main__":
    # Smoke test: print a context summary for a top flagged NPI
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--npi",  required=True)
    ap.add_argument("--year", type=int, required=True)
    args = ap.parse_args()

    r = ContextRetriever()
    ctx = r.get_context(args.npi, args.year)
    if ctx is None:
        print("NPI/year not found")
    else:
        print(f"NPI:       {ctx.npi}")
        print(f"Specialty: {ctx.specialty}  ({ctx.state})")
        print(f"Years active: {ctx.years_active}")
        print(f"Risk score: {ctx.risk_score:.2f} (specialty median {ctx.risk_score_specialty_median})")
        print(f"Trend:     {ctx.trend_direction}")
        print(f"\nMetrics ({args.year}):")
        for k, v in ctx.metrics.items():
            pct = ctx.percentile_ranks.get(k)
            pct_s = f" (P{pct:.0f})" if pct is not None else ""
            print(f"  {k}: {v:,.2f}{pct_s}")
        print(f"\nTop HCPCS ({len(ctx.top_hcpcs)} shown):")
        for h in ctx.top_hcpcs[:5]:
            print(f"  {h['HCPCS_Cd']}: {h['count']:,.0f} srvcs ({h['pct_of_total']}% of total) -- {h['HCPCS_Desc'][:60]}")
        print(f"\nBucket mix: {ctx.bucket_distribution}")
        print(f"\nData available: {ctx.data_available}")
