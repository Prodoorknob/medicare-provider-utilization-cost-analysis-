"""One-shot migration: Supabase Postgres -> Railway Postgres.

Reads from Supabase via PostgREST (service_role key bypasses RLS) and writes
to Railway via asyncpg COPY. Schema and indexes are recreated from scratch.

Usage:
    python scripts/migrate_supabase_to_railway.py \
        --supabase-url https://zdkoniqnvbklxtsviikl.supabase.co \
        --supabase-key "$SUPABASE_SERVICE_ROLE_KEY" \
        --railway-url "$RAILWAY_DATABASE_URL"

If env vars SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY, RAILWAY_DATABASE_URL are
set, the flags can be omitted.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
import time

import asyncpg
import httpx


SCHEMA_SQL = """
DROP TABLE IF EXISTS public.lookup_labels CASCADE;
DROP TABLE IF EXISTS public.stage1_allowed_amounts CASCADE;
DROP TABLE IF EXISTS public.stage2_oop_estimates CASCADE;
DROP TABLE IF EXISTS public.lstm_forecasts CASCADE;
DROP TABLE IF EXISTS public.state_summary CASCADE;
DROP TABLE IF EXISTS public.model_metrics CASCADE;
DROP TABLE IF EXISTS public.feature_importances CASCADE;
DROP TABLE IF EXISTS public.specialty_yearly_avg CASCADE;

CREATE TABLE public.lookup_labels (
    id        SERIAL PRIMARY KEY,
    category  TEXT NOT NULL,
    idx       INTEGER NOT NULL,
    label     TEXT NOT NULL,
    UNIQUE (category, idx)
);

CREATE TABLE public.stage1_allowed_amounts (
    id                SERIAL PRIMARY KEY,
    specialty_idx     SMALLINT NOT NULL,
    hcpcs_bucket      SMALLINT NOT NULL,
    state_idx         SMALLINT NOT NULL,
    place_of_service  SMALLINT NOT NULL,
    n_records         INTEGER NOT NULL,
    mean_allowed      REAL NOT NULL,
    median_allowed    REAL NOT NULL,
    p10_allowed       REAL NOT NULL,
    p90_allowed       REAL NOT NULL,
    mean_charge       REAL,
    mean_risk_score   REAL,
    UNIQUE (specialty_idx, hcpcs_bucket, state_idx, place_of_service)
);

CREATE TABLE public.stage2_oop_estimates (
    id                SERIAL PRIMARY KEY,
    specialty_idx     SMALLINT NOT NULL,
    hcpcs_bucket      SMALLINT NOT NULL,
    census_region     SMALLINT NOT NULL,
    dual_eligible     SMALLINT NOT NULL,
    has_supplemental  SMALLINT NOT NULL,
    age_group         SMALLINT NOT NULL,
    income_bracket    SMALLINT NOT NULL,
    n_records         INTEGER NOT NULL,
    oop_p10           REAL NOT NULL,
    oop_p50           REAL NOT NULL,
    oop_p90           REAL NOT NULL,
    mean_allowed      REAL,
    UNIQUE (specialty_idx, hcpcs_bucket, census_region, dual_eligible, has_supplemental, age_group, income_bracket)
);

CREATE TABLE public.lstm_forecasts (
    id                SERIAL PRIMARY KEY,
    specialty_idx     SMALLINT NOT NULL,
    hcpcs_bucket      SMALLINT NOT NULL,
    state_idx         SMALLINT NOT NULL,
    forecast_year     SMALLINT NOT NULL,
    forecast_mean     REAL NOT NULL,
    forecast_std      REAL,
    forecast_p10      REAL NOT NULL,
    forecast_p50      REAL NOT NULL,
    forecast_p90      REAL NOT NULL,
    last_known_year   SMALLINT,
    last_known_value  REAL,
    n_history_years   SMALLINT,
    UNIQUE (specialty_idx, hcpcs_bucket, state_idx, forecast_year)
);

CREATE TABLE public.state_summary (
    id              SERIAL PRIMARY KEY,
    state_abbrev    TEXT NOT NULL UNIQUE,
    state_idx       SMALLINT NOT NULL,
    mean_allowed    REAL NOT NULL,
    median_allowed  REAL NOT NULL,
    n_records       INTEGER NOT NULL
);

CREATE TABLE public.model_metrics (
    id            SERIAL PRIMARY KEY,
    model_name    TEXT NOT NULL,
    stage         SMALLINT NOT NULL DEFAULT 1,
    metric_name   TEXT NOT NULL,
    metric_value  REAL NOT NULL,
    UNIQUE (model_name, stage, metric_name)
);

CREATE TABLE public.feature_importances (
    id            SERIAL PRIMARY KEY,
    model_name    TEXT NOT NULL,
    feature_name  TEXT NOT NULL,
    importance    REAL NOT NULL,
    rank          SMALLINT,
    UNIQUE (model_name, feature_name)
);

CREATE TABLE public.specialty_yearly_avg (
    specialty_idx  INTEGER NOT NULL,
    year           INTEGER NOT NULL,
    mean_allowed   DOUBLE PRECISION NOT NULL,
    PRIMARY KEY (specialty_idx, year)
);
"""

INDEX_SQL = """
CREATE INDEX idx_lookup_category ON public.lookup_labels (category);
CREATE INDEX idx_s1_lookup
    ON public.stage1_allowed_amounts (specialty_idx, hcpcs_bucket, state_idx, place_of_service);
CREATE INDEX idx_s2_lookup
    ON public.stage2_oop_estimates (specialty_idx, hcpcs_bucket, census_region, dual_eligible, has_supplemental);
CREATE INDEX idx_lstm_specialty ON public.lstm_forecasts (specialty_idx);
"""


# (supabase_table, railway_table, columns_in_order)
TABLES: list[tuple[str, str, list[str]]] = [
    ("lookup_labels", "lookup_labels",
     ["id", "category", "idx", "label"]),
    ("stage1_allowed_amounts", "stage1_allowed_amounts",
     ["id", "specialty_idx", "hcpcs_bucket", "state_idx", "place_of_service",
      "n_records", "mean_allowed", "median_allowed", "p10_allowed", "p90_allowed",
      "mean_charge", "mean_risk_score"]),
    ("stage2_oop_estimates", "stage2_oop_estimates",
     ["id", "specialty_idx", "hcpcs_bucket", "census_region", "dual_eligible",
      "has_supplemental", "age_group", "income_bracket", "n_records",
      "oop_p10", "oop_p50", "oop_p90", "mean_allowed"]),
    ("lstm_forecasts", "lstm_forecasts",
     ["id", "specialty_idx", "hcpcs_bucket", "state_idx", "forecast_year",
      "forecast_mean", "forecast_std", "forecast_p10", "forecast_p50",
      "forecast_p90", "last_known_year", "last_known_value", "n_history_years"]),
    ("state_summary", "state_summary",
     ["id", "state_abbrev", "state_idx", "mean_allowed", "median_allowed", "n_records"]),
    ("model_metrics", "model_metrics",
     ["id", "model_name", "stage", "metric_name", "metric_value"]),
    ("feature_importances", "feature_importances",
     ["id", "model_name", "feature_name", "importance", "rank"]),
    ("specialty_yearly_avg", "specialty_yearly_avg",
     ["specialty_idx", "year", "mean_allowed"]),
]


# Tables with a SERIAL `id` whose sequence must be reset after bulk insert
ID_SEQUENCES = [
    ("lookup_labels", "lookup_labels_id_seq"),
    ("stage1_allowed_amounts", "stage1_allowed_amounts_id_seq"),
    ("stage2_oop_estimates", "stage2_oop_estimates_id_seq"),
    ("lstm_forecasts", "lstm_forecasts_id_seq"),
    ("state_summary", "state_summary_id_seq"),
    ("model_metrics", "model_metrics_id_seq"),
    ("feature_importances", "feature_importances_id_seq"),
]


PAGE = 1000  # PostgREST default cap


async def fetch_all(client: httpx.AsyncClient, base: str, table: str, columns: list[str]) -> list[tuple]:
    """Page through PostgREST and return a list of column-ordered tuples."""
    out: list[tuple] = []
    offset = 0
    select = ",".join(columns)
    while True:
        url = f"{base}/rest/v1/{table}"
        params = {"select": select, "limit": str(PAGE), "offset": str(offset)}
        r = await client.get(url, params=params)
        r.raise_for_status()
        rows = r.json()
        if not rows:
            break
        for row in rows:
            out.append(tuple(row.get(c) for c in columns))
        if len(rows) < PAGE:
            break
        offset += PAGE
    return out


async def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--supabase-url", default=os.environ.get("SUPABASE_URL"))
    parser.add_argument("--supabase-key", default=os.environ.get("SUPABASE_SERVICE_ROLE_KEY"))
    parser.add_argument("--railway-url", default=os.environ.get("RAILWAY_DATABASE_URL"))
    parser.add_argument("--drop", action="store_true", help="Drop+recreate schema on Railway")
    args = parser.parse_args()

    missing = [n for n, v in [("SUPABASE_URL", args.supabase_url),
                              ("SUPABASE_SERVICE_ROLE_KEY", args.supabase_key),
                              ("RAILWAY_DATABASE_URL", args.railway_url)] if not v]
    if missing:
        print(f"ERROR: missing required values: {missing}", file=sys.stderr)
        return 2

    headers = {
        "apikey": args.supabase_key,
        "Authorization": f"Bearer {args.supabase_key}",
        "Accept": "application/json",
    }

    t0 = time.time()
    async with httpx.AsyncClient(headers=headers, timeout=120.0) as http:
        rw = await asyncpg.connect(args.railway_url, statement_cache_size=0)
        try:
            if args.drop:
                print("[schema] dropping + recreating tables...")
                await rw.execute(SCHEMA_SQL)
                print("[schema] creating indexes...")
                await rw.execute(INDEX_SQL)

            for src, dst, cols in TABLES:
                print(f"[{dst}] reading from supabase...")
                t = time.time()
                records = await fetch_all(http, args.supabase_url, src, cols)
                read_s = time.time() - t

                print(f"[{dst}] {len(records):>6} rows fetched in {read_s:.1f}s; "
                      f"writing to railway...")
                t = time.time()
                await rw.copy_records_to_table(
                    dst, records=records, columns=cols, schema_name="public",
                )
                write_s = time.time() - t
                print(f"[{dst}] wrote in {write_s:.1f}s")

            print("[sequences] resetting...")
            for tbl, seq in ID_SEQUENCES:
                await rw.execute(
                    f"SELECT setval('public.{seq}', "
                    f"COALESCE((SELECT MAX(id) FROM public.{tbl}), 1))"
                )

            print("[verify] row counts:")
            for _, dst, _ in TABLES:
                n = await rw.fetchval(f"SELECT COUNT(*) FROM public.{dst}")
                print(f"   {dst:30s} {n:>7}")
        finally:
            await rw.close()

    print(f"Done in {time.time() - t0:.1f}s.")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
