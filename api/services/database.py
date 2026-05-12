"""Server-side Postgres client for reference data queries.

Replaces the previous Supabase/PostgREST data layer with direct asyncpg against
Railway Postgres. Query shapes (single-table SELECT + filters + ORDER BY) are
the same; the merge/aggregation logic for CMS specialty-rename pairs lives in
Python and is unchanged.
"""

from collections import defaultdict

import asyncpg

from config import settings
from services.specialty_canonicalization import (
    ALIAS_IDXS,
    canonicalize_idx,
    expand_canonical,
)


_pool: asyncpg.Pool | None = None


async def get_pool() -> asyncpg.Pool:
    """Lazy-init shared connection pool against Railway Postgres."""
    global _pool
    if _pool is None:
        _pool = await asyncpg.create_pool(
            dsn=settings.database_url,
            min_size=1,
            max_size=10,
            command_timeout=30,
        )
    return _pool


async def close_pool() -> None:
    global _pool
    if _pool is not None:
        await _pool.close()
        _pool = None


def _records_to_dicts(records: list[asyncpg.Record]) -> list[dict]:
    return [dict(r) for r in records]


async def fetch_labels(category: str | None = None) -> list[dict]:
    pool = await get_pool()
    if category:
        rows = await pool.fetch(
            "SELECT * FROM lookup_labels WHERE category = $1",
            category,
        )
    else:
        rows = await pool.fetch("SELECT * FROM lookup_labels")
    data = _records_to_dicts(rows)
    if category == "specialty":
        data = [row for row in data if row.get("idx") not in ALIAS_IDXS]
    return data


async def fetch_state_summary() -> list[dict]:
    pool = await get_pool()
    rows = await pool.fetch(
        "SELECT * FROM state_summary ORDER BY state_abbrev"
    )
    return _records_to_dicts(rows)


async def fetch_model_metrics() -> list[dict]:
    pool = await get_pool()
    rows = await pool.fetch("SELECT * FROM model_metrics")
    return _records_to_dicts(rows)


async def fetch_feature_importances() -> list[dict]:
    pool = await get_pool()
    rows = await pool.fetch(
        "SELECT * FROM feature_importances ORDER BY importance DESC"
    )
    return _records_to_dicts(rows)


async def fetch_specialty_history(specialty_idx: int) -> list[dict]:
    """Yearly average allowed amounts for a specialty (2013-2023).

    If `specialty_idx` belongs to a CMS-rename pair (Cardiology, Colorectal Surgery,
    Oral Surgery), rows across all indices in the group are unioned and aggregated
    (mean) per year so callers see full 2013-2023 coverage.
    """
    pool = await get_pool()
    idxs = expand_canonical(specialty_idx)
    canonical = canonicalize_idx(specialty_idx)
    rows = await pool.fetch(
        "SELECT * FROM specialty_yearly_avg "
        "WHERE specialty_idx = ANY($1::int[]) "
        "ORDER BY year",
        idxs,
    )
    data = _records_to_dicts(rows)
    if len(idxs) == 1:
        return data
    by_year: dict[int, list[dict]] = defaultdict(list)
    for r in data:
        year = r.get("year")
        if year is not None:
            by_year[year].append(r)
    merged: list[dict] = []
    for year in sorted(by_year):
        group = by_year[year]
        means = [g["mean_allowed"] for g in group if g.get("mean_allowed") is not None]
        merged.append({
            "specialty_idx": canonical,
            "year": year,
            "mean_allowed": sum(means) / len(means) if means else None,
        })
    return merged


async def fetch_forecasts(
    specialty_idx: int,
    state_idx: int | None = None,
    hcpcs_bucket: int | None = None,
) -> list[dict]:
    """LSTM 2024-2026 forecasts.

    If `specialty_idx` belongs to a CMS-rename pair, forecast rows from all indices in
    the group are unioned and aggregated (mean) per (state, bucket, year).
    """
    pool = await get_pool()
    idxs = expand_canonical(specialty_idx)
    canonical = canonicalize_idx(specialty_idx)
    clauses = ["specialty_idx = ANY($1::int[])"]
    params: list = [idxs]
    if state_idx is not None:
        params.append(state_idx)
        clauses.append(f"state_idx = ${len(params)}")
    if hcpcs_bucket is not None:
        params.append(hcpcs_bucket)
        clauses.append(f"hcpcs_bucket = ${len(params)}")
    sql = (
        "SELECT * FROM lstm_forecasts "
        f"WHERE {' AND '.join(clauses)} "
        "ORDER BY forecast_year"
    )
    rows = await pool.fetch(sql, *params)
    data = _records_to_dicts(rows)
    if len(idxs) == 1:
        return data
    buckets: dict[tuple, list[dict]] = defaultdict(list)
    for r in data:
        key = (r.get("state_idx"), r.get("hcpcs_bucket"), r.get("forecast_year"))
        buckets[key].append(r)
    numeric_cols = (
        "forecast_mean", "forecast_std",
        "forecast_p10", "forecast_p50", "forecast_p90",
        "last_known_value",
    )
    merged: list[dict] = []
    for group in buckets.values():
        merged_row = dict(group[0])
        merged_row["specialty_idx"] = canonical
        for col in numeric_cols:
            vals = [g[col] for g in group if g.get(col) is not None]
            if vals:
                merged_row[col] = sum(vals) / len(vals)
        n_history = [g["n_history_years"] for g in group if g.get("n_history_years") is not None]
        if n_history:
            merged_row["n_history_years"] = max(n_history)
        last_years = [g["last_known_year"] for g in group if g.get("last_known_year") is not None]
        if last_years:
            merged_row["last_known_year"] = max(last_years)
        merged.append(merged_row)
    merged.sort(key=lambda r: (r.get("forecast_year") or 0, r.get("state_idx") or 0, r.get("hcpcs_bucket") or 0))
    return merged
