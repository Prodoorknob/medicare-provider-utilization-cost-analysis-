"""Server-side Supabase client for reference data queries."""

from collections import defaultdict
from functools import lru_cache

from supabase import create_client, Client

from config import settings
from services.specialty_canonicalization import (
    ALIAS_IDXS,
    canonicalize_idx,
    expand_canonical,
)


@lru_cache(maxsize=1)
def get_client() -> Client:
    """Cached Supabase client using service role key."""
    return create_client(settings.supabase_url, settings.supabase_service_role_key)


async def fetch_labels(category: str | None = None) -> list[dict]:
    client = get_client()
    query = client.table("lookup_labels").select("*")
    if category:
        query = query.eq("category", category)
    result = query.execute()
    data = result.data
    if category == "specialty":
        data = [row for row in data if row.get("idx") not in ALIAS_IDXS]
    return data


async def fetch_state_summary() -> list[dict]:
    client = get_client()
    result = client.table("state_summary").select("*").order("state_abbrev").execute()
    return result.data


async def fetch_model_metrics() -> list[dict]:
    client = get_client()
    result = client.table("model_metrics").select("*").execute()
    return result.data


async def fetch_feature_importances() -> list[dict]:
    client = get_client()
    result = client.table("feature_importances").select("*").order("importance", desc=True).execute()
    return result.data


async def fetch_specialty_history(specialty_idx: int) -> list[dict]:
    """Yearly average allowed amounts for a specialty (2013-2023).

    If `specialty_idx` belongs to a CMS-rename pair (Cardiology, Colorectal Surgery,
    Oral Surgery), rows across all indices in the group are unioned and aggregated
    (mean) per year so callers see full 2013-2023 coverage.
    """
    client = get_client()
    idxs = expand_canonical(specialty_idx)
    canonical = canonicalize_idx(specialty_idx)
    result = (
        client.table("specialty_yearly_avg")
        .select("*")
        .in_("specialty_idx", idxs)
        .order("year")
        .execute()
    )
    rows = result.data
    if len(idxs) == 1:
        return rows
    by_year: dict[int, list[dict]] = defaultdict(list)
    for r in rows:
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
    client = get_client()
    idxs = expand_canonical(specialty_idx)
    canonical = canonicalize_idx(specialty_idx)
    query = (
        client.table("lstm_forecasts")
        .select("*")
        .in_("specialty_idx", idxs)
    )
    if state_idx is not None:
        query = query.eq("state_idx", state_idx)
    if hcpcs_bucket is not None:
        query = query.eq("hcpcs_bucket", hcpcs_bucket)
    result = query.order("forecast_year").execute()
    rows = result.data
    if len(idxs) == 1:
        return rows
    buckets: dict[tuple, list[dict]] = defaultdict(list)
    for r in rows:
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
