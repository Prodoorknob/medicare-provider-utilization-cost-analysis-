"""Server-side Supabase client for reference data queries."""

from functools import lru_cache

from supabase import create_client, Client

from config import settings


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
    return result.data


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


async def fetch_forecasts(
    specialty_idx: int,
    state_idx: int | None = None,
    hcpcs_bucket: int | None = None,
) -> list[dict]:
    client = get_client()
    query = (
        client.table("lstm_forecasts")
        .select("*")
        .eq("specialty_idx", specialty_idx)
    )
    if state_idx is not None:
        query = query.eq("state_idx", state_idx)
    if hcpcs_bucket is not None:
        query = query.eq("hcpcs_bucket", hcpcs_bucket)
    result = query.order("forecast_year").execute()
    return result.data
