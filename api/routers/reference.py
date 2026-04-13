"""Reference data endpoints — proxies Supabase tables."""

from fastapi import APIRouter, HTTPException

from services.supabase import (
    fetch_feature_importances,
    fetch_labels,
    fetch_model_metrics,
    fetch_state_summary,
)

router = APIRouter(tags=["reference"])


@router.get("/labels")
async def labels(category: str | None = None):
    """Lookup labels for dropdowns (specialty, state, etc.)."""
    try:
        data = await fetch_labels(category)
    except Exception as e:
        raise HTTPException(502, f"Supabase query failed: {e}")
    return data


@router.get("/state-summary")
async def state_summary():
    """Aggregate statistics per state."""
    try:
        data = await fetch_state_summary()
    except Exception as e:
        raise HTTPException(502, f"Supabase query failed: {e}")
    return data


@router.get("/model-metrics")
async def model_metrics():
    """Model performance metrics."""
    try:
        data = await fetch_model_metrics()
    except Exception as e:
        raise HTTPException(502, f"Supabase query failed: {e}")
    return data


@router.get("/feature-importances")
async def feature_importances():
    """Feature importance rankings."""
    try:
        data = await fetch_feature_importances()
    except Exception as e:
        raise HTTPException(502, f"Supabase query failed: {e}")
    return data
