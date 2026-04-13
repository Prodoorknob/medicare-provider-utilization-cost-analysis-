"""LSTM forecast endpoint — proxies Supabase lstm_forecasts table."""

from fastapi import APIRouter, HTTPException, Query

from services.supabase import fetch_forecasts

router = APIRouter(tags=["forecast"])


@router.get("/forecast")
async def forecast(
    specialty_idx: int = Query(..., description="Specialty index"),
    state_idx: int | None = Query(None, description="State index (optional filter)"),
    hcpcs_bucket: int | None = Query(None, ge=0, le=5, description="HCPCS bucket (optional filter)"),
):
    """LSTM 2024-2026 forecasts with confidence bounds."""
    try:
        data = await fetch_forecasts(specialty_idx, state_idx, hcpcs_bucket)
    except Exception as e:
        raise HTTPException(502, f"Supabase query failed: {e}")
    return data
