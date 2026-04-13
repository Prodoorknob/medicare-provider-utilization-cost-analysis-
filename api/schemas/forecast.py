"""Schemas for LSTM forecast endpoints."""

from pydantic import BaseModel


class LstmForecast(BaseModel):
    specialty_idx: int
    hcpcs_bucket: int
    state_idx: int
    forecast_year: int
    forecast_mean: float
    forecast_std: float | None = None
    forecast_p10: float
    forecast_p50: float
    forecast_p90: float
    last_known_year: int | None = None
    last_known_value: float | None = None
    n_history_years: int | None = None
