"""Schemas for reference/lookup data endpoints."""

from pydantic import BaseModel


class LookupLabel(BaseModel):
    id: int
    category: str
    idx: int
    label: str


class StateSummary(BaseModel):
    state_abbrev: str
    state_idx: int
    mean_allowed: float
    median_allowed: float
    n_records: int


class ModelMetric(BaseModel):
    model_name: str
    stage: int
    metric_name: str
    metric_value: float


class FeatureImportance(BaseModel):
    model_name: str
    feature_name: str
    importance: float
    rank: int | None = None
