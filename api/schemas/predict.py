"""Request/response schemas for prediction endpoints."""

from pydantic import BaseModel, Field


class Stage1Request(BaseModel):
    """Input for Stage 1 (Medicare allowed amount) prediction."""

    provider_type: str = Field(..., description="Provider specialty name, e.g. 'Cardiology'")
    state: str = Field(..., description="Two-letter state abbreviation, e.g. 'CA'")
    hcpcs_code: str | None = Field(None, description="HCPCS/CPT code, e.g. '99213'. If omitted, hcpcs_bucket is used.")
    hcpcs_bucket: int = Field(4, ge=0, le=5, description="Service category 0-5 (0=Anesthesia, 1=Surgery, 2=Radiology, 3=Lab, 4=Medicine/E&M, 5=HCPCS Level II)")
    place_of_service: int = Field(0, ge=0, le=1, description="0=Office, 1=Facility")

    # Optional continuous features — server applies defaults from training medians
    risk_score: float | None = Field(None, description="Beneficiary HCC risk score (default 1.0)")
    total_services: float | None = Field(None, description="Total service count (default 50)")
    total_beneficiaries: float | None = Field(None, description="Total beneficiary count (default 20)")
    avg_submitted_charge: float | None = Field(None, description="Average submitted charge in dollars (default 200)")


class Stage1Response(BaseModel):
    """Stage 1 prediction output."""

    predicted_allowed_amount: float = Field(..., description="Predicted Medicare allowed amount in dollars")
    provider_type: str
    state: str
    hcpcs_bucket: int
    hcpcs_bucket_name: str
    place_of_service: int


class Stage2Request(BaseModel):
    """Input for Stage 2 (out-of-pocket cost) prediction."""

    # Provider-side (same as Stage 1 or provided directly)
    provider_type: str = Field(..., description="Provider specialty name")
    state: str = Field(..., description="Two-letter state abbreviation")
    hcpcs_bucket: int = Field(4, ge=0, le=5)
    place_of_service: int = Field(0, ge=0, le=1)
    allowed_amount: float = Field(..., description="Stage 1 predicted allowed amount (or known value)")

    risk_score: float | None = Field(None)

    # Beneficiary demographics
    age: int = Field(70, ge=0, le=120, description="Beneficiary age")
    sex: int = Field(0, ge=0, le=1, description="0=Female, 1=Male")
    income: int = Field(1, ge=1, le=2, description="1=Below Median, 2=Above Median")
    chronic_count: int = Field(2, ge=0, le=20, description="Number of chronic conditions")
    dual_eligible: int = Field(0, ge=0, le=1, description="0=No, 1=Medicare+Medicaid")
    has_supplemental: int = Field(0, ge=0, le=1, description="0=No, 1=Has supplemental insurance")


class Stage2Response(BaseModel):
    """Stage 2 OOP prediction output."""

    oop_p10: float = Field(..., description="10th percentile OOP cost (best case)")
    oop_p50: float = Field(..., description="50th percentile OOP cost (typical)")
    oop_p90: float = Field(..., description="90th percentile OOP cost (high end)")
    census_region: int
    census_region_name: str


class FullPredictionRequest(BaseModel):
    """Combined Stage 1 + Stage 2 prediction input."""

    # Provider-side
    provider_type: str = Field(..., description="Provider specialty name")
    state: str = Field(..., description="Two-letter state abbreviation")
    hcpcs_code: str | None = Field(None, description="HCPCS/CPT code (optional)")
    hcpcs_bucket: int = Field(4, ge=0, le=5)
    place_of_service: int = Field(0, ge=0, le=1)

    # Optional continuous features
    risk_score: float | None = Field(None)
    total_services: float | None = Field(None)
    total_beneficiaries: float | None = Field(None)
    avg_submitted_charge: float | None = Field(None)

    # Beneficiary demographics (for Stage 2)
    age: int = Field(70, ge=0, le=120)
    sex: int = Field(0, ge=0, le=1)
    income: int = Field(1, ge=1, le=2)
    chronic_count: int = Field(2, ge=0, le=20)
    dual_eligible: int = Field(0, ge=0, le=1)
    has_supplemental: int = Field(0, ge=0, le=1)


class FullPredictionResponse(BaseModel):
    """Combined Stage 1 + Stage 2 output."""

    stage1: Stage1Response
    stage2: Stage2Response
