"""Prediction endpoints for Stage 1, Stage 2, and full pipeline."""

from fastapi import APIRouter, HTTPException, Request

from schemas.predict import (
    FullPredictionRequest,
    FullPredictionResponse,
    Stage1Request,
    Stage1Response,
    Stage2Request,
    Stage2Response,
)
from services.prediction import (
    HCPCS_BUCKET_NAMES,
    STATE_TO_REGION,
    predict_stage1,
    predict_stage2,
)

router = APIRouter(prefix="/predict", tags=["predict"])

CENSUS_REGION_NAMES = {1: "Northeast", 2: "Midwest", 3: "South", 4: "West"}


@router.post("/stage1", response_model=Stage1Response)
async def stage1(req: Stage1Request, request: Request):
    """Real-time Stage 1 prediction: Medicare allowed amount."""
    artifacts = request.app.state.models
    if not artifacts.stage1_ready:
        raise HTTPException(503, "Stage 1 model not loaded")

    try:
        predicted = predict_stage1(
            artifacts,
            provider_type=req.provider_type,
            state=req.state,
            hcpcs_code=req.hcpcs_code,
            hcpcs_bucket=req.hcpcs_bucket,
            place_of_service=req.place_of_service,
            risk_score=req.risk_score,
            total_services=req.total_services,
            total_beneficiaries=req.total_beneficiaries,
            avg_submitted_charge=req.avg_submitted_charge,
        )
    except Exception as e:
        raise HTTPException(500, f"Stage 1 prediction failed: {e}")

    # Resolve the actual bucket used (may differ if hcpcs_code was provided)
    bucket = req.hcpcs_bucket
    if req.hcpcs_code is not None:
        from services.prediction import hcpcs_code_to_bucket
        bucket = hcpcs_code_to_bucket(req.hcpcs_code)

    return Stage1Response(
        predicted_allowed_amount=predicted,
        provider_type=req.provider_type,
        state=req.state,
        hcpcs_bucket=bucket,
        hcpcs_bucket_name=HCPCS_BUCKET_NAMES.get(bucket, "Unknown"),
        place_of_service=req.place_of_service,
    )


@router.post("/stage2", response_model=Stage2Response)
async def stage2(req: Stage2Request, request: Request):
    """Real-time Stage 2 prediction: out-of-pocket costs (P10/P50/P90)."""
    artifacts = request.app.state.models
    if not artifacts.stage2_ready:
        raise HTTPException(503, "Stage 2 OOP models not loaded")

    try:
        p10, p50, p90, region = predict_stage2(
            artifacts,
            allowed_amount=req.allowed_amount,
            risk_score=req.risk_score,
            provider_type=req.provider_type,
            hcpcs_bucket=req.hcpcs_bucket,
            place_of_service=req.place_of_service,
            state=req.state,
            age=req.age,
            sex=req.sex,
            income=req.income,
            chronic_count=req.chronic_count,
            dual_eligible=req.dual_eligible,
            has_supplemental=req.has_supplemental,
        )
    except Exception as e:
        raise HTTPException(500, f"Stage 2 prediction failed: {e}")

    return Stage2Response(
        oop_p10=p10,
        oop_p50=p50,
        oop_p90=p90,
        census_region=region,
        census_region_name=CENSUS_REGION_NAMES.get(region, "Unknown"),
    )


@router.post("/full", response_model=FullPredictionResponse)
async def full(req: FullPredictionRequest, request: Request):
    """Full two-stage prediction: allowed amount -> OOP costs."""
    artifacts = request.app.state.models
    if not artifacts.stage1_ready:
        raise HTTPException(503, "Stage 1 model not loaded")

    # Stage 1
    try:
        allowed = predict_stage1(
            artifacts,
            provider_type=req.provider_type,
            state=req.state,
            hcpcs_code=req.hcpcs_code,
            hcpcs_bucket=req.hcpcs_bucket,
            place_of_service=req.place_of_service,
            risk_score=req.risk_score,
            total_services=req.total_services,
            total_beneficiaries=req.total_beneficiaries,
            avg_submitted_charge=req.avg_submitted_charge,
        )
    except Exception as e:
        raise HTTPException(500, f"Stage 1 prediction failed: {e}")

    bucket = req.hcpcs_bucket
    if req.hcpcs_code is not None:
        from services.prediction import hcpcs_code_to_bucket
        bucket = hcpcs_code_to_bucket(req.hcpcs_code)

    s1 = Stage1Response(
        predicted_allowed_amount=allowed,
        provider_type=req.provider_type,
        state=req.state,
        hcpcs_bucket=bucket,
        hcpcs_bucket_name=HCPCS_BUCKET_NAMES.get(bucket, "Unknown"),
        place_of_service=req.place_of_service,
    )

    # Stage 2 (uses Stage 1 output as input)
    if artifacts.stage2_ready:
        try:
            p10, p50, p90, region = predict_stage2(
                artifacts,
                allowed_amount=allowed,
                risk_score=req.risk_score,
                provider_type=req.provider_type,
                hcpcs_bucket=bucket,
                place_of_service=req.place_of_service,
                state=req.state,
                age=req.age,
                sex=req.sex,
                income=req.income,
                chronic_count=req.chronic_count,
                dual_eligible=req.dual_eligible,
                has_supplemental=req.has_supplemental,
            )
        except Exception as e:
            raise HTTPException(500, f"Stage 2 prediction failed: {e}")
    else:
        # Stage 2 not available — return zeros
        p10, p50, p90 = 0.0, 0.0, 0.0
        region = STATE_TO_REGION.get(req.state.upper(), 3)

    s2 = Stage2Response(
        oop_p10=p10,
        oop_p50=p50,
        oop_p90=p90,
        census_region=region,
        census_region_name=CENSUS_REGION_NAMES.get(region, "Unknown"),
    )

    return FullPredictionResponse(stage1=s1, stage2=s2)
