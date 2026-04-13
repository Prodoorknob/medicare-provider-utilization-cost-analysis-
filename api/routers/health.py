"""Health check endpoint."""

from fastapi import APIRouter, Request

router = APIRouter(tags=["health"])


@router.get("/health")
async def health(request: Request):
    models = request.app.state.models
    return {
        "status": "ok",
        "version": "1.0.0",
        "models": {
            "stage1_lgbm": models.stage1_ready,
            "stage2_oop": models.stage2_ready,
            "label_encoders": bool(models.label_encoders),
            "specialties": len(models.specialty_to_idx),
            "states": len(models.state_to_idx),
            "hcpcs_codes": len(models.hcpcs_to_idx),
        },
    }
