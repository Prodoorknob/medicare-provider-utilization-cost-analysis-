"""Feature engineering and model inference for Stage 1 and Stage 2."""

import numpy as np
import pandas as pd

from config import settings
from models.loader import ModelArtifacts

# -- HCPCS bucket ranges (from notebooks/03_gold_features_local.py:100-128) --

HCPCS_BUCKET_NAMES = {
    0: "Anesthesia",
    1: "Surgery",
    2: "Radiology",
    3: "Lab/Pathology",
    4: "Medicine/E&M",
    5: "HCPCS Level II",
}

# State -> census region mapping (from generate_synthetic_mcbs.py + web constants)
STATE_TO_REGION: dict[str, int] = {
    "CT": 1, "ME": 1, "MA": 1, "NH": 1, "RI": 1, "VT": 1, "NJ": 1, "NY": 1, "PA": 1,
    "DE": 3, "FL": 3, "GA": 3, "MD": 3, "NC": 3, "SC": 3, "VA": 3, "DC": 3, "WV": 3,
    "AL": 3, "KY": 3, "MS": 3, "TN": 3, "AR": 3, "LA": 3, "OK": 3, "TX": 3,
    "IL": 2, "IN": 2, "MI": 2, "OH": 2, "WI": 2, "IA": 2, "KS": 2, "MN": 2, "MO": 2,
    "NE": 2, "ND": 2, "SD": 2,
    "AZ": 4, "CO": 4, "ID": 4, "MT": 4, "NV": 4, "NM": 4, "UT": 4, "WY": 4,
    "AK": 4, "CA": 4, "HI": 4, "OR": 4, "WA": 4,
}


def hcpcs_code_to_bucket(code: str) -> int:
    """Map a single HCPCS/CPT code to bucket 0-5.

    Replicates notebooks/03_gold_features_local.py:100-128.
    """
    code = code.strip()
    if not code:
        return 4  # default Medicine/E&M

    # Alpha-prefix -> HCPCS Level II
    if code[0].isalpha():
        return 5

    try:
        num = int(code)
    except ValueError:
        return 4  # fallback

    if 100 <= num <= 1999:
        return 0   # Anesthesia
    elif 10000 <= num <= 69999:
        return 1   # Surgery
    elif 70000 <= num <= 79999:
        return 2   # Radiology
    elif 80000 <= num <= 89999:
        return 3   # Lab/Pathology
    elif 90000 <= num <= 99999:
        return 4   # Medicine/E&M
    else:
        return 4   # fallback


def build_stage1_features(
    artifacts: ModelArtifacts,
    provider_type: str,
    state: str,
    hcpcs_code: str | None,
    hcpcs_bucket: int,
    place_of_service: int,
    risk_score: float | None,
    total_services: float | None,
    total_beneficiaries: float | None,
    avg_submitted_charge: float | None,
) -> np.ndarray:
    """Build the 13-feature vector for Stage 1 LightGBM.

    Feature order matches train_lgbm_local.py:50-57 exactly.
    """
    # Encode categoricals (fallback = len(classes) for unknown values)
    n_specialties = len(artifacts.specialty_to_idx)
    n_states = len(artifacts.state_to_idx)
    n_hcpcs = len(artifacts.hcpcs_to_idx)

    provider_type_idx = artifacts.specialty_to_idx.get(provider_type, n_specialties)
    state_idx = artifacts.state_to_idx.get(state, n_states)

    # HCPCS code encoding + bucket derivation
    if hcpcs_code is not None:
        hcpcs_idx = artifacts.hcpcs_to_idx.get(hcpcs_code.strip(), n_hcpcs)
        hcpcs_bucket = hcpcs_code_to_bucket(hcpcs_code)
    else:
        hcpcs_idx = n_hcpcs  # unknown code, use bucket only

    # Apply defaults for optional continuous features
    risk = risk_score if risk_score is not None else settings.default_risk_score
    srvcs = total_services if total_services is not None else settings.default_total_services
    benes = total_beneficiaries if total_beneficiaries is not None else settings.default_total_beneficiaries
    charge = avg_submitted_charge if avg_submitted_charge is not None else settings.default_avg_submitted_charge

    # Derived features
    log_srvcs = np.log1p(srvcs)
    log_benes = np.log1p(benes)
    srvcs_per_bene = log_srvcs / log_benes if log_benes > 0 else 0.0
    specialty_bucket = provider_type_idx * 6.0 + hcpcs_bucket
    pos_bucket = place_of_service * 6.0 + hcpcs_bucket

    # HCPCS target encoding
    hcpcs_te = artifacts.hcpcs_global_mean
    if hcpcs_code is not None:
        hcpcs_te = artifacts.hcpcs_target_enc.get(
            hcpcs_code.strip(), artifacts.hcpcs_global_mean
        )

    # Build feature vector — auto-detect if model includes Avg_Sbmtd_Chrg
    # The no-charge model is more robust (user won't know the charge)
    model_features = artifacts.lgbm_booster.feature_name() if artifacts.lgbm_booster else []
    has_charge = "Avg_Sbmtd_Chrg" in model_features

    features = [
        provider_type_idx,          # Rndrng_Prvdr_Type_idx
        state_idx,                  # Rndrng_Prvdr_State_Abrvtn_idx
        hcpcs_idx,                  # HCPCS_Cd_idx
        hcpcs_bucket,               # hcpcs_bucket
        place_of_service,           # place_of_srvc_flag
        risk,                       # Bene_Avg_Risk_Scre
        log_srvcs,                  # log_srvcs
        log_benes,                  # log_benes
    ]
    if has_charge:
        features.append(charge)     # Avg_Sbmtd_Chrg (only if model uses it)
    features.extend([
        srvcs_per_bene,             # srvcs_per_bene
        specialty_bucket,           # specialty_bucket
        pos_bucket,                 # pos_bucket
        hcpcs_te,                   # hcpcs_target_enc
    ])

    return np.array([features], dtype=np.float64)


def predict_stage1(
    artifacts: ModelArtifacts,
    provider_type: str,
    state: str,
    hcpcs_code: str | None,
    hcpcs_bucket: int,
    place_of_service: int,
    risk_score: float | None = None,
    total_services: float | None = None,
    total_beneficiaries: float | None = None,
    avg_submitted_charge: float | None = None,
) -> float:
    """Run Stage 1 LightGBM prediction. Returns predicted allowed amount in dollars."""
    if not artifacts.stage1_ready:
        raise RuntimeError("Stage 1 model not loaded")

    features = build_stage1_features(
        artifacts, provider_type, state, hcpcs_code, hcpcs_bucket,
        place_of_service, risk_score, total_services, total_beneficiaries,
        avg_submitted_charge,
    )

    # Model was trained on log1p(target) — invert with expm1
    log_pred = artifacts.lgbm_booster.predict(features)[0]
    pred = float(np.expm1(log_pred))
    return max(0.0, round(pred, 2))


# Stage 2 feature names and categorical indices (from V2_04 notebook)
OOP_FEATURE_NAMES = [
    "Avg_Mdcr_Alowd_Amt", "Bene_Avg_Risk_Scre", "Rndrng_Prvdr_Type_idx",
    "hcpcs_bucket", "place_of_srvc_flag", "census_region",
    "age", "sex", "income", "chronic_count", "dual_eligible", "has_supplemental",
]
OOP_CAT_IDX = [2, 3, 4, 5]  # specialty, bucket, pos, region


def build_stage2_features(
    artifacts: ModelArtifacts,
    allowed_amount: float,
    risk_score: float | None,
    provider_type: str,
    hcpcs_bucket: int,
    place_of_service: int,
    census_region: int,
    age: int,
    sex: int,
    income: int,
    chronic_count: int,
    dual_eligible: int,
    has_supplemental: int,
) -> pd.DataFrame:
    """Build the 12-feature DataFrame for Stage 2 CatBoost OOP.

    Feature order matches V2_04 OOP_FEATURES exactly.
    CatBoost needs categorical columns cast to int.
    """
    n_specialties = len(artifacts.specialty_to_idx)
    provider_type_idx = artifacts.specialty_to_idx.get(provider_type, n_specialties)
    risk = risk_score if risk_score is not None else settings.default_risk_score

    row = {
        "Avg_Mdcr_Alowd_Amt": float(allowed_amount),
        "Bene_Avg_Risk_Scre": float(risk),
        "Rndrng_Prvdr_Type_idx": int(provider_type_idx),
        "hcpcs_bucket": int(hcpcs_bucket),
        "place_of_srvc_flag": int(place_of_service),
        "census_region": int(census_region),
        "age": int(age),
        "sex": int(sex),
        "income": int(income),
        "chronic_count": int(chronic_count),
        "dual_eligible": int(dual_eligible),
        "has_supplemental": int(has_supplemental),
    }
    return pd.DataFrame([row], columns=OOP_FEATURE_NAMES)


def predict_stage2(
    artifacts: ModelArtifacts,
    allowed_amount: float,
    risk_score: float | None,
    provider_type: str,
    hcpcs_bucket: int,
    place_of_service: int,
    state: str,
    age: int = 70,
    sex: int = 0,
    income: int = 1,
    chronic_count: int = 2,
    dual_eligible: int = 0,
    has_supplemental: int = 0,
) -> tuple[float, float, float, int]:
    """Run Stage 2 CatBoost monotonic quantile prediction.

    Returns (p10, p50, p90, census_region) with non-crossing enforcement.
    OOP model was trained on raw dollars — no inverse transform needed.
    """
    if not artifacts.stage2_ready:
        raise RuntimeError("Stage 2 OOP models not loaded")

    census_region = STATE_TO_REGION.get(state.upper(), 3)  # default South

    features = build_stage2_features(
        artifacts, allowed_amount, risk_score, provider_type,
        hcpcs_bucket, place_of_service, census_region,
        age, sex, income, chronic_count, dual_eligible, has_supplemental,
    )

    from catboost import Pool
    pool = Pool(features, cat_features=OOP_CAT_IDX, feature_names=OOP_FEATURE_NAMES)
    p10 = float(artifacts.oop_p10.predict(pool)[0])
    p50 = float(artifacts.oop_p50.predict(pool)[0])
    p90 = float(artifacts.oop_p90.predict(pool)[0])

    # Floor at 0 and enforce non-crossing
    p10 = max(0.0, round(p10, 2))
    p50 = max(p10, round(p50, 2))
    p90 = max(p50, round(p90, 2))

    return p10, p50, p90, census_region
