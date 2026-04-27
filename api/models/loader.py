"""Load model artifacts at startup."""

import json
import os
from dataclasses import dataclass, field
from typing import Any

import lightgbm as lgb


@dataclass
class ModelArtifacts:
    """All model artifacts held in memory for inference."""

    lgbm_booster: lgb.Booster | None = None
    # CatBoost monotonic quantile models for Stage 2 OOP
    oop_p10: Any = None
    oop_p50: Any = None
    oop_p90: Any = None

    # CQR calibration constants (asymmetric split, V2_04 reproduction)
    # Apply at inference: p10' = max(0, pred_p10 - oop_q_lo); p90' = pred_p90 + oop_q_hi
    oop_q_lo: float = 0.0
    oop_q_hi: float = 0.0
    oop_calibration_meta: dict = field(default_factory=dict)

    # Encoding lookups
    label_encoders: dict[str, list[str]] = field(default_factory=dict)
    hcpcs_target_enc: dict[str, float] = field(default_factory=dict)
    hcpcs_global_mean: float = 76.34

    # Reverse lookup dicts (name -> index) for O(1) encoding
    specialty_to_idx: dict[str, int] = field(default_factory=dict)
    state_to_idx: dict[str, int] = field(default_factory=dict)
    hcpcs_to_idx: dict[str, int] = field(default_factory=dict)

    @property
    def stage1_ready(self) -> bool:
        return self.lgbm_booster is not None and bool(self.label_encoders)

    @property
    def stage2_ready(self) -> bool:
        return all([self.oop_p10, self.oop_p50, self.oop_p90])


def load_all_models(artifacts_dir: str) -> ModelArtifacts:
    """Load all model artifacts from disk. Returns a populated ModelArtifacts."""

    artifacts = ModelArtifacts()

    # -- Label encoders --
    enc_path = os.path.join(artifacts_dir, "label_encoders.json")
    if os.path.exists(enc_path):
        with open(enc_path, "r") as f:
            artifacts.label_encoders = json.load(f)
        # Build reverse lookups
        if "Rndrng_Prvdr_Type" in artifacts.label_encoders:
            artifacts.specialty_to_idx = {
                name: i for i, name in enumerate(artifacts.label_encoders["Rndrng_Prvdr_Type"])
            }
        if "Rndrng_Prvdr_State_Abrvtn" in artifacts.label_encoders:
            artifacts.state_to_idx = {
                name: i for i, name in enumerate(artifacts.label_encoders["Rndrng_Prvdr_State_Abrvtn"])
            }
        if "HCPCS_Cd" in artifacts.label_encoders:
            artifacts.hcpcs_to_idx = {
                code: i for i, code in enumerate(artifacts.label_encoders["HCPCS_Cd"])
            }
        print(f"  Loaded label encoders: {len(artifacts.specialty_to_idx)} specialties, "
              f"{len(artifacts.state_to_idx)} states, {len(artifacts.hcpcs_to_idx)} HCPCS codes")

    # -- HCPCS target encoding --
    te_path = os.path.join(artifacts_dir, "hcpcs_target_enc.json")
    if os.path.exists(te_path):
        with open(te_path, "r") as f:
            te_data = json.load(f)
        artifacts.hcpcs_global_mean = te_data.get("global_mean", 76.34)
        artifacts.hcpcs_target_enc = te_data.get("codes", {})
        print(f"  Loaded HCPCS target encoding: {len(artifacts.hcpcs_target_enc)} codes, "
              f"global_mean={artifacts.hcpcs_global_mean:.2f}")

    # -- LightGBM Stage 1 (prefer no-charge model for robustness) --
    for name in ("lgbm_v2_no_charge.txt", "lgbm_model.txt", "lgbm_v2_full.txt"):
        lgbm_path = os.path.join(artifacts_dir, name)
        if os.path.exists(lgbm_path):
            artifacts.lgbm_booster = lgb.Booster(model_file=lgbm_path)
            print(f"  Loaded LightGBM model from {name}: {artifacts.lgbm_booster.num_trees()} trees")
            break
    else:
        print(f"  WARNING: No LightGBM model found in {artifacts_dir}")

    # -- CatBoost Stage 2 OOP monotonic quantile models --
    try:
        from catboost import CatBoostRegressor
    except ImportError:
        print("  WARNING: catboost not installed — Stage 2 OOP unavailable")
        return artifacts

    for quantile, attr in [("p10", "oop_p10"), ("p50", "oop_p50"), ("p90", "oop_p90")]:
        path = os.path.join(artifacts_dir, f"oop_mono_{quantile}.cbm")
        if os.path.exists(path):
            model = CatBoostRegressor()
            model.load_model(path)
            setattr(artifacts, attr, model)
            print(f"  Loaded CatBoost OOP {quantile} from {path}")
        else:
            print(f"  WARNING: CatBoost OOP {quantile} not found at {path}")

    # -- CQR calibration constants --
    cal_path = os.path.join(artifacts_dir, "oop_calibration.json")
    if os.path.exists(cal_path):
        with open(cal_path, "r") as f:
            cal = json.load(f)
        asym = cal.get("asymmetric", {})
        artifacts.oop_q_lo = float(asym.get("q_lo", 0.0))
        artifacts.oop_q_hi = float(asym.get("q_hi", 0.0))
        artifacts.oop_calibration_meta = cal
        print(
            f"  Loaded OOP CQR calibration: q_lo={artifacts.oop_q_lo:+.4f}, "
            f"q_hi={artifacts.oop_q_hi:+.4f} "
            f"(test coverage {asym.get('test_coverage', 0):.1%})"
        )
    else:
        print(f"  WARNING: No oop_calibration.json — bands served UNCALIBRATED")

    return artifacts
