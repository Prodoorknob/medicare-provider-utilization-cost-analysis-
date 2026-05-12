"""Application settings loaded from environment variables."""

import os
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # Postgres connection string (Railway-provided DATABASE_URL)
    database_url: str = ""

    # CORS
    allowed_origins: str = "http://localhost:3000"

    # Model artifact paths (relative to api/ directory)
    artifacts_dir: str = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "models", "artifacts"
    )

    # Defaults for optional prediction inputs
    default_risk_score: float = 1.0          # CMS national median HCC risk
    default_total_services: float = 50.0
    default_total_beneficiaries: float = 20.0
    default_avg_submitted_charge: float = 200.0

    model_config = {"env_file": ".env", "extra": "ignore"}

    @property
    def origins_list(self) -> list[str]:
        return [o.strip() for o in self.allowed_origins.split(",") if o.strip()]


settings = Settings()
