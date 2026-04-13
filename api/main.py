"""Medicare Cost Prediction API — FastAPI application."""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from config import settings
from models.loader import ModelArtifacts, load_all_models
from routers import forecast, health, predict, reference


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model artifacts at startup, release at shutdown."""
    print("Loading model artifacts...")
    app.state.models = load_all_models(settings.artifacts_dir)
    print(f"Stage 1 ready: {app.state.models.stage1_ready}")
    print(f"Stage 2 ready: {app.state.models.stage2_ready}")
    yield
    # Cleanup (Python GC handles the rest)
    app.state.models = ModelArtifacts()
    print("Models unloaded.")


app = FastAPI(
    title="Medicare Cost Prediction API",
    description="Real-time Medicare allowed amount and out-of-pocket cost predictions.",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.origins_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Routers
app.include_router(health.router)
app.include_router(predict.router)
app.include_router(forecast.router)
app.include_router(reference.router)
