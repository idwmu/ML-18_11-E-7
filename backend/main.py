"""
PathSnap – System C FastAPI Application Entry Point
=====================================================
Starts the FastAPI application with:
  - Model loading at startup (once)
  - API router registration
  - CORS middleware for cross-origin access from Member D's frontend
  - Swagger UI at /docs
  - ReDoc at /redoc

Usage (from project root)
--------------------------
    uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload

Environment variables
---------------------
  PATHSNAP_MODEL_PATH   Path to a joblib-serialised scikit-learn model file.
                        If unset, DummyModel is used automatically.
  PATHSNAP_LOG_LEVEL    Logging level: DEBUG / INFO / WARNING (default: INFO).
"""

from __future__ import annotations

import logging
import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.api.routes import router
from backend.inference.model_loader import load_model

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

log_level = os.environ.get("PATHSNAP_LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, log_level, logging.INFO),
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("pathsnap.main")

# ---------------------------------------------------------------------------
# FastAPI application
# ---------------------------------------------------------------------------

app = FastAPI(
    title="PathSnap – System C: HMM Route Reconstruction API",
    description=(
        "**System C** of the PathSnap distributed map-matching pipeline.\n\n"
        "Receives a trip ID, runs Hidden Markov Model / Viterbi decoding "
        "over candidate road arcs, and returns the most likely route.\n\n"
        "- Member A provides trip data loading and candidate generation.\n"
        "- Member B provides the trained scoring model.\n"
        "- **This system (C)** performs HMM/Viterbi and exposes the REST API.\n"
        "- Member D consumes this API for visualisation.\n\n"
        "Set `PATHSNAP_MODEL_PATH` to load Member B's trained model; "
        "otherwise the built-in DummyModel is used."
    ),
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
)

# ---------------------------------------------------------------------------
# CORS – allow Member D's frontend (Streamlit / browser) to call this API
# ---------------------------------------------------------------------------

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],        # Tighten to Laptop 2 IP in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Startup: load model once into application state
# ---------------------------------------------------------------------------

@app.on_event("startup")
async def startup_event() -> None:
    model_path = os.environ.get("PATHSNAP_MODEL_PATH")
    app.state.model = load_model(model_path)
    log.info("Startup complete. Model: %s", app.state.model.name)


# ---------------------------------------------------------------------------
# Include router
# ---------------------------------------------------------------------------

app.include_router(router)


# ---------------------------------------------------------------------------
# Root redirect to docs
# ---------------------------------------------------------------------------

from fastapi.responses import RedirectResponse

@app.get("/", include_in_schema=False)
async def root() -> RedirectResponse:
    return RedirectResponse(url="/docs")
