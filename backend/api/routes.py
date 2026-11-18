"""
PathSnap – API Routes
=======================
FastAPI router defining all HTTP endpoints for System C.

Endpoints
---------
  GET  /health    – Liveness check; confirms the model is loaded.
  GET  /trips     – List all available trip IDs (evaluator + train).
  POST /predict   – Submit a trip ID and receive the predicted route.

The router reads the model instance from application state so it is
loaded exactly once at startup (see main.py).
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Request, HTTPException, status
from fastapi.responses import JSONResponse

from backend.schemas.request_response import (
    PredictRequest,
    PredictResponse,
    HealthResponse,
    TripsResponse,
    ErrorResponse,
)
from backend.inference.pipeline import predict_route, list_all_trips

log = logging.getLogger(__name__)

router = APIRouter()


# ---------------------------------------------------------------------------
# GET /health
# ---------------------------------------------------------------------------

@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Liveness / readiness check",
    tags=["System"],
    responses={
        200: {"description": "Service is healthy and model is loaded"},
    },
)
async def health(request: Request) -> HealthResponse:
    """
    Returns service status and the name of the currently loaded model.

    Use this to confirm that the backend is running and the model loaded
    successfully before sending prediction requests.
    """
    model = request.app.state.model
    return HealthResponse(
        status="ok",
        model_name=model.name,
        version="0.1.0",
    )


# ---------------------------------------------------------------------------
# GET /trips
# ---------------------------------------------------------------------------

@router.get(
    "/trips",
    response_model=TripsResponse,
    summary="List all available trip IDs",
    tags=["System"],
    responses={
        200: {"description": "Trip ID lists returned successfully"},
    },
)
async def trips() -> TripsResponse:
    """
    Returns the actual trip IDs present in the dataset.

    - **student_test**: the 10 evaluator trips (the ones submitted for scoring).
    - **train**: the 80 training trips (available for development testing).

    Use a `student_test` trip ID with `POST /predict` for the official submission.
    """
    trip_map = list_all_trips()
    return TripsResponse(
        student_test=trip_map.get("student_test", []),
        train=trip_map.get("train", []),
    )


# ---------------------------------------------------------------------------
# POST /predict
# ---------------------------------------------------------------------------

@router.post(
    "/predict",
    response_model=PredictResponse,
    summary="Predict the most likely route for a trip",
    tags=["Prediction"],
    responses={
        200: {"description": "Route predicted successfully"},
        400: {
            "model": ErrorResponse,
            "description": "Invalid trip_id or malformed request",
        },
        404: {
            "model": ErrorResponse,
            "description": "Trip not found in the dataset",
        },
        500: {
            "model": ErrorResponse,
            "description": "Internal prediction error",
        },
    },
)
async def predict(
    body: PredictRequest,
    request: Request,
) -> PredictResponse:
    """
    Run the HMM/Viterbi route reconstruction pipeline for the given trip.

    - **trip_id**: 8-digit zero-padded string identifier, e.g. `"00000001"`.

    The response contains:
    - `predicted_arc_ids` – ordered arc ID sequence (0-based, per-trip indices)
    - `points` – the GPS observations used `[[lon, lat], …]`
    - `n_gps_points` – how many GPS observations were processed
    - `n_route_arcs` – length of the predicted route
    - `candidate_miss_rate` – fraction of GPS points with no candidates found
    - `model_name` – which model produced the scores

    **Note for Member D:** arc IDs are 0-based row indices into the trip's
    `.arcs` file and are only meaningful per-trip. Pair them with `trip_id`
    for any cross-reference.
    """
    trip_id = body.trip_id.strip()

    if not trip_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="trip_id must not be empty.",
        )

    model = request.app.state.model

    try:
        result = predict_route(trip_id, model)
    except FileNotFoundError as exc:
        log.warning("predict: trip not found: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        )
    except ValueError as exc:
        log.warning("predict: invalid trip data: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        )
    except RuntimeError as exc:
        log.error("predict: pipeline error for trip %s: %s", trip_id, exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Route reconstruction failed: {exc}",
        )
    except Exception as exc:
        log.exception("predict: unexpected error for trip %s", trip_id)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Unexpected error: {type(exc).__name__}: {exc}",
        )

    return PredictResponse(**result)
