"""
PathSnap – System C API Schemas
================================
Pydantic models for /predict, /trips and /health endpoints.

Coordinate with Member D for final field names before freezing.
"""

from __future__ import annotations

from typing import Optional
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Request
# ---------------------------------------------------------------------------

class PredictRequest(BaseModel):
    """
    POST /predict request body.

    Fields
    ------
    trip_id : str
        8-digit zero-padded trip folder name, e.g. "00000001".
        Must correspond to a trip present in the cleaned dataset.
    """
    trip_id: str = Field(
        ...,
        description="8-digit zero-padded trip ID, e.g. '00000001'",
        min_length=1,
        max_length=20,
        examples=["00000001"],
    )

    model_config = {"json_schema_extra": {"example": {"trip_id": "00000001"}}}


# ---------------------------------------------------------------------------
# Response
# ---------------------------------------------------------------------------

class PredictResponse(BaseModel):
    """
    POST /predict response body.

    Fields
    ------
    trip_id            : str            – echoed from request for correlation.
    predicted_arc_ids  : list[int]      – ordered sequence of arc_ids (0-based
                                         row indices in the trip's .arcs file)
                                         chosen by Viterbi as the most likely route.
    points             : list[list[float]]
                                        – GPS observations used (lon, lat pairs),
                                         in the same temporal order as arc_ids.
    n_gps_points       : int            – total GPS observations processed.
    n_route_arcs       : int            – length of the predicted arc sequence.
    candidate_miss_rate: float          – fraction of GPS points for which the
                                         Viterbi-chosen arc was not in the
                                         candidate set (0.0 = perfect coverage).
    model_name         : str            – identifier of the scoring model used.
    """
    trip_id:             str              = Field(..., description="Echoed trip ID")
    predicted_arc_ids:   list[int]        = Field(..., description="Ordered arc ID sequence")
    points:              list[list[float]]= Field(..., description="GPS observations [[lon, lat], …]")
    n_gps_points:        int              = Field(..., description="Number of GPS points processed")
    n_route_arcs:        int              = Field(..., description="Length of predicted route")
    candidate_miss_rate: float            = Field(..., description="Fraction of points with no candidate")
    model_name:          str              = Field(..., description="Scoring model identifier")
    true_candidate_miss_rate: Optional[float] = Field(default=None, description="Fraction of GPS points whose true arc was absent from the candidate set (needs reference route; None otherwise)")
    route_f1:        Optional[float] = Field(default=None, description="Route-arc F1 vs reference route (None if no ground truth)")
    route_precision: Optional[float] = Field(default=None, description="Route-arc precision vs reference route")
    route_recall:    Optional[float] = Field(default=None, description="Route-arc recall vs reference route")

    model_config = {
        "json_schema_extra": {
            "example": {
                "trip_id":             "00000001",
                "predicted_arc_ids":   [2641, 64973, 133948, 133949],
                "points":              [[8.5952, 49.8441], [8.5953, 49.8445]],
                "n_gps_points":        4,
                "n_route_arcs":        4,
                "candidate_miss_rate": 0.0,
                "model_name":          "DummyModel-v0",
            }
        }
    }


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------

class HealthResponse(BaseModel):
    """GET /health response body."""
    status:     str = Field(..., description="'ok' when service is healthy")
    model_name: str = Field(..., description="Currently loaded model identifier")
    version:    str = Field(default="0.1.0", description="API version string")


# ---------------------------------------------------------------------------
# Error
# ---------------------------------------------------------------------------

class ErrorResponse(BaseModel):
    """Standard error envelope returned on 4xx / 5xx responses."""
    detail: str = Field(..., description="Human-readable error description")


# ---------------------------------------------------------------------------
# Trips listing
# ---------------------------------------------------------------------------

class TripsResponse(BaseModel):
    """
    GET /trips response body.

    Fields
    ------
    student_test : list[str] – the 10 evaluator trip IDs (primary targets).
    train        : list[str] – the 80 training trip IDs (available for testing).
    """
    student_test: list[str] = Field(
        ...,
        description="Evaluator (student_test) trip IDs",
        examples=[["00000000", "00000007"]],
    )
    train: list[str] = Field(
        ...,
        description="Training trip IDs",
        examples=[["00000001", "00000002"]],
    )
