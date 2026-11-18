"""
PathSnap – Prediction Pipeline
================================
Orchestrates the full route-reconstruction pipeline for one trip.

Data flow (real pipeline — stubs removed)
-----------------------------------------
  trip_id
    → data_processing.loaders      → raw track, nodes, arcs DataFrames
    → data_processing.cleaning     → cleaned track (outlier / gap handling)
    → data_processing.candidates   → RoadNetwork, k-nearest arc candidates
                                     (UTM-projected, exact point-to-segment dist)
    → data_processing.features     → 9-column feature DataFrame per candidate row
    → model.predict_proba()        → P(label=1) per candidate row
    → backend.hmm.emissions        → log-emission probabilities per GPS point
    → backend.hmm.transitions      → log-transition probabilities (road adjacency)
    → backend.hmm.viterbi          → most likely arc sequence
    → PredictResponse              → returned by FastAPI

Import strategy for data_processing
-------------------------------------
Member A's modules live in data_processing/ and use bare sibling imports
(``from loaders import …``, ``from geo import …`` etc.).  They were written
to be run from that directory.  We add data_processing/ to sys.path once,
at module load time, so the imports resolve correctly without touching any of
Member A's files.

loaders.py resolves DATA_ROOT = Path("datasets") relative to CWD.
Uvicorn is always started from the project root, so this works as-is.
"""

from __future__ import annotations

import logging
import pathlib
import sys
from typing import Optional

# ---------------------------------------------------------------------------
# Inject data_processing/ onto sys.path so Member A's bare imports resolve.
# This must happen before any data_processing import.
# ---------------------------------------------------------------------------
_THIS_FILE    = pathlib.Path(__file__).resolve()
_BACKEND_DIR  = _THIS_FILE.parent.parent          # PROJECT/backend/
_PROJECT_ROOT = _BACKEND_DIR.parent               # PROJECT/
_DATA_PROC    = _PROJECT_ROOT / "data_processing"
DATASET_ROOT  = _PROJECT_ROOT / "datasets"

if str(_DATA_PROC) not in sys.path:
    sys.path.insert(0, str(_DATA_PROC))

# ---------------------------------------------------------------------------
# Member A's data-processing modules (imported after sys.path is set)
# ---------------------------------------------------------------------------
from loaders   import load_track, load_nodes, load_arcs, list_trip_ids  # noqa: E402
from cleaning  import clean_track                                          # noqa: E402
from features  import build_trip_features, FEATURE_COLUMNS                # noqa: E402

# ---------------------------------------------------------------------------
# System C HMM / Viterbi modules
# ---------------------------------------------------------------------------
from backend.hmm.emissions   import build_emission_table                   # noqa: E402
from backend.hmm.transitions import (                                       # noqa: E402
    build_adjacency,
    transition_log_prob,
    GPS_GAP_THRESHOLD_S,
    ROUTE_SIGMA_M,
)
from backend.hmm.transitions import haversine_m                            # noqa: E402
from backend.hmm.viterbi     import viterbi_decode, ViterbiResult          # noqa: E402
from backend.inference.model_loader import BaseModel                       # noqa: E402

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Trip discovery helpers
# ---------------------------------------------------------------------------

def list_evaluator_trips() -> list[str]:
    """Return sorted list of student_test trip IDs (the 10 evaluator trips)."""
    stest_dir = DATASET_ROOT / "student_test"
    if not stest_dir.exists():
        return []
    return sorted(p.name for p in stest_dir.iterdir() if p.is_dir())


def list_all_trips() -> dict[str, list[str]]:
    """Return dict with 'train' and 'student_test' trip ID lists."""
    return {
        "train":        list_trip_ids("train"),
        "student_test": list_evaluator_trips(),
    }


def _find_partition(trip_id: str) -> str:
    """Return the partition ('train' or 'student_test') for a trip, or raise."""
    for partition in ("student_test", "train"):
        if (DATASET_ROOT / partition / trip_id).exists():
            return partition
    raise FileNotFoundError(
        f"Trip '{trip_id}' not found in datasets/train/ or datasets/student_test/."
    )


# ---------------------------------------------------------------------------
# Core trip data loader (uses Member A's modules end-to-end)
# ---------------------------------------------------------------------------

def _load_trip(trip_id: str) -> dict:
    """
    Load a trip using Member A's data_processing pipeline.

    Returns
    -------
    dict with keys:
        track       : list[(lon, lat, ts)]  – cleaned GPS track
        arcs_list   : list[(from_node, to_node)]  – for transitions.build_adjacency
        feature_df  : pd.DataFrame  – per-candidate feature rows (no label columns)
    """
    partition = _find_partition(trip_id)

    # --- Member A's loaders ---
    track_raw  = load_track(trip_id, partition)   # DataFrame[lon, lat, t, point_idx]
    nodes_df   = load_nodes(trip_id, partition)   # DataFrame[lon, lat]
    arcs_df    = load_arcs(trip_id, partition)    # DataFrame[from_node, to_node]

    # --- Member A's cleaning ---
    track_clean = clean_track(track_raw, nodes_df)  # drops outliers, flags gaps

    if track_clean.empty:
        raise ValueError(f"Trip {trip_id}: GPS track is empty after cleaning.")

    # --- Member A's candidate generation + feature extraction ---
    # build_trip_features re-runs the full pipeline internally.
    # We call it here with route=None implicitly (the function only reads .route
    # when the file exists; for student_test it won't, so no label columns are added).
    # For train trips .route exists but we do NOT use label for prediction.
    feat_df = build_trip_features(trip_id, partition)

    # Drop label / candidate_miss if present (must not be used during prediction)
    for col in ("label", "candidate_miss"):
        if col in feat_df.columns:
            feat_df = feat_df.drop(columns=[col])

    # Build arcs list as (from_node, to_node) tuples for transitions.build_adjacency.
    # B5 fix: Do NOT build RoadNetwork here — build_trip_features already does
    # it internally.  Rebuilding would waste time on UTM projection + KD-tree.
    arcs_list = list(zip(arcs_df["from_node"].tolist(), arcs_df["to_node"].tolist()))

    # Build GPS track as list[(lon, lat, ts)] from the CLEANED track.
    # Uses the external clean_track call (same deterministic result as the
    # internal one inside build_trip_features).
    track_tuples = [
        (row.lon, row.lat, row.t)
        for row in track_clean.itertuples(index=False)
    ]

    # NOTE (B5 fix): We do NOT build RoadNetwork here.  build_trip_features
    # already constructs it internally.  Building it again would waste
    # significant time on UTM projection + KD-tree construction with no benefit.

    return {
        "track":      track_tuples,
        "arcs_list":  arcs_list,
        "feature_df": feat_df,
    }


# ---------------------------------------------------------------------------
# Score candidates using the real model
# ---------------------------------------------------------------------------

def _score_candidates(
    feature_df,
    model: BaseModel,
) -> list[Optional[dict[int, float]]]:
    """
    Run model.predict_proba on all candidate rows in one batch.

    Uses the exact contract from train_final.py:
        p = b["model"].predict_proba(df[b["feature_cols"]])[:, 1]

    For PathSnapModel this is handled inside predict_proba() itself
    (the bundle's feature_cols drives column selection).
    For DummyModel it uses dist_m as a fallback.

    Parameters
    ----------
    feature_df : pd.DataFrame  – candidate feature rows from build_trip_features
    model      : BaseModel

    Returns
    -------
    list[Optional[dict[int, float]]]
        One dict per GPS point (indexed by point_idx 0..n-1): arc_id → score.
        None means no candidates exist for that point.
    """
    if feature_df.empty:
        return []

    # Convert each row to a dict for the model interface
    flat_feats    = feature_df.to_dict(orient="records")
    flat_scores   = model.predict_proba(flat_feats)

    # Determine the total number of GPS points
    n_points = int(feature_df["point_idx"].max()) + 1
    scores_by_point: list[Optional[dict[int, float]]] = [None] * n_points

    for feat, score in zip(flat_feats, flat_scores):
        pidx   = int(feat["point_idx"])
        arc_id = int(feat["arc_id"])
        if scores_by_point[pidx] is None:
            scores_by_point[pidx] = {}
        scores_by_point[pidx][arc_id] = float(score)  # type: ignore[index]

    return scores_by_point


# ---------------------------------------------------------------------------
# Candidates-by-point list (for emissions.build_emission_table)
# ---------------------------------------------------------------------------

def _build_candidates_by_point(
    feature_df,
) -> list[list[dict]]:
    """
    Convert the feature DataFrame into the per-point candidate list expected
    by backend.hmm.emissions.build_emission_table.

    Returns
    -------
    list[list[dict]]
        One inner list per GPS point; each dict has keys: arc_id, dist_m.
    """
    n_points = int(feature_df["point_idx"].max()) + 1
    candidates: list[list[dict]] = [[] for _ in range(n_points)]

    for row in feature_df.itertuples(index=False):
        pidx = int(row.point_idx)
        candidates[pidx].append({
            "arc_id": int(row.arc_id),
            "dist_m": float(row.dist_m),
        })

    return candidates


# ---------------------------------------------------------------------------
# Transition function factory
# ---------------------------------------------------------------------------

def _make_transition_fn(
    adjacency: dict[int, set[int]],
    sigma: float = ROUTE_SIGMA_M,
    gap_threshold_s: float = GPS_GAP_THRESHOLD_S,
):
    """
    Return a closure with signature:
        log_p = fn(arc_i, arc_j, t_prev, t_curr)
    where t_prev / t_curr are (lon, lat, ts) tuples.
    """
    def _fn(arc_i: int, arc_j: int,
            t_prev: tuple, t_curr: tuple) -> float:
        lon1, lat1, ts1 = t_prev
        lon2, lat2, ts2 = t_curr
        gps_dist_m = haversine_m(lon1, lat1, lon2, lat2)
        dt = abs(ts2 - ts1)
        return transition_log_prob(
            arc_i, arc_j, adjacency,
            gps_distance_m=gps_dist_m,
            dt_seconds=dt,
            sigma=sigma,
            gap_threshold_s=gap_threshold_s,
        )
    return _fn


# ---------------------------------------------------------------------------
# Main prediction pipeline
# ---------------------------------------------------------------------------

def predict_route(trip_id: str, model: BaseModel) -> dict:
    """
    Full route prediction for one trip.

    Steps
    -----
    1. Load raw trip data with Member A's loaders.
    2. Clean track with Member A's cleaning module.
    3. Generate candidates + compute all 9 features with Member A's pipeline.
    4. Score every (point, candidate) row with the real model in one batch.
    5. Build log-emission table from model scores (blended with distance prior).
    6. Build directed road-graph adjacency from arc from_node/to_node pairs.
    7. Run Viterbi DP in log space.
    8. Return ordered arc ID sequence.

    Parameters
    ----------
    trip_id : str – 8-digit zero-padded trip ID
    model   : BaseModel – PathSnapModel or DummyModel

    Returns
    -------
    dict matching PredictResponse field names.

    Raises
    ------
    FileNotFoundError  – trip not found in datasets/
    ValueError         – track empty after cleaning
    RuntimeError       – Viterbi produced no valid path
    """
    log.info("predict_route: trip_id=%s  model=%s", trip_id, model.name)

    # --- Step 1–3: Load + clean + features ---
    trip = _load_trip(trip_id)

    track      = trip["track"]       # list[(lon, lat, ts)]
    arcs_list  = trip["arcs_list"]   # list[(from_node, to_node)]
    feature_df = trip["feature_df"]  # pd.DataFrame

    if not track:
        raise ValueError(f"Trip {trip_id}: GPS track is empty after cleaning.")

    # --- Step 4: Score candidates ---
    model_scores_by_point = _score_candidates(feature_df, model)

    # --- Step 5: Build emission table ---
    candidates_by_point = _build_candidates_by_point(feature_df)

    # Pad model_scores to match candidate list length if needed
    n_cand_pts = len(candidates_by_point)
    if len(model_scores_by_point) < n_cand_pts:
        model_scores_by_point.extend([None] * (n_cand_pts - len(model_scores_by_point)))

    emission_table = build_emission_table(
        candidates_by_point=candidates_by_point,
        model_scores_by_point=model_scores_by_point,
    )

    # Align emission_table length with track length
    # (track has the GPS observations; some may have no candidates)
    n_track = len(track)
    if len(emission_table) < n_track:
        emission_table.extend([{}] * (n_track - len(emission_table)))
    elif len(emission_table) > n_track:
        emission_table = emission_table[:n_track]

    # --- Step 6: Road-network adjacency ---
    adjacency = build_adjacency(arcs_list)

    # --- Step 7: Transition function + Viterbi ---
    transition_fn = _make_transition_fn(adjacency)

    result: ViterbiResult = viterbi_decode(
        emission_table=emission_table,
        transition_fn=transition_fn,
        gps_track=track,
    )

    if not result.success:
        raise RuntimeError(
            f"Viterbi failed for trip {trip_id}: {result.message}"
        )

    # --- Step 8: Format response ---
    points = [[lon, lat] for lon, lat, _ts in track]

    return {
        "trip_id":             trip_id,
        "predicted_arc_ids":   result.arc_sequence,
        "points":              points,
        "n_gps_points":        result.n_gps_points,
        "n_route_arcs":        len(result.arc_sequence),
        "candidate_miss_rate": result.candidate_miss_rate,
        "model_name":          model.name,
    }
