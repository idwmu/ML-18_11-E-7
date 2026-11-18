"""
PathSnap – Model Loader
=========================
Separates model loading logic from Viterbi and API code.

This module defines the abstract interface that any scoring model must
implement, and provides:

  1. DummyModel      – deterministic stand-in for testing (preserved).
  2. PathSnapModel   – loads models/model.pkl (a joblib bundle dict) produced
                       by models/train_final.py.
  3. load_model()    – factory that picks the right implementation.

Bundle format (from train_final.py)
------------------------------------
  joblib.dump({"model": sklearn_model, "feature_cols": feat_list}, path)

Loading it:
  bundle = joblib.load(path)
  clf    = bundle["model"]          # HistGradientBoostingClassifier
  cols   = bundle["feature_cols"]   # list[str] – exact trained order

Feature columns (self-described by the bundle, do NOT hardcode):
  dist_m, bearing_diff_deg, arc_length_m, speed_mps, heading_deg,
  arc_degree, proj_t, dist_start_node_m, dist_end_node_m
  (9 columns; exact list lives inside the .pkl)

Usage (from train_final.py's own comment):
  b = joblib.load("models/model.pkl")
  p = b["model"].predict_proba(df[b["feature_cols"]])[:, 1]

Default model path
------------------
  models/model.pkl  (relative to project root, resolved at runtime)
  Override with env var: PATHSNAP_MODEL_PATH=/absolute/path/model.pkl
"""

from __future__ import annotations

import logging
import os
import pathlib
from abc import ABC, abstractmethod
from typing import Optional

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Project root – used to resolve the default model path
# ---------------------------------------------------------------------------
_THIS_FILE   = pathlib.Path(__file__).resolve()
_PROJECT_ROOT = _THIS_FILE.parent.parent.parent   # backend/inference/model_loader.py → PROJECT/
DEFAULT_MODEL_PATH = _PROJECT_ROOT / "models" / "model.pkl"


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------

class BaseModel(ABC):
    """Abstract scoring model interface."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable model identifier."""

    @abstractmethod
    def predict_proba(self, features: list[dict]) -> list[float]:
        """
        Score a list of (GPS point, candidate arc) feature dicts.

        Parameters
        ----------
        features : list[dict]
            Each dict contains the feature columns produced by
            data_processing/features.py:
              dist_m               (float, metres)
              bearing_diff_deg     (float, degrees [0,180])
              arc_length_m         (float, metres)
              speed_mps            (float, m/s)
              heading_deg          (float, degrees [0,360))
              arc_degree           (int)
              proj_t               (float, [0,1])
              dist_start_node_m    (float, metres)
              dist_end_node_m      (float, metres)

        Returns
        -------
        list[float]
            P(label=1) scores in [0, 1], same length as features.
            Higher score = more likely to be the correct arc for this GPS point.
        """


# ---------------------------------------------------------------------------
# DummyModel – preserved for testing / fallback
# ---------------------------------------------------------------------------

class DummyModel(BaseModel):
    """
    Deterministic dummy model for development and testing.

    Scoring logic: score = 1 / (1 + dist_m)
    Uses 'dist_m' (Member A's actual key name) with fallback to the old
    'point_to_arc_distance' key so it still works in unit-test contexts.
    """

    @property
    def name(self) -> str:
        return "DummyModel-v0"

    def predict_proba(self, features: list[dict]) -> list[float]:
        scores: list[float] = []
        for feat in features:
            # Accept both old and new key names for backwards compat with tests
            dist_m = float(
                feat.get("dist_m",
                feat.get("point_to_arc_distance", 1e6))
            )
            scores.append(1.0 / (1.0 + dist_m))
        return scores


# ---------------------------------------------------------------------------
# PathSnapModel – the real trained model from models/model.pkl
# ---------------------------------------------------------------------------

class PathSnapModel(BaseModel):
    """
    Wraps the joblib bundle produced by models/train_final.py.

    The bundle is a dict:
        {
          "model":        HistGradientBoostingClassifier (fitted),
          "feature_cols": ["dist_m", "bearing_diff_deg", ...]   # 9 cols
        }

    The feature column list is read directly from the bundle so this class
    never hardcodes column names or order.

    predict_proba() accepts a list of feature dicts (one per candidate row),
    builds a numpy matrix in the bundle's own column order, and returns
    P(label=1) for each row.
    """

    def __init__(self, bundle_path: pathlib.Path):
        import joblib
        bundle = joblib.load(bundle_path)

        # Validate bundle structure
        if not isinstance(bundle, dict):
            raise TypeError(
                f"Expected a dict bundle from {bundle_path}, "
                f"got {type(bundle).__name__}. "
                "Re-check that the model was saved with train_final.py."
            )
        if "model" not in bundle or "feature_cols" not in bundle:
            raise KeyError(
                f"Bundle at {bundle_path} is missing 'model' or 'feature_cols'. "
                f"Keys present: {list(bundle.keys())}"
            )

        self._clf          = bundle["model"]
        self._feature_cols: list[str] = bundle["feature_cols"]
        self._path         = bundle_path

        log.info(
            "PathSnapModel loaded from %s | classifier=%s | features(%d)=%s",
            bundle_path,
            type(self._clf).__name__,
            len(self._feature_cols),
            self._feature_cols,
        )

    @property
    def name(self) -> str:
        return f"PathSnapModel({type(self._clf).__name__})"

    @property
    def feature_cols(self) -> list[str]:
        """The exact feature column list the model was trained on."""
        return list(self._feature_cols)

    def predict_proba(self, features: list[dict]) -> list[float]:
        """
        Score candidate rows using the trained HistGradientBoostingClassifier.

        Parameters
        ----------
        features : list[dict]
            Each dict must contain keys matching self._feature_cols.
            Missing keys default to 0.0 (HistGBM handles NaN internally,
            but we fill 0.0 to avoid any unexpected NaN propagation).

        Returns
        -------
        list[float]
            P(label=1) for each row (column index 1 of predict_proba output).
        """
        import numpy as np

        if not features:
            return []

        # Build feature matrix using the bundle's own column order
        X = [
            [float(feat.get(col, 0.0)) for col in self._feature_cols]
            for feat in features
        ]
        X_arr = np.array(X, dtype=np.float64)

        # predict_proba returns (n_rows, 2): col 0 = P(0), col 1 = P(1)
        proba = self._clf.predict_proba(X_arr)
        return [float(row[1]) for row in proba]


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def load_model(model_path: Optional[str] = None) -> BaseModel:
    """
    Load and return the appropriate model.

    Resolution order:
      1. Explicit model_path argument (if provided and file exists).
      2. PATHSNAP_MODEL_PATH environment variable (if set and file exists).
      3. Default path: models/model.pkl relative to project root.
      4. DummyModel fallback if none of the above resolve to an existing file.

    Parameters
    ----------
    model_path : str, optional

    Returns
    -------
    BaseModel instance ready for predict_proba() calls.
    """
    # Resolve candidate paths in priority order
    candidates: list[pathlib.Path] = []

    if model_path:
        candidates.append(pathlib.Path(model_path))

    env_path = os.environ.get("PATHSNAP_MODEL_PATH")
    if env_path:
        candidates.append(pathlib.Path(env_path))

    # Always try the default location last
    candidates.append(DEFAULT_MODEL_PATH)

    for p in candidates:
        if p.exists():
            try:
                model = PathSnapModel(p)
                log.info("Real model loaded: %s from %s", model.name, p)
                return model
            except Exception as exc:
                log.warning(
                    "Failed to load model from %s: %s — trying next candidate",
                    p, exc,
                )

    log.warning(
        "No model file found at any of: %s — falling back to DummyModel.",
        [str(c) for c in candidates],
    )
    return DummyModel()
