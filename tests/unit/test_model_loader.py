"""
Unit tests for backend/inference/model_loader.py

Coverage
--------
  TestBaseModelInterface    – abstract interface contract
  TestDummyModel            – DummyModel scoring logic and fallback key support
  TestPathSnapModelMocked   – PathSnapModel with mocked bundle (no .pkl I/O)
  TestLoadModelFactory      – load_model() factory resolution logic
"""

import math
import pathlib
import sys
import types
import unittest.mock as mock

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent.parent))

from backend.inference.model_loader import (
    BaseModel,
    DummyModel,
    PathSnapModel,
    load_model,
    DEFAULT_MODEL_PATH,
)


# ---------------------------------------------------------------------------
# TestBaseModelInterface
# ---------------------------------------------------------------------------

class TestBaseModelInterface:
    """Confirm the abstract interface is correctly defined."""

    def test_base_model_is_abstract(self):
        """Cannot instantiate BaseModel directly."""
        try:
            BaseModel()
            assert False, "Should have raised TypeError"
        except TypeError:
            pass  # Expected

    def test_dummy_model_is_base_model(self):
        assert isinstance(DummyModel(), BaseModel)


# ---------------------------------------------------------------------------
# TestDummyModel
# ---------------------------------------------------------------------------

class TestDummyModel:

    def test_name(self):
        m = DummyModel()
        assert isinstance(m.name, str)
        assert len(m.name) > 0

    def test_empty_features_returns_empty(self):
        m = DummyModel()
        assert m.predict_proba([]) == []

    def test_scores_in_zero_one(self):
        m = DummyModel()
        feats = [
            {"dist_m": 0.0},
            {"dist_m": 10.0},
            {"dist_m": 1000.0},
        ]
        scores = m.predict_proba(feats)
        assert len(scores) == 3
        for s in scores:
            assert 0.0 <= s <= 1.0, f"DummyModel score out of range: {s}"

    def test_closer_arc_scores_higher(self):
        m = DummyModel()
        scores = m.predict_proba([{"dist_m": 5.0}, {"dist_m": 50.0}])
        assert scores[0] > scores[1], "Closer arc must score higher"

    def test_fallback_key_point_to_arc_distance(self):
        """DummyModel must still work with the old key name for backwards compat."""
        m = DummyModel()
        scores = m.predict_proba([{"point_to_arc_distance": 10.0}])
        assert len(scores) == 1
        assert 0.0 < scores[0] <= 1.0

    def test_missing_distance_key_does_not_crash(self):
        """Feature dict with no dist_m or fallback key should use 1e6 → score ≈ 0."""
        m = DummyModel()
        scores = m.predict_proba([{}])
        assert len(scores) == 1
        assert 0.0 <= scores[0] <= 1.0


# ---------------------------------------------------------------------------
# TestPathSnapModelMocked
# ---------------------------------------------------------------------------

class _FakeClf:
    """Minimal sklearn-like classifier for testing."""

    def __init__(self, n_classes=2):
        self._n_classes = n_classes

    def predict_proba(self, X):
        import numpy as np
        n = len(X)
        # Return deterministic proba: col-0 = 0.3, col-1 = 0.7
        out = np.zeros((n, 2))
        out[:, 0] = 0.3
        out[:, 1] = 0.7
        return out


def _make_bundle(feature_cols=None):
    if feature_cols is None:
        feature_cols = ["dist_m", "bearing_diff_deg", "arc_length_m"]
    return {
        "model": _FakeClf(),
        "feature_cols": feature_cols,
    }


class TestPathSnapModelMocked:

    def _build(self, feature_cols=None):
        """Build a PathSnapModel with a mocked joblib.load."""
        bundle = _make_bundle(feature_cols)
        with mock.patch("backend.inference.model_loader.PathSnapModel.__init__") as _init:
            # Bypass __init__ and set attributes directly
            obj = PathSnapModel.__new__(PathSnapModel)
            obj._clf = bundle["model"]
            obj._feature_cols = bundle["feature_cols"]
            obj._path = pathlib.Path("mock/model.pkl")
        return obj

    def test_name_contains_classifier_type(self):
        m = self._build()
        assert "PathSnapModel" in m.name
        assert "_FakeClf" in m.name

    def test_feature_cols_property_returns_copy(self):
        m = self._build(["dist_m", "speed_mps"])
        cols = m.feature_cols
        cols.append("extra")
        assert "extra" not in m.feature_cols, "feature_cols property must return a copy"

    def test_predict_proba_empty_returns_empty(self):
        m = self._build()
        assert m.predict_proba([]) == []

    def test_predict_proba_scores_in_zero_one(self):
        m = self._build(["dist_m", "bearing_diff_deg", "arc_length_m"])
        feats = [
            {"dist_m": 10.0, "bearing_diff_deg": 5.0, "arc_length_m": 100.0},
            {"dist_m": 50.0, "bearing_diff_deg": 45.0, "arc_length_m": 200.0},
        ]
        scores = m.predict_proba(feats)
        assert len(scores) == 2
        for s in scores:
            assert 0.0 <= s <= 1.0, f"Score out of [0,1]: {s}"
        # _FakeClf always returns 0.7 for column-1
        for s in scores:
            assert abs(s - 0.7) < 1e-9

    def test_predict_proba_missing_features_default_to_zero(self):
        """Missing keys in feature dict should default to 0.0 without crashing."""
        m = self._build(["dist_m", "speed_mps"])
        # speed_mps is missing → should default to 0.0
        feats = [{"dist_m": 10.0}]
        scores = m.predict_proba(feats)
        assert len(scores) == 1

    def test_is_base_model(self):
        m = self._build()
        assert isinstance(m, BaseModel)


# ---------------------------------------------------------------------------
# TestLoadModelFactory
# ---------------------------------------------------------------------------

class TestLoadModelFactory:

    def test_returns_base_model(self):
        """load_model() must always return a BaseModel subclass."""
        # Patch joblib.load to avoid needing the real .pkl file
        fake_bundle = _make_bundle()
        with mock.patch("joblib.load", return_value=fake_bundle):
            m = load_model(str(DEFAULT_MODEL_PATH))
        assert isinstance(m, BaseModel)

    def test_no_model_anywhere_falls_back_to_dummy(self):
        """When NO path resolves (explicit, env-var, or default), return DummyModel.

        Patches Path.exists to False for all candidates so even the real
        models/model.pkl on disk is treated as absent.
        """
        with mock.patch("pathlib.Path.exists", return_value=False):
            m = load_model("/nonexistent/path/model.pkl")
        assert isinstance(m, DummyModel)

    def test_corrupt_bundle_falls_back_to_dummy(self):
        """A bundle that raises during load causes fallback to DummyModel."""
        with mock.patch("joblib.load", side_effect=Exception("corrupt file")):
            # Also patch path.exists() to pretend the file exists
            with mock.patch("pathlib.Path.exists", return_value=True):
                m = load_model("/fake/model.pkl")
        assert isinstance(m, DummyModel)

    def test_missing_bundle_key_falls_back_to_dummy(self):
        """A bundle missing 'model' or 'feature_cols' causes fallback."""
        incomplete_bundle = {"model": _FakeClf()}  # missing 'feature_cols'
        with mock.patch("joblib.load", return_value=incomplete_bundle):
            with mock.patch("pathlib.Path.exists", return_value=True):
                m = load_model("/fake/model.pkl")
        assert isinstance(m, DummyModel)

    def test_env_var_overrides_default_path(self):
        """PATHSNAP_MODEL_PATH env var should be tried before the default."""
        import os
        fake_bundle = _make_bundle()
        env_path = "/env/override/model.pkl"
        with mock.patch.dict(os.environ, {"PATHSNAP_MODEL_PATH": env_path}):
            with mock.patch("joblib.load", return_value=fake_bundle) as mock_load:
                with mock.patch("pathlib.Path.exists", return_value=True):
                    m = load_model()
        # The first call to joblib.load should use the env path
        first_call_path = str(mock_load.call_args_list[0][0][0])
        assert env_path in first_call_path


# ---------------------------------------------------------------------------
# Run with pytest
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
