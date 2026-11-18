"""
Unit tests for backend/hmm/emissions.py

Coverage
--------
  TestDistanceLogPrior         – Gaussian distance prior correctness
  TestProbToLog                – _prob_to_log() conversion helper
  TestComputeEmissionLogProbs  – main emission calculator
  TestBuildEmissionTable       – table builder and length/mismatch handling
  TestEmissionModelProbabilities – probability-score integration tests
"""

import math
import sys
import pathlib

# Make the project root importable from tests/
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent.parent))

from backend.hmm.emissions import (
    distance_log_prior,
    compute_emission_log_probs,
    build_emission_table,
    _prob_to_log,
    _log_softmax,
    LOG_PROB_FLOOR,
    DISTANCE_SIGMA_M,
    _PROB_FLOOR,
)


# ---------------------------------------------------------------------------
# distance_log_prior
# ---------------------------------------------------------------------------

class TestDistanceLogPrior:

    def test_zero_distance_returns_zero(self):
        result = distance_log_prior(0.0)
        assert result == 0.0, "log p at zero distance must be 0"

    def test_increases_with_distance(self):
        p0 = distance_log_prior(0.0)
        p10 = distance_log_prior(10.0)
        p50 = distance_log_prior(50.0)
        assert p0 > p10 > p50, "log-prior must decrease as distance increases"

    def test_negative_input_returns_floor(self):
        assert distance_log_prior(-1.0) == LOG_PROB_FLOOR

    def test_sigma_gaussian_property(self):
        # At dist == sigma, log p = -0.5
        sigma = 20.0
        result = distance_log_prior(sigma, sigma=sigma)
        assert abs(result - (-0.5)) < 1e-9, (
            f"At dist=sigma, log p should be -0.5, got {result}"
        )

    def test_large_distance_floored(self):
        result = distance_log_prior(1_000_000.0)
        assert result == LOG_PROB_FLOOR


# ---------------------------------------------------------------------------
# _prob_to_log  (new helper introduced in B1 fix)
# ---------------------------------------------------------------------------

class TestProbToLog:
    """Tests for the probability → log-probability conversion helper."""

    def test_prob_one_returns_zero(self):
        """log(1) = 0."""
        assert _prob_to_log(1.0) == 0.0

    def test_prob_zero_returns_floor(self):
        """log(0) is -inf; must be clamped to log(_PROB_FLOOR)."""
        expected = math.log(_PROB_FLOOR)
        assert abs(_prob_to_log(0.0) - expected) < 1e-12

    def test_non_finite_nan_returns_floor(self):
        expected = math.log(_PROB_FLOOR)
        assert abs(_prob_to_log(float("nan")) - expected) < 1e-12

    def test_non_finite_inf_returns_floor(self):
        """inf and -inf are not finite → both return log(_PROB_FLOOR), same as nan.
        The function does NOT clamp inf to 1.0; all non-finite values are invalid."""
        expected = math.log(_PROB_FLOOR)
        assert abs(_prob_to_log(float("inf"))  - expected) < 1e-12
        assert abs(_prob_to_log(float("-inf")) - expected) < 1e-12

    def test_typical_probability(self):
        """For p in (0, 1) the result equals log(p)."""
        for p in [0.1, 0.5, 0.8, 0.99]:
            assert abs(_prob_to_log(p) - math.log(p)) < 1e-12

    def test_out_of_range_above_one_clamped_to_zero(self):
        """Scores > 1.0 are clamped to 1.0 → log = 0."""
        assert _prob_to_log(1.5) == 0.0
        assert _prob_to_log(10.0) == 0.0

    def test_out_of_range_below_zero_clamped_to_floor(self):
        """Scores < 0 are clamped to _PROB_FLOOR → log = log(_PROB_FLOOR)."""
        expected = math.log(_PROB_FLOOR)
        assert abs(_prob_to_log(-0.5) - expected) < 1e-12
        assert abs(_prob_to_log(-100.0) - expected) < 1e-12

    def test_result_is_non_positive(self):
        """log(p) ≤ 0 for all p ∈ [0, 1]."""
        for p in [0.0, 0.01, 0.5, 0.99, 1.0]:
            assert _prob_to_log(p) <= 0.0


# ---------------------------------------------------------------------------
# compute_emission_log_probs
# ---------------------------------------------------------------------------

class TestComputeEmissionLogProbs:

    def _make_candidates(self, dists):
        """Helper: create candidate dicts with given distances."""
        return [{"arc_id": i, "dist_m": d} for i, d in enumerate(dists)]

    def test_empty_candidates_returns_empty(self):
        result = compute_emission_log_probs([])
        assert result == {}

    def test_all_arc_ids_present(self):
        cands = self._make_candidates([5.0, 20.0, 50.0])
        result = compute_emission_log_probs(cands)
        assert set(result.keys()) == {0, 1, 2}

    def test_probabilities_sum_to_one(self):
        """exp of log-probabilities should sum to approximately 1."""
        cands = self._make_candidates([5.0, 20.0, 50.0])
        result = compute_emission_log_probs(cands)
        total = sum(math.exp(v) for v in result.values())
        assert abs(total - 1.0) < 1e-6, f"Probabilities should sum to 1, got {total}"

    def test_closer_arc_gets_higher_probability(self):
        cands = self._make_candidates([5.0, 50.0])
        result = compute_emission_log_probs(cands)
        assert result[0] > result[1], (
            "Arc at 5 m should have higher log-prob than arc at 50 m"
        )

    def test_model_scores_shift_probabilities(self):
        """When model gives arc 1 a high probability score, it should beat
        arc 0's distance advantage.

        B6 fix: scores must be PROBABILITIES in [0, 1], not logits.
        Old test used {0: -10.0, 1: 10.0} (logits) which is incorrect after
        the B1 fix converts them via log() before normalisation.
        """
        cands = self._make_candidates([2.0, 100.0])
        # Arc 1 gets a very high model probability despite being far away.
        model_scores = {0: 0.01, 1: 0.99}
        result = compute_emission_log_probs(cands, model_scores=model_scores)
        assert result[1] > result[0], (
            "High model probability (0.99) for arc 1 should overcome "
            "distance disadvantage (100 m vs 2 m for arc 0)"
        )

    def test_single_candidate_gets_log_prob_zero(self):
        """With one candidate, it gets all probability → log p ≈ 0."""
        cands = [{"arc_id": 42, "dist_m": 10.0}]
        result = compute_emission_log_probs(cands)
        assert abs(result[42] - 0.0) < 1e-6, (
            f"Single candidate should have log p ≈ 0, got {result[42]}"
        )

    def test_no_model_scores_uses_distance_only(self):
        cands = self._make_candidates([10.0, 30.0])
        result_no_model  = compute_emission_log_probs(cands, model_scores=None)
        result_zero_wt   = compute_emission_log_probs(cands, model_scores=None, model_weight=0.0)
        # Both should give distance-only result; keys and ordering match
        assert set(result_no_model.keys()) == {0, 1}
        assert result_no_model[0] > result_no_model[1]
        assert result_zero_wt[0] > result_zero_wt[1]

    def test_all_values_are_finite(self):
        """All returned log-probs must be finite (no -inf, no nan)."""
        cands = self._make_candidates([5.0, 20.0, 50.0])
        result = compute_emission_log_probs(cands)
        for v in result.values():
            assert math.isfinite(v), f"Non-finite emission value: {v}"

    def test_all_values_at_or_above_floor(self):
        """No value should be below LOG_PROB_FLOOR."""
        cands = self._make_candidates([0.0, 1000.0])
        result = compute_emission_log_probs(cands)
        for v in result.values():
            assert v >= LOG_PROB_FLOOR


# ---------------------------------------------------------------------------
# TestEmissionModelProbabilities – probability-score specific tests
# ---------------------------------------------------------------------------

class TestEmissionModelProbabilities:
    """Tests that model scores are correctly interpreted as probabilities,
    not logits.  These tests would fail if log-softmax were applied directly
    to raw probability values."""

    def test_equal_model_probabilities_distance_wins(self):
        """When model gives equal scores (0.5/0.5), the closer arc should win."""
        cands = [{"arc_id": 0, "dist_m": 5.0}, {"arc_id": 1, "dist_m": 50.0}]
        model_scores = {0: 0.5, 1: 0.5}
        result = compute_emission_log_probs(cands, model_scores=model_scores)
        assert result[0] > result[1], (
            "Equal model scores: distance prior should break the tie"
        )

    def test_high_model_prob_overrides_distance(self):
        """High model probability on a far arc can override distance prior."""
        cands = [{"arc_id": 0, "dist_m": 5.0}, {"arc_id": 1, "dist_m": 100.0}]
        model_scores = {0: 0.01, 1: 0.99}
        # Very high model weight so the model dominates
        result = compute_emission_log_probs(cands, model_scores=model_scores, model_weight=0.95)
        assert result[1] > result[0], (
            "Arc 1 (p=0.99) should beat arc 0 (p=0.01) at high model_weight"
        )

    def test_invalid_scores_clamped_not_crash(self):
        """Out-of-range or non-finite model scores must not crash or produce -inf."""
        cands = [{"arc_id": 0, "dist_m": 10.0}, {"arc_id": 1, "dist_m": 20.0}]
        # Pass scores that are out-of-range (should warn and clamp)
        model_scores = {0: float("nan"), 1: 2.0}
        result = compute_emission_log_probs(cands, model_scores=model_scores)
        assert set(result.keys()) == {0, 1}
        for v in result.values():
            assert math.isfinite(v), f"Non-finite emission for invalid input: {v}"

    def test_model_prob_zero_treated_as_floor_not_negative_infinity(self):
        """A model score of exactly 0.0 should produce log(_PROB_FLOOR), not -inf."""
        cands = [{"arc_id": 0, "dist_m": 5.0}, {"arc_id": 1, "dist_m": 10.0}]
        model_scores = {0: 0.0, 1: 1.0}
        result = compute_emission_log_probs(cands, model_scores=model_scores)
        for v in result.values():
            assert math.isfinite(v), f"Zero probability score produced non-finite: {v}"

    def test_model_prob_one_dominates_when_weight_is_one(self):
        """With model_weight=1.0, the arc with p=1.0 should get log p ≈ 0."""
        cands = [{"arc_id": 0, "dist_m": 5.0}, {"arc_id": 1, "dist_m": 5.0}]
        model_scores = {0: 0.0, 1: 1.0}
        result = compute_emission_log_probs(cands, model_scores=model_scores, model_weight=1.0)
        assert result[1] > result[0], (
            "Arc with model prob=1.0 must dominate arc with prob=0.0"
        )

    def test_result_sums_to_one_in_probability_space(self):
        """After normalisation, exp of log-probs must sum to ≈ 1."""
        cands = [{"arc_id": i, "dist_m": float(5 * (i + 1))} for i in range(5)]
        model_scores = {i: (i + 1) / 10.0 for i in range(5)}  # 0.1 … 0.5
        result = compute_emission_log_probs(cands, model_scores=model_scores)
        total = sum(math.exp(v) for v in result.values())
        assert abs(total - 1.0) < 1e-6, f"Prob total should be 1.0, got {total}"


# ---------------------------------------------------------------------------
# build_emission_table
# ---------------------------------------------------------------------------

class TestBuildEmissionTable:

    def test_output_length_matches_input(self):
        candidates_by_point = [
            [{"arc_id": 0, "dist_m": 5.0}, {"arc_id": 1, "dist_m": 20.0}],
            [{"arc_id": 2, "dist_m": 10.0}],
            [],
        ]
        result = build_emission_table(candidates_by_point)
        assert len(result) == 3

    def test_empty_point_gives_empty_dict(self):
        result = build_emission_table([[]])
        assert result == [{}]

    def test_model_scores_length_mismatch_handled_gracefully(self):
        """Mismatched model_scores list should not crash."""
        cands = [
            [{"arc_id": 0, "dist_m": 5.0}],
            [{"arc_id": 1, "dist_m": 10.0}],
        ]
        # Only 1 score for 2 points — should pad with None
        result = build_emission_table(cands, model_scores_by_point=[{0: 0.9}])
        assert len(result) == 2

    def test_model_scores_as_probabilities(self):
        """Scores in [0,1] should produce valid normalised log-probs."""
        cands = [
            [{"arc_id": 0, "dist_m": 5.0}, {"arc_id": 1, "dist_m": 20.0}],
        ]
        model_scores = [{0: 0.3, 1: 0.7}]
        result = build_emission_table(cands, model_scores_by_point=model_scores)
        assert len(result) == 1
        total = sum(math.exp(v) for v in result[0].values())
        assert abs(total - 1.0) < 1e-6


# ---------------------------------------------------------------------------
# Run with pytest
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
