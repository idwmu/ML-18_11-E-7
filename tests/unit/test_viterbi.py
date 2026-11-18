"""
Unit tests for backend/hmm/viterbi.py
"""

import math
import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent.parent))

from backend.hmm.viterbi import viterbi_decode, ViterbiResult, LOG_PROB_FLOOR
from backend.hmm.transitions import build_adjacency, transition_log_prob, haversine_m


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_emission(candidates: list[tuple[int, float]]) -> dict[int, float]:
    """
    Build an emission dict from (arc_id, log_prob) pairs.
    Does NOT normalise – caller supplies correct log-probs.
    """
    return {arc_id: lp for arc_id, lp in candidates}


def _make_transition_fn(adjacency):
    """Create a simple transition function from adjacency for tests."""
    def _fn(arc_i, arc_j, t_prev, t_curr):
        lon1, lat1, ts1 = t_prev
        lon2, lat2, ts2 = t_curr
        gps_dist = haversine_m(lon1, lat1, lon2, lat2)
        dt = abs(ts2 - ts1)
        return transition_log_prob(arc_i, arc_j, adjacency,
                                   gps_distance_m=gps_dist, dt_seconds=dt)
    return _fn


def _dummy_transition_fn(arc_i, arc_j, t_prev, t_curr):
    """Transition function that returns 0 for all transitions (uniform)."""
    return 0.0


# ---------------------------------------------------------------------------
# TestViterbiDecode – basic correctness
# ---------------------------------------------------------------------------

class TestViterbiDecode:

    def test_single_gps_point_returns_best_emission(self):
        """With one GPS point, Viterbi should pick the arc with highest emission."""
        emission_table = [
            {0: math.log(0.1), 1: math.log(0.6), 2: math.log(0.3)}
        ]
        gps_track = [(8.595, 49.845, 1.0)]

        result = viterbi_decode(emission_table, _dummy_transition_fn, gps_track)

        assert result.success is True
        assert 1 in result.arc_sequence, (
            "Arc 1 (highest emission) must appear in the route"
        )

    def test_two_gps_points_simple_chain(self):
        """
        Arc chain: 0→1→2.
        GPS point 1 strongly prefers arc 0.
        GPS point 2 strongly prefers arc 1.
        Connected arc (0→1) should be chosen.
        """
        arcs = [(0, 1), (1, 2), (5, 6)]   # arc 2 disconnected
        adj = build_adjacency(arcs)

        emission_table = [
            {0: math.log(0.9), 1: math.log(0.05), 2: math.log(0.05)},
            {0: math.log(0.05), 1: math.log(0.9), 2: math.log(0.05)},
        ]
        gps_track = [
            (8.595, 49.845, 1.0),
            (8.596, 49.846, 2.0),
        ]

        result = viterbi_decode(emission_table, _make_transition_fn(adj), gps_track)

        assert result.success is True
        assert len(result.arc_sequence) >= 1
        # The route should contain arcs 0 and/or 1 (not disconnected arc 2)
        for arc in result.arc_sequence:
            assert arc in {0, 1}, f"Unexpected arc {arc} in route: {result.arc_sequence}"

    def test_empty_emission_table_fails_gracefully(self):
        result = viterbi_decode([], _dummy_transition_fn, [])
        assert result.success is False
        assert result.arc_sequence == []
        assert result.n_gps_points == 0

    def test_all_empty_candidate_sets_fails_gracefully(self):
        """All GPS points have no candidates."""
        emission_table = [{}, {}, {}]
        gps_track = [
            (8.595, 49.845, 1.0),
            (8.596, 49.846, 2.0),
            (8.597, 49.847, 3.0),
        ]
        result = viterbi_decode(emission_table, _dummy_transition_fn, gps_track)
        assert result.success is False
        assert result.arc_sequence == []

    def test_mixed_empty_and_non_empty_points(self):
        """Some GPS points have no candidates; should not crash."""
        emission_table = [
            {0: math.log(0.8), 1: math.log(0.2)},
            {},                                         # no candidates
            {0: math.log(0.3), 1: math.log(0.7)},
        ]
        gps_track = [
            (8.595, 49.845, 1.0),
            (8.596, 49.846, 2.0),
            (8.597, 49.847, 3.0),
        ]
        result = viterbi_decode(emission_table, _dummy_transition_fn, gps_track)
        assert result.success is True
        assert result.n_skipped_points == 1
        assert abs(result.candidate_miss_rate - (1 / 3)) < 1e-6

    def test_candidate_miss_rate_all_valid(self):
        emission_table = [
            {0: math.log(0.6), 1: math.log(0.4)},
            {0: math.log(0.4), 1: math.log(0.6)},
        ]
        gps_track = [(8.595, 49.845, 1.0), (8.596, 49.846, 2.0)]
        result = viterbi_decode(emission_table, _dummy_transition_fn, gps_track)
        assert result.candidate_miss_rate == 0.0
        assert result.n_skipped_points == 0


# ---------------------------------------------------------------------------
# TestViterbiResult – field correctness
# ---------------------------------------------------------------------------

class TestViterbiResult:

    def test_n_gps_points_matches_input(self):
        T = 5
        emission_table = [{0: 0.0} for _ in range(T)]
        gps_track = [(8.0 + i * 0.001, 49.0, float(i)) for i in range(T)]
        result = viterbi_decode(emission_table, _dummy_transition_fn, gps_track)
        assert result.n_gps_points == T

    def test_log_probability_is_finite_on_success(self):
        emission_table = [{0: math.log(0.8), 1: math.log(0.2)}]
        gps_track = [(8.595, 49.845, 1.0)]
        result = viterbi_decode(emission_table, _dummy_transition_fn, gps_track)
        assert result.success is True
        assert math.isfinite(result.log_probability)

    def test_arc_sequence_non_empty_on_success(self):
        emission_table = [{0: math.log(0.6), 1: math.log(0.4)}]
        gps_track = [(8.595, 49.845, 1.0)]
        result = viterbi_decode(emission_table, _dummy_transition_fn, gps_track)
        assert result.success is True
        assert len(result.arc_sequence) > 0

    def test_consecutive_duplicates_removed(self):
        """When all GPS points assign the same arc, route should not repeat it."""
        emission_table = [
            {5: 0.0},  # arc 5 is best at every step
            {5: 0.0},
            {5: 0.0},
        ]
        gps_track = [(8.595, 49.845, t) for t in [1.0, 2.0, 3.0]]
        result = viterbi_decode(emission_table, _dummy_transition_fn, gps_track)
        assert result.success is True
        # Consecutive duplicates should be deduped
        for i in range(len(result.arc_sequence) - 1):
            assert result.arc_sequence[i] != result.arc_sequence[i + 1], (
                f"Consecutive duplicate arc {result.arc_sequence[i]} found "
                f"in sequence {result.arc_sequence}"
            )


# ---------------------------------------------------------------------------
# TestViterbiTransitionPreference
# ---------------------------------------------------------------------------

class TestViterbiTransitionPreference:

    def test_connected_path_beats_disconnected(self):
        """
        Topology:  arc 0 → arc 1 → arc 2  (a simple chain)
        Arc 3 is disconnected from everything.

        At time 0, all arcs are equally probable.
        At time 1, arc 1 (connected to 0) and arc 3 (disconnected from 0) are equal.
        Viterbi should prefer arc 1 due to connectivity.
        """
        arcs = [(0, 1), (1, 2), (2, 3), (10, 11)]  # arc 3 disconnected
        adj = build_adjacency(arcs)

        emission_table = [
            {0: 0.0},                       # t=0: only arc 0
            {1: math.log(0.5), 3: math.log(0.5)},  # t=1: equal emission
        ]
        gps_track = [
            (8.595, 49.845, 1.0),
            (8.596, 49.846, 2.0),
        ]

        result = viterbi_decode(emission_table, _make_transition_fn(adj), gps_track)

        assert result.success is True
        # Arc 1 (connected from arc 0) should be preferred over arc 3 (disconnected)
        if len(result.arc_sequence) > 1:
            assert result.arc_sequence[-1] == 1, (
                f"Expected arc 1 at t=1, got {result.arc_sequence}"
            )


# ---------------------------------------------------------------------------
# Run with pytest
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
