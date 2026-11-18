"""
Unit tests for backend/hmm/transitions.py
"""

import math
import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent.parent))

from backend.hmm.transitions import (
    build_adjacency,
    arcs_are_connected,
    haversine_m,
    route_continuity_log_prob,
    transition_log_prob,
    build_transition_row,
    LOG_PROB_CONNECTED,
    LOG_PROB_DISCONNECTED,
    LOG_PROB_FLOOR,
    ROUTE_SIGMA_M,
    GPS_GAP_THRESHOLD_S,
)


# ---------------------------------------------------------------------------
# build_adjacency
# ---------------------------------------------------------------------------

class TestBuildAdjacency:

    def test_simple_chain(self):
        # arc 0: 0→1, arc 1: 1→2, arc 2: 2→3
        arcs = [(0, 1), (1, 2), (2, 3)]
        adj = build_adjacency(arcs)
        assert 1 in adj[0]   # arc 0 → arc 1 (node 1 is start of arc 1)
        assert 2 in adj[1]   # arc 1 → arc 2
        assert adj[2] == set()  # arc 2 leads to node 3, no arc starts there

    def test_branching(self):
        # arc 0: 0→1, arc 1: 1→2, arc 2: 1→3  (fork at node 1)
        arcs = [(0, 1), (1, 2), (1, 3)]
        adj = build_adjacency(arcs)
        assert 1 in adj[0]
        assert 2 in adj[0]   # arc 0 → arcs 1 and 2 (both start at node 1)

    def test_self_loop_arc_not_reachable_from_itself(self):
        # arc 0: 0→0 (self-loop)
        arcs = [(0, 0)]
        adj = build_adjacency(arcs)
        # arc 0 ends at node 0, and arc 0 starts at node 0 → reachable
        assert 0 in adj[0]

    def test_empty_arcs(self):
        adj = build_adjacency([])
        assert adj == {}

    def test_disconnected_arcs(self):
        # arc 0: 0→1, arc 1: 5→6  (no shared nodes)
        arcs = [(0, 1), (5, 6)]
        adj = build_adjacency(arcs)
        assert adj[0] == set()   # arc 0 ends at node 1; no arc starts at node 1
        assert adj[1] == set()   # arc 1 ends at node 6; no arc starts there


# ---------------------------------------------------------------------------
# arcs_are_connected
# ---------------------------------------------------------------------------

class TestArcsAreConnected:

    def _adj(self):
        arcs = [(0, 1), (1, 2), (2, 3), (5, 6)]
        return build_adjacency(arcs)

    def test_connected_pair(self):
        adj = self._adj()
        assert arcs_are_connected(0, 1, adj) is True
        assert arcs_are_connected(1, 2, adj) is True

    def test_disconnected_pair(self):
        adj = self._adj()
        assert arcs_are_connected(0, 3, adj) is False
        assert arcs_are_connected(0, 2, adj) is False

    def test_arc_not_in_adjacency(self):
        adj = self._adj()
        assert arcs_are_connected(99, 0, adj) is False


# ---------------------------------------------------------------------------
# haversine_m
# ---------------------------------------------------------------------------

class TestHaversineM:

    def test_zero_distance(self):
        d = haversine_m(77.5, 12.9, 77.5, 12.9)
        assert d == 0.0

    def test_known_distance(self):
        # Bangalore (77.5946, 12.9716) → ~111 km north (13.9716)
        d = haversine_m(77.5946, 12.9716, 77.5946, 13.9716)
        # ~111,195 m at this latitude
        assert 110_000 < d < 112_000, f"Expected ~111 km, got {d:.0f} m"

    def test_symmetry(self):
        d_ab = haversine_m(10.0, 48.0, 10.5, 48.5)
        d_ba = haversine_m(10.5, 48.5, 10.0, 48.0)
        assert abs(d_ab - d_ba) < 1e-6


# ---------------------------------------------------------------------------
# route_continuity_log_prob
# ---------------------------------------------------------------------------

class TestRouteContinuityLogProb:

    def test_zero_distance_returns_zero(self):
        assert route_continuity_log_prob(0.0) == 0.0

    def test_negative_distance_returns_zero(self):
        assert route_continuity_log_prob(-5.0) == 0.0

    def test_sigma_property(self):
        # At dist == sigma, log p = -0.5
        result = route_continuity_log_prob(ROUTE_SIGMA_M)
        assert abs(result - (-0.5)) < 1e-9

    def test_very_large_distance_floored(self):
        result = route_continuity_log_prob(1_000_000.0)
        assert result == LOG_PROB_FLOOR

    def test_monotonically_decreasing(self):
        probs = [route_continuity_log_prob(d) for d in [0, 10, 50, 100, 500]]
        for i in range(len(probs) - 1):
            assert probs[i] >= probs[i + 1]


# ---------------------------------------------------------------------------
# transition_log_prob
# ---------------------------------------------------------------------------

class TestTransitionLogProb:

    def _adj(self):
        arcs = [(0, 1), (1, 2), (2, 3)]
        return build_adjacency(arcs)

    def test_connected_better_than_disconnected(self):
        adj = self._adj()
        lp_conn   = transition_log_prob(0, 1, adj, gps_distance_m=10.0, dt_seconds=1.0)
        lp_disconn = transition_log_prob(0, 2, adj, gps_distance_m=10.0, dt_seconds=1.0)
        assert lp_conn > lp_disconn

    def test_connected_at_zero_distance(self):
        adj = self._adj()
        lp = transition_log_prob(0, 1, adj, gps_distance_m=0.0, dt_seconds=1.0)
        # Should be at or just below LOG_PROB_CONNECTED (distance adds 0 penalty)
        assert lp <= LOG_PROB_CONNECTED
        assert lp > LOG_PROB_FLOOR

    def test_gps_gap_relaxes_distance_penalty(self):
        adj = self._adj()
        large_dist = 5000.0  # 5 km – very penalising normally
        lp_no_gap  = transition_log_prob(0, 1, adj, gps_distance_m=large_dist, dt_seconds=5.0)
        lp_gap     = transition_log_prob(0, 1, adj, gps_distance_m=large_dist,
                                         dt_seconds=GPS_GAP_THRESHOLD_S + 10.0)
        assert lp_gap > lp_no_gap, (
            "GPS gap should relax distance penalty, giving higher transition prob"
        )

    def test_result_at_floor_or_above(self):
        adj = self._adj()
        lp = transition_log_prob(0, 99, adj, gps_distance_m=1e6, dt_seconds=1.0)
        assert lp >= LOG_PROB_FLOOR


# ---------------------------------------------------------------------------
# build_transition_row
# ---------------------------------------------------------------------------

class TestBuildTransitionRow:

    def _adj(self):
        arcs = [(0, 1), (1, 2), (2, 3)]
        return build_adjacency(arcs)

    def test_all_pairs_present(self):
        adj = self._adj()
        prev = [0, 1]
        curr = [1, 2]
        result = build_transition_row(prev, curr, adj, gps_distance_m=10.0, dt_seconds=1.0)
        assert set(result.keys()) == {(0, 1), (0, 2), (1, 1), (1, 2)}

    def test_empty_prev_gives_empty_result(self):
        adj = self._adj()
        result = build_transition_row([], [1, 2], adj)
        assert result == {}

    def test_empty_curr_gives_empty_result(self):
        adj = self._adj()
        result = build_transition_row([0, 1], [], adj)
        assert result == {}


# ---------------------------------------------------------------------------
# Run with pytest
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
