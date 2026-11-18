"""
PathSnap – Transition Probabilities
======================================
Computes log-transition probabilities between consecutive HMM states
(road arcs) for the Viterbi decoder.

Theory
------
The transition probability p(state_t = arc_j | state_{t-1} = arc_i) models
how likely it is for the vehicle to move from arc i to arc j between two
consecutive GPS observations.

We use a two-component model:

  1. CONNECTIVITY TERM:
     Arcs that are directly connected (arc_i ends at a node that is arc_j's
     start node, or they share a node) receive a HIGH transition probability.
     Arcs that are disconnected in the road graph receive a LOW penalty.

  2. ROUTE CONTINUITY TERM (distance penalty):
     The difference between the great-circle distance (GPS point to GPS point)
     and the road-network distance (along the arc sequence) is penalised.
     A large discrepancy indicates an unlikely transition.

GPS Gap Handling
----------------
When the timestamp difference between two GPS observations exceeds
GPS_GAP_THRESHOLD_S seconds, the transition probability is relaxed:
connectivity is still enforced but the distance penalty is zeroed out
(we cannot reliably estimate what path was taken during a large gap).
"""

from __future__ import annotations

import math
import logging
from typing import Optional

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Log-probability assigned to a connected transition (arc i → arc j where
# the endpoint of i is reachable from the start of j, or they share a node).
LOG_PROB_CONNECTED: float = math.log(0.95)

# Log-probability penalty for a topologically disconnected transition.
# This is intentionally lenient (not -inf) to handle:
#   - GPS noise that places the vehicle on a parallel road
#   - Missing arcs in the per-trip subgraph
#   - U-turns not captured in the arc topology
LOG_PROB_DISCONNECTED: float = math.log(0.05)

# If |ts_{t} - ts_{t-1}| > this threshold (seconds), relax the distance penalty.
GPS_GAP_THRESHOLD_S: float = 30.0

# Standard deviation for the route-continuity Gaussian (metres).
# A GPS-to-GPS great-circle distance vs road-network path length discrepancy
# of ROUTE_SIGMA_M metres produces a factor of exp(-0.5) ≈ 0.6.
ROUTE_SIGMA_M: float = 50.0

# Log-probability floor.
# Must be more negative than any naturally computed log-probability for realistic
# GPS distances, but less negative than the Gaussian output at absurd distances
# (e.g. 1,000,000 m with sigma=50 m gives -2e8, which must be caught by the floor).
# -1e8 satisfies both constraints: realistic distances (< ~100 km) produce values
# above -1e8, while implausible distances produce values below -1e8 and are floored.
LOG_PROB_FLOOR: float = -1e8


# ---------------------------------------------------------------------------
# Road connectivity helpers
# ---------------------------------------------------------------------------

def build_adjacency(
    arcs: list[tuple[int, int]],
) -> dict[int, set[int]]:
    """
    Build a directed adjacency mapping from the arc list.

    For the HMM we need to know: "if I am on arc i, which arcs can I
    naturally drive onto next?"

    Two arcs i and j are connected if:
      arc_i.to_node == arc_j.from_node

    Parameters
    ----------
    arcs : list[(from_node, to_node)]
        0-based arc index = position in list.

    Returns
    -------
    dict[int, set[int]]
        arc_id_i → set of arc_ids reachable directly from arc_i
        (i.e. arcs whose from_node == arc_i.to_node).
    """
    # Map: node_id → set of arc_ids that START at that node
    node_to_outgoing: dict[int, set[int]] = {}
    for arc_id, (from_node, to_node) in enumerate(arcs):
        node_to_outgoing.setdefault(from_node, set()).add(arc_id)

    adjacency: dict[int, set[int]] = {}
    for arc_id, (from_node, to_node) in enumerate(arcs):
        adjacency[arc_id] = node_to_outgoing.get(to_node, set())

    return adjacency


def arcs_are_connected(
    arc_i: int,
    arc_j: int,
    adjacency: dict[int, set[int]],
) -> bool:
    """
    Return True if arc_j is directly reachable from arc_i.

    Parameters
    ----------
    arc_i, arc_j : int – arc IDs (0-based)
    adjacency    : dict built by build_adjacency()
    """
    return arc_j in adjacency.get(arc_i, set())


# ---------------------------------------------------------------------------
# Great-circle distance (Haversine) for GPS gap distance estimation
# ---------------------------------------------------------------------------

def haversine_m(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    """
    Haversine great-circle distance in metres between two WGS-84 points.

    Parameters
    ----------
    lon1, lat1, lon2, lat2 : float – decimal degrees
    """
    R = 6_371_000.0  # Earth radius in metres
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = (math.sin(dphi / 2) ** 2
         + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2)
    return R * 2 * math.asin(min(1.0, math.sqrt(a)))


# ---------------------------------------------------------------------------
# Route-continuity distance penalty
# ---------------------------------------------------------------------------

def route_continuity_log_prob(
    gps_distance_m: float,
    sigma: float = ROUTE_SIGMA_M,
) -> float:
    """
    Log-Gaussian penalty for the discrepancy between GPS straight-line
    distance and the expected road travel distance.

    When gps_distance_m is 0 (stationary point), returns 0.0.

    Parameters
    ----------
    gps_distance_m : float – great-circle distance between consecutive GPS pts
    sigma          : float – tolerance in metres

    Returns
    -------
    float – log-probability (non-positive)
    """
    if gps_distance_m <= 0:
        return 0.0
    log_p = -(gps_distance_m ** 2) / (2.0 * sigma ** 2)
    return max(log_p, LOG_PROB_FLOOR)


# ---------------------------------------------------------------------------
# Single transition log-probability
# ---------------------------------------------------------------------------

def transition_log_prob(
    arc_i: int,
    arc_j: int,
    adjacency: dict[int, set[int]],
    gps_distance_m: float = 0.0,
    dt_seconds: float = 0.0,
    sigma: float = ROUTE_SIGMA_M,
    gap_threshold_s: float = GPS_GAP_THRESHOLD_S,
) -> float:
    """
    Compute log p(state_t = arc_j | state_{t-1} = arc_i).

    Parameters
    ----------
    arc_i, arc_j     : int   – consecutive Viterbi state arc IDs
    adjacency        : dict  – built by build_adjacency()
    gps_distance_m   : float – haversine distance between the two GPS points (m)
    dt_seconds       : float – timestamp difference (seconds)
    sigma            : float – route-continuity Gaussian sigma (m)
    gap_threshold_s  : float – GPS gap threshold; above this, skip distance penalty

    Returns
    -------
    float – log-transition probability
    """
    # --- Connectivity term ---
    if arcs_are_connected(arc_i, arc_j, adjacency):
        log_conn = LOG_PROB_CONNECTED
    else:
        log_conn = LOG_PROB_DISCONNECTED

    # --- Distance continuity term ---
    if dt_seconds > gap_threshold_s:
        # Large GPS gap: vehicle may have taken any path; skip penalty
        log_dist = 0.0
    else:
        log_dist = route_continuity_log_prob(gps_distance_m, sigma)

    return max(log_conn + log_dist, LOG_PROB_FLOOR)


# ---------------------------------------------------------------------------
# Build full transition table for one Viterbi step
# ---------------------------------------------------------------------------

def build_transition_row(
    prev_candidates: list[int],
    curr_candidates: list[int],
    adjacency: dict[int, set[int]],
    gps_distance_m: float = 0.0,
    dt_seconds: float = 0.0,
    sigma: float = ROUTE_SIGMA_M,
    gap_threshold_s: float = GPS_GAP_THRESHOLD_S,
) -> dict[tuple[int, int], float]:
    """
    Compute log-transition probabilities for all (arc_i, arc_j) pairs
    between two consecutive GPS observations.

    Parameters
    ----------
    prev_candidates : list[int] – arc IDs from time t-1
    curr_candidates : list[int] – arc IDs from time t
    adjacency       : dict      – road-network adjacency
    gps_distance_m  : float
    dt_seconds      : float
    sigma           : float
    gap_threshold_s : float

    Returns
    -------
    dict[(arc_i, arc_j), float]
        Log-transition probability for each pair.
    """
    result: dict[tuple[int, int], float] = {}
    for ai in prev_candidates:
        for aj in curr_candidates:
            result[(ai, aj)] = transition_log_prob(
                ai, aj, adjacency,
                gps_distance_m=gps_distance_m,
                dt_seconds=dt_seconds,
                sigma=sigma,
                gap_threshold_s=gap_threshold_s,
            )
    return result
