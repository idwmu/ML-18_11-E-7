"""
PathSnap – Viterbi Decoder
============================
Implements the Viterbi algorithm for Hidden Markov Model route reconstruction.

HMM formulation
---------------
  Hidden states : road arcs (arc_id ∈ candidates for each GPS point)
  Observations  : GPS points (lon, lat, ts)
  Emissions     : p(GPS_obs_t | state=arc_i)  ← from emissions.py
  Transitions   : p(state_t=arc_j | state_{t-1}=arc_i) ← from transitions.py
  Initial dist. : uniform over first GPS point's candidates

The Viterbi algorithm finds the state sequence that maximises the joint
log-probability of all observations and state transitions:

  argmax  [ log π(s_1) + sum_t log p(o_t | s_t) + sum_t log p(s_t | s_{t-1}) ]

All computations are in log-space to avoid numerical underflow.

Edge cases handled
------------------
  - GPS points with zero candidates: skip the DP step; carry forward current
    Viterbi states unchanged.  Backtracking skips (not breaks) these steps.
  - GPS gaps (dt > threshold): transition penalty is relaxed (see transitions.py).
  - No valid path found: returns an empty arc sequence with a flag.
  - Single GPS point: returns the highest-emission candidate with no transitions.

Backtracking note
-----------------
  When a GPS point is skipped (no candidates), `backptr[t]` is an empty dict.
  During backtracking we `continue` over these empty steps rather than
  `break`ing — breaking would silently truncate the returned arc sequence.

  When all predecessors at time t are floored (score ≤ LOG_PROB_FLOOR),
  we start the DP fresh from the emission of arc_j, leaving `best_prev = -1`
  to signal the chain break.  The old code incorrectly used `prev_arcs[0]`
  as an arbitrary predecessor, creating a false backtrack path.
"""

from __future__ import annotations

import math
import logging
from dataclasses import dataclass, field
from typing import Optional

log = logging.getLogger(__name__)

LOG_PROB_FLOOR: float = -1e9


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class ViterbiResult:
    """Output of the Viterbi decoder for one trip."""
    arc_sequence:        list[int]   # Predicted arc IDs in GPS temporal order
    log_probability:     float       # Total log-probability of the path
    n_gps_points:        int         # Total GPS observations processed
    n_skipped_points:    int         # Points with no candidates (skipped)
    candidate_miss_rate: float       # Fraction of skipped points
    success:             bool        # False if no valid path could be found
    message:             str = ""    # Human-readable status or error note


# ---------------------------------------------------------------------------
# Viterbi algorithm
# ---------------------------------------------------------------------------

def viterbi_decode(
    emission_table: list[dict[int, float]],
    transition_fn,
    gps_track: list[tuple[float, float, float]],
) -> ViterbiResult:
    """
    Run the Viterbi algorithm over a sequence of GPS observations.

    Parameters
    ----------
    emission_table : list[dict[int, float]]
        One dict per GPS point: arc_id → log-emission probability.
        Built by emissions.build_emission_table().

    transition_fn : callable
        Signature: transition_fn(arc_i, arc_j, t_prev, t_curr) -> float
            Returns log p(state_t=arc_j | state_{t-1}=arc_i).
            t_prev, t_curr are (lon, lat, ts) tuples from the GPS track.

    gps_track : list[(lon, lat, ts)]
        GPS observations in temporal order. Must be the same length as
        emission_table.

    Returns
    -------
    ViterbiResult

    Algorithm
    ---------
    For each time step t:
      For each candidate arc j at time t:
        viterbi[t][j] = max_i { viterbi[t-1][i] + log_trans(i→j) + log_emit(j) }
        backptr[t][j] = argmax_i { viterbi[t-1][i] + log_trans(i→j) }

    After processing all time steps, backtrack from the highest-probability
    final state to recover the full arc sequence.

    Backtracking skips (not breaks) steps where backptr[t] is empty, which
    occur when a GPS point has no candidates and was skipped in the DP.
    """
    T = len(emission_table)

    if T == 0:
        return ViterbiResult(
            arc_sequence=[], log_probability=LOG_PROB_FLOOR,
            n_gps_points=0, n_skipped_points=0,
            candidate_miss_rate=0.0, success=False,
            message="Empty emission table: no GPS points to decode.",
        )

    # --- Initialisation (t=0) ---
    t0_emissions = emission_table[0]

    if not t0_emissions:
        # No candidates at the first point – try to find the first point with candidates
        log.warning("viterbi_decode: first GPS point has no candidates; "
                    "scanning forward for initial state.")
        first_valid_t = None
        for t in range(T):
            if emission_table[t]:
                first_valid_t = t
                break
        if first_valid_t is None:
            return ViterbiResult(
                arc_sequence=[], log_probability=LOG_PROB_FLOOR,
                n_gps_points=T, n_skipped_points=T,
                candidate_miss_rate=1.0, success=False,
                message="No candidates for any GPS point in this trip.",
            )
        # Re-enter with trimmed emission table
        return viterbi_decode(
            emission_table[first_valid_t:],
            transition_fn,
            gps_track[first_valid_t:],
        )

    # viterbi[arc_id] = best log-prob reaching this arc at current time step
    viterbi: dict[int, float] = {}
    # backptr[t][arc_j] = arc_i that led to arc_j at time t with best score.
    # backptr[t] is {} when t=0 (no predecessor) or when t was skipped.
    backptr: list[dict[int, int]] = []

    # Uniform initial distribution over candidates
    n_init = len(t0_emissions)
    log_uniform = math.log(1.0 / n_init)
    for arc_id, log_emit in t0_emissions.items():
        viterbi[arc_id] = log_uniform + log_emit

    backptr.append({})  # no predecessor at t=0

    n_skipped = 0

    # --- Recursion (t = 1 … T-1) ---
    for t in range(1, T):
        curr_emissions = emission_table[t]

        if not curr_emissions:
            # No candidates for this GPS point.
            # Carry forward current Viterbi states unchanged.
            # Store an empty backptr so backtracking knows to skip this step
            # (B3 fix: backtracking must `continue`, not `break`, here).
            backptr.append({})
            n_skipped += 1
            log.debug("viterbi_decode: t=%d has no candidates, skipping.", t)
            continue

        prev_arcs = list(viterbi.keys())
        new_viterbi: dict[int, float] = {}
        new_backptr: dict[int, int] = {}

        t_prev = gps_track[t - 1]
        t_curr = gps_track[t]

        for arc_j, log_emit_j in curr_emissions.items():
            best_score = LOG_PROB_FLOOR
            best_prev  = -1

            for arc_i in prev_arcs:
                score_i = viterbi[arc_i]
                if score_i <= LOG_PROB_FLOOR:
                    continue

                log_trans = transition_fn(arc_i, arc_j, t_prev, t_curr)
                total = score_i + log_trans + log_emit_j

                if total > best_score:
                    best_score = total
                    best_prev  = arc_i

            if best_prev == -1:
                # B4 fix: All predecessors were floored — start fresh from
                # this point's emission only.  Leave best_prev = -1 so that
                # backtracking recognises this as a genuine chain break and
                # does NOT follow a spurious arbitrary predecessor.
                # (The old code set best_prev = prev_arcs[0], creating a
                # false backtrack path.)
                best_score = log_emit_j

            new_viterbi[arc_j] = best_score
            new_backptr[arc_j] = best_prev

        viterbi = new_viterbi
        backptr.append(new_backptr)

    # --- Termination ---
    if not viterbi:
        return ViterbiResult(
            arc_sequence=[], log_probability=LOG_PROB_FLOOR,
            n_gps_points=T, n_skipped_points=n_skipped,
            candidate_miss_rate=n_skipped / T,
            success=False,
            message="Viterbi produced no final states; all points were skipped.",
        )

    best_final_arc   = max(viterbi, key=lambda a: viterbi[a])
    best_final_score = viterbi[best_final_arc]

    # --- Backtracking ---
    arc_sequence: list[int] = [best_final_arc]
    current_arc = best_final_arc

    for t in range(len(backptr) - 1, 0, -1):
        bp = backptr[t]

        # B3 fix: empty backptr means this GPS point was skipped (no candidates).
        # `continue` here — do not append an arc, do not update current_arc,
        # and critically do NOT break.  The old code broke here, silently
        # truncating the arc sequence whenever any intermediate point was skipped.
        if not bp:
            continue

        # -1 means this state has no valid predecessor (fresh-start after
        # a floored-out segment, or genuinely disconnected).  Stop here.
        if current_arc not in bp or bp[current_arc] == -1:
            break

        current_arc = bp[current_arc]
        arc_sequence.append(current_arc)

    arc_sequence.reverse()

    # Deduplicate consecutive identical arcs (can happen when GPS points
    # are stationary and all assigned to the same arc — the vehicle has
    # not moved to a new arc, so it should appear only once in the route).
    deduped: list[int] = []
    for arc in arc_sequence:
        if not deduped or arc != deduped[-1]:
            deduped.append(arc)

    miss_rate = n_skipped / T if T > 0 else 0.0

    log.info(
        "viterbi_decode: T=%d, skipped=%d, route_len=%d, log_prob=%.2f",
        T, n_skipped, len(deduped), best_final_score,
    )

    return ViterbiResult(
        arc_sequence=deduped,
        log_probability=best_final_score,
        n_gps_points=T,
        n_skipped_points=n_skipped,
        candidate_miss_rate=miss_rate,
        success=True,
        message="OK",
    )
