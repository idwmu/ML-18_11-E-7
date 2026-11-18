"""
PathSnap – Emission Probabilities
====================================
Converts raw candidate scores (from the ML model) into log-emission
probabilities for the HMM.

Theory
------
The emission probability p(GPS_obs_t | state=arc_i) represents how likely
it is to observe the GPS measurement at time t given that the vehicle is on
arc i.

We receive **probabilities** (output of predict_proba()[:, 1], values in
[0, 1]) from the scoring model for each (GPS point, candidate arc) pair.

Conversion pipeline per GPS observation
-----------------------------------------
  1.  For each candidate arc i:
        model_log_i = log( max(p_i, _PROB_FLOOR) )   ← safe log of probability
        If p_i is missing (arc not in model_scores), fall back to the
        distance-prior log-value as a neutral substitute.

  2.  Normalise across all candidates using log-softmax so that the per-GPS
        distribution sums to 1 in probability space.

  3.  Compute an analogous normalised log-prior from the Gaussian distance term.

  4.  Blend model and distance terms in log-space with configurable weights.

  5.  Final log-softmax normalisation to produce a valid log-distribution.

IMPORTANT: Do NOT pass logits to this module. The model scores must be
probabilities in [0, 1]. Applying log-softmax directly to raw probabilities
would treat 0.9 as a logit (≫ a logit of 0), grossly distorting the scores.
This module applies log() first, then log-softmax.

Additionally, a distance-based prior is blended in to handle cases where
the model is absent or returns flat scores.
"""

from __future__ import annotations

import math
import logging
from typing import Optional

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Gaussian sigma for distance-based prior (metres).
# A GPS point 20 m from an arc gets weight ~exp(-(20^2)/(2*20^2)) ≈ 0.6.
DISTANCE_SIGMA_M: float = 20.0

# Minimum log-probability floor to avoid -inf in the DP.
LOG_PROB_FLOOR: float = -1e9

# Blend weight for model score vs distance prior.
# 1.0 = use model only. 0.0 = use distance prior only.
MODEL_WEIGHT: float = 0.7
DISTANCE_WEIGHT: float = 1.0 - MODEL_WEIGHT

# Minimum probability before log() to avoid log(0) = -inf.
# log(1e-9) ≈ -20.7, which is a safely distinguishable floor relative to
# LOG_PROB_FLOOR of -1e9.  This floor is applied to individual candidate
# scores, not to the final emission value.
_PROB_FLOOR: float = 1e-9


# ---------------------------------------------------------------------------
# Distance-based prior
# ---------------------------------------------------------------------------

def distance_log_prior(dist_m: float, sigma: float = DISTANCE_SIGMA_M) -> float:
    """
    Log-Gaussian prior on perpendicular distance from GPS point to arc.

    p(obs | arc) ∝ exp(-dist^2 / (2 * sigma^2))
    log p = -dist^2 / (2 * sigma^2)   (unnormalised; constants cancel in Viterbi)

    Parameters
    ----------
    dist_m : float – perpendicular distance in metres (must be >= 0)
    sigma  : float – standard deviation in metres

    Returns
    -------
    float – log-probability (non-positive; 0 at dist=0)
    """
    if dist_m < 0:
        return LOG_PROB_FLOOR
    log_p = -(dist_m ** 2) / (2.0 * sigma ** 2)
    return max(log_p, LOG_PROB_FLOOR)


# ---------------------------------------------------------------------------
# Probability → log-probability conversion  (FIX for Bug B1 / B2)
# ---------------------------------------------------------------------------

def _prob_to_log(p: float) -> float:
    """
    Safely convert a model probability p ∈ [0, 1] to a log-probability.

    - Clamps p to [_PROB_FLOOR, 1.0] before log() to avoid log(0) = -inf
      and log(>1) = positive (the latter should never occur with a valid model).
    - Non-finite inputs (NaN, inf) are replaced with the floor value so they
      do not pollute the downstream HMM computation.

    Parameters
    ----------
    p : float – probability output of model.predict_proba()[:, 1],
                expected in [0, 1].

    Returns
    -------
    float – log(p), ≥ log(_PROB_FLOOR) ≈ -20.7
    """
    if not math.isfinite(p):
        return math.log(_PROB_FLOOR)
    return math.log(max(min(float(p), 1.0), _PROB_FLOOR))


# ---------------------------------------------------------------------------
# Log-softmax normalisation
# ---------------------------------------------------------------------------

def _log_softmax(scores: list[float]) -> list[float]:
    """
    Numerically stable log-softmax over a list of log-scores.

    log_softmax(x_i) = x_i - log(sum(exp(x_j)))
                     = x_i - max(x) - log(sum(exp(x_j - max(x))))

    Parameters
    ----------
    scores : list[float] – unnormalised log-scores (may be negative)

    Returns
    -------
    list[float] – normalised log-probabilities; sum(exp(output)) = 1
    """
    if not scores:
        return []
    max_s = max(scores)
    log_sum_exp = math.log(sum(math.exp(s - max_s) for s in scores)) + max_s
    return [s - log_sum_exp for s in scores]


# ---------------------------------------------------------------------------
# Main emission calculator
# ---------------------------------------------------------------------------

def compute_emission_log_probs(
    candidates: list[dict],
    model_scores: Optional[dict[int, float]] = None,
    sigma: float = DISTANCE_SIGMA_M,
    model_weight: float = MODEL_WEIGHT,
) -> dict[int, float]:
    """
    Compute log-emission probabilities for one GPS observation's candidates.

    Parameters
    ----------
    candidates : list[dict]
        List of candidate dicts for one GPS point. Each dict must contain:
            arc_id : int   – candidate arc ID
            dist_m : float – perpendicular distance in metres
        Additional keys (e.g. features) are ignored here.

    model_scores : dict[int, float] or None
        arc_id → model probability (output of predict_proba()[:, 1],
        **must be in [0, 1]**).
        If None or empty, falls back to distance-only prior.
        Out-of-range or non-finite scores are warned about and clamped.

    sigma : float
        Gaussian sigma for the distance prior (metres).

    model_weight : float
        Blend weight for model score (1 − model_weight used for distance prior).

    Returns
    -------
    dict[int, float]
        arc_id → log-emission probability.
        Values are log-normalised over the candidate set so that
        log(sum(exp(v))) ≈ 0.  Empty dict if no candidates.

    Notes
    -----
    - Model scores are PROBABILITIES, not logits. This function converts them
      to log-probabilities using log(p) before normalisation.  Applying
      log-softmax directly to raw probabilities would treat probability values
      as unnormalised logits, which is incorrect.
    - If a candidate's arc_id is missing from model_scores, its model
      component falls back to the distance prior log-value.
    - Log-probabilities are floored at LOG_PROB_FLOOR to avoid -inf.
    """
    if not candidates:
        return {}

    arc_ids = [c["arc_id"] for c in candidates]
    dists_m = [c["dist_m"]  for c in candidates]

    # --- Distance log-prior ---
    dist_log_priors = [distance_log_prior(d, sigma) for d in dists_m]

    # --- Model log-probabilities (B1 fix: convert p→log(p), not log-softmax of p) ---
    if model_scores:
        raw_model = [model_scores.get(arc_id, None) for arc_id in arc_ids]

        model_log_raw: list[float] = []
        for arc_id, rm, dp in zip(arc_ids, raw_model, dist_log_priors):
            if rm is None:
                # Missing score for this arc: fall back to distance prior
                model_log_raw.append(dp)
            else:
                # B2 fix: validate that the score is a probability in [0, 1]
                if not math.isfinite(rm) or rm < 0.0 or rm > 1.0:
                    log.warning(
                        "arc %d: model score %.6g is outside [0, 1] "
                        "or non-finite; clamping to [%g, 1].",
                        arc_id, rm, _PROB_FLOOR,
                    )
                # Convert probability → log-probability (safe, with floor)
                model_log_raw.append(_prob_to_log(rm))

        # Normalise the log-probabilities across candidates
        model_log = _log_softmax(model_log_raw)
    else:
        # No model → pure distance prior (normalised)
        model_log = _log_softmax(dist_log_priors)

    # --- Blend model and distance ---
    dist_log_norm = _log_softmax(dist_log_priors)

    blended: list[float] = []
    for m, d in zip(model_log, dist_log_norm):
        # Log-space blend: log(w1*exp(m) + w2*exp(d))
        # Numerically stable: factor out max_v
        max_v = max(m, d)
        combined = (
            max_v
            + math.log(
                model_weight         * math.exp(m - max_v)
                + (1.0 - model_weight) * math.exp(d - max_v)
            )
        )
        blended.append(max(combined, LOG_PROB_FLOOR))

    # Final normalisation over candidate set
    normalised = _log_softmax(blended)

    return {arc_id: lp for arc_id, lp in zip(arc_ids, normalised)}


# ---------------------------------------------------------------------------
# Convenience: build emission table for all GPS points in a trip
# ---------------------------------------------------------------------------

def build_emission_table(
    candidates_by_point: list[list[dict]],
    model_scores_by_point: Optional[list[Optional[dict[int, float]]]] = None,
    sigma: float = DISTANCE_SIGMA_M,
    model_weight: float = MODEL_WEIGHT,
) -> list[dict[int, float]]:
    """
    Build a list of per-GPS-point emission log-probability dicts.

    Parameters
    ----------
    candidates_by_point : list[list[dict]]
        One inner list per GPS point, each containing candidate dicts
        with keys: arc_id, dist_m.

    model_scores_by_point : list[dict[int, float] | None] or None
        Parallel list of model scores per GPS point.
        Each dict maps arc_id → probability in [0, 1].
        Pass None to use distance-only priors for all points.

    sigma : float
    model_weight : float

    Returns
    -------
    list[dict[int, float]]
        One dict per GPS point: arc_id → log-emission probability.
    """
    n = len(candidates_by_point)
    scores_list = model_scores_by_point or ([None] * n)

    if len(scores_list) != n:
        log.warning(
            "build_emission_table: model_scores_by_point length %d != "
            "candidates length %d; padding with None for mismatched points.",
            len(scores_list), n,
        )
        # Pad or truncate to match candidates length
        scores_list = list(scores_list)
        if len(scores_list) < n:
            scores_list.extend([None] * (n - len(scores_list)))
        else:
            scores_list = scores_list[:n]

    return [
        compute_emission_log_probs(
            cands, model_sc, sigma=sigma, model_weight=model_weight
        )
        for cands, model_sc in zip(candidates_by_point, scores_list)
    ]
