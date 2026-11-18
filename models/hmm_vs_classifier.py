"""
Section 5 experiment: classifier-only vs classifier+HMM/Viterbi.

For every trip that ships a reference route, reconstructs the route two ways:
  * classifier-only : argmax candidate per GPS point (no transitions)
  * classifier+HMM  : full Viterbi decoding
and reports route-arc F1 vs the reference route and route continuity (fraction
of consecutive predicted arcs that actually share a node, i.e. are connected).

Run from the repo ROOT:  python models/hmm_vs_classifier.py
"""
import sys
import pathlib

# Put the repo root on sys.path so `import backend` works when run as
# `python models/hmm_vs_classifier.py` (sys.path[0] would otherwise be models/).
_ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

import pandas as pd

# Importing the pipeline also puts data_processing/ on sys.path.
from backend.inference.pipeline import (
    _load_trip, _score_candidates, _build_candidates_by_point,
    _make_transition_fn, _find_partition, list_all_trips,
)
from backend.hmm.emissions import build_emission_table
from backend.hmm.transitions import build_adjacency
from backend.hmm.viterbi import viterbi_decode
from backend.inference.model_loader import load_model
from loaders import load_route

OUT = pathlib.Path("results/hmm_vs_classifier.csv")


def dedup(seq):
    out = []
    for a in seq:
        if not out or a != out[-1]:
            out.append(a)
    return out


def route_f1(pred_seq, true_arcs):
    pred = {int(a) for a in pred_seq}
    true = {int(a) for a in true_arcs}
    if not pred or not true:
        return 0.0
    inter = len(pred & true)
    p = inter / len(pred)
    r = inter / len(true)
    return 2 * p * r / (p + r) if (p + r) > 0 else 0.0


def continuity(seq, arcs_list):
    """Fraction of consecutive arcs that share a node (are connected)."""
    if len(seq) < 2:
        return 1.0
    ok = tot = 0
    for a, b in zip(seq[:-1], seq[1:]):
        tot += 1
        try:
            if set(arcs_list[a]) & set(arcs_list[b]):
                ok += 1
        except Exception:
            pass
    return ok / tot if tot else 1.0


def classifier_only(scores_by_point):
    """Argmax model score per GPS point, then de-dup consecutive arcs."""
    seq = []
    for sc in scores_by_point:
        if sc:
            seq.append(max(sc, key=sc.get))
    return dedup(seq)


def main():
    model = load_model()
    trips = list_all_trips()
    target = trips.get("student_test", [])  # the evaluator/student trips with .route
    print(f"Evaluating {len(target)} trips with model {model.name}", flush=True)

    rows = []
    for tid in target:
        try:
            part = _find_partition(tid)
            true_route = load_route(tid, part)
            if true_route is None or len(true_route) == 0:
                continue

            trip = _load_trip(tid)
            feat_df, track, arcs_list = trip["feature_df"], trip["track"], trip["arcs_list"]

            scores = _score_candidates(feat_df, model)
            cands = _build_candidates_by_point(feat_df)
            emission = build_emission_table(cands, scores)
            adjacency = build_adjacency(arcs_list)

            if len(emission) < len(track):
                emission += [{}] * (len(track) - len(emission))
            else:
                emission = emission[:len(track)]

            clf_seq = classifier_only(scores)
            hmm_seq = viterbi_decode(emission, _make_transition_fn(adjacency), track).arc_sequence

            r = dict(
                trip_id=tid,
                clf_f1=route_f1(clf_seq, true_route),
                hmm_f1=route_f1(hmm_seq, true_route),
                clf_continuity=continuity(clf_seq, arcs_list),
                hmm_continuity=continuity(hmm_seq, arcs_list),
            )
            rows.append(r)
            print(f"{tid}: F1 clf {r['clf_f1']:.3f} -> HMM {r['hmm_f1']:.3f} | "
                  f"continuity clf {r['clf_continuity']:.3f} -> HMM {r['hmm_continuity']:.3f}",
                  flush=True)
        except Exception as e:
            print(f"{tid}: skipped ({e})", flush=True)

    if not rows:
        print("No trips evaluated.")
        return

    df = pd.DataFrame(rows)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False)
    print(f"\n=== Mean over {len(df)} trips ===")
    print(f"  Route F1:    classifier-only {df.clf_f1.mean():.4f}   classifier+HMM {df.hmm_f1.mean():.4f}")
    print(f"  Continuity:  classifier-only {df.clf_continuity.mean():.4f}   classifier+HMM {df.hmm_continuity.mean():.4f}")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
