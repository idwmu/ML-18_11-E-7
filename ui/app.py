"""
PathSnap - Evaluator UI (System D) — single page.
Wired to System C's real API. No mock data.

Accuracy: uses the API's route_f1 if present (server-side, compliant);
otherwise computes route precision/recall/F1 locally from the trip's
reference .route in datasets/ (works for student_test/train trips that
ship a .route). Reserved evaluator trips have no .route -> accuracy n/a.
"""
from pathlib import Path
import numpy as np
import streamlit as st
import pandas as pd
import folium
from streamlit_folium import st_folium
import requests

st.set_page_config(page_title="PathSnap Evaluator", page_icon="🗺️", layout="wide")

DEFAULT_API = "http://localhost:8000"
# Repo root = two levels up from ui/app.py, so datasets/ resolves regardless of cwd.
DATASETS = Path(__file__).resolve().parent.parent / "datasets"

st.session_state.setdefault("trips", [])
st.session_state.setdefault("result", None)
st.session_state.setdefault("trip_id", None)


# --- API client ------------------------------------------------------------
def fetch_trips(base):
    r = requests.get(f"{base}/trips", timeout=10); r.raise_for_status()
    return r.json().get("student_test", [])

def predict(base, tid):
    r = requests.post(f"{base}/predict", json={"trip_id": tid}, timeout=60); r.raise_for_status()
    return r.json()


# --- Local accuracy fallback (reference route in datasets/) -----------------
def load_route_arcs(trip_id):
    for part in ("evaluator_test", "student_test", "train"):
        p = DATASETS / part / trip_id / f"{trip_id}.route"
        if p.exists():
            try:
                return [int(x) for x in p.read_text().split()]
            except Exception:
                return None
    return None

def load_trip_geom(trip_id):
    """Load nodes [lon,lat] and arcs [from_node,to_node] from the local dataset."""
    for part in ("evaluator_test", "student_test", "train"):
        d = DATASETS / part / trip_id
        if (d / f"{trip_id}.nodes").exists():
            try:
                nodes = np.loadtxt(d / f"{trip_id}.nodes")
                arcs = np.loadtxt(d / f"{trip_id}.arcs", dtype=int)
                if arcs.ndim == 1:
                    arcs = arcs.reshape(-1, 2)
                return nodes, arcs
            except Exception:
                return None, None
    return None, None


def arc_segments(arc_ids, nodes, arcs):
    """Return [[lat,lon],[lat,lon]] segments for each arc id (folium order)."""
    segs = []
    for aid in arc_ids:
        aid = int(aid)
        if 0 <= aid < len(arcs):
            fn, tn = arcs[aid]
            if 0 <= fn < len(nodes) and 0 <= tn < len(nodes):
                segs.append([[nodes[fn][1], nodes[fn][0]], [nodes[tn][1], nodes[tn][0]]])
    return segs


def route_scores(pred_ids, true_ids):
    pred, true = {int(a) for a in pred_ids}, {int(a) for a in true_ids}
    if not pred or not true:
        return None
    inter = len(pred & true)
    prec = inter / len(pred)
    rec  = inter / len(true)
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    return {"f1": f1, "precision": prec, "recall": rec}


# --- Sidebar ---------------------------------------------------------------
st.sidebar.title("🗺️ PathSnap")
st.sidebar.caption("Distributed GPS Map-Matching")
api = st.sidebar.text_input("API Base URL", DEFAULT_API).rstrip("/")
if st.sidebar.button("Check API"):
    try:
        h = requests.get(f"{api}/health", timeout=5).json()
        st.sidebar.success(f"OK · model: {h.get('model_name', '?')}")
    except Exception as e:
        st.sidebar.error(f"Unreachable: {e}")

# --- Main (single page) ----------------------------------------------------
st.title("🗺️ PathSnap — Evaluator")

c1, c2, c3 = st.columns([1, 2, 1])
with c1:
    if st.button("Fetch trips", use_container_width=True):
        try:
            st.session_state.trips = fetch_trips(api)
        except Exception as e:
            st.error(f"Could not reach API: {e}")
with c2:
    trip = st.selectbox("Evaluator trip", st.session_state.trips,
                        index=0 if st.session_state.trips else None,
                        placeholder="Fetch trips first")
with c3:
    st.write("")
    run = st.button("Run map matching", type="primary", use_container_width=True,
                    disabled=not st.session_state.trips)

if run and st.session_state.trips:
    with st.spinner(f"POST /predict for {trip}..."):
        try:
            st.session_state.result = predict(api, trip)
            st.session_state.trip_id = trip
        except Exception as e:
            st.error(f"Prediction failed: {e}")
            st.session_state.result = None

res = st.session_state.result
if res:
    st.divider()
    st.subheader(f"Trip {st.session_state.trip_id}")

    pred_ids = res.get("predicted_arc_ids", [])

    # Accuracy: API value first (server-side), else local reference route
    acc, src = None, ""
    if res.get("route_f1") is not None:
        acc = {"f1": res["route_f1"], "precision": res.get("route_precision", 0.0),
               "recall": res.get("route_recall", 0.0)}
        src = "server-side (API)"
    else:
        true_ids = load_route_arcs(st.session_state.trip_id)
        if true_ids:
            acc = route_scores(pred_ids, true_ids)
            src = "computed from reference route"

    if acc:
        a1, a2, a3 = st.columns(3)
        a1.metric("Route F1", f"{acc['f1']:.1%}")
        a2.metric("Precision", f"{acc['precision']:.1%}")
        a3.metric("Recall", f"{acc['recall']:.1%}")
        st.caption(f"Accuracy {src}. Route-arc overlap vs the reference route.")
    else:
        st.info("No reference route for this trip (reserved evaluator trip) — accuracy n/a.")

    # Honest coverage: true-arc miss (needs ground truth) vs skipped points.
    tmiss = res.get("true_candidate_miss_rate")
    tmiss_disp = f"{tmiss:.1%}" if tmiss is not None else "n/a"
    s1, s2, s3, s4 = st.columns(4)
    s1.metric("GPS points", res.get("n_gps_points", 0))
    s2.metric("Route arcs", res.get("n_route_arcs", 0))
    s3.metric("True-arc miss", tmiss_disp,
              help="Fraction of GPS points whose true road was NOT in the candidate "
                   "set — the honest coverage ceiling (needs reference route).")
    s4.metric("Skipped pts", f"{res.get('candidate_miss_rate', 0):.1%}",
              help="Fraction of points with zero candidates. NOT the same as "
                   "true-arc miss — this is usually ~0.")
    st.caption(f"Scoring model: {res.get('model_name', '?')}")

    # Map: raw GPS observations vs the predicted (matched) route, as separate layers.
    pts = res.get("points", [])
    coords = [[p[1], p[0]] for p in pts]  # API gives [lon,lat]; folium needs [lat,lon]
    if coords:
        m = folium.Map(location=[sum(c[0] for c in coords) / len(coords),
                                 sum(c[1] for c in coords) / len(coords)],
                       zoom_start=14, tiles="OpenStreetMap")

        # Layer 1: raw GPS observations (red)
        raw = folium.FeatureGroup(name="Raw GPS observations", show=True)
        folium.PolyLine(coords, color="red", weight=2, opacity=0.5).add_to(raw)
        for c in coords:
            folium.CircleMarker(c, radius=2, color="red", fill=True, fill_opacity=0.7).add_to(raw)
        raw.add_to(m)

        nodes, arcs = load_trip_geom(st.session_state.trip_id)
        if nodes is not None and arcs is not None:
            # Layer 2: predicted / matched route (blue) from predicted_arc_ids
            pred = folium.FeatureGroup(name="Predicted route (matched arcs)", show=True)
            for seg in arc_segments(pred_ids, nodes, arcs):
                folium.PolyLine(seg, color="blue", weight=5, opacity=0.85).add_to(pred)
            pred.add_to(m)

            # Layer 3: reference route (green), off by default
            ref_ids = load_route_arcs(st.session_state.trip_id)
            if ref_ids:
                ref = folium.FeatureGroup(name="Reference route", show=False)
                for seg in arc_segments(ref_ids, nodes, arcs):
                    folium.PolyLine(seg, color="green", weight=3, opacity=0.7).add_to(ref)
                ref.add_to(m)
        else:
            st.caption("Road geometry not found locally, showing GPS observations only.")

        folium.CircleMarker(coords[0], radius=6, color="green", fill=True, popup="Start").add_to(m)
        folium.CircleMarker(coords[-1], radius=6, color="black", fill=True, popup="End").add_to(m)
        folium.LayerControl(collapsed=False).add_to(m)
        st_folium(m, width=1150, height=540)
        st.caption("Red = raw GPS observations · Blue = predicted/matched route · "
                   "Green (toggle) = reference route.")
    else:
        st.info("No GPS points returned for this trip.")

    with st.expander("Predicted arc sequence"):
        st.dataframe(pd.DataFrame({"predicted_arc_ids": pred_ids}), use_container_width=True)