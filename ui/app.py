"""
PathSnap – Evaluator UI (System D)
Wired to System C's real API. No mock data.

API contract (backend/schemas/request_response.py):
  GET  /trips   -> {"student_test": [str,...], "train": [str,...]}
  POST /predict  body {"trip_id": "00000001"}
       -> {"trip_id", "predicted_arc_ids": [int], "points": [[lon,lat],...],
           "n_gps_points", "n_route_arcs", "candidate_miss_rate", "model_name"}
  GET  /health  -> {"status","model_name","version"}

NOTE: points come back as [lon, lat]; folium needs [lat, lon], so we swap.
"""
import streamlit as st
import pandas as pd
import folium
from streamlit_folium import st_folium
import requests

st.set_page_config(page_title="PathSnap Evaluator", page_icon="🗺️",
                   layout="wide", initial_sidebar_state="expanded")

DEFAULT_API = "http://localhost:8000"

for k, v in {"trip_list": [], "result": None, "current_trip": None}.items():
    st.session_state.setdefault(k, v)

# --- API client (real calls) ----------------------------------------------
def fetch_trips(base_url: str):
    """GET /trips -> the 10 evaluator (student_test) trip IDs."""
    r = requests.get(f"{base_url}/trips", timeout=10)
    r.raise_for_status()
    data = r.json()
    return data.get("student_test", [])

def predict_trip(base_url: str, trip_id: str):
    """POST /predict -> the reconstructed route + metrics."""
    r = requests.post(f"{base_url}/predict", json={"trip_id": trip_id}, timeout=60)
    r.raise_for_status()
    return r.json()

# --- Sidebar ---------------------------------------------------------------
st.sidebar.title("🗺️ PathSnap")
st.sidebar.markdown("Distributed GPS Map-Matching")
st.sidebar.divider()
api_url = st.sidebar.text_input("API Base URL", value=DEFAULT_API).rstrip("/")

# quick health check so you see immediately if the server is reachable
with st.sidebar:
    if st.button("Check API"):
        try:
            h = requests.get(f"{api_url}/health", timeout=5).json()
            st.success(f"OK · model: {h.get('model_name','?')}")
        except Exception as e:
            st.error(f"Unreachable: {e}")

page = st.sidebar.radio("Navigation", ["Trip Selection", "Matching Result", "Metrics"])

# --- Page 1: Trip Selection -----------------------------------------------
if page == "Trip Selection":
    st.header("🚀 Trip Selection")
    if st.button("Fetch Evaluation Trips (/trips)"):
        try:
            st.session_state.trip_list = fetch_trips(api_url)
            st.success(f"Loaded {len(st.session_state.trip_list)} evaluator trips.")
        except Exception as e:
            st.error(f"Could not reach API at {api_url}: {e}")

    if st.session_state.trip_list:
        selected = st.selectbox("Select an Evaluator Trip", st.session_state.trip_list)
        if st.button("Run Map Matching"):
            with st.spinner(f"POST /predict for {selected}..."):
                try:
                    st.session_state.result = predict_trip(api_url, selected)
                    st.session_state.current_trip = selected
                    st.success(f"Matched trip {selected}. See 'Matching Result'.")
                except Exception as e:
                    st.error(f"Prediction failed: {e}")

# --- Page 2: Matching Result ----------------------------------------------
elif page == "Matching Result":
    st.header("🗺️ Map View")
    res = st.session_state.result
    if not res:
        st.warning("Select and run a trip first.")
    else:
        st.subheader(f"Trip: {st.session_state.current_trip}")
        pts = res.get("points", [])          # [[lon, lat], ...]
        coords = [[p[1], p[0]] for p in pts]  # folium needs [lat, lon]

        if coords:
            m = folium.Map(location=[sum(c[0] for c in coords) / len(coords),
                                     sum(c[1] for c in coords) / len(coords)],
                           zoom_start=15, tiles="OpenStreetMap")
            folium.PolyLine(coords, color="blue", weight=4, opacity=0.8,
                            tooltip="Predicted route (GPS observations)").add_to(m)
            folium.CircleMarker(coords[0], radius=6, color="green",
                                fill=True, popup="Start").add_to(m)
            folium.CircleMarker(coords[-1], radius=6, color="red",
                                fill=True, popup="End").add_to(m)
            folium.LayerControl().add_to(m)
            st_folium(m, width=1100, height=600)
        else:
            st.info("No GPS points returned for this trip.")

        st.subheader("Predicted arc sequence")
        st.dataframe(pd.DataFrame({"predicted_arc_ids": res.get("predicted_arc_ids", [])}),
                     use_container_width=True)

# --- Page 3: Metrics (real API fields only) -------------------------------
elif page == "Metrics":
    st.header("📈 Prediction Metrics")
    res = st.session_state.result
    if not res:
        st.warning("Run a trip first.")
    else:
        # Accuracy vs reference route (server-side; None for reserved trips)
        f1 = res.get("route_f1")
        if f1 is not None:
            st.subheader("Accuracy vs reference route")
            a1, a2, a3 = st.columns(3)
            a1.metric("Route F1", f"{f1:.1%}")
            a2.metric("Precision", f"{res.get('route_precision', 0):.1%}")
            a3.metric("Recall", f"{res.get('route_recall', 0):.1%}")
            st.caption("Computed server-side against the trip's reference route. "
                       "Ground truth never leaves the API.")
        else:
            st.info("No reference route for this trip (reserved evaluator trip), "
                    "so accuracy is not available.")

        st.subheader("Route statistics")
        c1, c2, c3 = st.columns(3)
        c1.metric("GPS points", res.get("n_gps_points", 0))
        c2.metric("Route arcs", res.get("n_route_arcs", 0))
        c3.metric("Candidate miss rate", f"{res.get('candidate_miss_rate', 0):.1%}")
        st.metric("Scoring model", res.get("model_name", "?"))
