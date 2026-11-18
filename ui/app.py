import streamlit as st
import pandas as pd
import numpy as np
import folium
from streamlit_folium import st_folium
import plotly.express as px
import requests
from datetime import datetime

# Page configuration
st.set_page_config(
    page_title="PathSnap Evaluator",
    page_icon="🗺️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Constants
API_BASE_URL = "http://localhost:8000"  # Default, can be changed in UI

# Initialize session state
if "raw_data" not in st.session_state:
    st.session_state.raw_data = None
if "matched_data" not in st.session_state:
    st.session_state.matched_data = None
if "trip_list" not in st.session_state:
    st.session_state.trip_list = []

# --- API Client ---
def fetch_trips():
    """Calls GET /trips to get the list of 10 evaluation trips."""
    try:
        # In a real scenario: response = requests.get(f"{API_BASE_URL}/trips", timeout=2)
        # Mocking response as per Spec: 10 evaluator-test trips
        return [f"000000{i:02d}" for i in range(1, 11)]
    except Exception as e:
        st.error(f"API Error fetching trips: {e}")
        return []

def predict_trip(trip_id):
    """Calls POST /predict to get the matched route."""
    try:
        # In a real scenario: 
        # response = requests.post(f"{API_BASE_URL}/predict", json={"trip_id": trip_id}, timeout=10)
        # response.raise_for_status()
        # return response.json()
        
        # Mocking response based on C · API.md: {predicted_arc_ids, points, metrics}
        # points: list of [lat, lon]
        n_points = 50
        points = np.linspace([47.42, 11.38], [47.45, 11.41], n_points).tolist()
        # Add some noise to make it look like a path
        points = [[p[0] + np.random.normal(0, 0.0001), p[1] + np.random.normal(0, 0.0001)] for p in points]
        
        return {
            "predicted_arc_ids": [f"arc_{np.random.randint(100, 500)}" for _ in range(n_points)],
            "points": points,
            "metrics": {
                "accuracy": 0.92,
                "mean_lateral_error": 1.5,
                "f1": 0.88,
                "total_dist_km": 1.2
            }
        }
    except Exception as e:
        st.error(f"API Error predicting trip {trip_id}: {e}")
        return None

# --- Navigation ---
st.sidebar.title("🗺️ PathSnap")
st.sidebar.markdown("Distributed GPS Map-Matching")
st.sidebar.divider()

# API Config in sidebar
api_url = st.sidebar.text_input("API Base URL", value=API_BASE_URL)

page = st.sidebar.radio(
    "Navigation",
    ["Trip Selection", "Matching Result", "Metrics & Analysis"]
)

# --- Page 1: Trip Selection ---
if page == "Trip Selection":
    st.header("🚀 Trip Selection")
    
    if not st.session_state.trip_list:
        if st.button("Fetch Evaluation Trips"):
            with st.spinner("Calling /trips..."):
                st.session_state.trip_list = fetch_trips()
                st.rerun()
    
    if st.session_state.trip_list:
        selected_trip = st.selectbox("Select an Evaluator Trip", st.session_state.trip_list)
        
        if st.button("Run Map Matching"):
            with st.spinner(f"Requesting prediction for {selected_trip}..."):
                res = predict_trip(selected_trip)
                if res:
                    st.session_state.raw_data = None # Clear previous
                    # Create a DataFrame for the matched points
                    matched_df = pd.DataFrame(res["points"], columns=["lat", "lon"])
                    matched_df["arc_id"] = res["predicted_arc_ids"]
                    st.session_state.matched_data = matched_df
                    st.session_state.metrics = res["metrics"]
                    st.session_state.current_trip = selected_trip
                    st.success(f"Successfully matched trip {selected_trip}!")
                    st.info("Navigate to 'Matching Result' to see the map.")

# --- Page 2: Matching Result ---
elif page == "Matching Result":
    st.header("🗺️ Map View")
    
    if st.session_state.matched_data is None:
        st.warning("Please select and run a trip first.")
    else:
        df = st.session_state.matched_data
        trip_id = st.session_state.get("current_trip", "Unknown")
        st.subheader(f"Trip: {trip_id}")
        
        # Map configuration
        m = folium.Map(
            location=[df['lat'].mean(), df['lon'].mean()],
            zoom_start=14,
            tiles="CartoDB dark_matter"
        )
        
        # Matched Path
        coords = df[['lat', 'lon']].values.tolist()
        folium.PolyLine(coords, color="blue", weight=4, opacity=0.8, label="Predicted Route").add_to(m)
        
        # Start/End
        folium.Marker([coords[0][0], coords[0][1]], popup="Start", icon=folium.Icon(color="green")).add_to(m)
        folium.Marker([coords[-1][0], coords[-1][1]], popup="End", icon=folium.Icon(color="red")).add_to(m)
        
        st_folium(m, width=1200, height=600)
        
        st.subheader("Predicted Segments")
        st.dataframe(df, use_container_width=True)

# --- Page 3: Metrics & Analysis ---
elif page == "Metrics & Analysis":
    st.header("📈 Performance Metrics")
    
    if "metrics" not in st.session_state:
        st.warning("Please run a trip first to see metrics.")
    else:
        m = st.session_state.metrics
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Accuracy", f"{m.get('accuracy', 0):.1%}")
        col2.metric("Mean Lateral Error", f"{m.get('mean_lateral_error', 0):.2f} m")
        col3.metric("F1 Score", f"{m.get('f1', 0):.2f}")
        
        st.divider()
        st.metric("Total Matched Distance", f"{m.get('total_dist_km', 0):.2f} km")
        
        st.info("Metrics are calculated server-side against hidden reference data.")

