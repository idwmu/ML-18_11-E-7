import numpy as np
import pandas as pd
from loaders import load_track, load_nodes, load_arcs

def clean_track(track: pd.DataFrame, nodes_df: pd.DataFrame) -> pd.DataFrame:
    """
    Spec Step 4:
    - Sort by timestamp (already done in load_track)
    - Drop duplicate timestamps (already done in load_track)
    - Flag gaps > 5s
    - Drop points outside network bounding box
    """
    df = track.copy()
    
    # Flag gaps > 5 seconds
    dt = df["t"].diff()
    df["gap_flag"] = (dt > 5.0).astype(np.int8)
    df.loc[df.index[0], "gap_flag"] = 0 # First point has no gap
    
    # Drop points outside network bounding box
    lon_min, lat_min = nodes_df["lon"].min(), nodes_df["lat"].min()
    lon_max, lat_max = nodes_df["lon"].max(), nodes_df["lat"].max()
    
    # Keep points within bounding box
    mask = (df["lon"] >= lon_min) & (df["lon"] <= lon_max) & \
           (df["lat"] >= lat_min) & (df["lat"] <= lat_max)
    
    df = df[mask].reset_index(drop=True)
    df["point_idx"] = df.index
    
    return df
