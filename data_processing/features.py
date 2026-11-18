"""
Feature columns (fixed, exact order - this is the data contract, do not reorder
once teammates B/C start consuming features_*.parquet):

  trip_id, point_idx, candidate_rank, arc_id,
  dist_m, bearing_diff_deg, arc_length_m, speed_mps, heading_deg,
  arc_degree, proj_t, dist_start_node_m, dist_end_node_m,
  label   (only present when ground truth available, i.e. train/student_test)
  candidate_miss (only present when ground truth available)

Model I/O: input = feature row per candidate, output = P(correct candidate).
"""
import numpy as np
import pandas as pd

from loaders import load_track, load_nodes, load_arcs, load_route
from cleaning import clean_track
from candidates import RoadNetwork, K_DEFAULT
from labeling import label_points_to_route
from geo import bearing, angular_diff, point_to_segment_dist

FEATURE_COLUMNS = [
    "trip_id", "point_idx", "candidate_rank", "arc_id",
    "dist_m", "bearing_diff_deg", "arc_length_m", "speed_mps", "heading_deg",
    "arc_degree", "proj_t", "dist_start_node_m", "dist_end_node_m",
]


def _vehicle_heading_and_speed(x, y, t):
    """Per-point heading (bearing of travel) and speed, using the segment to
    the NEXT point (last point repeats the previous segment's heading/speed)."""
    n = len(x)
    heading = np.zeros(n)
    speed = np.zeros(n)
    if n >= 2:
        dx = np.diff(x)
        dy = np.diff(y)
        dt = np.diff(t)
        dt_safe = np.where(dt <= 0, np.nan, dt)
        seg_dist = np.sqrt(dx ** 2 + dy ** 2)
        seg_bearing = bearing(x[:-1], y[:-1], x[1:], y[1:])
        seg_speed = seg_dist / dt_safe
        heading[:-1] = seg_bearing
        heading[-1] = seg_bearing[-1] if n > 1 else 0.0
        speed[:-1] = np.nan_to_num(seg_speed, nan=0.0)
        speed[-1] = speed[-2] if n > 1 else 0.0
    return heading, speed


def build_trip_features(trip_id: str, partition: str, k: int = K_DEFAULT):
    """Returns a DataFrame with FEATURE_COLUMNS (+ 'label' and 'candidate_miss' if route available)."""
    track_raw = load_track(trip_id, partition)
    nodes = load_nodes(trip_id, partition)
    arcs = load_arcs(trip_id, partition)
    route = load_route(trip_id, partition)  # None for evaluator-test

    track = clean_track(track_raw, nodes)

    net = RoadNetwork(nodes, arcs)
    px, py = net.project_points(track["lon"].to_numpy(), track["lat"].to_numpy())
    px, py = np.asarray(px), np.asarray(py)

    heading, speed = _vehicle_heading_and_speed(px, py, track["t"].to_numpy())

    cand_arc_ids, cand_dist = net.k_nearest_candidates(px, py, k=k)
    n_points = len(track)

    arc_bearing_all = bearing(net.ax, net.ay, net.bx, net.by)  # per arc_id

    rows_arc_id = cand_arc_ids.reshape(-1)
    rows_dist = cand_dist.reshape(-1)
    point_idx_rep = np.repeat(np.arange(n_points), k)
    rank_rep = np.tile(np.arange(k), n_points)
    heading_rep = np.repeat(heading, k)
    speed_rep = np.repeat(speed, k)

    valid = rows_arc_id >= 0
    df = pd.DataFrame({
        "trip_id": trip_id,
        "point_idx": point_idx_rep[valid],
        "candidate_rank": rank_rep[valid],
        "arc_id": rows_arc_id[valid],
        "dist_m": rows_dist[valid],
        "arc_length_m": net.arc_len[rows_arc_id[valid]],
        "speed_mps": speed_rep[valid],
        "heading_deg": heading_rep[valid],
        "arc_degree": net.arc_degree[rows_arc_id[valid]],
    })
    df["bearing_diff_deg"] = angular_diff(
        arc_bearing_all[df["arc_id"].to_numpy()], df["heading_deg"].to_numpy()
    )

    # New Features from Spec Step 7
    # Along-arc position (t value from projection)
    # Distance to endpoints separately

    # Vectorized projection for selected rows
    p_x = px[df["point_idx"].to_numpy()]
    p_y = py[df["point_idx"].to_numpy()]
    a_x = net.ax[df["arc_id"].to_numpy()]
    a_y = net.ay[df["arc_id"].to_numpy()]
    b_x = net.bx[df["arc_id"].to_numpy()]
    b_y = net.by[df["arc_id"].to_numpy()]

    # Call point_to_segment_dist to get t and dist
    _, t_vals = point_to_segment_dist(p_x, p_y, a_x, a_y, b_x, b_y)
    df["proj_t"] = t_vals

    # Dist to start node (a) and end node (b)
    df["dist_start_node_m"] = np.sqrt((p_x - a_x)**2 + (p_y - a_y)**2)
    df["dist_end_node_m"] = np.sqrt((p_x - b_x)**2 + (p_y - b_y)**2)

    df = df[FEATURE_COLUMNS]

    if route is not None and len(route) > 0:
        route = route[route < net.n_arcs]  # guard against stray out-of-range ids
        true_arc_id, _true_dist, _route_pos = label_points_to_route(px, py, net, route)
        true_map = pd.Series(true_arc_id, index=np.arange(n_points))

        # Label = 1 if candidate arc id == true arc id
        df["label"] = (df["arc_id"].to_numpy() == true_map.loc[df["point_idx"]].to_numpy()).astype(np.int8)

        # Candidate-miss flag (Spec Step 6)
        # True if the true arc for this point is not in its top-k candidate set
        point_has_positive = df.groupby("point_idx")["label"].max().to_numpy()
        # Re-map back to the candidate rows
        df["candidate_miss"] = (1 - point_has_positive[df["point_idx"].to_numpy()]).astype(np.int8)

    return df


if __name__ == "__main__":
    df = build_trip_features("00000002", "train")
    print(df.shape)
    print(df.head(10))
    print("label rate (should be roughly 1/k per point if always findable):",
          df["label"].mean(), "| points with a positive candidate:",
          df.groupby("point_idx")["label"].max().mean())
