"""
PathSnap - System A (Data & Features)
Loaders: functions returning track points, nodes, arcs, route per trip id.

File formats (confirmed from Dataset Exploration.pdf + raw inspection):
  .track  -> lon, lat, timestamp_seconds   (whitespace-sep, no header, ~1Hz, NOT always evenly spaced)
  .nodes  -> lon, lat                      (whitespace-sep, no header, 0-based row index = node_id)
  .arcs   -> from_node_id, to_node_id      (whitespace/tab-sep, no header, 0-based row index = arc_id)
  .route  -> arc_id                        (one per line, chronological sequence of ground-truth arcs)
"""
from pathlib import Path
import numpy as np
import pandas as pd

DATA_ROOT = Path("datasets")


def trip_dir(trip_id: str, partition: str) -> Path:
    """partition: 'train' or 'student_test'"""
    return DATA_ROOT / partition / trip_id


def load_track(trip_id: str, partition: str) -> pd.DataFrame:
    """Returns DataFrame[lon, lat, t] sorted by time, t in seconds (float).
    Drops duplicate timestamps."""
    f = trip_dir(trip_id, partition) / f"{trip_id}.track"
    df = pd.read_csv(f, sep=r"\s+", header=None, names=["lon", "lat", "t"])
    df = df.sort_values("t").reset_index(drop=True)
    df = df.drop_duplicates(subset="t").reset_index(drop=True)
    df["point_idx"] = df.index
    return df


def load_nodes(trip_id: str, partition: str) -> pd.DataFrame:
    """Returns DataFrame[lon, lat] indexed by node_id (0-based row index)."""
    f = trip_dir(trip_id, partition) / f"{trip_id}.nodes"
    df = pd.read_csv(f, sep=r"\s+", header=None, names=["lon", "lat"])
    df.index.name = "node_id"
    return df


def load_arcs(trip_id: str, partition: str) -> pd.DataFrame:
    """Returns DataFrame[from_node, to_node] indexed by arc_id (0-based row index)."""
    f = trip_dir(trip_id, partition) / f"{trip_id}.arcs"
    df = pd.read_csv(f, sep=r"\s+", header=None, names=["from_node", "to_node"])
    df.index.name = "arc_id"
    return df


def load_route(trip_id: str, partition: str) -> np.ndarray | None:
    """Returns array of arc_ids (chronological ground truth), or None if not supplied
    (evaluator-test trips won't have a .route file available to us)."""
    f = trip_dir(trip_id, partition) / f"{trip_id}.route"
    if not f.exists():
        return None
    return pd.read_csv(f, header=None)[0].to_numpy()


def list_trip_ids(partition: str) -> list[str]:
    base = DATA_ROOT / partition
    return sorted(p.name for p in base.iterdir() if p.is_dir())


if __name__ == "__main__":
    tid = "00000002"
    track = load_track(tid, "train")
    nodes = load_nodes(tid, "train")
    arcs = load_arcs(tid, "train")
    route = load_route(tid, "train")
    print("track:", track.shape, track.head(2).to_dict("records"))
    print("nodes:", nodes.shape)
    print("arcs:", arcs.shape)
    print("route:", None if route is None else (route.shape, route[:5]))
    print("train trips:", len(list_trip_ids("train")))
    print("student_test trips:", len(list_trip_ids("student_test")))
