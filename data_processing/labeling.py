"""
Label rule: a candidate arc is positive if it equals the ground-truth arc for
that point (from .route). ONLY used while building TRAINING labels - the
route file is never touched at inference time.

Assumption (document this in the report): the .route file is a chronological
sequence of arc-ids, but it is not guaranteed to be one arc-per-GPS-point.
We therefore project each point onto the route's own polyline to find which
route arc it's geometrically closest to, using a forward-moving window (size
WINDOW) around a monotonic pointer so the match respects chronological order
instead of just picking the globally nearest route arc (which can mis-assign
on routes that pass close to themselves, e.g. loops / parallel streets).
"""
import numpy as np
from geo import point_to_segment_dist

WINDOW_BACK = 2
WINDOW_FWD = 25


def label_points_to_route(px, py, network, route_arc_ids):
    """
    px, py: projected point coords, shape (n,)
    network: RoadNetwork instance (same trip)
    route_arc_ids: np.ndarray of arc_ids, chronological (from .route)

    Returns:
      true_arc_id: (n,) int array - the route arc assigned to each point
      true_dist:   (n,) float array - point-to-segment distance to that arc (metres)
      route_pos:   (n,) int array - index into route_arc_ids that was matched (diagnostic)
    """
    n = len(px)
    r_ax = network.ax[route_arc_ids]
    r_ay = network.ay[route_arc_ids]
    r_bx = network.bx[route_arc_ids]
    r_by = network.by[route_arc_ids]
    n_route = len(route_arc_ids)

    true_arc_id = np.full(n, -1, dtype=np.int64)
    true_dist = np.full(n, np.inf, dtype=np.float64)
    route_pos = np.full(n, -1, dtype=np.int64)

    pointer = 0
    for i in range(n):
        lo = max(0, pointer - WINDOW_BACK)
        hi = min(n_route, pointer + WINDOW_FWD + 1)
        if lo >= hi:
            lo, hi = 0, n_route
        d, _t = point_to_segment_dist(
            px[i], py[i], r_ax[lo:hi], r_ay[lo:hi], r_bx[lo:hi], r_by[lo:hi]
        )
        j = int(np.argmin(d)) + lo
        true_arc_id[i] = route_arc_ids[j]
        true_dist[i] = d[j - lo]
        route_pos[i] = j
        pointer = max(pointer, j)  # monotonic: don't move window backward

    return true_arc_id, true_dist, route_pos
