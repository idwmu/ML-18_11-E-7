"""
Candidate set: for each GPS point, the k nearest arcs (k ~= 5-8), in projected
metres (UTM), never raw lon/lat degrees.
"""
import numpy as np
from scipy.spatial import cKDTree
from geo import make_projector, point_to_segment_dist

K_DEFAULT = 6


class RoadNetwork:
    """Projected road-network geometry + connectivity for one trip."""

    def __init__(self, nodes_df, arcs_df):
        sample = nodes_df.iloc[0]
        self.project, self.epsg = make_projector(sample["lon"], sample["lat"])
        nx, ny = self.project(nodes_df["lon"].to_numpy(), nodes_df["lat"].to_numpy())
        self.node_x, self.node_y = nx, ny  # indexed by node_id

        self.from_node = arcs_df["from_node"].to_numpy()
        self.to_node = arcs_df["to_node"].to_numpy()
        self.ax, self.ay = self.node_x[self.from_node], self.node_y[self.from_node]
        self.bx, self.by = self.node_x[self.to_node], self.node_y[self.to_node]
        self.arc_len = np.sqrt((self.bx - self.ax) ** 2 + (self.by - self.ay) ** 2)
        self.mid_x = (self.ax + self.bx) / 2
        self.mid_y = (self.ay + self.by) / 2
        self.n_arcs = len(arcs_df)

        # node degree = number of arc endpoints touching that node (network given
        # as directed pairs, both directions listed separately in practice)
        deg = np.zeros(len(nodes_df), dtype=np.int64)
        np.add.at(deg, self.from_node, 1)
        np.add.at(deg, self.to_node, 1)
        self.node_degree = deg
        # arc "degree" feature = connectivity at its two endpoints
        self.arc_degree = self.node_degree[self.from_node] + self.node_degree[self.to_node]

        self.tree = cKDTree(np.column_stack([self.mid_x, self.mid_y]))

    def project_points(self, lon, lat):
        return self.project(lon, lat)

    def k_nearest_candidates(self, px, py, k=K_DEFAULT):
        """For arrays of projected points (px, py), returns:
        cand_arc_ids: (n_points, k) int array
        cand_dist:    (n_points, k) float array, true point-to-segment distance (metres)
        Uses midpoint-KDTree over-fetch (3x k) then refines with exact segment distance
        and re-sorts, since nearest-midpoint != nearest-segment in general.
        """
        n = len(px)
        overfetch = min(max(k * 3, 12), self.n_arcs)
        _, idx = self.tree.query(np.column_stack([px, py]), k=overfetch)
        if overfetch == 1:
            idx = idx.reshape(-1, 1)

        cand_arc_ids = np.full((n, k), -1, dtype=np.int64)
        cand_dist = np.full((n, k), np.inf, dtype=np.float64)

        ax, ay, bx, by = self.ax[idx], self.ay[idx], self.bx[idx], self.by[idx]
        dist, _t = point_to_segment_dist(
            px[:, None], py[:, None], ax, ay, bx, by
        )
        order = np.argsort(dist, axis=1)[:, :k]
        rows = np.arange(n)[:, None]
        cand_arc_ids = idx[rows, order]
        cand_dist = dist[rows, order]
        return cand_arc_ids, cand_dist
