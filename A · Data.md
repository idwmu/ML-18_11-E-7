# A · Data & Features

Critical path for 11:30. Your feature table unblocks B, C and D, so ship a usable version early and freeze the schema. Reuse-friendly code: C imports your loaders and candidate/feature functions inside the API.

## Workflow

1. **Verify the dataset structure.** List the three partitions (training/80, student-test/10, evaluator-test/10). `head` one `.track`, `.nodes`, `.arcs`, `.route` and confirm columns: track = lon lat ts; nodes = lon lat (row index = node id); arcs = from\_node to\_node (row index = arc id); route = sequence of arc ids. Do not touch the evaluator-test partition.
2. **Write `loaders.py`.** `read_track(trip)` returns an Nx3 array (lon, lat, ts); `read_nodes`, `read_arcs`, `read_route`. Build an arc table mapping arc id to (from\_node, to\_node, both endpoint coords).
3. **Project to metres.** Pick the UTM zone from the data centroid and use `pyproj.Transformer` EPSG:4326 to that UTM. Do every distance and bearing calculation in metres, never in degrees.
4. **Clean.** Sort each track by timestamp, drop duplicate timestamps, flag gaps over a threshold (e.g. 5 s), drop points with implausible coordinates (outside the network bbox).
5. **Candidate generation.** Build each arc as a segment between its two node coords. For each GPS point, get the k nearest arcs (k = 5-8) with a spatial index (`shapely.STRtree`, or a `cKDTree` on arc midpoints then exact point-to-segment distance). Store rows of (trip, point\_idx, cand\_arc\_id, perp\_dist\_m).
6. **Labelling.** `.route` is the ordered list of true arc ids. Assign each GPS point to the route arc it is closest to along the route (project the point onto the route polyline). A candidate is positive (label = 1) if its arc id equals that true arc. Record a candidate-miss flag when the true arc is absent from the candidate set.
7. **Features per (point, candidate).** Perpendicular distance to the arc, bearing difference (GPS heading vs arc direction), arc length, along-arc position of the projection, point speed and heading from neighbours, arc degree (node connectivity), distance to each arc endpoint. Freeze the exact column list and order.
8. **Emit tables.** `features_train.parquet` and `features_studenttest.parquet`, each with the feature columns + `trip_id` (the GroupKFold group) + `label`. Commit a one-line schema note.

**Deliverable by \~11:15:** frozen feature tables + `loaders.py` + candidate/feature code, pushed. Tell B the moment the tables land.

**Handoff:** keep the loader and candidate/feature functions importable and side-effect free so C calls them in the API path.

## Setup (macOS)

- `python3 -m venv .venv && source .venv/bin/activate`
- `pip install numpy pandas pyproj shapely scipy scikit-learn pyarrow`
- Prefer `shapely.STRtree` for the spatial index so you avoid rtree. If you want rtree: `brew install spatialindex` then `pip install rtree`.
- Apple Silicon has arm64 wheels for all of these, so no special steps.
