"""
Geographic helper functions. Everything downstream works in projected metres
(UTM), never raw lon/lat degrees - degree-Euclidean distance is not physically
meaningful at this latitude (~47.6N, Innsbruck/Tyrol area, lon ~11E).
"""
import numpy as np
from pyproj import Transformer


def utm_epsg_for(lon: float, lat: float) -> int:
    zone = int((lon + 180) / 6) + 1
    return (32600 if lat >= 0 else 32700) + zone


def make_projector(sample_lon: float, sample_lat: float):
    """Returns a function project(lon_array, lat_array) -> (x, y) in metres."""
    epsg = utm_epsg_for(sample_lon, sample_lat)
    transformer = Transformer.from_crs("EPSG:4326", f"EPSG:{epsg}", always_xy=True)

    def project(lon, lat):
        x, y = transformer.transform(np.asarray(lon), np.asarray(lat))
        return x, y

    return project, epsg


def point_to_segment_dist(px, py, ax, ay, bx, by):
    """Vectorised point-to-segment distance in metres.
    p: point(s), a/b: segment endpoints (can be arrays, broadcast against p)."""
    px, py = np.asarray(px), np.asarray(py)
    ax, ay, bx, by = map(np.asarray, (ax, ay, bx, by))
    abx, aby = bx - ax, by - ay
    apx, apy = px - ax, py - ay
    ab_len2 = abx * abx + aby * aby
    ab_len2_safe = np.where(ab_len2 == 0, 1.0, ab_len2)
    t = (apx * abx + apy * aby) / ab_len2_safe
    t = np.clip(t, 0.0, 1.0)
    t = np.where(ab_len2 == 0, 0.0, t)
    projx = ax + t * abx
    projy = ay + t * aby
    dx, dy = px - projx, py - projy
    dist = np.sqrt(dx * dx + dy * dy)
    return dist, t


def bearing(x1, y1, x2, y2):
    """Bearing in degrees [0,360) from point1->point2, in projected (x=east,y=north) metres."""
    dx, dy = x2 - x1, y2 - y1
    ang = np.degrees(np.arctan2(dx, dy))  # 0=north, 90=east
    return np.mod(ang, 360.0)


def angular_diff(a, b):
    """Smallest absolute difference between two bearings in degrees, range [0,180]."""
    d = np.abs(np.asarray(a) - np.asarray(b)) % 360.0
    return np.minimum(d, 360.0 - d)
