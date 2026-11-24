"""
Geometric analysis of a hexasphere grid (numpy only).

Everything here is additive: it only reads the public ``HexGrid`` / ``Hexagon``
API.  The functions answer questions such as

* how many tiles does resolution ``n`` have, and which are pentagons?
* what is the exact spherical area of every tile (equal-area quality)?
* which spherical-harmonic degrees does the tile-area field contain?

The last one has a clean theoretical answer.  The grid is invariant under the
rotation group I of the icosahedron, so the area field, viewed as a function on
the sphere, lies in the I-invariant harmonics.  Those only exist in degrees
l = 0, 6, 10, 12, 15, 16, ...  (Molien series 1/((1-t^6)(1-t^10))).  Degrees
1-5, 7-9, 11 must therefore vanish up to sampling error, which
``multipole_spectrum`` lets you verify.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from hexasphere.geometry import latlon_to_X
from hexasphere.hexgrid import Hexagon

# Degrees l <= 20 that carry I-invariant spherical harmonics (dimension 1 each
# except l = 30, which is outside this range).
ICOSAHEDRAL_DEGREES = (0, 6, 10, 12, 15, 16, 18, 20)


def n_tiles(n: int) -> int:
    """Number of tiles of the grid with resolution n: 10 (n+1)^2 + 2."""
    return 10 * (n + 1) ** 2 + 2


def enumerate_tiles(grid, n: int) -> list[Hexagon]:
    """All distinct tiles of resolution ``n`` (pentagons included), once each."""
    m = n + 1
    seen: dict[str, Hexagon] = {}
    for face in range(20):
        for a in range(m + 1):
            for b in range(m + 1):
                c = 2 * m - a - b
                if not 0 <= c <= m:
                    continue
                h = Hexagon(grid, face, (a, b, c), res=m, solve_conflicts=True)
                seen.setdefault(h.to_str_id(), h)
    return list(seen.values())


def tile_center(grid, hexagon: Hexagon) -> np.ndarray:
    """Unit vector of the tile centre."""
    return grid.projection.inv_project(hexagon.P, hexagon.face)


def tile_vertices(hexagon: Hexagon, tol: float = 1e-9) -> np.ndarray:
    """Corner vectors in order; a pentagon yields 5 rows, a hexagon 6."""
    pts = [np.asarray(v, float) for v in hexagon.retrieve_polygon()]
    out = [pts[0]]
    for p in pts[1:]:
        if np.linalg.norm(p - out[-1]) > tol:
            out.append(p)
    if len(out) > 1 and np.linalg.norm(out[0] - out[-1]) <= tol:
        out.pop()
    return np.array(out)


def _triangle_solid_angle(a, b, c) -> float:
    """Van Oosterom-Strackee solid angle of the spherical triangle abc."""
    num = abs(np.dot(a, np.cross(b, c)))
    den = 1.0 + np.dot(a, b) + np.dot(b, c) + np.dot(c, a)
    return 2.0 * np.arctan2(num, den)


def spherical_polygon_area(center: np.ndarray, verts: np.ndarray) -> float:
    """Area (steradians) of a star-shaped spherical polygon around ``center``."""
    k = len(verts)
    return float(sum(
        _triangle_solid_angle(center, verts[i], verts[(i + 1) % k])
        for i in range(k)
    ))


@dataclass
class GridData:
    n: int
    ids: list[str]
    centers: np.ndarray          # (N, 3) unit vectors
    areas: np.ndarray            # (N,) steradians
    n_sides: np.ndarray          # (N,) 5 or 6
    polygons: list[np.ndarray]   # per tile corner vectors

    @property
    def relative_areas(self) -> np.ndarray:
        """Tile area divided by the ideal 4*pi/N."""
        return self.areas * len(self.areas) / (4 * np.pi)


def grid_data(grid, n: int) -> GridData:
    """Enumerate the grid at resolution n and measure every tile."""
    tiles = enumerate_tiles(grid, n)
    ids, centers, areas, sides, polys = [], [], [], [], []
    for h in tiles:
        c = tile_center(grid, h)
        v = tile_vertices(h)
        ids.append(h.to_str_id())
        centers.append(c)
        areas.append(spherical_polygon_area(c, v))
        sides.append(len(v))
        polys.append(v)
    return GridData(n, ids, np.array(centers), np.array(areas),
                    np.array(sides), polys)


def area_stats(data: GridData) -> dict:
    """Equal-area quality summary (relative to the mean tile area)."""
    r = data.relative_areas
    pent = data.n_sides == 5
    return {
        "n": data.n,
        "tiles": int(len(r)),
        "pentagons": int(pent.sum()),
        "total_area_over_4pi": float(data.areas.sum() / (4 * np.pi)),
        "min": float(r.min()),
        "max": float(r.max()),
        "max_over_min": float(r.max() / r.min()),
        "cv": float(r.std() / r.mean()),
        "pentagon_mean": float(r[pent].mean()) if pent.any() else float("nan"),
        "hexagon_mean": float(r[~pent].mean()),
    }


def multipole_spectrum(data: GridData, values: np.ndarray | None = None,
                       lmax: int = 20) -> np.ndarray:
    """Power per degree Q_l = sum_m |a_lm|^2 of a tile field.

    ``values`` defaults to the relative tile area.  Each tile contributes its
    own area as quadrature weight, so the l=0 term is the mean and exact
    constants have Q_l = 0 for l > 0 (up to roundoff).
    """
    from scipy.special import sph_harm_y

    f = data.relative_areas if values is None else np.asarray(values, float)
    w = data.areas
    x, y, z = data.centers.T
    theta = np.arccos(np.clip(z, -1, 1))
    phi = np.arctan2(y, x)
    Q = np.zeros(lmax + 1)
    for l in range(lmax + 1):
        for m in range(-l, l + 1):
            Y = sph_harm_y(l, m, theta, phi)
            Q[l] += abs(np.sum(w * f * np.conj(Y))) ** 2
    return Q


def hit_check(grid, n: int, samples: int = 2000, seed: int = 0) -> float:
    """Fraction of random points that land in a tile whose polygon contains
    them (consistency of ``latlon_to_hex`` with ``retrieve_polygon``)."""
    rng = np.random.default_rng(seed)
    ok = 0
    for _ in range(samples):
        p = rng.normal(size=3)
        p /= np.linalg.norm(p)
        lat = np.degrees(np.arcsin(p[2]))
        lon = np.degrees(np.arctan2(p[1], p[0]))
        h = grid.latlon_to_hex(lat, lon, n)[0]
        c = tile_center(grid, h)
        v = tile_vertices(h)
        # point is inside if it is closer to this centre than to any vertex
        # is far: compare against the polygon's circumradius
        if np.arccos(np.clip(p @ c, -1, 1)) <= np.max(
                np.arccos(np.clip(v @ c, -1, 1))) + 1e-9:
            ok += 1
    return ok / samples
