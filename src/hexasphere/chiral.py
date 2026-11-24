"""
Chiral hexagonal tilings of the sphere from an equivariant, area-preserving flow.

Take the vector field  v(p) = p x grad f(p)  with f(p) = sum_a (p.a)^6 over the
six five-fold axes a of the icosahedron.  Because f is invariant under the
icosahedral rotation group I, the flow Phi_t of v

* commutes with every rotation in I           (the image tiling is still I-symmetric),
* preserves area exactly (v is divergence free, f is its Hamiltonian),
* fixes every point with a non-trivial stabiliser: the 12 pentagon centres, the
  30 edge midpoints and the 20 face centres,
* is odd under reflections: for t != 0 the image of a mirror-symmetric tiling
  has no mirror symmetry left (it is chiral), and Phi_{-t} gives the mirror image.

So pushing a hexasphere tiling through Phi_t gives tiles that have exactly the
areas of the original ones, share their seams exactly, and have swirled
boundaries.  This is the same construction as the chiral "Trionda" family in
the ballgen repository, applied to the hexagonal grid.

The flow time is normalised so that ``tau`` is the largest displacement (in
radians of arc) any point can travel.
"""
from __future__ import annotations

import numpy as np

from hexasphere.analysis import GridData, spherical_polygon_area


def five_fold_axes(grid) -> np.ndarray:
    """The six five-fold axes (one unit vector per +/- pair of vertices)."""
    V = np.array([grid.abc[f][i] for f in range(20) for i in range(3)])
    axes: list[np.ndarray] = []
    for v in V:
        if not any(abs(abs(v @ a) - 1) < 1e-9 for a in axes):
            axes.append(v / np.linalg.norm(v))
    assert len(axes) == 6
    return np.array(axes)


class IcosahedralFlow:
    """Flow of p x grad f for f = sum (p.a)^6, normalised to unit top speed."""

    def __init__(self, axes: np.ndarray):
        self.axes = np.asarray(axes, float)
        s = _fibonacci(20000)
        speed = np.linalg.norm(self._raw(s), axis=1).max()
        self.scale = 1.0 / speed

    def f(self, p):
        return np.sum((np.atleast_2d(p) @ self.axes.T) ** 6, axis=1)

    def _raw(self, p):
        g = 6 * ((p @ self.axes.T) ** 5) @ self.axes
        return np.cross(p, g)

    def field(self, p):
        return self.scale * self._raw(p)

    def advance(self, p, tau: float, steps: int | None = None) -> np.ndarray:
        """Phi_tau(p) by RK4 with renormalisation (``tau`` may be negative)."""
        p = np.atleast_2d(np.asarray(p, float)).copy()
        if tau == 0:
            return p
        steps = steps or max(8, int(np.ceil(abs(tau) / 0.01)))
        h = tau / steps
        for _ in range(steps):
            k1 = self.field(p)
            k2 = self.field(p + 0.5 * h * k1)
            k3 = self.field(p + 0.5 * h * k2)
            k4 = self.field(p + h * k3)
            p = p + (h / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
            p /= np.linalg.norm(p, axis=1, keepdims=True)
        return p


def _fibonacci(n):
    i = np.arange(n) + 0.5
    z = 1 - 2 * i / n
    r = np.sqrt(1 - z * z)
    a = np.pi * (3 - np.sqrt(5)) * i
    return np.c_[r * np.cos(a), r * np.sin(a), z]


def _slerp_ring(verts: np.ndarray, samples: int) -> np.ndarray:
    """Closed ring through ``verts`` with ``samples`` great-circle points per edge."""
    out = []
    k = len(verts)
    t = np.linspace(0, 1, samples, endpoint=False)[:, None]
    for i in range(k):
        a, b = verts[i], verts[(i + 1) % k]
        ang = np.arccos(np.clip(a @ b, -1, 1))
        if ang < 1e-12:
            out.append(np.repeat(a[None], samples, axis=0))
            continue
        out.append((np.sin((1 - t) * ang) * a + np.sin(t * ang) * b) / np.sin(ang))
    return np.vstack(out)


def chiral_grid(grid, data: GridData, tau: float, edge_samples: int = 16,
                flow: IcosahedralFlow | None = None) -> GridData:
    """Push a measured grid through the flow for time ``tau``.

    Tile identities and ordering are preserved.  Polygons become densified rings
    (``edge_samples`` points per original edge) so the swirled edges are resolved.
    """
    flow = flow or IcosahedralFlow(five_fold_axes(grid))
    centers = flow.advance(data.centers, tau)
    polys, areas = [], []
    for v in data.polygons:
        ring = flow.advance(_slerp_ring(v, edge_samples), tau)
        polys.append(ring)
    for c, ring in zip(centers, polys):
        areas.append(spherical_polygon_area(c, ring))
    return GridData(data.n, list(data.ids), centers, np.array(areas),
                    data.n_sides.copy(), polys)


def perimeter(ring: np.ndarray) -> float:
    d = np.arccos(np.clip(np.sum(ring * np.roll(ring, -1, axis=0), axis=1), -1, 1))
    return float(d.sum())


def perimeters(data: GridData) -> np.ndarray:
    return np.array([perimeter(r) for r in data.polygons])


def icosahedral_rotations(grid) -> np.ndarray:
    """The 60 rotation matrices of the icosahedron whose vertices are ``grid.abc``."""
    F = [np.array(grid.abc[f]) for f in range(20)]
    inv = np.linalg.inv(F[0].T)
    R = []
    for f in range(20):
        for k in range(3):
            M = np.roll(F[f], k, axis=0).T @ inv
            if np.allclose(M @ M.T, np.eye(3), atol=1e-8) and np.linalg.det(M) > 0:
                R.append(M)
    assert len(R) == 60
    return np.array(R)


def symmetry_counts(grid, centers: np.ndarray, tol: float = 1e-6) -> dict:
    """How many of the 120 symmetries of the icosahedron map the tile-centre set
    onto itself: 60 rotations (proper) and 60 rotation-inversions (improper).

    The achiral grid gives 60 + 60.  A chiral tiling keeps the 60 rotations and
    loses every improper one, which is the exact meaning of "chiral" here.
    """
    from scipy.spatial import cKDTree
    tree = cKDTree(centers)
    proper = improper = 0
    for M in icosahedral_rotations(grid):
        if tree.query(centers @ M.T)[0].max() < tol:
            proper += 1
        if tree.query(-(centers @ M.T))[0].max() < tol:
            improper += 1
    return {"proper": proper, "improper": improper, "chiral": improper == 0}
