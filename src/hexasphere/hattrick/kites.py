"""
Kites, the hat, and the kite surface of a geodesic sphere.

The *hat* (David Smith's aperiodic monotile) is a polykite: 8 kites of the deltoidal
trihexagonal tiling.  Cut a hexagonal grid into kites and the hat becomes a combinatorial
object that can be searched for on a sphere.

A triangulation (the dual of the hexagonal grid: one vertex per tile) is cut like this.  A
triangle ``t = (v0, v1, v2)``, counter-clockwise seen from outside, is split into three kites
around its centroid; kite ``3t + i`` belongs to vertex ``v_i``.  Its corners, counter-clockwise,

    v_i  ->  m(v_i, v_{i+1})  ->  centroid(t)  ->  m(v_{i+2}, v_i)

(``m`` is an edge midpoint) and its four sides are numbered 0..3 starting at ``v_i``.  With a
triangulation edge of length ``2 sqrt(3)`` the hexagon side is 2 and the kite sides are

    side 0 : long  (sqrt 3)      side 1 : short (1)
    side 2 : short (1)           side 3 : long  (sqrt 3)

which are exactly the hat's two edge lengths.  A tile that is a pentagon (a degree-5 vertex of the
triangulation) is left out: its five kites are a *hole*, the pentagon panel of the ball.
"""
from __future__ import annotations

import math

import numpy as np

R3 = math.sqrt(3.0)

KITE_SIDE = (R3, 1.0, 1.0, R3)
KITE_DIAG = {(0, 2): 2.0, (1, 3): R3}      # vertex-to-centroid, midpoint-to-midpoint
KITE_ANGLE = {"v": 60, "m": 90, "c": 120}  # interior angle (degrees) at each corner type
HAT_ANGLES = [90] * 4 + [120] * 5 + [240] * 2 + [270] * 2   # the hat's 13 interior angles


class KiteComplex:
    """The kites of a triangulated sphere (or plane patch), with their side adjacency."""

    def __init__(self, V, F, skip=()):
        self.V = np.asarray(V, float)
        self.F = [tuple(int(x) for x in f) for f in F]
        self.skip = set(skip)
        e2t = {}
        for t, f in enumerate(self.F):
            for i in range(3):
                e2t[(f[i], f[(i + 1) % 3])] = t
        self.e2t = e2t
        self.nb = [[None] * 4 for _ in range(3 * len(self.F))]
        for t, f in enumerate(self.F):
            for i in range(3):
                v = f[i]
                if v in self.skip:
                    continue
                nb = self.nb[3 * t + i]
                t1, t2 = e2t.get((f[(i + 1) % 3], v)), e2t.get((v, f[(i + 2) % 3]))
                if t1 is not None:
                    nb[0] = 3 * t1 + self.F[t1].index(v)
                if f[(i + 1) % 3] not in self.skip:
                    nb[1] = 3 * t + (i + 1) % 3
                if f[(i + 2) % 3] not in self.skip:
                    nb[2] = 3 * t + (i + 2) % 3
                if t2 is not None:
                    nb[3] = 3 * t2 + self.F[t2].index(v)
        self.kites = [k for k in range(3 * len(self.F)) if self.F[k // 3][k % 3] not in self.skip]

    def corners(self, k):
        """Combinatorial ids of the four corners of kite k: ('v', vertex), ('m', {a, b}), ('c', triangle)."""
        t, i = divmod(k, 3)
        f = self.F[t]
        v, b, c = f[i], f[(i + 1) % 3], f[(i + 2) % 3]
        return [("v", v), ("m", frozenset((v, b))), ("c", t), ("m", frozenset((c, v)))]

    def corner_position(self, cid):
        """Starting position of a corner on the unit sphere."""
        kind, x = cid
        if kind == "v":
            return self.V[x] / np.linalg.norm(self.V[x])
        if kind == "m":
            a, b = tuple(x)
            q = self.V[a] / np.linalg.norm(self.V[a]) + self.V[b] / np.linalg.norm(self.V[b])
        else:
            q = sum(self.V[i] / np.linalg.norm(self.V[i]) for i in self.F[x])
        return q / np.linalg.norm(q)


def hat_adjacency(mirror=False):
    """The hat as 8 kites: ``(a, side, b)`` for every internal adjacency (kite 0 is the anchor).

    Read off the hat outline; ``flat_hat`` lays it out in the plane and checks that the
    adjacency closes up and reproduces the hat's 13 corners.
    """
    adj = [(0, 3, 1), (1, 0, 0), (1, 3, 2), (2, 0, 1), (2, 1, 3), (2, 2, 4), (2, 3, 5), (3, 0, 6),
           (3, 1, 4), (3, 2, 2), (4, 0, 7), (4, 1, 2), (4, 2, 3), (5, 0, 2), (5, 1, 7), (6, 3, 3),
           (7, 2, 5), (7, 3, 4)]
    if mirror:               # a kite is its own mirror image with sides 0<->3, 1<->2
        adj = [(a, 3 - s, b) for a, s, b in adj]
    return adj


def embed_hat(S: KiteComplex, anchor: int, mirror: bool = False):
    """Embed the hat with its kite 0 on kite ``anchor`` of ``S``.

    Returns the 8 surface kites in hat order, or ``None``.  Sides are labelled, so only
    orientation-preserving embeddings exist (a hat is chiral; its mirror image is not used).
    """
    adj = hat_adjacency(mirror)
    out = {0: anchor}
    todo = [0]
    while todo:
        h = todo.pop()
        for a, s, b in adj:
            if a != h:
                continue
            nxt = S.nb[out[h]][s]
            if nxt is None:
                return None
            if b in out:
                if out[b] != nxt:
                    return None
            else:
                out[b] = nxt
                todo.append(b)
    if len(out) != 8 or len(set(out.values())) != 8:
        return None
    if any(S.nb[out[a]][s] != out[b] for a, s, b in adj):
        return None
    return tuple(out[i] for i in range(8))


def flat_hat():
    """The hat laid out in the plane: ``(8 kite polygons, outline loop)``.  Raises if the
    kite adjacency is not planar-consistent."""
    c = np.array([2.0, 0.0])
    m1 = R3 * np.array([math.cos(-math.pi / 6), math.sin(-math.pi / 6)])
    m2 = R3 * np.array([math.cos(math.pi / 6), math.sin(math.pi / 6)])
    base = np.array([[0, 0], m1, c, m2])
    adj = hat_adjacency()
    back = {(a, b): next(s for aa, s, bb in adj if aa == b and bb == a) for a, s, b in adj}
    pose = {0: (np.eye(2), np.zeros(2))}
    todo = [0]
    while todo:
        h = todo.pop()
        R, t = pose[h]
        for a, s, b in adj:
            if a != h or b in pose:
                continue
            Q = base @ R.T + t
            e0, e1 = Q[s], Q[(s + 1) % 4]
            s2 = back[(a, b)]
            vb, va = base[(s2 + 1) % 4] - base[s2], e0 - e1
            ang = math.atan2(va[1], va[0]) - math.atan2(vb[1], vb[0])
            Rb = np.array([[math.cos(ang), -math.sin(ang)], [math.sin(ang), math.cos(ang)]])
            pose[b] = (Rb, e1 - Rb @ base[s2])
            todo.append(b)
    kites = [base @ pose[i][0].T + pose[i][1] for i in range(8)]
    for a, s, b in adj:
        d = (np.linalg.norm(kites[a][s] - kites[b][(back[(a, b)] + 1) % 4])
             + np.linalg.norm(kites[a][(s + 1) % 4] - kites[b][back[(a, b)]]))
        if d > 1e-9:
            raise ValueError("hat adjacency is not planar-consistent")
    key = lambda p: (round(float(p[0]), 3), round(float(p[1]), 3))
    edges = {}
    for Q in kites:
        for s in range(4):
            a, b = key(Q[s]), key(Q[(s + 1) % 4])
            if (b, a) in edges:
                del edges[(b, a)]
            else:
                edges[(a, b)] = True
    nxt = {a: b for a, b in edges}
    start = next(iter(nxt))
    outline = [start]
    while nxt[outline[-1]] != start:
        outline.append(nxt[outline[-1]])
    return kites, np.array(outline)


def corner_angles(outline):
    """Interior angles (degrees) at the true corners of a counter-clockwise planar loop."""
    n = len(outline)
    out = []
    for i in range(n):
        a, b = outline[i - 1] - outline[i], outline[(i + 1) % n] - outline[i]
        t = math.degrees(math.atan2(b[0] * a[1] - b[1] * a[0], a @ b)) % 360
        if abs(t - 180) > 0.5:
            out.append(int(round(t)))
    return out


# ---------------------------------------------------------------------------------------------
# geodesic spheres (the hexagonal grid's dual, for any frequency)
# ---------------------------------------------------------------------------------------------

def icosahedron():
    p = (1 + 5 ** 0.5) / 2
    V = np.array([(-1, p, 0), (1, p, 0), (-1, -p, 0), (1, -p, 0), (0, -1, p), (0, 1, p),
                  (0, -1, -p), (0, 1, -p), (p, 0, -1), (p, 0, 1), (-p, 0, -1), (-p, 0, 1)], float)
    V /= np.linalg.norm(V, axis=1)[:, None]
    F = [(0, 11, 5), (0, 5, 1), (0, 1, 7), (0, 7, 10), (0, 10, 11), (1, 5, 9), (5, 11, 4), (11, 10, 2),
         (10, 7, 6), (7, 1, 8), (3, 9, 4), (3, 4, 2), (3, 2, 6), (3, 6, 8), (3, 8, 9), (4, 9, 5),
         (2, 4, 11), (6, 2, 10), (8, 6, 7), (9, 8, 1)]
    return V, [f if np.dot(np.cross(*(V[[f[1], f[2]]] - V[f[0]])), V[list(f)].sum(0)) > 0 else (f[0], f[2], f[1])
               for f in F]


def geodesic_sphere(freq: int = 3):
    """Icosahedron with every face cut into freq^2 triangles, projected to the sphere.
    ``10 freq^2 + 2`` vertices = the tiles of the hexagonal grid with n = freq - 1."""
    V0, F0 = icosahedron()
    verts, index, F = [], {}, []

    def vid(p):
        k = tuple(np.round(p, 6))
        if k not in index:
            index[k] = len(verts)
            verts.append(p)
        return index[k]

    for f in F0:
        A, B, C = V0[list(f)]
        P = lambda i, j: (A * (freq - i - j) + B * i + C * j) / freq
        for i in range(freq):
            for j in range(freq - i):
                F.append((vid(P(i, j)), vid(P(i + 1, j)), vid(P(i, j + 1))))
                if i + j < freq - 1:
                    F.append((vid(P(i + 1, j)), vid(P(i + 1, j + 1)), vid(P(i, j + 1))))
    V = np.array(verts)
    return V / np.linalg.norm(V, axis=1)[:, None], F


def rotation_group():
    """The 60 proper rotations of the (built-in) icosahedron."""
    V0, F0 = icosahedron()

    def turn(ax, ang):
        ax = ax / np.linalg.norm(ax)
        K = np.array([[0, -ax[2], ax[1]], [ax[2], 0, -ax[0]], [-ax[1], ax[0], 0]])
        return np.eye(3) + math.sin(ang) * K + (1 - math.cos(ang)) * K @ K

    gens = [turn(V0[0], 2 * math.pi / 5), turn(V0[list(F0[0])].sum(0), 2 * math.pi / 3)]
    G = [np.eye(3)]
    grew = True
    while grew:
        grew = False
        for g in list(G):
            for h in gens:
                m = h @ g
                if not any(np.allclose(m, x, atol=1e-6) for x in G):
                    G.append(m)
                    grew = True
    assert len(G) == 60
    return G


def triangulate_points(points):
    """Counter-clockwise (outward) triangulation of points on the sphere: their convex hull.
    For hexagonal-grid tile centres this is the dual triangulation (degrees 5 and 6)."""
    from scipy.spatial import ConvexHull
    P = np.asarray(points, float)
    hull = ConvexHull(P)
    F = []
    for a, b, c in hull.simplices:
        F.append((int(a), int(b), int(c)) if np.dot(np.cross(P[b] - P[a], P[c] - P[a]), P[a] + P[b] + P[c]) > 0
                 else (int(a), int(c), int(b)))
    return F


def _closure(gens, cap=None):
    """Group generated by rotation matrices (None if it grows beyond ``cap`` elements)."""
    key = lambda m: tuple(np.round(m, 5).ravel() + 0.0)
    G = {key(np.eye(3)): np.eye(3)}
    todo = [np.eye(3)]
    while todo:
        g = todo.pop()
        for h in gens:
            m = h @ g
            k = key(m)
            if k not in G:
                G[k] = m
                todo.append(m)
                if cap is not None and len(G) > cap:
                    return None
    return list(G.values())


def _element_order(g):
    m, n = g.copy(), 1
    while not np.allclose(m, np.eye(3), atol=1e-6):
        m = g @ m
        n += 1
    return n


# element-order histograms of the rotation subgroups of the icosahedral group
SUBGROUPS = {
    "I": {1: 1, 2: 15, 3: 20, 5: 24},
    "T": {1: 1, 2: 3, 3: 8},
    "D5": {1: 1, 2: 5, 5: 4},
    "D3": {1: 1, 2: 3, 3: 2},
    "D2": {1: 1, 2: 3},
    "C5": {1: 1, 5: 4},
    "C3": {1: 1, 3: 2},
    "C2": {1: 1, 2: 1},
    "C1": {1: 1},
}


def rotation_subgroup(G, name):
    """A subgroup of the icosahedral rotation group ``G`` of the given type (all subgroups of one
    type are conjugate, so the choice does not matter up to a rotation of the ball)."""
    if name not in SUBGROUPS:
        raise ValueError(f"unknown symmetry {name!r}; choose from {sorted(SUBGROUPS)}")
    want = SUBGROUPS[name]
    size = sum(want.values())
    if size == len(G):
        return list(G)
    if size == 1:
        return [np.eye(3)]
    for g in G:
        for h in G:
            H = _closure([g, h], cap=size)
            if H is not None and len(H) == size:
                hist = {}
                for x in H:
                    o = _element_order(x)
                    hist[o] = hist.get(o, 0) + 1
                if hist == want:
                    return H
    raise RuntimeError(f"no subgroup of type {name}")
