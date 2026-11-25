"""
The Hat Trick ball: 60 hats + 12 pentagons (and its larger cousins).

    from hexasphere.hattrick import HatBall
    ball = HatBall.from_frequency(3)          # GP(3,0): 92 tiles = hexasphere grid with n = 2
    ball.relax()                              # kite corners on the unit sphere
    ball.report()

Idea.  The hexagonal grid with ``n = 2`` has 92 tiles (12 pentagons + 80 hexagons).  Every hexagon is
6 kites, so there are 480 kites = 60 hats of 8 kites.  The search embeds the hat in every
orientation-preserving way into the kite surface (the 12 pentagons are holes), and keeps the
placements whose icosahedral orbits cover all kites exactly: an icosahedrally symmetric hat tiling.
Hats are chiral, so only the 60 rotations act; a hat has no symmetry, so its orbit is 60 hats.
The pentagon is the hole in the middle of 5 hats; every hat puts 240 degrees at a pentagon corner
(two hat corners of 120), the pentagon is a 120-degree pentagon with K = 60 degrees and
12 x 60 = 720 closes the Gauss-Bonnet ledger with no curvature anywhere else.

Larger frequencies work the same way: the number of hats is 7.5 (f^2 - 1), an integer for odd f, and
f = 3, 5, 7 give 60, 180, 360 hats.
"""
from __future__ import annotations

import math

import numpy as np

from hexasphere.hattrick.kites import (KITE_ANGLE, KITE_DIAG, KITE_SIDE, R3, KiteComplex, embed_hat,
                                       geodesic_sphere, rotation_group, rotation_subgroup,
                                       triangulate_points)


class HatBall:
    def __init__(self, V, F, rotations, variant=0, max_tilings=200, symmetry="I", max_nodes=2_000_000, both_hands=False):
        self.V = np.asarray(V, float)
        self.V = self.V / np.linalg.norm(self.V, axis=1)[:, None]
        self.F = [tuple(int(x) for x in f) for f in F]
        deg = np.zeros(len(self.V), int)
        for f in self.F:
            for v in f:
                deg[v] += 1
        self.pent = [int(v) for v in np.where(deg == 5)[0]]
        if len(self.pent) != 12 or set(deg) - {5, 6}:
            raise ValueError("expected an icosahedral hexagonal grid (12 pentagons, rest hexagons)")
        self.freq = int(round(math.sqrt((len(self.V) - 2) / 10)))
        self.S = KiteComplex(self.V, self.F, skip=self.pent)
        self.G = rotation_subgroup([np.asarray(g, float) for g in rotations], symmetry)
        self.symmetry = symmetry
        self.max_nodes = max_nodes
        self.both_hands = both_hands
        self.kperm = [self._kite_perm(g) for g in self.G]
        self.hand = {}                      # placement -> +1 (the hat) or -1 (its mirror image)
        for mirror in ((False, True) if both_hands else (False,)):
            for k in self.S.kites:
                p = embed_hat(self.S, k, mirror)
                if p:
                    self.hand.setdefault(frozenset(p), -1 if mirror else 1)
        self.placements = sorted(self.hand, key=lambda s: sorted(s))
        self.tilings = self._symmetric_tilings(max_tilings)
        if not self.tilings:
            raise ValueError(f"no hat tiling with symmetry {symmetry} found ({self.search}) (hats = 7.5 (f^2-1) must be an "
                             "integer, so f must be odd; larger f often needs a smaller symmetry group)")
        self.select(variant)
        self.ids = self.X = None

    # ---- construction ----------------------------------------------------------------------
    @classmethod
    def from_frequency(cls, freq=3, variant=0, **kw):
        V, F = geodesic_sphere(freq)
        return cls(V, F, rotation_group(), variant=variant, **kw)

    @classmethod
    def from_grid(cls, grid, n=2, variant=0, **kw):
        """Build on the tile centres of a ``hexasphere`` grid (any projection).  ``n`` must be even
        (f = n + 1 odd)."""
        from hexasphere import analysis, chiral
        tiles = analysis.enumerate_tiles(grid, n)
        V = np.array([analysis.tile_center(grid, h) for h in tiles])
        return cls(V, triangulate_points(V), chiral.icosahedral_rotations(grid), variant=variant, **kw)

    def _kite_perm(self, g):
        from scipy.spatial import cKDTree
        d, p = cKDTree(self.V).query(self.V @ g.T)
        if d.max() > 1e-6:
            raise ValueError("rotation does not map the vertex set to itself")
        fidx = {frozenset(f): t for t, f in enumerate(self.F)}
        out = {}
        for k in range(3 * len(self.F)):
            t, i = divmod(k, 3)
            f = self.F[t]
            t2 = fidx[frozenset(int(p[x]) for x in f)]
            out[k] = 3 * t2 + self.F[t2].index(int(p[f[i]]))
        return out

    def _symmetric_tilings(self, limit):
        """All tilings by hats whose set of hats is invariant under the 60 rotations.

        Candidates are free orbits (60 disjoint hats); a tiling is a set of candidates whose
        kite-orbits partition the kite-orbits (exact cover, solved by backtracking)."""
        ng = len(self.G)
        orbit = {}
        for k in self.S.kites:
            orbit[k] = min(kp[k] for kp in self.kperm)
        cands = {}
        for P in self.placements:
            imgs = {frozenset(kp[k] for k in P) for kp in self.kperm}
            if len(imgs) == ng and len(set().union(*imgs)) == 8 * ng:
                cov = frozenset(orbit[k] for k in P)
                if len(cov) == 8:
                    cands.setdefault(frozenset(imgs), cov)
        keys = sorted(cands, key=lambda c: sorted(map(sorted, c)))
        Y = {i: sorted(cands[c]) for i, c in enumerate(keys)}
        X = {}
        for i, cov in Y.items():
            for o in cov:
                X.setdefault(o, set()).add(i)
        for o in set(orbit.values()):
            X.setdefault(o, set())
        sols, nodes = [], [0]

        def select(r):
            cols = []
            for j in Y[r]:
                for i in X[j]:
                    for k in Y[i]:
                        if k != j:
                            X[k].remove(i)
                cols.append(X.pop(j))
            return cols

        def deselect(r, cols):
            for j in reversed(Y[r]):
                X[j] = cols.pop()
                for i in X[j]:
                    for k in Y[i]:
                        if k != j:
                            X[k].add(i)

        def solve(chosen):
            nodes[0] += 1
            if len(sols) >= limit or nodes[0] > self.max_nodes:
                return
            if not X:
                sols.append(frozenset().union(*[keys[i] for i in chosen]))
                return
            c = min(X, key=lambda o: len(X[o]))
            for r in sorted(X[c]):
                cols = select(r)
                solve(chosen + [r])
                deselect(r, cols)

        import sys
        old = sys.getrecursionlimit()
        sys.setrecursionlimit(max(old, 5000))
        try:
            solve([])
        finally:
            sys.setrecursionlimit(old)
        self.search = {"nodes": nodes[0], "exhausted": nodes[0] <= self.max_nodes and len(sols) < limit,
                       "candidates": len(keys), "kite_orbits": len(set(orbit.values()))}
        return sorted(set(sols), key=lambda T: sorted(map(sorted, T)))

    def select(self, variant):
        if not 0 <= variant < len(self.tilings):
            raise ValueError(f"variant must be in 0..{len(self.tilings) - 1}")
        self.variant = variant
        self.hats = sorted(map(sorted, self.tilings[variant]))
        self.hat_of = {k: i for i, h in enumerate(self.hats) for k in h}
        self._outlines = [self._outline(h) for h in self.hats]

    @property
    def n_hats(self):
        return len(self.hats)

    # ---- outlines --------------------------------------------------------------------------
    def _outline(self, kites):
        edges = {}
        for k in kites:
            c = self.S.corners(k)
            for s in range(4):
                a, b = c[s], c[(s + 1) % 4]
                if (b, a) in edges:
                    del edges[(b, a)]
                else:
                    edges[(a, b)] = True
        nxt = {a: b for (a, b) in edges}
        start = next(iter(nxt))
        loop, p = [start], nxt[start]
        while p != start:
            loop.append(p)
            p = nxt[p]
        assert len(loop) == len(nxt), "outline is not a single simple loop"
        ang = {}
        for k in kites:
            for cid in self.S.corners(k):
                ang[cid] = ang.get(cid, 0) + KITE_ANGLE[cid[0]]
        true = [i for i, cid in enumerate(loop) if ang[cid] != 180]
        return loop, true, [ang[c] for c in loop]

    def hat_outline(self, i):
        """``(loop, corner indices, interior angle at every loop point)`` of hat i."""
        return self._outlines[i]

    def pentagon_loop(self, p):
        """The pentagon around vertex p: centroid, midpoint, centroid, ... (10 corner ids)."""
        t = next(t for t, f in enumerate(self.F) if p in f)
        order = [t]
        while len(order) < 5:
            f = self.F[order[-1]]
            order.append(self.S.e2t[(p, f[(f.index(p) + 2) % 3])])
        loop = []
        for t in order:
            f = self.F[t]
            loop += [("c", t), ("m", frozenset((p, f[(f.index(p) + 2) % 3])))]
        return loop

    def pentagon_hats(self, p):
        """Hats that touch the pentagon around vertex p."""
        out = set()
        for k in self.S.kites:
            t, i = divmod(k, 3)
            if any(self.F[t][(i + s) % 3] == p for s in (1, 2)):
                out.add(self.hat_of[k])
        return sorted(out)

    def patch(self, p):
        """The patch of pentagon p: the sorted kites of its hats."""
        return sorted(k for h in self.pentagon_hats(p) for k in self.hats[h])

    # ---- geometry --------------------------------------------------------------------------
    def corner_ids(self):
        ids = {}
        for k in self.S.kites:
            for cid in self.S.corners(k):
                ids.setdefault(cid, len(ids))
        return ids

    def initial_positions(self):
        ids = self.corner_ids()
        X = np.zeros((len(ids), 3))
        for cid, i in ids.items():
            X[i] = self.S.corner_position(cid)
        return ids, X

    def _constraints(self, ids):
        pairs, target = [], []
        for k in self.S.kites:
            c = [ids[x] for x in self.S.corners(k)]
            for s in range(4):
                pairs.append((c[s], c[(s + 1) % 4])); target.append(KITE_SIDE[s])
            for (i, j), d in KITE_DIAG.items():
                pairs.append((c[i], c[j])); target.append(d)
        seen = set()
        for t, f in enumerate(self.F):
            for i in range(3):
                a, b = f[i], f[(i + 1) % 3]
                e = frozenset((a, b))
                if e in seen:
                    continue
                seen.add(e)
                t2 = self.S.e2t[(b, a)]
                pairs.append((ids[("c", t)], ids[("c", t2)])); target.append(2.0)   # hexagon edge = 2 short sides
                if a not in self.S.skip and b not in self.S.skip:
                    pairs.append((ids[("v", a)], ids[("v", b)])); target.append(2 * R3)
        return np.array(pairs), np.array(target)

    def relax(self, max_nfev=200, positions=None):
        """Move every kite corner on the unit sphere so each kite is as close as possible to the
        flat kite times a common scale ``a`` (sides 1, sqrt3; diagonals 2, sqrt3) and the straight
        hexagon / triangulation edges stay straight.  Sets ``ids, X, a, info`` and returns ``info``.
        ``max_nfev=0`` only measures the starting geometry."""
        from scipy.optimize import least_squares
        from scipy.sparse import lil_matrix
        ids, X0 = self.initial_positions() if positions is None else positions
        n = len(X0)
        P, T = self._constraints(ids)
        m = len(P)

        def unit(Z):
            Z = Z.reshape(n, 3)
            return Z / np.linalg.norm(Z, axis=1)[:, None]

        def resid(z):
            return np.linalg.norm(unit(z[1:])[P[:, 0]] - unit(z[1:])[P[:, 1]], axis=1) - z[0] * T

        d0 = np.linalg.norm(X0[P[:, 0]] - X0[P[:, 1]], axis=1)
        a0 = float(d0 @ T / (T @ T))                     # best common scale for the start
        z0 = np.concatenate([[a0], X0.ravel()])
        if max_nfev == 0:
            z = z0
        else:
            A = lil_matrix((m, 1 + 3 * n), dtype=int)
            A[:, 0] = 1
            for r, (i, j) in enumerate(P):
                for q in (i, j):
                    A[r, 1 + 3 * q:4 + 3 * q] = 1
            z = least_squares(resid, z0, jac_sparsity=A, method="trf", x_scale="jac", max_nfev=max_nfev).x
        a, X = float(z[0]), unit(z[1:])
        rel = resid(z) / (a * T)
        self.ids, self.X, self.a = ids, X, a
        self._pairs, self._target = P, T
        self.info = {"scale": a, "rms_strain": float(np.sqrt(np.mean(rel ** 2))),
                     "max_strain": float(np.max(np.abs(rel))), "n_constraints": int(m)}
        return self.info

    def kite_strain(self):
        """Per-kite rms relative deviation of its six distances (needs ``relax``)."""
        P, T = self._pairs, self._target
        d = np.linalg.norm(self.X[P[:, 0]] - self.X[P[:, 1]], axis=1)
        rel = (d - self.a * T) / (self.a * T)
        out = {}
        for r, (i, j) in enumerate(P):
            pass
        # the first 6 constraints of every kite are its sides and diagonals, in kite order
        per = {}
        for q, k in enumerate(self.S.kites):
            per[k] = float(np.sqrt(np.mean(rel[6 * q:6 * q + 6] ** 2)))
        return per

    def polygon(self, loop):
        return self.X[[self.ids[c] for c in loop]]

    # ---- reports ---------------------------------------------------------------------------
    @staticmethod
    def spherical_area(pts):
        c = pts.mean(0)
        c = c / np.linalg.norm(c)
        tot = 0.0
        for i in range(len(pts)):
            a, b = pts[i], pts[(i + 1) % len(pts)]
            tot += 2 * math.atan2(abs(np.dot(c, np.cross(a, b))), 1 + c @ a + a @ b + b @ c)
        return tot

    def panel_areas(self):
        hats = np.array([self.spherical_area(self.polygon(self.hat_outline(i)[0])) for i in range(self.n_hats)])
        pents = np.array([self.spherical_area(self.polygon(self.pentagon_loop(p))) for p in self.pent])
        return hats, pents

    def report(self):
        if self.X is None:
            self.relax()
        hats, pents = self.panel_areas()
        return {
            "frequency": self.freq, "tiles": len(self.V), "kites": len(self.S.kites),
            "hats": self.n_hats, "pentagons": len(self.pent), "panels": self.n_hats + len(self.pent),
            "hat_placements": len(self.placements), "symmetric_tilings": len(self.tilings),
            "variant": self.variant,
            "hats_per_pentagon": sorted({len(self.pentagon_hats(p)) for p in self.pent}),
            "hat_corners": sorted({len(self.hat_outline(i)[1]) for i in range(self.n_hats)}),
            "hat_to_pentagon_area": float(hats.mean() / pents.mean()),
            "hat_area_spread": float((hats.max() - hats.min()) / hats.mean()),
            "total_area_over_4pi": float((hats.sum() + pents.sum()) / (4 * math.pi)),
            **self.info,
        }

    def panels(self):
        """JSON-ready panels: outline points on the unit sphere and the indices of the true corners."""
        out = []
        for i in range(self.n_hats):
            loop, true, _ = self.hat_outline(i)
            out.append({"kind": "hat", "id": i, "corners": true, "outline": self.polygon(loop).round(6).tolist()})
        for p in self.pent:
            out.append({"kind": "pentagon", "id": p, "corners": [0, 2, 4, 6, 8],
                        "outline": self.polygon(self.pentagon_loop(p)).round(6).tolist()})
        return out
