"""Matplotlib pictures of a Hat Trick ball (matplotlib is imported lazily)."""
from __future__ import annotations

import numpy as np

PALETTE = ["#e07a5f", "#3d85c6", "#81b29a", "#f2cc8f"]
PENTAGON_COLOUR = "#2b2b2b"


def _plt():
    import matplotlib
    matplotlib.use("Agg", force=False)
    import matplotlib.pyplot as plt
    return plt


def _basis(d):
    d = np.asarray(d, float) / np.linalg.norm(d)
    up = np.array([0, 0, 1.0]) if abs(d[2]) < 0.9 else np.array([0, 1.0, 0])
    x = np.cross(up, d)
    x /= np.linalg.norm(x)
    return x, np.cross(d, x), d


def symmetry_axes(ball):
    """Directions of a 5-fold, a 3-fold and a 2-fold axis of the ball (from its pentagons)."""
    P = ball.V[ball.pent]
    near = lambda i: [j for j in range(12) if j != i and P[i] @ P[j] > 0.2]
    a = 0
    b = near(a)[0]
    c = next(j for j in near(a) if j in near(b))
    return {"5-fold (pentagon)": P[a], "3-fold": P[a] + P[b] + P[c], "2-fold": P[a] + P[b]}


def patch_colours(ball, colours=4):
    """Colour the 12 patches so that touching patches differ (4-colouring of the icosahedron)."""
    P = ball.V[ball.pent]
    adj = {i: {j for j in range(12) if j != i and P[i] @ P[j] > 0.2} for i in range(12)}
    order = sorted(adj, key=lambda i: -len(adj[i]))
    col = {}

    def solve(i):
        if i == len(order):
            return True
        v = order[i]
        for c in range(colours):
            if all(col.get(w) != c for w in adj[v]):
                col[v] = c
                if solve(i + 1):
                    return True
                del col[v]
        return False

    assert solve(0)
    return {ball.pent[i]: col[i] for i in range(12)}


def _kite_grid(P, n=3):
    p0, p1, p2, p3 = P
    out = []
    for i in range(n):
        for j in range(n):
            q = []
            for u, w in ((i / n, j / n), ((i + 1) / n, j / n), ((i + 1) / n, (j + 1) / n), (i / n, (j + 1) / n)):
                v = (1 - u) * (1 - w) * p0 + u * (1 - w) * p1 + u * w * p2 + (1 - u) * w * p3
                out_ = v / np.linalg.norm(v)
                q.append(out_)
            out.append(np.array(q))
    return out


def _arc(a, b, n=6):
    t = np.linspace(0, 1, n)[:, None]
    p = (1 - t) * a + t * b
    return p / np.linalg.norm(p, axis=1)[:, None]


def draw_ball(ax, ball, view, colour="patch", zoom=1.0, highlight=None, lines=True, light=(0.35, 0.55, 1.0),
              cmap="viridis", vmax=None):
    """Orthographic view of the ball from direction ``view``.

    colour: 'patch' (hats coloured by the pentagon they touch), 'hat' (one colour per hat),
    'strain' (kites coloured by their rms deviation from the flat kite).
    highlight: pentagon vertex id; its patch is drawn in colour, everything else is faded."""
    from matplotlib.collections import LineCollection, PolyCollection
    import matplotlib.colors as mc
    from matplotlib import cm
    ex, ey, ez = _basis(view)
    L = light[0] * ex + light[1] * ey + light[2] * ez
    L = L / np.linalg.norm(L)
    pc = patch_colours(ball)
    patch_of_hat = {h: pc[p] for p in ball.pent for h in ball.pentagon_hats(p)}
    hl_hats = set(ball.pentagon_hats(highlight)) if highlight is not None else None
    strain = ball.kite_strain() if colour == "strain" else None
    if strain is not None:
        norm = mc.Normalize(0, vmax or max(strain.values()))
        cm_ = cm.get_cmap(cmap) if hasattr(cm, "get_cmap") else cm.viridis
    polys, cols = [], []

    def add(q, base):
        nrm = q.mean(0)
        nrm /= np.linalg.norm(nrm)
        if nrm @ ez <= 0:
            return
        sh = 0.45 + 0.55 * max(0.0, nrm @ L)
        polys.append(np.c_[q @ ex, q @ ey])
        cols.append(np.clip(np.array(mc.to_rgb(base)) * sh, 0, 1))

    for k in ball.S.kites:
        P = ball.polygon(ball.S.corners(k))
        h = ball.hat_of[k]
        if colour == "strain":
            base = cm_(norm(strain[k]))[:3]
        elif colour == "hat":
            base = PALETTE[h % 4]
        else:
            base = PALETTE[patch_of_hat[h]]
        if hl_hats is not None and h not in hl_hats:
            base = tuple(0.5 * np.array(mc.to_rgb(base)) + 0.5 * 0.88)
        for q in _kite_grid(P, 3):
            add(q, base)
    for p in ball.pent:
        Q = ball.polygon(ball.pentagon_loop(p))
        c = Q.mean(0)
        c /= np.linalg.norm(c)
        base = PENTAGON_COLOUR if (highlight is None or p == highlight) else "#9a9a9a"
        for i in range(len(Q)):
            tri = np.array([c, Q[i], Q[(i + 1) % len(Q)]])
            for q in _kite_grid([tri[0], tri[1], tri[2], tri[0]], 2):
                add(q, base)
    ax.add_collection(PolyCollection(polys, facecolors=cols, edgecolors=cols, linewidths=0.3))
    if lines:
        segs = []
        loops = [ball.polygon(ball.hat_outline(i)[0]) for i in range(ball.n_hats)]
        loops += [ball.polygon(ball.pentagon_loop(p)) for p in ball.pent]
        for Q in loops:
            for j in range(len(Q)):
                arc = _arc(Q[j], Q[(j + 1) % len(Q)])
                if arc.mean(0) @ ez > 0.02:
                    segs.append(np.c_[arc @ ex, arc @ ey])
        ax.add_collection(LineCollection(segs, colors="#101010", linewidths=1.1 if ball.n_hats <= 100 else 0.6))
    th = np.linspace(0, 2 * np.pi, 200)
    ax.plot(np.cos(th), np.sin(th), color="#101010", lw=1.0)
    r = 1.03 / zoom
    ax.set_xlim(-r, r)
    ax.set_ylim(-r, r)
    ax.set_aspect("equal")
    ax.axis("off")


def views(ball, path=None, title=None):
    """The ball seen along a 5-fold, a 3-fold and a 2-fold axis."""
    plt = _plt()
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.3))
    for ax, (name, d) in zip(axes, symmetry_axes(ball).items()):
        draw_ball(ax, ball, d)
        ax.set_title(f"along a {name} axis", fontsize=11)
    fig.suptitle(title or f"{ball.n_hats} hats + 12 pentagons (variant {ball.variant}); "
                          f"rms kite strain {100 * ball.info['rms_strain']:.1f}%", fontsize=13)
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=130)
    return fig


def patch(ball, path=None):
    """One patch (a pentagon and its 5 hats) highlighted, seen along its 5-fold axis."""
    plt = _plt()
    fig, ax = plt.subplots(figsize=(6, 6))
    p = ball.pent[0]
    draw_ball(ax, ball, ball.V[p], zoom=1.7, highlight=p)
    ax.set_title("a patch: a pentagon with 5 hats around it; 12 patches make the ball", fontsize=10)
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=130)
    return fig


def strain_map(ball, path=None):
    """Kites coloured by how far they are from the flat kite (the price of closing the sphere)."""
    plt = _plt()
    from matplotlib import cm, colors
    s = ball.kite_strain()
    vmax = max(s.values())
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.5))
    for ax, name in zip(axes, ("5-fold (pentagon)", "3-fold")):
        draw_ball(ax, ball, symmetry_axes(ball)[name], colour="strain", vmax=vmax)
        ax.set_title(f"along a {name} axis", fontsize=10)
    sm = cm.ScalarMappable(norm=colors.Normalize(0, 100 * vmax), cmap="viridis")
    fig.colorbar(sm, ax=axes, shrink=0.7, label="kite strain (rms of six distances, %)")
    fig.suptitle("Where the hats are strained", fontsize=12)
    if path:
        fig.savefig(path, dpi=130)
    return fig


def hat(path=None):
    """The hat as 8 kites in the plane."""
    from hexasphere.hattrick.kites import flat_hat
    plt = _plt()
    kites, outline = flat_hat()
    fig, ax = plt.subplots(figsize=(5.5, 4.2))
    for Q in kites:
        ax.fill(Q[:, 0], Q[:, 1], color=PALETTE[1], ec="#9fc3e6", lw=0.8, zorder=1)
    ax.fill(outline[:, 0], outline[:, 1], fill=False, ec="#101010", lw=2.0, zorder=3, joinstyle="round")
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("the hat: 8 kites (edges 1 and $\\sqrt{3}$; one straight edge of length 2)")
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=130)
    return fig
