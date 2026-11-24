"""
Matplotlib views of a hexasphere grid (matplotlib is imported lazily, so the
core package stays numpy-only).

    from hexasphere import hexgrid, projection, analysis, plot
    data = analysis.grid_data(grid, n)
    fig = plot.grid_3d(data)            # tiles on a sphere, shaded by area
    fig = plot.area_map(data)           # equirectangular map of the same
    fig = plot.spectrum(data)           # multipole power per degree
"""
from __future__ import annotations

import numpy as np

from hexasphere.analysis import GridData, ICOSAHEDRAL_DEGREES, multipole_spectrum


def _plt():
    import matplotlib
    import matplotlib.pyplot as plt
    return matplotlib, plt


def _shade(data: GridData, vmin=None, vmax=None, values=None):
    from matplotlib import cm, colors
    r = data.relative_areas if values is None else np.asarray(values, float)
    norm = colors.Normalize(vmin=r.min() if vmin is None else vmin,
                            vmax=r.max() if vmax is None else vmax)
    cmap = cm.get_cmap("Blues") if hasattr(cm, "get_cmap") else cm.Blues
    return r, norm, cmap


def grid_3d(data: GridData, elev=20, azim=30, ax=None, title=None,
            vmin=None, vmax=None, values=None, label="tile area / mean", zoom=1.0):
    """Tiles drawn as flat spherical-polygon fans on a 3D sphere."""
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    _, plt = _plt()
    r, norm, cmap = _shade(data, vmin, vmax, values)
    if ax is None:
        fig = plt.figure(figsize=(5.5, 5.5))
        ax = fig.add_subplot(111, projection="3d")
    else:
        fig = ax.figure
    polys, cols = [], []
    for c, v, ri in zip(data.centers, data.polygons, r):
        # subdivide each edge a few times so edges follow the sphere
        ring = []
        for i in range(len(v)):
            a, b = v[i], v[(i + 1) % len(v)]
            for t in np.linspace(0, 1, 5, endpoint=False):
                p = (1 - t) * a + t * b
                ring.append(p / np.linalg.norm(p))
        polys.append(np.array(ring))
        cols.append(cmap(norm(ri)))
    # only draw tiles facing the viewer: cheap hidden-surface removal
    e, a_ = np.radians(elev), np.radians(azim)
    view = np.array([np.cos(e) * np.cos(a_), np.cos(e) * np.sin(a_), np.sin(e)])
    keep = [i for i, c in enumerate(data.centers) if c @ view > -0.15]
    coll = Poly3DCollection([polys[i] for i in keep],
                            facecolors=[cols[i] for i in keep],
                            edgecolors="#1f2a37", linewidths=0.4)
    ax.add_collection3d(coll)
    ax.set_xlim(-1, 1); ax.set_ylim(-1, 1); ax.set_zlim(-1, 1)
    ax.set_box_aspect((1, 1, 1), zoom=zoom); ax.view_init(elev, azim); ax.set_axis_off()
    if title:
        ax.set_title(title, fontsize=10)
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    fig.colorbar(sm, ax=ax, shrink=0.55, pad=0.0, label=label)
    return fig


def area_map(data: GridData, ax=None, title=None, vmin=None, vmax=None,
             values=None, label="tile area / mean"):
    """Equirectangular map; tiles that wrap the antimeridian are split."""
    from matplotlib.collections import PolyCollection
    _, plt = _plt()
    r, norm, cmap = _shade(data, vmin, vmax, values)
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 4.2))
    else:
        fig = ax.figure
    verts, cols = [], []
    for v, ri in zip(data.polygons, r):
        lon = np.degrees(np.arctan2(v[:, 1], v[:, 0]))
        lat = np.degrees(np.arcsin(np.clip(v[:, 2], -1, 1)))
        for shift in (0, 360):
            lo = np.where(lon < 0, lon + 360, lon) if shift else lon
            if shift == 0 and lon.max() - lon.min() > 180:
                continue          # wraps: draw the unwrapped copy instead
            if shift == 360 and lon.max() - lon.min() <= 180:
                continue
            pts = np.c_[lo, lat]
            verts.append(pts); cols.append(cmap(norm(ri)))
            if shift == 360:       # also draw its copy shifted by -360
                verts.append(pts - [360, 0]); cols.append(cmap(norm(ri)))
    ax.add_collection(PolyCollection(verts, facecolors=cols,
                                     edgecolors="#1f2a37", linewidths=0.3))
    ax.set_xlim(-180, 180); ax.set_ylim(-90, 90)
    ax.set_xlabel("longitude (deg)"); ax.set_ylabel("latitude (deg)")
    ax.set_aspect("equal")
    if title:
        ax.set_title(title, fontsize=10)
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    fig.colorbar(sm, ax=ax, shrink=0.8, label=label)
    return fig


def spectrum(data: GridData, lmax: int = 20, ax=None, title=None):
    """Bar chart of Q_l for the tile-area field.

    Degrees that carry icosahedral-invariant harmonics are hatched; all other
    degrees are solid.  The theory says the solid bars must be (numerically)
    empty, so the y axis is logarithmic.
    """
    _, plt = _plt()
    Q = multipole_spectrum(data, lmax=lmax)[1:]
    ls = np.arange(1, lmax + 1)
    floor = Q.max() * 1e-18
    allowed = np.isin(ls, ICOSAHEDRAL_DEGREES)
    if ax is None:
        fig, ax = plt.subplots(figsize=(6.2, 3.4))
    else:
        fig = ax.figure
    ax.bar(ls[allowed], np.maximum(Q[allowed], floor), color="#2c6fb7",
           hatch="//", edgecolor="white", label="I-invariant degree")
    ax.bar(ls[~allowed], np.maximum(Q[~allowed], floor), color="#8a8f98",
           label="no I-invariant harmonic")
    ax.set_yscale("log"); ax.set_ylim(floor, Q.max() * 10)
    ax.set_xlabel("degree l"); ax.set_ylabel("Q_l  (power of area field)")
    ax.set_xticks(ls[::1] if lmax <= 20 else ls[::2])
    ax.legend(fontsize=8, loc="lower left")
    if title:
        ax.set_title(title, fontsize=10)
    return fig
