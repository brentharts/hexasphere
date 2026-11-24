"""Chiral hexagonal tilings from the icosahedral equivariant flow.

    python examples/chiral_tilings.py [n]       # default n = 8
Writes docs/chiral_tilings.png, docs/chiral_sweep.json
"""
import json
import pathlib
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from hexasphere import analysis, chiral, hexgrid, plot, projection

n = int(sys.argv[1]) if len(sys.argv) > 1 else 8
out = pathlib.Path(__file__).resolve().parent.parent / "docs"
out.mkdir(exist_ok=True)

grid = hexgrid.HexGrid()
grid.projection = projection.GnomonicProj(grid)
base = analysis.grid_data(grid, n)
flow = chiral.IcosahedralFlow(chiral.five_fold_axes(grid))
P0 = chiral.perimeters(base)

sweep = []
for tau in [0.0, 0.025, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3]:
    c = chiral.chiral_grid(grid, base, tau, edge_samples=16, flow=flow)
    per = chiral.perimeters(c)
    sym = chiral.symmetry_counts(grid, c.centers)
    sweep.append({"tau": tau,
                  "max_rel_area_change": float(np.abs(c.areas / base.areas - 1).max()),
                  "mean_perimeter_ratio": float(np.mean(per / P0)),
                  "max_perimeter_ratio": float(np.max(per / P0)),
                  "rotations": sym["proper"], "improper": sym["improper"]})
(out / "chiral_sweep.json").write_text(json.dumps({"n": n, "rows": sweep}, indent=2))

TAUS = [0.0, 0.1, 0.2]
VMIN, VMAX = 1.0, 1.45
fig = plt.figure(figsize=(11, 8.4))
gs = fig.add_gridspec(2, 3, height_ratios=[1, 0.85], hspace=0.08, wspace=0.28)
for i, tau in enumerate(TAUS):
    c = chiral.chiral_grid(grid, base, tau, edge_samples=12, flow=flow)
    ax = fig.add_subplot(gs[0, i], projection="3d")
    plot.grid_3d(c, ax=ax, elev=25, azim=20, zoom=1.45, vmin=VMIN, vmax=VMAX,
                 values=chiral.perimeters(c) / P0, label="perimeter / original",
                 title=f"tau = {tau:g}" + (" (achiral)" if tau == 0 else ""))
    if i < 2:   # keep one colour bar
        for a in fig.axes[-1:]:
            if a is not ax:
                a.remove()
c = chiral.chiral_grid(grid, base, 0.2, edge_samples=12, flow=flow)
axm = fig.add_subplot(gs[1, :2])
plot.area_map(c, ax=axm, vmin=VMIN, vmax=VMAX, values=chiral.perimeters(c) / P0,
              label="perimeter / original", title="tau = 0.2, equirectangular")
axw = fig.add_subplot(gs[1, 2])
tt = [r["tau"] for r in sweep]
axw.plot(tt, [r["mean_perimeter_ratio"] for r in sweep], "o-", color="#2c6fb7", label="mean")
axw.plot(tt, [r["max_perimeter_ratio"] for r in sweep], "s--", color="#1b3a5c", label="max")
axw.set_xlabel("flow time tau (radians of arc)")
axw.set_ylabel("tile perimeter / original")
axw.set_title("cost of chirality: longer seams\n(areas unchanged to < 1e-4)", fontsize=10)
axw.legend(fontsize=8); axw.grid(alpha=0.3)
fig.savefig(out / "chiral_tilings.png", dpi=150, bbox_inches="tight")
print(json.dumps(sweep, indent=1))
