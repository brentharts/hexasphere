"""Compare the Gnomonic and Snyder grids: area quality and multipole spectrum.

    python examples/analyze_grids.py [n]        # default n = 8
Writes docs/grid_analysis.json and docs/grid_*.png.
"""
import json
import pathlib
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from hexasphere import analysis, hexgrid, plot, projection

n = int(sys.argv[1]) if len(sys.argv) > 1 else 8
out = pathlib.Path(__file__).resolve().parent.parent / "docs"
out.mkdir(exist_ok=True)

projs = {"gnomonic": projection.GnomonicProj, "snyder": projection.SnyderEAProj}
report, datas = {}, {}
for name, cls in projs.items():
    grid = hexgrid.HexGrid()
    grid.projection = cls(grid)
    datas[name] = analysis.grid_data(grid, n)
    report[name] = analysis.area_stats(datas[name])
(out / "grid_analysis.json").write_text(json.dumps(report, indent=2))

vmin = min(d.relative_areas.min() for d in datas.values())
vmax = max(d.relative_areas.max() for d in datas.values())
for name, d in datas.items():
    s = report[name]
    t = f"{name}, n={n}: {s['tiles']} tiles, max/min area {s['max_over_min']:.2f}"
    plot.grid_3d(d, title=t, vmin=vmin, vmax=vmax).savefig(
        out / f"grid_3d_{name}.png", dpi=150, bbox_inches="tight")
    plot.area_map(d, title=t, vmin=vmin, vmax=vmax).savefig(
        out / f"grid_map_{name}.png", dpi=150, bbox_inches="tight")
    plot.spectrum(d, title=f"{name}: area-field spectrum").savefig(
        out / f"grid_spectrum_{name}.png", dpi=150, bbox_inches="tight")
    plt.close("all")
print(json.dumps(report, indent=2))
