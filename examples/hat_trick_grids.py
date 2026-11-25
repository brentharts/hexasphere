#!/usr/bin/env python3
"""
Hat Trick ball on the tile centres of the two hexasphere projections (n = 2: 92 tiles).

The combinatorial tiling is the same for any projection; what changes is the starting geometry.  This
measures, for Gnomonic and Snyder equal-area centres, how far the hat kites are from flat kites before and
after the relaxation, and the hat : pentagon area ratio.

    python examples/hat_trick_grids.py          # -> docs/hat_trick_grids.json
"""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from hexasphere import hexgrid, projection                      # noqa: E402
from hexasphere.hattrick import HatBall                          # noqa: E402

DOCS = os.path.join(os.path.dirname(__file__), "..", "docs")


def main():
    rows = {}
    for name, P in (("gnomonic", projection.GnomonicProj), ("snyder", projection.SnyderEAProj)):
        g = hexgrid.HexGrid()
        g.projection = P(g)
        b = HatBall.from_grid(g, 2)
        start = b.relax(max_nfev=0)
        h0, p0 = b.panel_areas()
        end = b.relax()
        h1, p1 = b.panel_areas()
        rows[name] = {
            "tilings": len(b.tilings),
            "start_rms_strain": start["rms_strain"], "start_max_strain": start["max_strain"],
            "start_hat_to_pentagon_area": float(h0.mean() / p0.mean()),
            "relaxed_rms_strain": end["rms_strain"], "relaxed_max_strain": end["max_strain"],
            "relaxed_hat_to_pentagon_area": float(h1.mean() / p1.mean()),
        }
        print(name, rows[name])
    with open(os.path.join(DOCS, "hat_trick_grids.json"), "w") as fh:
        json.dump(rows, fh, indent=1)


if __name__ == "__main__":
    main()
