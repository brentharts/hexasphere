#!/usr/bin/env python3
"""
Build the Hat Trick ball (60 hats + 12 pentagons) and draw it.

    python examples/hat_trick_ball.py                    # variant 0, figures + json in docs/
    python examples/hat_trick_ball.py --variant 1
    python examples/hat_trick_ball.py --grid snyder      # build on the tile centres of a hexasphere grid
    python examples/hat_trick_ball.py --mirror           # the other hand

Writes docs/hat_trick_ball.png (three views), hat_trick_patch.png, hat_trick_strain.png, hat_trick_hat.png,
hat_trick_variants.png (both icosahedral tilings, same patch) and docs/hat_trick_ball.json.
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from hexasphere import hexgrid, projection                      # noqa: E402
from hexasphere.hattrick import HatBall, plot                    # noqa: E402

DOCS = os.path.join(os.path.dirname(__file__), "..", "docs")


def build(args, variant):
    if args.grid == "none":
        return HatBall.from_frequency(3, variant=variant)
    g = hexgrid.HexGrid()
    g.projection = {"gnomonic": projection.GnomonicProj, "snyder": projection.SnyderEAProj}[args.grid](g)
    return HatBall.from_grid(g, 2, variant=variant)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--variant", type=int, default=0)
    ap.add_argument("--grid", choices=["none", "gnomonic", "snyder"], default="none",
                    help="start from the tile centres of a hexasphere grid (n = 2)")
    ap.add_argument("--mirror", action="store_true", help="mirror image of the ball")
    ap.add_argument("--out", default=DOCS)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    ball = build(args, args.variant)
    ball.relax()
    if args.mirror:
        ball.X = ball.X * np.array([-1.0, 1.0, 1.0])
        ball.V = ball.V * np.array([-1.0, 1.0, 1.0])
    rep = ball.report()
    for k, v in rep.items():
        print(f"{k:>24}: {v}")

    plot.views(ball, os.path.join(args.out, "hat_trick_ball.png"))
    plot.patch(ball, os.path.join(args.out, "hat_trick_patch.png"))
    plot.strain_map(ball, os.path.join(args.out, "hat_trick_strain.png"))
    plot.hat(os.path.join(args.out, "hat_trick_hat.png"))

    # both tilings, the same patch
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, len(ball.tilings), figsize=(5.8 * len(ball.tilings), 6))
    for v, ax in zip(range(len(ball.tilings)), np.atleast_1d(axes)):
        b = build(args, v)
        b.relax()
        plot.draw_ball(ax, b, b.V[b.pent[0]], zoom=1.7, highlight=b.pent[0], colour="hat")
        ax.set_title(f"variant {v}", fontsize=11)
    fig.suptitle("The two icosahedrally symmetric hat tilings: same patch outline, different cuts", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(os.path.join(args.out, "hat_trick_variants.png"), dpi=130)

    with open(os.path.join(args.out, "hat_trick_ball.json"), "w") as f:
        json.dump({"report": rep, "panels": ball.panels()}, f)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
