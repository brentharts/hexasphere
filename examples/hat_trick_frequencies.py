#!/usr/bin/env python3
"""
For which hexagonal-grid resolutions can the hat tile the sphere with 12 pentagon holes?

The hexasphere grid with resolution n has 10 (n+1)^2 + 2 tiles; f = n + 1 is the frequency.  Cutting the
80-or-more hexagons into kites gives 60 (f^2 - 1) kites = 7.5 (f^2 - 1) hats, an integer only for odd f.
For every odd f this script
  * embeds the hat in every position (same hand, and with its mirror image too),
  * searches exact covers of all kites by hats (Knuth's algorithm X) for several symmetry groups,
and reports whether the search was exhaustive.

    python examples/hat_trick_frequencies.py            # f = 3, 5, 7, 9 -> docs/hat_trick_frequencies.json
"""
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from hexasphere.hattrick import HatBall                                  # noqa: E402

DOCS = os.path.join(os.path.dirname(__file__), "..", "docs")
CASES = [("I", False), ("T", False), ("C5", False), ("C1", False), ("C1", True)]


def main():
    out = []
    for f in (3, 5, 7, 9):
        row = {"f": f, "n": f - 1, "tiles": 10 * f * f + 2, "kites": 60 * (f * f - 1), "hats": 7.5 * (f * f - 1)}
        for sym, both in CASES:
            key = f"{sym}{'_both_hands' if both else ''}"
            t = time.time()
            try:
                b = HatBall.from_frequency(f, symmetry=sym, both_hands=both, max_tilings=1000, max_nodes=500000)
                row[key] = {"tilings": len(b.tilings), "exhausted": b.search["exhausted"], "nodes": b.search["nodes"]}
                row["placements"] = len(b.placements)
            except ValueError as e:
                s = str(e)
                row[key] = {"tilings": 0, "exhausted": "'exhausted': True" in s}
            row[key]["seconds"] = round(time.time() - t, 1)
            print(f, key, row[key], flush=True)
        out.append(row)
    with open(os.path.join(DOCS, "hat_trick_frequencies.json"), "w") as fh:
        json.dump(out, fh, indent=1)


if __name__ == "__main__":
    main()
