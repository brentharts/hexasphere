"""Hat Trick ball: the hat on the kite surface of the hexagonal grid with 12 pentagon holes."""
import math

import numpy as np
import pytest

from hexasphere import hexgrid, projection
from hexasphere.hattrick import HatBall, corner_angles, flat_hat, hat_adjacency, rotation_group
from hexasphere.hattrick.kites import HAT_ANGLES, KiteComplex, embed_hat, geodesic_sphere, rotation_subgroup


@pytest.fixture(scope="module")
def ball():
    b = HatBall.from_frequency(3)
    b.relax()
    return b


def test_flat_hat_is_the_hat():
    kites, outline = flat_hat()                       # raises if the adjacency is inconsistent
    assert len(kites) == 8
    assert sorted(corner_angles(outline)) == HAT_ANGLES
    n = len(outline)
    area = 0.5 * sum(outline[i, 0] * outline[(i + 1) % n, 1] - outline[(i + 1) % n, 0] * outline[i, 1]
                     for i in range(n))
    assert area == pytest.approx(8 * math.sqrt(3), abs=1e-2)
    L = np.array([np.linalg.norm(outline[(i + 1) % n] - outline[i]) for i in range(n)])
    assert np.sum(abs(L - 1) < 1e-2) == 8 and np.sum(abs(L - math.sqrt(3)) < 1e-2) == 6   # 2-edge = 2 segments


def test_mirror_hat_is_the_mirror():
    a, b = hat_adjacency(), hat_adjacency(mirror=True)
    assert sorted(a) != sorted(b) and len(a) == len(b) == 18


def test_grid_is_goldberg_3_0(ball):
    assert len(ball.V) == 92 == 10 * (2 + 1) ** 2 + 2        # hexasphere n = 2
    assert len(ball.pent) == 12 and len(ball.S.kites) == 480
    assert len(ball.G) == 60


def test_placements_and_tilings(ball):
    assert len(ball.placements) == 360
    assert len(ball.tilings) == 2
    assert ball.search["exhausted"]
    for t in ball.tilings:
        assert len(t) == 60 and len(set().union(*t)) == 480


@pytest.mark.parametrize("variant", [0, 1])
def test_72_panels(variant):
    b = HatBall.from_frequency(3, variant=variant)
    assert b.n_hats == 60 and sorted(k for h in b.hats for k in h) == sorted(b.S.kites)
    for p in b.pent:
        assert len(b.pentagon_hats(p)) == 5
        assert len(b.patch(p)) == 40
    for i in range(60):
        loop, true, ang = b.hat_outline(i)
        assert len(true) == 13 and sorted(ang[j] for j in true) == HAT_ANGLES


def test_one_orbit(ball):
    h0 = frozenset(ball.hats[0])
    assert {frozenset(kp[k] for k in h0) for kp in ball.kperm} == {frozenset(h) for h in ball.hats}


def test_pentagon_ledger(ball):
    """Every pentagon corner (a triangle centroid next to the pentagon) carries two kites, 240 degrees: a
    120-degree pentagon with K = 60 degrees; 12 pentagons give 720."""
    for p in ball.pent:
        tris = [t for t, f in enumerate(ball.F) if p in f]
        assert len(tris) == 5
        for t in tris:
            assert sum(1 for j in range(3) if ball.F[t][j] not in ball.S.skip) == 2
    assert 12 * (5 * 120 - 540) == 720


def test_relaxation(ball):
    assert np.allclose(np.linalg.norm(ball.X, axis=1), 1)
    assert ball.info["rms_strain"] < 0.03 and ball.info["max_strain"] < 0.07
    hats, pents = ball.panel_areas()
    assert hats.sum() + pents.sum() == pytest.approx(4 * math.pi, rel=1e-6)
    assert (hats.max() - hats.min()) / hats.mean() < 1e-4
    assert 2.0 < hats.mean() / pents.mean() < 2.6
    assert max(ball.kite_strain().values()) < 0.04


@pytest.mark.parametrize("proj", [projection.GnomonicProj, projection.SnyderEAProj])
def test_on_hexasphere_grid(proj, ball):
    g = hexgrid.HexGrid()
    g.projection = proj(g)
    b = HatBall.from_grid(g, 2)
    assert len(b.tilings) == 2 and len(b.pent) == 12 and b.n_hats == 60
    start = b.relax(max_nfev=0)
    end = b.relax()
    assert end["rms_strain"] < start["rms_strain"]
    assert end["rms_strain"] == pytest.approx(ball.info["rms_strain"], rel=1e-3)   # relaxation forgets the projection


def test_subgroups():
    G = rotation_group()
    for name, size in (("I", 60), ("T", 12), ("D5", 10), ("D3", 6), ("D2", 4), ("C5", 5), ("C3", 3), ("C2", 2), ("C1", 1)):
        assert len(rotation_subgroup(G, name)) == size
    with pytest.raises(ValueError):
        rotation_subgroup(G, "O")


def test_symmetry_choices_at_f3():
    assert len(HatBall.from_frequency(3, symmetry="T").tilings) == 2        # no extra tilings with only T
    assert len(HatBall.from_frequency(3, symmetry="C5", max_tilings=100).tilings) >= 2


def test_mirror_hats_add_placements():
    b = HatBall.from_frequency(3, both_hands=True)
    assert set(b.hand.values()) == {1, -1} and len(b.placements) == 720


@pytest.mark.parametrize("f", [5, 7])
def test_no_tiling_at_larger_frequency(f):
    """Exhaustive searches: hats cannot cover the f = 5 and 7 kite surfaces for any symmetry."""
    for sym in ("I", "C1"):
        with pytest.raises(ValueError, match="exhausted': True"):
            HatBall.from_frequency(f, symmetry=sym, max_nodes=500000)
