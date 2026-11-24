import numpy as np
import pytest
from scipy.spatial import cKDTree

from hexasphere import analysis, chiral, hexgrid, projection


@pytest.fixture(scope="module")
def base():
    g = hexgrid.HexGrid()
    g.projection = projection.GnomonicProj(g)
    d = analysis.grid_data(g, 5)
    return g, d, chiral.IcosahedralFlow(chiral.five_fold_axes(g))


def _rotations(g):
    F = [np.array(g.abc[f]) for f in range(20)]
    inv = np.linalg.inv(F[0].T)
    R = []
    for f in range(20):
        for k in range(3):
            M = np.roll(F[f], k, axis=0).T @ inv
            if np.allclose(M @ M.T, np.eye(3), atol=1e-8) and np.linalg.det(M) > 0:
                R.append(M)
    return R


def test_six_axes_and_field_is_tangent_and_bounded(base):
    g, d, fl = base
    assert fl.axes.shape == (6, 3)
    p = d.centers
    v = fl.field(p)
    assert np.abs(np.sum(v * p, axis=1)).max() < 1e-12
    assert np.linalg.norm(v, axis=1).max() <= 1.0 + 1e-9


def test_flow_is_equivariant_under_all_rotations(base):
    g, d, fl = base
    p = d.centers[::7]
    for M in _rotations(g)[::5]:
        a = fl.advance(p @ M.T, 0.15)
        b = fl.advance(p, 0.15) @ M.T
        assert np.abs(a - b).max() < 1e-10


def test_flow_inverse(base):
    _, d, fl = base
    p = d.centers
    assert np.abs(fl.advance(fl.advance(p, 0.2), -0.2) - p).max() < 1e-8


def test_fixed_points_of_stabilisers(base):
    g, _, fl = base
    V = np.array([g.abc[f][i] for f in range(20) for i in range(3)])
    mids = np.array([(g.abc[f][i] + g.abc[f][(i + 1) % 3]) for f in range(20) for i in range(3)])
    mids /= np.linalg.norm(mids, axis=1, keepdims=True)
    faces = np.array([g.k[f] for f in range(20)])
    for pts in (V, mids, faces):
        assert np.abs(fl.field(pts)).max() < 1e-12


def test_chiral_tiling_keeps_areas_and_symmetry(base):
    g, d, fl = base
    c = chiral.chiral_grid(g, d, 0.15, edge_samples=24, flow=fl)
    assert np.abs(c.areas / d.areas - 1).max() < 1e-4        # area preserving
    assert c.areas.sum() == pytest.approx(4 * np.pi, rel=1e-9)
    tree = cKDTree(c.centers)
    for M in _rotations(g)[::4]:
        dist, idx = tree.query(c.centers @ M.T)
        assert dist.max() < 1e-9                             # still I-symmetric
        assert np.abs(c.areas[idx] - c.areas).max() < 1e-4


def test_zero_tau_is_identity(base):
    g, d, fl = base
    c = chiral.chiral_grid(g, d, 0.0, flow=fl)
    assert np.abs(c.centers - d.centers).max() < 1e-14
    assert np.abs(c.areas / d.areas - 1).max() < 1e-9


def test_mirror_image_is_the_negative_flow_time(base):
    g, d, fl = base
    R = np.diag([-1.0, 1.0, 1.0])          # mirror plane of the icosahedron
    p = d.centers
    assert cKDTree(p).query(p @ R.T)[0].max() < 1e-9      # the base grid has this mirror
    a = fl.advance(p @ R.T, 0.2)
    b = fl.advance(p, -0.2) @ R.T
    assert np.abs(a - b).max() < 1e-9


def test_symmetry_counts_achiral_and_chiral(base):
    g, d, fl = base
    assert chiral.symmetry_counts(g, d.centers) == {"proper": 60, "improper": 60, "chiral": False}
    for tau in (0.02, 0.15):
        c = chiral.chiral_grid(g, d, tau, flow=fl)
        assert chiral.symmetry_counts(g, c.centers) == {"proper": 60, "improper": 0, "chiral": True}
    # the mirror image (flow time -tau) is chiral with the opposite handedness but the same counts
    m = chiral.chiral_grid(g, d, -0.15, flow=fl)
    assert chiral.symmetry_counts(g, m.centers)["improper"] == 0


def test_enantiomers_are_distinct_tilings(base):
    g, d, fl = base
    a = chiral.chiral_grid(g, d, 0.15, flow=fl).centers
    b = chiral.chiral_grid(g, d, -0.15, flow=fl).centers
    # no rotation of the sphere in the icosahedral group maps one onto the other
    tree = cKDTree(b)
    assert all(tree.query(a @ M.T)[0].max() > 1e-3 for M in chiral.icosahedral_rotations(g))
