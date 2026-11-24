import numpy as np
import pytest
from scipy.spatial import cKDTree

from hexasphere import analysis, hexgrid, projection
from hexasphere.geometry import phi

PROJS = {"gnomonic": projection.GnomonicProj, "snyder": projection.SnyderEAProj}


def _grid(name):
    g = hexgrid.HexGrid()
    g.projection = PROJS[name](g)
    return g


@pytest.fixture(scope="module", params=["gnomonic", "snyder"])
def setup(request):
    g = _grid(request.param)
    return request.param, g, analysis.grid_data(g, 6)


@pytest.mark.parametrize("n", [0, 1, 2, 5])
def test_tile_count_and_pentagons(n):
    g = _grid("gnomonic")
    d = analysis.grid_data(g, n)
    assert len(d.ids) == analysis.n_tiles(n) == len(set(d.ids))
    assert (d.n_sides == 5).sum() == 12
    assert (d.n_sides == 6).sum() == analysis.n_tiles(n) - 12


def test_areas_tile_the_sphere(setup):
    name, g, d = setup
    assert d.areas.sum() == pytest.approx(4 * np.pi, rel=1e-9)
    assert np.all(d.areas > 0)
    assert np.all(np.isfinite(d.centers))


def test_pentagon_centres_are_icosahedron_vertices(setup):
    # regression: the Snyder inverse used an exact float comparison and
    # returned NaN at some pentagon centres
    _, g, d = setup
    V = np.unique(np.round(np.array([g.abc[f][i] for f in range(20)
                                      for i in range(3)]), 9), axis=0)
    pent = d.centers[d.n_sides == 5]
    dist, _ = cKDTree(V).query(pent)
    assert dist.max() < 1e-9


def test_snyder_is_more_equal_area_than_gnomonic():
    s = {k: analysis.area_stats(analysis.grid_data(_grid(k), 5)) for k in PROJS}
    assert s["snyder"]["cv"] < 0.5 * s["gnomonic"]["cv"]
    assert s["snyder"]["max_over_min"] < s["gnomonic"]["max_over_min"]


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


def test_grid_invariant_under_all_60_icosahedral_rotations(setup):
    _, g, d = setup
    R = _rotations(g)
    assert len(R) == 60
    tree = cKDTree(d.centers)
    for M in R:
        dist, idx = tree.query(d.centers @ M.T)
        assert dist.max() < 1e-9
        assert np.abs(d.areas[idx] - d.areas).max() < 1e-9


def test_area_field_is_inversion_symmetric(setup):
    # makes odd degrees (including l=15) vanish
    _, g, d = setup
    dist, idx = cKDTree(d.centers).query(-d.centers)
    assert dist.max() < 1e-9
    assert np.abs(d.areas[idx] - d.areas).max() < 1e-9


def test_multipole_selection_rule(setup):
    """Only degrees with I-invariant harmonics may carry power."""
    _, g, d = setup
    Q = analysis.multipole_spectrum(d, lmax=16)
    allowed = np.isin(np.arange(17), analysis.ICOSAHEDRAL_DEGREES)
    assert Q[6] > 0                      # the leading non-trivial degree
    assert Q[1:6].max() < 1e-20 * Q[6]
    assert Q[~allowed][1:].max() < 1e-18 * Q[6]


def test_constant_field_has_no_higher_multipoles(setup):
    _, g, d = setup
    Q = analysis.multipole_spectrum(d, values=np.ones(len(d.ids)), lmax=8)
    # quadrature by tile areas is not exact for a pure l=0 field at l>0, but
    # it must be tiny
    assert Q[1:].max() < 1e-3 * Q[0]


def test_hit_check_consistent():
    g = _grid("gnomonic")
    assert analysis.hit_check(g, 4, samples=300) > 0.99


def test_snyder_inv_project_vertex_exact():
    g = _grid("snyder")
    v = g.abc[3][1]
    P = g.projection.project(v, 3)
    assert np.allclose(g.projection.inv_project(P, 3), v, atol=1e-9)


def test_plot_smoke(tmp_path):
    import matplotlib
    matplotlib.use("Agg")
    from hexasphere import plot
    d = analysis.grid_data(_grid("snyder"), 2)
    for fn in (plot.grid_3d, plot.area_map, plot.spectrum):
        fig = fn(d)
        fig.savefig(tmp_path / f"{fn.__name__}.png", dpi=40)
    assert len(list(tmp_path.glob("*.png"))) == 3
