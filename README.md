# hexasphere
A module to create (almost) hexagonal grids of variable size on a sphere, fully implemented in python.

## description of the grid

### base polyhedron

The grid is built by subdividing a base polyhedron into hexagon tiles.
The base polyhedron is an icosahedron, which comprises:
- 20 faces (equilateral triangles)
- 12 vertices
- 30 edges

At the vertices of the icosahedron, instead of an hexagon, a pentagon is constructed.

For a given grid resolution `n`, each edge of the icosahedron goes through exactly `n` hexagon centers (not counting the polygons found at the ends). The inside of faces is then filled with tile-centers by following a triangular pattern.
This allows to cover the icosahedron. Its surface is then mapped to the sphere using a projection.

![icogrid](https://user-images.githubusercontent.com/70936497/186472661-e6255c76-46ae-4ce8-9b13-32c0683ee48b.jpeg)
*Here, `n = 7`*

### projection

There are two available projections in `projection` module:
- `GnomonicProj`: a simple projection, which produces hexagonal tiles about 60% larger (in area) at the corners of a face than at its center.

- `SnyderEAProj`: a more complex projection, slower to compute (roughly 3x slower than Gnomonic projection), but which preserves areas. The implementation is based on [Brenton R S Recht's blog](https://brsr.github.io/2021/08/31/snyder-equal-area.html). See there for more details.

### hexagon tile identifier

A tile identifier has the following pattern: `?XXXXX-YYYYY-ZZZZZ`
- `?` is one of the 20 letters `A ... T`, each letter corresponding to one face of the icosahedron
- `XXXXX`, `YYYYY`, `ZZZZZ` are the integer coordinates of the tile in the triangular mesh covering face `?`. An useful property holds:

`XXXXX + YYYYY + ZZZZZ = 2 * (n + 1)`

---

## getting started

1. Install the package with pip

`$ pip install hexasphere`

2. Import the library in python

`from hexasphere import hexgrid, projection`

## usage

### construction of a grid

1. Create a `HexGrid` object:

`my_grid = hexgrid.HexGrid()`

2. Instantiate a projection system `Projection` associated with this grid:

`my_projection = projection.MyProjection(my_grid)`

3. Provide the projection system to the grid:

`my_grid.projection = my_projection`

### playing with grid resolutions

- Compute closest grid resolution `n` for any desired hex dimension **(in kilometers)**:

```
n = my_grid.compute_n_for_radius(0.25)
n = my_grid.compute_n_for_height(0.25)
n = my_grid.compute_n_for_side(0.25)
```

- Retrieve average hex dimension **(in kilometers)** for any given resolution `n`:

```
my_grid.compute_radius_for_n(n)
my_grid.compute_height_for_n(n)
my_grid.compute_side_for_n(n)
```

### encoding and decoding

- To find the string identifier of the hexagon to which a geographic point `(lat, lon)` belongs, call:

```
hex_identifier = my_grid.latlon_to_hex(lat, lon, n, out_str=True)[0]
```

- To find the `(lat, lon)` coordinates of the center of an hex, call:

```
my_grid.hex_to_latlon(hex_identifier, in_str=True)
my_grid.hex_to_latlon(hex_identifier, n, in_str=True) # n is here not required
```

### overlapping grids

`grid.latlon_to_hex` also supports overlapping grids:

```
value = 12 # Overlap distance (in km)
my_grid.set_overlap(value)
```

The method `grid.latlon_to_hex` returns the list of distinct hexes a point of coordinates `(lat, lon)` belongs to:

```
hexes_identifier = my_grid.latlon_to_hex(lat, lon, n, out_str=True)
```

<img width="634" alt="Screenshot 2022-09-13 at 12 09 33" src="https://user-images.githubusercontent.com/70936497/189876027-fc9e3867-613d-41b5-beb0-5b28f9e0117d.png">

### playing with hexagons

#### retrieving shape data

One can also deal with an `Hexagon` object instead of an hexagon string identifier:

```
hex_object = my_grid.latlon_to_hex(lat, lon, n)[0]
hex_object = hexgrid.Hexagon(my_grid, str_id=hexagon_identifier)
```

The coordinates of the vertices of the corresponding shape can then be retrieved:

```
shape_coordinates = hex_object.retrieve_polygon(out_latlon=True)
```

#### moving on the grid

To retrieve a neighboring hex:

```
hex_neighbor = hex_object.compute_neighbor(dP=(0, 1, -1))
```

To retrieve the list of hexes in the k-ring centered on the hex object:

```
hexes = hex_object.k_ring(k, out_str=True)
```

---

## grid analysis and plots (optional)

`hexasphere.analysis` (numpy + scipy) and `hexasphere.plot` (matplotlib) measure and draw a grid.
Install the extras with `pip install -e ".[analysis]"`.

```python
from hexasphere import hexgrid, projection, analysis, plot

grid = hexgrid.HexGrid()
grid.projection = projection.SnyderEAProj(grid)

data = analysis.grid_data(grid, n=8)      # 812 tiles: centres, exact areas, corners
analysis.area_stats(data)                 # min / max / cv of tile area, pentagon mean
analysis.multipole_spectrum(data)         # power per spherical-harmonic degree
plot.grid_3d(data); plot.area_map(data); plot.spectrum(data)
```

`python examples/analyze_grids.py 8` writes `docs/grid_analysis.json` and the figures in `docs/`.

### What it shows (n = 8, 812 tiles, 12 pentagons)

| projection | max/min tile area | cv of area | mean pentagon area / mean |
|---|---|---|---|
| Gnomonic | 2.51 | 0.139 | 0.479 |
| Snyder   | 1.29 | 0.027 | 0.780 |

Snyder is far more uniform, but its pentagons are still about 22% smaller than average.
Areas are those of the polygon through the tile corners with great-circle edges, which is exact for
Gnomonic (straight lines in the gnomonic plane are great circles) and an approximation for Snyder.

**Symmetry selection rule.** Both grids are invariant under all 60 rotations of the icosahedron
(checked numerically), and the tile-area field is also inversion symmetric. A function on the sphere
with that symmetry can only contain spherical-harmonic degrees l = 0, 6, 10, 12, 16, 18, 20, ...
(l = 1-5, 7-9, 11, 13-15, 17, 19 are absent). The measured spectrum agrees to about 1e-25 of the leading
l = 6 term, for both projections. So the lowest-order imperfection of an icosahedral grid is always an l = 6 pattern.

### Fixes to `SnyderEAProj`

- `inv_project` compared floats with `==` to detect a vertex, so some pentagon centres produced NaN
  (`0/0`); it now uses a tolerance.
- `project` returned the unscaled unit vector at an exact vertex instead of the face-plane point.

Both are covered by `tests/test_analysis.py`.

---

## chiral hexagonal tilings (`hexasphere.chiral`)

The hexagonal grid is mirror symmetric. Pushing it through an area-preserving, icosahedrally equivariant flow
gives a **chiral** tiling that keeps every tile's area and every shared seam, but loses all mirror symmetry
(a "pinwheel" hex grid, with a left- and a right-handed version).

```python
from hexasphere import hexgrid, projection, analysis, chiral, plot

grid = hexgrid.HexGrid(); grid.projection = projection.GnomonicProj(grid)
data = analysis.grid_data(grid, n=8)
twisted = chiral.chiral_grid(grid, data, tau=0.2)     # tau < 0 gives the mirror image
chiral.symmetry_counts(grid, twisted.centers)         # {'proper': 60, 'improper': 0, 'chiral': True}
plot.grid_3d(twisted, values=chiral.perimeters(twisted) / chiral.perimeters(data))
```

The flow is `v = p x grad f` with `f = sum_a (p.a)^6` over the six five-fold axes `a`. Since `f` is invariant
under the 60 icosahedral rotations, the flow commutes with them; it is divergence free, so it preserves area;
it vanishes at the 12 pentagon centres, 30 edge midpoints and 20 face centres; and it is odd under reflections,
so a mirrored tiling equals the flow at time `-tau`. This is the same construction as the chiral Trionda family in
the [ballgen](https://github.com/brentharts/ballgen) repository. `tau` is the largest distance in radians of arc
a point can move.

Measured on the Gnomonic grid, n = 8 (`python examples/chiral_tilings.py`, results in `docs/chiral_sweep.json`):

| tau | max change of a tile's area | mean tile perimeter | max tile perimeter | symmetries kept (of 60 + 60) |
|---|---|---|---|---|
| 0 | 1e-14 | 1.000 | 1.000 | 60 + 60 |
| 0.05 | 3e-6 | 1.009 | 1.025 | 60 + 0 |
| 0.1 | 1e-5 | 1.035 | 1.098 | 60 + 0 |
| 0.2 | 4e-5 | 1.135 | 1.379 | 60 + 0 |
| 0.3 | 6e-5 | 1.286 | 1.779 | 60 + 0 |

Area changes are discretisation error of the polygon rings (`edge_samples` points per edge), not of the flow.
Chirality has a cost: seams get longer and the tiles near the edge midpoints are sheared hard, so the useful
range is small `tau`. The test suite checks equivariance under the rotations, the inverse flow, the fixed points,
area preservation, the mirror relation, and that the two enantiomers are different tilings.
