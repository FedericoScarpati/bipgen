# Specification format

A specification is a YAML file (or a dictionary with the same layout) that fully
determines an ensemble of graphs. Arrays, such as empirical degree sequences, live in
an optional `.npz` file next to it, referenced by `arrays`.

All lengths are in units of the domain radius: the domain is the unit disk in 2D and
the unit ball in 3D. Unknown keys raise `SpecError`.

## Top level

| Key | Default | Meaning |
|---|---|---|
| `schema` | `0` | version of the format |
| `seed` | `42` | default seed; `generate(spec, seed=...)` overrides it |
| `dim` | `2` | `2` or `3` |
| `sizes` | required | `n_bcn`, `n_tgt`: number of nodes of each side |
| `positions` | uniform | see below |
| `degrees` | required for the kernel model | see below |
| `edges` | required | see below |
| `background` | none | see below |
| `arrays` | `null` | companion `.npz` file, relative to the YAML file |

## `positions`

| Key | Values | Meaning |
|---|---|---|
| `method` | `uniform`, `metropolis` | uniform in the domain, or a spatially correlated density (2D only) |
| `density.n_bins` | default `20` | grid of `n_bins x n_bins` cells over the disk |
| `density.occupancy` | `{source: gamma, cv: 0.5}` or `{source: file, key: ...}` | cell counts: Gamma weights with coefficient of variation `cv`, or counts from the arrays file |
| `density.correlogram` | default `[0.3, 0.1]` | target Moran's I at lags 1, 2, ... |
| `density.sweeps` | default `400` | annealing sweeps |

## `degrees`

Target degrees are integers, and the totals of the two sides are equal. The TGT
distribution only gives a shape: its scale follows from
$n_\mathrm{bcn} \langle k_\mathrm{bcn} \rangle = n_\mathrm{tgt} \langle k_\mathrm{tgt} \rangle$.

| `distribution` | `bcn` | `tgt` |
|---|---|---|
| `lognormal` | `median`, `sigma` | `sigma` |
| `constant` | `value` | (empty) |
| `empirical` | `key` (array in the `.npz`) | `key` |

Empirical sequences are resampled with replacement when their length differs from the
number of nodes. Degrees are assigned to nodes at random.

## `edges`

| Key | Meaning |
|---|---|
| `model` | `kernel` (degree-matched kernel model) or `knn` |
| `kernel` | name in the kernel registry; `gaussian` |
| `scale` | kernel length $L$, required with `kernel` |
| `sinkhorn` | optional solver settings: `tol` (default `1e-3`), `max_iter` (default `2000`) |
| `k` | with `knn`: each BCN links to its `k` nearest TGT; `degrees` is not used |

## `background`

| Key | Meaning |
|---|---|
| `model` | `none`, `uniform` (endpoints uniform), `configuration` (endpoints proportional to target degrees, or realized degrees for `knn`) |
| `ratio` | background edges per geometric edge |

## Example

```yaml
--8<-- "docs/spec_example.yaml"
```
