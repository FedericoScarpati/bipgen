# Specification format

A specification is a YAML file (or a dictionary with the same layout) that fully
determines an ensemble of graphs. Arrays, such as empirical degree sequences or given
positions, live in an optional `.npz` file next to it, referenced by `arrays`; from
Python, `arrays` can also be a dictionary of arrays.

All lengths are in units of the domain radius: the domain is the unit disk in 2D and
the unit ball in 3D. Unknown keys raise `SpecError`.

## Top level

| Key | Default | Meaning |
|---|---|---|
| `seed` | `42` | default seed; `generate(spec, seed=...)` overrides it |
| `dim` | `2` | dimension of the domain: `2` (unit disk) or `3` (unit ball) |
| `sizes` | required | `n_bcn`, `n_tgt`: number of nodes of each side |
| `positions` | uniform | see below |
| `degrees` | required for the kernel model, absent for `knn` | see below |
| `edges` | required | see below |
| `background` | none | see below |
| `arrays` | `null` | companion `.npz` file, relative to the YAML file, or a dictionary of arrays |
| `schema` | `0` | version of the format (compatibility check) |

## `positions`

The `density` keys are used only with `method: metropolis`; see
[The model](model.md#positions).

| Key | Values | Meaning |
|---|---|---|
| `method` | `uniform`, `metropolis`, `file` | uniform in the domain, a spatially correlated density, or given positions |
| `density.n_bins` | default `20` in 2D, `12` in 3D | cells per axis of the grid on which the density is built |
| `density.occupancy` | `{source: gamma, cv: 0.5}` or `{source: file, key: ...}` | how many nodes each cell holds: drawn from a Gamma distribution with coefficient of variation `cv` (larger `cv`, more uneven cells), or taken from counts in the arrays file |
| `density.correlogram` | default `[0.3, 0.1]` | how alike nearby cells are: target spatial correlation between nearby cells (_Moran's I_), for cells 1, 2, ... cells apart; 0 means no spatial structure, values towards 1 give large smooth patches |
| `density.sweeps` | default `400` | length of the annealing that arranges the cells; one sweep proposes one swap per cell |
| `bcn`, `tgt` | array names | with `file`: the `(n_bcn, dim)` and `(n_tgt, dim)` position arrays, in the same units as `edges.scale` |

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
| `kernel` | name of the kernel; only `gaussian` in this version |
| `scale` | kernel length $L$, required with `kernel` |
| `sinkhorn` | optional solver settings: `tol`, the iterations stop when no node weight changes by more than this (default `1e-3`); `max_iter` (default `2000`) |
| `k` | with `knn`: each BCN links to its `k` nearest TGT; there is no `degrees` section |

## `background`

| Key | Meaning |
|---|---|
| `model` | `none`, `uniform` (endpoints uniform), `configuration` (endpoints proportional to the node weights $\theta$ of the kernel model, or to realized degrees for `knn`) |
| `ratio` | background edges per geometric edge |

## Example file

```yaml
--8<-- "docs/spec_example.yaml"
```

## Building a specification in Python

Only what differs from the defaults has to be given; `spec.to_dict()` shows the full
specification, with the defaults filled in.

```python
import bipgen

spec = bipgen.spec_from_dict({
    "sizes": {"n_bcn": 2000, "n_tgt": 3000},
    "degrees": {"distribution": "lognormal", "bcn": {"median": 15, "sigma": 0.5},
                "tgt": {"sigma": 0.6}},  # shape only: scale set by the BCN total
    "edges": {"model": "kernel", "scale": 0.12},
})
spec.save("my_spec.yaml")
spec = bipgen.load_spec("my_spec.yaml")
```

Arrays can be passed directly as a dictionary, here for given positions; `save` then
writes them to an `.npz` file next to the YAML file.

```python
spec = bipgen.spec_from_dict({
    "sizes": {"n_bcn": 2000, "n_tgt": 3000},
    "positions": {"method": "file", "bcn": "pos_bcn", "tgt": "pos_tgt"},
    "edges": {"model": "knn", "k": 5},
    "arrays": {"pos_bcn": pos_bcn, "pos_tgt": pos_tgt},   # (2000, 2) and (3000, 2)
})
spec.save("given.yaml")                                    # given.yaml + given.npz
```
