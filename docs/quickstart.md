# Quickstart

| Step | What it does | Functions |
|---|---|---|
| [1. Describe](#1-describe) | Describe the ensemble, in a specification | `load_spec`, `spec_from_dict`, `presets.load`, `spec_from_graph` |
| [2. Generate](#2-generate) | Generate one realization per seed | `generate` |
| [3. Inspect](#3-inspect) | Inspect contents, figures | `graph.edges()`, `plot_graph`, `plot_degrees` |
| [4. Compare](#4-compare) | Compare with an observed graph | `compare`, `degree_ks` |
| [5. Save](#5-save) | Save to disk and read back | `save`, `load` |

## 1. Describe

Describe the **specification** of a desired graph in a YAML file, or a dictionary of YAMLs with the same layout. Unknown keys
are rejected, so a typo raises an error instead of being ignored.

```python
import bipgen

spec = bipgen.load_spec("my_spec.yaml")
# or
spec = bipgen.spec_from_dict({
    "sizes": {"n_bcn": 2000, "n_tgt": 3000},
    "degrees": {"distribution": "lognormal", "bcn": {"median": 15, "sigma": 0.5},
                "tgt": {"sigma": 0.6}},  # shape only: scale set by the BCN total
    "edges": {"model": "kernel", "kernel": "gaussian", "scale": 0.12},
})
```

- **From a preset**, built from published data: `bipgen.presets.load("c58_4")`.
  See [Presets](presets.md).
- **From an observed graph**: sizes and degrees are read off its biadjacency matrix.
  See [Observed graphs](observed.md).

```python
adjacency = bipgen.read_edge_list("edges.csv")
spec = bipgen.spec_from_graph(adjacency, scale=0.15)
```

See [Specification format](specification.md) for every field.

## 2. Generate

```python
graph = bipgen.generate(spec)            # seed from the specification, default 42
other = bipgen.generate(spec, seed=7)    # another realization of the same ensemble
ensemble = [bipgen.generate(spec, seed=s) for s in range(10)]
```

The seed actually used is stored in `graph.seed` and `graph.info["seed"]`. The same
seed gives the same graph.

## 3. Inspect

**Contents**

```python
graph.pos_bcn, graph.pos_tgt       # positions, (n, dim) arrays
graph.geometric, graph.background  # disjoint (n_bcn, n_tgt) sparse 0/1 matrices
graph.biadjacency                  # all edges
graph.edges()                      # table: bcn, tgt, origin
graph.degrees()                    # realized degrees of both sides
graph.target_degrees               # targets of the kernel model
graph.info                         # seed, target and expected edges, solver diagnostics
```

**Figures**

```python
bipgen.plot_graph(graph)          # nodes and a subset of edges, 2D or 3D
bipgen.plot_degrees(graph)        # realized and target degree distributions
bipgen.plot_edge_lengths(graph)   # edge lengths by origin
```

## 4. Compare

```python
table = bipgen.compare(adjacency, ensemble)   # observed value, ensemble mean, sd, z-score
bipgen.plot_comparison(table)
bipgen.degree_ks(adjacency, ensemble)         # degree distributions
```

Statistics are computed in the same way on the observed graph and on each
realization. See [Observed graphs](observed.md).

## 5. Save

```python
bipgen.save(graph, "out/graph_042")      # edge table, node table, specification
again = bipgen.load("out/graph_042")
```

See [Saving and loading](io.md). The [tutorials](tutorials/01_getting_started.ipynb)
go through all of this with figures.
