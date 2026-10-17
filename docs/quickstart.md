# Quickstart

## A specification

A specification is a YAML file, or a dictionary with the same layout. Unknown keys
are rejected, so a typo raises an error instead of being ignored.

```python
import bipgen

spec = bipgen.load_spec("my_spec.yaml")
# or
spec = bipgen.spec_from_dict({
    "sizes": {"n_bcn": 2000, "n_tgt": 3000},
    "degrees": {"distribution": "lognormal", "bcn": {"median": 15, "sigma": 0.5},
                "tgt": {"sigma": 0.6}},
    "edges": {"model": "kernel", "kernel": "gaussian", "scale": 0.12},
})
```

See [Specification format](specification.md) for every field.

## Realizations and seeds

```python
graph = bipgen.generate(spec)            # seed from the specification, default 42
other = bipgen.generate(spec, seed=7)    # another realization of the same ensemble
ensemble = [bipgen.generate(spec, seed=s) for s in range(10)]
```

The seed actually used is stored in `graph.seed` and `graph.info["seed"]`. The same
seed gives the same graph.

## What a graph holds

```python
graph.pos_bcn, graph.pos_tgt     # positions, (n, dim) arrays
graph.geometric, graph.background  # disjoint (n_bcn, n_tgt) sparse 0/1 matrices
graph.biadjacency                # all edges
graph.edges()                    # table: bcn, tgt, origin
graph.degrees()                  # realized degrees of both sides
graph.target_degrees             # targets of the kernel model
graph.info                       # seed, target and expected edges, solver diagnostics
```

## Figures

```python
bipgen.plot_graph(graph)          # nodes and a subset of edges, 2D or 3D
bipgen.plot_degrees(graph)        # realized and target degree distributions
bipgen.plot_edge_lengths(graph)   # edge lengths by origin
```

## Saving

```python
bipgen.save(graph, "out/graph_042")
again = bipgen.load("out/graph_042")
```

The [tutorials](tutorials/01_getting_started.ipynb) go through all of this with figures.
