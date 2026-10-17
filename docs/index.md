# bipgen

**bipgen** generates spatial bipartite proximity graphs: two populations of nodes,
called BCN and TGT, placed in the unit disk (2D) or the unit ball (3D), linked by
edges whose probability decays with distance.

Graphs of this kind are the raw output of sequencing-based spatial methods, where
molecular barcodes that end up close to each other get linked, and the spatial
layout has to be inferred from the links alone. bipgen produces graphs in which every
node has a known position, with sizes, degree distributions and spatial density that
can be taken from an observed graph.

## The model in four steps

1. **Positions**, uniform or with a spatially correlated density.
2. **Target degrees** for both sides, from a distribution or from an observed graph.
3. **Geometric edges**: a distance kernel whose node weights are fitted by Sinkhorn
   scaling, so that expected degrees match the targets; or each BCN linked to its k
   nearest TGT.
4. **Background edges**, unrelated to distance, added in a chosen proportion.

Everything is set by one YAML [specification](specification.md); a seed selects one
realization of the ensemble it describes.

```python
import bipgen

spec = bipgen.spec_from_dict({
    "sizes": {"n_bcn": 2000, "n_tgt": 3000},
    "degrees": {"distribution": "lognormal", "bcn": {"median": 15, "sigma": 0.5},
                "tgt": {"sigma": 0.6}},
    "edges": {"model": "kernel", "kernel": "gaussian", "scale": 0.12},
    "background": {"model": "uniform", "ratio": 0.05},
})
graph = bipgen.generate(spec)          # seed 42 unless set otherwise
print(graph)
bipgen.plot_graph(graph)
```

Next: [installation](installation.md), the [quickstart](quickstart.md), and the
[model](model.md) with its assumptions and known limitations.
