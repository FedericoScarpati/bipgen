# bipgen

**bipgen** is a convenient tool to generate spatial bipartite proximity graphs: two populations of nodes, called BCN (_beacons_) and TGT (_targets_), placed in the unit disk (2D) or the unit ball (3D), linked by edges whose probability decays with distance.

Graphs of this kind are the raw output of sequencing-based spatial methods, where
molecular barcodes that end up close to each other get linked, and the spatial
layout has to be inferred from the links alone. bipgen produces graphs in which every
node has a known position, with sizes, degree distributions and spatial density that
can be taken from an observed graph or simply specified by the user.

## The model in four steps

1. **Positions**: node positions are drawn uniformly or from a spatially correlated
   density, or given by the user.
2. **Target degrees**: node degrees are drawn from a distribution or taken from an
   observed graph.
3. **Geometric edges**: nodes are linked according to their distance. In the kernel
   model the probability of a link decays with distance and is adjusted so that the
   expected degrees match the targets. In the k-nearest-neighbour model each BCN is
   linked to its k nearest TGT, and degree targets are not used.
4. **Background edges**: links unrelated to distance, which carry no information on
   how close two nodes are, added in a chosen proportion.

Everything is set by one YAML [specification](specification.md); a seed selects one
realization of the ensemble it describes.

```python
import bipgen

spec = bipgen.spec_from_dict({
    "sizes": {"n_bcn": 2000, "n_tgt": 3000},
    "degrees": {"distribution": "lognormal", "bcn": {"median": 15, "sigma": 0.5},
                "tgt": {"sigma": 0.6}},  # shape only: scale set by the BCN total
    "edges": {"model": "kernel", "kernel": "gaussian", "scale": 0.12},
    "background": {"model": "uniform", "ratio": 0.05},
})
graph = bipgen.generate(spec)          # seed 42 unless set otherwise
print(graph)
bipgen.plot_graph(graph)
```

Next: [installation](installation.md), the [quickstart](quickstart.md), and the
[model](model.md) with its assumptions and known limitations.

## Funding

<p>
  <a href="https://www.voluminex-project.eu/"><img src="assets/funding/voluminex.png" alt="VOLUMINEX" height="56" style="height: 56px; width: auto"></a>
  &nbsp;&nbsp;
  <img src="assets/funding/eu_funded.png" alt="Funded by the European Union" height="64" style="height: 64px; width: auto">
</p>

bipgen is developed within [VOLUMINEX](https://www.voluminex-project.eu/), *Instrument-free
3D molecular imaging with the VOLumetric UMI-Network EXplorer*. This project has received
funding from the European Innovation Council (EIC) established under the EU Horizon Europe
programme, [grant 101184647](https://cordis.europa.eu/project/id/101184647). Views and
opinions expressed are however those of the authors only and do not necessarily reflect
those of the European Union and European Innovation Council (EIC). Neither the European
Union nor the granting authority can be held responsible for them.
