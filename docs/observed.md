# Observed graphs

## From a graph to a specification

`spec_from_graph` builds a kernel-model specification from a biadjacency matrix
(BCN as rows, TGT as columns):

```python
import bipgen

adj = bipgen.read_edge_list("edges.csv", bcn="bcn", tgt="tgt")
spec = bipgen.spec_from_graph(adj, scale=0.15)
spec.save("observed.yaml")      # writes observed.yaml and observed.npz
```

What is read off the graph:

- the number of nodes and the degree sequence of each side, after binarizing the
  matrix and dropping isolated nodes; they become an `empirical` degree section;
- if `positions` of some nodes are given, the cell occupancies and the correlogram of
  their density (see below); positions are then drawn with the Metropolis method.

What is **not** inferred, and has to be chosen: the kernel scale and the background.

## Density measured on points

`bipgen.density.measure` takes 2D points, maps their bounding box to $[-1, 1]^2$, counts
the points in the cells of the grid inside the disk, and computes Moran's I of the
counts at each lag. When the points are only part of the nodes, counting noise
inflates the variance of the counts and attenuates the correlation; the corrected
correlogram multiplies it by $\mathrm{var}/(\mathrm{var} - \mathrm{mean})$, which
treats the counts as Poisson draws around the underlying density.

Assumptions and limitations:

- the known positions are a random sample of the nodes, and BCN and TGT share the
  same density, since both are drawn from it;
- the correction tends to overestimate the correlation when the known points are a
  large fraction of the nodes, and becomes unstable when cells hold few points.

## Comparing an observed graph with an ensemble

```python
ensemble = [bipgen.generate(spec, seed=s) for s in range(20)]
table = bipgen.compare(adj, ensemble)
bipgen.plot_comparison(table)
bipgen.degree_ks(adj, ensemble)
```

`statistics` computes, on the observed graph and on each realization alike: numbers of
nodes and edges, mean, median, 90th percentile and maximum degree of each side, and
the fraction of nodes in the largest connected component. Only nodes with at least
one edge are counted. `compare` reports the observed value, the ensemble mean and
standard deviation, the z-score and the percentile of the observed value in the
ensemble: a Monte Carlo goodness-of-fit check. `degree_ks` compares the degree
distributions with the two-sample Kolmogorov-Smirnov distance, next to the distance
between two realizations, which is what sampling alone produces.

`compare` also accepts a stored summary instead of a graph, such as
`bipgen.presets.observed(name)`.
