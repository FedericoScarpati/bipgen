# The model

This page describes how bipgen builds one graph, in four steps: node positions,
target degrees, geometric edges and background edges. For each step, the table gives
the [specification](specification.md) section that sets it and the attributes of the
generated graph that hold its result; the text names each key where it acts. The last
sections cover reproducibility and known limitations.

| Step | Set by | Result |
|---|---|---|
| [Positions](#positions) | `dim`, `sizes`, `positions` | `graph.pos_bcn`, `graph.pos_tgt` |
| [Target degrees](#target-degrees) | `degrees` | `graph.target_degrees` |
| [Geometric edges](#geometric-edges) | `edges` | `graph.geometric` |
| [Background edges](#background-edges) | `background` | `graph.background` |
| [Reproducibility](#reproducibility) | `seed` | `graph.seed` |

The graph also keeps the specification it was built from, in `graph.spec`, and the
diagnostics of each step, such as the correlogram reached or the Sinkhorn iterations,
in `graph.info`, together with the seconds each step took (`graph.info["timing"]`).

## Positions

`sizes.n_bcn` BCN and `sizes.n_tgt` TGT are placed in the domain set by `dim`: the
unit disk (`dim: 2`) or the unit ball (`dim: 3`). Lengths are in units of the domain
radius. `positions.method` chooses one of three methods.

- **Uniform** (`uniform`, the default). Points are spread uniformly over the domain.
- **Spatially correlated density** (`metropolis`). Nodes are spread over a grid of
  cells, with a chosen variability of the counts between cells and a chosen
  correlation between nearby cells. BCN and TGT are drawn together from this density,
  then split at random.

    1. **Grid**: `density.n_bins` cells per axis, keeping the cells whose centre lies
       in the domain; 20 by default in 2D, 12 in 3D, where there are many more cells.
    2. **Counts**: one count per cell, drawn from `density.occupancy`, rescaled so
       that the counts add up to the number of nodes. With `{source: gamma, cv: ...}`
       the counts follow a Gamma distribution with coefficient of variation `cv`;
       with `{source: file, key: ...}` they are resampled from the array `key` of the
       arrays file, for instance measured on another graph (see
       [Observed graphs](observed.md)).
    3. **Arrangement**: simulated annealing swaps the counts of two cells at a time,
       to bring the correlogram $\rho_k$ towards the target $\rho_k^\ast$ given by
       `density.correlogram`.
        - $\rho_k$ is the spatial correlation between nearby cells (_Moran's I_ [1]),
          computed over the pairs of cells $k$ cells apart (distance between centres,
          rounded): lag 1 is the 8 surrounding cells in 2D, the 18 sharing a face or
          an edge in 3D.
        - A swap that changes the error $E = \sum_k (\rho_k - \rho_k^\ast)^2$ by
          $\Delta E$ is accepted with probability $\min(1, e^{-\Delta E / T})$
          (Metropolis rule); the temperature $T$ decreases geometrically over
          `density.sweeps` sweeps, of one proposed swap per cell.
        - Swaps keep the set of counts and change only their arrangement. The
          correlogram reached is stored in `graph.info["correlogram"]`.
    4. **Points**: drawn uniformly inside each cell; points outside the domain are
       discarded, then random points are removed down to the exact total. The density
       is constant within each cell.

- **Given positions** (`file`). Positions are read from two arrays of the arrays
  file, named by `positions.bcn` and `positions.tgt`, of shape `(n_bcn, dim)` and
  `(n_tgt, dim)`; from Python, `arrays` can also be a dictionary of arrays. Positions
  are used as given, in the same units as `edges.scale`.

## Target degrees

Target degrees are used only by the kernel model, and are stored in
`graph.target_degrees`. `degrees.distribution` sets the distribution, and
`degrees.bcn` and `degrees.tgt` its parameters on each side:

- `lognormal`: median `median` and standard deviation of the logarithm `sigma`; the
  TGT side takes only `sigma`;
- `constant`: every BCN gets `value`;
- `empirical`: values taken from the array `key` of the arrays file, resampled with
  replacement when its length differs from the number of nodes.

The BCN degrees are rounded and clipped to $[1, n_\mathrm{tgt}]$. The TGT values give
only a shape: they are rescaled to the BCN total, rounded, clipped to
$[1, n_\mathrm{bcn}]$ and adjusted by one unit at a time until the two totals are
equal. Degrees are assigned to nodes at random, so they are independent of position.

## Geometric edges

Geometric edges depend on the distance $d_{ij}$ between a BCN $i$ and a TGT $j$, and
are stored in `graph.geometric`. Two models are available, set by `edges.model`:

- `kernel`: each pair is linked at random, with probability
  $P_{ij} = \min(1, \theta_i K(d_{ij}) \theta_j)$: the kernel $K$ decays with
  distance, and the weights $\theta$ of the nodes are fitted so that the expected
  degrees match the targets;
- `knn`: each BCN is linked to its `k` nearest TGT, by distance alone.

### Kernel model

Each pair $(i, j)$ is linked independently with probability

$$
P_{ij} = \min\left(1,\ \theta_i \, K(d_{ij}) \, \theta_j\right),
\qquad K(d) = e^{-(d/L)^2},
$$

where $K$ is the `gaussian` kernel (`edges.kernel`), $L$ is `edges.scale`, and the
node weights $\theta$ solve

$$
\sum_j \theta_i K(d_{ij}) \theta_j = k_i, \qquad \sum_i \theta_i K(d_{ij}) \theta_j = k_j ,
$$

so that expected degrees match the targets $k$ (before the clipping at 1, see
[Known limitations](#known-limitations)). The weights are found by Sinkhorn scaling [2],
alternating $\theta_i \leftarrow k_i / \sum_j K_{ij}\theta_j$ and
$\theta_j \leftarrow k_j / \sum_i K_{ij}\theta_i$ until no weight changes by more than
`edges.sinkhorn.tol`, for at most `edges.sinkhorn.max_iter` iterations. `graph.info`
records the iterations (`sinkhorn_iterations`), whether they converged
(`sinkhorn_converged`) and the largest relative gap between expected and target
degrees (`sinkhorn_max_error`).

### k-nearest-neighbour model

Each BCN is linked to its `edges.k` nearest TGT. BCN degrees are exactly `k`; TGT
degrees follow from the geometry, and the `degrees` section is not used.

## Background edges

Background edges link pairs regardless of distance, and are stored in
`graph.background`. Their number is `background.ratio` times the number of geometric
edges. `background.model` sets how the endpoints are drawn:

- `none` (the default): no background edges;
- `uniform`: every pair equally likely;
- `configuration`: pair $(i, j)$ drawn with probability proportional to the
    geometric degrees of its nodes,

    $$
    \Pr(i, j) \propto g_i\, g_j, \qquad g_i = \text{number of geometric edges of } i,
    $$

    so that every node gets on average the same share of background edges,
    `ratio / (1 + ratio)`. Nodes without geometric edges get none.

Pairs that are already linked are redrawn, so the two kinds of edges are disjoint; the
`origin` column of `graph.edges()` tells them apart.

## Reproducibility

One seed spawns independent random streams for positions, degrees, geometric edges
and background edges (`numpy.random.SeedSequence`). The seed is `seed` in the
specification (default 42), `generate(spec, seed=...)` overrides it, and `graph.seed`
records the one used. Sinkhorn scaling runs on one
BLAS thread, so its result does not depend on the number of cores. On a given
machine and library build the same seed gives the same graph.

## Known limitations

- **Clipping lowers degrees.** Sinkhorn scaling matches the targets before the
  clipping at 1. For kernels short compared with the distance between linked nodes,
  clipping removes edges: with 800 x 1000 nodes, median BCN degree 15 and uniform
  positions, the expected number of edges is 90% of the target at scale 0.15 and 99%
  at scale 0.30. `graph.info` reports `target_edges`, the target number of
  edges, and `expected_edges`, the expected number after clipping.
- **Short kernels.** When too few partners lie within the kernel range, the weights
  keep growing and Sinkhorn scaling stops at `edges.sinkhorn.max_iter` with a warning,
  also kept in `graph.info["warning"]`; if they overflow, generation stops with an
  error. Increase `edges.scale` or lower the degrees.
- **Memory.** The kernel matrix is dense during scaling ($4\,n_\mathrm{bcn} n_\mathrm{tgt}$
  bytes).
- **Densities are measured on 2D points only**: `spec_from_graph` accepts planar
  positions, while generation works in 2D and 3D. Measurement in 3D is
  planned for a future version.
- **Densities measured on points** carry the assumptions listed in
  [Observed graphs](observed.md).

## References

1. P. A. P. Moran, Notes on continuous stochastic phenomena, *Biometrika* 37, 17 (1950).
2. R. Sinkhorn, A relationship between arbitrary positive matrices and doubly
   stochastic matrices, *Ann. Math. Statist.* 35, 876 (1964).
