# Changelog

## Unreleased (first public version, 0.1.0)

- **Specification**: one YAML file (or dictionary) per ensemble, with defaults and strict
  validation; default seed 42, overridable when generating; the seed used is recorded in
  every graph.
- **Positions**: uniform in the unit disk or ball; spatially correlated densities in 2D and 3D,
  with cell counts arranged by Metropolis annealing to match a target correlogram (Moran's I
  at several lags).
- **Degrees**: lognormal, constant or empirical target sequences, integer, with equal totals
  on the two sides.
- **Geometric edges**: Gaussian kernel with node weights fitted by Sinkhorn scaling to the
  target degrees, sampled block by block; or each BCN linked to its k nearest TGT. Kernels
  live in a registry.
- **Background edges**: uniform or proportional to node degrees, in a given ratio to the
  geometric edges; each edge records its origin.
- **Reproducibility**: independent random streams spawned from one seed; Sinkhorn scaling on
  one BLAS thread, so results do not depend on the number of cores.
- **Observed graphs**: specification from a biadjacency matrix (sizes, degree sequences,
  density of known positions); summary statistics and Monte Carlo comparison of an observed
  graph with an ensemble; Kolmogorov-Smirnov distances of degree distributions.
- **Input and output**: native directory format, CSV edge lists, export for the
  spatial-graph-algorithms package.
- **Descriptors**: registry with node degree and edge degree sum.
- **Presets**: `c58_4`, `c58_5`, `c58_16`, built from published Slide-recon arrays.
- **Figures**: layout in 2D and 3D, degree distributions, edge lengths, comparison with an
  ensemble.
- **Documentation**: user guide, model with its literature sources, two tutorials.

Known limitations are listed in the documentation (*The model* and *Observed graphs*):
clipping of probabilities at 1 lowers the degrees of dense nodes for short kernels; very
short kernels prevent Sinkhorn scaling from converging; the kernel matrix is dense while it
is fitted; the Poisson correction of a correlogram measured on points tends to overestimate
it when the known points are a large fraction of the nodes.
