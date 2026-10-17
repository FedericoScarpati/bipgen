# The model

## Positions

**Uniform.** Each point gets an isotropic direction and a radius $u^{1/d}$ with $u$
uniform in $[0, 1]$, which gives a flat density in the unit ball of dimension $d$.

**Spatially correlated density.** The domain is covered by a grid with `n_bins`
cells per axis, keeping the cells whose centre lies in the disk or ball. Each cell
receives a count of points. The counts come from Gamma weights with a given
coefficient of variation, or from counts given by the user (for instance measured
on a tissue), rescaled to the total number of points. The counts are then rearranged
on the grid by swaps accepted with the Metropolis rule [6] under a geometric
annealing schedule [5], until the spatial autocorrelation of the counts matches a
target correlogram. The autocorrelation at lag $k$ is Moran's I [4] computed over the
pairs of cells at rounded grid distance $k$; lag 1 is the 8-cell neighbourhood in 2D
and the 18-cell neighbourhood (faces and edges) in 3D.
Swaps conserve the set of counts, so only their arrangement changes. Finally, the
points of each cell are drawn uniformly inside it, kept inside the domain and thinned
at random to the exact total. The density is piecewise constant on the cells, and near
the boundary it follows the cells whose centre lies inside the domain. In 3D the
annealing has many more cells to arrange, hence the coarser default grid.

BCN and TGT are drawn together from the same density, then split at random.

## Target degrees

The BCN degrees are drawn from the chosen distribution, rounded and clipped to
$[1, n_\mathrm{tgt}]$. The TGT values give a shape that is rescaled to the BCN total,
rounded, clipped to $[1, n_\mathrm{bcn}]$ and adjusted by one unit at a time until the
two totals are equal. Degrees are assigned to nodes at random, so they are
independent of position.

## Kernel model

Each pair $(i, j)$ of a BCN and a TGT at distance $d_{ij}$ is linked independently
with probability

$$
P_{ij} = \min\left(1,\ \theta_i \, K(d_{ij}) \, \theta_j\right),
\qquad K(d) = e^{-(d/L)^2},
$$

where the node weights $\theta$ solve

$$
\sum_j \theta_i K(d_{ij}) \theta_j = k_i, \qquad \sum_i \theta_i K(d_{ij}) \theta_j = k_j ,
$$

so that expected degrees match the targets $k$. The weights are found by Sinkhorn
scaling [1, 2], alternating $\theta_i \leftarrow k_i / \sum_j K_{ij}\theta_j$ and
$\theta_j \leftarrow k_j / \sum_i K_{ij}\theta_i$ until no weight changes by more than
`tol`. The same structure appears in the doubly constrained gravity model of spatial
interaction [3] and in soft random geometric graphs with a general connection
function [7, 8], here with node weights fitted to a degree sequence. The Gaussian
kernel is the diffusion profile used in the simulations of [9]. Other kernels can be
added to the registry in `bipgen.kernels`.

## k-nearest-neighbour model

Each BCN is linked to its `k` nearest TGT. BCN degrees are exactly `k`; TGT degrees
follow from the geometry.

## Background edges

Background edges link pairs regardless of distance. Their number is `ratio` times the
number of geometric edges. Endpoints are drawn uniformly, or with probability
proportional to the target degrees (realized degrees for the kNN model). Pairs that
are already linked are redrawn, so the two kinds of edges are disjoint; each edge
records its origin.

## Reproducibility

One seed spawns independent random streams for positions, degrees, geometric edges
and background edges (`numpy.random.SeedSequence`). Sinkhorn scaling runs on one
BLAS thread, so its result does not depend on the number of cores. On a given
machine and library build the same seed gives the same graph.

## What comes from the literature

| Element | Source |
|---|---|
| Matrix scaling to prescribed margins | Sinkhorn [1], Sinkhorn and Knopp [2] |
| Weights times distance function | doubly constrained gravity model [3]; soft random geometric graphs [7, 8] |
| Gaussian diffusion kernel | simulations of [9] |
| Moran's I, correlogram | Moran [4] |
| Metropolis rule, simulated annealing | [5, 6] |

Choices of this package: Gamma cell weights, the shells of the correlogram on the
grid, the integer degree sequences and their matching rule, the clipping at 1 after
scaling, the kNN rule, the background models and the measurement of occupancies and
correlogram from points (see [Observed graphs](observed.md)).

## Known limitations

- **Clipping lowers degrees.** Sinkhorn scaling matches the targets before the
  clipping at 1. For kernels short compared with the distance between linked nodes,
  clipping removes edges: with 800 x 1000 nodes, median BCN degree 15 and uniform
  positions, the expected number of edges is 90% of the target at scale 0.15 and 99%
  at scale 0.30. `graph.info` reports `target_edges` and `expected_edges`.
- **Short kernels.** When too few partners lie within the kernel range, the weights
  keep growing and Sinkhorn scaling stops at `max_iter` with a warning; if they
  overflow, generation stops with an error. Increase `scale` or lower the degrees.
- **Memory.** The kernel matrix is dense during scaling ($4\,n_\mathrm{bcn} n_\mathrm{tgt}$
  bytes).
- **Densities are measured on 2D points only**: `bipgen.density.measure` works on
  planar positions, while generation works in 2D and 3D.
- **Densities measured on points** carry the assumptions listed in
  [Observed graphs](observed.md).

## References

1. R. Sinkhorn, A relationship between arbitrary positive matrices and doubly
   stochastic matrices, *Ann. Math. Statist.* 35, 876 (1964).
2. R. Sinkhorn and P. Knopp, Concerning nonnegative matrices and doubly stochastic
   matrices, *Pacific J. Math.* 21, 343 (1967).
3. A. G. Wilson, A statistical theory of spatial distribution models,
   *Transportation Research* 1, 253 (1967).
4. P. A. P. Moran, Notes on continuous stochastic phenomena, *Biometrika* 37, 17 (1950).
5. S. Kirkpatrick, C. D. Gelatt and M. P. Vecchi, Optimization by simulated annealing,
   *Science* 220, 671 (1983).
6. N. Metropolis, A. W. Rosenbluth, M. N. Rosenbluth, A. H. Teller and E. Teller,
   Equation of state calculations by fast computing machines, *J. Chem. Phys.* 21,
   1087 (1953).
7. C. P. Dettmann and O. Georgiou, Random geometric graphs with general connection
   functions, *Phys. Rev. E* 93, 032313 (2016).
8. M. D. Penrose, Connectivity of soft random geometric graphs, *Ann. Appl. Probab.*
   26, 986 (2016).
9. C. Hu et al., Scalable spatial transcriptomics through computational array
   reconstruction, *Nat. Biotechnol.* (2025).
