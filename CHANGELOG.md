# Changelog

## Unreleased

- Package skeleton and draft specification format (default seed 42).
- Seed streams, kernel registry (Gaussian), degree sequences, uniform positions in 2D and 3D.
- Kernel model with Sinkhorn degree matching and block-wise Bernoulli sampling; k-nearest-neighbour model.
- Background edges, uniform or proportional to node weights.
- YAML specification with defaults and strict validation (unknown keys are rejected).
- `generate(spec, seed=None)`: one realization; the seed used is recorded in the graph, together with
  the target and expected numbers of edges and the Sinkhorn diagnostics.
- `BipartiteGraph` container with geometric and background edges and an edge table.
- Metropolis positions (2D): Gamma or given cell occupancies, rearranged by simulated annealing to
  match a target correlogram.
- Quick plots: graph layout (2D/3D), complementary cumulative degree distributions with targets,
  edge lengths by origin. Development notebook `notebooks/00_dev_playground.ipynb`.
- Sinkhorn runs on one BLAS thread (faster on shared machines, potentials independent of the
  thread count); overflowing potentials raise an error instead of returning an empty graph;
  the non-convergence warning reports the gap before clipping and points to the clipping loss.
- Known behaviour: probabilities are clipped at 1 after Sinkhorn scaling, so dense nodes stay below
  their target degree when the kernel is short.
