# Changelog

## Unreleased

- Package skeleton and draft specification format (default seed 42).
- Seed streams, kernel registry (Gaussian), degree sequences, uniform positions in 2D and 3D.
- Kernel model with Sinkhorn degree matching and block-wise Bernoulli sampling; k-nearest-neighbour model.
- Background edges, uniform or proportional to node weights.
- Known behaviour: probabilities are clipped at 1 after Sinkhorn scaling, so dense nodes stay below
  their target degree when the kernel is short.
