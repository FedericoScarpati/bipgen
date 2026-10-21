# Presets

Specifications built from the published Slide-recon datasets (Hu et al., *Nature
Biotechnology* 2025; Single Cell Portal SCP2577) with `bipgen.spec_from_graph`.
The presets are derived from the public Single Cell Portal study SCP2577 and are
provided for research use, under the portal's terms of service; please cite Hu et al.
(2025) when using them.

| Preset | Source dataset |
|---|---|
| `c58_4`, `c58_5`, `c58_16` | Slide-recon arrays c58_4, c58_5, c58_16 |

What is taken from the data:

- sizes and degree sequences of the observed graph, with counts binarized and
  isolated nodes dropped (`degrees_bcn`, `degrees_tgt` in the `.npz`);
- cell occupancies and correlogram (lags 1 and 2) of the known BCN positions on
  a 20 x 20 grid, with the Poisson correction for counting noise (see Observed
  graphs in the documentation); these
  drive the Metropolis positions. Only part of the BCN have a known position and
  no TGT does: the presets assume that known positions are a random sample of the
  nodes and that BCN and TGT share the same density.

What is chosen by hand, not inferred from the data:

- the Gaussian kernel and its scale, 0.15 domain radii;
- no background edges.

`<name>_observed.json` holds the summary statistics of the observed graph, so that
`bipgen.compare(bipgen.presets.observed(name), ensemble)` works without the raw data.

With scale 0.15, clipping probabilities at 1 lowers the expected number of edges below the
observed one (seed 42, one realization each):

| Preset | Expected edges / observed edges |
|---|---|
| `c58_4` | 0.995 |
| `c58_5` | 0.892 |
| `c58_16` | 0.964 |

`c58_16` deserves care: in that array BCN and TGT appear to be distributed differently in
space, while the preset draws both sides from one shared density.
