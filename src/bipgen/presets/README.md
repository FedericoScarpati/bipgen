# Presets

Specifications built from published datasets with `bipgen.spec_from_graph`.

| Preset | Source dataset |
|---|---|
| `c58_4`, `c58_5`, `c58_16` | Slide-recon arrays c58_4, c58_5, c58_16 (Hu et al., *Nature Biotechnology* 2025; Single Cell Portal SCP2577) |
| `slidetags_tonsil` | Slide-tags human tonsil (Russell et al., *Nature* 2024; Single Cell Portal SCP2169), cell-bead network as used in Dahlberg et al., *Nature Communications* 16, 9652 (2025) |

The code of bipgen is distributed under the package license. The preset data are derived
from the datasets above and follow the terms of their sources: the Single Cell Portal
terms of service for the Slide-recon arrays and the Slide-tags data (research use), and
CC BY 4.0 for the cell-bead network of Dahlberg et al. Please cite the source papers when
using a preset.

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
| `slidetags_tonsil` | 0.998 |

`c58_16` and `slidetags_tonsil` deserve care. In `c58_16` BCN and TGT appear to be
distributed differently in space; in `slidetags_tonsil` the two sides are cell nuclei and
beads, which are distributed differently too. The presets draw both sides from one shared
density.
