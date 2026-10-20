# Saving and loading

## Native format

`bipgen.save(graph, path)` writes one directory per graph:

| File | Content |
|---|---|
| `edges.csv` | `bcn`, `tgt`, `origin` (`geometric` or `background`) |
| `nodes.csv` | `side`, `index`, `x`, `y` (and `z` in 3D), `target_degree` (empty for the kNN model) |
| `spec.yaml`, `spec.npz` | the specification, and its arrays if any |
| `info.json` | seed and generation diagnostics |

`bipgen.load(path)` reads it back.

## Edge lists

`bipgen.read_edge_list(path, bcn=..., tgt=...)` reads a CSV file with integer node
indices into a binary biadjacency matrix; repeated pairs are merged.
