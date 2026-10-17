# Saving and exporting

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

## Export for spatial-graph-algorithms

`bipgen.to_sga(graph, path)` writes the files read by `load_benchmark_graph` of the
spatial-graph-algorithms package: a symmetric adjacency matrix with BCN numbered
first, positions, and edge metadata in which background edges are marked
`is_false`. That package locates the two sides of a bipartite graph by position,
assuming `n_bcn == n // ratio` for an integer `ratio`; with other side sizes the
graph loads, but its bipartite-specific routines may misplace the split.
