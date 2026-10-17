"""Reading and writing graphs.

Native format, one directory per graph:

* ``edges.csv``: columns ``bcn``, ``tgt``, ``origin`` (``geometric`` or ``background``);
* ``nodes.csv``: columns ``side`` (``bcn`` or ``tgt``), ``index``, ``x``, ``y`` (and ``z``
  in 3D), ``target_degree`` (empty for the kNN model);
* ``spec.yaml`` (and ``spec.npz`` when the specification holds arrays);
* ``info.json``: seed and generation diagnostics.

:func:`to_sga` writes the layout read by ``load_benchmark_graph`` of the
spatial-graph-algorithms package.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, save_npz

from .graph import ORIGINS, BipartiteGraph
from .spec import load_spec

AXES = ("x", "y", "z")


def _node_table(graph: BipartiteGraph) -> pd.DataFrame:
    parts = []
    targets = graph.target_degrees or (None, None)
    for side, pos, target in zip(("bcn", "tgt"), (graph.pos_bcn, graph.pos_tgt), targets,
                                 strict=True):
        frame = pd.DataFrame(pos, columns=list(AXES[: graph.dim]))
        frame.insert(0, "index", np.arange(len(pos)))
        frame.insert(0, "side", side)
        frame["target_degree"] = pd.array(target if target is not None else [None] * len(pos),
                                          dtype="Int64")
        parts.append(frame)
    return pd.concat(parts, ignore_index=True)


def save(graph: BipartiteGraph, path) -> Path:
    """Write ``graph`` to the directory ``path`` (created if needed)."""
    root = Path(path)
    root.mkdir(parents=True, exist_ok=True)
    graph.edges().to_csv(root / "edges.csv", index=False)
    _node_table(graph).to_csv(root / "nodes.csv", index=False)
    if graph.spec is not None:
        graph.spec.save(root / "spec.yaml")
    (root / "info.json").write_text(json.dumps({"seed": graph.seed, **graph.info}, indent=2))
    return root


def load(path) -> BipartiteGraph:
    """Read a graph written by :func:`save`."""
    root = Path(path)
    nodes = pd.read_csv(root / "nodes.csv")
    edges = pd.read_csv(root / "edges.csv")
    axes = [a for a in AXES if a in nodes.columns]
    sides = {s: nodes[nodes["side"] == s].sort_values("index") for s in ("bcn", "tgt")}
    pos = {s: frame[axes].to_numpy(dtype=float) for s, frame in sides.items()}
    shape = (len(pos["bcn"]), len(pos["tgt"]))
    mats = {}
    for origin in ORIGINS:
        sel = edges[edges["origin"] == origin]
        data = np.ones(len(sel), dtype=np.uint8)
        mats[origin] = csr_matrix((data, (sel["bcn"], sel["tgt"])), shape=shape)
    info = json.loads((root / "info.json").read_text()) if (root / "info.json").exists() else {}
    spec = load_spec(root / "spec.yaml") if (root / "spec.yaml").exists() else None
    target = None
    if sides["bcn"]["target_degree"].notna().all() and sides["tgt"]["target_degree"].notna().all():
        target = tuple(sides[s]["target_degree"].to_numpy(dtype=np.int64) for s in ("bcn", "tgt"))
    seed = int(info.get("seed", spec.seed if spec else 0))
    return BipartiteGraph(mats["geometric"], mats["background"], pos["bcn"], pos["tgt"], seed,
                          spec, info, target)


def read_edge_list(path, bcn: str = "bcn", tgt: str = "tgt", shape=None) -> csr_matrix:
    """Biadjacency matrix from a CSV edge list with integer columns ``bcn`` and ``tgt``.

    Repeated pairs are merged; the result is binary.
    """
    table = pd.read_csv(path, usecols=[bcn, tgt])
    rows, cols = table[bcn].to_numpy(), table[tgt].to_numpy()
    shape = shape or (int(rows.max()) + 1, int(cols.max()) + 1)
    adj = csr_matrix((np.ones(len(rows), dtype=np.uint8), (rows, cols)), shape=shape)
    adj.data[:] = 1
    return adj


def to_sga(graph: BipartiteGraph, path) -> Path:
    """Write the files read by ``load_benchmark_graph`` of spatial-graph-algorithms.

    Nodes are numbered BCN first (``0 .. n_bcn - 1``), then TGT. Background
    edges are marked ``is_false``. Load with ``keep_lcc=False`` semantics
    (``load_benchmark_graph`` does so) to keep this numbering. That package
    locates the two sides of a bipartite graph by position, assuming
    ``n_bcn == n // ratio`` for an integer ``ratio``; graphs with other side
    sizes load fine but its bipartite-specific routines may misplace the split.
    """
    root = Path(path)
    root.mkdir(parents=True, exist_ok=True)
    n_bcn, n = graph.n_bcn, graph.n_bcn + graph.n_tgt
    coo = graph.biadjacency.tocoo()
    rows, cols = coo.row.astype(np.int64), coo.col.astype(np.int64) + n_bcn
    sym = csr_matrix((np.ones(2 * len(rows)), (np.r_[rows, cols], np.r_[cols, rows])), shape=(n, n))
    save_npz(root / "adjacency.npz", sym)
    np.save(root / "positions.npy", np.vstack([graph.pos_bcn, graph.pos_tgt]))
    table = graph.edges()
    pd.DataFrame({
        "source": table["bcn"].to_numpy(),
        "target": table["tgt"].to_numpy() + n_bcn,
        "edge_id": np.arange(len(table)),
        "is_false": (table["origin"] == "background").to_numpy(),
    }).to_csv(root / "edge_metadata.csv", index=False)
    pd.DataFrame({"node_id": np.arange(n),
                  "side": ["bcn"] * n_bcn + ["tgt"] * graph.n_tgt}).to_csv(
        root / "node_metadata.csv", index=False)
    meta = {"generator": "bipgen", "seed": graph.seed, "n_bcn": n_bcn, "n_tgt": graph.n_tgt,
            **{k: v for k, v in graph.info.items() if k != "seed"}}
    (root / "metadata.json").write_text(json.dumps(meta, indent=2))
    return root
