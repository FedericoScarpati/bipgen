"""Container for one realization of a spatial bipartite graph."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

ORIGINS = ("geometric", "background")


@dataclass
class BipartiteGraph:
    """A bipartite graph between side A (BCN, rows) and side B (TGT, columns).

    ``geometric`` and ``background`` are disjoint ``(n_bcn, n_tgt)`` sparse
    matrices of 0/1 entries; their sum is the full biadjacency matrix. The
    ``data`` array of these matrices is the slot for edge weights in later
    versions. ``info`` records how the realization was produced: the seed, the
    target and expected numbers of edges, solver diagnostics, and the seconds
    spent in each step (``timing``).
    ``target_degrees`` holds the ``(bcn, tgt)`` target degrees of the kernel
    model, and is ``None`` for the kNN model.
    """

    geometric: csr_matrix
    background: csr_matrix
    pos_bcn: np.ndarray
    pos_tgt: np.ndarray
    seed: int
    spec: object = None
    info: dict = field(default_factory=dict)
    target_degrees: tuple | None = None

    def __post_init__(self) -> None:
        shape = (len(self.pos_bcn), len(self.pos_tgt))
        for name in ("geometric", "background"):
            if getattr(self, name).shape != shape:
                raise ValueError(f"{name} has shape {getattr(self, name).shape}, expected {shape}")
        if self.geometric.multiply(self.background).nnz:
            raise ValueError("geometric and background edges overlap")

    @property
    def n_bcn(self) -> int:
        return len(self.pos_bcn)

    @property
    def n_tgt(self) -> int:
        return len(self.pos_tgt)

    @property
    def dim(self) -> int:
        return self.pos_bcn.shape[1]

    @property
    def biadjacency(self) -> csr_matrix:
        """All edges, ``(n_bcn, n_tgt)``."""
        return (self.geometric + self.background).tocsr()

    @property
    def n_edges(self) -> int:
        return self.geometric.nnz + self.background.nnz

    def degrees(self) -> tuple[np.ndarray, np.ndarray]:
        """Realized degrees ``(bcn, tgt)`` counting all edges."""
        adj = self.biadjacency
        return np.asarray(adj.sum(axis=1)).ravel(), np.asarray(adj.sum(axis=0)).ravel()

    def edge_lengths(self) -> pd.DataFrame:
        """Euclidean length of every edge, with its origin."""
        table = self.edges()
        diff = self.pos_bcn[table["bcn"].to_numpy()] - self.pos_tgt[table["tgt"].to_numpy()]
        table["length"] = np.linalg.norm(diff, axis=1)
        return table

    def edges(self) -> pd.DataFrame:
        """Edge table with columns ``bcn``, ``tgt`` and ``origin``."""
        parts = []
        for origin in ORIGINS:
            coo = getattr(self, origin).tocoo()
            parts.append(pd.DataFrame({"bcn": coo.row, "tgt": coo.col, "origin": origin}))
        table = pd.concat(parts, ignore_index=True)
        table["origin"] = pd.Categorical(table["origin"], categories=ORIGINS)
        return table.sort_values(["bcn", "tgt"], ignore_index=True)

    def __repr__(self) -> str:
        return (
            f"BipartiteGraph(n_bcn={self.n_bcn}, n_tgt={self.n_tgt}, dim={self.dim}, "
            f"edges={self.geometric.nnz} geometric + {self.background.nnz} background, "
            f"seed={self.seed})"
        )
