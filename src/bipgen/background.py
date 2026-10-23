"""Background edges: pairs linked independently of their distance.

The number of background edges is ``ratio`` times the number of geometric
edges. Endpoints are drawn uniformly (``uniform``) or with probability
proportional to node weights (``configuration``): in :func:`bipgen.generate`, the
geometric degrees, so that every node gets on average the same share of background
edges. Pairs that are already linked, and repeated pairs, are rejected.
"""

from __future__ import annotations

import warnings

import numpy as np
from scipy.sparse import csr_matrix

MODELS = ("none", "uniform", "configuration")


def background_edges(
    geometric: csr_matrix,
    ratio: float,
    model: str,
    rng: np.random.Generator,
    weights_a=None,
    weights_b=None,
    max_rounds: int = 50,
) -> tuple[np.ndarray, np.ndarray]:
    """Row and column indices of the background edges.

    Returns two int64 arrays of equal length, disjoint from the edges of
    ``geometric`` and free of repeats.
    """
    if model not in MODELS:
        raise ValueError(f"unknown background model {model!r}; available: {MODELS}")
    if ratio < 0:
        raise ValueError(f"background ratio must be >= 0, got {ratio}")
    empty = np.empty(0, dtype=np.int64)
    if model == "none" or ratio == 0:
        return empty, empty
    n_a, n_b = geometric.shape
    n_wanted = max(1, int(ratio * geometric.nnz))
    free = n_a * n_b - geometric.nnz
    if n_wanted > free:
        raise ValueError(f"{n_wanted} background edges requested but only {free} free pairs")

    if model == "uniform":
        p_a = p_b = None
    else:
        if weights_a is None or weights_b is None:
            raise ValueError("the configuration model needs node weights for both sides")
        p_a = np.asarray(weights_a, dtype=np.float64)
        p_b = np.asarray(weights_b, dtype=np.float64)
        if p_a.sum() <= 0 or p_b.sum() <= 0:
            raise ValueError("the configuration model needs positive weights on both sides")
        p_a, p_b = p_a / p_a.sum(), p_b / p_b.sum()

    coo = geometric.tocoo()
    taken = np.sort(coo.row.astype(np.int64) * n_b + coo.col)
    chosen = empty
    for _ in range(max_rounds):
        missing = n_wanted - chosen.size
        if missing == 0:
            break
        size = 2 * missing + 16
        rows = rng.choice(n_a, size=size, p=p_a)
        cols = rng.choice(n_b, size=size, p=p_b)
        keys = rows.astype(np.int64) * n_b + cols
        pos = np.searchsorted(taken, keys)
        pos[pos == taken.size] = 0
        keys = keys[taken[pos] != keys] if taken.size else keys
        keys = keys[~np.isin(keys, chosen)]
        _, first = np.unique(keys, return_index=True)
        chosen = np.concatenate([chosen, keys[np.sort(first)][:missing]])
    if chosen.size < n_wanted:
        warnings.warn(
            f"only {chosen.size} of {n_wanted} background edges placed (graph too dense?)",
            stacklevel=2,
        )
    return chosen // n_b, chosen % n_b
