"""A specification built from an existing graph.

Only quantities that can be read off the graph are taken from it: the sizes,
the degree sequences and, when positions of some nodes are known, the cell
occupancies and the correlogram of their density. The kernel scale and the
background are not inferred: they are chosen by the user.
"""

from __future__ import annotations

import warnings

import numpy as np
from scipy.sparse import csr_matrix

from . import density
from .spec import Spec


def binarize(biadjacency) -> csr_matrix:
    """0/1 copy of a (possibly weighted) biadjacency matrix."""
    adj = csr_matrix(biadjacency, copy=True)
    adj.eliminate_zeros()
    adj.data = np.ones_like(adj.data, dtype=np.uint8)
    return adj.astype(np.uint8)


def spec_from_graph(
    biadjacency,
    *,
    scale: float,
    positions=None,
    background: dict | None = None,
    n_bins: int = 20,
    n_lags: int = 2,
    sweeps: int = 400,
    seed: int = 42,
) -> Spec:
    """Kernel-model specification with the sizes and degrees of an observed graph.

    Parameters
    ----------
    biadjacency
        ``(n_bcn, n_tgt)`` sparse matrix; weights are ignored and isolated
        nodes are dropped.
    scale
        Kernel length, in units of the domain radius. Not fitted: choose it.
    positions
        Optional ``(m, 2)`` array of known positions of some nodes (rows with
        NaN are ignored). When given, positions are drawn with the Metropolis
        method, with the cell occupancies (``n_bins`` cells per axis) and the
        correlogram (lags 1 to ``n_lags``, corrected for counting noise)
        measured on these points; otherwise uniformly.
    background
        Background section of the specification; default no background.
    """
    adj = binarize(biadjacency)
    deg_bcn = np.asarray(adj.sum(axis=1)).ravel()
    deg_tgt = np.asarray(adj.sum(axis=0)).ravel()
    deg_bcn, deg_tgt = deg_bcn[deg_bcn > 0], deg_tgt[deg_tgt > 0]
    arrays = {"degrees_bcn": deg_bcn.astype(np.int64), "degrees_tgt": deg_tgt.astype(np.int64)}
    positions_section = {"method": "uniform"}
    if positions is not None:
        m = density.measure(positions, n_bins=n_bins, n_lags=n_lags)
        rho = m["correlogram"]
        if not all(np.isfinite(rho)) or max(abs(r) for r in rho) >= 1:
            warnings.warn(
                f"corrected correlogram {np.round(rho, 3).tolist()} is not usable; "
                f"using the raw one {np.round(m['correlogram_raw'], 3).tolist()}",
                stacklevel=2,
            )
            rho = m["correlogram_raw"]
        arrays["occupancy"] = m["occupancy"]
        positions_section = {"method": "metropolis", "density": {
            "n_bins": n_bins, "occupancy": {"source": "file", "key": "occupancy"},
            "correlogram": [round(float(r), 4) for r in rho], "sweeps": sweeps}}
    return Spec(
        sizes={"n_bcn": int(deg_bcn.size), "n_tgt": int(deg_tgt.size)},
        degrees={"distribution": "empirical", "bcn": {"key": "degrees_bcn"},
                 "tgt": {"key": "degrees_tgt"}},
        edges={"model": "kernel", "kernel": "gaussian", "scale": float(scale)},
        positions=positions_section,
        background=background or {"model": "none", "ratio": 0.0},
        seed=seed,
        arrays=arrays,
    )
