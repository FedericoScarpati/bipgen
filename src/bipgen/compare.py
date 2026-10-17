"""An observed graph against an ensemble of generated ones.

Every statistic is computed the same way on the observed graph and on each
realization, counting only nodes with at least one edge. The table reports the
observed value, the ensemble mean and standard deviation, the z-score and the
percentile of the observed value in the ensemble (a Monte Carlo
goodness-of-fit check). Degree distributions are compared with the
two-sample Kolmogorov-Smirnov distance, calibrated against the distance
between two realizations.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.sparse import bmat, csr_matrix
from scipy.sparse.csgraph import connected_components
from scipy.stats import ks_2samp

from .fit import binarize
from .graph import BipartiteGraph


def _adjacency(obj) -> csr_matrix:
    return obj.biadjacency if isinstance(obj, BipartiteGraph) else binarize(obj)


def _degrees(adj: csr_matrix) -> tuple[np.ndarray, np.ndarray]:
    deg_bcn = np.asarray(adj.sum(axis=1)).ravel()
    deg_tgt = np.asarray(adj.sum(axis=0)).ravel()
    return deg_bcn[deg_bcn > 0], deg_tgt[deg_tgt > 0]


def statistics(obj) -> dict:
    """Summary statistics of a graph or a biadjacency matrix."""
    adj = _adjacency(obj)
    deg_bcn, deg_tgt = _degrees(adj)
    out = {"n_bcn": deg_bcn.size, "n_tgt": deg_tgt.size, "n_edges": int(adj.nnz)}
    for side, deg in (("bcn", deg_bcn), ("tgt", deg_tgt)):
        out[f"{side}_degree_mean"] = float(deg.mean())
        out[f"{side}_degree_median"] = float(np.median(deg))
        out[f"{side}_degree_p90"] = float(np.percentile(deg, 90))
        out[f"{side}_degree_max"] = int(deg.max())
    full = bmat([[None, adj], [adj.T, None]], format="csr")
    _, labels = connected_components(full, directed=False)
    active = np.r_[np.asarray(adj.sum(axis=1)).ravel(), np.asarray(adj.sum(axis=0)).ravel()] > 0
    sizes = np.bincount(labels[active])
    out["giant_component_fraction"] = float(sizes.max() / active.sum())
    return out


def _observed(real):
    """Statistics and degree sequences of the observed graph (or of a stored summary)."""
    if isinstance(real, dict):
        degrees = real.get("degrees")
        return real["statistics"], degrees
    adj = _adjacency(real)
    return statistics(adj), _degrees(adj)


def compare(real, ensemble) -> pd.DataFrame:
    """Observed statistics against the ensemble.

    ``real`` is a :class:`BipartiteGraph`, a biadjacency matrix, or a
    dictionary with keys ``statistics`` (and optionally ``degrees``), as
    returned by :func:`bipgen.presets.observed`.
    """
    observed, _ = _observed(real)
    rows = [statistics(g) for g in ensemble]
    table = []
    for name, value in observed.items():
        values = np.array([r[name] for r in rows], dtype=float)
        mean, sd = values.mean(), values.std(ddof=1) if len(values) > 1 else float("nan")
        z = (value - mean) / sd if sd and np.isfinite(sd) and sd > 0 else float("nan")
        percentile = float(np.mean(values < value) + 0.5 * np.mean(values == value))
        table.append({"statistic": name, "observed": float(value), "mean": mean, "sd": sd,
                      "z": z, "percentile": percentile})
    return pd.DataFrame(table).set_index("statistic")


def degree_ks(real, ensemble) -> pd.DataFrame:
    """Kolmogorov-Smirnov distance of the degree distributions, per side.

    ``to_observed`` is the distance between the observed degrees and those of
    each realization; ``between_realizations`` the distance between consecutive
    realizations, which sets the scale of distances due to sampling alone.
    """
    _, observed = _observed(real)
    if observed is None:
        raise ValueError("the observed summary has no degree sequences")
    sequences = [_degrees(g.biadjacency) for g in ensemble]
    out = []
    for i, side in enumerate(("bcn", "tgt")):
        to_obs = [ks_2samp(observed[i], s[i]).statistic for s in sequences]
        between = [ks_2samp(a[i], b[i]).statistic for a, b in zip(sequences, sequences[1:],
                                                                  strict=False)]
        out.append({"side": side, "to_observed_mean": np.mean(to_obs),
                    "to_observed_sd": np.std(to_obs, ddof=1) if len(to_obs) > 1 else np.nan,
                    "between_realizations_mean": np.mean(between) if between else np.nan,
                    "between_realizations_sd": np.std(between, ddof=1) if len(between) > 1
                    else np.nan})
    return pd.DataFrame(out).set_index("side")
