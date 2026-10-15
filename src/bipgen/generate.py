"""One realization of the ensemble described by a specification."""

from __future__ import annotations

import numpy as np
from scipy.sparse import csr_matrix

from . import edges, positions
from .background import background_edges
from .degrees import sample_degrees
from .graph import BipartiteGraph
from .rng import resolve_seed, streams
from .spec import Spec


def _positions(spec: Spec, rng: np.random.Generator):
    method = spec.positions["method"]
    if method == "uniform":
        return (
            positions.uniform(spec.n_bcn, spec.dim, rng),
            positions.uniform(spec.n_tgt, spec.dim, rng),
            {},
        )
    if method == "metropolis":
        sites, diag = positions.metropolis(spec.n_bcn + spec.n_tgt, spec.positions["density"],
                                           spec.arrays, rng)
        is_tgt = np.zeros(len(sites), dtype=bool)
        is_tgt[: spec.n_tgt] = True
        rng.shuffle(is_tgt)
        return sites[~is_tgt], sites[is_tgt], diag
    raise ValueError(f"unknown positions method {method!r}")


def _degree_params(spec: Spec, side: str) -> dict:
    params = dict(spec.degrees[side])
    if spec.degrees["distribution"] == "empirical":
        params = {"values": spec.arrays[params["key"]]}
    return params


def generate(spec: Spec, seed: int | None = None) -> BipartiteGraph:
    """Draw one graph from ``spec``.

    ``seed`` overrides the seed of the specification (default 42); the seed
    actually used is stored in ``graph.seed`` and ``graph.info["seed"]``.
    """
    seed = resolve_seed(seed, spec.seed)
    rng = streams(seed)
    pos_bcn, pos_tgt, pos_info = _positions(spec, rng["positions"])
    info = {"seed": seed, **pos_info}

    model = spec.edges["model"]
    if model == "kernel":
        k_bcn, k_tgt = sample_degrees(
            spec.degrees["distribution"], _degree_params(spec, "bcn"),
            _degree_params(spec, "tgt"), spec.n_bcn, spec.n_tgt, rng["degrees"],
        )
        name, scale = spec.edges["kernel"], float(spec.edges["scale"])
        solver = spec.edges["sinkhorn"]
        fit = edges.sinkhorn(pos_bcn, pos_tgt, k_bcn, k_tgt, name, scale,
                             tol=float(solver["tol"]), max_iter=int(solver["max_iter"]))
        if not fit.converged:
            info["warning"] = "Sinkhorn did not converge: compare expected_edges with target_edges"
        geometric, expected = edges.sample_kernel_edges(
            pos_bcn, pos_tgt, fit, name, scale, rng["edges"], return_expected=True
        )
        weights = targets = (k_bcn, k_tgt)
        info.update(target_edges=int(k_bcn.sum()), expected_edges=expected,
                    sinkhorn_iterations=fit.n_iter, sinkhorn_converged=fit.converged,
                    sinkhorn_max_error=fit.max_error)
    elif model == "knn":
        targets = None
        geometric = edges.knn_edges(pos_bcn, pos_tgt, int(spec.edges["k"]))
        weights = (np.asarray(geometric.sum(axis=1)).ravel(),
                   np.asarray(geometric.sum(axis=0)).ravel())
    else:
        raise ValueError(f"unknown edge model {model!r}")

    rows, cols = background_edges(
        geometric, spec.background["ratio"], spec.background["model"], rng["background"],
        *weights,
    )
    data = np.ones(rows.size, dtype=np.uint8)
    background = csr_matrix((data, (rows, cols)), shape=geometric.shape)
    info.update(geometric_edges=int(geometric.nnz), background_edges=int(background.nnz))
    return BipartiteGraph(geometric, background, pos_bcn, pos_tgt, seed, spec, info, targets)
