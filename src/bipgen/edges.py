"""Geometric edges between side A (BCN) and side B (TGT).

Two models:

* **kernel**: degree-matched null model. Each pair has probability
  ``P_ij = clip(theta_i K(d_ij) theta_j, 0, 1)``, where ``K`` is a distance
  kernel and the potentials ``theta`` are found by Sinkhorn scaling so that the
  expected degrees ``sum_j theta_i K_ij theta_j`` match the targets. Edges are
  independent Bernoulli draws.
* **knn**: each BCN links to its ``k`` nearest TGT.

Degrees and clipping: Sinkhorn scaling matches the target degrees for the
unclipped product ``theta_i K_ij theta_j``. Clipping at 1 then lowers the
expected degree of nodes whose nearby pairs would exceed probability 1, so the
realized graph has fewer edges than the target total. The deficit grows when
the kernel is short compared with the typical distance between linked nodes,
and vanishes for long kernels. For example, with 800 x 1000 nodes, median BCN
degree 15 and uniform positions in the unit disk, the expected number of edges
is 90% of the target at scale 0.15 and 99% at scale 0.30. The expected number
of edges after clipping is the sum of the clipped matrix; sampled graphs
fluctuate around it.

The kernel matrix is dense during Sinkhorn scaling: memory is about
``4 * n_bcn * n_tgt`` bytes. Sampling works on row blocks and never holds the
full probability matrix. Sinkhorn runs its matrix-vector products on one BLAS
thread: they are limited by memory bandwidth, more threads barely help, and a
fixed thread count keeps the potentials independent of the machine.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
from scipy.sparse import csr_matrix, vstack
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist
from threadpoolctl import threadpool_limits

from . import kernels

BLOCK = 256


@dataclass
class SinkhornResult:
    """Potentials of the kernel model and convergence information."""

    theta_a: np.ndarray
    theta_b: np.ndarray
    n_iter: int
    converged: bool
    max_error: float  # largest relative gap between expected and target degree, before clipping


def kernel_rows(pos_a, pos_b, kernel: str, scale: float, start: int, stop: int) -> np.ndarray:
    """Kernel values for rows ``start:stop`` of side A against all of side B (float32)."""
    dist = cdist(pos_a[start:stop], pos_b, metric="euclidean").astype(np.float32)
    return kernels.get(kernel)(dist, scale).astype(np.float32, copy=False)


def kernel_matrix(pos_a, pos_b, kernel: str, scale: float) -> np.ndarray:
    """Dense kernel matrix ``(n_a, n_b)`` in float32, built by row blocks."""
    K = np.empty((len(pos_a), len(pos_b)), dtype=np.float32)
    for start in range(0, len(pos_a), BLOCK):
        stop = min(start + BLOCK, len(pos_a))
        K[start:stop] = kernel_rows(pos_a, pos_b, kernel, scale, start, stop)
    return K


def _scale(k: np.ndarray, field: np.ndarray) -> np.ndarray:
    """``k / field``; 0 for zero-degree nodes, inf where a positive degree meets no partner."""
    out = np.full_like(k, np.inf)
    np.divide(k, field, out=out, where=field > 0)
    out[k == 0] = 0.0
    return out


def sinkhorn(
    pos_a,
    pos_b,
    k_a,
    k_b,
    kernel: str = "gaussian",
    scale: float = 0.1,
    tol: float = 1e-3,
    max_iter: int = 2000,
) -> SinkhornResult:
    """Potentials such that ``theta_a K theta_b`` has row sums ``k_a`` and column sums ``k_b``.

    Alternating updates ``theta_a = k_a / (K theta_b)``, ``theta_b = k_b / (K^T theta_a)``
    in float32, stopped when the largest change of a potential falls below ``tol``.
    The two degree totals must be equal. Clipping ``P`` at 1 afterwards lowers the
    degrees of dense nodes below their target (see the module docstring).
    """
    k_a = np.asarray(k_a, dtype=np.float32)
    k_b = np.asarray(k_b, dtype=np.float32)
    if not np.isclose(k_a.sum(), k_b.sum(), rtol=1e-3):
        raise ValueError(f"degree totals differ: {k_a.sum():.1f} vs {k_b.sum():.1f}")
    K = kernel_matrix(pos_a, pos_b, kernel, scale)
    theta_a = np.ones(len(k_a), dtype=np.float32)
    theta_b = np.ones(len(k_b), dtype=np.float32)
    converged = False
    n_iter = 0
    with threadpool_limits(limits=1, user_api="blas"), np.errstate(over="ignore", invalid="ignore"):
        while n_iter < max_iter:
            n_iter += 1
            new_a = _scale(k_a, K @ theta_b)
            new_b = _scale(k_b, K.T @ new_a)
            if not (np.isfinite(new_a).all() and np.isfinite(new_b).all()):
                raise ValueError(
                    f"Sinkhorn potentials overflowed after {n_iter} iterations: the target "
                    "degrees cannot be reached with this kernel, because too few partners lie "
                    "within its range. Increase edges.scale or lower the degrees."
                )
            change = max(np.max(np.abs(new_a - theta_a)), np.max(np.abs(new_b - theta_b)))
            theta_a, theta_b = new_a, new_b
            if change < tol:
                converged = True
                break
        exp_a = theta_a * (K @ theta_b)
        exp_b = theta_b * (K.T @ theta_a)
    with np.errstate(divide="ignore", invalid="ignore"):
        gaps = np.concatenate([np.abs(exp_a / k_a - 1), np.abs(exp_b / k_b - 1)])
    max_error = float(np.nanmax(np.where(np.isfinite(gaps), gaps, 1.0)))
    if not converged:
        warnings.warn(
            f"Sinkhorn did not converge in {max_iter} iterations (largest change of a potential "
            f"{change:.2e}); before clipping, expected degrees are within {max_error:.1%} of the "
            "targets. Potentials that keep growing push probabilities above 1, and clipping then "
            "removes edges: compare expected_edges with target_edges. The kernel is probably too "
            "short for these degrees: increase edges.scale or lower the degrees.",
            stacklevel=2,
        )
    return SinkhornResult(theta_a, theta_b, n_iter, converged, max_error)


def probability_rows(pos_a, pos_b, fit: SinkhornResult, kernel, scale, start, stop) -> np.ndarray:
    """Rows ``start:stop`` of ``P = clip(theta_a K theta_b, 0, 1)`` (float32)."""
    P = fit.theta_a[start:stop, None] * kernel_rows(pos_a, pos_b, kernel, scale, start, stop)
    P *= fit.theta_b[None, :]
    np.clip(P, 0.0, 1.0, out=P)
    return P


def sample_kernel_edges(
    pos_a, pos_b, fit: SinkhornResult, kernel, scale, rng, return_expected: bool = False
):
    """Independent Bernoulli draws of every pair, row block by row block.

    Uniform numbers are drawn in row order, so the result does not depend on
    the block size. With ``return_expected`` the expected number of edges, the
    sum of the clipped probabilities, is returned as well.
    """
    blocks, expected = [], 0.0
    for start in range(0, len(pos_a), BLOCK):
        stop = min(start + BLOCK, len(pos_a))
        P = probability_rows(pos_a, pos_b, fit, kernel, scale, start, stop)
        expected += float(P.sum(dtype=np.float64))
        blocks.append(csr_matrix(rng.random(P.shape, dtype=np.float32) < P, dtype=np.uint8))
    graph = vstack(blocks, format="csr")
    return (graph, expected) if return_expected else graph


def knn_edges(pos_a, pos_b, k: int) -> csr_matrix:
    """Each node of side A linked to its ``k`` nearest nodes of side B."""
    n_a, n_b = len(pos_a), len(pos_b)
    if not 1 <= k <= n_b:
        raise ValueError(f"k must be in [1, n_tgt={n_b}], got {k}")
    _, idx = cKDTree(pos_b).query(pos_a, k=k)
    idx = np.asarray(idx).reshape(n_a, k)
    rows = np.repeat(np.arange(n_a), k)
    data = np.ones(n_a * k, dtype=np.uint8)
    return csr_matrix((data, (rows, idx.ravel())), shape=(n_a, n_b))
