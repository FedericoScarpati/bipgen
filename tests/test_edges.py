from pathlib import Path

import numpy as np
import pytest

from bipgen import edges, positions
from bipgen.degrees import sample_degrees

REFERENCE = Path(__file__).parent / "data" / "reference_gaussian.npz"


@pytest.fixture
def small():
    g = np.random.default_rng(3)
    pa, pb = positions.uniform(300, 2, g), positions.uniform(400, 2, g)
    k_a, k_b = sample_degrees("lognormal", {"median": 8, "sigma": 0.4}, {"sigma": 0.4}, 300, 400, g)
    return pa, pb, k_a, k_b


def test_sinkhorn_matches_target_degrees(small):
    pa, pb, k_a, k_b = small
    fit = edges.sinkhorn(pa, pb, k_a, k_b, "gaussian", 0.4, tol=1e-6)
    raw = fit.theta_a[:, None] * edges.kernel_matrix(pa, pb, "gaussian", 0.4) * fit.theta_b
    assert fit.converged
    assert np.allclose(raw.sum(axis=1), k_a, rtol=1e-3)
    assert np.allclose(raw.sum(axis=0), k_b, rtol=1e-3)


def test_sinkhorn_rejects_unequal_totals(small):
    pa, pb, k_a, k_b = small
    with pytest.raises(ValueError):
        edges.sinkhorn(pa, pb, k_a, k_b * 2, "gaussian", 0.4)


def test_sampling_is_reproducible_and_block_independent(small, monkeypatch):
    pa, pb, k_a, k_b = small
    fit = edges.sinkhorn(pa, pb, k_a, k_b, "gaussian", 0.3)
    a = edges.sample_kernel_edges(pa, pb, fit, "gaussian", 0.3, np.random.default_rng(1))
    monkeypatch.setattr(edges, "BLOCK", 7)
    b = edges.sample_kernel_edges(pa, pb, fit, "gaussian", 0.3, np.random.default_rng(1))
    c = edges.sample_kernel_edges(pa, pb, fit, "gaussian", 0.3, np.random.default_rng(2))
    assert (a != b).nnz == 0
    assert (a != c).nnz > 0


def test_knn_degrees(small):
    pa, pb, _, _ = small
    G = edges.knn_edges(pa, pb, 5)
    assert np.all(np.diff(G.indptr) == 5)
    # the linked TGT are the 5 closest ones
    d = np.linalg.norm(pa[0] - pb, axis=1)
    assert set(G[0].indices) == set(np.argsort(d)[:5])


def test_reference_values():
    """Same potentials, probabilities and draws as the reference implementation."""
    ref = np.load(REFERENCE)
    scale = float(ref["scale"])
    fit = edges.sinkhorn(ref["pos_a"], ref["pos_b"], ref["k_a"], ref["k_b"], "gaussian", scale)
    assert np.allclose(fit.theta_a, ref["theta_a"], rtol=1e-4)
    assert np.allclose(fit.theta_b, ref["theta_b"], rtol=1e-4)
    n_a = len(ref["pos_a"])
    P = edges.probability_rows(ref["pos_a"], ref["pos_b"], fit, "gaussian", scale, 0, n_a)
    assert np.allclose(P, ref["P"], atol=1e-6)
    G = edges.sample_kernel_edges(ref["pos_a"], ref["pos_b"], fit, "gaussian", scale,
                                  np.random.default_rng(int(ref["seed"])))
    got = set(zip(*G.nonzero(), strict=True))
    want = set(zip(ref["rows"].tolist(), ref["cols"].tolist(), strict=True))
    assert len(got ^ want) <= 2  # last-ulp differences of exp/BLAS across machines


def test_potentials_do_not_depend_on_blas_threads(small):
    from threadpoolctl import threadpool_limits

    pa, pb, k_a, k_b = small
    with threadpool_limits(limits=4, user_api="blas"):
        a = edges.sinkhorn(pa, pb, k_a, k_b, "gaussian", 0.3)
    b = edges.sinkhorn(pa, pb, k_a, k_b, "gaussian", 0.3)
    assert np.array_equal(a.theta_a, b.theta_a) and np.array_equal(a.theta_b, b.theta_b)


def test_unreachable_degrees_raise(small):
    pa, pb, k_a, k_b = small
    with pytest.raises(ValueError, match="Increase edges.scale"):
        edges.sinkhorn(pa, pb, k_a * 5, k_b * 5, "gaussian", 0.02, max_iter=200)


def test_no_convergence_warns_with_gap(small):
    pa, pb, k_a, k_b = small
    with pytest.warns(UserWarning, match="increase edges.scale"):
        fit = edges.sinkhorn(pa, pb, k_a, k_b, "gaussian", 0.3, max_iter=3)
    assert not fit.converged and 0 < fit.max_error < 1
