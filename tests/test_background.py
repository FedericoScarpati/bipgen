import numpy as np
import pytest
from scipy.sparse import random as sparse_random

from bipgen.background import background_edges


@pytest.fixture
def geometric():
    G = sparse_random(400, 500, density=0.02, format="csr", random_state=0)
    G.data[:] = 1
    return G


@pytest.mark.parametrize("model", ["uniform", "configuration"])
def test_count_disjoint_unique(geometric, model):
    w_a, w_b = np.arange(1, 401), np.arange(1, 501)
    r, c = background_edges(geometric, 0.1, model, np.random.default_rng(0), w_a, w_b)
    assert r.size == int(0.1 * geometric.nnz)
    assert geometric[r, c].sum() == 0
    assert np.unique(r * 500 + c).size == r.size


def test_configuration_prefers_heavy_nodes(geometric):
    w_a = np.ones(400)
    w_a[:40] = 50.0
    g = np.random.default_rng(1)
    r, _ = background_edges(geometric, 0.5, "configuration", g, w_a, np.ones(500))
    assert np.mean(r < 40) > 0.5


def test_none_and_zero(geometric):
    for model, ratio in (("none", 0.1), ("uniform", 0.0)):
        r, c = background_edges(geometric, ratio, model, np.random.default_rng(0))
        assert r.size == c.size == 0


def test_bad_inputs(geometric):
    with pytest.raises(ValueError):
        background_edges(geometric, 0.1, "erdos", np.random.default_rng(0))
    with pytest.raises(ValueError):
        background_edges(geometric, 0.1, "configuration", np.random.default_rng(0))
    with pytest.raises(ValueError, match="positive weights"):
        background_edges(geometric, 0.1, "configuration", np.random.default_rng(0),
                         np.zeros(400), np.ones(500))
