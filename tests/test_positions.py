import numpy as np
import pytest
from scipy import stats

from bipgen import positions


@pytest.mark.parametrize("dim", [2, 3])
def test_uniform_in_unit_ball(dim):
    x = positions.uniform(20000, dim, np.random.default_rng(0))
    r = np.linalg.norm(x, axis=1)
    assert x.shape == (20000, dim)
    assert r.max() <= 1.0
    # uniform density in the ball: r**dim is uniform on [0, 1]
    assert stats.kstest(r**dim, "uniform").pvalue > 1e-3


def test_unsupported_dim():
    with pytest.raises(ValueError):
        positions.uniform(10, 4, np.random.default_rng(0))
