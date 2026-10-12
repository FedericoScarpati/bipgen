import numpy as np
import pytest

from bipgen import kernels


def test_gaussian_values():
    d = np.array([0.0, 0.1, 0.2])
    assert np.allclose(kernels.get("gaussian")(d, 0.1), np.exp(-np.array([0.0, 1.0, 4.0])))


def test_registry():
    assert "gaussian" in kernels.available()
    with pytest.raises(KeyError):
        kernels.get("nope")
    with pytest.raises(ValueError):
        kernels.register("gaussian")(lambda d, scale: d)


def test_bad_scale():
    with pytest.raises(ValueError):
        kernels.gaussian(np.ones(3), 0.0)
