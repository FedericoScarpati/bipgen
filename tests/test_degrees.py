import numpy as np
import pytest

from bipgen.degrees import sample_degrees


@pytest.mark.parametrize(
    "dist,bcn,tgt",
    [
        ("lognormal", {"median": 20, "sigma": 0.5}, {"sigma": 0.8}),
        ("constant", {"value": 12}, {}),
        ("empirical", {"values": [3, 5, 8, 40]}, {"values": [1, 2, 2, 9]}),
    ],
)
def test_totals_match_and_bounds(dist, bcn, tgt):
    k_a, k_b = sample_degrees(dist, bcn, tgt, 500, 700, np.random.default_rng(0))
    assert k_a.sum() == k_b.sum()
    assert k_a.min() >= 1 and k_a.max() <= 700
    assert k_b.min() >= 1 and k_b.max() <= 500
    assert k_a.dtype == np.int64 and k_b.dtype == np.int64


def test_lognormal_median():
    k_a, _ = sample_degrees("lognormal", {"median": 20, "sigma": 0.3}, {"sigma": 0.3}, 20000, 20000,
                            np.random.default_rng(1))
    assert abs(np.median(k_a) - 20) <= 1


def test_tgt_scale_follows_totals():
    k_a, k_b = sample_degrees("constant", {"value": 10}, {}, 1000, 2000, np.random.default_rng(2))
    assert np.isclose(k_b.mean(), 5.0)


def test_unknown_distribution():
    with pytest.raises(ValueError):
        sample_degrees("poisson", {}, {}, 10, 10, np.random.default_rng(0))
