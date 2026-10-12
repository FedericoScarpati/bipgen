import numpy as np
import pytest

from bipgen import rng


def test_default_seed_is_42():
    assert rng.resolve_seed(None) == 42
    assert rng.resolve_seed(None, spec_seed=7) == 7
    assert rng.resolve_seed(3, spec_seed=7) == 3


def test_negative_seed_rejected():
    with pytest.raises(ValueError):
        rng.resolve_seed(-1)


def test_same_seed_same_streams():
    a, b = rng.streams(5), rng.streams(5)
    for name in rng.STREAMS:
        assert np.array_equal(a[name].random(10), b[name].random(10))


def test_streams_are_distinct():
    s = rng.streams(5)
    draws = [s[name].random(10) for name in rng.STREAMS]
    assert len({d.tobytes() for d in draws}) == len(rng.STREAMS)


def test_existing_streams_survive_new_ones():
    old = np.random.SeedSequence(5).spawn(len(rng.STREAMS))
    new = np.random.SeedSequence(5).spawn(len(rng.STREAMS) + 3)
    for a, b in zip(old, new, strict=False):
        x, y = np.random.default_rng(a).random(5), np.random.default_rng(b).random(5)
        assert np.array_equal(x, y)
