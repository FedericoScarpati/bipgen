from pathlib import Path

import numpy as np
import pytest

from bipgen import SpecError, load_spec, spec_from_dict

EXAMPLE = Path(__file__).parents[1] / "docs" / "spec_example.yaml"

BASE = {
    "sizes": {"n_bcn": 100, "n_tgt": 150},
    "degrees": {"distribution": "lognormal", "bcn": {"median": 5, "sigma": 0.5},
                "tgt": {"sigma": 0.5}},
    "edges": {"model": "kernel", "scale": 0.2},
}


def make(**changes):
    data = {**BASE, **changes}
    return spec_from_dict(data)


def test_example_loads():
    spec = load_spec(EXAMPLE)
    assert spec.seed == 42 and spec.n_bcn == 5000
    assert spec.edges["sinkhorn"] == {"tol": 1e-3, "max_iter": 2000}


def test_defaults():
    spec = make()
    assert spec.seed == 42 and spec.dim == 2
    assert spec.positions == {"method": "uniform"}
    assert spec.background == {"model": "none", "ratio": 0.0}


@pytest.mark.parametrize(
    "changes,message",
    [
        ({"colour": 1}, "unknown keys"),
        ({"sizes": {"n_bcn": 0, "n_tgt": 5}}, "positive integer"),
        ({"dim": 4}, "dim"),
        ({"seed": -3}, "seed"),
        ({"edges": {"model": "kernel", "scale": 0.2, "kernal": "x"}}, "unknown keys"),
        ({"edges": {"model": "kernel", "kernel": "nope", "scale": 0.2}}, "not registered"),
        ({"edges": {"model": "kernel"}}, "scale"),
        ({"edges": {"model": "knn"}}, "edges.k"),
        ({"edges": {"model": "knn", "k": 500}}, "exceeds"),
        ({"degrees": {"distribution": "lognormal", "bcn": {"median": 5, "sigma": 0.5},
                      "tgt": {"median": 5, "sigma": 0.5}}}, "not a free parameter"),
        ({"degrees": {"distribution": "lognormal", "bcn": {"median": 5}, "tgt": {"sigma": 1}}},
         "missing"),
        ({"positions": {"method": "metropolis"}, "dim": 3}, "2D only"),
        ({"background": {"model": "erdos", "ratio": 0.1}}, "background.model"),
    ],
)
def test_invalid(changes, message):
    with pytest.raises(SpecError, match=message):
        make(**changes)


def test_knn_needs_no_degrees():
    spec = spec_from_dict({"sizes": {"n_bcn": 10, "n_tgt": 10}, "edges": {"model": "knn", "k": 3}})
    assert spec.degrees is None


def test_roundtrip_with_arrays(tmp_path):
    np.savez(tmp_path / "a.npz", deg_b=np.arange(1, 101), deg_t=np.arange(1, 151))
    data = {**BASE, "degrees": {"distribution": "empirical", "bcn": {"key": "deg_b"},
                                "tgt": {"key": "deg_t"}}, "arrays": "a.npz"}
    spec = spec_from_dict(data, base_dir=tmp_path)
    path = spec.save(tmp_path / "copy.yaml")
    again = load_spec(path)
    assert again.to_dict() == {**spec.to_dict(), "arrays": "copy.npz"}
    assert np.array_equal(again.arrays["deg_b"], np.arange(1, 101))


def test_missing_array_key(tmp_path):
    data = {**BASE, "degrees": {"distribution": "empirical", "bcn": {"key": "x"},
                                "tgt": {"key": "y"}}}
    with pytest.raises(SpecError, match="not found"):
        spec_from_dict(data)
