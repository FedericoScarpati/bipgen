import numpy as np
import pytest

import bipgen

KERNEL = {
    "sizes": {"n_bcn": 300, "n_tgt": 400},
    "degrees": {"distribution": "lognormal", "bcn": {"median": 8, "sigma": 0.4},
                "tgt": {"sigma": 0.4}},
    "edges": {"model": "kernel", "scale": 0.25},
    "background": {"model": "configuration", "ratio": 0.1},
}


@pytest.fixture
def spec():
    return bipgen.spec_from_dict(KERNEL)


def test_seed_default_override_and_record(spec):
    g = bipgen.generate(spec)
    assert g.seed == 42 and g.info["seed"] == 42
    assert bipgen.generate(spec, seed=7).seed == 7
    spec.seed = 9
    assert bipgen.generate(spec).seed == 9


def test_reproducible(spec):
    a, b, c = bipgen.generate(spec), bipgen.generate(spec), bipgen.generate(spec, seed=1)
    assert (a.biadjacency != b.biadjacency).nnz == 0
    assert np.array_equal(a.pos_bcn, b.pos_bcn)
    assert (a.biadjacency != c.biadjacency).nnz > 0


def test_kernel_graph(spec):
    g = bipgen.generate(spec)
    assert g.geometric.shape == (300, 400)
    assert g.info["sinkhorn_converged"]
    assert g.info["expected_edges"] <= g.info["target_edges"]
    assert abs(g.geometric.nnz - g.info["expected_edges"]) < 5 * np.sqrt(g.info["expected_edges"])
    assert g.background.nnz == int(0.1 * g.geometric.nnz)
    table = g.edges()
    assert len(table) == g.n_edges
    assert (table["origin"] == "background").sum() == g.background.nnz


def test_3d_and_knn():
    spec = bipgen.spec_from_dict({"sizes": {"n_bcn": 200, "n_tgt": 250}, "dim": 3,
                                  "edges": {"model": "knn", "k": 4}})
    g = bipgen.generate(spec)
    assert g.dim == 3
    assert np.all(np.linalg.norm(g.pos_tgt, axis=1) <= 1)
    assert np.all(g.degrees()[0] == 4) and g.background.nnz == 0


def test_metropolis_positions():
    spec = bipgen.spec_from_dict({**KERNEL, "positions": {
        "method": "metropolis",
        "density": {"n_bins": 10, "correlogram": [0.4, 0.2], "sweeps": 40}}})
    g = bipgen.generate(spec)
    assert g.pos_bcn.shape == (300, 2) and len(g.info["correlogram"]) == 2


def test_overlapping_edges_rejected(spec):
    g = bipgen.generate(spec)
    with pytest.raises(ValueError, match="overlap"):
        bipgen.BipartiteGraph(g.geometric, g.geometric, g.pos_bcn, g.pos_tgt, 0)
