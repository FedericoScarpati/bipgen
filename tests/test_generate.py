import importlib

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


def test_configuration_background_uses_geometric_degrees(spec, monkeypatch):
    module = importlib.import_module("bipgen.generate")
    seen = {}
    background_edges = module.background_edges

    def background(geometric, ratio, model, rng, weights_a, weights_b):
        seen.update(geometric=geometric, weights=(weights_a, weights_b))
        return background_edges(geometric, ratio, model, rng, weights_a, weights_b)

    monkeypatch.setattr(module, "background_edges", background)
    bipgen.generate(spec)
    geometric = seen["geometric"]
    np.testing.assert_array_equal(seen["weights"][0], geometric.sum(axis=1).A1)
    np.testing.assert_array_equal(seen["weights"][1], geometric.sum(axis=0).A1)


def test_configuration_background_share_is_uniform():
    # every node gets on average ratio / (1 + ratio) of its edges from the background,
    # also nodes with few partners nearby (short kernel, nodes near the border)
    spec = bipgen.spec_from_dict({
        "sizes": {"n_bcn": 800, "n_tgt": 1200},
        "degrees": {"distribution": "lognormal", "bcn": {"median": 8, "sigma": 0.5},
                    "tgt": {"sigma": 0.6}},
        "edges": {"model": "kernel", "scale": 0.08},
        "background": {"model": "configuration", "ratio": 0.2},
    })
    outer, inner = np.zeros(2), np.zeros(2)
    for seed in range(5):
        g = bipgen.generate(spec, seed=seed)
        border = np.linalg.norm(g.pos_bcn, axis=1) > 0.9
        counts = np.stack([g.geometric.sum(axis=1).A1, g.background.sum(axis=1).A1])
        outer += counts[:, border].sum(axis=1)
        inner += counts[:, ~border].sum(axis=1)
    for geometric, background in (outer, inner):
        assert abs(background / (geometric + background) - 0.2 / 1.2) < 0.02


def test_overlapping_edges_rejected(spec):
    g = bipgen.generate(spec)
    with pytest.raises(ValueError, match="overlap"):
        bipgen.BipartiteGraph(g.geometric, g.geometric, g.pos_bcn, g.pos_tgt, 0)


def test_metropolis_positions_3d():
    spec = bipgen.spec_from_dict({**KERNEL, "dim": 3, "edges": {"model": "kernel", "scale": 0.35},
                                  "positions": {"method": "metropolis", "density": {
                                      "n_bins": 8, "correlogram": [0.3, 0.1], "sweeps": 30}}})
    g = bipgen.generate(spec)
    assert g.pos_tgt.shape == (400, 3)
    assert np.linalg.norm(g.pos_bcn, axis=1).max() <= 1.0


def test_file_positions_used_as_given():
    rng = np.random.default_rng(0)
    p, q = rng.uniform(-0.7, 0.7, (300, 2)), rng.uniform(-0.7, 0.7, (400, 2))
    spec = bipgen.spec_from_dict({**KERNEL, "positions": {"method": "file", "bcn": "p", "tgt": "q"},
                                  "arrays": {"p": p, "q": q}})
    g = bipgen.generate(spec)
    assert np.array_equal(g.pos_bcn, p) and np.array_equal(g.pos_tgt, q)
    assert g.geometric.nnz > 0


def test_timing_recorded(spec):
    t = bipgen.generate(spec).info["timing"]
    assert set(t) == {"positions", "degrees", "sinkhorn", "edges", "background", "total"}
    assert all(v >= 0 for v in t.values())
    assert t["total"] >= max(v for k, v in t.items() if k != "total")
    knn = bipgen.generate(bipgen.spec_from_dict({"sizes": {"n_bcn": 50, "n_tgt": 60},
                                                 "edges": {"model": "knn", "k": 3}})).info["timing"]
    assert set(knn) == {"positions", "edges", "background", "total"}


def test_graph_keeps_a_snapshot_of_the_spec(spec):
    g = bipgen.generate(spec)
    spec.edges["scale"] = 0.8
    assert g.spec.edges["scale"] == 0.25
    again = bipgen.generate(g.spec)
    np.testing.assert_array_equal(again.geometric.toarray(), g.geometric.toarray())
