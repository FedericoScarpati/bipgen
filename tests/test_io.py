import json

import numpy as np
import pytest

import bipgen

SPEC = {
    "sizes": {"n_bcn": 120, "n_tgt": 150}, "dim": 3,
    "degrees": {"distribution": "lognormal", "bcn": {"median": 6, "sigma": 0.4},
                "tgt": {"sigma": 0.4}},
    "edges": {"model": "kernel", "scale": 0.4},
    "background": {"model": "uniform", "ratio": 0.1},
}


@pytest.fixture
def graph():
    return bipgen.generate(bipgen.spec_from_dict(SPEC), seed=3)


def test_roundtrip(graph, tmp_path):
    again = bipgen.load(bipgen.save(graph, tmp_path / "g"))
    assert again.seed == 3 and again.dim == 3
    assert (again.geometric != graph.geometric).nnz == 0
    assert (again.background != graph.background).nnz == 0
    assert np.allclose(again.pos_tgt, graph.pos_tgt)
    assert all(np.array_equal(a, b) for a, b in zip(again.target_degrees, graph.target_degrees,
                                                    strict=True))
    assert again.spec.to_dict() == graph.spec.to_dict()


def test_roundtrip_knn(tmp_path):
    g = bipgen.generate(bipgen.spec_from_dict({"sizes": {"n_bcn": 30, "n_tgt": 40},
                                               "edges": {"model": "knn", "k": 3}}))
    again = bipgen.load(bipgen.save(g, tmp_path / "k"))
    assert again.target_degrees is None and again.n_edges == 90


def test_read_edge_list(tmp_path):
    path = tmp_path / "e.csv"
    path.write_text("bcn,tgt,weight\n0,1,5\n0,1,2\n2,0,1\n")
    adj = bipgen.read_edge_list(path)
    assert adj.shape == (3, 2) and adj.nnz == 2 and adj.max() == 1


def test_to_sga_files(graph, tmp_path):
    from scipy.sparse import load_npz

    root = bipgen.to_sga(graph, tmp_path / "sga")
    adj = load_npz(root / "adjacency.npz")
    n = graph.n_bcn + graph.n_tgt
    assert adj.shape == (n, n) and (adj != adj.T).nnz == 0 and adj.nnz == 2 * graph.n_edges
    assert np.load(root / "positions.npy").shape == (n, 3)
    meta = json.loads((root / "metadata.json").read_text())
    assert meta["seed"] == 3 and meta["n_bcn"] == graph.n_bcn


def test_to_sga_loads_in_sga(graph, tmp_path):
    benchmark_io = pytest.importorskip("spatial_graph_algorithms.compare.benchmark_io")
    sg = benchmark_io.load_benchmark_graph(bipgen.to_sga(graph, tmp_path / "sga"))
    assert sg.adjacency_matrix.shape[0] == graph.n_bcn + graph.n_tgt
    assert int(sg.edge_metadata["is_false"].sum()) == graph.background.nnz
