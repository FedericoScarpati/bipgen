import numpy as np
import pytest

import bipgen
from bipgen import descriptors


@pytest.fixture
def graph():
    spec = {"sizes": {"n_bcn": 50, "n_tgt": 60}, "edges": {"model": "knn", "k": 3},
            "background": {"model": "uniform", "ratio": 0.2}}
    return bipgen.generate(bipgen.spec_from_dict(spec))


def test_node_degree(graph):
    tables = descriptors.node_descriptors(graph)
    assert np.array_equal(tables["bcn"]["degree"], graph.degrees()[0])
    assert len(tables["tgt"]) == graph.n_tgt


def test_edge_degree_sum(graph):
    table = descriptors.edge_descriptors(graph)
    deg_bcn, deg_tgt = graph.degrees()
    row = table.iloc[0]
    assert row["degree_sum"] == deg_bcn[row["bcn"]] + deg_tgt[row["tgt"]]
    assert len(table) == graph.n_edges


def test_registry(graph):
    @descriptors.register_node("is_leaf_test")
    def _leaf(g):
        b, t = g.degrees()
        return b == 1, t == 1

    assert "is_leaf_test" in descriptors.available()["node"]
    assert "is_leaf_test" in descriptors.node_descriptors(graph, ["is_leaf_test"])["tgt"]
    with pytest.raises(KeyError):
        descriptors.edge_descriptors(graph, ["nope"])
