import numpy as np
import pytest
from scipy.sparse import csr_matrix
from scipy.sparse import random as sparse_random

import bipgen
from bipgen import density


@pytest.fixture
def observed():
    adj = sparse_random(200, 260, density=0.04, format="csr", random_state=1)
    adj.data = np.rint(adj.data * 9) + 1  # weighted, as counts would be
    return adj


def test_spec_from_graph_uniform(observed):
    spec = bipgen.spec_from_graph(observed, scale=0.3)
    assert spec.degrees["distribution"] == "empirical"
    assert spec.arrays["degrees_bcn"].sum() == observed.nnz
    assert spec.positions == {"method": "uniform"}
    g = bipgen.generate(spec)
    assert g.target_degrees[0].sum() == g.target_degrees[1].sum()


def test_spec_from_graph_with_positions(observed):
    pts = np.random.default_rng(0).normal(size=(3000, 2))
    spec = bipgen.spec_from_graph(observed, scale=0.3, positions=pts, n_bins=10, sweeps=20)
    assert spec.positions["method"] == "metropolis"
    assert "occupancy" in spec.arrays and len(spec.positions["density"]["correlogram"]) == 2


def test_measure_correction():
    pts = np.random.default_rng(0).normal(size=(20000, 2))
    m = density.measure(pts, n_bins=16)
    assert m["attenuation"] > 1
    assert np.allclose(m["correlogram"], np.array(m["correlogram_raw"]) * m["attenuation"])


def test_statistics_and_compare(observed):
    spec = bipgen.spec_from_graph(observed, scale=0.5)
    ensemble = [bipgen.generate(spec, seed=s) for s in range(6)]
    stats = bipgen.statistics(observed)
    assert stats["n_edges"] == observed.nnz and 0 < stats["giant_component_fraction"] <= 1
    table = bipgen.compare(observed, ensemble)
    assert set(table.columns) == {"observed", "mean", "sd", "z", "percentile"}
    assert table.loc["n_edges", "observed"] == observed.nnz
    assert table["percentile"].between(0, 1).all()
    ks = bipgen.degree_ks(observed, ensemble)
    assert list(ks.index) == ["bcn", "tgt"] and (ks.to_numpy() >= 0).all()


def test_presets():
    names = bipgen.presets.available()
    assert {"c58_4", "c58_5", "c58_16", "slidetags_tonsil"} <= set(names)
    spec = bipgen.presets.load("c58_4")
    obs = bipgen.presets.observed("c58_4")
    assert spec.n_bcn == obs["statistics"]["n_bcn"]
    assert obs["degrees"][0].sum() == obs["statistics"]["n_edges"]
    assert spec.positions["method"] == "metropolis"
    tonsil = bipgen.presets.observed("slidetags_tonsil")
    assert bipgen.presets.load("slidetags_tonsil").n_tgt == tonsil["statistics"]["n_tgt"]


def test_statistics_of_a_graph_without_edges():
    stats = bipgen.statistics(csr_matrix((2, 2)))
    assert stats["n_edges"] == 0 and stats["bcn_degree_max"] == 0
    assert np.isnan(stats["bcn_degree_mean"]) and np.isnan(stats["giant_component_fraction"])


def test_duplicate_entries_are_one_edge():
    adj = csr_matrix((np.array([1, 1]), np.array([0, 0]), np.array([0, 2, 2])), shape=(2, 2))
    spec = bipgen.spec_from_graph(adj, scale=0.2)
    assert spec.arrays["degrees_bcn"].tolist() == [1]


@pytest.mark.parametrize("positions,message", [
    (np.random.default_rng(0).random((50, 3)), r"\(m, 2\)"),
    (np.zeros((50, 2)), "line or a point"),
    (np.full((5, 2), np.nan), "at least two points"),
])
def test_measured_positions_are_checked(observed, positions, message):
    with pytest.raises(ValueError, match=message):
        bipgen.spec_from_graph(observed, scale=0.2, positions=positions)
