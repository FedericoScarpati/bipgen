import numpy as np
import pytest
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
