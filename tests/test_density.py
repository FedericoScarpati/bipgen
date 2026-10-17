from pathlib import Path

import numpy as np

from bipgen import density

REFERENCE = Path(__file__).parent / "data" / "reference_metropolis.npz"


def test_shell_lag1_is_queen():
    mask = np.ones((5, 5), dtype=bool)
    _, cnt = density.shell_index(mask, 1)
    assert cnt.reshape(5, 5)[2, 2] == 8 and cnt.reshape(5, 5)[0, 0] == 3


def test_population_sums_exactly():
    pop = density.occupancy_population([1.0, 5.0, 2.0], 50, 1234, np.random.default_rng(0))
    assert pop.sum() == 1234 and pop.size == 50 and pop.min() >= 0


def test_anneal_reaches_target():
    mask, _ = density.grid_mask(16)
    pop = density.occupancy_population(np.random.default_rng(1).gamma(2.0, 1.0, 300),
                                       int(mask.sum()), 6000, np.random.default_rng(2))
    counts, rho = density.anneal(pop, mask, [0.5, 0.3], np.random.default_rng(3), sweeps=150)
    assert counts.sum() == 6000
    assert np.allclose(rho, [0.5, 0.3], atol=0.03)


def test_reference_values():
    """Same points as the reference implementation, with x and y in the other order."""
    ref = np.load(REFERENCE)
    spec = {"n_bins": int(ref["n_bins"]), "occupancy": {"source": "file", "key": "occ"},
            "correlogram": ref["correlogram"].tolist(), "sweeps": int(ref["sweeps"])}
    pts, _ = density.sample_points(int(ref["n_points"]), spec, {"occ": ref["occupancies"]},
                                   np.random.default_rng(int(ref["seed"])))
    assert np.allclose(pts, ref["points"][:, ::-1])


def test_grid_and_shells_in_3d():
    mask, dx = density.grid_mask(12, 3)
    assert mask.shape == (12, 12, 12) and dx == 2.0 / 12
    assert abs(int(mask.sum()) - np.pi / 6 * 12**3) < 0.03 * 12**3
    _, cnt = density.shell_index(mask, 1)
    assert cnt.max() == 18  # faces and edges; corners are at rounded distance 2


def test_anneal_reaches_target_in_3d():
    mask, _ = density.grid_mask(10, 3)
    pop = density.occupancy_population(np.random.default_rng(1).gamma(2.0, 1.0, 300),
                                       int(mask.sum()), 8000, np.random.default_rng(2))
    counts, rho = density.anneal(pop, mask, [0.5, 0.3], np.random.default_rng(3), sweeps=150)
    assert counts.sum() == 8000
    assert np.allclose(rho, [0.5, 0.3], atol=0.03)


def test_points_in_3d_ball():
    spec = {"n_bins": 8, "occupancy": {"source": "gamma", "cv": 0.5}, "correlogram": [0.3],
            "sweeps": 30}
    pts, info = density.sample_points(3000, spec, {}, np.random.default_rng(0), dim=3)
    assert pts.shape == (3000, 3)
    assert np.linalg.norm(pts, axis=1).max() <= 1.0
    assert len(info["correlogram"]) == 1
