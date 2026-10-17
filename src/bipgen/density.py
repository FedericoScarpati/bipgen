"""Spatially correlated density maps in the unit disk (2D), by simulated annealing.

Stages:

1. **occupancies**: one count per grid cell, either drawn from a Gamma
   distribution with a given coefficient of variation, or resampled from
   counts measured elsewhere (for instance on a real tissue);
2. **layout**: the counts are rearranged on the grid by Metropolis swaps with a
   geometric annealing schedule, until the spatial autocorrelation (Moran's I)
   at lags 1, 2, ... matches a target correlogram. Swaps conserve the multiset
   of counts exactly, so only the arrangement changes;
3. **points**: each cell receives its number of points, uniformly inside the
   cell; points outside the disk are discarded and the set is thinned at random
   to exactly ``N``.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree


# --------------------------------------------------------------------- grid
def grid_mask(n_bins: int) -> tuple[np.ndarray, float]:
    """Cells of an ``n_bins x n_bins`` grid over ``[-1, 1]^2`` whose centre lies in the disk."""
    dx = 2.0 / n_bins
    centers = -1.0 + (np.arange(n_bins) + 0.5) * dx
    yy, xx = np.meshgrid(centers, centers, indexing="ij")
    return xx**2 + yy**2 <= 1.0, dx


def shell_index(mask: np.ndarray, lag: int) -> tuple[np.ndarray, np.ndarray]:
    """For every valid cell, the valid cells at rounded grid distance ``lag``.

    Returns neighbour ids padded with ``n`` (a sentinel) and the number of
    valid neighbours of each cell. Lag 1 is the 8-cell queen neighbourhood.
    """
    side = mask.shape[0]
    cells = np.column_stack(np.nonzero(mask))
    n = len(cells)
    flat = np.full(mask.shape, -1, dtype=np.int64)
    flat[cells[:, 0], cells[:, 1]] = np.arange(n)
    offsets = [
        (dr, dc)
        for dr in range(-lag, lag + 1)
        for dc in range(-lag, lag + 1)
        if (dr, dc) != (0, 0) and int(round(np.hypot(dr, dc))) == lag
    ]
    idx = np.full((n, len(offsets)), n, dtype=np.int32)
    cnt = np.zeros(n, dtype=np.int32)
    for k, (r, c) in enumerate(cells):
        m = 0
        for dr, dc in offsets:
            rr, cc = r + dr, c + dc
            if 0 <= rr < side and 0 <= cc < side and mask[rr, cc]:
                idx[k, m] = flat[rr, cc]
                m += 1
        cnt[k] = m
    return idx, cnt


def _neighbour_sums(z: np.ndarray, idx: np.ndarray) -> np.ndarray:
    return np.append(z, 0.0)[idx].sum(axis=1)


# ------------------------------------------------------------------- target
class CorrelogramTarget:
    """Energy ``sum_k (rho_k - target_k)^2``, with ``rho_k`` Moran's I on the lag-k shell.

    A swap of cells ``i`` and ``j`` changes each ``S_k = sum_<ab>_k z_a z_b``
    by ``(z_j - z_i)(A_k[i] - A_k[j])``, minus ``(z_i - z_j)^2`` for the one
    shell that contains the pair itself, so every proposal costs O(lags).
    """

    def __init__(self, values: np.ndarray, shells: list, target: list[float]):
        self.shells = shells
        self.target = np.asarray(target, dtype=float)
        self.mean = float(values.mean())
        self.z = values.astype(float) - self.mean
        var = float((self.z**2).mean())
        self.A = [_neighbour_sums(self.z, idx) for idx, _ in shells]
        pairs = np.array([0.5 * float(cnt.sum()) for _, cnt in shells])
        self.S = np.array([0.5 * float((self.z * A).sum()) for A in self.A])
        self.norm = pairs * var if var > 0 else np.full(len(shells), np.inf)

    def rho(self) -> np.ndarray:
        return self.S / self.norm

    def energy(self) -> float:
        return float(((self.rho() - self.target) ** 2).sum())

    def _shell_of_pair(self, i: int, j: int) -> int:
        for k, (idx, cnt) in enumerate(self.shells):
            if (idx[i, : cnt[i]] == j).any():
                return k
        return -1

    def delta(self, i: int, j: int):
        a, b = self.z[i], self.z[j]
        if a == b:
            return 0.0, None
        dS = np.array([(b - a) * (A[i] - A[j]) for A in self.A])
        shell = self._shell_of_pair(i, j)
        if shell >= 0:
            dS[shell] -= (a - b) ** 2
        now = ((self.rho() - self.target) ** 2).sum()
        new = (((self.S + dS) / self.norm - self.target) ** 2).sum()
        return float(new - now), dS

    def commit(self, i: int, j: int, dS: np.ndarray) -> None:
        a, b = self.z[i], self.z[j]
        for A, (idx, cnt) in zip(self.A, self.shells, strict=True):
            A[idx[i, : cnt[i]]] += b - a
            A[idx[j, : cnt[j]]] += a - b
        self.z[i], self.z[j] = b, a
        self.S += dS


# ---------------------------------------------------------------- annealing
def anneal(
    values: np.ndarray,
    mask: np.ndarray,
    target: list[float],
    rng: np.random.Generator,
    sweeps: int = 400,
    local_radius: float = 3.0,
    p_global: float = 0.1,
    t_end_frac: float = 1e-3,
) -> tuple[np.ndarray, np.ndarray]:
    """Arrange ``values`` on the valid cells of ``mask`` to match the target correlogram.

    One sweep proposes as many swaps as there are cells. Partners are mostly
    within ``local_radius`` cells, sometimes anywhere. The initial temperature
    accepts about 80% of worsening moves; it decreases geometrically to
    ``t_end_frac`` of that value. Returns the count map and the final
    correlogram.
    """
    values = rng.permutation(np.asarray(values, dtype=float))
    n = int(mask.sum())
    if values.size != n:
        raise ValueError(f"{values.size} values for {n} cells")
    shells = [shell_index(mask, lag) for lag in range(1, len(target) + 1)]
    state = CorrelogramTarget(values, shells, target)

    cells = np.column_stack(np.nonzero(mask)).astype(float)
    balls = cKDTree(cells).query_ball_point(cells, local_radius)
    near = [np.asarray(x, dtype=np.int32) for x in balls]

    def propose() -> tuple[int, int]:
        i = int(rng.integers(n))
        if rng.random() < p_global:
            return i, int(rng.integers(n))
        cand = near[i]
        return i, int(cand[rng.integers(cand.size)])

    worse = [d for d, _ in (state.delta(*propose()) for _ in range(300)) if d > 0]
    t0 = float(-np.mean(worse) / np.log(0.8)) if worse else 1e-12
    temps = t0 * (max(t0 * t_end_frac, 1e-300) / t0) ** (np.arange(sweeps) / max(sweeps - 1, 1))
    for temp in temps:
        for _ in range(n):
            i, j = propose()
            if i == j:
                continue
            dE, dS = state.delta(i, j)
            if dS is None:
                continue
            if dE <= 0 or rng.random() < np.exp(-dE / temp):
                state.commit(i, j, dS)
    counts = np.zeros(mask.shape, dtype=np.int64)
    counts[mask] = np.rint(state.z + state.mean).astype(np.int64)
    return counts, state.rho()


# ----------------------------------------------------------- occupancies
def gamma_occupancies(n_cells: int, cv: float, rng: np.random.Generator) -> np.ndarray:
    """Per-cell weights from a Gamma distribution with coefficient of variation ``cv``."""
    return rng.gamma(1.0 / cv**2, 1.0, n_cells)


def occupancy_population(occupancies, n_cells: int, n_points: int, rng) -> np.ndarray:
    """Resample ``n_cells`` counts from ``occupancies`` and rescale them to sum to ``n_points``."""
    occupancies = np.asarray(occupancies, dtype=float)
    if occupancies.size == 0 or occupancies.sum() <= 0:
        raise ValueError("occupancies must contain positive values")
    draw = rng.choice(occupancies, size=n_cells, replace=True)
    pop = np.maximum(np.rint(draw * (n_points / draw.sum())).astype(np.int64), 0)
    delta = n_points - int(pop.sum())
    if delta > 0:
        np.add.at(pop, rng.integers(0, pop.size, size=delta), 1)
    for _ in range(max(-delta, 0)):
        positive = np.flatnonzero(pop > 0)
        pop[positive[rng.integers(positive.size)]] -= 1
    return pop


def points_from_counts(counts: np.ndarray, dx: float, n_points: int, rng) -> np.ndarray:
    """Points uniform inside their cells, kept in the disk, thinned to exactly ``n_points``."""
    idx = np.nonzero(counts > 0)
    per_cell = counts[idx]
    cells = np.column_stack(idx)
    oversample = 1.15
    for _ in range(60):
        rep = np.repeat(cells, np.ceil(oversample * per_cell).astype(int), axis=0)
        cand = -1.0 + (rep + rng.uniform(0.0, 1.0, size=rep.shape)) * dx
        pts = cand[np.sum(cand**2, axis=1) <= 1.0]
        if len(pts) >= n_points:
            rng.shuffle(pts, axis=0)
            return pts[:n_points, ::-1].copy()  # (row, col) -> (x, y)
        oversample *= 1.3
    raise RuntimeError("could not place enough points inside the disk")


def sample_points(n_points: int, density: dict, arrays: dict, rng) -> tuple[np.ndarray, dict]:
    """``n_points`` positions in the unit disk following ``density`` (a spec section)."""
    mask, dx = grid_mask(int(density["n_bins"]))
    n_cells = int(mask.sum())
    occ = density["occupancy"]
    if occ["source"] == "gamma":
        weights = gamma_occupancies(n_cells, float(occ["cv"]), rng)
    else:
        weights = arrays[occ["key"]]
    pop = occupancy_population(weights, n_cells, n_points, rng)
    counts, rho = anneal(pop, mask, density["correlogram"], rng, sweeps=int(density["sweeps"]))
    pts = points_from_counts(counts, dx, n_points, rng)
    return pts, {"correlogram": [float(r) for r in rho]}


# ------------------------------------------------------------- measurement
def _moran(values: np.ndarray, idx: np.ndarray, cnt: np.ndarray) -> float:
    z = values - values.mean()
    var = float((z**2).mean())
    if var <= 0:
        return float("nan")
    return 0.5 * float((z * _neighbour_sums(z, idx)).sum()) / (0.5 * float(cnt.sum()) * var)


def measure(points: np.ndarray, n_bins: int = 20, n_lags: int = 2) -> dict:
    """Cell occupancies and correlogram of a set of 2D points (for example, measured positions).

    The bounding box of the points is mapped to ``[-1, 1]^2`` axis by axis and the
    counts are taken on the cells of :func:`grid_mask`. When the points are only a
    sample of the nodes, Poisson counting noise inflates the variance of the
    counts and attenuates Moran's I; the corrected correlogram multiplies it by
    ``var / (var - mean)``, which estimates the autocorrelation of the underlying
    density (counts modelled as Poisson draws around it). The correction is
    undefined when the variance does not exceed the mean.

    Limitation: the correction assumes Poisson counts and tends to overestimate
    the correlation when the known points are a large fraction of the nodes; it
    also becomes unstable when cells hold few points, because the factor
    ``var / (var - mean)`` grows large.
    """
    pts = np.asarray(points, dtype=float)
    pts = pts[np.isfinite(pts).all(axis=1)]
    lo, hi = pts.min(axis=0), pts.max(axis=0)
    norm = 2 * (pts - lo) / (hi - lo) - 1
    mask, _ = grid_mask(n_bins)
    H, _, _ = np.histogram2d(norm[:, 0], norm[:, 1], bins=n_bins, range=[[-1, 1], [-1, 1]])
    occ = H[mask]
    raw = np.array([_moran(occ, *shell_index(mask, lag)) for lag in range(1, n_lags + 1)])
    var, mean = float(occ.var()), float(occ.mean())
    factor = var / (var - mean) if var > mean else float("nan")
    return {"occupancy": occ, "correlogram_raw": raw.tolist(),
            "correlogram": (raw * factor).tolist(), "attenuation": factor,
            "n_points": int(len(pts))}
