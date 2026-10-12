"""Target degree sequences for the two sides.

Side A (BCN) degrees come from the chosen distribution. Side B (TGT) degrees
take only their shape from it: they are rescaled to the side-A total, because
both sides of a bipartite graph share the same number of edges. Degrees are
integers in ``[1, n_other]`` and the two totals are equal exactly.
"""

from __future__ import annotations

import numpy as np


def _match_sum(deg: np.ndarray, target: int, cap: int, rng: np.random.Generator) -> np.ndarray:
    """Add or remove 1 on random entries until ``deg.sum() == target``, within ``[1, cap]``."""
    if not len(deg) <= target <= len(deg) * cap:
        raise ValueError(f"cannot reach total {target} with {len(deg)} degrees in [1, {cap}]")
    diff = int(target - deg.sum())
    while diff != 0:
        idx = rng.integers(len(deg), size=min(abs(diff), len(deg)))
        if diff > 0:
            idx = idx[deg[idx] < cap]
            np.add.at(deg, idx[: diff], 1)
        else:
            idx = idx[deg[idx] > 1]
            np.subtract.at(deg, idx[: -diff], 1)
        np.clip(deg, 1, cap, out=deg)
        diff = int(target - deg.sum())
    return deg


def _shape(dist: str, params: dict, n: int, rng: np.random.Generator) -> np.ndarray:
    """Unnormalized positive values drawn from the requested distribution."""
    if dist == "lognormal":
        median = float(params.get("median", 1.0))
        sigma = float(params["sigma"])
        if median <= 0 or sigma < 0:
            raise ValueError("lognormal degrees need median > 0 and sigma >= 0")
        return rng.lognormal(np.log(median), sigma, n)
    if dist == "constant":
        return np.full(n, float(params.get("value", 1.0)))
    if dist == "empirical":
        values = np.asarray(params["values"], dtype=float)
        values = values[values > 0]
        if values.size == 0:
            raise ValueError("empirical degrees contain no positive value")
        return values if values.size == n else rng.choice(values, size=n, replace=True)
    raise ValueError(f"unknown degree distribution {dist!r}")


def sample_degrees(
    dist: str,
    bcn: dict,
    tgt: dict,
    n_bcn: int,
    n_tgt: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Integer target degrees ``(k_bcn, k_tgt)`` with equal totals.

    ``bcn`` and ``tgt`` hold the distribution parameters of each side. For
    ``empirical`` they hold a ``values`` array. Only the shape of the side-B
    distribution matters.
    """
    k_bcn = np.clip(np.rint(_shape(dist, bcn, n_bcn, rng)), 1, n_tgt).astype(np.int64)
    total = int(k_bcn.sum())
    raw = _shape(dist, {**tgt, "median": 1.0} if dist == "lognormal" else tgt, n_tgt, rng)
    k_tgt = np.clip(np.rint(raw * total / raw.sum()), 1, n_bcn).astype(np.int64)
    k_tgt = _match_sum(k_tgt, total, n_bcn, rng)
    return k_bcn, k_tgt
