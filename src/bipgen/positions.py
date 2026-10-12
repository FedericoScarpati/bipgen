"""Node positions in the unit disk (2D) or the unit ball (3D)."""

from __future__ import annotations

import numpy as np

SUPPORTED_DIMS = (2, 3)


def _check_dim(dim: int) -> None:
    if dim not in SUPPORTED_DIMS:
        raise ValueError(f"dim must be one of {SUPPORTED_DIMS}, got {dim}")


def uniform(n: int, dim: int, rng: np.random.Generator) -> np.ndarray:
    """``n`` points drawn uniformly in the unit ball of dimension ``dim``.

    Directions are isotropic Gaussian vectors normalized to unit length; the
    radius is ``u**(1/dim)`` with ``u`` uniform, which makes the density flat.
    """
    _check_dim(dim)
    direction = rng.standard_normal((n, dim))
    direction /= np.linalg.norm(direction, axis=1, keepdims=True)
    radius = rng.random(n) ** (1.0 / dim)
    return direction * radius[:, None]
