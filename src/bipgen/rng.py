"""Random streams derived from a single seed.

Every source of randomness in a realization draws from its own stream, spawned
from one :class:`numpy.random.SeedSequence`. Streams are identified by their
position in :data:`STREAMS`; new streams are only ever appended, so existing
streams keep producing the same numbers across versions.
"""

from __future__ import annotations

import numpy as np

DEFAULT_SEED = 42

STREAMS = ("positions", "degrees", "edges", "background")


def resolve_seed(seed: int | None, spec_seed: int | None = None) -> int:
    """Return the seed to use: the explicit one, else the spec one, else 42."""
    for value in (seed, spec_seed):
        if value is not None:
            value = int(value)
            if value < 0:
                raise ValueError(f"seed must be a non-negative integer, got {value}")
            return value
    return DEFAULT_SEED


def streams(seed: int) -> dict[str, np.random.Generator]:
    """One independent generator per entry of :data:`STREAMS`."""
    children = np.random.SeedSequence(seed).spawn(len(STREAMS))
    pairs = zip(STREAMS, children, strict=True)
    return {name: np.random.default_rng(child) for name, child in pairs}
