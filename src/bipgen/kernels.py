"""Registry of distance kernels.

A kernel maps distances ``d`` (array) and a length ``scale`` to non-negative
weights of the same shape. New kernels are added with :func:`register`.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

Kernel = Callable[..., np.ndarray]

_REGISTRY: dict[str, Kernel] = {}


def register(name: str) -> Callable[[Kernel], Kernel]:
    """Decorator that adds a kernel to the registry under ``name``."""

    def decorator(func: Kernel) -> Kernel:
        if name in _REGISTRY:
            raise ValueError(f"kernel {name!r} is already registered")
        _REGISTRY[name] = func
        return func

    return decorator


def get(name: str) -> Kernel:
    """Return the kernel registered under ``name``."""
    try:
        return _REGISTRY[name]
    except KeyError:
        raise KeyError(f"unknown kernel {name!r}; available: {available()}") from None


def available() -> list[str]:
    """Names of the registered kernels."""
    return sorted(_REGISTRY)


@register("gaussian")
def gaussian(d: np.ndarray, scale: float) -> np.ndarray:
    """Gaussian kernel ``exp(-(d / scale)**2)``."""
    if scale <= 0:
        raise ValueError(f"kernel scale must be positive, got {scale}")
    return np.exp(-((d / scale) ** 2))
