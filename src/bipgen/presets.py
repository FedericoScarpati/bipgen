"""Specifications shipped with the package, built from published datasets.

Each preset is a YAML specification with a companion ``.npz`` of arrays and a
JSON summary of the observed graph it was built from. See ``presets/README.md``
for sources and choices.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .spec import Spec, load_spec

DIRECTORY = Path(__file__).parent / "presets"


def available() -> list[str]:
    """Names of the shipped presets."""
    return sorted(p.stem for p in DIRECTORY.glob("*.yaml"))


def _path(name: str) -> Path:
    path = DIRECTORY / f"{name}.yaml"
    if not path.exists():
        raise KeyError(f"unknown preset {name!r}; available: {available()}")
    return path


def load(name: str) -> Spec:
    """The specification of preset ``name``."""
    return load_spec(_path(name))


def observed(name: str) -> dict:
    """Summary of the observed graph behind preset ``name``, for :func:`bipgen.compare.compare`."""
    path = _path(name)
    summary = json.loads(path.with_name(f"{name}_observed.json").read_text())
    with np.load(path.with_suffix(".npz")) as arrays:
        degrees = (arrays["degrees_bcn"], arrays["degrees_tgt"])
    return {"statistics": summary["statistics"], "degrees": degrees, "source": summary["source"]}
