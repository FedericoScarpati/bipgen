"""Generation specification: a YAML file, plus an optional ``.npz`` of arrays.

A specification fully determines an ensemble of graphs; the seed selects one
realization. See ``docs/spec_example.yaml`` for a commented example.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import yaml

from . import kernels
from .rng import DEFAULT_SEED

SCHEMA = 0

_TOP = {"schema", "seed", "dim", "sizes", "positions", "degrees", "edges", "background", "arrays"}
_SECTIONS = {
    "sizes": {"n_bcn", "n_tgt"},
    "positions": {"method", "density", "bcn", "tgt"},
    "degrees": {"distribution", "bcn", "tgt"},
    "edges": {"model", "kernel", "scale", "sinkhorn", "k"},
    "background": {"model", "ratio"},
}
_DENSITY = {"n_bins", "occupancy", "correlogram", "sweeps"}
_DEGREE_PARAMS = {
    "lognormal": ({"median", "sigma"}, {"sigma"}),
    "constant": ({"value"}, set()),
    "empirical": ({"key"}, {"key"}),
}


class SpecError(ValueError):
    """Raised when a specification is incomplete or inconsistent."""


def _check_keys(where: str, given: dict, allowed: set) -> None:
    unknown = set(given) - allowed
    if unknown:
        raise SpecError(f"{where}: unknown keys {sorted(unknown)}; allowed: {sorted(allowed)}")


def _positive_int(where: str, value) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 1:
        raise SpecError(f"{where} must be a positive integer, got {value!r}")
    return int(value)


@dataclass
class Spec:
    """Validated specification. Sections are plain dictionaries mirroring the YAML."""

    sizes: dict
    degrees: dict | None = None
    edges: dict = field(default_factory=lambda: {"model": "kernel", "kernel": "gaussian"})
    positions: dict = field(default_factory=lambda: {"method": "uniform"})
    background: dict = field(default_factory=lambda: {"model": "none", "ratio": 0.0})
    dim: int = 2
    seed: int = DEFAULT_SEED
    schema: int = SCHEMA
    arrays: dict = field(default_factory=dict, repr=False)
    arrays_file: str | None = None

    def __post_init__(self) -> None:
        self.validate()

    @property
    def n_bcn(self) -> int:
        return self.sizes["n_bcn"]

    @property
    def n_tgt(self) -> int:
        return self.sizes["n_tgt"]

    # ------------------------------------------------------------------ checks
    def validate(self) -> None:
        if self.schema != SCHEMA:
            raise SpecError(f"unsupported schema {self.schema}; this version reads schema {SCHEMA}")
        if self.seed is None:
            self.seed = DEFAULT_SEED
        integer = isinstance(self.seed, (int, np.integer)) and not isinstance(self.seed, bool)
        if not integer or self.seed < 0:
            raise SpecError(f"seed must be a non-negative integer, got {self.seed!r}")
        if self.dim not in (2, 3):
            raise SpecError(f"dim must be 2 or 3, got {self.dim!r}")
        for name in ("sizes", "positions", "edges", "background"):
            _check_keys(name, getattr(self, name), _SECTIONS[name])
        for side in ("n_bcn", "n_tgt"):
            if side not in self.sizes:
                raise SpecError(f"sizes.{side} is required")
            self.sizes[side] = _positive_int(f"sizes.{side}", self.sizes[side])
        self._validate_positions()
        self._validate_edges()
        self._validate_background()

    def _validate_positions(self) -> None:
        method = self.positions.setdefault("method", "uniform")
        if method not in ("uniform", "metropolis", "file"):
            raise SpecError("positions.method must be 'uniform', 'metropolis' or 'file', "
                            f"got {method!r}")
        if method != "metropolis" and "density" in self.positions:
            raise SpecError("positions.density is only used with method 'metropolis'")
        if method != "file" and {"bcn", "tgt"} & set(self.positions):
            raise SpecError("positions.bcn and positions.tgt are only used with method 'file'")
        if method == "uniform":
            return
        if method == "file":
            for side, n in (("bcn", self.n_bcn), ("tgt", self.n_tgt)):
                if side not in self.positions:
                    raise SpecError(f"positions.{side} is required with method 'file'")
                key = self.positions[side]
                self._require_array(key, f"positions.{side}")
                shape = np.shape(self.arrays[key])
                if shape != (n, self.dim):
                    raise SpecError(f"positions.{side}: array {key!r} has shape {shape}, "
                                    f"expected ({n}, {self.dim})")
                if not np.isfinite(self.arrays[key]).all():
                    raise SpecError(f"positions.{side}: array {key!r} has non-finite values")
            return
        density = self.positions.setdefault("density", {})
        _check_keys("positions.density", density, _DENSITY)
        default_bins = 20 if self.dim == 2 else 12
        density["n_bins"] = _positive_int("positions.density.n_bins",
                                          density.get("n_bins", default_bins))
        density["sweeps"] = _positive_int("positions.density.sweeps", density.get("sweeps", 400))
        occ = density.setdefault("occupancy", {"source": "gamma", "cv": 0.5})
        if occ.get("source") == "gamma":
            _check_keys("positions.density.occupancy", occ, {"source", "cv"})
            if float(occ.get("cv", 0)) <= 0:
                raise SpecError("positions.density.occupancy.cv must be positive")
        elif occ.get("source") == "file":
            _check_keys("positions.density.occupancy", occ, {"source", "key"})
            self._require_array(occ.get("key"), "positions.density.occupancy.key")
        else:
            raise SpecError("positions.density.occupancy.source must be 'gamma' or 'file'")
        rho = density.setdefault("correlogram", [0.3, 0.1])
        if not isinstance(rho, (list, tuple)) or not rho or not all(-1 < float(r) < 1 for r in rho):
            raise SpecError("positions.density.correlogram must be a list of values in (-1, 1)")
        density["correlogram"] = [float(r) for r in rho]

    def _validate_edges(self) -> None:
        model = self.edges.setdefault("model", "kernel")
        if model == "knn":
            if self.degrees is not None:
                raise SpecError("the degrees section is not used with edges.model 'knn': each BCN "
                                "links to its k nearest TGT; remove it")
            if "k" not in self.edges:
                raise SpecError("edges.k is required with model 'knn'")
            k = _positive_int("edges.k", self.edges["k"])
            if k > self.n_tgt:
                raise SpecError(f"edges.k = {k} exceeds sizes.n_tgt = {self.n_tgt}")
            return
        if model != "kernel":
            raise SpecError(f"edges.model must be 'kernel' or 'knn', got {model!r}")
        name = self.edges.setdefault("kernel", "gaussian")
        if name not in kernels.available():
            known = kernels.available()
            raise SpecError(f"edges.kernel {name!r} is not registered; available: {known}")
        if float(self.edges.get("scale", 0)) <= 0:
            raise SpecError("edges.scale is required with model 'kernel' and must be positive")
        solver = self.edges.setdefault("sinkhorn", {})
        _check_keys("edges.sinkhorn", solver, {"tol", "max_iter"})
        solver.setdefault("tol", 1e-3)
        solver.setdefault("max_iter", 2000)
        self._validate_degrees()

    def _validate_degrees(self) -> None:
        if self.degrees is None:
            raise SpecError("the degrees section is required with edges.model 'kernel'")
        _check_keys("degrees", self.degrees, _SECTIONS["degrees"])
        dist = self.degrees.get("distribution")
        if dist not in _DEGREE_PARAMS:
            raise SpecError(f"degrees.distribution must be one of {sorted(_DEGREE_PARAMS)}")
        allowed_bcn, allowed_tgt = _DEGREE_PARAMS[dist]
        bcn = self.degrees.setdefault("bcn", {})
        tgt = self.degrees.setdefault("tgt", {})
        if dist == "lognormal" and "median" in tgt:
            raise SpecError(
                "degrees.tgt.median is not a free parameter: both sides share the same number "
                "of edges, so n_bcn * <k_bcn> = n_tgt * <k_tgt> sets the TGT scale; give only sigma"
            )
        _check_keys("degrees.bcn", bcn, allowed_bcn)
        _check_keys("degrees.tgt", tgt, allowed_tgt)
        missing = (allowed_bcn - set(bcn)) | {f"tgt.{k}" for k in allowed_tgt - set(tgt)}
        if missing:
            raise SpecError(f"degrees: missing {sorted(missing)} for distribution {dist!r}")
        if dist == "empirical":
            self._require_array(bcn["key"], "degrees.bcn.key")
            self._require_array(tgt["key"], "degrees.tgt.key")

    def _validate_background(self) -> None:
        model = self.background.setdefault("model", "none")
        if model not in ("none", "uniform", "configuration"):
            raise SpecError("background.model must be 'none', 'uniform' or 'configuration'")
        ratio = float(self.background.setdefault("ratio", 0.0))
        if ratio < 0:
            raise SpecError("background.ratio must be >= 0")
        self.background["ratio"] = ratio

    def _require_array(self, key, where: str) -> None:
        if key not in self.arrays:
            raise SpecError(f"{where}: array {key!r} not found in the companion arrays file")

    # --------------------------------------------------------------------- I/O
    def to_dict(self) -> dict:
        """Plain dictionary in the YAML layout (arrays are referenced, not embedded)."""
        out = {"schema": self.schema, "seed": int(self.seed), "dim": self.dim}
        for name in ("sizes", "positions", "degrees", "edges", "background"):
            value = getattr(self, name)
            if value is not None:
                out[name] = copy.deepcopy(value)
        out["arrays"] = self.arrays_file
        return out

    def save(self, path) -> Path:
        """Write the YAML file and, if arrays are present, the ``.npz`` next to it."""
        path = Path(path)
        data = self.to_dict()
        if self.arrays:
            npz = path.with_suffix(".npz")
            np.savez_compressed(npz, **self.arrays)
            data["arrays"] = npz.name
        path.write_text(yaml.safe_dump(data, sort_keys=False))
        return path


def spec_from_dict(data: dict, base_dir=None) -> Spec:
    """Build and validate a :class:`Spec` from a dictionary in the YAML layout.

    ``arrays`` is the name of a ``.npz`` file, relative to ``base_dir``, or a
    dictionary of arrays.
    """
    data = copy.deepcopy(data or {})
    _check_keys("specification", data, _TOP)
    arrays, arrays_file = {}, data.pop("arrays", None)
    if isinstance(arrays_file, dict):
        arrays, arrays_file = {k: np.asarray(v) for k, v in arrays_file.items()}, None
    elif arrays_file:
        npz = Path(base_dir or ".") / arrays_file
        with np.load(npz) as f:
            arrays = {k: f[k] for k in f.files}
    return Spec(
        sizes=data.get("sizes", {}),
        degrees=data.get("degrees"),
        edges=data.get("edges", {"model": "kernel", "kernel": "gaussian"}),
        positions=data.get("positions", {"method": "uniform"}),
        background=data.get("background", {"model": "none", "ratio": 0.0}),
        dim=data.get("dim", 2),
        seed=data.get("seed", DEFAULT_SEED),
        schema=data.get("schema", SCHEMA),
        arrays=arrays,
        arrays_file=arrays_file,
    )


def load_spec(path) -> Spec:
    """Read a specification from a YAML file."""
    path = Path(path)
    return spec_from_dict(yaml.safe_load(path.read_text()), base_dir=path.parent)
