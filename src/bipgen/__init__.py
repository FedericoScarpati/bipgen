"""bipgen: spatial bipartite proximity graphs from degree-matched null models.

Typical use::

    import bipgen

    spec = bipgen.load_spec("spec.yaml")
    graph = bipgen.generate(spec)            # seed from the spec, default 42
    other = bipgen.generate(spec, seed=7)    # another realization
"""

from . import descriptors, presets
from .compare import compare, degree_ks, statistics
from .fit import spec_from_graph
from .generate import generate
from .graph import BipartiteGraph
from .io import load, read_edge_list, save
from .plot import plot_comparison, plot_degrees, plot_edge_lengths, plot_graph
from .spec import Spec, SpecError, load_spec, spec_from_dict

__version__ = "0.1.0"

__all__ = [
    "BipartiteGraph",
    "Spec",
    "SpecError",
    "__version__",
    "compare",
    "degree_ks",
    "descriptors",
    "generate",
    "load",
    "load_spec",
    "plot_comparison",
    "plot_degrees",
    "plot_edge_lengths",
    "plot_graph",
    "presets",
    "read_edge_list",
    "save",
    "spec_from_dict",
    "spec_from_graph",
    "statistics",
]
