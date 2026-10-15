"""bipgen: spatial bipartite proximity graphs from degree-matched null models.

Typical use::

    import bipgen

    spec = bipgen.load_spec("spec.yaml")
    graph = bipgen.generate(spec)            # seed from the spec, default 42
    other = bipgen.generate(spec, seed=7)    # another realization
"""

from .generate import generate
from .graph import BipartiteGraph
from .plot import plot_degrees, plot_edge_lengths, plot_graph
from .spec import Spec, SpecError, load_spec, spec_from_dict

__version__ = "0.0.1"

__all__ = [
    "BipartiteGraph",
    "Spec",
    "SpecError",
    "__version__",
    "generate",
    "load_spec",
    "plot_degrees",
    "plot_edge_lengths",
    "plot_graph",
    "spec_from_dict",
]
