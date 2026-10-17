"""Registry of node and edge descriptors.

A node descriptor maps a graph to one array per side; an edge descriptor maps
it to one value per edge, in the row order of :meth:`BipartiteGraph.edges`.
New descriptors are added with :func:`register_node` or :func:`register_edge`.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np
import pandas as pd

_NODE: dict[str, Callable] = {}
_EDGE: dict[str, Callable] = {}


def _register(table: dict, name: str):
    def decorator(func):
        if name in table:
            raise ValueError(f"descriptor {name!r} is already registered")
        table[name] = func
        return func

    return decorator


def register_node(name: str):
    """Decorator for ``func(graph) -> (array_bcn, array_tgt)``."""
    return _register(_NODE, name)


def register_edge(name: str):
    """Decorator for ``func(graph, edges) -> array`` with one value per row of ``edges``."""
    return _register(_EDGE, name)


def available() -> dict[str, list[str]]:
    """Names of the registered descriptors."""
    return {"node": sorted(_NODE), "edge": sorted(_EDGE)}


@register_node("degree")
def _degree(graph):
    return graph.degrees()


@register_edge("degree_sum")
def _degree_sum(graph, edges):
    deg_bcn, deg_tgt = graph.degrees()
    return deg_bcn[edges["bcn"].to_numpy()] + deg_tgt[edges["tgt"].to_numpy()]


def node_descriptors(graph, names: Sequence[str] = ("degree",)) -> dict[str, pd.DataFrame]:
    """One table per side (``"bcn"``, ``"tgt"``), one column per descriptor."""
    out = {"bcn": pd.DataFrame(index=np.arange(graph.n_bcn)),
           "tgt": pd.DataFrame(index=np.arange(graph.n_tgt))}
    for name in names:
        if name not in _NODE:
            raise KeyError(f"unknown node descriptor {name!r}; available: {sorted(_NODE)}")
        values_bcn, values_tgt = _NODE[name](graph)
        out["bcn"][name], out["tgt"][name] = values_bcn, values_tgt
    return out


def edge_descriptors(graph, names: Sequence[str] = ("degree_sum",)) -> pd.DataFrame:
    """The edge table of ``graph`` with one extra column per descriptor."""
    edges = graph.edges()
    for name in names:
        if name not in _EDGE:
            raise KeyError(f"unknown edge descriptor {name!r}; available: {sorted(_EDGE)}")
        edges[name] = _EDGE[name](graph, edges)
    return edges
