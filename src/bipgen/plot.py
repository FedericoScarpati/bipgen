"""Quick views of one graph: layout, degrees, edge lengths."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.ticker import NullFormatter

# Categorical slots 1-3 of a colour-blind-checked palette, plus neutral greys.
COLORS = {
    "bcn": "#2a78d6",
    "tgt": "#eb6834",
    "background": "#1baf7a",
    "geometric": "#8a8984",
    "ink": "#52514e",
}


def _style(ax):
    for side in ("top", "right"):
        if side in ax.spines:
            ax.spines[side].set_visible(False)
    ax.tick_params(colors=COLORS["ink"], labelsize=8)


def _legend_below(ax, ncol):
    ax.legend(fontsize=8, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.08),
              ncol=ncol, markerscale=3)


def plot_graph(graph, ax=None, max_edges=3000, node_size=3.0, seed=0):
    """Nodes of both sides and a random subset of edges (2D or 3D).

    Geometric and background edges are subsampled in the same proportion, so
    the picture keeps their relative abundance. ``max_edges=0`` hides edges.
    """
    rng = np.random.default_rng(seed)
    three_d = graph.dim == 3
    if ax is None:
        fig = plt.figure(figsize=(5, 5.4))
        ax = fig.add_subplot(projection="3d" if three_d else None)
    fraction = min(1.0, max_edges / max(graph.n_edges, 1))
    for origin, z in (("background", 1), ("geometric", 2)):
        coo = getattr(graph, origin).tocoo()
        n_keep = int(round(fraction * coo.nnz))
        if n_keep == 0:
            continue
        keep = rng.choice(coo.nnz, size=n_keep, replace=False)
        segments = np.stack([graph.pos_bcn[coo.row[keep]], graph.pos_tgt[coo.col[keep]]], axis=1)
        style = dict(colors=COLORS[origin], linewidths=0.5, alpha=0.6)
        if three_d:
            from mpl_toolkits.mplot3d.art3d import Line3DCollection

            ax.add_collection3d(Line3DCollection(segments, **style))
        else:
            ax.add_collection(LineCollection(segments, zorder=z, **style))
        ax.plot([], [], color=COLORS[origin], lw=2, label=f"{origin} edges")
    for side, pos in (("bcn", graph.pos_bcn), ("tgt", graph.pos_tgt)):
        ax.scatter(*pos.T, s=node_size, color=COLORS[side], label=side.upper(), linewidths=0,
                   zorder=3)
    if not three_d:
        ax.set_aspect("equal")
        _style(ax)
    # below 1% one significant digit, so that a small sample does not read "0%"
    shown = f", {100 * fraction:.1g}% of edges drawn" if 0 < fraction < 0.01 else (
        f", {fraction:.0%} of edges drawn" if 0 < fraction < 1 else "")
    ax.set_title(f"{graph.n_bcn} BCN, {graph.n_tgt} TGT, {graph.n_edges} edges{shown}",
                 fontsize=9, color=COLORS["ink"])
    _legend_below(ax, ncol=4)
    return ax


def _ccdf(values):
    """Fraction of nodes with degree >= k, for each distinct k >= 1."""
    values = np.sort(np.asarray(values)[np.asarray(values) >= 1])
    k, first = np.unique(values, return_index=True)
    return k, 1.0 - first / values.size


def plot_degrees(graph, axes=None):
    """Complementary cumulative degree distributions, realized and target, on log-log axes."""
    if axes is None:
        _, axes = plt.subplots(1, 2, figsize=(8, 3.2))
    targets = graph.target_degrees or (None, None)
    for ax, side, deg, target in zip(axes, ("bcn", "tgt"), graph.degrees(), targets, strict=True):
        ax.step(*_ccdf(deg), where="post", color=COLORS[side], lw=2, label="realized")
        if target is not None:
            ax.step(*_ccdf(target), where="post", color=COLORS["ink"], lw=1.5, ls="--",
                    label="target")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.xaxis.set_minor_formatter(NullFormatter())  # no labels on minor ticks: they overlap
        ax.set_xlabel(f"{side.upper()} degree k", fontsize=9)
        ax.set_ylabel("fraction of nodes with degree ≥ k", fontsize=9)
        ax.legend(fontsize=8, frameon=False)
        _style(ax)
    return axes


def plot_edge_lengths(graph, ax=None, bins=60):
    """Distribution of edge lengths, geometric and background edges separately."""
    if ax is None:
        _, ax = plt.subplots(figsize=(5, 3.2))
    table = graph.edge_lengths()
    edges = np.linspace(0, table["length"].max(), bins)
    for origin in ("geometric", "background"):
        values = table.loc[table["origin"] == origin, "length"]
        if len(values):
            ax.hist(values, bins=edges, histtype="step", lw=2, color=COLORS[origin], label=origin)
    ax.set_xlabel("edge length (domain radius = 1)", fontsize=9)
    ax.set_ylabel("edges", fontsize=9)
    ax.set_yscale("log")
    ax.legend(fontsize=8, frameon=False)
    _style(ax)
    return ax


def plot_comparison(table, ax=None):
    """z-score of each observed statistic within the ensemble; the band marks |z| <= 2.

    Statistics with no spread in the ensemble have no z-score: they are drawn as
    hollow markers at 0 and labelled "no spread".
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(5, 0.3 * len(table) + 1))
    z = table["z"].to_numpy(dtype=float)
    y = np.arange(len(table))[::-1]
    finite = np.isfinite(z)
    ax.axvspan(-2, 2, color=COLORS["geometric"], alpha=0.15, lw=0)
    ax.axvline(0, color=COLORS["geometric"], lw=1)
    ax.scatter(z[finite], y[finite], s=36, color=COLORS["bcn"], zorder=3,
               edgecolors="white", linewidths=2)
    ax.scatter(np.zeros((~finite).sum()), y[~finite], s=36, facecolors="white", zorder=3,
               edgecolors=COLORS["ink"], linewidths=1.2)
    labels = [name if ok else f"{name} (no spread)" for name, ok in zip(table.index, finite,
                                                                        strict=True)]
    ax.set_yticks(y, labels=labels, fontsize=8)
    ax.set_xlabel("z-score of the observed value in the ensemble", fontsize=9)
    _style(ax)
    return ax
