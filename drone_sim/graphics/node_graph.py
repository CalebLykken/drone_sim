"""Draw node graphs like rqt_graph: nodes, topics, and who talks to whom.

Nodes are rounded boxes colored by kind (sim, sensor, estimator, controller,
other); topics are plain boxes showing the message type and measured rate.
Arrows go node -> topic (publish) and topic -> node (subscribe). Columns are
assigned left to right along the data flow; feedback edges (e.g. motor
commands back into the simulator) are drawn as arcs underneath.

Drones whose stacks have the same structure are drawn once, labeled with
every drone that uses it.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from ..nodes.core import Graph
from ..nodes.layout import BOX_H, BOX_W, KIND_COLORS, layout, route, short, signature

TOPIC_COLOR = "white"


def draw_graphs(graphs: list[Graph], title: str | None = None):
    """Draw one or more graphs in a new figure and return it."""
    groups: dict = {}
    for g in graphs:
        groups.setdefault(signature(g), []).append(g)

    fig, ax = plt.subplots()
    ax.set_axis_off()
    y0, xmax = 0.0, 0.0
    for gs in groups.values():
        g = gs[0]
        pos, succ, back, long_topics = layout(g)
        ys = [y for _, y in pos.values()]
        lanes_up = 0.6 + 0.3 * len(back)
        offset = y0 - (max(ys) + BOX_H / 2 + lanes_up + 0.5)
        P = {v: (x, y + offset) for v, (x, y) in pos.items()}
        names = ", ".join(x.name for x in gs)
        label = f"{names}  ({g.namespace or '/'})" if len(gs) == 1 else f"{names}  (same stack)"
        ax.text(-BOX_W / 2, y0, label, fontsize=11, weight="bold", va="top")
        _draw_one(ax, g, P, succ, back, long_topics, top=max(ys) + offset + BOX_H / 2)
        xmax = max(xmax, max(x for x, _ in P.values()))
        y0 = min(ys) + offset - BOX_H / 2 - 0.9
    xmin, ymin, ymax = -BOX_W / 2 - 0.4, y0 + 0.4, 0.4
    ax.set_xlim(xmin, xmax + BOX_W / 2 + 0.4)
    ax.set_ylim(ymin, ymax)
    ax.set_aspect("equal")
    scale = 0.62  # inches per unit
    fig.set_size_inches(min(max((xmax + BOX_W + 0.8) * scale, 6), 24), min(max((ymax - ymin) * scale, 3), 18))
    if title:
        ax.set_title(title)
    fig.tight_layout()
    return fig


def _arrow(ax, verts, color="0.35", alpha=1.0, z=2):
    from matplotlib.path import Path

    ax.add_patch(FancyArrowPatch(path=Path(verts), arrowstyle="-|>", mutation_scale=11,
                                 color=color, lw=1.0, alpha=alpha, zorder=z))


def _draw_one(ax, g: Graph, P, succ, back, long_topics, top):
    nodes = {n.name: n for n in g.nodes}
    for (kind, name), (x, y) in P.items():
        if kind == "node":
            n = nodes[name]
            face = KIND_COLORS.get(n.kind, KIND_COLORS["node"])
            ax.add_patch(FancyBboxPatch((x - BOX_W / 2, y - BOX_H / 2), BOX_W, BOX_H,
                                        boxstyle="round,pad=0.02,rounding_size=0.3",
                                        fc=face, ec="0.25", lw=1.2, zorder=3))
            ax.text(x, y + 0.11, name, ha="center", va="center", fontsize=8.5, weight="bold", zorder=4)
            ax.text(x, y - 0.17, type(n).__name__, ha="center", va="center",
                    fontsize=6.5, color="0.3", zorder=4)
        else:
            t = g.topics[name]
            ax.add_patch(FancyBboxPatch((x - BOX_W / 2, y - BOX_H / 2), BOX_W, BOX_H,
                                        boxstyle="square,pad=0.02", fc=TOPIC_COLOR, ec="0.45",
                                        lw=1.0, ls="--", zorder=3))
            hz = t.rate()
            rate = f"  {hz:.0f} Hz" if hz else ""
            ax.text(x, y + 0.11, short(name, g.namespace), ha="center", va="center", fontsize=8,
                    family="monospace", zorder=4)
            ax.text(x, y - 0.17, f"{t.msg_type.__name__}{rate}", ha="center", va="center",
                    fontsize=6.5, color="0.35", zorder=4)

    for verts, feedback in route(P, succ, back, long_topics, top):
        if feedback:
            _arrow(ax, verts, color="tab:red", alpha=0.75, z=1)
        else:
            _arrow(ax, verts)


def save_graphs(graphs: list[Graph], path: str) -> None:
    """Write .dot or .mmd text, or render .png/.svg/.pdf."""
    from ..nodes.core import to_dot, to_mermaid

    if path.endswith(".dot") or path.endswith(".gv"):
        text = to_dot(graphs)
    elif path.endswith(".mmd") or path.endswith(".md"):
        text = to_mermaid(graphs)
    else:
        fig = draw_graphs(graphs)
        fig.savefig(path, dpi=130)
        plt.close(fig)
        return
    with open(path, "w") as f:
        f.write(text + "\n")
