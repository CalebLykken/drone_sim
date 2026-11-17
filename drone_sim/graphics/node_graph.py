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

from collections import defaultdict

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from ..nodes.core import Graph

KIND_COLORS = {
    "sim": "#d9d9d9",
    "sensor": "#a6cee3",
    "estimator": "#b2df8a",
    "controller": "#fdbf6f",
    "planner": "#cab2d6",
    "node": "#ffffb3",
}
TOPIC_COLOR = "white"
COL_W, ROW_H = 3.2, 1.1
BOX_W, BOX_H = 2.5, 0.72


def _short(topic: str, namespace: str) -> str:
    prefix = namespace.rstrip("/") + "/"
    return topic[len(prefix):] if namespace and topic.startswith(prefix) else topic


def _signature(g: Graph):
    return (
        tuple((type(n).__name__, n.name) for n in g.nodes),
        tuple(sorted((_short(s, g.namespace), _short(d, g.namespace), k) for s, d, k in g.edges())),
    )


def _layout(g: Graph):
    """Place vertices on a grid and route the edges.

    Columns follow the data flow (longest path along forward edges); each
    topic sits just left of its first reader. Rows within a column are
    ordered by the rows of what feeds them. Topics that would need a long
    arrow across other columns go in lanes below everything, and feedback
    edges run in lanes above, so no line crosses a box.
    """
    vertices = [("node", n.name) for n in g.nodes] + [("topic", t) for t in g.topics]
    succ = defaultdict(list)
    for src, dst, kind in g.edges():
        a = ("node", src) if kind == "pub" else ("topic", src)
        b = ("topic", dst) if kind == "pub" else ("node", dst)
        succ[a].append(b)

    # Depth-first search from the nodes in the order they were added (the sim
    # bridge first); edges back to a vertex on the current path are feedback.
    state, back = {}, set()

    def visit(v):
        state[v] = "open"
        for w in succ[v]:
            if state.get(w) == "open":
                back.add((v, w))
            elif w not in state:
                visit(w)
        state[v] = "done"

    for v in vertices:
        if v not in state:
            visit(v)

    pred = defaultdict(list)
    for v in vertices:
        for w in succ[v]:
            if (v, w) not in back:
                pred[w].append(v)

    col = {v: 0 for v in vertices}
    for _ in range(len(vertices)):  # longest path; the forward graph is a DAG
        changed = False
        for v in vertices:
            for w in succ[v]:
                if (v, w) not in back and col[w] < col[v] + 1:
                    col[w] = col[v] + 1
                    changed = True
        if not changed:
            break
    for v in vertices:  # pull each topic next to its first reader
        if v[0] == "topic":
            readers = [col[w] for w in succ[v] if (v, w) not in back]
            if readers:
                col[v] = max(col[v], min(readers) - 1)

    def is_long(v):
        readers = [col[w] for w in succ[v] if (v, w) not in back]
        return v[0] == "topic" and pred[v] and readers and \
            min(readers) - 1 > max(col[p] for p in pred[v]) + 1

    long_topics = [v for v in vertices if is_long(v)]
    columns = defaultdict(list)
    for v in vertices:
        if v not in long_topics:
            columns[col[v]].append(v)
    row = {}
    for c in sorted(columns):
        if c > 0:  # order by the average row of what feeds each vertex
            columns[c].sort(key=lambda v: sum(row.get(p, 0) for p in pred[v]) / len(pred[v]) if pred[v] else 0)
        n = len(columns[c])
        for i, v in enumerate(columns[c]):
            row[v] = i - (n - 1) / 2
    bottom = max(row.values(), default=0)
    for i, v in enumerate(sorted(long_topics, key=lambda v: col[v])):
        row[v] = bottom + 1 + i
    pos = {v: (col[v] * COL_W, -row[v] * ROW_H) for v in vertices}
    return pos, succ, back, set(long_topics)


def draw_graphs(graphs: list[Graph], title: str | None = None):
    """Draw one or more graphs in a new figure and return it."""
    groups: dict = {}
    for g in graphs:
        groups.setdefault(_signature(g), []).append(g)

    fig, ax = plt.subplots()
    ax.set_axis_off()
    y0, xmax = 0.0, 0.0
    for gs in groups.values():
        g = gs[0]
        pos, succ, back, long_topics = _layout(g)
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
            ax.text(x, y + 0.11, _short(name, g.namespace), ha="center", va="center", fontsize=8,
                    family="monospace", zorder=4)
            ax.text(x, y - 0.17, f"{t.msg_type.__name__}{rate}", ha="center", va="center",
                    fontsize=6.5, color="0.35", zorder=4)

    half, gap = BOX_W / 2, (COL_W - BOX_W) / 2
    n_back = 0
    for v, ws in succ.items():
        for w in ws:
            (x1, y1), (x2, y2) = P[v], P[w]
            if (v, w) in back:
                # Up from the source, along a lane above everything, down into the target.
                n_back += 1
                lane = top + 0.3 + 0.3 * n_back
                _arrow(ax, [(x1, y1 + BOX_H / 2), (x1, lane), (x2, lane), (x2, y2 + BOX_H / 2)],
                       color="tab:red", alpha=0.75, z=1)
            elif w in long_topics:
                # Out of the publisher's right side, down to the lane, across.
                xa = x1 + half + gap * 0.6
                _arrow(ax, [(x1 + half, y1), (xa, y1), (xa, y2), (x2 - half, y2)])
            elif v in long_topics:
                # A lane topic feeds its reader in the next column.
                xa = x2 - half - gap * 0.5
                _arrow(ax, [(x1 + half, y1), (xa, y1), (xa, y2), (x2 - half, y2)])
            else:
                xa = x1 + half + gap
                _arrow(ax, [(x1 + half, y1), (xa, y1), (xa, y2), (x2 - half, y2)])


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
