"""Graph layout shared by every node-graph view (matplotlib's --graph and
the web app's Stacks tab). Pure Python: no drawing here.

Coordinates are in abstract units: columns are COL_W apart, rows ROW_H, and
each node or topic box is BOX_W x BOX_H, centered on its position. y grows
upward.
"""

from __future__ import annotations

from collections import defaultdict

from .core import Graph

KIND_COLORS = {
    "sim": "#d9d9d9",
    "sensor": "#a6cee3",
    "estimator": "#b2df8a",
    "controller": "#fdbf6f",
    "planner": "#cab2d6",
    "node": "#ffffb3",
}
COL_W, ROW_H = 3.2, 1.1
BOX_W, BOX_H = 2.5, 0.72


def short(topic: str, namespace: str) -> str:
    """Topic name relative to the namespace ("/drone0/imu" -> "imu")."""
    prefix = namespace.rstrip("/") + "/"
    return topic[len(prefix):] if namespace and topic.startswith(prefix) else topic


def signature(g: Graph):
    """Identical for graphs with the same structure (so they can be drawn once)."""
    return (
        tuple((type(n).__name__, n.name) for n in g.nodes),
        tuple(sorted((short(s, g.namespace), short(d, g.namespace), k) for s, d, k in g.edges())),
    )


def layout(g: Graph):
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


def route(P, succ, back, long_topics, top: float) -> list[tuple[list[tuple[float, float]], bool]]:
    """Orthogonal polylines for every edge, as (points, is_feedback).
    `P` maps vertices to positions; `top` is the y of the highest box edge
    (feedback lanes go above it)."""
    half, gap = BOX_W / 2, (COL_W - BOX_W) / 2
    out, n_back = [], 0
    for v, ws in succ.items():
        for w in ws:
            (x1, y1), (x2, y2) = P[v], P[w]
            if (v, w) in back:
                # Up from the source, along a lane above everything, down into the target.
                n_back += 1
                lane = top + 0.3 + 0.3 * n_back
                out.append(([(x1, y1 + BOX_H / 2), (x1, lane), (x2, lane), (x2, y2 + BOX_H / 2)], True))
            elif w in long_topics:
                # Out of the publisher's right side, down to the lane, across.
                xa = x1 + half + gap * 0.6
                out.append(([(x1 + half, y1), (xa, y1), (xa, y2), (x2 - half, y2)], False))
            elif v in long_topics:
                # A lane topic feeds its reader in the next column.
                xa = x2 - half - gap * 0.5
                out.append(([(x1 + half, y1), (xa, y1), (xa, y2), (x2 - half, y2)], False))
            else:
                xa = x1 + half + gap
                out.append(([(x1 + half, y1), (xa, y1), (xa, y2), (x2 - half, y2)], False))
    return out
