"""JSON views of a stack for the web app's Stacks tab (web/src/python/bridge.py).

`StackEditor` holds one open stack file and answers in plain dicts: the
nodes and their editable properties, the graph laid out like `--graph`
(drone_sim/nodes/layout.py), warnings about loose ends, and the file's
current source after every change.
"""

from __future__ import annotations

import math
from pathlib import Path

from ..nodes import Graph
from ..nodes.layout import BOX_H, KIND_COLORS, layout, route, short
from .model import EditError, StackDocument

# Sim outputs a stack is free to ignore.
OPTIONAL_TOPICS = {"ground_truth", "setpoint", "sticks", "/swarm"}


def graph_json(g: Graph) -> dict:
    """Vertices with positions (y up) and routed edges, ready to draw."""
    pos, succ, back, long_topics = layout(g)
    top = max(y for _, y in pos.values()) + BOX_H / 2
    nodes = {n.name: n for n in g.nodes}
    vertices = []
    for (kind, name), (x, y) in pos.items():
        if kind == "node":
            n = nodes[name]
            vertices.append({
                "id": f"node:{name}", "type": "node", "name": name, "label": name,
                "sub": type(n).__name__, "kind": n.kind,
                "color": KIND_COLORS.get(n.kind, KIND_COLORS["node"]),
                "editable": n.kind != "sim", "x": x, "y": y,
            })
        else:
            t = g.topics[name]
            hz = t.rate()
            vertices.append({
                "id": f"topic:{name}", "type": "topic", "name": name,
                "label": short(name, g.namespace),
                "sub": t.msg_type.__name__ + (f"  {hz:.0f} Hz" if hz else ""),
                "msg_type": t.msg_type.__name__, "rate": hz,
                "publishers": list(dict.fromkeys(p.name for p in t.publishers)),
                "subscribers": list(dict.fromkeys(s.node.name for s in t.subscriptions)),
                "x": x, "y": y,
            })
    edges = [{"points": [list(p) for p in pts], "feedback": fb}
             for pts, fb in route(pos, succ, back, long_topics, top)]
    return {"vertices": vertices, "edges": edges}


def warnings_for(g: Graph) -> list[str]:
    """Topics published with no reader, or read with nothing publishing them."""
    out = []
    for t in g.topics.values():
        rel = short(t.name, g.namespace)
        pubs = ", ".join(dict.fromkeys(p.name for p in t.publishers))
        subs = ", ".join(dict.fromkeys(s.node.name for s in t.subscriptions))
        if t.publishers and not t.subscriptions and rel not in OPTIONAL_TOPICS:
            out.append(f"Nothing reads {rel} (published by {pubs}).")
        if t.subscriptions and not t.publishers:
            out.append(f"Nothing publishes {rel}, which {subs} needs.")
    return out


def _clean(x):
    if isinstance(x, float) and not math.isfinite(x):
        return None
    if isinstance(x, dict):
        return {k: _clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_clean(v) for v in x]
    if hasattr(x, "item") and not isinstance(x, (str, bytes)):  # numpy scalars
        return _clean(x.item())
    return x


class StackEditor:
    def __init__(self, path: str | Path, root: str | Path):
        self.doc = StackDocument(path, root=root)
        self.flight: dict | None = None
        self.flight_graph: Graph | None = None  # has measured topic rates

    def state(self) -> dict:
        doc = self.doc
        g = self.flight_graph or doc.graph()
        return _clean({
            "nodes": doc.describe(),
            "graph": graph_json(g),
            "warnings": warnings_for(g),
            "source": doc.source,
            "flight": self.flight,
        })

    def catalog(self) -> list[dict]:
        return _clean(self.doc.catalog())

    def _changed(self) -> dict:
        self.doc.save()  # rewrite the stack() function in the working copy
        self.flight = self.flight_graph = None  # rates measured on the old stack
        return self.state()

    def add(self, key: str, params: dict) -> dict:
        added = self.doc.add(key, params)
        return {**self._changed(), "selected": added}

    def update(self, name: str, params: dict) -> dict:
        renamed = self.doc.update(name, params)
        return {**self._changed(), "selected": renamed}

    def remove(self, name: str) -> dict:
        self.doc.remove(name)
        return self._changed()

    def test_flight(self, scenario: str, duration: float) -> dict:
        result, graph = self.doc.test_flight(scenario, duration)
        self.flight_graph = graph
        self.flight = {"scenario": scenario, "duration": duration,
                       "metrics": result.metrics(), "class": "StackController"}
        return self.state()


__all__ = ["EditError", "StackEditor", "graph_json", "warnings_for"]
