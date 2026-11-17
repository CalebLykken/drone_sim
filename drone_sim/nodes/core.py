"""ROS-style nodes and topics, run deterministically in simulation time.

A `Graph` holds `Node`s. In `setup()` each node creates publishers,
subscriptions, and timers, much like rclpy:

    class Doubler(Node):
        def setup(self):
            self.out = self.create_publisher("doubled", Number)
            self.create_subscription("number", Number, self.on_number)

        def on_number(self, msg):
            self.out.publish(Number(self.now, 2 * msg.value))

Topics are typed by message class. Relative topic names ("imu") live in the
graph's namespace ("/drone0/imu"); names starting with "/" are global. Each
node can remap topic names (`remap={"imu": "imu_filtered"}`) before
namespacing, like ROS remapping.

Differences from ROS, on purpose:
- Time is simulation time. `Graph.spin_until(t)` fires every timer due up to
  t in time order, so runs are exactly reproducible.
- Publishing delivers to subscribers immediately, in subscription order (like
  intra-process communication with no queueing delay).
- Timers due at the same instant fire in the order the nodes were added, so
  list a stack like a pipeline: sensors, then estimators, then controllers.
"""

from __future__ import annotations

import heapq
import inspect
import itertools
import re
from dataclasses import dataclass, field
from typing import Any, Callable

from ..dynamics import DroneParams

_EPS = 1e-9


def _snake(name: str) -> str:
    return re.sub(r"(?<!^)(?=[A-Z])", "_", name).lower()


@dataclass
class Topic:
    name: str
    msg_type: type
    publishers: list["Node"] = field(default_factory=list)
    subscriptions: list["Subscription"] = field(default_factory=list)
    count: int = 0
    first_stamp: float | None = None
    last: Any = None

    @property
    def last_stamp(self) -> float | None:
        return getattr(self.last, "stamp", None)

    def rate(self) -> float | None:
        """Measured publish rate in Hz (like `ros2 topic hz`), once known."""
        if self.count < 2 or self.last_stamp is None or self.last_stamp <= self.first_stamp:
            return None
        return (self.count - 1) / (self.last_stamp - self.first_stamp)


class Publisher:
    def __init__(self, node: "Node", topic: Topic):
        self.node = node
        self.topic = topic

    def publish(self, msg) -> None:
        topic = self.topic
        if not isinstance(msg, topic.msg_type):
            raise TypeError(f"{self.node.name}: {topic.name} carries {topic.msg_type.__name__}, "
                            f"not {type(msg).__name__}")
        if topic.count == 0:
            topic.first_stamp = getattr(msg, "stamp", None)
        topic.count += 1
        topic.last = msg
        for sub in topic.subscriptions:
            sub.callback(msg)


@dataclass
class Subscription:
    node: "Node"
    topic: Topic
    callback: Callable[[Any], None]


@dataclass
class Timer:
    node: "Node"
    period: float
    callback: Callable[[], None]
    start: float = 0.0
    k: int = 0  # number of times fired; next fire is start + k * period (no drift)

    @property
    def next_time(self) -> float:
        return self.start + self.k * self.period


class Node:
    """Base class for nodes. Override `setup()` to create publishers,
    subscriptions, and timers; it runs every time the graph (re)starts, so
    initialize the node's state there too."""

    kind = "node"  # used to group and color nodes in the graph view

    def __new__(cls, *args, **kwargs):
        # Remember how the node was constructed, so tools (the stack editor)
        # can show its settings and write it back out. See constructor_args().
        node = super().__new__(cls)
        node._ctor = (args, kwargs)
        return node

    def __init__(self, name: str | None = None, *, remap: dict[str, str] | None = None, **params):
        self.name = name or _snake(type(self).__name__)
        self.remap = dict(remap or {})
        self.params = params
        self.graph: Graph | None = None

    # --- Overridable ---

    def setup(self) -> None:
        """Create publishers, subscriptions, and timers here."""

    # --- API for subclasses ---

    @property
    def now(self) -> float:
        """Current simulation time."""
        return self.graph.time

    @property
    def drone_params(self) -> DroneParams:
        """Physical parameters of the drone this graph flies."""
        return self.graph.drone_params

    @property
    def namespace(self) -> str:
        return self.graph.namespace

    def param(self, name: str, default=None):
        return self.params.get(name, default)

    def resolve(self, topic: str) -> str:
        """Full topic name after remapping and namespacing."""
        topic = self.remap.get(topic, topic)
        if topic.startswith("/"):
            return topic
        ns = self.graph.namespace.rstrip("/")
        return f"{ns}/{topic}" if ns else f"/{topic}"

    def create_publisher(self, topic: str, msg_type: type) -> Publisher:
        t = self.graph._topic(self.resolve(topic), msg_type, self)
        t.publishers.append(self)
        return Publisher(self, t)

    def create_subscription(self, topic: str, msg_type: type, callback: Callable[[Any], None]) -> Subscription:
        t = self.graph._topic(self.resolve(topic), msg_type, self)
        sub = Subscription(self, t, callback)
        t.subscriptions.append(sub)
        return sub

    def create_timer(self, period: float, callback: Callable[[], None]) -> Timer:
        if period <= 0:
            raise ValueError(f"{self.name}: timer period must be positive")
        timer = Timer(self, period, callback, start=self.graph.time)
        self.graph._add_timer(timer)
        return timer

    def constructor_args(self) -> dict[str, Any]:
        """The keyword arguments this node was constructed with (positional
        ones named), e.g. Imu(rate=250) -> {"rate": 250}."""
        args, kwargs = self._ctor
        if not args:
            return dict(kwargs)
        params = [p for p in constructor_params(type(self))
                  if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)]
        return {**{p.name: a for p, a in zip(params, args)}, **kwargs}

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.name!r})"


def constructor_params(cls: type) -> list[inspect.Parameter]:
    """Every named constructor parameter of a node class, following **kwargs
    up the class hierarchy (Imu -> Sensor -> Node), in declaration order."""
    seen: dict[str, inspect.Parameter] = {}
    for klass in cls.__mro__:
        init = klass.__dict__.get("__init__")
        if init is None:
            continue
        has_var_kw = False
        for p in list(inspect.signature(init).parameters.values())[1:]:  # skip self
            if p.kind is p.VAR_KEYWORD:
                has_var_kw = True
            elif p.kind is not p.VAR_POSITIONAL and p.name not in seen:
                seen[p.name] = p
        if not has_var_kw or klass is Node:
            break
    return list(seen.values())


class Graph:
    """A set of nodes wired together by topics, plus a sim-time executor."""

    def __init__(self, nodes: list[Node] = (), namespace: str = "", name: str | None = None):
        self.nodes: list[Node] = []
        self.namespace = namespace
        self.name = name or (namespace.strip("/") or "graph")
        self.drone_params = DroneParams()
        self.topics: dict[str, Topic] = {}
        self.time = 0.0
        self._timers: list[tuple[float, int, int, Timer]] = []
        self._order = itertools.count()
        for node in nodes:
            self.add(node)

    def add(self, node: Node) -> Node:
        if any(n.name == node.name for n in self.nodes):
            raise ValueError(f"duplicate node name {node.name!r}; pass name=... to tell them apart")
        if node.graph is not None and node.graph is not self:
            raise ValueError(f"{node.name} already belongs to another graph")
        node.graph = self
        self.nodes.append(node)
        return node

    def node(self, name: str) -> Node:
        for n in self.nodes:
            if n.name == name:
                return n
        raise KeyError(name)

    def start(self, drone_params: DroneParams | None = None, time: float = 0.0) -> None:
        """(Re)start: clear topics and timers, then run every node's setup()."""
        if drone_params is not None:
            self.drone_params = drone_params
        self.topics = {}
        self._timers = []
        self.time = time
        for node in self.nodes:
            node.setup()

    def spin_until(self, t: float) -> None:
        """Fire every timer due at or before t, in time order."""
        while self._timers and self._timers[0][0] <= t + _EPS:
            when, _, _, timer = heapq.heappop(self._timers)
            self.time = when
            timer.callback()
            timer.k += 1
            self._push(timer)
        self.time = max(self.time, t)

    # --- Introspection ---

    def edges(self) -> list[tuple[str, str, str]]:
        """(source, target, kind) for every publish (node -> topic) and
        subscribe (topic -> node) connection."""
        out = []
        for topic in self.topics.values():
            for pub in dict.fromkeys(topic.publishers):
                out.append((pub.name, topic.name, "pub"))
            for sub in topic.subscriptions:
                out.append((topic.name, sub.node.name, "sub"))
        return list(dict.fromkeys(out))

    def describe(self) -> str:
        """Human-readable summary, like `ros2 node list` + `ros2 topic list -v`."""
        lines = [f"graph {self.name} ({self.namespace or '/'})", "nodes:"]
        lines += [f"  {n.name} [{n.kind}] {type(n).__name__}" for n in self.nodes]
        lines.append("topics:")
        for t in self.topics.values():
            hz = t.rate()
            rate = f" {hz:.0f} Hz" if hz else ""
            pubs = ", ".join(dict.fromkeys(p.name for p in t.publishers)) or "-"
            subs = ", ".join(dict.fromkeys(s.node.name for s in t.subscriptions)) or "-"
            lines.append(f"  {t.name} <{t.msg_type.__name__}>{rate}  pub: {pubs}  sub: {subs}")
        return "\n".join(lines)

    def to_dot(self) -> str:
        """Graphviz DOT source (render with `dot -Tsvg`)."""
        return to_dot([self])

    def to_mermaid(self) -> str:
        """Mermaid flowchart source (renders on GitHub in ```mermaid blocks)."""
        return to_mermaid([self])

    # --- Internals ---

    def _topic(self, name: str, msg_type: type, node: Node) -> Topic:
        t = self.topics.get(name)
        if t is None:
            t = self.topics[name] = Topic(name, msg_type)
        elif t.msg_type is not msg_type:
            raise TypeError(f"{node.name}: {name} carries {t.msg_type.__name__}, "
                            f"not {msg_type.__name__}")
        return t

    def _add_timer(self, timer: Timer) -> None:
        timer.order = (self.nodes.index(timer.node), next(self._order))
        self._push(timer)

    def _push(self, timer: Timer) -> None:
        heapq.heappush(self._timers, (timer.next_time, *timer.order, timer))


# --- Text exports (no graphics dependencies) ------------------------------------

def _dot_id(s: str) -> str:
    return '"' + s.replace('"', r"\"") + '"'


def to_dot(graphs: list[Graph]) -> str:
    """DOT for one or more graphs, each drawn as a cluster."""
    lines = ["digraph drone_sim {", "  rankdir=LR;", '  node [fontname="Helvetica"];']
    shared: dict[str, Topic] = {}
    for i, g in enumerate(graphs):
        lines.append(f"  subgraph cluster_{i} {{")
        lines.append(f"    label={_dot_id(g.namespace or g.name)};")
        for n in g.nodes:
            lines.append(f"    {_dot_id(g.namespace + '::' + n.name)} "
                         f"[label={_dot_id(n.name)}, shape=ellipse];")
        for t in g.topics.values():
            if t.name.startswith(g.namespace + "/") or not g.namespace:
                lines.append(f"    {_dot_id(t.name)} [label={_dot_id(t.name + chr(10) + t.msg_type.__name__)}, shape=box];")
            else:
                shared[t.name] = t
        lines.append("  }")
    for t in shared.values():
        lines.append(f"  {_dot_id(t.name)} [label={_dot_id(t.name + chr(10) + t.msg_type.__name__)}, shape=box];")
    for g in graphs:
        for src, dst, kind in g.edges():
            a = g.namespace + "::" + src if kind == "pub" else src
            b = dst if kind == "pub" else g.namespace + "::" + dst
            lines.append(f"  {_dot_id(a)} -> {_dot_id(b)};")
    lines.append("}")
    return "\n".join(lines)


def to_mermaid(graphs: list[Graph]) -> str:
    """Mermaid flowchart for one or more graphs."""
    ids: dict[str, str] = {}

    def nid(key: str) -> str:
        return ids.setdefault(key, f"n{len(ids)}")

    lines = ["flowchart LR"]
    for g in graphs:
        lines.append(f'  subgraph {nid("cluster" + g.name)}["{g.namespace or g.name}"]')
        for n in g.nodes:
            lines.append(f'    {nid(g.namespace + "::" + n.name)}(["{n.name}"])')
        for t in g.topics.values():
            lines.append(f'    {nid(t.name)}["{t.name}<br/><small>{t.msg_type.__name__}</small>"]')
        lines.append("  end")
    for g in graphs:
        for src, dst, kind in g.edges():
            a = nid(g.namespace + "::" + src) if kind == "pub" else nid(src)
            b = nid(dst) if kind == "pub" else nid(g.namespace + "::" + dst)
            lines.append(f"  {a} --> {b}")
    return "\n".join(lines)
