"""The stack being edited: load a stack file, change it, write it back.

A stack is held as a list of NodeSpec (node class + constructor keyword
arguments). Properties shown in the editor come from each class's
constructor signature, so any node, including ones defined in the stack
file itself, gets an editor form automatically.

Saving rewrites only the `stack()` function of the file (plus any imports
it now needs); everything else in the file, such as custom node classes,
constants, and comments, is kept.
"""

from __future__ import annotations

import ast
import importlib.util
import inspect
import os
import pathlib
import sys
import textwrap
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .. import nodes as builtin_nodes
from ..dynamics import DroneParams
from ..nodes import ControllerNode, Graph, Node, SimBridge, StackController
from ..nodes.core import constructor_params

ABSTRACT = {Node, builtin_nodes.Sensor, builtin_nodes.InertialEstimator, SimBridge}
KIND_ORDER = ["sensor", "estimator", "controller", "planner", "node"]


class EditError(ValueError):
    """A change the editor can't make; the message is shown to the user."""


@dataclass
class NodeSpec:
    cls: type
    args: dict[str, Any] = field(default_factory=dict)

    def build(self) -> Node:
        return self.cls(**self.args)

    @property
    def name(self) -> str:
        return self.build().name


# --- Parameter types ---------------------------------------------------------------

def _param_type(p: inspect.Parameter) -> str:
    ann = str(p.annotation) if p.annotation is not inspect.Parameter.empty else ""
    d = p.default
    if p.name == "remap" or isinstance(d, dict) or "dict" in ann:
        return "mapping"
    if "Controller" in ann:
        return "controller"
    if isinstance(d, bool) or ann == "bool":
        return "bool"
    if isinstance(d, (tuple, list, np.ndarray)) or "tuple" in ann:
        return "vector"
    if isinstance(d, int) or ann.startswith("int"):
        return "integer"
    if isinstance(d, float) or "float" in ann:
        return "number"
    if isinstance(d, Path) or "Path" in ann:
        return "path"
    return "string"


def _to_json(value, ptype: str, root: Path):
    if value is inspect.Parameter.empty:
        return None
    if isinstance(value, Path) or (ptype in ("controller", "path") and isinstance(value, str)):
        p = Path(value)
        try:
            return os.path.relpath(p.resolve(), root)
        except ValueError:
            return str(p)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, Node) or not isinstance(value, (int, float, str, bool, list, dict, type(None))):
        return repr(value)
    return value


def _from_json(value, ptype: str, p: inspect.Parameter, root: Path):
    if value is None or value == "":
        if p.default is inspect.Parameter.empty:
            raise EditError(f"{p.name} is required")
        return p.default
    try:
        if ptype == "number":
            return float(value)
        if ptype == "integer":
            return int(value)
        if ptype == "bool":
            return bool(value)
        if ptype == "vector":
            return tuple(float(v) for v in value)
        if ptype == "mapping":
            return {str(k): str(v) for k, v in dict(value).items() if str(k)}
        if ptype in ("controller", "path"):
            path = (root / str(value)).resolve()
            if not path.is_file():
                raise EditError(f"{p.name}: no such file {value}")
            return path
        return str(value)
    except (TypeError, ValueError) as e:
        raise EditError(f"{p.name}: {e}") from None


def _same(a, b) -> bool:
    try:
        return bool(np.all(np.asarray(a) == np.asarray(b))) if not isinstance(a, dict) else a == b
    except Exception:
        return a is b


# --- The document ------------------------------------------------------------------

class StackDocument:
    def __init__(self, path: str | Path | None = None, root: str | Path | None = None):
        self.root = Path(root or Path.cwd()).resolve()
        self.path = Path(path).resolve() if path else None
        self.source: str | None = None
        self.module = None
        self.specs: list[NodeSpec] = []
        self.dirty = False
        if self.path is not None:
            self._load()

    # Loading

    def _load(self) -> None:
        if not self.path.is_file():
            raise EditError(f"no such file: {self.path}")
        self.source = self.path.read_text()
        name = f"_drone_stack_{self.path.stem}_{abs(hash(self.path))}"
        spec = importlib.util.spec_from_file_location(name, self.path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        if not callable(getattr(module, "stack", None)):
            raise EditError(f"{self.path.name} has no stack() function")
        self.module = module
        self.specs = [NodeSpec(type(n), n.constructor_args()) for n in module.stack()]

    # Catalog of node types

    def node_classes(self) -> dict[str, type]:
        classes = {}
        for name in builtin_nodes.__all__:
            obj = getattr(builtin_nodes, name)
            if inspect.isclass(obj) and issubclass(obj, Node) and obj not in ABSTRACT:
                classes[f"drone_sim.nodes.{name}"] = obj
        if self.module is not None:
            for name, obj in vars(self.module).items():
                if inspect.isclass(obj) and issubclass(obj, Node) and obj not in ABSTRACT \
                        and obj.__module__ == self.module.__name__:
                    classes[f"{self.path.name}:{name}"] = obj
        return classes

    def class_key(self, cls: type) -> str:
        for key, c in self.node_classes().items():
            if c is cls:
                return key
        return f"{cls.__module__}.{cls.__qualname__}"

    def catalog(self) -> list[dict]:
        out = []
        for key, cls in self.node_classes().items():
            out.append({
                "key": key,
                "class": cls.__name__,
                "kind": cls.kind,
                "doc": _first_paragraph(cls),
                "local": ":" in key,
                "params": self.params(cls, {}),
            })
        out.sort(key=lambda c: (KIND_ORDER.index(c["kind"]) if c["kind"] in KIND_ORDER else 99, c["class"]))
        return out

    def params(self, cls: type, args: dict) -> list[dict]:
        out = []
        for p in constructor_params(cls):
            ptype = _param_type(p)
            default = p.default
            if p.name == "rate" and default is None and hasattr(cls, "default_rate"):
                default = cls.default_rate
            if p.name == "name" and default is None:
                default = _snake(cls.__name__)
            item = {
                "name": p.name,
                "type": ptype,
                "value": _to_json(args.get(p.name, p.default), ptype, self.root),
                "default": _to_json(default, ptype, self.root),
                "required": p.default is inspect.Parameter.empty,
            }
            if ptype == "controller":
                item["choices"] = [os.path.relpath(f, self.root)
                                   for f in sorted((self.root / "controllers").glob("*.py"))]
            if ptype == "vector":
                item["length"] = len(default) if isinstance(default, (tuple, list)) else 3
            out.append(item)
        return out

    # Editing

    def _spec(self, name: str) -> NodeSpec:
        for s in self.specs:
            if s.name == name:
                return s
        raise EditError(f"no node named {name!r}")

    def _parse_args(self, cls: type, values: dict, current: dict | None = None) -> dict:
        """Constructor arguments from form values. Fields not in `values` keep
        their `current` value; values equal to the default are left out."""
        args = dict(current or {})
        for p in constructor_params(cls):
            if p.name not in values:
                continue
            ptype = _param_type(p)
            v = _from_json(values[p.name], ptype, p, self.root)
            if ptype == "mapping" and not v and p.default is None:
                v = None  # an empty remap is no remap
            if p.default is inspect.Parameter.empty or not _same(v, p.default):
                args[p.name] = v
            else:
                args.pop(p.name, None)
        missing = [p.name for p in constructor_params(cls)
                   if p.default is inspect.Parameter.empty and p.name not in args]
        if missing:
            raise EditError(f"missing required: {', '.join(missing)}")
        return args

    def _try(self, specs: list[NodeSpec]) -> None:
        """Raise EditError if `specs` don't make a valid stack."""
        try:
            build_graph(specs)
        except EditError:
            raise
        except Exception as e:  # anything a node's constructor or setup() raises
            raise EditError(f"{type(e).__name__}: {e}") from None

    def add(self, key: str, values: dict) -> str:
        cls = self.node_classes().get(key)
        if cls is None:
            raise EditError(f"unknown node type {key}")
        args = self._parse_args(cls, values)
        spec = NodeSpec(cls, args)
        taken = {s.name for s in self.specs} | {"sim"}
        if spec.name in taken:  # second IMU -> imu_2
            base, i = spec.name, 2
            while f"{base}_{i}" in taken:
                i += 1
            spec.args["name"] = f"{base}_{i}"
        # New sensors go after the existing sensors, everything else at the end,
        # keeping the stack in pipeline order (see drone_sim/nodes/core.py).
        if cls.kind == "sensor":
            index = max((i + 1 for i, s in enumerate(self.specs) if s.cls.kind == "sensor"), default=0)
        else:
            index = len(self.specs)
        specs = self.specs[:index] + [spec] + self.specs[index:]
        self._try(specs)
        self.specs, self.dirty = specs, True
        return spec.name

    def update(self, name: str, values: dict) -> str:
        old = self._spec(name)
        spec = NodeSpec(old.cls, self._parse_args(old.cls, values, current=old.args))
        if spec.name != name and spec.name in {s.name for s in self.specs} | {"sim"}:
            raise EditError(f"a node named {spec.name!r} already exists")
        specs = [spec if s is old else s for s in self.specs]
        self._try(specs)
        self.specs, self.dirty = specs, True
        return spec.name

    def remove(self, name: str) -> None:
        old = self._spec(name)
        specs = [s for s in self.specs if s is not old]
        self._try(specs)
        self.specs, self.dirty = specs, True

    def describe(self) -> list[dict]:
        return [{
            "name": s.name,
            "key": self.class_key(s.cls),
            "class": s.cls.__name__,
            "kind": s.cls.kind,
            "doc": _first_paragraph(s.cls),
            "params": self.params(s.cls, s.args),
        } for s in self.specs]

    # Saving

    def save(self, path: str | Path | None = None) -> Path:
        target = Path(path).resolve() if path else self.path
        if target is None:
            raise EditError("choose a file name to save to")
        if target.suffix != ".py":
            raise EditError("stack files end in .py")
        if self.root not in target.parents:
            raise EditError(f"save inside {self.root}")
        local = [s.cls.__name__ for s in self.specs if self._is_local(s.cls)]
        if local and target != self.path:
            raise EditError(f"{', '.join(sorted(set(local)))} are defined in {self.path.name}; "
                            f"save to that file, or move them to a module first")
        if target == self.path and not self.dirty:
            return target  # nothing changed; leave the file alone
        text = self.to_source(target)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text)
        self.path, self.source, self.dirty = target, text, False
        return target

    def _is_local(self, cls: type) -> bool:
        return self.module is not None and cls.__module__ == self.module.__name__

    def to_source(self, target: Path | None = None) -> str:
        target = target or self.path
        uses_path = False
        imports: dict[str, list[str]] = {}
        entries = []
        for s in self.specs:
            cls = s.cls
            if not self._is_local(cls):
                module = "drone_sim.nodes" if getattr(builtin_nodes, cls.__name__, None) is cls else cls.__module__
                if self.module is None or vars(self.module).get(cls.__name__) is not cls:
                    imports.setdefault(module, [])
                    if cls.__name__ not in imports[module]:
                        imports[module].append(cls.__name__)
            parts = []
            for p in constructor_params(cls):
                if p.name not in s.args:
                    continue
                code, is_path = _format(s.args[p.name], target.parent, self._path_constants(target))
                uses_path |= is_path
                parts.append(f"{p.name}={code}")
            entries.append(f"{cls.__name__}({', '.join(parts)}),")

        func = _stack_function(entries, self._stack_docstring())
        if uses_path and (self.module is None or vars(self.module).get("Path") is not pathlib.Path):
            imports.setdefault("pathlib", []).append("Path")
        import_lines = [f"from {m} import {', '.join(sorted(names))}" for m, names in sorted(imports.items())]

        if self.source is None or target != self.path:
            header = '"""Stack made with the drone_sim stack editor."""\n\n'
            return header + "\n".join(import_lines) + ("\n\n\n" if import_lines else "") + func

        tree = ast.parse(self.source)
        lines = self.source.splitlines(keepends=True)
        fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "stack")
        # Edit bottom-up so earlier line numbers stay valid.
        edits = [(fn.lineno - 1, fn.end_lineno, [func])]
        extra = []
        for module, names in imports.items():
            existing = next((n for n in tree.body if isinstance(n, ast.ImportFrom) and n.module == module
                             and not any(a.asname for a in n.names)), None)
            if existing is None:
                extra.append(f"from {module} import {', '.join(sorted(names))}\n")
            else:
                merged = sorted({a.name for a in existing.names} | set(names))
                edits.append((existing.lineno - 1, existing.end_lineno, [f"from {module} import {', '.join(merged)}\n"]))
        if extra:
            last_import = max((n.end_lineno for n in tree.body if isinstance(n, (ast.Import, ast.ImportFrom))),
                              default=_docstring_end(tree))
            edits.append((last_import, last_import, extra))
        for start, end, repl in sorted(edits, key=lambda e: e[0], reverse=True):
            lines[start:end] = repl
        return "".join(lines)

    def _path_constants(self, target: Path) -> dict[str, Path]:
        """Module-level Path constants (e.g. CONTROLLERS) to write paths with,
        when saving back to the file that defines them."""
        if self.module is None or target != self.path:
            return {}
        return {k: v.resolve() for k, v in vars(self.module).items()
                if isinstance(v, Path) and k.isupper()}

    def _stack_docstring(self) -> str | None:
        if self.source is None:
            return None
        tree = ast.parse(self.source)
        fn = next((n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "stack"), None)
        return ast.get_docstring(fn) if fn else None

    # Running

    def graph(self) -> Graph:
        return build_graph(self.specs)

    def test_flight(self, scenario: str = "square", duration: float = 6.0):
        from ..sim import SimConfig, run

        if self.specs and not any(issubclass(s.cls, ControllerNode) for s in self.specs):
            raise EditError("add a controller node to fly")
        specs = list(self.specs)
        controller = StackController(lambda: [s.build() for s in specs], label="editor")
        result = run(controller, SimConfig(scenario=scenario, duration=duration))
        return result, controller.graph


def build_graph(specs: list[NodeSpec]) -> Graph:
    g = Graph([SimBridge(), *(s.build() for s in specs)], namespace="/drone0", name="drone0")
    g.start(DroneParams())
    return g


def _format(value, base: Path, constants: dict[str, Path] | None = None) -> tuple[str, bool]:
    """Python source for a value. Paths are written relative to a matching
    module constant (CONTROLLERS / "pid.py") or to the stack file."""
    if isinstance(value, Path):
        for name, root in (constants or {}).items():
            if root in value.resolve().parents:
                return f"{name} / {_literal(str(value.resolve().relative_to(root)))}", False
        rel = os.path.relpath(value, base)
        return f"Path(__file__).resolve().parent / {_literal(rel)}", True
    if isinstance(value, np.ndarray):
        value = tuple(value.tolist())
    if isinstance(value, (int, float, str, bool, tuple, list, dict, type(None))):
        return _literal(value), False
    raise EditError(f"can't write {value!r} to a stack file")


def _literal(v) -> str:
    """repr(), but 5.0 -> 5 and double-quoted strings, matching the repo style."""
    if isinstance(v, bool) or v is None:
        return repr(v)
    if isinstance(v, float):
        return str(int(v)) if v.is_integer() and abs(v) < 1e15 else repr(v)
    if isinstance(v, str):
        return '"' + v.replace("\\", "\\\\").replace('"', '\\"') + '"'
    if isinstance(v, tuple):
        inner = ", ".join(_literal(x) for x in v)
        return f"({inner},)" if len(v) == 1 else f"({inner})"
    if isinstance(v, list):
        return "[" + ", ".join(_literal(x) for x in v) + "]"
    if isinstance(v, dict):
        return "{" + ", ".join(f"{_literal(k)}: {_literal(x)}" for k, x in v.items()) + "}"
    return repr(v)


def _stack_function(entries: list[str], docstring: str | None) -> str:
    body = ""
    if docstring:
        body += textwrap.indent(f'"""{docstring}"""', "    ") + "\n"
    body += "    return [\n" + "".join(f"        {e}\n" for e in entries) + "    ]\n"
    return "def stack():\n" + body


def _docstring_end(tree: ast.Module) -> int:
    first = tree.body[0] if tree.body else None
    if isinstance(first, ast.Expr) and isinstance(getattr(first, "value", None), ast.Constant):
        return first.end_lineno
    return 0


def _first_paragraph(cls: type) -> str:
    doc = inspect.getdoc(cls) or ""
    if not doc or (cls.__doc__ is None):
        return ""
    return doc.split("\n\n")[0].replace("\n", " ")


def _snake(name: str) -> str:
    import re

    return re.sub(r"(?<!^)(?=[A-Z])", "_", name).lower()
