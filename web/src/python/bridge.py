"""Bridge between the web app (web/) and the drone_sim package.

Runs inside Pyodide in a Web Worker. The browser only calls what is defined
here and only ever gets JSON strings back, so the JavaScript side never
touches numpy and drone_sim itself needs no changes to run in a browser.

Simulators are set up the way `python -m drone_sim` does it: drones start
side by side `spacing` apart and fly the scenario shifted by the same amount.
In live mode the first drone is piloted from the keyboard and the others fly
the `others` controller (the CLI uses controllers/pid.py for them).

This module also imports under plain CPython; tests/test_web_bridge.py uses
that to catch core changes that would break the web app.
"""

import json
import math
import sys
import tempfile
from pathlib import Path

import numpy as np

from drone_sim import DroneParams, DroneSpec, SimConfig, Simulator, State, load_controller, offset_target
from drone_sim.editor import EditError, StackEditor
from drone_sim.session import LiveSession
from drone_sim.sim import SCENARIOS

# The page re-loads controllers from source after every edit. A cached .pyc
# with the same size and timestamp as the edited file would hide the change.
sys.dont_write_bytecode = True

CONTROLLER_DIR = Path(tempfile.gettempdir()) / "drone_sim_web_controllers"
# The Stacks tab works on a copy of the repo layout, so a stack file's
# CONTROLLERS = Path(__file__).parent.parent / "controllers" finds the page's
# current controllers (including edits made in the browser).
STACK_ROOT = Path(tempfile.gettempdir()) / "drone_sim_web_stacks"
MAX_DRONES = 10
LIVE_LOG_KEEP = 2000  # control steps of log kept per drone while flying live


# --- Helpers ----------------------------------------------------------------------

def _num(x):
    """A JSON-safe scalar: JSON has no NaN or infinity, so those become null."""
    if isinstance(x, (bool, np.bool_)):
        return bool(x)
    if isinstance(x, (int, float, np.integer, np.floating)):
        x = float(x)
        return x if math.isfinite(x) else None
    return x


def _flat(a, digits: int = 5) -> list:
    """Row-major flat list of floats, rounded to keep the JSON small."""
    a = np.nan_to_num(np.asarray(a, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
    return np.round(a, digits).ravel().tolist()


def _params(p: DroneParams) -> dict:
    return {
        "mass": p.mass,
        "arm_length": p.arm_length,
        "max_thrust": p.max_thrust,
        "radius": p.radius,
        "yaw_signs": list(p.yaw_signs),
        "rotor_positions": p.rotor_positions.tolist(),
    }


def _load(ctrl: dict):
    """Write a controller's source to a file and load it with drone_sim's own
    loader, so the web app accepts exactly what the command line accepts."""
    name = Path(str(ctrl["name"])).name
    if not name.endswith(".py") or not name[:-3].isidentifier():
        raise ValueError(f"controller file names look like my_controller.py, not {name!r}")
    CONTROLLER_DIR.mkdir(parents=True, exist_ok=True)
    path = CONTROLLER_DIR / name
    path.write_text(str(ctrl["source"]))
    return load_controller(str(path))


def _build(opts: dict) -> Simulator:
    n = int(opts.get("drones", 1))
    if not 1 <= n <= MAX_DRONES:
        raise ValueError(f"drones must be between 1 and {MAX_DRONES}, not {n}")
    spacing = float(opts.get("spacing", 1.0))
    scenario_name = str(opts.get("scenario", "step"))
    if scenario_name not in SCENARIOS:
        raise ValueError(f"unknown scenario {scenario_name!r}; choose from {', '.join(SCENARIOS)}")
    scenario = SCENARIOS[scenario_name]
    pilot = opts["controller"]
    others = opts.get("others") or pilot

    specs = []
    for i in range(n):
        y = (i - (n - 1) / 2) * spacing
        specs.append(DroneSpec(
            _load(pilot if i == 0 else others),  # a fresh instance per drone
            initial_state=State(pos=np.array([0.0, y, 0.0])),
            target_fn=offset_target(scenario, (0.0, y, 0.0)),
        ))
    wind = [float(w) for w in opts.get("wind", ())][:3]
    config = SimConfig(
        duration=float(opts.get("duration", 12.0)),
        scenario=scenario_name,
        wind_mean=tuple(wind + [0.0] * (3 - len(wind))),
        wind_gust=float(opts.get("gust", 0.0)),
        sensor_noise=float(opts.get("noise", 0.0)),
        seed=int(opts.get("seed", 0)),
        crash_on_collision=not opts.get("ghost", False),
    )
    return Simulator(drones=specs, config=config)


def _collisions(sim: Simulator) -> list:
    return [{"t": c.t, "a": c.a, "b": c.b, "distance": c.distance} for c in sim.collisions]


# --- What the page calls ----------------------------------------------------------

def info() -> str:
    """Versions, scenario names, and the default drone geometry for drawing."""
    return json.dumps({
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "scenarios": list(SCENARIOS),
        "params": _params(DroneParams()),
    })


def check(controller_json: str) -> str:
    """Load a controller and fly it for two seconds on the hover scenario.
    Raises with the Python traceback if anything goes wrong."""
    ctrl = _load(json.loads(controller_json))
    sim = Simulator(ctrl, SimConfig(scenario="hover", duration=2.0))
    for _ in range(int(round(sim.config.duration / sim.control_dt))):
        sim.step()
    metrics = sim.result().metrics()
    return json.dumps({
        "class": type(ctrl).__name__,
        "metrics": {k: _num(v) for k, v in metrics.items()},
    })


class Playback:
    """A fixed-duration run, advanced in chunks so the page can show progress
    (and stay responsive) while it computes."""

    def __init__(self, opts_json: str):
        self.sim = _build(json.loads(opts_json))
        self.steps = int(round(self.sim.config.duration / self.sim.control_dt))
        if self.steps < 2:
            raise ValueError("duration is too short to simulate")

    def advance(self, n: int) -> float:
        """Step up to n control periods. Returns the fraction done."""
        for _ in range(min(int(n), self.steps - self.sim.k)):
            self.sim.step()
        return self.sim.k / self.steps

    def result(self) -> str:
        sim = self.sim
        results = sim.results()
        first = results[0]
        return json.dumps({
            "scenario": sim.config.scenario,
            "dt": sim.control_dt,
            "t": _flat(first.t, 4),
            "wind": _flat(first.wind, 3),
            "params": _params(sim.params),
            "collisions": _collisions(sim),
            "drones": [{
                "name": r.name,
                "pos": _flat(r.pos),
                "quat": _flat(r.states[:, 6:10]),
                "euler": _flat(r.euler),
                "target": _flat(r.targets),
                "thrust": _flat(r.thrusts, 3),
                "metrics": {k: _num(v) for k, v in r.metrics().items()},
                "crashed_at": _num(r.crashed_at),
            } for r in results],
        })


class Live:
    """Real-time keyboard flight around drone_sim's LiveSession. The page
    forwards key events here and calls `tick` with the wall time elapsed."""

    def __init__(self, opts_json: str):
        self.sim = _build(json.loads(opts_json))
        self.session = LiveSession(self.sim)

    def key(self, key: str, down: bool) -> None:
        if down:
            self.session.key_down(key)
        else:
            self.session.key_up(key)

    def release_all(self) -> None:
        self.session.release_all()

    def reset(self) -> None:
        self.session.reset()

    def tick(self, elapsed: float) -> str:
        self.session.tick(float(elapsed))
        self._trim_logs()
        return self.snapshot()

    def _trim_logs(self) -> None:
        # The Simulator logs every step so it can build a SimResult. Live
        # flight never asks for one, so cap the log to keep memory flat on
        # long flights. (Uses Drone._log, so it quietly does nothing if that
        # ever changes.)
        for d in self.sim.drones:
            log = getattr(d, "_log", None)
            if isinstance(log, dict) and len(log.get("t", ())) > 2 * LIVE_LOG_KEEP:
                for values in log.values():
                    del values[:-LIVE_LOG_KEEP]

    def snapshot(self) -> str:
        sim, st = self.sim, self.session.sticks
        return json.dumps({
            "t": sim.t,
            "sticks": {
                "throttle": float(st.throttle),
                "roll": float(st.roll),
                "pitch": float(st.pitch),
                "yaw": float(st.yaw),
            },
            "wind": _flat(sim.wind, 3),
            "collisions": _collisions(sim),
            "drones": [{
                "name": d.name,
                "pos": _flat(d.state.pos),
                "quat": _flat(d.state.quat),
                "vel": _flat(d.state.vel),
                "euler": _flat(d.state.euler),
                "target": _flat(d.target),
                "thrust": _flat(d.thrust, 3),
                "crashed": d.crashed,
            } for d in sim.drones],
        })


# --- Stacks tab --------------------------------------------------------------------

_stack: StackEditor | None = None


def _file_name(name: str) -> str:
    name = Path(str(name)).name
    if not name.endswith(".py") or not name[:-3].isidentifier():
        raise ValueError(f"file names look like my_file.py, not {name!r}")
    return name


def stack_open(request_json: str) -> str:
    """Open a stack for editing: {"name", "source", "controllers": [{"name", "source"}]}.
    Returns the editor state plus the catalog of node types that can be added."""
    global _stack
    req = json.loads(request_json)
    controllers = STACK_ROOT / "controllers"
    controllers.mkdir(parents=True, exist_ok=True)
    for c in req.get("controllers", []):
        (controllers / _file_name(c["name"])).write_text(str(c["source"]))
    stacks = STACK_ROOT / "stacks"
    stacks.mkdir(parents=True, exist_ok=True)
    path = stacks / _file_name(req["name"])
    path.write_text(str(req["source"]))
    try:
        _stack = StackEditor(path, root=STACK_ROOT)
    except EditError as e:
        _stack = None
        return json.dumps({"ok": False, "error": str(e)})
    return json.dumps({"ok": True, **_stack.state(), "catalog": _stack.catalog()})


def stack_edit(request_json: str) -> str:
    """Change the open stack. {"op": "add", "key", "params"}, {"op": "update",
    "name", "params"}, {"op": "remove", "name"}, or {"op": "test", "scenario",
    "duration"}. Edits that would break the stack come back as {"ok": false,
    "error"} and leave it unchanged."""
    if _stack is None:
        raise RuntimeError("no stack is open")
    req = json.loads(request_json)
    op = req.get("op")
    try:
        if op == "add":
            state = _stack.add(str(req["key"]), dict(req.get("params", {})))
        elif op == "update":
            state = _stack.update(str(req["name"]), dict(req.get("params", {})))
        elif op == "remove":
            state = _stack.remove(str(req["name"]))
        elif op == "test":
            state = _stack.test_flight(str(req.get("scenario", "square")), float(req.get("duration", 6.0)))
        else:
            raise ValueError(f"unknown stack operation {op!r}")
    except EditError as e:
        return json.dumps({"ok": False, "error": str(e)})
    return json.dumps({"ok": True, **state})
