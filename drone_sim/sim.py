"""Simulation loop, scenarios, and controller loading."""

import importlib.util
import inspect
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np

from .controller import Controller, Neighbor
from .dynamics import DroneParams, State, motor_step, step

# Scenario functions may return (x, y, z) or (x, y, z, yaw); yaw defaults to 0.
Target = tuple[float, ...]


# --- Scenarios: functions of time returning the target --------------------------

def _hover(t: float) -> Target:
    return (0.0, 0.0, 1.0)


def _step(t: float) -> Target:
    return (0.0, 0.0, 1.0) if t < 3.0 else (2.0, 1.0, 2.0)


def _square(t: float) -> Target:
    corners = [(0.0, 0.0, 1.0), (2.0, 0.0, 1.5), (2.0, 2.0, 2.0), (0.0, 2.0, 1.5)]
    return corners[int(t // 3.0) % len(corners)]


def _figure8(t: float) -> Target:
    w = 2 * math.pi / 8.0
    return (2.0 * math.sin(w * t), 1.0 * math.sin(2 * w * t), 1.5 + 0.5 * math.sin(w * t))


def _helix(t: float) -> Target:
    w = 2 * math.pi / 6.0
    # Climb in a circle while turning to face the direction of travel.
    return (1.5 * math.cos(w * t) - 1.5, 1.5 * math.sin(w * t), 1.0 + 0.15 * t, w * t + math.pi / 2)


def _spin(t: float) -> Target:
    return (0.0, 0.0, 1.0, 0.0 if t < 3.0 else math.pi / 2)


SCENARIOS: dict[str, Callable[[float], Target]] = {
    "hover": _hover,
    "step": _step,
    "square": _square,
    "figure8": _figure8,
    "helix": _helix,
    "spin": _spin,
}


@dataclass
class SimConfig:
    duration: float = 10.0  # s
    physics_hz: float = 1000.0
    control_hz: float = 250.0
    scenario: str = "step"
    initial_state: State = field(default_factory=State)
    wind_mean: tuple[float, float, float] = (0.0, 0.0, 0.0)  # N, steady force
    wind_gust: float = 0.0  # N, std dev of gusts on each horizontal axis
    sensor_noise: float = 0.0  # std dev added to every measured state component
    seed: int = 0
    crash_on_collision: bool = True  # drones that touch lose their motors and fall


@dataclass
class SimResult:
    t: np.ndarray  # (N,)
    states: np.ndarray  # (N, 13) true state: pos(3), vel(3), quat(4), omega(3)
    euler: np.ndarray  # (N, 3) roll, pitch, yaw in radians
    targets: np.ndarray  # (N, 4) x, y, z, yaw
    commands: np.ndarray  # (N, 4) controller output, FL FR BR BL
    thrusts: np.ndarray  # (N, 4) actual rotor thrust after lag/saturation
    wind: np.ndarray  # (N, 3)
    params: DroneParams
    config: SimConfig
    name: str = "drone0"
    crashed_at: float | None = None  # sim time of a crash, if any

    @property
    def pos(self) -> np.ndarray:
        return self.states[:, 0:3]

    def metrics(self) -> dict[str, float]:
        err = np.linalg.norm(self.pos - self.targets[:, :3], axis=1)
        R22 = 1 - 2 * (self.states[:, 7] ** 2 + self.states[:, 8] ** 2)  # cos(tilt)
        tilt = np.arccos(np.clip(R22, -1, 1))
        return {
            "rms_pos_error_m": float(np.sqrt(np.mean(err**2))),
            "final_pos_error_m": float(err[-1]),
            "max_tilt_deg": float(np.degrees(tilt.max())),
            "flipped": bool(np.any(tilt > math.pi / 2)),
            "crashed": self.crashed_at is not None,
        }


# --- Controller loading ---------------------------------------------------------

def load_controller(spec: str) -> Controller:
    """Load a controller from "path/to/file.py" or "path/to/file.py:ClassName".

    Without a class name, the file must define exactly one Controller subclass.
    """
    path_str, _, class_name = spec.partition(":")
    path = Path(path_str).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"controller file not found: {path}")

    module_name = f"_drone_controller_{path.stem}"
    mod_spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(mod_spec)
    sys.modules[module_name] = module
    mod_spec.loader.exec_module(module)

    if class_name:
        cls = getattr(module, class_name, None)
        if cls is None:
            raise AttributeError(f"{path.name} has no class {class_name!r}")
    else:
        found = [
            obj
            for obj in vars(module).values()
            if inspect.isclass(obj)
            and issubclass(obj, Controller)
            and obj is not Controller
            and obj.__module__ == module_name
        ]
        if len(found) != 1:
            names = ", ".join(c.__name__ for c in found) or "none"
            raise ValueError(
                f"{path.name} must define exactly one Controller subclass "
                f"(found: {names}); use {path_str}:ClassName to pick one"
            )
        cls = found[0]
    return cls()


# --- Main loop ------------------------------------------------------------------

def _noisy(state: State, sigma: float, rng: np.random.Generator) -> State:
    a = state.as_array() + rng.normal(0, sigma, 13)
    a[6:10] /= np.linalg.norm(a[6:10])
    return State.from_array(a)


@dataclass
class DroneSpec:
    """How to set up one drone in a multi-drone Simulator. Anything left as
    None falls back to the SimConfig / scenario defaults."""

    controller: Controller
    params: DroneParams | None = None
    initial_state: State | None = None
    target_fn: Callable[[float], Target] | None = None
    name: str | None = None


@dataclass(frozen=True)
class Collision:
    t: float
    a: str  # drone names
    b: str
    distance: float  # m, center to center


class Drone:
    """One drone's runtime state and log inside a Simulator."""

    def __init__(self, spec: DroneSpec, name: str, config: SimConfig, default_target_fn):
        self.name = name
        self.controller = spec.controller
        self.params = spec.params or DroneParams()
        self.initial_state = spec.initial_state or config.initial_state
        self.target_fn = spec.target_fn or default_target_fn
        self.reset()

    def reset(self) -> None:
        self.state = State.from_array(self.initial_state.as_array())
        self.thrust = np.zeros(4)
        self.target = (0.0, 0.0, 0.0, 0.0)
        self.crashed_at: float | None = None
        self._log: dict[str, list] = {
            k: [] for k in ("t", "state", "euler", "target", "cmd", "thrust", "wind")
        }

    @property
    def crashed(self) -> bool:
        return self.crashed_at is not None

    def positions(self, last: int | None = None) -> np.ndarray:
        """(N, 3) logged positions, optionally only the most recent `last`."""
        states = self._log["state"] if last is None else self._log["state"][-last:]
        return np.array(states, dtype=float).reshape(-1, 13)[:, :3]

    def result(self, config: SimConfig) -> SimResult:
        log = {k: np.array(v, dtype=float) for k, v in self._log.items()}
        return SimResult(
            log["t"], log["state"], log["euler"], log["target"], log["cmd"],
            log["thrust"], log["wind"], self.params, config, self.name, self.crashed_at,
        )


class Simulator:
    """Steps one or more drones one control period at a time and logs them.

    One drone:     Simulator(controller, config, params, target_fn)
    Several:       Simulator(drones=[DroneSpec(...), ...], config=config)

    All drones share the clock and the wind. Each control period every
    drone's controller is called (seeing the others via `controller.others`),
    then physics advances and drone pairs closer than the sum of their
    `params.radius` are recorded in `collisions`. With
    `config.crash_on_collision`, both drones' motors cut out and they fall.

    `run` drives this for a fixed duration; live mode drives it in real time.
    Set `controller.sticks` between steps to feed pilot inputs.

    The single-drone attributes (`controller`, `params`, `state`, `target`,
    `thrust`, `positions()`, `result()`) refer to the first drone, so code
    written for one drone keeps working.
    """

    def __init__(
        self,
        controller: Controller | None = None,
        config: SimConfig | None = None,
        params: DroneParams | None = None,
        target_fn: Callable[[float], Target] | None = None,
        *,
        drones: list[DroneSpec] | None = None,
    ):
        if (controller is None) == (drones is None):
            raise TypeError("pass either a controller or drones=[DroneSpec, ...]")
        if drones is not None and (params is not None or target_fn is not None):
            raise TypeError("with drones=..., set params and target_fn on each DroneSpec")
        specs = drones if drones is not None else [DroneSpec(controller, params, None, target_fn)]
        if not specs:
            raise ValueError("drones=[] has no drones")

        self.config = config or SimConfig()
        default_target_fn = SCENARIOS[self.config.scenario]
        names = [spec.name or f"drone{i}" for i, spec in enumerate(specs)]
        if len(set(names)) != len(names):
            raise ValueError(f"drone names must be unique: {names}")
        if len({id(spec.controller) for spec in specs}) != len(specs):
            raise ValueError("each drone needs its own controller instance")
        self.drones = [Drone(spec, name, self.config, default_target_fn) for spec, name in zip(specs, names)]
        if self.config.crash_on_collision:
            for i, a in enumerate(self.drones):
                for b in self.drones[i + 1:]:
                    gap = np.linalg.norm(a.initial_state.pos - b.initial_state.pos)
                    if gap < a.params.radius + b.params.radius:
                        raise ValueError(
                            f"{a.name} and {b.name} start {gap:.2f} m apart, closer than their "
                            f"radii allow ({a.params.radius + b.params.radius:.2f} m); spread them "
                            f"out or set crash_on_collision=False"
                        )

        self.dt = 1.0 / self.config.physics_hz
        self.substeps = max(1, round(self.config.physics_hz / self.config.control_hz))
        self.control_dt = self.substeps * self.dt
        self.reset()

    def reset(self) -> None:
        """Back to the initial state and an empty log."""
        self.rng = np.random.default_rng(self.config.seed)
        self.k = 0
        self.gust = np.zeros(3)
        self.wind = np.asarray(self.config.wind_mean, dtype=float)
        self.collisions: list[Collision] = []
        for d in self.drones:
            d.reset()
            d.controller.name = d.name
            d.controller.others = self._neighbors(d)
            d.controller.reset(d.params)

    @property
    def t(self) -> float:
        return self.k * self.control_dt

    def drone(self, name: str) -> Drone:
        for d in self.drones:
            if d.name == name:
                return d
        raise KeyError(name)

    def _neighbors(self, me: Drone) -> tuple[Neighbor, ...]:
        return tuple(Neighbor(d.name, d.state, d.crashed) for d in self.drones if d is not me)

    def step(self) -> None:
        """Advance one control period: call every controller once, then physics."""
        cfg, t = self.config, self.t

        commands = []
        for d in self.drones:
            target = tuple(float(v) for v in d.target_fn(t))
            if len(target) == 3:
                target = (*target, 0.0)
            d.target = target

            if d.crashed:
                commands.append(np.zeros(4))
                continue
            d.controller.others = self._neighbors(d)
            measured = _noisy(d.state, cfg.sensor_noise, self.rng) if cfg.sensor_noise > 0 else d.state
            command = np.asarray(d.controller.update(t, measured, target, self.control_dt), dtype=float)
            if command.shape != (4,) or not np.all(np.isfinite(command)):
                raise ValueError(
                    f"{d.name}: controller returned {command!r}; expected four finite thrusts"
                )
            commands.append(command)

        # Gusts: first-order filtered noise with a ~0.5 s correlation time.
        if cfg.wind_gust > 0:
            a = self.control_dt / 0.5
            self.gust[:2] += -a * self.gust[:2] + cfg.wind_gust * math.sqrt(2 * a) * self.rng.normal(size=2)
        self.wind = np.asarray(cfg.wind_mean, dtype=float) + self.gust

        for d, command in zip(self.drones, commands):
            log = d._log
            log["t"].append(t)
            log["state"].append(d.state.as_array())
            log["euler"].append(d.state.euler)
            log["target"].append(d.target)
            log["cmd"].append(command)
            log["thrust"].append(d.thrust)
            log["wind"].append(self.wind)

            for _ in range(self.substeps):
                d.thrust = motor_step(d.thrust, command, d.params, self.dt)
                d.state = step(d.state, d.thrust, d.params, self.dt, wind=self.wind)

        self.k += 1
        self._check_collisions()

    def _check_collisions(self) -> None:
        """Record drone pairs that are touching. Checked once per control
        period, which at 250 Hz catches closing speeds up to ~100 m/s."""
        t = self.t
        for i, a in enumerate(self.drones):
            for b in self.drones[i + 1:]:
                if a.crashed and b.crashed:
                    continue
                dist = float(np.linalg.norm(a.state.pos - b.state.pos))
                if dist >= a.params.radius + b.params.radius:
                    continue
                already = any({c.a, c.b} == {a.name, b.name} for c in self.collisions)
                if not already:
                    self.collisions.append(Collision(t, a.name, b.name, dist))
                if self.config.crash_on_collision:
                    for d in (a, b):
                        if not d.crashed:
                            d.crashed_at = t
                            d.thrust = np.zeros(4)

    def results(self) -> list[SimResult]:
        """One SimResult per drone, in the order they were given."""
        return [d.result(self.config) for d in self.drones]

    # --- Single-drone view (the first drone) ---

    @property
    def controller(self) -> Controller:
        return self.drones[0].controller

    @property
    def params(self) -> DroneParams:
        return self.drones[0].params

    @property
    def state(self) -> State:
        return self.drones[0].state

    @property
    def target(self) -> tuple[float, float, float, float]:
        return self.drones[0].target

    @property
    def thrust(self) -> np.ndarray:
        return self.drones[0].thrust

    def positions(self, last: int | None = None) -> np.ndarray:
        """First drone's logged positions; see Drone.positions."""
        return self.drones[0].positions(last)

    def result(self) -> SimResult:
        """First drone's result; see `results` for all of them."""
        return self.drones[0].result(self.config)


def run(
    controller: Controller,
    config: SimConfig | None = None,
    params: DroneParams | None = None,
    target_fn: Callable[[float], Target] | None = None,
) -> SimResult:
    """Simulate one flight. Pass `target_fn` to override the named scenario."""
    sim = Simulator(controller, config, params, target_fn)
    for _ in range(int(round(sim.config.duration / sim.control_dt))):
        sim.step()
    return sim.result()


def run_many(drones: list[DroneSpec], config: SimConfig | None = None) -> tuple[list[SimResult], list[Collision]]:
    """Simulate several drones together. Returns one SimResult per drone and
    the list of collisions."""
    sim = Simulator(drones=drones, config=config)
    for _ in range(int(round(sim.config.duration / sim.control_dt))):
        sim.step()
    return sim.results(), sim.collisions


def offset_target(target_fn: Callable[[float], Target], offset) -> Callable[[float], Target]:
    """Shift a target function by a fixed (x, y, z), e.g. to fly a scenario in formation."""
    dx, dy, dz = offset

    def shifted(t: float) -> Target:
        x, y, z, *yaw = target_fn(t)
        return (x + dx, y + dy, z + dz, *yaw)

    return shifted
