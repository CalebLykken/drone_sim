"""Simulation loop, scenarios, and controller loading."""

import importlib.util
import inspect
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np

from .controller import Controller
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


def run(
    controller: Controller,
    config: SimConfig | None = None,
    params: DroneParams | None = None,
    target_fn: Callable[[float], Target] | None = None,
) -> SimResult:
    """Simulate one flight. Pass `target_fn` to override the named scenario."""
    config = config or SimConfig()
    params = params or DroneParams()
    target_fn = target_fn or SCENARIOS[config.scenario]
    rng = np.random.default_rng(config.seed)

    dt = 1.0 / config.physics_hz
    substeps = max(1, round(config.physics_hz / config.control_hz))
    control_dt = substeps * dt
    n = int(round(config.duration / control_dt))

    state = State.from_array(config.initial_state.as_array())
    thrust = np.zeros(4)
    gust = np.zeros(3)
    wind_mean = np.asarray(config.wind_mean, dtype=float)
    controller.reset(params)

    log_t = np.empty(n)
    log_state = np.empty((n, 13))
    log_euler = np.empty((n, 3))
    log_target = np.empty((n, 4))
    log_cmd = np.empty((n, 4))
    log_thrust = np.empty((n, 4))
    log_wind = np.empty((n, 3))

    for k in range(n):
        t = k * control_dt
        target = tuple(float(v) for v in target_fn(t))
        if len(target) == 3:
            target = (*target, 0.0)

        measured = _noisy(state, config.sensor_noise, rng) if config.sensor_noise > 0 else state
        command = np.asarray(controller.update(t, measured, target, control_dt), dtype=float)
        if command.shape != (4,) or not np.all(np.isfinite(command)):
            raise ValueError(f"controller returned {command!r}; expected four finite thrusts")

        # Gusts: first-order filtered noise with a ~0.5 s correlation time.
        if config.wind_gust > 0:
            a = control_dt / 0.5
            gust[:2] += -a * gust[:2] + config.wind_gust * math.sqrt(2 * a) * rng.normal(size=2)
        wind = wind_mean + gust

        log_t[k] = t
        log_state[k] = state.as_array()
        log_euler[k] = state.euler
        log_target[k] = target
        log_cmd[k] = command
        log_thrust[k] = thrust
        log_wind[k] = wind

        for _ in range(substeps):
            thrust = motor_step(thrust, command, params, dt)
            state = step(state, thrust, params, dt, wind=wind)

    return SimResult(
        log_t, log_state, log_euler, log_target, log_cmd, log_thrust, log_wind, params, config
    )
