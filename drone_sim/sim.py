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

Target = tuple[float, float]


# --- Scenarios: functions of time returning the (x, z) target -----------------

def _hover(t: float) -> Target:
    return (0.0, 1.0)


def _step(t: float) -> Target:
    return (0.0, 1.0) if t < 3.0 else (2.0, 2.0)


def _square(t: float) -> Target:
    corners = [(0.0, 1.0), (2.0, 1.0), (2.0, 3.0), (0.0, 3.0)]
    return corners[int(t // 3.0) % len(corners)]


def _figure8(t: float) -> Target:
    w = 2 * math.pi / 8.0
    return (2.0 * math.sin(w * t), 2.0 + 0.75 * math.sin(2 * w * t))


SCENARIOS: dict[str, Callable[[float], Target]] = {
    "hover": _hover,
    "step": _step,
    "square": _square,
    "figure8": _figure8,
}


@dataclass
class SimConfig:
    duration: float = 10.0  # s
    physics_hz: float = 1000.0
    control_hz: float = 250.0
    scenario: str = "step"
    initial_state: State = field(default_factory=State)
    wind_mean: float = 0.0  # N, steady sideways force
    wind_gust: float = 0.0  # N, std dev of gusts
    sensor_noise: float = 0.0  # std dev added to every measured state component
    seed: int = 0


@dataclass
class SimResult:
    t: np.ndarray  # (N,)
    states: np.ndarray  # (N, 6) true state: x, z, theta, vx, vz, omega
    targets: np.ndarray  # (N, 2)
    commands: np.ndarray  # (N, 2) controller output (left, right)
    thrusts: np.ndarray  # (N, 2) actual rotor thrust after lag/saturation
    wind: np.ndarray  # (N,)
    params: DroneParams
    config: SimConfig

    def metrics(self) -> dict[str, float]:
        err = np.linalg.norm(self.states[:, :2] - self.targets, axis=1)
        return {
            "rms_pos_error_m": float(np.sqrt(np.mean(err**2))),
            "final_pos_error_m": float(err[-1]),
            "max_tilt_deg": float(np.degrees(np.max(np.abs(self.states[:, 2])))),
            "flipped": bool(np.any(np.abs(self.states[:, 2]) > math.pi / 2)),
        }


# --- Controller loading -------------------------------------------------------

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


# --- Main loop ----------------------------------------------------------------

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

    state = config.initial_state
    thrust = np.zeros(2)
    gust = 0.0
    controller.reset(params)

    log_t = np.empty(n)
    log_state = np.empty((n, 6))
    log_target = np.empty((n, 2))
    log_cmd = np.empty((n, 2))
    log_thrust = np.empty((n, 2))
    log_wind = np.empty(n)

    for k in range(n):
        t = k * control_dt
        target = target_fn(t)

        measured = state
        if config.sensor_noise > 0:
            measured = State.from_array(
                state.as_array() + rng.normal(0, config.sensor_noise, 6)
            )
        command = np.asarray(controller.update(t, measured, target, control_dt), dtype=float)
        if command.shape != (2,) or not np.all(np.isfinite(command)):
            raise ValueError(f"controller returned {command!r}; expected two finite thrusts")

        # Gusts: first-order filtered noise with a ~0.5 s correlation time.
        if config.wind_gust > 0:
            a = control_dt / 0.5
            gust += -a * gust + config.wind_gust * math.sqrt(2 * a) * rng.normal()
        wind = config.wind_mean + gust

        log_t[k] = t
        log_state[k] = state.as_array()
        log_target[k] = target
        log_cmd[k] = command
        log_thrust[k] = thrust
        log_wind[k] = wind

        for _ in range(substeps):
            thrust = motor_step(thrust, command, params, dt)
            state = step(state, (thrust[0], thrust[1]), params, dt, wind_x=wind)

    return SimResult(log_t, log_state, log_target, log_cmd, log_thrust, log_wind, params, config)
