"""Planar (2D) quadrotor dynamics.

The drone lives in the x-z plane with two rotors, left and right, separated by
2 * arm_length. Pitch angle theta is measured counter-clockwise from level, so
more thrust on the right rotor rotates the drone counter-clockwise (theta > 0)
and tilts the thrust vector to the left (-x).

    x_ddot     = -(T_l + T_r) * sin(theta) / m - drag * x_dot / m + wind_x / m
    z_ddot     =  (T_l + T_r) * cos(theta) / m - g - drag * z_dot / m
    theta_ddot =  (T_r - T_l) * arm_length / I
"""

from dataclasses import dataclass, replace

import numpy as np


@dataclass(frozen=True)
class DroneParams:
    mass: float = 1.0  # kg
    inertia: float = 0.01  # kg m^2, about the pitch axis
    arm_length: float = 0.15  # m, rotor to center of mass
    gravity: float = 9.81  # m/s^2
    max_thrust: float = 8.0  # N, per rotor
    motor_tau: float = 0.03  # s, first-order motor spin-up time constant
    drag: float = 0.1  # N s/m, linear translational drag

    @property
    def hover_thrust(self) -> float:
        """Per-rotor thrust needed to hover."""
        return self.mass * self.gravity / 2.0


@dataclass(frozen=True)
class State:
    x: float = 0.0
    z: float = 0.0
    theta: float = 0.0
    vx: float = 0.0
    vz: float = 0.0
    omega: float = 0.0

    def as_array(self) -> np.ndarray:
        return np.array([self.x, self.z, self.theta, self.vx, self.vz, self.omega])

    @classmethod
    def from_array(cls, a) -> "State":
        return cls(*(float(v) for v in a))


def step(
    state: State,
    thrust: tuple[float, float],
    params: DroneParams,
    dt: float,
    wind_x: float = 0.0,
) -> State:
    """Advance the rigid body by dt with semi-implicit Euler. `thrust` is the
    actual (post-motor-lag, saturated) (left, right) rotor thrust in newtons."""
    t_l, t_r = thrust
    total = t_l + t_r
    m = params.mass

    ax = (-total * np.sin(state.theta) - params.drag * state.vx + wind_x) / m
    az = (total * np.cos(state.theta) - params.drag * state.vz) / m - params.gravity
    alpha = (t_r - t_l) * params.arm_length / params.inertia

    vx = state.vx + ax * dt
    vz = state.vz + az * dt
    omega = state.omega + alpha * dt
    new = State(
        x=state.x + vx * dt,
        z=state.z + vz * dt,
        theta=state.theta + omega * dt,
        vx=vx,
        vz=vz,
        omega=omega,
    )

    # Ground contact: the drone can sit on the ground but not pass through it.
    if new.z < 0.0:
        new = replace(new, z=0.0, vz=max(new.vz, 0.0), vx=0.0, omega=0.0, theta=0.0)
    return new


def motor_step(actual: np.ndarray, command: np.ndarray, params: DroneParams, dt: float) -> np.ndarray:
    """First-order lag from commanded to actual rotor thrust, with saturation."""
    command = np.clip(command, 0.0, params.max_thrust)
    alpha = dt / (params.motor_tau + dt)
    return actual + alpha * (command - actual)
