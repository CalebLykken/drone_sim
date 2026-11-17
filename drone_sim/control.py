"""Building blocks for controllers: thrust vector -> attitude -> torque.

Most position or velocity controllers end the same way: decide which way to
push (a desired acceleration), tilt the drone to push that way, and spin the
rotors to get there. These helpers do that last part so controllers can focus
on the first.
"""

import math

import numpy as np

from .dynamics import DroneParams, State


def vee(M: np.ndarray) -> np.ndarray:
    """Skew-symmetric matrix -> vector."""
    return np.array([M[2, 1], M[0, 2], M[1, 0]])


def thrust_vector(acc, params: DroneParams, max_tilt: float = math.radians(35)) -> np.ndarray:
    """Desired specific thrust (m/s^2, world frame) for a desired acceleration,
    with gravity compensated and the tilt limited to max_tilt radians."""
    g = params.gravity
    f = np.array([acc[0], acc[1], g + acc[2]], dtype=float)
    f[2] = max(f[2], 0.2 * g)
    horiz = math.hypot(f[0], f[1])
    limit = f[2] * math.tan(max_tilt)
    if horiz > limit:
        f[:2] *= limit / horiz
    return f


def attitude_from_thrust(f: np.ndarray, yaw: float) -> np.ndarray:
    """Rotation matrix with body z along f and body x pointing toward yaw."""
    z_b = f / np.linalg.norm(f)
    x_c = np.array([math.cos(yaw), math.sin(yaw), 0.0])
    y_b = np.cross(z_b, x_c)
    y_b /= np.linalg.norm(y_b)
    return np.column_stack([np.cross(y_b, z_b), y_b, z_b])


def track_thrust_vector(
    f: np.ndarray,
    yaw: float,
    state: State,
    params: DroneParams,
    k_att=(60.0, 60.0, 15.0),
    k_rate=(12.0, 12.0, 6.0),
) -> np.ndarray:
    """Rotor thrusts that push along specific thrust f (from `thrust_vector`)
    while turning to face `yaw`. Uses geometric attitude control on SO(3)."""
    R = state.R
    R_des = attitude_from_thrust(f, yaw)
    e_R = 0.5 * vee(R_des.T @ R - R.T @ R_des)
    alpha = -np.asarray(k_att) * e_R - np.asarray(k_rate) * state.omega
    I = params.inertia_matrix
    torque = I @ alpha + np.cross(state.omega, I @ state.omega)
    total = params.mass * f @ R[:, 2]  # only the part of f along the current thrust axis
    return params.allocate(total, torque)
