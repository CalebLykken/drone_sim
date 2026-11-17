"""3D quadrotor dynamics (X configuration).

Frames
------
World: z up, x and y horizontal (ENU-style). Gravity is -z.
Body:  x forward, y left, z up through the rotors.

Attitude is a unit quaternion q = (w, x, y, z) rotating body vectors into the
world frame; R(q) is the matching rotation matrix. Angular velocity `omega`
is expressed in the body frame.

Rotors (seen from above, front is +x):

        front
     0 FL   1 FR
         \\ /
          X
         / \\
     3 BL   2 BR

Each rotor pushes along body +z with thrust T_i and twists the body about
body z by yaw_sign_i * yaw_coeff * T_i (the reaction to its spin).

    m  v_dot     = R @ [0, 0, sum T] - m g e_z - drag * v + wind
    I  omega_dot = tau - omega x (I omega)
    tau          = sum_i r_i x [0, 0, T_i] + [0, 0, yaw_sign_i * yaw_coeff * T_i]
"""

from dataclasses import dataclass, field
from functools import cached_property

import numpy as np


# --- Rotation helpers ----------------------------------------------------------

def quat_to_rot(q: np.ndarray) -> np.ndarray:
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def quat_multiply(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return np.array([
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ])


def euler_to_quat(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """ZYX (yaw, then pitch, then roll) Euler angles in radians to a quaternion."""
    cr, sr = np.cos(roll / 2), np.sin(roll / 2)
    cp, sp = np.cos(pitch / 2), np.sin(pitch / 2)
    cy, sy = np.cos(yaw / 2), np.sin(yaw / 2)
    return np.array([
        cr * cp * cy + sr * sp * sy,
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
    ])


def rot_to_euler(R: np.ndarray) -> np.ndarray:
    """Rotation matrix to ZYX Euler angles (roll, pitch, yaw) in radians."""
    roll = np.arctan2(R[2, 1], R[2, 2])
    pitch = -np.arcsin(np.clip(R[2, 0], -1.0, 1.0))
    yaw = np.arctan2(R[1, 0], R[0, 0])
    return np.array([roll, pitch, yaw])


# --- Drone ---------------------------------------------------------------------

@dataclass(frozen=True)
class DroneParams:
    mass: float = 1.0  # kg
    inertia: tuple[float, float, float] = (0.01, 0.01, 0.02)  # kg m^2 about body x, y, z
    arm_length: float = 0.17  # m, center to each rotor
    gravity: float = 9.81  # m/s^2
    max_thrust: float = 5.0  # N, per rotor
    yaw_coeff: float = 0.015  # m, yaw torque per newton of rotor thrust
    motor_tau: float = 0.03  # s, first-order motor spin-up time constant
    drag: float = 0.1  # N s/m, linear translational drag
    yaw_signs: tuple[float, ...] = (1.0, -1.0, 1.0, -1.0)  # FL, FR, BR, BL
    radius: float = 0.25  # m, bounding sphere (arm + prop) for drone-drone collisions

    @property
    def hover_thrust(self) -> float:
        """Per-rotor thrust needed to hover."""
        return self.mass * self.gravity / 4.0

    @property
    def inertia_matrix(self) -> np.ndarray:
        return np.diag(self.inertia)

    @cached_property
    def rotor_positions(self) -> np.ndarray:
        """(4, 3) rotor positions in the body frame: FL, FR, BR, BL."""
        a = self.arm_length / np.sqrt(2)
        return np.array([[a, a, 0], [a, -a, 0], [-a, -a, 0], [-a, a, 0]])

    @cached_property
    def mixer(self) -> np.ndarray:
        """(4, 4) matrix M with [total_thrust, tau_x, tau_y, tau_z] = M @ thrusts."""
        r = self.rotor_positions
        return np.array([
            np.ones(4),
            r[:, 1],
            -r[:, 0],
            self.yaw_coeff * np.array(self.yaw_signs),
        ])

    def allocate(self, total_thrust: float, torque) -> np.ndarray:
        """Rotor thrusts that produce a total thrust (N) and body torque (N m).

        This is the inverse of `mixer`. The result can be negative or above
        max_thrust when you ask for more than the motors can give; the sim
        clips it.
        """
        return np.linalg.solve(self.mixer, np.array([total_thrust, *torque]))


@dataclass
class State:
    pos: np.ndarray = field(default_factory=lambda: np.zeros(3))  # world, m
    vel: np.ndarray = field(default_factory=lambda: np.zeros(3))  # world, m/s
    quat: np.ndarray = field(default_factory=lambda: np.array([1.0, 0, 0, 0]))  # body -> world
    omega: np.ndarray = field(default_factory=lambda: np.zeros(3))  # body, rad/s

    @property
    def R(self) -> np.ndarray:
        """Rotation matrix, body -> world. Column 2 is the thrust direction."""
        return quat_to_rot(self.quat)

    @property
    def euler(self) -> np.ndarray:
        """(roll, pitch, yaw) in radians, ZYX convention."""
        return rot_to_euler(self.R)

    x = property(lambda s: s.pos[0])
    y = property(lambda s: s.pos[1])
    z = property(lambda s: s.pos[2])

    def as_array(self) -> np.ndarray:
        return np.concatenate([self.pos, self.vel, self.quat, self.omega])

    @classmethod
    def from_array(cls, a) -> "State":
        a = np.asarray(a, dtype=float)
        return cls(a[0:3].copy(), a[3:6].copy(), a[6:10].copy(), a[10:13].copy())


def step(
    state: State,
    thrust: np.ndarray,
    params: DroneParams,
    dt: float,
    wind=(0.0, 0.0, 0.0),
) -> State:
    """Advance the rigid body by dt with semi-implicit Euler. `thrust` is the
    actual (post-motor-lag, saturated) thrust of each rotor in newtons."""
    R = quat_to_rot(state.quat)
    wrench = params.mixer @ thrust
    total, torque = wrench[0], wrench[1:]

    acc = (R[:, 2] * total - params.drag * state.vel + np.asarray(wind)) / params.mass
    acc[2] -= params.gravity
    I = params.inertia_matrix
    alpha = np.linalg.solve(I, torque - np.cross(state.omega, I @ state.omega))

    vel = state.vel + acc * dt
    omega = state.omega + alpha * dt
    pos = state.pos + vel * dt
    dq = quat_multiply(state.quat, np.array([0.0, *omega])) * (0.5 * dt)
    quat = state.quat + dq
    quat /= np.linalg.norm(quat)

    # Ground contact: the drone can sit on the ground but not pass through it.
    # Landing levels it out and keeps only its heading.
    if pos[2] < 0.0:
        pos[2] = 0.0
        vel = np.array([0.0, 0.0, max(vel[2], 0.0)])
        omega = np.zeros(3)
        quat = euler_to_quat(0.0, 0.0, rot_to_euler(quat_to_rot(quat))[2])
    return State(pos, vel, quat, omega)


def motor_step(actual: np.ndarray, command: np.ndarray, params: DroneParams, dt: float) -> np.ndarray:
    """First-order lag from commanded to actual rotor thrust, with saturation."""
    command = np.clip(command, 0.0, params.max_thrust)
    alpha = dt / (params.motor_tau + dt)
    return actual + alpha * (command - actual)
