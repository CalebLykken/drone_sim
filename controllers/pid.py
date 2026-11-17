"""Example: cascaded position -> attitude controller.

Outer loop: position PID -> desired acceleration -> desired thrust vector.
The thrust vector plus the target yaw define a desired attitude.
Inner loop: attitude error on SO(3) (geometric control) -> body torque.
`params.allocate` turns (total thrust, torque) into four rotor thrusts.

Copy this file and edit it to try your own ideas.
"""

import math

import numpy as np

from drone_sim import Controller


class PID:
    def __init__(self, kp, ki=0.0, kd=0.0, i_limit=float("inf")):
        self.kp, self.ki, self.kd, self.i_limit = (np.asarray(v, dtype=float) for v in (kp, ki, kd, i_limit))
        self.integral = 0.0

    def __call__(self, error, error_rate, dt):
        self.integral = np.clip(self.integral + error * dt, -self.i_limit, self.i_limit)
        return self.kp * error + self.ki * self.integral + self.kd * error_rate


def vee(M):
    """Skew-symmetric matrix -> vector."""
    return np.array([M[2, 1], M[0, 2], M[1, 0]])


class CascadedPID(Controller):
    max_tilt = math.radians(35)

    def reset(self, params):
        self.p = params
        # Per-axis gains for x, y, z.
        self.pos_pid = PID(kp=[2.0, 2.0, 6.0], ki=[0.0, 0.0, 0.5], kd=[2.5, 2.5, 4.0], i_limit=1.0)
        self.k_att = np.array([60.0, 60.0, 15.0])  # roll, pitch, yaw
        self.k_rate = np.array([12.0, 12.0, 6.0])

    def update(self, t, state, target, dt):
        p = self.p
        g = p.gravity
        *goal, yaw_des = target

        # Outer loop. Targets are treated as stationary, so error rate = -velocity.
        acc = self.pos_pid(np.array(goal) - state.pos, -state.vel, dt)

        # Desired thrust vector, with horizontal part limited to max_tilt.
        f = np.array([acc[0], acc[1], g + acc[2]])
        f[2] = max(f[2], 0.2 * g)
        horiz = np.hypot(f[0], f[1])
        limit = f[2] * math.tan(self.max_tilt)
        if horiz > limit:
            f[:2] *= limit / horiz

        # Desired attitude: body z along f, body x pointing toward yaw_des.
        z_b = f / np.linalg.norm(f)
        x_c = np.array([math.cos(yaw_des), math.sin(yaw_des), 0.0])
        y_b = np.cross(z_b, x_c)
        y_b /= np.linalg.norm(y_b)
        R_des = np.column_stack([np.cross(y_b, z_b), y_b, z_b])

        # Inner loop: attitude error on SO(3), then torque = I * angular accel.
        R = state.R
        e_R = 0.5 * vee(R_des.T @ R - R.T @ R_des)
        alpha = -self.k_att * e_R - self.k_rate * state.omega
        torque = p.inertia_matrix @ alpha + np.cross(state.omega, p.inertia_matrix @ state.omega)

        total = p.mass * f @ R[:, 2]  # only the part of f along the current thrust axis
        return p.allocate(total, torque)
