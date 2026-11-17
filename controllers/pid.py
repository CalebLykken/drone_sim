"""Example: cascaded position -> attitude controller.

Outer loop: position PID -> desired acceleration -> desired thrust vector.
Inner loop: the thrust vector plus the target yaw define a desired attitude,
tracked with geometric attitude control (see drone_sim/control.py).

Copy this file and edit it to try your own ideas.
"""

import math

import numpy as np

from drone_sim import Controller
from drone_sim.control import thrust_vector, track_thrust_vector


class PID:
    def __init__(self, kp, ki=0.0, kd=0.0, i_limit=float("inf")):
        self.kp, self.ki, self.kd, self.i_limit = (np.asarray(v, dtype=float) for v in (kp, ki, kd, i_limit))
        self.integral = 0.0

    def __call__(self, error, error_rate, dt):
        self.integral = np.clip(self.integral + error * dt, -self.i_limit, self.i_limit)
        return self.kp * error + self.ki * self.integral + self.kd * error_rate


class CascadedPID(Controller):
    max_tilt = math.radians(35)

    def reset(self, params):
        self.p = params
        # Per-axis gains for x, y, z.
        self.pos_pid = PID(kp=[2.0, 2.0, 6.0], ki=[0.0, 0.0, 0.5], kd=[2.5, 2.5, 4.0], i_limit=1.0)

    def update(self, t, state, target, dt):
        *goal, yaw_des = target

        # Outer loop. Targets are treated as stationary, so error rate = -velocity.
        acc = self.pos_pid(np.array(goal) - state.pos, -state.vel, dt)

        f = thrust_vector(acc, self.p, self.max_tilt)
        return track_thrust_vector(f, yaw_des, state, self.p)
