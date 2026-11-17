"""Example: cascaded PID controller.

Outer loop: position error -> desired acceleration -> desired tilt + thrust.
Inner loop: tilt error -> torque -> differential rotor thrust.

Copy this file and edit it to try your own ideas.
"""

import math

from drone_sim import Controller


class PID:
    def __init__(self, kp, ki=0.0, kd=0.0, i_limit=float("inf")):
        self.kp, self.ki, self.kd, self.i_limit = kp, ki, kd, i_limit
        self.integral = 0.0

    def __call__(self, error, error_rate, dt):
        self.integral = max(-self.i_limit, min(self.i_limit, self.integral + error * dt))
        return self.kp * error + self.ki * self.integral + self.kd * error_rate


class CascadedPID(Controller):
    max_tilt = math.radians(30)

    def reset(self, params):
        self.p = params
        self.x_pid = PID(kp=1.5, kd=2.0)
        self.z_pid = PID(kp=6.0, ki=0.5, kd=4.0, i_limit=1.0)
        self.theta_pid = PID(kp=40.0, kd=6.0)

    def update(self, t, state, target, dt):
        p = self.p
        tx, tz = target

        # Outer loop. Targets are treated as stationary, so error rate = -velocity.
        ax = self.x_pid(tx - state.x, -state.vx, dt)
        az = self.z_pid(tz - state.z, -state.vz, dt)

        # Tilting by theta tilts thrust to -x (see dynamics.py), so theta = -ax / g.
        theta_des = max(-self.max_tilt, min(self.max_tilt, -ax / p.gravity))
        total = p.mass * (p.gravity + az) / max(math.cos(state.theta), 0.5)

        # Inner loop: attitude.
        alpha = self.theta_pid(theta_des - state.theta, -state.omega, dt)
        diff = alpha * p.inertia / p.arm_length  # T_right - T_left

        return (total - diff) / 2, (total + diff) / 2
