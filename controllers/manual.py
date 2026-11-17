"""Manual flight: fly the drone yourself with the keyboard.

    python -m drone_sim --live --controller controllers/manual.py

Velocity mode, like a camera drone's normal mode: the sticks ask for a speed,
and letting go brakes to a hover in place. Forward/right are relative to where
the drone is facing. The target (red X) is ignored, so try flying to it.
"""

import math

import numpy as np

from drone_sim import Controller
from drone_sim.control import thrust_vector, track_thrust_vector


class Manual(Controller):
    max_speed = 3.0  # m/s horizontal at full stick
    max_climb = 1.5  # m/s at full throttle
    yaw_rate = math.radians(90)  # rad/s at full yaw stick
    max_tilt = math.radians(30)
    k_vel = np.array([2.5, 2.5, 4.0])  # how hard to chase the commanded velocity

    def reset(self, params):
        self.p = params
        self.yaw = None

    def update(self, t, state, target, dt):
        s = self.sticks
        if self.yaw is None:
            self.yaw = state.euler[2]
        self.yaw -= s.yaw * self.yaw_rate * dt  # stick right = clockwise = -yaw

        # Sticks -> velocity in the drone's heading frame -> world frame.
        fwd, left = s.pitch * self.max_speed, -s.roll * self.max_speed
        c, sn = math.cos(self.yaw), math.sin(self.yaw)
        v_des = np.array([c * fwd - sn * left, sn * fwd + c * left, s.throttle * self.max_climb])

        acc = self.k_vel * (v_des - state.vel)
        f = thrust_vector(acc, self.p, self.max_tilt)
        return track_thrust_vector(f, self.yaw, state, self.p)
