"""Minimal example: the smallest controller that flies.

Holds altitude with a P-D loop and keeps the drone level with a P-D attitude
loop, but ignores x/y position entirely, so it drifts in wind. A good
starting template.
"""

import numpy as np

from drone_sim import Controller


class HoverOnly(Controller):
    def reset(self, params):
        self.p = params

    def update(self, t, state, target, dt):
        p = self.p
        tz = target[2]

        total = p.mass * (p.gravity + 3.0 * (tz - state.z) - 2.0 * state.vel[2])
        roll, pitch, yaw = state.euler
        torque = p.inertia_matrix @ (
            -40.0 * np.array([roll, pitch, 0.0]) - 8.0 * state.omega
        )
        return p.allocate(total, torque)
