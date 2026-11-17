"""Minimal example: the smallest possible controller.

Only holds altitude with a P-D loop and ignores x entirely, so it drifts in
wind. A good starting template.
"""

from drone_sim import Controller


class HoverOnly(Controller):
    def reset(self, params):
        self.hover = params.hover_thrust

    def update(self, t, state, target, dt):
        _, tz = target
        extra = 3.0 * (tz - state.z) - 2.0 * state.vz
        return self.hover + extra, self.hover + extra
