"""Real-time flight without any graphics.

A LiveSession turns key presses into stick inputs and advances a Simulator
by wall-clock time. A front end (see drone_sim/graphics/live.py) forwards
its key events here, calls `tick` once per frame, and draws whatever it
likes from `session.sim`.

Key names follow matplotlib's: single lowercase letters, "up", "down",
"left", "right". Other front ends translate their own key names to these.
"""

import numpy as np

from .controller import Sticks
from .sim import Simulator

# key -> (stick, direction)
STICK_KEYS = {
    "w": ("pitch", 1), "s": ("pitch", -1),
    "d": ("roll", 1), "a": ("roll", -1),
    "up": ("throttle", 1), "down": ("throttle", -1),
    "right": ("yaw", 1), "left": ("yaw", -1),
}
RESET_KEY = "r"
HELP = "W/S fwd/back  A/D left/right  ↑/↓ up/down  ←/→ turn  R reset"

STICK_TIME = 0.15  # s, how quickly sticks follow the keys (keys are all-or-nothing)
MAX_TICK = 0.1  # s, don't try to catch up after a stall longer than this


class LiveSession:
    def __init__(self, sim: Simulator):
        self.sim = sim
        self.pressed: set[str] = set()
        self.sticks = Sticks()
        self._debt = 0.0

    def key_down(self, key: str) -> None:
        key = key.lower()
        if key == RESET_KEY:
            self.reset()
        else:
            self.pressed.add(key)

    def key_up(self, key: str) -> None:
        self.pressed.discard(key.lower())

    def release_all(self) -> None:
        """Forget held keys, e.g. when the window loses focus."""
        self.pressed.clear()

    def reset(self) -> None:
        self.sim.reset()
        self.sticks = Sticks()
        self._debt = 0.0

    def tick(self, elapsed: float) -> None:
        """Advance by `elapsed` seconds of wall time: update the sticks from
        the held keys, then step the simulator as many times as fit."""
        elapsed = min(elapsed, MAX_TICK)

        goal = dict.fromkeys(("throttle", "roll", "pitch", "yaw"), 0.0)
        for key in self.pressed:
            if key in STICK_KEYS:
                name, sign = STICK_KEYS[key]
                goal[name] += sign
        blend = min(1.0, elapsed / STICK_TIME)
        self.sticks = Sticks(**{
            name: v + blend * (np.clip(goal[name], -1, 1) - v)
            for name, v in vars(self.sticks).items()
        })
        self.sim.controller.sticks = self.sticks

        self._debt += elapsed
        while self._debt >= self.sim.control_dt:
            self.sim.step()
            self._debt -= self.sim.control_dt
