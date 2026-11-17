"""The interface every controller implements.

Write a subclass in any .py file and pass its path to the simulator:

    python -m drone_sim --controller path/to/my_controller.py

    from drone_sim import Controller

    class MyController(Controller):
        def reset(self, params):
            self.hover = params.hover_thrust

        def update(self, t, state, target, dt):
            return self.hover, self.hover   # (left, right) thrust in newtons
"""

from .dynamics import DroneParams, State


class Controller:
    """Base class for drone controllers.

    The simulator calls `reset` once before a run, then `update` at the
    control rate (SimConfig.control_hz). Commands are clipped to
    [0, params.max_thrust] and pass through the motor lag before acting on
    the drone, just like real ESCs and props.
    """

    def reset(self, params: DroneParams) -> None:
        """Called once before the run. Store whatever physical params you need."""

    def update(
        self,
        t: float,
        state: State,
        target: tuple[float, float],
        dt: float,
    ) -> tuple[float, float]:
        """Return (left, right) rotor thrust commands in newtons.

        t:      sim time in seconds
        state:  measured state (with sensor noise, if enabled)
        target: desired (x, z) position in meters
        dt:     seconds since the previous update
        """
        raise NotImplementedError
