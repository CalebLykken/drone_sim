"""The interface every controller implements.

Write a subclass in any .py file and pass its path to the simulator:

    python -m drone_sim --controller path/to/my_controller.py

    from drone_sim import Controller

    class MyController(Controller):
        def reset(self, params):
            self.hover = params.hover_thrust

        def update(self, t, state, target, dt):
            return [self.hover] * 4   # FL, FR, BR, BL thrust in newtons
"""

from .dynamics import DroneParams, State


class Controller:
    """Base class for drone controllers.

    The simulator calls `reset` once before a run, then `update` at the
    control rate (SimConfig.control_hz). Commands are clipped to
    [0, params.max_thrust] and pass through the motor lag before acting on
    the drone, just like real ESCs and props.

    Tip: compute a total thrust and a body torque, then let
    `params.allocate(total_thrust, torque)` split it across the four rotors.
    """

    def reset(self, params: DroneParams) -> None:
        """Called once before the run. Store whatever physical params you need."""

    def update(
        self,
        t: float,
        state: State,
        target: tuple[float, float, float, float],
        dt: float,
    ):
        """Return four rotor thrust commands in newtons: (FL, FR, BR, BL).

        t:      sim time in seconds
        state:  measured State (.pos .vel .quat .omega, plus .R and .euler),
                with sensor noise if enabled
        target: desired (x, y, z, yaw) in meters and radians
        dt:     seconds since the previous update
        """
        raise NotImplementedError
