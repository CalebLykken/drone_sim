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

from dataclasses import dataclass

from .dynamics import DroneParams, State


@dataclass(frozen=True)
class Sticks:
    """Pilot inputs, each in [-1, 1]. All zero unless flying with --live.

    throttle: +1 up       roll: +1 right
    pitch:    +1 forward  yaw:  +1 turn right (clockwise from above)
    """

    throttle: float = 0.0
    roll: float = 0.0
    pitch: float = 0.0
    yaw: float = 0.0


@dataclass(frozen=True)
class Neighbor:
    """Another drone in the same simulation, as seen by a controller."""

    name: str
    state: State  # true state (no sensor noise), at the same instant as yours
    crashed: bool = False
    # Where it is trying to go, (x, y, z, yaw). This is privileged information
    # (a real drone would have to infer it), as assumed by planners such as
    # drone_sim.phri that model the other agents' goals.
    target: tuple[float, float, float, float] | None = None


class Controller:
    """Base class for drone controllers.

    The simulator calls `reset` once before a run, then `update` at the
    control rate (SimConfig.control_hz). Commands are clipped to
    [0, params.max_thrust] and pass through the motor lag before acting on
    the drone, just like real ESCs and props.

    Tip: compute a total thrust and a body torque, then let
    `params.allocate(total_thrust, torque)` split it across the four rotors.

    `self.sticks` holds the latest pilot inputs (see Sticks). In live mode the
    simulator updates it from the keyboard before every `update` call.

    `self.name` is this drone's name and `self.others` is a tuple of Neighbor,
    one per other drone in the simulation (empty when flying alone). Both are
    set by the simulator before `reset`/`update`, so formation or collision-
    avoidance controllers can react to the rest of the swarm.
    """

    sticks: Sticks = Sticks()
    name: str = "drone0"
    others: tuple[Neighbor, ...] = ()

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
