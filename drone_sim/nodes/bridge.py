"""Connecting node graphs to the simulator.

`StackController` is an ordinary Controller whose insides are a node graph,
so a stack flies anywhere a controller does (multi-drone runs, live mode,
the graphics). Each control period it:

1. has the `sim` node (SimBridge) publish the true state, the setpoint, the
   swarm, and the pilot's sticks,
2. runs the graph's timers up to the current time (sensors, estimators,
   controllers, ...),
3. returns the most recent `motor_command`.

`ControllerNode` runs any existing Controller as a node, so controllers
written for the plain simulator work unchanged inside a stack.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np

from ..controller import Controller
from .core import Graph, Node
from .msgs import GroundTruth, MotorCommand, Setpoint, StateEstimate, SticksInput, SwarmState


class SimBridge(Node):
    """The simulator side of a stack, like a Gazebo plugin: publishes
    `ground_truth`, `setpoint`, `sticks`, and `/swarm`, and applies
    `motor_command` to the drone."""

    kind = "sim"

    def __init__(self, name: str = "sim", **kw):
        super().__init__(name, **kw)

    def setup(self):
        self.truth_pub = self.create_publisher("ground_truth", GroundTruth)
        self.setpoint_pub = self.create_publisher("setpoint", Setpoint)
        self.sticks_pub = self.create_publisher("sticks", SticksInput)
        self.swarm_pub = self.create_publisher("/swarm", SwarmState)
        self.create_subscription("motor_command", MotorCommand, self._on_command)
        self.command: MotorCommand | None = None
        self._prev = None  # (t, vel) for the acceleration estimate

    def _on_command(self, msg: MotorCommand) -> None:
        self.command = msg

    def push(self, t, state, target, others, sticks) -> None:
        # World acceleration by finite difference over one control period
        # (the plant doesn't expose it); IMUs turn it into specific force.
        accel = np.zeros(3)
        if self._prev is not None and t > self._prev[0]:
            accel = (state.vel - self._prev[1]) / (t - self._prev[0])
        self._prev = (t, state.vel.copy())
        self.truth_pub.publish(GroundTruth(t, state, accel))
        self.setpoint_pub.publish(Setpoint(t, tuple(target)))
        self.sticks_pub.publish(SticksInput(t, sticks))
        self.swarm_pub.publish(SwarmState(t, tuple(others)))


class StackController(Controller):
    """A Controller made of nodes. `build` returns a fresh list of nodes;
    it's called on every reset so each run starts clean."""

    def __init__(self, build: Callable[[], list[Node]], label: str = "stack"):
        self.build = build
        self.label = label
        self.graph: Graph | None = None

    def reset(self, params):
        self.bridge = SimBridge()
        self.graph = Graph([self.bridge, *self.build()], namespace=f"/{self.name}", name=self.name)
        self.graph.start(params)

    def update(self, t, state, target, dt):
        self.bridge.push(t, state, target, self.others, self.sticks)
        self.graph.spin_until(t)
        cmd = self.bridge.command
        return np.zeros(4) if cmd is None else cmd.thrusts

    def __repr__(self) -> str:
        return f"StackController({self.label!r})"


class ControllerNode(Node):
    """Runs a Controller at `rate` Hz. Inputs: `state_estimate`, `setpoint`,
    `/swarm`, `sticks`. Output: `motor_command`. Until the first estimate
    arrives the motors stay off."""

    kind = "controller"

    def __init__(self, controller: Controller | str, rate: float = 250.0, name: str = "controller", **kw):
        super().__init__(name, **kw)
        self.source = controller
        self.rate = rate

    def setup(self):
        if isinstance(self.source, (str, Path)):
            from ..sim import load_controller

            self.controller = load_controller(str(self.source))
        else:
            self.controller = self.source
        self.controller.name = self.graph.name
        self.controller.others = ()
        self.controller.reset(self.drone_params)

        self.cmd_pub = self.create_publisher("motor_command", MotorCommand)
        self.create_subscription("state_estimate", StateEstimate, self._on_estimate)
        self.create_subscription("setpoint", Setpoint, self._on_setpoint)
        self.create_subscription("/swarm", SwarmState, self._on_swarm)
        self.create_subscription("sticks", SticksInput, self._on_sticks)
        self.estimate: StateEstimate | None = None
        self.target = None
        self.create_timer(1.0 / self.rate, self._tick)

    def _on_estimate(self, msg):
        self.estimate = msg

    def _on_setpoint(self, msg):
        self.target = msg.target

    def _on_swarm(self, msg):
        self.controller.others = msg.neighbors

    def _on_sticks(self, msg):
        self.controller.sticks = msg.sticks

    def _tick(self):
        if self.estimate is None or self.target is None:
            thrusts = np.zeros(4)
        else:
            thrusts = np.asarray(
                self.controller.update(self.now, self.estimate.state, self.target, 1.0 / self.rate),
                dtype=float,
            )
        self.cmd_pub.publish(MotorCommand(self.now, thrusts))


def default_stack(controller: Controller) -> StackController:
    """The stack equivalent of flying `controller` directly: perfect state
    feedback at the controller's rate. Used to draw plain controllers."""
    from .sensors import PerfectState

    return StackController(lambda: [PerfectState(), ControllerNode(controller)], label=type(controller).__name__)

