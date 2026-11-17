"""ROS-style nodes for building full drone stacks.

    from drone_sim.nodes import ControllerNode, Gps, GpsImuEstimator, Imu

    def stack():            # in a stack file, e.g. stacks/gps_imu_pid.py
        return [Imu(), Gps(), GpsImuEstimator(), ControllerNode("controllers/pid.py")]

    python -m drone_sim -c stacks/gps_imu_pid.py --graph

See core.py for Node/Graph, msgs.py for message types, sensors.py and
estimators.py for the built-in nodes, and bridge.py for how a stack plugs
into the simulator.
"""

from .bridge import ControllerNode, SimBridge, StackController, default_stack
from .core import Graph, Node, Publisher, Subscription, Timer, Topic, to_dot, to_mermaid
from .estimators import GpsImuEstimator, InertialEstimator, MocapImuEstimator
from .sensors import Gps, Imu, Mocap, PerfectState, Sensor

__all__ = [
    "Graph", "Node", "Publisher", "Subscription", "Timer", "Topic", "to_dot", "to_mermaid",
    "ControllerNode", "SimBridge", "StackController", "default_stack",
    "Sensor", "Imu", "Gps", "Mocap", "PerfectState",
    "InertialEstimator", "GpsImuEstimator", "MocapImuEstimator",
]
