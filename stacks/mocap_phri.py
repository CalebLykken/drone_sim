"""Indoor swarm stack: IMU + motion capture -> estimator -> PHRI planner.

    python -m drone_sim -c stacks/mocap_phri.py --layout circle -n 4
"""

from pathlib import Path

from drone_sim.nodes import ControllerNode, Imu, Mocap, MocapImuEstimator

CONTROLLERS = Path(__file__).resolve().parent.parent / "controllers"


def stack():
    return [
        Imu(rate=250),
        Mocap(rate=100, pos_noise=0.002),
        MocapImuEstimator(),
        ControllerNode(CONTROLLERS / "phri.py", rate=250),
    ]
