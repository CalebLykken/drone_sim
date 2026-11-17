"""Fly it yourself on real-ish sensors: IMU + GPS -> estimator -> manual.

    python -m drone_sim --live -c stacks/manual_gps.py
"""

from pathlib import Path

from drone_sim.nodes import ControllerNode, Gps, GpsImuEstimator, Imu

CONTROLLERS = Path(__file__).resolve().parent.parent / "controllers"


def stack():
    return [
        Imu(),
        Gps(),
        GpsImuEstimator(),
        ControllerNode(CONTROLLERS / "manual.py"),
    ]
