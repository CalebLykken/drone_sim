"""Outdoor stack: IMU + GPS -> complementary filter -> PID.

    python -m drone_sim -c stacks/gps_imu_pid.py -s square
    python -m drone_sim -c stacks/gps_imu_pid.py --graph
"""

from pathlib import Path

from drone_sim.nodes import ControllerNode, Gps, GpsImuEstimator, Imu

CONTROLLERS = Path(__file__).resolve().parent.parent / "controllers"


def stack():
    return [
        Imu(rate=250, accel_noise=0.05, gyro_noise=0.002),
        Gps(rate=10, pos_noise=0.1, vel_noise=0.05),
        GpsImuEstimator(),
        ControllerNode(CONTROLLERS / "pid.py", rate=250),
    ]
