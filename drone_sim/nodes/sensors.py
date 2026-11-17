"""Simulated sensors. Each reads `ground_truth` and publishes a noisy
measurement at its own rate.

Noise is seeded per drone and per sensor, so runs are reproducible and two
drones' sensors are independent.
"""

from __future__ import annotations

import zlib

import numpy as np

from ..dynamics import State, quat_multiply
from .core import Node
from .msgs import GpsFix, GroundTruth, MocapPose, StateEstimate
from .msgs import Imu as ImuMsg  # sensors.Imu publishes msgs.Imu


class Sensor(Node):
    """Base class: subclasses set `output` (topic), `msg_type`, and
    `default_rate`, and implement `measure(truth)`."""

    kind = "sensor"
    default_rate = 100.0
    output = "measurement"
    msg_type: type = object

    def __init__(self, name: str | None = None, rate: float | None = None, seed: int = 0, **kw):
        super().__init__(name, **kw)
        self.rate = rate or self.default_rate
        self.seed = seed

    def setup(self):
        self.truth: GroundTruth | None = None
        key = f"{self.graph.namespace}/{self.name}".encode()
        self.rng = np.random.default_rng([self.seed, zlib.crc32(key)])
        self.create_subscription("ground_truth", GroundTruth, self._on_truth)
        self.pub = self.create_publisher(self.output, self.msg_type)
        self.create_timer(1.0 / self.rate, self._tick)

    def _on_truth(self, msg):
        self.truth = msg

    def _tick(self):
        if self.truth is not None:
            self.pub.publish(self.measure(self.truth))

    def noise(self, sigma: float, n: int = 3) -> np.ndarray:
        return self.rng.normal(0.0, sigma, n) if sigma > 0 else np.zeros(n)

    def measure(self, truth: GroundTruth):
        raise NotImplementedError


class Imu(Sensor):
    """Accelerometer (specific force, body frame) and gyro, with noise and a
    constant gyro bias."""

    default_rate = 250.0
    output = "imu"
    msg_type = ImuMsg

    def __init__(self, name: str = "imu", rate: float | None = None, accel_noise: float = 0.05,
                 gyro_noise: float = 0.002, gyro_bias=(0.0, 0.0, 0.0), **kw):
        super().__init__(name, rate, **kw)
        self.accel_noise = accel_noise
        self.gyro_noise = gyro_noise
        self.gyro_bias = np.asarray(gyro_bias, dtype=float)

    def measure(self, truth):
        s = truth.state
        g = self.drone_params.gravity
        specific_force = s.R.T @ (truth.accel + np.array([0.0, 0.0, g]))
        return ImuMsg(
            self.now,
            specific_force + self.noise(self.accel_noise),
            s.omega + self.gyro_bias + self.noise(self.gyro_noise),
        )


class Gps(Sensor):
    """Position and velocity fixes."""

    default_rate = 10.0
    output = "gps"
    msg_type = GpsFix

    def __init__(self, name: str = "gps", rate: float | None = None, pos_noise: float = 0.1,
                 vel_noise: float = 0.05, **kw):
        super().__init__(name, rate, **kw)
        self.pos_noise = pos_noise
        self.vel_noise = vel_noise

    def measure(self, truth):
        s = truth.state
        return GpsFix(self.now, s.pos + self.noise(self.pos_noise), s.vel + self.noise(self.vel_noise))


class Mocap(Sensor):
    """Motion capture: precise position and attitude, like Vicon/OptiTrack."""

    default_rate = 100.0
    output = "mocap"
    msg_type = MocapPose

    def __init__(self, name: str = "mocap", rate: float | None = None, pos_noise: float = 0.002,
                 att_noise: float = 0.002, **kw):
        super().__init__(name, rate, **kw)
        self.pos_noise = pos_noise
        self.att_noise = att_noise

    def measure(self, truth):
        s = truth.state
        dq = np.r_[1.0, 0.5 * self.noise(self.att_noise)]
        q = quat_multiply(s.quat, dq)
        return MocapPose(self.now, s.pos + self.noise(self.pos_noise), q / np.linalg.norm(q))


class PerfectState(Sensor):
    """Publishes the exact state as `state_estimate`: the plain simulator's
    behavior, for when you don't want sensors or an estimator."""

    default_rate = 250.0
    output = "state_estimate"
    msg_type = StateEstimate

    def __init__(self, name: str = "perfect_state", rate: float | None = None, **kw):
        super().__init__(name, rate, **kw)

    def measure(self, truth):
        s = truth.state
        return StateEstimate(self.now, State(s.pos.copy(), s.vel.copy(), s.quat.copy(), s.omega.copy()))

