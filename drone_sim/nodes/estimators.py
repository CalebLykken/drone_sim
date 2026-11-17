"""State estimators: turn sensor messages into `state_estimate`.

Both estimators here are complementary filters around inertial navigation:
the IMU propagates attitude (gyro) and velocity/position (accelerometer) at
its rate, and slower position fixes pull the estimate back.

- GpsImuEstimator: GPS position + velocity fixes. Velocity errors also
  correct roll/pitch, because a tilt error shows up as a horizontal
  acceleration error (velocity-aided attitude). Yaw comes from the gyro alone.
- MocapImuEstimator: motion-capture position + attitude fixes.

They publish one estimate per IMU message.
"""

from __future__ import annotations

import numpy as np

from ..dynamics import State, quat_multiply, quat_to_rot
from .core import Node
from .msgs import GpsFix, Imu, MocapPose, StateEstimate


def _quat_exp(rotvec: np.ndarray) -> np.ndarray:
    angle = np.linalg.norm(rotvec)
    if angle < 1e-12:
        return np.r_[1.0, 0.5 * rotvec]
    axis = rotvec / angle
    return np.r_[np.cos(angle / 2), np.sin(angle / 2) * axis]


class InertialEstimator(Node):
    kind = "estimator"

    def __init__(self, name: str | None = None, *, k_pos: float = 0.3, k_vel_from_pos: float = 0.5,
                 **kw):
        super().__init__(name, **kw)
        self.k_pos = k_pos  # fraction of a position error removed per fix
        self.k_vel_from_pos = k_vel_from_pos  # velocity correction (1/s) per meter of position error

    def setup(self):
        self.pub = self.create_publisher("state_estimate", StateEstimate)
        self.create_subscription("imu", Imu, self._on_imu)
        self.ready = False
        self.p = np.zeros(3)
        self.v = np.zeros(3)
        self.q = np.array([1.0, 0.0, 0.0, 0.0])
        self.last_imu: float | None = None
        self.last_fix: float | None = None

    def _on_imu(self, msg: Imu) -> None:
        dt = 0.0 if self.last_imu is None else msg.stamp - self.last_imu
        self.last_imu = msg.stamp
        if not self.ready:
            return
        if dt > 0:
            self.q = quat_multiply(self.q, _quat_exp(msg.gyro * dt))
            self.q /= np.linalg.norm(self.q)
            a = quat_to_rot(self.q) @ msg.accel - np.array([0.0, 0.0, self.drone_params.gravity])
            self.p = self.p + self.v * dt + 0.5 * a * dt * dt
            self.v = self.v + a * dt
        self.pub.publish(StateEstimate(msg.stamp, State(self.p.copy(), self.v.copy(), self.q.copy(), msg.gyro.copy())))

    def _fix_dt(self, stamp: float) -> float:
        dt = 0.0 if self.last_fix is None else stamp - self.last_fix
        self.last_fix = stamp
        return dt

    def _correct_position(self, pos: np.ndarray, dt: float) -> None:
        err = pos - self.p
        self.p = self.p + self.k_pos * err
        self.v = self.v + self.k_vel_from_pos * dt * err


class GpsImuEstimator(InertialEstimator):
    def __init__(self, name: str = "gps_imu_estimator", *, k_vel: float = 0.3, k_tilt: float = 0.02, **kw):
        super().__init__(name, **kw)
        self.k_vel = k_vel  # fraction of a velocity error removed per fix
        self.k_tilt = k_tilt  # rad of tilt correction per m/s of velocity error

    def setup(self):
        super().setup()
        self.create_subscription("gps", GpsFix, self._on_gps)

    def _on_gps(self, msg: GpsFix) -> None:
        dt = self._fix_dt(msg.stamp)
        if not self.ready:  # assume a level start, heading +x
            self.p, self.v, self.ready = msg.pos.copy(), msg.vel.copy(), True
            return
        dv = msg.vel - self.v
        # The estimate drifts toward where its thrust axis is tilted, so
        # rotate the thrust axis toward the velocity error to cancel it.
        tilt = self.k_tilt * np.cross([0.0, 0.0, 1.0], dv)
        self.q = quat_multiply(_quat_exp(tilt), self.q)
        self.q /= np.linalg.norm(self.q)
        self.v = self.v + self.k_vel * dv
        self._correct_position(msg.pos, dt)


class MocapImuEstimator(InertialEstimator):
    def __init__(self, name: str = "mocap_imu_estimator", *, k_att: float = 0.2, **kw):
        super().__init__(name, **kw)
        self.k_att = k_att  # fraction of an attitude error removed per fix

    def setup(self):
        super().setup()
        self.create_subscription("mocap", MocapPose, self._on_mocap)

    def _on_mocap(self, msg: MocapPose) -> None:
        dt = self._fix_dt(msg.stamp)
        if not self.ready:
            self.p, self.q, self.ready = msg.pos.copy(), msg.quat.copy(), True
            return
        q = msg.quat if np.dot(msg.quat, self.q) >= 0 else -msg.quat  # same hemisphere
        self.q = self.q + self.k_att * (q - self.q)  # small-step nlerp
        self.q /= np.linalg.norm(self.q)
        self._correct_position(msg.pos, dt)
