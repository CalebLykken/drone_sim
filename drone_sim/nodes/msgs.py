"""Message types. Every message carries `stamp`, the sim time it describes.

Define your own the same way: a frozen dataclass with a `stamp` field.
"""

from dataclasses import dataclass

import numpy as np

from ..controller import Neighbor, Sticks
from ..dynamics import State


@dataclass(frozen=True)
class GroundTruth:
    """The true state, published by the simulator bridge. Simulated sensors
    read it; flight software should not (use a sensor instead)."""

    stamp: float
    state: State
    accel: np.ndarray  # world-frame acceleration, m/s^2


@dataclass(frozen=True)
class Imu:
    stamp: float
    accel: np.ndarray  # specific force in the body frame, m/s^2 (reads +g at rest)
    gyro: np.ndarray  # body angular rate, rad/s


@dataclass(frozen=True)
class GpsFix:
    stamp: float
    pos: np.ndarray  # world position, m
    vel: np.ndarray  # world velocity, m/s


@dataclass(frozen=True)
class MocapPose:
    """Motion-capture pose (e.g. Vicon/OptiTrack)."""

    stamp: float
    pos: np.ndarray
    quat: np.ndarray  # (w, x, y, z), body -> world


@dataclass(frozen=True)
class StateEstimate:
    stamp: float
    state: State


@dataclass(frozen=True)
class Setpoint:
    stamp: float
    target: tuple[float, float, float, float]  # x, y, z, yaw


@dataclass(frozen=True)
class MotorCommand:
    stamp: float
    thrusts: np.ndarray  # newtons, FL FR BR BL


@dataclass(frozen=True)
class SticksInput:
    stamp: float
    sticks: Sticks


@dataclass(frozen=True)
class SwarmState:
    """The other drones in the simulation (like a shared swarm broadcast)."""

    stamp: float
    neighbors: tuple[Neighbor, ...]
