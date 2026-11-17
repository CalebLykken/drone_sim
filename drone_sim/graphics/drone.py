"""The drone model drawn on a matplotlib 3D axes."""

import numpy as np

from ..dynamics import DroneParams

DRAW_SCALE = 2.0  # draw the drone bigger than life so it's visible
ROTOR_COLORS = ["tab:red", "tab:red", "0.3", "0.3"]  # front rotors red


def _rotor_ring(center, radius, n=16):
    a = np.linspace(0, 2 * np.pi, n)
    return center + radius * np.column_stack([np.cos(a), np.sin(a), np.zeros(n)])


class DroneArtist:
    """Arms, rotor rings, and a thrust-direction line for one drone."""

    def __init__(self, ax, params: DroneParams):
        self.rotors_body = params.rotor_positions * DRAW_SCALE
        ring_r = 0.35 * params.arm_length * DRAW_SCALE
        self.rings_body = [_rotor_ring(r, ring_r) for r in self.rotors_body]
        self.arms = [ax.plot([], [], [], "-", color="k", lw=3)[0] for _ in range(2)]
        self.rings = [ax.plot([], [], [], "-", color=c, lw=2)[0] for c in ROTOR_COLORS]
        self.up, = ax.plot([], [], [], "-", color="tab:green", lw=1.5)

    def update(self, pos: np.ndarray, R: np.ndarray) -> None:
        """Move to world position `pos` with body->world rotation `R`."""
        world = self.rotors_body @ R.T + pos
        for arm, (i, j) in zip(self.arms, [(0, 2), (1, 3)]):
            seg = world[[i, j]]
            arm.set_data_3d(seg[:, 0], seg[:, 1], seg[:, 2])
        for ring, rb in zip(self.rings, self.rings_body):
            w = rb @ R.T + pos
            ring.set_data_3d(w[:, 0], w[:, 1], w[:, 2])
        tip = pos + R[:, 2] * 0.3 * DRAW_SCALE
        self.up.set_data_3d([pos[0], tip[0]], [pos[1], tip[1]], [pos[2], tip[2]])
