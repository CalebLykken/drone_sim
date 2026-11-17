"""Matplotlib 3D animation and plots for a SimResult."""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation

from .dynamics import quat_to_rot
from .sim import SimResult

DRAW_SCALE = 2.0  # draw the drone bigger than life so it's visible
ROTOR_COLORS = ["tab:red", "tab:red", "0.3", "0.3"]  # front rotors red


def _rotor_ring(center, radius, n=16):
    a = np.linspace(0, 2 * np.pi, n)
    return center + radius * np.column_stack([np.cos(a), np.sin(a), np.zeros(n)])


def animate(result: SimResult, speed: float = 1.0, save: str | None = None, fps: int = 30):
    """Animate the flight in 3D next to time-series plots. Returns the animation."""
    t, pos, tgt = result.t, result.pos, result.targets
    eul = np.degrees(result.euler)
    eul[:, 2] = np.degrees(np.unwrap(result.euler[:, 2]))  # no jumps at +-180
    rotors_body = result.params.rotor_positions * DRAW_SCALE
    ring_r = 0.35 * result.params.arm_length * DRAW_SCALE
    rings_body = [_rotor_ring(r, ring_r) for r in rotors_body]

    fig = plt.figure(figsize=(14, 7))
    gs = fig.add_gridspec(4, 2, width_ratios=[1.4, 1])
    ax3 = fig.add_subplot(gs[:, 0], projection="3d")
    ax_p, ax_e, ax_a, ax_u = (fig.add_subplot(gs[i, 1]) for i in range(4))

    # World box: fit the flight and target path, keep true proportions.
    pts = np.vstack([pos, tgt[:, :3]])
    lo = pts.min(axis=0) - 0.75
    hi = pts.max(axis=0) + 0.75
    lo[2] = 0.0
    ax3.set_xlim(lo[0], hi[0])
    ax3.set_ylim(lo[1], hi[1])
    ax3.set_zlim(lo[2], hi[2])
    ax3.set_box_aspect(hi - lo)
    ax3.set_xlabel("x [m]")
    ax3.set_ylabel("y [m]")
    ax3.set_zlabel("z [m]")
    ax3.set_title(f"scenario: {result.config.scenario}   (drag to rotate)")
    ax3.plot(tgt[:, 0], tgt[:, 1], tgt[:, 2], ":", color="tab:red", alpha=0.35, lw=1)

    trail, = ax3.plot([], [], [], "-", color="tab:blue", alpha=0.6, lw=1)
    shadow, = ax3.plot([], [], [], "-", color="0.6", alpha=0.5, lw=1)
    target_pt, = ax3.plot([], [], [], "x", color="tab:red", ms=10, mew=2)
    arms = [ax3.plot([], [], [], "-", color="k", lw=3)[0] for _ in range(2)]
    rings = [ax3.plot([], [], [], "-", color=c, lw=2)[0] for c in ROTOR_COLORS]
    up, = ax3.plot([], [], [], "-", color="tab:green", lw=1.5)
    info = ax3.text2D(0.02, 0.97, "", transform=ax3.transAxes, va="top", family="monospace")

    def series(ax, ys, labels, ylabel, styles=None):
        styles = styles or ["-"] * len(ys)
        for y, label, st in zip(ys, labels, styles):
            ax.plot(t, y, st, label=label, lw=1.1)
        ax.set_ylabel(ylabel)
        ax.legend(loc="upper right", fontsize=7, ncol=3)
        ax.grid(alpha=0.3)
        return ax.axvline(0, color="k", lw=0.8)

    ax_p.set_prop_cycle(color=["tab:blue", "tab:orange", "tab:green"] * 2)
    err = np.linalg.norm(pos - tgt[:, :3], axis=1)
    cursors = [
        series(ax_p, [*pos.T, *tgt[:, :3].T], ["x", "y", "z", "", "", ""], "position [m]",
               ["-"] * 3 + ["--"] * 3),
        series(ax_e, [err], ["distance to target"], "error [m]"),
        series(ax_a, [*eul.T, np.degrees(np.unwrap(tgt[:, 3]))],
               ["roll", "pitch", "yaw", "yaw target"], "attitude [deg]", ["-", "-", "-", ":"]),
        series(ax_u, list(result.thrusts.T), ["FL", "FR", "BR", "BL"], "thrust [N]"),
    ]
    ax_u.axhline(result.params.max_thrust, color="tab:red", ls=":", lw=1)
    ax_u.set_xlabel("time [s]")
    fig.tight_layout()

    frame_dt = speed / fps
    frames = np.searchsorted(t, np.arange(0, t[-1], frame_dt))

    def draw(k):
        p = pos[k]
        R = quat_to_rot(result.states[k, 6:10])
        world = rotors_body @ R.T + p
        for arm, (i, j) in zip(arms, [(0, 2), (1, 3)]):
            seg = world[[i, j]]
            arm.set_data_3d(seg[:, 0], seg[:, 1], seg[:, 2])
        for ring, rb in zip(rings, rings_body):
            w = rb @ R.T + p
            ring.set_data_3d(w[:, 0], w[:, 1], w[:, 2])
        tip = p + R[:, 2] * 0.3 * DRAW_SCALE
        up.set_data_3d([p[0], tip[0]], [p[1], tip[1]], [p[2], tip[2]])
        trail.set_data_3d(pos[: k + 1, 0], pos[: k + 1, 1], pos[: k + 1, 2])
        shadow.set_data_3d(pos[: k + 1, 0], pos[: k + 1, 1], np.zeros(k + 1))
        target_pt.set_data_3d([tgt[k, 0]], [tgt[k, 1]], [tgt[k, 2]])
        wx, wy, _ = result.wind[k]
        info.set_text(f"t = {t[k]:5.2f} s\nwind = ({wx:+.2f}, {wy:+.2f}) N")
        for c in cursors:
            c.set_xdata([t[k], t[k]])
        return []

    anim = FuncAnimation(fig, draw, frames=frames, interval=1000 / fps, blit=False, repeat=False)
    if save:
        writer = "pillow" if save.endswith(".gif") else "ffmpeg"
        anim.save(save, writer=writer, fps=fps)
    return anim
