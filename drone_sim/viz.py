"""Matplotlib animation and plots for a SimResult."""

import math

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation

from .sim import SimResult


def _drone_shape(x, z, theta, arm):
    """Endpoints of the arm and the two rotor bars, in world coordinates."""
    c, s = math.cos(theta), math.sin(theta)
    # Body frame points: arm from -arm to +arm, rotors slightly above each end.
    body = np.array([[-arm, 0], [arm, 0]])
    rot_l = np.array([[-arm - 0.06, 0.04], [-arm + 0.06, 0.04]])
    rot_r = np.array([[arm - 0.06, 0.04], [arm + 0.06, 0.04]])
    R = np.array([[c, -s], [s, c]])
    return [pts @ R.T + [x, z] for pts in (body, rot_l, rot_r)]


def animate(result: SimResult, speed: float = 1.0, save: str | None = None, fps: int = 30):
    """Animate the flight next to time-series plots. Returns the animation."""
    t, s, tgt = result.t, result.states, result.targets
    arm = result.params.arm_length * 2  # drawn 2x size so it's visible

    fig = plt.figure(figsize=(13, 6.5))
    gs = fig.add_gridspec(4, 2, width_ratios=[1.3, 1])
    ax_w = fig.add_subplot(gs[:, 0])
    ax_x, ax_z, ax_th, ax_u = (fig.add_subplot(gs[i, 1]) for i in range(4))

    pad = 1.0
    xs = np.concatenate([s[:, 0], tgt[:, 0]])
    zs = np.concatenate([s[:, 1], tgt[:, 1]])
    ax_w.set_xlim(xs.min() - pad, xs.max() + pad)
    ax_w.set_ylim(-0.2, zs.max() + pad)
    ax_w.set_aspect("equal")
    ax_w.axhspan(-1, 0, color="0.85")
    ax_w.set_xlabel("x [m]")
    ax_w.set_ylabel("z [m]")
    ax_w.set_title(f"scenario: {result.config.scenario}")

    trail, = ax_w.plot([], [], "-", color="tab:blue", alpha=0.4, lw=1)
    target_pt, = ax_w.plot([], [], "x", color="tab:red", ms=12, mew=2)
    arm_line, = ax_w.plot([], [], "-", color="k", lw=3)
    rotor_l, = ax_w.plot([], [], "-", color="tab:orange", lw=4)
    rotor_r, = ax_w.plot([], [], "-", color="tab:orange", lw=4)
    wind_arrow = ax_w.annotate("", xy=(0, 0), xytext=(0, 0), xycoords="axes fraction",
                               arrowprops=dict(arrowstyle="->", color="tab:cyan", lw=2))
    info = ax_w.text(0.02, 0.97, "", transform=ax_w.transAxes, va="top", family="monospace")

    def series(ax, ys, labels, ylabel):
        for y, label in zip(ys, labels):
            ax.plot(t, y, label=label, lw=1.2)
        ax.set_ylabel(ylabel)
        ax.legend(loc="upper right", fontsize=7)
        ax.grid(alpha=0.3)
        return ax.axvline(0, color="k", lw=0.8)

    cursors = [
        series(ax_x, [s[:, 0], tgt[:, 0]], ["x", "target"], "x [m]"),
        series(ax_z, [s[:, 1], tgt[:, 1]], ["z", "target"], "z [m]"),
        series(ax_th, [np.degrees(s[:, 2])], ["theta"], "tilt [deg]"),
        series(ax_u, [result.thrusts[:, 0], result.thrusts[:, 1]], ["left", "right"], "thrust [N]"),
    ]
    ax_u.axhline(result.params.max_thrust, color="tab:red", ls=":", lw=1)
    ax_u.set_xlabel("time [s]")
    fig.tight_layout()

    frame_dt = speed / fps
    frames = np.searchsorted(t, np.arange(0, t[-1], frame_dt))

    def draw(k):
        x, z, th = s[k, :3]
        arm_pts, l_pts, r_pts = _drone_shape(x, z, th, arm)
        arm_line.set_data(*arm_pts.T)
        rotor_l.set_data(*l_pts.T)
        rotor_r.set_data(*r_pts.T)
        trail.set_data(s[: k + 1, 0], s[: k + 1, 1])
        target_pt.set_data([tgt[k, 0]], [tgt[k, 1]])
        w = result.wind[k]
        wind_arrow.set_position((0.85, 0.93))
        wind_arrow.xy = (0.85 + 0.03 * w, 0.93)
        info.set_text(f"t = {t[k]:5.2f} s\nwind = {w:+.2f} N")
        for c in cursors:
            c.set_xdata([t[k], t[k]])
        return [arm_line, rotor_l, rotor_r, trail, target_pt, wind_arrow, info, *cursors]

    anim = FuncAnimation(fig, draw, frames=frames, interval=1000 / fps, blit=False, repeat=False)
    if save:
        writer = "pillow" if save.endswith(".gif") else "ffmpeg"
        anim.save(save, writer=writer, fps=fps)
    return anim
