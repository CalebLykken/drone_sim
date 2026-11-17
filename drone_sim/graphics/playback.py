"""Matplotlib 3D animation and plots for finished flights."""

from collections.abc import Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation

from ..dynamics import quat_to_rot
from ..sim import Collision, SimResult
from .drone import DroneArtist, drone_color

COLLISION_FLASH = 1.0  # s, how long a collision is drawn big


def animate(
    results: SimResult | Sequence[SimResult],
    speed: float = 1.0,
    save: str | None = None,
    fps: int = 30,
    collisions: Sequence[Collision] = (),
):
    """Animate one or more drones in 3D next to time-series plots.

    With one drone the plots show its position, error, attitude, and each
    rotor; with several they compare the drones (one color each). Returns the
    animation.
    """
    if isinstance(results, SimResult):
        results = [results]
    results = list(results)
    multi = len(results) > 1
    t = results[0].t
    colors = [drone_color(i) for i in range(len(results))]

    fig = plt.figure(figsize=(14, 7))
    gs = fig.add_gridspec(4, 2, width_ratios=[1.4, 1])
    ax3 = fig.add_subplot(gs[:, 0], projection="3d")
    plot_axes = [fig.add_subplot(gs[i, 1]) for i in range(4)]

    # World box: fit every flight and target path, keep true proportions.
    pts = np.vstack([np.vstack([r.pos, r.targets[:, :3]]) for r in results])
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
    count = f"   {len(results)} drones" if multi else ""
    ax3.set_title(f"scenario: {results[0].config.scenario}{count}   (drag to rotate)")

    drones = []
    for r, color in zip(results, colors):
        tgt = r.targets
        ax3.plot(tgt[:, 0], tgt[:, 1], tgt[:, 2], ":", color="tab:red", alpha=0.35, lw=1)
        drones.append({
            "result": r,
            "artist": DroneArtist(ax3, r.params, color=color if multi else "k"),
            "trail": ax3.plot([], [], [], "-", color=color, alpha=0.6, lw=1)[0],
            "shadow": ax3.plot([], [], [], "-", color="0.6", alpha=0.5, lw=1)[0],
            "target": ax3.plot([], [], [], "x", color="tab:red" if not multi else color, ms=10, mew=2)[0],
            "label": ax3.text(0, 0, 0, r.name, fontsize=8, color=color) if multi else None,
        })
    flash = ax3.plot([], [], [], "*", color="tab:red", ms=24)[0]
    marks = ax3.plot([], [], [], "*", color="tab:red", ms=10, alpha=0.6)[0]
    info = ax3.text2D(0.02, 0.97, "", transform=ax3.transAxes, va="top", family="monospace")

    # Where each collision happened (midpoint of the two drones at that step).
    by_name = {r.name: r for r in results}
    hits = []
    for c in collisions:
        if c.a in by_name and c.b in by_name:
            k = min(np.searchsorted(t, c.t), len(t) - 1)
            hits.append((c, (by_name[c.a].pos[k] + by_name[c.b].pos[k]) / 2))

    cursors = (_multi_plots if multi else _single_plots)(plot_axes, results, colors)
    plot_axes[-1].set_xlabel("time [s]")
    fig.tight_layout()

    frame_dt = speed / fps
    frames = np.searchsorted(t, np.arange(0, t[-1], frame_dt))

    def draw(k):
        now = t[k]
        for d in drones:
            r = d["result"]
            pos = r.pos
            crashed = r.crashed_at is not None and now >= r.crashed_at
            d["artist"].update(pos[k], quat_to_rot(r.states[k, 6:10]), crashed=crashed)
            d["trail"].set_data_3d(pos[: k + 1, 0], pos[: k + 1, 1], pos[: k + 1, 2])
            d["shadow"].set_data_3d(pos[: k + 1, 0], pos[: k + 1, 1], np.zeros(k + 1))
            d["target"].set_data_3d([r.targets[k, 0]], [r.targets[k, 1]], [r.targets[k, 2]])
            d["target"].set_visible(not crashed)
            if d["label"] is not None:
                d["label"].set_position((pos[k, 0], pos[k, 1]))
                d["label"].set_3d_properties(pos[k, 2] + 0.35, None)
                d["label"].set_text(r.name + (" (crashed)" if crashed else ""))

        past = [p for c, p in hits if c.t <= now]
        recent = [p for c, p in hits if c.t <= now < c.t + COLLISION_FLASH]
        marks.set_data_3d(*(np.array(past).T if past else ([], [], [])))
        flash.set_data_3d(*(np.array(recent).T if recent else ([], [], [])))

        wx, wy, _ = results[0].wind[k]
        lines = [f"t = {now:5.2f} s", f"wind = ({wx:+.2f}, {wy:+.2f}) N"]
        crashed = sum(r.crashed_at is not None and now >= r.crashed_at for r in results)
        if crashed:
            lines.append(f"crashed: {crashed}/{len(results)}")
        lines += [f"collision: {c.a} + {c.b} at {c.t:.2f} s" for c, _ in hits if c.t <= now]
        info.set_text("\n".join(lines))
        for c in cursors:
            c.set_xdata([now, now])
        return []

    anim = FuncAnimation(fig, draw, frames=frames, interval=1000 / fps, blit=False, repeat=False)
    if save:
        writer = "pillow" if save.endswith(".gif") else "ffmpeg"
        anim.save(save, writer=writer, fps=fps)
    return anim


def _series(ax, t, ys, labels, ylabel, styles=None, colors=None):
    styles = styles or ["-"] * len(ys)
    for i, (y, label, st) in enumerate(zip(ys, labels, styles)):
        kw = {"color": colors[i]} if colors else {}
        ax.plot(t, y, st, label=label, lw=1.1, **kw)
    ax.set_ylabel(ylabel)
    ax.legend(loc="upper right", fontsize=7, ncol=3)
    ax.grid(alpha=0.3)
    return ax.axvline(0, color="k", lw=0.8)


def _single_plots(axes, results, colors):
    """One drone: position, error, attitude, and each rotor."""
    (r,) = results
    t, pos, tgt = r.t, r.pos, r.targets
    eul = np.degrees(r.euler)
    eul[:, 2] = np.degrees(np.unwrap(r.euler[:, 2]))  # no jumps at +-180
    ax_p, ax_e, ax_a, ax_u = axes
    ax_p.set_prop_cycle(color=["tab:blue", "tab:orange", "tab:green"] * 2)
    err = np.linalg.norm(pos - tgt[:, :3], axis=1)
    cursors = [
        _series(ax_p, t, [*pos.T, *tgt[:, :3].T], ["x", "y", "z", "", "", ""], "position [m]",
                ["-"] * 3 + ["--"] * 3),
        _series(ax_e, t, [err], ["distance to target"], "error [m]"),
        _series(ax_a, t, [*eul.T, np.degrees(np.unwrap(tgt[:, 3]))],
                ["roll", "pitch", "yaw", "yaw target"], "attitude [deg]", ["-", "-", "-", ":"]),
        _series(ax_u, t, list(r.thrusts.T), ["FL", "FR", "BR", "BL"], "thrust [N]"),
    ]
    ax_u.axhline(r.params.max_thrust, color="tab:red", ls=":", lw=1)
    return cursors


def _multi_plots(axes, results, colors):
    """Several drones: one line per drone for altitude, error, tilt, total thrust."""
    t = results[0].t
    names = [r.name for r in results]
    altitude = [r.pos[:, 2] for r in results]
    error = [np.linalg.norm(r.pos - r.targets[:, :3], axis=1) for r in results]
    tilt = [
        np.degrees(np.arccos(np.clip(1 - 2 * (r.states[:, 7] ** 2 + r.states[:, 8] ** 2), -1, 1)))
        for r in results
    ]
    thrust = [r.thrusts.sum(axis=1) for r in results]
    ax_z, ax_e, ax_a, ax_u = axes
    cursors = [
        _series(ax_z, t, altitude, names, "altitude [m]", colors=colors),
        _series(ax_e, t, error, names, "error [m]", colors=colors),
        _series(ax_a, t, tilt, names, "tilt [deg]", colors=colors),
        _series(ax_u, t, thrust, names, "total thrust [N]", colors=colors),
    ]
    for r, color in zip(results, colors):
        if r.crashed_at is not None:
            for ax in axes:
                ax.axvline(r.crashed_at, color=color, ls=":", lw=1)
    # With many drones, legends would cover the plots; the 3D labels name them.
    if len(results) > 6:
        for ax in axes:
            ax.get_legend().remove()
    return cursors
