"""Matplotlib window for real-time keyboard flight.

All flight logic (keys -> sticks, stepping the sim) lives in LiveSession;
this module only forwards key events to it and draws `session.sim`.
"""

import time

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation

from ..session import HELP, LiveSession
from .drone import DroneArtist

VIEW = 4.0  # m, half-width of the chase view
TRAIL = 2000  # control steps of trail to draw


def fly(session: LiveSession, fps: int = 30):
    """Open a window and fly `session` in real time until it's closed."""
    # Free up keys that matplotlib would otherwise use (s = save, a/d/l/k = axes, ...).
    for name in plt.rcParams:
        if name.startswith("keymap."):
            plt.rcParams[name] = []

    sim = session.sim
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(projection="3d")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_zlabel("z [m]")
    ax.set_title(HELP + "  Esc quit", fontsize=9)

    trail, = ax.plot([], [], [], "-", color="tab:blue", alpha=0.6, lw=1)
    shadow, = ax.plot([], [], [], "o", color="0.4", alpha=0.5, ms=8)
    drop, = ax.plot([], [], [], ":", color="0.5", lw=1)
    target_pt, = ax.plot([], [], [], "x", color="tab:red", ms=10, mew=2)
    drone = DroneArtist(ax, sim.params)
    hud = ax.text2D(0.02, 0.97, "", transform=ax.transAxes, va="top", family="monospace")

    def on_press(event):
        if event.key == "escape":
            plt.close(fig)
        elif event.key:
            session.key_down(event.key)

    def on_release(event):
        if event.key:
            session.key_up(event.key)

    fig.canvas.mpl_connect("key_press_event", on_press)
    fig.canvas.mpl_connect("key_release_event", on_release)
    # Releasing a key outside the window never reaches us, so forget held keys.
    fig.canvas.mpl_connect("figure_leave_event", lambda e: session.release_all())

    last = [time.perf_counter()]

    def frame(_):
        now = time.perf_counter()
        session.tick(now - last[0])
        last[0] = now

        s = sim.state
        p = s.pos
        drone.update(p, s.R)
        shadow.set_data_3d([p[0]], [p[1]], [0.0])
        drop.set_data_3d([p[0], p[0]], [p[1], p[1]], [0.0, p[2]])

        # Chase view: keep the drone centered horizontally.
        top = max(1.5 * VIEW, p[2] + 1.0)
        ax.set_xlim(p[0] - VIEW, p[0] + VIEW)
        ax.set_ylim(p[1] - VIEW, p[1] + VIEW)
        ax.set_zlim(0.0, top)
        ax.set_box_aspect((1, 1, top / (2 * VIEW)))

        # 3D axes don't clip, so hide anything outside the view (NaN breaks the line).
        def clip(pts):
            pts = np.array(pts, dtype=float).reshape(-1, 3)
            outside = np.any(np.abs(pts[:, :2] - p[:2]) > VIEW, axis=1)
            pts[outside] = np.nan
            return pts.T

        trail.set_data_3d(*clip(sim.positions(last=TRAIL)))
        tx, ty, tz, _ = sim.target
        target_pt.set_data_3d(*clip([tx, ty, tz]))

        roll, pitch, yaw = np.degrees(s.euler)
        dist = np.linalg.norm(np.array([tx, ty, tz]) - p)
        st = session.sticks
        hud.set_text(
            f"t {sim.t:6.1f} s   alt {p[2]:5.2f} m   speed {np.linalg.norm(s.vel):4.1f} m/s\n"
            f"roll {roll:+5.0f}°  pitch {pitch:+5.0f}°  heading {yaw:+5.0f}°\n"
            f"distance to ✕ {dist:5.2f} m\n"
            f"sticks  thr {st.throttle:+.1f}  roll {st.roll:+.1f}  "
            f"pitch {st.pitch:+.1f}  yaw {st.yaw:+.1f}"
        )
        return []

    anim = FuncAnimation(fig, frame, interval=1000 / fps, blit=False, cache_frame_data=False)
    plt.show()
    return anim
