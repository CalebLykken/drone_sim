"""Matplotlib window for real-time keyboard flight.

All flight logic (keys -> sticks, stepping the sim) lives in LiveSession;
this module only forwards key events to it and draws `session.sim`.
"""

import time

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation

from ..session import HELP, LiveSession
from .drone import DroneArtist, drone_color

VIEW = 4.0  # m, half-width of the chase view
TRAIL = 2000  # control steps of trail to draw


def fly(session: LiveSession, fps: int = 30):
    """Open a window and fly `session` in real time until it's closed."""
    # Free up keys that matplotlib would otherwise use (s = save, a/d/l/k = axes, ...).
    for name in plt.rcParams:
        if name.startswith("keymap."):
            plt.rcParams[name] = []

    sim = session.sim
    multi = len(sim.drones) > 1
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(projection="3d")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_zlabel("z [m]")
    pilot_note = f"\nyou fly {sim.drones[0].name}; the camera follows it" if multi else ""
    ax.set_title(HELP + "  Esc quit" + pilot_note, fontsize=9)

    # One set of artists per drone. The first drone is the piloted one.
    drones = []
    for i, d in enumerate(sim.drones):
        color = drone_color(i)
        drones.append({
            "drone": d,
            "artist": DroneArtist(ax, d.params, color=color if multi else "k"),
            "trail": ax.plot([], [], [], "-", color=color, alpha=0.6, lw=1)[0],
            "shadow": ax.plot([], [], [], "o", color="0.4", alpha=0.5, ms=8)[0],
            "drop": ax.plot([], [], [], ":", color="0.5", lw=1)[0],
            "target": ax.plot([], [], [], "x", color=color if multi else "tab:red", ms=10, mew=2)[0],
            "label": ax.text(0, 0, 0, d.name, fontsize=8, color=color) if multi else None,
        })
    marks = ax.plot([], [], [], "*", color="tab:red", ms=14)[0]
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
    collision_pos: dict[tuple[str, str, float], np.ndarray] = {}

    def frame(_):
        now = time.perf_counter()
        session.tick(now - last[0])
        last[0] = now

        pilot = sim.drones[0]
        p = pilot.state.pos

        # Chase view: keep the piloted drone centered horizontally.
        top = max(1.5 * VIEW, p[2] + 1.0)
        ax.set_xlim(p[0] - VIEW, p[0] + VIEW)
        ax.set_ylim(p[1] - VIEW, p[1] + VIEW)
        ax.set_zlim(0.0, top)
        ax.set_box_aspect((1, 1, top / (2 * VIEW)))

        # 3D axes don't clip, so hide anything outside the view (NaN breaks the line).
        def outside(pts):
            return np.any(np.abs(np.asarray(pts).reshape(-1, 3)[:, :2] - p[:2]) > VIEW, axis=1)

        def clip(pts):
            pts = np.array(pts, dtype=float).reshape(-1, 3)
            pts[outside(pts)] = np.nan
            return pts.T

        for g in drones:
            d = g["drone"]
            q = d.state.pos
            g["artist"].update(q, d.state.R, crashed=d.crashed)
            visible = not outside(q)[0]
            g["artist"].set_visible(visible)
            g["shadow"].set_data_3d(*clip([q[0], q[1], 0.0]))
            g["drop"].set_data_3d(*clip([[q[0], q[1], 0.0], [q[0], q[1], q[2]]]))
            g["trail"].set_data_3d(*clip(d.positions(last=TRAIL)))
            g["target"].set_data_3d(*clip(d.target[:3]))
            g["target"].set_visible(not d.crashed)
            if g["label"] is not None:
                g["label"].set_visible(visible)
                g["label"].set_position((q[0], q[1]))
                g["label"].set_3d_properties(q[2] + 0.35, None)
                g["label"].set_text(d.name + (" (crashed)" if d.crashed else ""))

        # Remember where each collision happened (sessions can reset, so key by time too).
        for c in sim.collisions:
            key = (c.a, c.b, c.t)
            if key not in collision_pos:
                collision_pos[key] = (sim.drone(c.a).state.pos + sim.drone(c.b).state.pos) / 2
        if sim.k == 0:
            collision_pos.clear()
        hits = [collision_pos[(c.a, c.b, c.t)] for c in sim.collisions]
        marks.set_data_3d(*(clip(hits) if hits else ([], [], [])))

        s = pilot.state
        roll, pitch, yaw = np.degrees(s.euler)
        dist = np.linalg.norm(np.array(pilot.target[:3]) - p)
        st = session.sticks
        lines = [
            f"t {sim.t:6.1f} s   alt {p[2]:5.2f} m   speed {np.linalg.norm(s.vel):4.1f} m/s",
            f"roll {roll:+5.0f}\u00b0  pitch {pitch:+5.0f}\u00b0  heading {yaw:+5.0f}\u00b0",
            f"distance to \u2715 {dist:5.2f} m",
            f"sticks  thr {st.throttle:+.1f}  roll {st.roll:+.1f}  "
            f"pitch {st.pitch:+.1f}  yaw {st.yaw:+.1f}",
        ]
        if pilot.crashed:
            lines.append("CRASHED: press R to reset")
        if multi:
            crashed = sum(d.crashed for d in sim.drones)
            lines.append(f"drones {len(sim.drones)}   crashed {crashed}   collisions {len(sim.collisions)}")
        hud.set_text("\n".join(lines))
        return []

    anim = FuncAnimation(fig, frame, interval=1000 / fps, blit=False, cache_frame_data=False)
    plt.show()
    return anim
