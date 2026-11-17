"""Real-time flight with keyboard input.

The simulator runs as fast as the wall clock, the keyboard sets the
controller's `sticks`, and a chase view follows the drone.
"""

import time

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation

from .controller import Sticks
from .dynamics import quat_to_rot
from .sim import Simulator
from .viz import DRAW_SCALE, ROTOR_COLORS, _rotor_ring

# key -> (stick, direction)
KEYS = {
    "w": ("pitch", 1), "s": ("pitch", -1),
    "d": ("roll", 1), "a": ("roll", -1),
    "up": ("throttle", 1), "down": ("throttle", -1),
    "right": ("yaw", 1), "left": ("yaw", -1),
}
HELP = "W/S fwd/back  A/D left/right  ↑/↓ up/down  ←/→ turn  R reset  Esc quit"
STICK_TIME = 0.15  # s, how quickly sticks follow the keys (keys are all-or-nothing)
VIEW = 4.0  # m, half-width of the chase view
TRAIL = 2000  # control steps of trail to draw


def fly(sim: Simulator, fps: int = 30):
    """Open a window and fly `sim` in real time until it's closed."""
    # Free up keys that matplotlib would otherwise use (s = save, a/d/l/k = axes, ...).
    for name in plt.rcParams:
        if name.startswith("keymap."):
            plt.rcParams[name] = []

    params = sim.params
    rotors_body = params.rotor_positions * DRAW_SCALE
    rings_body = [_rotor_ring(r, 0.35 * params.arm_length * DRAW_SCALE) for r in rotors_body]

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(projection="3d")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_zlabel("z [m]")
    ax.set_title(HELP, fontsize=9)

    trail, = ax.plot([], [], [], "-", color="tab:blue", alpha=0.6, lw=1)
    shadow, = ax.plot([], [], [], "o", color="0.4", alpha=0.5, ms=8)
    drop, = ax.plot([], [], [], ":", color="0.5", lw=1)
    target_pt, = ax.plot([], [], [], "x", color="tab:red", ms=10, mew=2)
    arms = [ax.plot([], [], [], "-", color="k", lw=3)[0] for _ in range(2)]
    rings = [ax.plot([], [], [], "-", color=c, lw=2)[0] for c in ROTOR_COLORS]
    up, = ax.plot([], [], [], "-", color="tab:green", lw=1.5)
    hud = ax.text2D(0.02, 0.97, "", transform=ax.transAxes, va="top", family="monospace")

    pressed: set[str] = set()
    sticks = dict.fromkeys(("throttle", "roll", "pitch", "yaw"), 0.0)
    clock = {"last": time.perf_counter(), "debt": 0.0}

    def on_press(event):
        key = (event.key or "").lower()
        if key == "r":
            sim.reset()
            for k in sticks:
                sticks[k] = 0.0
        elif key == "escape":
            plt.close(fig)
        else:
            pressed.add(key)

    def on_release(event):
        pressed.discard((event.key or "").lower())

    fig.canvas.mpl_connect("key_press_event", on_press)
    fig.canvas.mpl_connect("key_release_event", on_release)
    # Releasing a key outside the window never reaches us, so forget held keys.
    fig.canvas.mpl_connect("figure_leave_event", lambda e: pressed.clear())

    def frame(_):
        now = time.perf_counter()
        elapsed = min(now - clock["last"], 0.1)  # don't try to catch up after a stall
        clock["last"] = now

        # Keys -> smoothed sticks.
        goal = dict.fromkeys(sticks, 0.0)
        for key in pressed:
            if key in KEYS:
                name, sign = KEYS[key]
                goal[name] += sign
        blend = min(1.0, elapsed / STICK_TIME)
        for k in sticks:
            sticks[k] += blend * (np.clip(goal[k], -1, 1) - sticks[k])
        sim.controller.sticks = Sticks(**sticks)

        # Advance the sim by however much wall time passed.
        clock["debt"] += elapsed
        while clock["debt"] >= sim.control_dt:
            sim.step()
            clock["debt"] -= sim.control_dt

        s = sim.state
        p, R = s.pos, s.R
        world = rotors_body @ R.T + p
        for arm, (i, j) in zip(arms, [(0, 2), (1, 3)]):
            seg = world[[i, j]]
            arm.set_data_3d(seg[:, 0], seg[:, 1], seg[:, 2])
        for ring, rb in zip(rings, rings_body):
            w = rb @ R.T + p
            ring.set_data_3d(w[:, 0], w[:, 1], w[:, 2])
        tip = p + R[:, 2] * 0.3 * DRAW_SCALE
        up.set_data_3d([p[0], tip[0]], [p[1], tip[1]], [p[2], tip[2]])
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

        hist = np.array(sim._log["state"][-TRAIL:]).reshape(-1, 13)[:, :3]
        trail.set_data_3d(*clip(hist))
        tx, ty, tz, _ = sim.target
        target_pt.set_data_3d(*clip([tx, ty, tz]))

        roll, pitch, yaw = np.degrees(s.euler)
        dist = np.linalg.norm(np.array([tx, ty, tz]) - p)
        hud.set_text(
            f"t {sim.t:6.1f} s   alt {p[2]:5.2f} m   speed {np.linalg.norm(s.vel):4.1f} m/s\n"
            f"roll {roll:+5.0f}°  pitch {pitch:+5.0f}°  heading {yaw:+5.0f}°\n"
            f"distance to ✕ {dist:5.2f} m\n"
            f"sticks  thr {sticks['throttle']:+.1f}  roll {sticks['roll']:+.1f}  "
            f"pitch {sticks['pitch']:+.1f}  yaw {sticks['yaw']:+.1f}"
        )
        return []

    anim = FuncAnimation(fig, frame, interval=1000 / fps, blit=False, cache_frame_data=False)
    plt.show()
    return anim
