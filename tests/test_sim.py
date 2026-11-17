"""Run with: python -m pytest tests/"""

import math
from pathlib import Path

import numpy as np
import pytest

from drone_sim import (
    Controller, DroneParams, DroneSpec, SimConfig, Simulator, State, Sticks,
    load_controller, offset_target, run, run_many,
)
from drone_sim.dynamics import euler_to_quat, quat_to_rot, rot_to_euler

ROOT = Path(__file__).resolve().parent.parent
FL, FR, BR, BL = range(4)


class Constant(Controller):
    def __init__(self, *cmd):
        self.cmd = cmd

    def update(self, t, state, target, dt):
        return self.cmd


def hover_plus(delta):
    """Hover thrust on every rotor, plus a per-rotor delta."""
    return Constant(*(DroneParams().hover_thrust + np.asarray(delta)))


def fly(controller, duration, z0=5.0):
    return run(controller, SimConfig(duration=duration, initial_state=State(pos=np.array([0, 0, z0]))))


def test_hover_thrust_holds_position():
    # Spin motors up instantly so the drone doesn't sag while they spool.
    res = run(hover_plus([0, 0, 0, 0]),
              SimConfig(duration=2.0, initial_state=State(pos=np.array([0, 0, 1.0]))),
              DroneParams(motor_tau=1e-6))
    assert res.pos[-1] == pytest.approx([0, 0, 1.0], abs=1e-3)
    assert np.abs(res.euler).max() < 1e-9


def test_zero_thrust_stays_on_ground():
    res = run(Constant(0, 0, 0, 0), SimConfig(duration=1.0))
    assert res.pos[:, 2].max() == 0.0


def test_front_thrust_pitches_nose_up_and_flies_backward():
    res = fly(hover_plus([0.05, 0.05, -0.05, -0.05]), 0.3)
    assert res.euler[-1, 1] < 0  # pitch < 0 is nose up
    assert res.pos[-1, 0] < 0


def test_left_thrust_rolls_right_side_down_and_flies_right():
    res = fly(hover_plus([0.05, -0.05, -0.05, 0.05]), 0.3)
    assert res.euler[-1, 0] > 0  # roll > 0 lifts the left (+y) side
    assert res.pos[-1, 1] < 0


def test_diagonal_pair_yaws_ccw():
    res = fly(hover_plus([0.2, -0.2, 0.2, -0.2]), 0.5)
    assert res.euler[-1, 2] > 0
    assert np.abs(res.euler[-1, :2]).max() < 1e-6
    assert np.abs(res.pos[-1, :2]).max() < 1e-6


def test_allocate_inverts_mixer():
    p = DroneParams()
    thrusts = p.allocate(10.0, [0.01, -0.02, 0.003])
    assert p.mixer @ thrusts == pytest.approx([10.0, 0.01, -0.02, 0.003])


def test_euler_round_trip():
    angles = np.array([0.3, -0.4, 2.0])
    assert rot_to_euler(quat_to_rot(euler_to_quat(*angles))) == pytest.approx(angles)


def test_thrust_saturates():
    res = run(Constant(100, 100, 100, 100), SimConfig(duration=0.5))
    assert res.thrusts.max() <= DroneParams().max_thrust + 1e-9


# Durations end each run ~3 s after the last target change, so the drone has settled.
@pytest.mark.parametrize(
    "scenario, duration",
    [("hover", 5.0), ("step", 6.0), ("square", 8.9), ("spin", 6.0)],
)
def test_example_pid_tracks(scenario, duration):
    cfg = SimConfig(scenario=scenario, duration=duration)
    res = run(load_controller(str(ROOT / "controllers" / "pid.py")), cfg)
    m = res.metrics()
    assert not m["flipped"]
    assert m["final_pos_error_m"] < 0.2
    yaw_err = (res.euler[-1, 2] - res.targets[-1, 3] + math.pi) % (2 * math.pi) - math.pi
    assert abs(yaw_err) < math.radians(5)


def test_hover_only_template_flies():
    res = run(load_controller(str(ROOT / "controllers" / "hover_only.py")),
              SimConfig(scenario="hover", duration=5.0))
    assert res.metrics()["final_pos_error_m"] < 0.05


def test_load_by_class_name():
    ctrl = load_controller(f"{ROOT / 'controllers' / 'pid.py'}:CascadedPID")
    assert type(ctrl).__name__ == "CascadedPID"


def test_load_rejects_ambiguous_file(tmp_path):
    f = tmp_path / "two.py"
    f.write_text(
        "from drone_sim import Controller\n"
        "class A(Controller): pass\n"
        "class B(Controller): pass\n"
    )
    with pytest.raises(ValueError, match="exactly one"):
        load_controller(str(f))
    assert type(load_controller(f"{f}:B")).__name__ == "B"


@pytest.mark.parametrize("cmd", [(math.nan, 1, 1, 1), (1, 1)])
def test_bad_controller_output_is_reported(cmd):
    with pytest.raises(ValueError, match="expected four finite thrusts"):
        run(Constant(*cmd), SimConfig(duration=0.1))


# --- Manual (stick) control ------------------------------------------------------

def fly_sticks(sim, seconds, **sticks):
    sim.controller.sticks = Sticks(**sticks)
    for _ in range(int(seconds / sim.control_dt)):
        sim.step()


def manual_sim():
    return Simulator(load_controller(str(ROOT / "controllers" / "manual.py")), SimConfig())


def test_manual_takes_off_and_hovers_when_sticks_released():
    sim = manual_sim()
    fly_sticks(sim, 2.0, throttle=1.0)
    assert sim.state.z > 1.5
    fly_sticks(sim, 2.0)  # brakes over a few tenths of a meter, then holds
    z = sim.state.z
    fly_sticks(sim, 2.0)
    assert np.linalg.norm(sim.state.vel) < 0.05
    assert sim.state.z == pytest.approx(z, abs=0.02)


def test_manual_sticks_move_relative_to_heading():
    sim = manual_sim()
    fly_sticks(sim, 1.5, throttle=1.0)
    fly_sticks(sim, 2.0, pitch=1.0)  # forward = +x at heading 0
    assert sim.state.vel[0] > 2.0
    fly_sticks(sim, 2.0)

    fly_sticks(sim, 1.0, yaw=1.0)  # turn right 90 degrees
    fly_sticks(sim, 1.0)
    assert np.degrees(sim.state.euler[2]) == pytest.approx(-90, abs=5)

    fly_sticks(sim, 2.0, pitch=1.0)  # now forward is -y
    assert sim.state.vel[1] < -2.0
    assert abs(sim.state.vel[0]) < 0.3

    fly_sticks(sim, 2.0)
    fly_sticks(sim, 2.0, roll=1.0)  # right of heading -90 is -x
    assert sim.state.vel[0] < -2.0


def test_simulator_reset_starts_over():
    sim = manual_sim()
    fly_sticks(sim, 1.0, throttle=1.0)
    sim.reset()
    assert sim.k == 0 and sim.state.z == 0.0 and not sim.result().t.size


# --- Separation between graphics and everything else -----------------------------

def test_core_and_controllers_do_not_import_graphics():
    """Controllers and the sim must run without matplotlib or drone_sim.graphics."""
    import subprocess
    import sys

    code = (
        "import sys, drone_sim, drone_sim.control, drone_sim.session\n"
        "from drone_sim import load_controller, run, SimConfig\n"
        "for name in ('pid', 'manual', 'hover_only'):\n"
        f"    run(load_controller({str(ROOT / 'controllers')!r} + f'/{{name}}.py'), SimConfig(duration=0.1))\n"
        "bad = sorted(m for m in sys.modules if m.startswith(('matplotlib', 'drone_sim.graphics')))\n"
        "assert not bad, bad\n"
    )
    subprocess.run([sys.executable, "-c", code], cwd=ROOT, check=True)


# --- LiveSession (real-time flight logic, no window) -------------------------------

def live_session():
    from drone_sim.session import LiveSession

    return LiveSession(manual_sim())


def tick_for(session, seconds, frame=1 / 30):
    for _ in range(int(seconds / frame)):
        session.tick(frame)


def test_live_session_keys_fly_the_drone():
    session = live_session()
    session.key_down("up")
    tick_for(session, 1.5)
    assert session.sticks.throttle == pytest.approx(1.0, abs=1e-3)
    assert session.sim.state.z > 1.0
    assert session.sim.t == pytest.approx(1.5, abs=0.05)  # sim time follows wall time

    session.key_up("up")
    session.key_down("W")  # case-insensitive
    tick_for(session, 1.5)
    assert session.sim.state.vel[0] > 2.0

    session.release_all()
    tick_for(session, 1.0)
    assert session.sticks.pitch == pytest.approx(0.0, abs=1e-3)


def test_live_session_caps_stalls_and_resets():
    session = live_session()
    session.tick(5.0)  # e.g. the window was dragged; don't jump ahead 5 s
    assert session.sim.t <= 0.1 + 1e-9
    session.key_down("up")
    tick_for(session, 0.5)
    session.key_down("r")
    assert session.sim.k == 0 and session.sticks == Sticks()
    assert "r" not in session.pressed


# --- Multiple drones ---------------------------------------------------------------

PID_PATH = str(ROOT / "controllers" / "pid.py")


def at(x, y=0.0, z=0.0):
    return State(pos=np.array([x, y, z], dtype=float))


def test_single_drone_api_matches_multi_drone_with_one_drone():
    cfg = SimConfig(scenario="square", duration=3.0, wind_gust=0.5, sensor_noise=0.002)
    single = run(load_controller(PID_PATH), cfg)
    (multi,), collisions = run_many([DroneSpec(load_controller(PID_PATH))], cfg)
    assert np.array_equal(single.states, multi.states)
    assert np.array_equal(single.commands, multi.commands)
    assert collisions == []


def test_formation_drones_fly_independently():
    scenario = lambda t: (0.0, 0.0, 1.0) if t < 1.5 else (1.0, 0.5, 1.5)  # noqa: E731
    specs = [
        DroneSpec(load_controller(PID_PATH), initial_state=at(0, y), target_fn=offset_target(scenario, (0, y, 0)))
        for y in (-1.0, 0.0, 1.0)
    ]
    results, collisions = run_many(specs, SimConfig(duration=4.0))
    assert collisions == []
    assert [r.name for r in results] == ["drone0", "drone1", "drone2"]
    # Same controller and wind, so identical flights shifted sideways.
    for r, y in zip(results, (-1.0, 0.0, 1.0)):
        assert r.pos - [0, y, 0] == pytest.approx(results[1].pos, abs=1e-9)
        assert r.metrics()["final_pos_error_m"] < 0.2


def test_each_drone_can_have_its_own_params():
    specs = [
        DroneSpec(load_controller(PID_PATH), initial_state=at(0)),
        DroneSpec(load_controller(PID_PATH), initial_state=at(2), params=DroneParams(mass=1.5),
                  target_fn=lambda t: (2.0, 0.0, 1.0)),
    ]
    sim = Simulator(drones=specs, config=SimConfig(scenario="hover"))
    for _ in range(250):
        sim.step()
    light, heavy = sim.results()
    assert heavy.params.mass == 1.5
    assert heavy.thrusts[-1].sum() > light.thrusts[-1].sum() * 1.3


class Recorder(Controller):
    def reset(self, params):
        self.seen = []
        self.hover = params.hover_thrust

    def update(self, t, state, target, dt):
        self.seen.append({n.name: n.state.x for n in self.others})
        return [self.hover] * 4


def test_controllers_see_the_other_drones():
    a, b = Recorder(), Recorder()
    sim = Simulator(drones=[DroneSpec(a, initial_state=at(0), name="alpha"),
                            DroneSpec(b, initial_state=at(3), name="bravo")])
    sim.step()
    assert (a.name, b.name) == ("alpha", "bravo")
    assert a.seen == [{"bravo": 3.0}] and b.seen == [{"alpha": 0.0}]
    assert sim.drone("bravo").controller is b
    assert run(Recorder(), SimConfig(duration=0.01)) is not None  # alone: others == ()


def head_on(**config):
    specs = [
        DroneSpec(load_controller(PID_PATH), initial_state=at(x), name=name,
                  target_fn=lambda t, x=x: (x, 0.0, 1.5) if t < 2 else (-x, 0.0, 1.5))
        for x, name in ((-2.0, "A"), (2.0, "B"))
    ]
    return run_many(specs, SimConfig(duration=6.0, **config))


def test_head_on_collision_crashes_both_drones():
    (a, b), collisions = head_on()
    assert len(collisions) == 1
    c = collisions[0]
    assert {c.a, c.b} == {"A", "B"} and 2.0 < c.t < 5.0
    assert a.crashed_at == b.crashed_at == c.t
    assert a.metrics()["crashed"] and b.metrics()["crashed"]
    after = a.t > c.t
    assert np.all(a.commands[after] == 0)  # motors are dead
    assert a.pos[-1, 2] == 0.0 and b.pos[-1, 2] == 0.0  # fell to the ground


def test_ghost_mode_records_collision_but_keeps_flying():
    (a, b), collisions = head_on(crash_on_collision=False)
    assert len(collisions) == 1  # recorded once, not every step they overlap
    assert not a.metrics()["crashed"]
    assert a.pos[-1, 0] == pytest.approx(2.0, abs=0.2)
    assert b.pos[-1, 0] == pytest.approx(-2.0, abs=0.2)


def test_multi_drone_setup_errors():
    c = lambda: load_controller(PID_PATH)  # noqa: E731
    with pytest.raises(ValueError, match="start 0.30 m apart"):
        Simulator(drones=[DroneSpec(c(), initial_state=at(0)), DroneSpec(c(), initial_state=at(0.3))])
    Simulator(drones=[DroneSpec(c(), initial_state=at(0)), DroneSpec(c(), initial_state=at(0.3))],
              config=SimConfig(crash_on_collision=False))
    with pytest.raises(ValueError, match="unique"):
        Simulator(drones=[DroneSpec(c(), initial_state=at(0), name="x"),
                          DroneSpec(c(), initial_state=at(2), name="x")])
    shared = c()
    with pytest.raises(ValueError, match="own controller"):
        Simulator(drones=[DroneSpec(shared, initial_state=at(0)), DroneSpec(shared, initial_state=at(2))])
    with pytest.raises(TypeError):
        Simulator(c(), drones=[DroneSpec(c())])
    with pytest.raises(TypeError):
        Simulator()


def test_live_session_pilots_the_first_drone():
    from drone_sim.session import LiveSession

    sim = Simulator(drones=[
        DroneSpec(load_controller(str(ROOT / "controllers" / "manual.py")), initial_state=at(0)),
        DroneSpec(load_controller(PID_PATH), initial_state=at(0, 2), target_fn=lambda t: (0, 2, 1)),
    ])
    session = LiveSession(sim)
    session.key_down("up")
    tick_for(session, 1.0)
    assert sim.drones[0].controller.sticks.throttle > 0.9
    assert sim.drones[1].controller.sticks == Sticks()
    assert sim.drones[0].state.z > 0.5 and sim.drones[1].state.z > 0.5


# --- Graphics smoke tests (headless) ---------------------------------------------

@pytest.fixture
def agg():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    yield plt
    plt.close("all")


def test_playback_draws_every_drone(agg):
    from drone_sim.graphics.playback import animate

    (a, b), collisions = head_on()
    anim = animate([a, b], collisions=collisions)
    ax3 = agg.gcf().axes[0]
    agg.gcf().canvas.draw()  # the first draw starts the animation at frame 0
    for k in (0, len(a.t) // 2, len(a.t) - 1):
        anim._func(k)
    labels = {t.get_text() for t in ax3.texts}
    assert {"A (crashed)", "B (crashed)"} <= labels
    stars = [l for l in ax3.lines if l.get_marker() == "*"]
    assert any(len(l.get_data_3d()[0]) for l in stars)  # collision marked
    assert len(agg.gcf().axes[1].lines) >= 2  # one line per drone in the plots


def test_playback_single_result_still_works(agg):
    from drone_sim.graphics.playback import animate

    anim = animate(run(load_controller(PID_PATH), SimConfig(duration=1.0)))
    anim._func(10)
    agg.gcf().canvas.draw()
    assert agg.gcf().axes[1].get_ylabel() == "position [m]"


@pytest.mark.filterwarnings("ignore:FigureCanvasAgg is non-interactive")
def test_live_window_draws_every_drone(agg):
    from drone_sim.graphics.live import fly
    from drone_sim.session import LiveSession

    sim = Simulator(drones=[
        DroneSpec(load_controller(str(ROOT / "controllers" / "manual.py")), initial_state=at(0), name="you"),
        DroneSpec(load_controller(PID_PATH), initial_state=at(1), name="near"),
        DroneSpec(load_controller(PID_PATH), initial_state=at(30), name="far"),
    ])
    anim = fly(LiveSession(sim))
    anim._func(0)
    agg.gcf().canvas.draw()
    labels = {t.get_text(): t.get_visible() for t in agg.gcf().axes[0].texts if t.get_text() in ("you", "near", "far")}
    assert labels == {"you": True, "near": True, "far": False}  # far is outside the chase view
