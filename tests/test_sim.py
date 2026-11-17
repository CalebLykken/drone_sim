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
        "import sys, drone_sim, drone_sim.control, drone_sim.session, drone_sim.nodes\n"
        "from drone_sim import load_controller, run, SimConfig\n"
        "for name in ('pid', 'manual', 'hover_only', 'phri'):\n"
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


# --- Prosocial planning (PHRI) -------------------------------------------------------

PHRI_PATH = str(ROOT / "controllers" / "phri.py")


def phri_agents(params, a_start, a_goal, b_start, b_goal):
    from drone_sim.phri import GameAgent

    def agent(name, start, goal):
        return GameAgent(name, np.r_[start, 0, 0, 0].astype(float), np.r_[goal, 0, 0, 0].astype(float), params)

    return [agent("A", a_start, a_goal), agent("B", b_start, b_goal)]


def test_phri_head_on_both_keep_right_and_share_the_detour():
    pytest.importorskip("cvxpy")
    from drone_sim.phri import GameCache, PHRIParams, PHRIPlanner, convenience

    GameCache.clear()
    p = PHRIParams()
    planner = PHRIPlanner(p)
    for _ in range(3):  # a few receding-horizon replans from the same spot
        game = phri_agents(p, [-4, 0, 1.5], [4, 0, 1.5], [4, 0, 1.5], [-4, 0, 1.5])
        planner.plan(game)
    a, b = game
    assert planner.last_stats.failures == 0
    sep = np.linalg.norm(a.xs[:, :3] - b.xs[:, :3], axis=1)
    assert sep.min() > 0.9 * p.collision_radius
    # A flies +x so its right is -y; B flies -x so its right is +y. Equal shares.
    assert a.xs[-1, 1] < -0.3 and b.xs[-1, 1] > 0.3
    assert a.xs[-1, 1] == pytest.approx(-b.xs[-1, 1], abs=0.02)
    for agent in (a, b):  # the inconvenience budget holds
        assert convenience(agent.xs, agent.goal, p.inconvenience_weights) <= agent.budget + p.budget_floor + 1e-3


def test_phri_identical_games_are_shared_not_resolved():
    pytest.importorskip("cvxpy")
    from drone_sim.phri import GameCache, PHRIParams, PHRIPlanner

    GameCache.clear()
    p = PHRIParams()
    first, second = PHRIPlanner(p), PHRIPlanner(p)
    game_a = phri_agents(p, [-4, 0.3, 1.5], [4, 0, 1.5], [4, 0, 1.5], [-4, 0, 1.5])
    game_b = list(reversed(phri_agents(p, [-4, 0.3, 1.5], [4, 0, 1.5], [4, 0, 1.5], [-4, 0, 1.5])))
    first.plan(game_a)
    second.plan(game_b)  # B's point of view, same game
    assert not first.last_stats.cached and second.last_stats.cached
    by_name = {g.name: g for g in game_b}
    assert np.array_equal(by_name["A"].xs, game_a[0].xs)


def swap_flight(ctrl_a, ctrl_b, duration=10.0):
    specs = [
        DroneSpec(load_controller(ctrl_a), initial_state=at(-4), name="A",
                  target_fn=lambda t: (-4.0, 0.0, 1.5) if t < 2 else (4.0, 0.0, 1.5)),
        DroneSpec(load_controller(ctrl_b), initial_state=at(4), name="B",
                  target_fn=lambda t: (4.0, 0.0, 1.5) if t < 2 else (-4.0, 0.0, 1.5)),
    ]
    return run_many(specs, SimConfig(duration=duration))


def test_phri_drones_swap_places_without_colliding():
    pytest.importorskip("cvxpy")
    (_, _), pid_collisions = swap_flight(PID_PATH, PID_PATH)
    assert pid_collisions  # the baseline crashes head-on

    (a, b), collisions = swap_flight(PHRI_PATH, PHRI_PATH)
    assert collisions == []
    assert np.linalg.norm(a.pos - b.pos, axis=1).min() > 0.8
    assert a.metrics()["final_pos_error_m"] < 0.2 and b.metrics()["final_pos_error_m"] < 0.2


def test_phri_avoids_a_drone_that_does_not_cooperate():
    pytest.importorskip("cvxpy")
    (a, b), collisions = swap_flight(PHRI_PATH, PID_PATH)
    assert collisions == []
    assert a.metrics()["final_pos_error_m"] < 0.2


def test_phri_non_interactive_baseline_runs():
    pytest.importorskip("cvxpy")
    from drone_sim.phri import GameCache, PHRIParams, PHRIPlanner

    GameCache.clear()
    p = PHRIParams()
    planner = PHRIPlanner(p, ibr_iterations=1)
    game = phri_agents(p, [-4, 0.2, 1.5], [4, 0, 1.5], [4, 0, 1.5], [-4, 0, 1.5])
    planner.plan(game)
    assert planner.last_stats.failures == 0
    assert planner.last_stats.solves == 2 * (p.ideal_iterations + 5)


def test_neighbors_see_each_others_targets():
    a, b = Recorder(), Recorder()
    sim = Simulator(drones=[DroneSpec(a, initial_state=at(0), name="alpha", target_fn=lambda t: (1, 2, 3)),
                            DroneSpec(b, initial_state=at(3), name="bravo", target_fn=lambda t: (4, 5, 6, 0.5))])
    sim.step()
    assert b.others[0].target == (1.0, 2.0, 3.0, 0.0)
    assert a.others[0].target == (4.0, 5.0, 6.0, 0.5)  # set before alpha's controller ran


# --- Node framework -----------------------------------------------------------------

from dataclasses import dataclass as _dataclass  # noqa: E402

from drone_sim.nodes import (  # noqa: E402
    ControllerNode, Gps, GpsImuEstimator, Graph, Imu, Mocap, MocapImuEstimator, Node,
    PerfectState, StackController, to_dot, to_mermaid,
)


@_dataclass(frozen=True)
class Num:
    stamp: float
    value: float


class Counter(Node):
    def setup(self):
        self.pub = self.create_publisher("count", Num)
        self.n = 0
        self.create_timer(self.param("period", 0.1), self.tick)

    def tick(self):
        self.n += 1
        self.pub.publish(Num(self.now, self.n))


class Doubler(Node):
    def setup(self):
        self.seen = []
        self.out = self.create_publisher("doubled", Num)
        self.create_subscription("count", Num, self.on_count)

    def on_count(self, msg):
        self.seen.append((self.now, msg.value))
        self.out.publish(Num(msg.stamp, 2 * msg.value))


def test_pub_sub_timers_and_namespaces():
    g = Graph([Counter(), Doubler()], namespace="/robot")
    g.start()
    g.spin_until(0.35)
    assert g.node("doubler").seen == [(0.0, 1), (0.1, 2), pytest.approx((0.2, 3)), pytest.approx((0.3, 4))]
    assert set(g.topics) == {"/robot/count", "/robot/doubled"}
    assert g.topics["/robot/doubled"].last.value == 8
    assert g.topics["/robot/count"].rate() == pytest.approx(10.0)
    assert ("counter", "/robot/count", "pub") in g.edges()
    assert ("/robot/count", "doubler", "sub") in g.edges()
    g.start()  # restart resets everything
    assert g.node("doubler").seen == [] and g.topics["/robot/count"].count == 0


def test_remapping_and_global_topics():
    g = Graph([Counter(remap={"count": "/shared/ticks"}), Doubler(remap={"count": "/shared/ticks"})],
              namespace="/a")
    g.start()
    g.spin_until(0.0)
    assert "/shared/ticks" in g.topics and "/a/doubled" in g.topics
    assert g.node("doubler").seen == [(0.0, 1)]


def test_topic_types_are_enforced():
    class Wrong(Node):
        def setup(self):
            self.create_subscription("count", int, lambda m: None)

    g = Graph([Counter(), Wrong()])
    with pytest.raises(TypeError, match="carries Num"):
        g.start()
    with pytest.raises(ValueError, match="duplicate node name"):
        Graph([Counter(), Counter()])


def test_timers_at_the_same_time_fire_in_node_order():
    order = []

    class Tick(Node):
        def setup(self):
            self.create_timer(0.5, lambda: order.append(self.name))

    g = Graph([Tick("b"), Tick("a")])
    g.start()
    g.spin_until(1.0)
    assert order == ["b", "a", "b", "a", "b", "a"]


def test_perfect_state_stack_matches_the_plain_controller_exactly():
    cfg = SimConfig(scenario="square", duration=4.0)
    plain = run(load_controller(PID_PATH), cfg)
    stack = run(StackController(lambda: [PerfectState(), ControllerNode(PID_PATH)]), cfg)
    assert np.array_equal(plain.states, stack.states)


@pytest.mark.parametrize("sensors, settled", [
    (lambda: [Imu(), Gps(), GpsImuEstimator()], 0.2),
    (lambda: [Imu(), Mocap(), MocapImuEstimator()], 0.08),
])
def test_pid_flies_on_estimated_state(sensors, settled):
    c = StackController(lambda: [*sensors(), ControllerNode(PID_PATH)])
    r = run(c, SimConfig(scenario="step", duration=10.0))
    err = np.linalg.norm(r.pos - r.targets[:, :3], axis=1)
    assert err[r.t > 8].max() < settled
    est = c.graph.topics["/drone0/state_estimate"].last.state
    assert np.linalg.norm(est.pos - r.pos[-1]) < 0.3


def test_sensor_noise_is_reproducible_and_independent_per_drone():
    def flight():
        specs = [DroneSpec(load_controller(str(ROOT / "stacks" / "gps_imu_pid.py")), initial_state=at(0, y))
                 for y in (0.0, 2.0)]
        sim = Simulator(drones=specs, config=SimConfig(duration=1.0))
        for _ in range(250):
            sim.step()
        return [d.controller.graph.topics[f"/{d.name}/gps"].last.pos - d.state.pos for d in sim.drones]

    first, second = flight(), flight()
    assert np.array_equal(first[0], second[0])  # reproducible
    assert not np.allclose(first[0], first[1], atol=1e-6)  # drones' GPS noise differs


def test_stack_files_load_with_dash_c_and_fly_with_others():
    stack = load_controller(str(ROOT / "stacks" / "gps_imu_pid.py"))
    assert isinstance(stack, StackController) and stack.label == "gps_imu_pid"
    sim = Simulator(drones=[DroneSpec(stack, initial_state=at(0)),
                            DroneSpec(load_controller(PID_PATH), initial_state=at(2))])
    for _ in range(500):
        sim.step()
    swarm = stack.graph.topics["/swarm"].last
    assert [n.name for n in swarm.neighbors] == ["drone1"]
    assert sim.drones[0].state.z > 0.5


def test_live_sticks_reach_a_controller_inside_a_stack():
    from drone_sim.session import LiveSession

    sim = Simulator(load_controller(str(ROOT / "stacks" / "manual_gps.py")), SimConfig())
    session = LiveSession(sim)
    session.key_down("up")
    tick_for(session, 1.5)
    assert sim.state.z > 0.5


def test_graph_exports():
    c = StackController(lambda: [Imu(), Gps(), GpsImuEstimator(), ControllerNode(PID_PATH)])
    c.name = "drone0"
    c.reset(DroneParams())
    dot, mmd = to_dot([c.graph]), to_mermaid([c.graph])
    assert dot.startswith("digraph") and '"/drone0/gps"' in dot and '"/drone0::gps" -> "/drone0/gps"' in dot
    assert mmd.startswith("flowchart LR") and "GpsFix" in mmd
    assert "gps_imu_estimator [estimator]" in c.graph.describe()


def test_node_graph_drawing(agg, tmp_path):
    from drone_sim.graphics.node_graph import draw_graphs, save_graphs

    graphs = []
    for name in ("drone0", "drone1"):
        c = load_controller(str(ROOT / "stacks" / "gps_imu_pid.py"))
        c.name = name
        c.reset(DroneParams())
        graphs.append(c.graph)
    fig = draw_graphs(graphs)
    titles = [t.get_text() for t in fig.axes[0].texts if "same stack" in t.get_text()]
    assert titles == ["drone0, drone1  (same stack)"]
    save_graphs(graphs, str(tmp_path / "g.png"))
    assert (tmp_path / "g.png").stat().st_size > 1000


# --- PHRI in parallel -----------------------------------------------------------------

def circle_game(n, params):
    from drone_sim.phri import GameAgent

    agents = []
    for i in range(n):
        a = 2 * np.pi * i / n
        x0 = np.r_[4 * np.cos(a), 4 * np.sin(a), 1.5, 0, 0, 0]
        agents.append(GameAgent(f"d{i}", x0, np.r_[-x0[:3], 0, 0, 0], params))
    return agents


def test_phri_worker_pool_gives_identical_plans():
    pytest.importorskip("cvxpy")
    from drone_sim.phri import GameCache, PHRIParams, PHRIPlanner

    p = PHRIParams()
    plans = {}
    for workers in (1, 3):
        GameCache.clear()
        planner = PHRIPlanner(p, workers=workers)
        for _ in range(2):  # a replan warm-starts from the first plan
            game = circle_game(4, p)
            planner.plan(game)
        assert planner.last_stats.failures == 0
        plans[workers] = {a.name: a.xs for a in game}
    for name in plans[1]:
        assert np.array_equal(plans[1][name], plans[3][name])


def test_phri_worker_errors_surface_with_the_traceback():
    pytest.importorskip("cvxpy")
    from drone_sim.phri import PHRIParams, pool

    p = PHRIParams()
    bad = ("ideal", "x", p, np.zeros(6), np.zeros(6), np.zeros((3, 6)), np.zeros((3, 3)))  # wrong shapes
    with pytest.raises(RuntimeError, match="PHRI worker .* failed"):
        pool(2).run([bad, bad])


def test_phri_default_workers(monkeypatch):
    from drone_sim import phri

    monkeypatch.setenv("DRONE_SIM_PHRI_WORKERS", "3")
    assert phri.default_workers() == 3
    monkeypatch.delenv("DRONE_SIM_PHRI_WORKERS")
    monkeypatch.setattr(phri.sys, "platform", "emscripten")  # the browser: no processes
    assert phri.default_workers() == 1
