"""Run with: python -m pytest tests/"""

import math
from pathlib import Path

import numpy as np
import pytest

from drone_sim import Controller, DroneParams, SimConfig, Simulator, State, Sticks, load_controller, run
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
