"""Run with: python -m pytest tests/"""

import math
from pathlib import Path

import pytest

from drone_sim import Controller, DroneParams, SimConfig, State, load_controller, run

ROOT = Path(__file__).resolve().parent.parent


class Constant(Controller):
    def __init__(self, left, right):
        self.cmd = (left, right)

    def update(self, t, state, target, dt):
        return self.cmd


def test_hover_thrust_holds_altitude():
    p = DroneParams()
    cfg = SimConfig(duration=2.0, initial_state=State(z=1.0), scenario="hover")
    # Spin motors up instantly so the drone doesn't sag while they spool.
    res = run(Constant(p.hover_thrust, p.hover_thrust), cfg, DroneParams(motor_tau=1e-6))
    assert res.states[-1, 1] == pytest.approx(1.0, abs=1e-3)


def test_zero_thrust_stays_on_ground():
    res = run(Constant(0.0, 0.0), SimConfig(duration=1.0))
    assert res.states[:, 1].max() == 0.0


def test_more_right_thrust_rotates_ccw_and_moves_left():
    p = DroneParams()
    cfg = SimConfig(duration=0.3, initial_state=State(z=5.0))
    res = run(Constant(p.hover_thrust - 0.1, p.hover_thrust + 0.1), cfg)
    assert res.states[-1, 2] > 0  # theta
    assert res.states[-1, 0] < 0  # x


def test_thrust_saturates():
    res = run(Constant(100.0, 100.0), SimConfig(duration=0.5))
    assert res.thrusts.max() <= DroneParams().max_thrust + 1e-9


# Durations end each run ~3 s after the last target change, so the drone has settled.
@pytest.mark.parametrize("scenario, duration", [("hover", 5.0), ("step", 6.0), ("square", 8.9)])
def test_example_pid_tracks(scenario, duration):
    cfg = SimConfig(scenario=scenario, duration=duration)
    res = run(load_controller(str(ROOT / "controllers" / "pid.py")), cfg)
    m = res.metrics()
    assert not m["flipped"]
    assert m["final_pos_error_m"] < 0.2


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


def test_bad_controller_output_is_reported():
    with pytest.raises(ValueError, match="expected two finite thrusts"):
        run(Constant(math.nan, 1.0), SimConfig(duration=0.1))
