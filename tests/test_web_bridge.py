"""The web app (web/) drives drone_sim only through web/src/python/bridge.py,
which runs in the browser under Pyodide. These tests run that same file under
CPython, so a core change that would break the web app fails here.

Run with: python -m pytest tests/
"""

import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
CONTROLLERS = ROOT / "controllers"


@pytest.fixture(scope="module")
def bridge():
    spec = importlib.util.spec_from_file_location("webbridge", ROOT / "web" / "src" / "python" / "bridge.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def ctrl(name):
    return {"name": name, "source": (CONTROLLERS / name).read_text()}


def run_playback(bridge, **opts):
    pb = bridge.Playback(json.dumps(opts))
    while pb.advance(200) < 1.0:
        pass
    return json.loads(pb.result())


def test_info_lists_scenarios_and_geometry(bridge):
    info = json.loads(bridge.info())
    assert {"hover", "step", "helix"} <= set(info["scenarios"])
    assert len(info["params"]["rotor_positions"]) == 4


def test_playback_tracks_like_the_cli(bridge):
    r = run_playback(bridge, scenario="step", duration=6.0, controller=ctrl("pid.py"))
    n = len(r["t"])
    (d,) = r["drones"]
    assert len(d["pos"]) == 3 * n and len(d["quat"]) == 4 * n and len(d["thrust"]) == 4 * n
    assert d["metrics"]["final_pos_error_m"] < 0.2
    assert not d["metrics"]["flipped"]


def test_playback_swarm_flies_in_formation(bridge):
    r = run_playback(bridge, scenario="hover", duration=4.0, drones=3, spacing=1.0, controller=ctrl("pid.py"))
    assert [d["name"] for d in r["drones"]] == ["drone0", "drone1", "drone2"]
    ys = [d["pos"][-2] for d in r["drones"]]  # final y of each drone
    assert ys == pytest.approx([-1.0, 0.0, 1.0], abs=0.1)
    assert r["collisions"] == []


def test_live_throttle_climbs_and_r_resets(bridge):
    live = bridge.Live(json.dumps({"scenario": "hover", "controller": ctrl("manual.py")}))
    live.key("up", True)
    for _ in range(30):
        snap = json.loads(live.tick(1 / 30))
    assert snap["t"] == pytest.approx(1.0, abs=0.01)
    assert snap["drones"][0]["pos"][2] > 0.5
    assert snap["sticks"]["throttle"] > 0.9

    live.key("up", False)
    live.key("r", True)
    snap = json.loads(live.tick(0.0))
    assert snap["t"] == 0.0
    assert snap["drones"][0]["pos"] == [0.0, 0.0, 0.0]


def test_live_wingmen_fly_their_own_controller(bridge):
    live = bridge.Live(json.dumps({
        "scenario": "hover", "drones": 2, "controller": ctrl("manual.py"), "others": ctrl("pid.py"),
    }))
    for _ in range(60):
        snap = json.loads(live.tick(1 / 30))
    piloted, wingman = snap["drones"]
    assert piloted["pos"][2] == 0.0  # nobody touched the keys
    assert wingman["pos"][2] > 0.8  # pid climbs to the hover target


def test_check_reports_class_and_metrics(bridge):
    out = json.loads(bridge.check(json.dumps(ctrl("hover_only.py"))))
    assert out["class"] == "HoverOnly"
    assert not out["metrics"]["flipped"]


def test_check_rejects_files_without_a_controller(bridge):
    with pytest.raises(ValueError, match="exactly one Controller"):
        bridge.check(json.dumps({"name": "empty.py", "source": "x = 1\n"}))


def test_check_rejects_bad_file_names(bridge):
    with pytest.raises(ValueError, match="file names"):
        bridge.check(json.dumps({"name": "my-controller.py", "source": ""}))


def test_edited_source_is_reloaded(bridge):
    src = (CONTROLLERS / "hover_only.py").read_text()
    first = json.loads(bridge.check(json.dumps({"name": "mine.py", "source": src})))
    renamed = src.replace("class HoverOnly", "class Renamed")
    second = json.loads(bridge.check(json.dumps({"name": "mine.py", "source": renamed})))
    assert (first["class"], second["class"]) == ("HoverOnly", "Renamed")


# --- Stacks tab ---------------------------------------------------------------------

STACKS = ROOT / "stacks"
ALL_CONTROLLERS = [ctrl(p.name) for p in sorted(CONTROLLERS.glob("*.py"))]


def open_stack(bridge, name="gps_imu_pid.py", source=None):
    reply = json.loads(bridge.stack_open(json.dumps({
        "name": name,
        "source": source if source is not None else (STACKS / name).read_text(),
        "controllers": ALL_CONTROLLERS,
    })))
    assert reply["ok"], reply
    return reply


def edit(bridge, **req):
    return json.loads(bridge.stack_edit(json.dumps(req)))


def test_stack_open_lists_nodes_graph_and_catalog(bridge):
    s = open_stack(bridge)
    assert [n["name"] for n in s["nodes"]] == ["imu", "gps", "gps_imu_estimator", "controller"]
    labels = {v["label"] for v in s["graph"]["vertices"]}
    assert {"sim", "imu", "gps", "state_estimate", "motor_command"} <= labels
    assert any(e["feedback"] for e in s["graph"]["edges"])  # motor_command back into sim
    sensors = [c["class"] for c in s["catalog"] if c["kind"] == "sensor"]
    assert sensors == ["Gps", "Imu", "Mocap", "PerfectState"]
    gps = next(n for n in s["nodes"] if n["name"] == "gps")
    rate = next(p for p in gps["params"] if p["name"] == "rate")
    assert rate == {**rate, "type": "number", "value": 10, "default": 10.0}
    controller = next(p for p in s["nodes"][-1]["params"] if p["name"] == "controller")
    assert controller["value"] == "controllers/pid.py" and "controllers/phri.py" in controller["choices"]
    assert s["warnings"] == []


def test_stack_edits_rewrite_only_the_stack_function(bridge):
    original = (STACKS / "gps_imu_pid.py").read_text()
    open_stack(bridge)
    s = edit(bridge, op="update", name="gps", params={"rate": 5, "pos_noise": 0.5})
    assert s["ok"] and s["selected"] == "gps"
    assert "Gps(rate=5, pos_noise=0.5, vel_noise=0.05)" in s["source"]
    s = edit(bridge, op="add", key="drone_sim.nodes.Mocap", params={"rate": 50})
    assert s["selected"] == "mocap"
    assert "from drone_sim.nodes import ControllerNode, Gps, GpsImuEstimator, Imu, Mocap" in s["source"]
    assert "Mocap(rate=50)," in s["source"]
    assert [n["name"] for n in s["nodes"]][:4] == ["imu", "gps", "mocap", "gps_imu_estimator"]  # after the sensors
    assert s["warnings"] == ["Nothing reads mocap (published by mocap)."]
    # Everything outside stack() is untouched.
    head = original.split("def stack")[0].replace(
        "import ControllerNode, Gps, GpsImuEstimator, Imu", "import ControllerNode, Gps, GpsImuEstimator, Imu, Mocap")
    assert s["source"].startswith(head)
    s = edit(bridge, op="remove", name="mocap")
    assert "Mocap(" not in s["source"].split("def stack")[1]


def test_stack_rejects_bad_edits_without_changing_anything(bridge):
    before = open_stack(bridge)["source"]
    for req in [
        {"op": "update", "name": "gps", "params": {"name": "imu"}},
        {"op": "update", "name": "gps", "params": {"rate": "fast"}},
        {"op": "update", "name": "controller", "params": {"controller": "controllers/nope.py"}},
        {"op": "add", "key": "drone_sim.nodes.NoSuchSensor", "params": {}},
        {"op": "remove", "name": "ghost"},
    ]:
        reply = edit(bridge, **req)
        assert reply == {"ok": False, "error": reply["error"]} and reply["error"]
    assert edit(bridge, op="test", scenario="hover", duration=0.2)["source"] == before


def test_stack_remap_warns_about_loose_ends(bridge):
    open_stack(bridge)
    s = edit(bridge, op="update", name="gps", params={"remap": {"gps": "gps_raw"}})
    assert "Nothing reads gps_raw (published by gps)." in s["warnings"]
    assert "Nothing publishes gps, which gps_imu_estimator needs." in s["warnings"]
    assert 'remap={"gps": "gps_raw"}' in s["source"]
    s = edit(bridge, op="update", name="gps", params={"remap": {}})
    assert s["warnings"] == [] and "remap" not in s["source"].split("def stack")[1]


def test_stack_test_flight_measures_rates(bridge):
    open_stack(bridge)
    edit(bridge, op="update", name="gps", params={"rate": 5})
    s = edit(bridge, op="test", scenario="hover", duration=3)
    assert s["flight"]["metrics"]["flipped"] is False
    gps = next(v for v in s["graph"]["vertices"] if v["type"] == "topic" and v["label"] == "gps")
    assert gps["rate"] == pytest.approx(5.0)


def test_stack_files_keep_custom_nodes(bridge):
    source = '''"""A stack with its own node."""

from pathlib import Path

from drone_sim.nodes import ControllerNode, Node, PerfectState
from drone_sim.nodes.msgs import StateEstimate

CONTROLLERS = Path(__file__).resolve().parent.parent / "controllers"


class Watcher(Node):
    """Counts estimates."""

    def __init__(self, name="watcher", every: float = 1.0, **kw):
        super().__init__(name, **kw)
        self.every = every

    def setup(self):
        self.create_subscription("state_estimate", StateEstimate, lambda m: None)


def stack():
    return [PerfectState(), ControllerNode(CONTROLLERS / "pid.py")]
'''
    s = open_stack(bridge, "custom_stack.py", source)
    watcher = next(c for c in s["catalog"] if c["class"] == "Watcher")
    assert watcher["local"] and [p["name"] for p in watcher["params"]] == ["name", "every", "remap"]
    s = edit(bridge, op="add", key=watcher["key"], params={"every": 2.5})
    assert "class Watcher(Node):" in s["source"] and "Watcher(every=2.5)," in s["source"]
    assert "watcher" in [n["name"] for n in s["nodes"]]
