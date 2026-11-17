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
