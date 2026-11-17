"""The local Python backend (web/src/python/server.py): the page's messages
over a WebSocket instead of to the in-browser worker.

Run with: python -m pytest tests/
"""

import asyncio
import importlib.util
import json
import threading
import time
from pathlib import Path

import pytest

websockets = pytest.importorskip("websockets")
from websockets.exceptions import InvalidStatus  # noqa: E402
from websockets.sync.client import connect  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
TOKEN = "test-token"
ORIGIN = "http://127.0.0.1:5173"


@pytest.fixture(scope="module")
def server():
    spec = importlib.util.spec_from_file_location("drone_sim_local_server", ROOT / "web" / "src" / "python" / "server.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    started = threading.Event()
    loop = asyncio.new_event_loop()

    def run():
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(module.main_async("127.0.0.1", 0, TOKEN, "http://127.0.0.1:5173", False, started))
        except asyncio.CancelledError:
            pass

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    assert started.wait(30), "server didn't start"
    yield f"ws://127.0.0.1:{started.port}"
    for task in asyncio.all_tasks(loop):
        loop.call_soon_threadsafe(task.cancel)
    thread.join(5)


class Client:
    def __init__(self, url):
        self.ws = connect(f"{url}/?token={TOKEN}", origin=ORIGIN, max_size=2**26)
        self.next_id = 1
        self.ready = self.recv()

    def recv(self, timeout=60):
        return json.loads(self.ws.recv(timeout=timeout))

    def send(self, msg):
        self.ws.send(json.dumps(msg))

    def request(self, msg, on_message=None):
        msg_id = self.next_id
        self.next_id += 1
        self.send({**msg, "id": msg_id})
        while True:
            m = self.recv()
            if m.get("type") == "reply" and m.get("id") == msg_id:
                return m
            if on_message:
                on_message(m)

    def close(self):
        self.ws.close()


@pytest.fixture
def client(server):
    c = Client(server)
    yield c
    c.close()


def ctrl(name):
    return {"name": name, "source": (ROOT / "controllers" / name).read_text()}


def test_rejects_other_sites_and_wrong_tokens(server):
    with pytest.raises(InvalidStatus) as e:
        connect(f"{server}/?token={TOKEN}", origin="https://evil.example")
    assert e.value.response.status_code == 403
    with pytest.raises(InvalidStatus) as e:
        connect(f"{server}/?token=nope", origin=ORIGIN)
    assert e.value.response.status_code == 403
    with pytest.raises(InvalidStatus):
        connect(f"{server}/", origin=ORIGIN)


def test_ready_reports_local_python(client):
    assert client.ready["type"] == "ready"
    info = client.ready["info"]
    assert info["backend"] == "local" and "hover" in info["scenarios"]


def test_check_and_playback_with_progress(client):
    check = client.request({"type": "check", "controller": ctrl("pid.py")})
    assert check["ok"] and check["value"]["class"] == "CascadedPID"

    progress = []
    reply = client.request(
        {"type": "playback", "opts": {"scenario": "step", "duration": 3, "controller": ctrl("pid.py")}},
        on_message=lambda m: m["type"] == "progress" and progress.append(m["fraction"]),
    )
    assert reply["ok"]
    result = reply["value"]
    assert len(result["t"]) == 750 and len(result["drones"][0]["pos"]) == 750 * 3
    assert progress and progress[-1] == 1.0


def test_errors_come_back_as_tracebacks(client):
    reply = client.request({"type": "check", "controller": {"name": "bad.py", "source": "raise ValueError('boom')"}})
    assert not reply["ok"] and "ValueError: boom" in reply["error"] and "Traceback" in reply["error"]


def test_controller_print_reaches_the_console(client):
    source = ctrl("pid.py")["source"].replace("    def reset(self, params):", "    def reset(self, params):\n        print('hello from reset')")
    client.request({"type": "check", "controller": {"name": "chatty.py", "source": source}})
    deadline = time.time() + 5
    while time.time() < deadline:
        m = client.recv(timeout=5)
        if m["type"] == "stdout" and "hello from reset" in m["text"]:
            return
    pytest.fail("print() output never arrived")


def test_live_frames_keys_and_stop(client):
    reply = client.request({"type": "live:start", "opts": {"scenario": "hover", "controller": ctrl("manual.py")}})
    assert reply["ok"]
    client.send({"type": "live:key", "key": "up", "down": True})
    frames = []
    while len(frames) < 40:
        m = client.recv()
        if m["type"] == "live:frame":
            frames.append(m["frame"])
    assert frames[-1]["sticks"]["throttle"] > 0.5
    assert frames[-1]["drones"][0]["pos"][2] > frames[0]["drones"][0]["pos"][2]
    client.send({"type": "live:stop"})


def test_stack_editing_runs_locally(client):
    reply = client.request({
        "type": "stack:open",
        "name": "gps_imu_pid.py",
        "source": (ROOT / "stacks" / "gps_imu_pid.py").read_text(),
        "controllers": [ctrl("pid.py")],
    })
    assert reply["ok"] and reply["value"]["ok"]
    edited = client.request({"type": "stack:edit", "edit": {"op": "update", "name": "gps", "params": {"rate": 5}}})
    assert "Gps(rate=5" in edited["value"]["source"]
