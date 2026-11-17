"""Run the web app's Python on this computer instead of in the browser.

    pip install websockets
    python web/src/python/server.py

The page normally runs drone_sim in Pyodide (WebAssembly) inside a Web
Worker. This server runs the very same bridge.py in your local Python and
speaks the same messages (src/sim/protocol.ts) over a WebSocket, so the
page works unchanged but simulations run at native speed with your
installed packages (e.g. cvxpy for the PHRI planner).

It prints two links:
- for `npm run dev`: the dev server URL with ?local=...&token=... added
- if web/dist exists (`npm run build`): the built app, served right here

Security: the page sends controller and stack code that this server runs,
so it only listens on 127.0.0.1, only accepts pages from localhost, and
requires a random token that is only printed here (and embedded in the
app it serves itself). Stop it with Ctrl+C.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import importlib.util
import io
import json
import mimetypes
import secrets
import sys
import threading
import time
import traceback
import urllib.parse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]  # the repo: web/src/python -> repo
DIST = ROOT / "web" / "dist"
LOCAL_HOSTS = {"localhost", "127.0.0.1", "[::1]", "::1"}
PLAYBACK_CHUNK = 250  # control steps between progress reports (1 s of sim time)
FRAME_S = 1 / 60

try:
    from websockets.asyncio.server import ServerConnection, serve
    from websockets.datastructures import Headers
    from websockets.exceptions import ConnectionClosed
    from websockets.http11 import Request, Response
except ImportError:  # pragma: no cover - explained at startup
    serve = None


def load_bridge():
    """Import bridge.py exactly as the browser does (as `webbridge`)."""
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    spec = importlib.util.spec_from_file_location("webbridge", HERE / "bridge.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules["webbridge"] = module
    spec.loader.exec_module(module)
    return module


def origin_allowed(origin: str | None) -> bool:
    """Only pages served from this computer may connect."""
    if not origin:
        return False
    parsed = urllib.parse.urlparse(origin)
    return parsed.scheme in ("http", "https") and (parsed.hostname or "") in LOCAL_HOSTS


class Output(io.TextIOBase):
    """Collects print() output from controllers for the page's console."""

    def __init__(self):
        self.lock = threading.Lock()
        self.parts: list[str] = []

    def write(self, s: str) -> int:
        with self.lock:
            self.parts.append(s)
        return len(s)

    def take(self) -> str:
        with self.lock:
            text, self.parts = "".join(self.parts), []
        return text


class Backend:
    """One Python, shared by connections one at a time (like the worker).

    Every bridge call runs on a single thread, in order, so bridge.py's
    module-level state (the open stack, the live session) behaves exactly
    as it does in the browser.
    """

    def __init__(self, token: str):
        self.token = token
        self.bridge = load_bridge()
        self.pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="bridge")
        self.output = Output()
        self.info = json.loads(self.bridge.info())
        self.info["backend"] = "local"
        self.active: Session | None = None

    async def call(self, fn, *args):
        def run():
            # Anything controllers print goes to the page's console.
            with contextlib.redirect_stdout(self.output), contextlib.redirect_stderr(self.output):
                return fn(*args)

        return await asyncio.get_running_loop().run_in_executor(self.pool, run)


class Session:
    def __init__(self, backend: Backend, ws: "ServerConnection"):
        self.backend = backend
        self.ws = ws
        self.cancelled: set[int] = set()
        self.live = None
        self.live_paused = False
        self.live_task: asyncio.Task | None = None
        self.live_generation = 0
        self.closed = False

    async def send(self, text: str) -> None:
        if not self.closed:
            try:
                await self.ws.send(text)
            except ConnectionClosed:
                self.closed = True

    async def send_json(self, obj) -> None:
        await self.send(json.dumps(obj))

    async def reply(self, msg_id: int, value_json: str) -> None:
        await self.send(f'{{"type":"reply","id":{msg_id},"ok":true,"value":{value_json}}}')

    async def fail(self, msg_id: int, error: str) -> None:
        await self.send_json({"type": "reply", "id": msg_id, "ok": False, "error": error})

    # --- Live flight ---

    async def stop_live(self) -> None:
        self.live_generation += 1
        if self.live_task:
            self.live_task.cancel()
            self.live_task = None
        if self.live is not None:
            self.live = None

    async def live_loop(self, generation: int) -> None:
        last = time.perf_counter()
        while self.live is not None and generation == self.live_generation and not self.closed:
            if self.live_paused:
                await asyncio.sleep(FRAME_S)
                last = time.perf_counter()
                continue
            now = time.perf_counter()
            elapsed, last = now - last, now
            try:
                frame = await self.backend.call(self.live.tick, elapsed)
            except Exception:
                await self.stop_live()
                await self.send_json({"type": "live:error", "error": traceback.format_exc()})
                return
            await self.send(f'{{"type":"live:frame","frame":{frame}}}')
            await asyncio.sleep(max(0.0, FRAME_S - (time.perf_counter() - now)))

    # --- Requests ---

    async def handle(self, msg: dict) -> None:
        kind = msg.get("type")
        b = self.backend
        bridge = b.bridge
        if kind == "cancel":
            self.cancelled.add(msg["id"])
            return
        if kind in ("live:key", "live:release", "live:reset"):
            if self.live is not None:
                live = self.live
                if kind == "live:key":
                    await b.call(live.key, msg["key"], msg["down"])
                elif kind == "live:release":
                    await b.call(live.release_all)
                else:
                    await b.call(live.reset)
            return
        if kind == "live:pause":
            self.live_paused = bool(msg["paused"])
            return
        if kind == "live:stop":
            await self.stop_live()
            return

        msg_id = msg.get("id")
        try:
            if kind == "playback":
                await self.playback(msg_id, msg["opts"])
            elif kind == "check":
                out = await b.call(bridge.check, json.dumps(msg["controller"]))
                await self.reply(msg_id, out)
            elif kind == "live:start":
                await self.stop_live()
                generation = self.live_generation
                session = await b.call(bridge.Live, json.dumps(msg["opts"]))
                if generation != self.live_generation:
                    return  # superseded by a newer start or a stop
                self.live, self.live_paused = session, False
                frame = await b.call(session.tick, 0.0)
                await self.send(f'{{"type":"live:frame","frame":{frame}}}')
                self.live_task = asyncio.create_task(self.live_loop(generation))
                await self.reply(msg_id, "null")
            elif kind == "stack:open":
                request = {"name": msg["name"], "source": msg["source"], "controllers": msg["controllers"]}
                await self.reply(msg_id, await b.call(bridge.stack_open, json.dumps(request)))
            elif kind == "stack:edit":
                await self.reply(msg_id, await b.call(bridge.stack_edit, json.dumps(msg["edit"])))
            else:
                await self.fail(msg_id, f"unknown request {kind!r}")
        except Exception as e:
            text = traceback.format_exc()
            if isinstance(e, ModuleNotFoundError):
                text += f"\nInstall it for the local Python with: pip install {e.name}"
            await self.fail(msg_id, text)

    async def playback(self, msg_id: int, opts: dict) -> None:
        b = self.backend
        pb = await b.call(b.bridge.Playback, json.dumps(opts))
        done = 0.0
        while done < 1.0:
            done = await b.call(pb.advance, PLAYBACK_CHUNK)
            await self.send_json({"type": "progress", "id": msg_id, "fraction": done})
            if msg_id in self.cancelled:
                self.cancelled.discard(msg_id)
                raise RuntimeError("Cancelled")
        await self.reply(msg_id, await b.call(pb.result))


async def flush_output(backend: Backend, session: Session) -> None:
    while not session.closed:
        await asyncio.sleep(0.15)
        text = backend.output.take()
        if text:
            lines = text.rstrip("\n").split("\n")[-400:]
            await session.send_json({"type": "stdout", "text": "\n".join(lines)})


def make_handler(backend: Backend):
    async def handler(ws: "ServerConnection") -> None:
        if backend.active is not None:  # a newer tab takes over
            old = backend.active
            await old.stop_live()
            await old.ws.close(4000, "another tab connected")
        session = Session(backend, ws)
        backend.active = session
        flusher = asyncio.create_task(flush_output(backend, session))
        await session.send_json({"type": "ready", "info": backend.info})
        tasks: set[asyncio.Task] = set()
        try:
            async for raw in ws:
                msg = json.loads(raw)
                # Live controls and cancels must not wait behind a long playback.
                task = asyncio.create_task(session.handle(msg))
                tasks.add(task)
                task.add_done_callback(tasks.discard)
        except ConnectionClosed:
            pass
        finally:
            session.closed = True
            await session.stop_live()
            flusher.cancel()
            if backend.active is session:
                backend.active = None
    return handler


# --- HTTP: serve the built app with the connection details filled in --------------

def static_response(backend: Backend, path: str, ws_url: str) -> "Response":
    rel = urllib.parse.unquote(urllib.parse.urlparse(path).path).lstrip("/") or "index.html"
    if not DIST.is_dir():
        return _text(404, "No built app here. Run `npm run build` in web/, or use the dev-server link "
                          "printed in the terminal.\n")
    file = (DIST / rel).resolve()
    if DIST.resolve() not in file.parents or not file.is_file():
        file = DIST / "index.html"  # single-page app
    body = file.read_bytes()
    if file.name == "index.html":
        config = json.dumps({"url": ws_url, "token": backend.token})
        body = body.replace(b"</head>", f"<script>window.__DRONE_SIM_LOCAL__={config}</script></head>".encode(), 1)
    ctype = mimetypes.guess_type(file.name)[0] or "application/octet-stream"
    if ctype.startswith("text/") or ctype in ("application/javascript", "application/json"):
        ctype += "; charset=utf-8"
    headers = Headers([("Content-Type", ctype), ("Content-Length", str(len(body))), ("Cache-Control", "no-store")])
    return Response(200, "OK", headers, body)


def _text(status: int, text: str) -> "Response":
    body = text.encode()
    headers = Headers([("Content-Type", "text/plain; charset=utf-8"), ("Content-Length", str(len(body)))])
    return Response(status, "Forbidden" if status == 403 else "Not Found", headers, body)


def make_process_request(backend: Backend, where: dict):
    """`where["url"]` is the WebSocket address, filled in once the port is known."""
    def process_request(connection, request: "Request"):
        if request.headers.get("Upgrade", "").lower() != "websocket":
            return static_response(backend, request.path, where["url"])
        if not origin_allowed(request.headers.get("Origin")):
            return _text(403, "Only pages served from this computer may connect.\n")
        query = urllib.parse.parse_qs(urllib.parse.urlparse(request.path).query)
        if not secrets.compare_digest(query.get("token", [""])[0], backend.token):
            return _text(403, "Wrong or missing token. Use the link printed by the server.\n")
        return None  # accept the WebSocket
    return process_request


async def main_async(host: str, port: int, token: str, dev_url: str, open_browser: bool,
                     started: threading.Event | None = None) -> None:
    backend = Backend(token)
    where = {"url": f"ws://{host}:{port}"}
    async with serve(make_handler(backend), host, port, process_request=make_process_request(backend, where),
                     max_size=2**26, compression=None) as server:
        actual = server.sockets[0].getsockname()[1]
        ws_url = where["url"] = f"ws://{host}:{actual}"
        log = sys.__stdout__
        query = urllib.parse.urlencode({"local": ws_url, "token": token})
        print(f"drone_sim local Python ({backend.info['python']}, numpy {backend.info['numpy']}) on {ws_url}", file=log)
        print(f"  with `npm run dev`:   {dev_url}/?{query}", file=log)
        if DIST.is_dir():
            print(f"  built app (web/dist): http://{host}:{actual}/", file=log)
        print("Ctrl+C to stop.", file=log, flush=True)
        if open_browser:
            import webbrowser

            webbrowser.open(f"http://{host}:{actual}/" if DIST.is_dir() else f"{dev_url}/?{query}")
        if started is not None:
            started.port = actual  # type: ignore[attr-defined]
            started.set()
        await asyncio.get_running_loop().create_future()  # run until cancelled


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="Run the web app's Python on this computer.")
    ap.add_argument("--port", type=int, default=8765, help="0 picks a free port")
    ap.add_argument("--dev-url", default="http://127.0.0.1:5173", help="where `npm run dev` serves the app")
    ap.add_argument("--token", help="use this token instead of a random one")
    ap.add_argument("--open", action="store_true", help="open the app in a browser")
    args = ap.parse_args(argv)
    if serve is None:
        sys.exit("The local backend needs the websockets package: pip install websockets")
    token = args.token or secrets.token_urlsafe(18)
    try:
        asyncio.run(main_async("127.0.0.1", args.port, token, args.dev_url.rstrip("/"), args.open))
    except KeyboardInterrupt:
        print(file=sys.__stdout__)


if __name__ == "__main__":
    main()
