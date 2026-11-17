// Runs drone_sim in Pyodide, off the main thread so the 3D view never stalls.
//
// The Python sources are bundled straight from the repo (../drone_sim and
// ../controllers), so the web app always runs the same code as
// `python -m drone_sim`. Only src/python/bridge.py is web-specific.

import bridgeSource from "../python/bridge.py?raw";
import type {
  ControllerFile,
  FromWorker,
  LiveFrame,
  PlaybackResult,
  SimOptions,
  ToWorker,
} from "./protocol";

const PYODIDE_URL = "https://cdn.jsdelivr.net/pyodide/v0.29.4/full/";
const HOME = "/home/pyodide";

const packageFiles = import.meta.glob(
  ["../../../drone_sim/**/*.py", "!../../../drone_sim/graphics/**"],
  { query: "?raw", import: "default", eager: true },
) as Record<string, string>;

const controllerFiles = import.meta.glob("../../../controllers/*.py", {
  query: "?raw",
  import: "default",
  eager: true,
}) as Record<string, string>;

// --- Minimal typings for the parts of Pyodide used here ---------------------------

interface PyProxy {
  destroy(): void;
}
interface Bridge extends PyProxy {
  info(): string;
  check(controllerJson: string): string;
  Playback(optsJson: string): PlaybackProxy;
  Live(optsJson: string): LiveProxy;
  stack_open(requestJson: string): string;
  stack_edit(requestJson: string): string;
}
interface PlaybackProxy extends PyProxy {
  advance(n: number): number;
  result(): string;
}
interface LiveProxy extends PyProxy {
  key(key: string, down: boolean): void;
  release_all(): void;
  reset(): void;
  tick(elapsed: number): string;
}
interface Pyodide {
  FS: { mkdirTree(path: string): void; writeFile(path: string, data: string): void };
  loadPackage(name: string, options?: { messageCallback?: (msg: string) => void }): Promise<void>;
  loadPackagesFromImports(code: string, options?: { messageCallback?: (msg: string) => void }): Promise<void>;
  loadedPackages: Record<string, string>;
  runPython(code: string): unknown;
  pyimport(name: string): Bridge;
  setStdout(opts: { batched: (text: string) => void }): void;
  setStderr(opts: { batched: (text: string) => void }): void;
}

const ctx = self as unknown as {
  postMessage(message: FromWorker, transfer?: Transferable[]): void;
  onmessage: ((e: MessageEvent<ToWorker>) => void) | null;
};

function post(message: FromWorker, transfer?: Transferable[]) {
  ctx.postMessage(message, transfer);
}

function errorText(e: unknown): string {
  if (e instanceof Error) return e.message;
  return String(e);
}

// --- Startup ------------------------------------------------------------------------

let bridge: Bridge | null = null;

function writeTree(files: Record<string, string>) {
  for (const [path, source] of Object.entries(files)) {
    const rel = path.replace(/^(\.\.\/)+/, ""); // "drone_sim/sim.py"
    const full = `${HOME}/${rel}`;
    py!.FS.mkdirTree(full.slice(0, full.lastIndexOf("/")));
    py!.FS.writeFile(full, source);
  }
}

// Controllers may print every control step; batch output so it can't flood the page.
let stdoutBuffer: string[] = [];
let stdoutTimer: ReturnType<typeof setTimeout> | null = null;
function queueStdout(text: string) {
  stdoutBuffer.push(text);
  if (stdoutBuffer.length > 400) stdoutBuffer = stdoutBuffer.slice(-200);
  stdoutTimer ??= setTimeout(() => {
    post({ type: "stdout", text: stdoutBuffer.join("\n") });
    stdoutBuffer = [];
    stdoutTimer = null;
  }, 150);
}

let py: Pyodide | null = null;

const ready = (async () => {
  try {
    post({ type: "status", text: "Downloading Python" });
    const { loadPyodide } = await import(/* @vite-ignore */ `${PYODIDE_URL}pyodide.mjs`);
    py = (await loadPyodide({ indexURL: PYODIDE_URL })) as Pyodide;
    post({ type: "status", text: "Loading numpy" });
    await py.loadPackage("numpy", { messageCallback: () => {} });
    // Only controller output from here on goes to the page's Python console.
    py.setStdout({ batched: queueStdout });
    py.setStderr({ batched: queueStdout });
    post({ type: "status", text: "Loading drone_sim" });
    writeTree(packageFiles);
    writeTree(controllerFiles);
    py.FS.writeFile(`${HOME}/webbridge.py`, bridgeSource);
    py.runPython(`import sys\nif "${HOME}" not in sys.path: sys.path.insert(0, "${HOME}")`);
    bridge = py.pyimport("webbridge");
    post({ type: "ready", info: JSON.parse(bridge.info()) });
  } catch (e) {
    post({ type: "fatal", error: errorText(e) });
    throw e;
  }
})();

// --- Optional packages ---------------------------------------------------------------

/** Run `fn`; if Python can't find a module that Pyodide ships (e.g. cvxpy for
 * controllers/phri.py), download it and try again. Only numpy loads up front. */
async function withPackages<T>(fn: () => T): Promise<T> {
  for (let attempt = 0; ; attempt++) {
    try {
      return fn();
    } catch (e) {
      const missing = /No module named '([\w.]+)'/.exec(errorText(e))?.[1].split(".")[0];
      if (!missing || attempt >= 5) throw e;
      const before = Object.keys(py!.loadedPackages).length;
      post({ type: "notice", text: `Loading ${missing}` });
      try {
        await py!.loadPackagesFromImports(`import ${missing}`, { messageCallback: () => {} });
      } finally {
        post({ type: "notice", text: null });
      }
      if (Object.keys(py!.loadedPackages).length === before) throw e; // not something Pyodide has
    }
  }
}

// --- Playback ----------------------------------------------------------------------

const cancelled = new Set<number>();
const CHUNK = 125; // control steps (0.5 s of sim time) between progress reports

const yieldToMessages = () => new Promise((resolve) => setTimeout(resolve, 0));

function toFloat64(values: number[]): Float64Array {
  return Float64Array.from(values);
}

async function runPlayback(id: number, opts: SimOptions): Promise<[PlaybackResult, Transferable[]]> {
  const pb = await withPackages(() => bridge!.Playback(JSON.stringify(opts)));
  try {
    let done = 0;
    while (done < 1) {
      done = pb.advance(CHUNK);
      post({ type: "progress", id, fraction: done });
      await yieldToMessages();
      if (cancelled.delete(id)) throw new Error("Cancelled");
    }
    const raw = JSON.parse(pb.result());
    const transfer: Transferable[] = [];
    const arr = (values: number[]) => {
      const a = toFloat64(values);
      transfer.push(a.buffer);
      return a;
    };
    const result: PlaybackResult = {
      ...raw,
      options: opts,
      t: arr(raw.t),
      wind: arr(raw.wind),
      drones: raw.drones.map((d: Record<string, unknown>) => ({
        ...d,
        pos: arr(d.pos as number[]),
        quat: arr(d.quat as number[]),
        euler: arr(d.euler as number[]),
        target: arr(d.target as number[]),
        thrust: arr(d.thrust as number[]),
      })),
    };
    return [result, transfer];
  } finally {
    pb.destroy();
  }
}

// --- Live flight -------------------------------------------------------------------

let live: LiveProxy | null = null;
let livePaused = false;
let liveTimer: ReturnType<typeof setTimeout> | null = null;
let lastTick = 0;
const FRAME_MS = 1000 / 60;

let liveGeneration = 0; // bumped by every start/stop, so a slow start can tell it was superseded

function stopLive() {
  liveGeneration++;
  if (liveTimer) clearTimeout(liveTimer);
  liveTimer = null;
  live?.destroy();
  live = null;
}

function liveLoop() {
  liveTimer = null;
  if (!live || livePaused) return;
  const now = performance.now();
  const elapsed = (now - lastTick) / 1000;
  lastTick = now;
  try {
    const frame = JSON.parse(live.tick(elapsed)) as LiveFrame;
    post({ type: "live:frame", frame });
  } catch (e) {
    stopLive();
    post({ type: "live:error", error: errorText(e) });
    return;
  }
  liveTimer = setTimeout(liveLoop, FRAME_MS);
}

async function startLive(opts: SimOptions) {
  stopLive();
  const generation = liveGeneration;
  const session = await withPackages(() => bridge!.Live(JSON.stringify(opts)));
  if (generation !== liveGeneration) {
    session.destroy(); // a newer start or a stop arrived while packages loaded
    return;
  }
  live = session;
  livePaused = false;
  lastTick = performance.now();
  post({ type: "live:frame", frame: JSON.parse(live.tick(0)) });
  liveTimer = setTimeout(liveLoop, FRAME_MS);
}

// --- Messages ----------------------------------------------------------------------

async function handle(msg: ToWorker) {
  switch (msg.type) {
    case "cancel":
      cancelled.add(msg.id);
      return;
    case "live:key":
      live?.key(msg.key, msg.down);
      return;
    case "live:release":
      live?.release_all();
      return;
    case "live:reset":
      live?.reset();
      return;
    case "live:pause":
      livePaused = msg.paused;
      if (live && !livePaused && !liveTimer) {
        lastTick = performance.now();
        liveLoop();
      }
      return;
    case "live:stop":
      stopLive();
      return;
  }

  // Everything below needs Python and sends a reply.
  try {
    await ready;
    switch (msg.type) {
      case "playback": {
        const [result, transfer] = await runPlayback(msg.id, msg.opts);
        post({ type: "reply", id: msg.id, ok: true, value: result }, transfer);
        return;
      }
      case "check": {
        const controller: ControllerFile = msg.controller;
        const out = await withPackages(() => bridge!.check(JSON.stringify(controller)));
        post({ type: "reply", id: msg.id, ok: true, value: JSON.parse(out) });
        return;
      }
      case "live:start":
        await startLive(msg.opts);
        post({ type: "reply", id: msg.id, ok: true, value: null });
        return;
      case "stack:open": {
        const request = JSON.stringify({ name: msg.name, source: msg.source, controllers: msg.controllers });
        const out = await withPackages(() => bridge!.stack_open(request));
        post({ type: "reply", id: msg.id, ok: true, value: JSON.parse(out) });
        return;
      }
      case "stack:edit": {
        const out = await withPackages(() => bridge!.stack_edit(JSON.stringify(msg.edit)));
        post({ type: "reply", id: msg.id, ok: true, value: JSON.parse(out) });
        return;
      }
    }
  } catch (e) {
    post({ type: "reply", id: msg.id, ok: false, error: errorText(e) });
  }
}

ctx.onmessage = (e) => {
  void handle(e.data);
};
