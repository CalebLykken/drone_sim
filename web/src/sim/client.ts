// Main-thread handle on the simulator: promise-based requests plus
// subscriptions for live frames, progress, and Python output.
//
// Python runs in one of two places, with the same messages (protocol.ts):
// - in this browser, in a Web Worker with Pyodide (worker.ts), or
// - on this computer, in `python web/src/python/server.py`, over a WebSocket.
//   The page uses the local server when it was opened through the link the
//   server prints (?local=...&token=...) or is served by the server itself,
//   and falls back to the browser if the server can't be reached.

import { useSyncExternalStore } from "react";
import { toPlaybackResult } from "./convert";
import type {
  CheckResult,
  ControllerFile,
  FromWorker,
  LiveFrame,
  PlaybackResult,
  SimInfo,
  SimKey,
  SimOptions,
  StackEdit,
  StackReply,
  ToWorker,
} from "./protocol";

export type SimStatus =
  | { state: "loading"; text: string }
  | { state: "ready"; info: SimInfo }
  | { state: "error"; error: string; local?: boolean };

type Pending = {
  resolve: (value: unknown) => void;
  reject: (error: Error) => void;
  onProgress?: (fraction: number) => void;
  /** Turn the JSON reply into its final shape (only needed off the worker). */
  convert?: (value: unknown) => unknown;
};

// --- Choosing where Python runs ------------------------------------------------------

interface LocalConfig {
  url: string; // ws://127.0.0.1:8765
  token: string;
}

const LOCAL_KEY = "drone_sim.web.local"; // remembered for this tab (sessionStorage)

/** The local Python server to use, if any. `?local=<ws url>&token=<token>`
 * (removed from the address bar once read) selects it for this tab, the
 * server injects it into the app it serves, and `?local=off` forgets it. */
function localConfig(): LocalConfig | null {
  const params = new URLSearchParams(window.location.search);
  const local = params.get("local");
  if (local !== null) {
    const token = params.get("token");
    params.delete("local");
    params.delete("token");
    const query = params.toString();
    history.replaceState(null, "", window.location.pathname + (query ? `?${query}` : "") + window.location.hash);
    if (local === "off" || !token) {
      forgetLocal();
      return null;
    }
    const config = { url: local.replace(/\/+$/, ""), token };
    try {
      sessionStorage.setItem(LOCAL_KEY, JSON.stringify(config));
    } catch {
      // storage blocked: works until reload
    }
    return config;
  }
  const injected = (window as unknown as { __DRONE_SIM_LOCAL__?: LocalConfig }).__DRONE_SIM_LOCAL__;
  if (injected) return injected;
  try {
    const saved = sessionStorage.getItem(LOCAL_KEY);
    return saved ? (JSON.parse(saved) as LocalConfig) : null;
  } catch {
    return null;
  }
}

function forgetLocal() {
  try {
    sessionStorage.removeItem(LOCAL_KEY);
  } catch {
    // nothing to forget
  }
}

const LOCAL_TIMEOUT_MS = 3000;

// A request without its id (the client assigns one). Distributes over the union.
type RequestMessage = Extract<ToWorker, { id: number }> extends infer M
  ? M extends { id: number }
    ? Omit<M, "id">
    : never
  : never;

/** A Python exception, with the traceback as the message. */
export class PythonError extends Error {}

class SimClient {
  private send: (msg: ToWorker) => void = () => {};
  private local = false; // talking to the local Python server
  private nextId = 1;
  private pending = new Map<number, Pending>();
  private frameListeners = new Set<(f: LiveFrame) => void>();
  private liveErrorListeners = new Set<(error: string) => void>();
  private statusListeners = new Set<() => void>();
  private outputListeners = new Set<() => void>();
  status: SimStatus = { state: "loading", text: "Starting" };
  notice: string | null = null; // an optional package downloading, e.g. "Loading cvxpy"
  output: string[] = [];
  /** Why the local Python server isn't being used, when it was asked for. */
  localUnavailable: string | null = null;

  constructor() {
    const local = localConfig();
    if (local) this.connectLocal(local);
    else this.startWorker();
  }

  private startWorker() {
    this.local = false;
    const worker = new Worker(new URL("./worker.ts", import.meta.url), { type: "module" });
    worker.onmessage = (e: MessageEvent<FromWorker>) => this.handle(e.data);
    worker.onerror = (e) => this.setStatus({ state: "error", error: e.message || "The simulation worker failed to start." });
    this.send = (msg) => worker.postMessage(msg);
  }

  private connectLocal(config: LocalConfig) {
    this.local = true;
    this.setStatus({ state: "loading", text: "Connecting to local Python" });
    let ws: WebSocket;
    try {
      ws = new WebSocket(`${config.url}/?token=${encodeURIComponent(config.token)}`);
    } catch {
      this.useBrowserInstead(`${config.url} isn't a valid address`);
      return;
    }
    let opened = false;
    const queued: ToWorker[] = [];
    this.send = (msg) => (ws.readyState === WebSocket.OPEN ? ws.send(JSON.stringify(msg)) : queued.push(msg));
    const timer = setTimeout(() => {
      if (opened) return;
      ws.close();
      this.useBrowserInstead(`no answer from ${config.url}`);
    }, LOCAL_TIMEOUT_MS);
    ws.onopen = () => {
      opened = true;
      clearTimeout(timer);
      queued.splice(0).forEach((m) => ws.send(JSON.stringify(m)));
    };
    ws.onmessage = (e: MessageEvent<string>) => this.handle(JSON.parse(e.data) as FromWorker);
    ws.onclose = (e) => {
      clearTimeout(timer);
      if (!opened) {
        // Server not running, an old token (it changes on every start), or another site.
        this.useBrowserInstead(`couldn't connect to ${config.url}`);
        return;
      }
      for (const p of this.pending.values()) p.reject(new PythonError("Lost the connection to the local Python."));
      this.pending.clear();
      this.liveErrorListeners.forEach((f) => f("Lost the connection to the local Python."));
      const why = e.code === 4000 ? "Another tab connected to the local Python, so this one was disconnected." : "The local Python stopped.";
      this.setStatus({ state: "error", local: true, error: `${why} Start it again (python web/src/python/server.py) and open the link it prints, or reload to run Python in the browser.` });
    };
  }

  private useBrowserInstead(reason: string) {
    if (!this.local) return; // already switched
    forgetLocal();
    this.localUnavailable = reason;
    this.startWorker();
    this.setStatus({ state: "loading", text: "Starting" });
  }

  private setStatus(status: SimStatus) {
    this.status = status;
    this.statusListeners.forEach((f) => f());
  }

  private handle(msg: FromWorker) {
    switch (msg.type) {
      case "status":
        this.setStatus({ state: "loading", text: msg.text });
        break;
      case "ready":
        this.setStatus({ state: "ready", info: msg.info });
        break;
      case "fatal":
        this.setStatus({ state: "error", error: msg.error });
        break;
      case "progress":
        this.pending.get(msg.id)?.onProgress?.(msg.fraction);
        break;
      case "reply": {
        const p = this.pending.get(msg.id);
        this.pending.delete(msg.id);
        if (!p) break;
        if (msg.ok) p.resolve(this.local && p.convert ? p.convert(msg.value) : msg.value);
        else p.reject(new PythonError(msg.error));
        break;
      }
      case "live:frame":
        this.frameListeners.forEach((f) => f(msg.frame));
        break;
      case "live:error":
        this.liveErrorListeners.forEach((f) => f(msg.error));
        break;
      case "notice":
        this.notice = msg.text;
        this.statusListeners.forEach((f) => f());
        break;
      case "stdout":
        this.output = [...this.output, ...msg.text.split("\n")].slice(-500);
        this.outputListeners.forEach((f) => f());
        break;
    }
  }

  private request<T>(msg: RequestMessage, onProgress?: (f: number) => void, convert?: (v: unknown) => unknown) {
    const id = this.nextId++;
    const promise = new Promise<T>((resolve, reject) => {
      this.pending.set(id, { resolve: resolve as (v: unknown) => void, reject, onProgress, convert });
    });
    this.send({ ...msg, id } as ToWorker);
    return { id, promise };
  }

  // --- Requests ---

  runPlayback(opts: SimOptions, onProgress?: (fraction: number) => void) {
    const { id, promise } = this.request<PlaybackResult>(
      { type: "playback", opts },
      onProgress,
      (raw) => toPlaybackResult(raw as Record<string, unknown>, opts)[0], // the worker converts its own
    );
    return { promise, cancel: () => this.send({ type: "cancel", id }) };
  }

  check(controller: ControllerFile) {
    return this.request<CheckResult>({ type: "check", controller }).promise;
  }

  startLive(opts: SimOptions) {
    return this.request<null>({ type: "live:start", opts }).promise;
  }

  /** Open a stack file for editing. `controllers` are the page's current
   * controller files, so the stack sees edits made in the Controllers tab. */
  openStack(name: string, source: string, controllers: ControllerFile[]) {
    return this.request<StackReply>({ type: "stack:open", name, source, controllers }).promise;
  }

  editStack(edit: StackEdit) {
    return this.request<StackReply>({ type: "stack:edit", edit }).promise;
  }

  // --- Live controls ---

  key(key: SimKey, down: boolean) {
    this.send({ type: "live:key", key, down });
  }
  releaseKeys() {
    this.send({ type: "live:release" });
  }
  resetLive() {
    this.send({ type: "live:reset" });
  }
  pauseLive(paused: boolean) {
    this.send({ type: "live:pause", paused });
  }
  stopLive() {
    this.send({ type: "live:stop" });
  }

  // --- Subscriptions ---

  onFrame(f: (frame: LiveFrame) => void) {
    this.frameListeners.add(f);
    return () => void this.frameListeners.delete(f);
  }
  onLiveError(f: (error: string) => void) {
    this.liveErrorListeners.add(f);
    return () => void this.liveErrorListeners.delete(f);
  }
  subscribeStatus = (f: () => void) => {
    this.statusListeners.add(f);
    return () => void this.statusListeners.delete(f);
  };
  subscribeOutput = (f: () => void) => {
    this.outputListeners.add(f);
    return () => void this.outputListeners.delete(f);
  };
  clearOutput() {
    this.output = [];
    this.outputListeners.forEach((f) => f());
  }
}

export const sim = new SimClient();

export function useSimStatus(): SimStatus {
  return useSyncExternalStore(sim.subscribeStatus, () => sim.status);
}

export function useSimNotice(): string | null {
  return useSyncExternalStore(sim.subscribeStatus, () => sim.notice);
}

export function useSimOutput(): string[] {
  return useSyncExternalStore(sim.subscribeOutput, () => sim.output);
}
