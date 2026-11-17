// Main-thread handle on the simulation worker: promise-based requests plus
// subscriptions for live frames, progress, and Python output.

import { useSyncExternalStore } from "react";
import type {
  CheckResult,
  ControllerFile,
  FromWorker,
  LiveFrame,
  PlaybackResult,
  SimInfo,
  SimKey,
  SimOptions,
  ToWorker,
} from "./protocol";

export type SimStatus =
  | { state: "loading"; text: string }
  | { state: "ready"; info: SimInfo }
  | { state: "error"; error: string };

type Pending = {
  resolve: (value: unknown) => void;
  reject: (error: Error) => void;
  onProgress?: (fraction: number) => void;
};

// A request without its id (the client assigns one). Distributes over the union.
type RequestMessage = Extract<ToWorker, { id: number }> extends infer M
  ? M extends { id: number }
    ? Omit<M, "id">
    : never
  : never;

/** A Python exception, with the traceback as the message. */
export class PythonError extends Error {}

class SimClient {
  private worker: Worker;
  private nextId = 1;
  private pending = new Map<number, Pending>();
  private frameListeners = new Set<(f: LiveFrame) => void>();
  private liveErrorListeners = new Set<(error: string) => void>();
  private statusListeners = new Set<() => void>();
  private outputListeners = new Set<() => void>();
  status: SimStatus = { state: "loading", text: "Starting" };
  notice: string | null = null; // an optional package downloading, e.g. "Loading cvxpy"
  output: string[] = [];

  constructor() {
    this.worker = new Worker(new URL("./worker.ts", import.meta.url), { type: "module" });
    this.worker.onmessage = (e: MessageEvent<FromWorker>) => this.handle(e.data);
    this.worker.onerror = (e) => this.setStatus({ state: "error", error: e.message || "The simulation worker failed to start." });
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
        if (msg.ok) p.resolve(msg.value);
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

  private request<T>(msg: RequestMessage, onProgress?: (f: number) => void) {
    const id = this.nextId++;
    const promise = new Promise<T>((resolve, reject) => {
      this.pending.set(id, { resolve: resolve as (v: unknown) => void, reject, onProgress });
    });
    this.worker.postMessage({ ...msg, id } as ToWorker);
    return { id, promise };
  }

  private send(msg: ToWorker) {
    this.worker.postMessage(msg);
  }

  // --- Requests ---

  runPlayback(opts: SimOptions, onProgress?: (fraction: number) => void) {
    const { id, promise } = this.request<PlaybackResult>({ type: "playback", opts }, onProgress);
    return { promise, cancel: () => this.send({ type: "cancel", id }) };
  }

  check(controller: ControllerFile) {
    return this.request<CheckResult>({ type: "check", controller }).promise;
  }

  startLive(opts: SimOptions) {
    return this.request<null>({ type: "live:start", opts }).promise;
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
