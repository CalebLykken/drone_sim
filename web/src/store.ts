import { useSyncExternalStore } from "react";

/** A tiny observable value, for things that change every frame (the playback
 * clock) where re-rendering React on each change would be wasteful. */
export interface Store<T> {
  get(): T;
  set(value: T): void;
  subscribe(listener: () => void): () => void;
}

export function createStore<T>(initial: T): Store<T> {
  let value = initial;
  const listeners = new Set<() => void>();
  return {
    get: () => value,
    set(next) {
      if (Object.is(next, value)) return;
      value = next;
      listeners.forEach((f) => f());
    },
    subscribe(listener) {
      listeners.add(listener);
      return () => void listeners.delete(listener);
    },
  };
}

export function useStore<T>(store: Store<T>): T {
  return useSyncExternalStore(store.subscribe, store.get);
}

// --- Color scheme ---

const darkQuery = typeof window !== "undefined" ? window.matchMedia("(prefers-color-scheme: dark)") : null;
const subscribeScheme = (f: () => void) => {
  darkQuery?.addEventListener("change", f);
  return () => darkQuery?.removeEventListener("change", f);
};

export function useDarkMode(): boolean {
  return useSyncExternalStore(subscribeScheme, () => darkQuery?.matches ?? false);
}

/** Read a CSS custom property from :root (e.g. "--ink"). */
export function cssVar(name: string): string {
  return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
}

// --- localStorage, which can be missing or throw (private windows, blocked storage) ---

export function loadJson<T>(key: string, fallback: T): T {
  try {
    const raw = localStorage.getItem(key);
    return raw ? (JSON.parse(raw) as T) : fallback;
  } catch {
    return fallback;
  }
}

export function saveJson(key: string, value: unknown) {
  try {
    localStorage.setItem(key, JSON.stringify(value));
  } catch {
    // Storage full or unavailable: edits still live for this session.
  }
}

// --- Scenery for the 3D views (shared by the Fly and Scenarios tabs, kept per browser) ---

export type WorldName = "grid" | "forest";
export type TimeOfDay = "day" | "evening";

export const WORLDS: { name: WorldName; label: string }[] = [
  { name: "grid", label: "Grid" },
  { name: "forest", label: "Pine forest" },
];

export const TIMES: { name: TimeOfDay; label: string }[] = [
  { name: "day", label: "Daytime" },
  { name: "evening", label: "Evening" },
];

/** A store holding one of `options`, saved under `key` whenever it changes. */
function savedChoice<T extends string>(key: string, options: { name: T }[], fallback: T): Store<T> {
  const saved = loadJson<string>(key, fallback);
  const store = createStore<T>(options.find((o) => o.name === saved)?.name ?? fallback);
  store.subscribe(() => saveJson(key, store.get()));
  return store;
}

export const worldStore = savedChoice<WorldName>("drone_sim.web.world.v1", WORLDS, "grid");
export const timeStore = savedChoice<TimeOfDay>("drone_sim.web.time.v1", TIMES, "day");
