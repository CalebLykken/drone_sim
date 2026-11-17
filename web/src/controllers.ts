// The controller files the page knows about: the repo's controllers/*.py
// (bundled at build time) plus any the person creates. Edits are kept in this
// browser's localStorage; the repo files themselves never change.

import { useSyncExternalStore } from "react";
import type { ControllerFile } from "./sim/protocol";
import { loadJson, saveJson } from "./store";

const bundled = import.meta.glob("../../controllers/*.py", {
  query: "?raw",
  import: "default",
  eager: true,
}) as Record<string, string>;

export const BUILTIN: Record<string, string> = Object.fromEntries(
  Object.entries(bundled).map(([path, src]) => [path.slice(path.lastIndexOf("/") + 1), src]),
);

const STORAGE_KEY = "drone_sim.web.controllers.v1";

interface Saved {
  edits: Record<string, string>; // changed copies of built-in files
  custom: Record<string, string>; // files the person created
}

export interface ControllerEntry {
  name: string;
  builtin: boolean;
  edited: boolean;
}

let saved: Saved = { edits: {}, custom: {}, ...loadJson<Partial<Saved>>(STORAGE_KEY, {}) };
let entries: ControllerEntry[] = [];
const listeners = new Set<() => void>();

function refresh() {
  entries = [
    ...Object.keys(BUILTIN)
      .sort()
      .map((name) => ({ name, builtin: true, edited: name in saved.edits })),
    ...Object.keys(saved.custom)
      .filter((name) => !(name in BUILTIN))
      .sort()
      .map((name) => ({ name, builtin: false, edited: false })),
  ];
}
refresh();

let saveTimer: ReturnType<typeof setTimeout> | null = null;
function commit() {
  refresh();
  listeners.forEach((f) => f());
  if (saveTimer) clearTimeout(saveTimer);
  saveTimer = setTimeout(() => saveJson(STORAGE_KEY, saved), 300);
}

export const controllers = {
  list: () => entries,

  has: (name: string) => name in BUILTIN || name in saved.custom,

  source(name: string): string {
    return saved.edits[name] ?? saved.custom[name] ?? BUILTIN[name] ?? "";
  },

  file(name: string): ControllerFile {
    return { name, source: controllers.source(name) };
  },

  save(name: string, source: string) {
    if (name in BUILTIN) {
      if (source === BUILTIN[name]) delete saved.edits[name];
      else saved.edits = { ...saved.edits, [name]: source };
    } else {
      saved.custom = { ...saved.custom, [name]: source };
    }
    saved = { ...saved };
    commit();
  },

  revert(name: string) {
    const { [name]: _, ...rest } = saved.edits;
    saved = { ...saved, edits: rest };
    commit();
  },

  create(name: string, source: string) {
    saved = { ...saved, custom: { ...saved.custom, [name]: source } };
    commit();
  },

  remove(name: string) {
    const { [name]: _, ...rest } = saved.custom;
    saved = { ...saved, custom: rest };
    commit();
  },

  subscribe(f: () => void) {
    listeners.add(f);
    return () => void listeners.delete(f);
  },
};

export function useControllerList(): ControllerEntry[] {
  return useSyncExternalStore(controllers.subscribe, controllers.list);
}

/** Re-renders when any controller's source changes. */
export function useControllerSource(name: string): string {
  return useSyncExternalStore(controllers.subscribe, () => controllers.source(name));
}

/** Why a new file name won't work, or null if it's fine. */
export function fileNameProblem(name: string): string | null {
  if (!name.endsWith(".py")) return "End the name with .py";
  const stem = name.slice(0, -3);
  if (!/^[A-Za-z_][A-Za-z0-9_]*$/.test(stem)) return "Use letters, digits and underscores, starting with a letter";
  if (controllers.has(name)) return `${name} already exists`;
  return null;
}

export const NEW_CONTROLLER_TEMPLATE = `"""My controller.

update() runs 250 times a second. Return four rotor thrusts in newtons:
(front-left, front-right, back-right, back-left).

state:  .pos .vel (world, m and m/s), .quat, .omega, .R, .euler (roll, pitch, yaw)
target: (x, y, z, yaw) to fly to
self.sticks: pilot inputs (throttle, roll, pitch, yaw in [-1, 1]) when you fly it live
"""

import numpy as np

from drone_sim import Controller
from drone_sim.control import thrust_vector, track_thrust_vector


class MyController(Controller):
    def reset(self, params):
        self.p = params

    def update(self, t, state, target, dt):
        *goal, yaw = target
        # Spring toward the target, damped by velocity.
        acc = 1.5 * (np.array(goal) - state.pos) - 2.0 * state.vel
        f = thrust_vector(acc, self.p)
        return track_thrust_vector(f, yaw, state, self.p)
`;
