// The stack files the page knows about: the repo's stacks/*.py (bundled at
// build time) plus any the person creates. Like controllers.ts, edits are kept
// in this browser's localStorage; the repo files themselves never change.

import { useSyncExternalStore } from "react";
import { loadJson, saveJson } from "./store";

const bundled = import.meta.glob("../../stacks/*.py", {
  query: "?raw",
  import: "default",
  eager: true,
}) as Record<string, string>;

export const BUILTIN_STACKS: Record<string, string> = Object.fromEntries(
  Object.entries(bundled).map(([path, src]) => [path.slice(path.lastIndexOf("/") + 1), src]),
);

const STORAGE_KEY = "drone_sim.web.stacks.v1";

interface Saved {
  edits: Record<string, string>; // changed copies of built-in files
  custom: Record<string, string>; // files the person created
}

export interface StackEntry {
  name: string;
  builtin: boolean;
  edited: boolean;
}

let saved: Saved = { edits: {}, custom: {}, ...loadJson<Partial<Saved>>(STORAGE_KEY, {}) };
let entries: StackEntry[] = [];
const listeners = new Set<() => void>();

function refresh() {
  entries = [
    ...Object.keys(BUILTIN_STACKS)
      .sort()
      .map((name) => ({ name, builtin: true, edited: name in saved.edits })),
    ...Object.keys(saved.custom)
      .filter((name) => !(name in BUILTIN_STACKS))
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

export const stacks = {
  list: () => entries,

  has: (name: string) => name in BUILTIN_STACKS || name in saved.custom,

  source(name: string): string {
    return saved.edits[name] ?? saved.custom[name] ?? BUILTIN_STACKS[name] ?? "";
  },

  save(name: string, source: string) {
    if (name in BUILTIN_STACKS) {
      if (source === BUILTIN_STACKS[name]) {
        const { [name]: _, ...rest } = saved.edits;
        saved = { ...saved, edits: rest };
      } else {
        saved = { ...saved, edits: { ...saved.edits, [name]: source } };
      }
    } else {
      saved = { ...saved, custom: { ...saved.custom, [name]: source } };
    }
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

export function useStackList(): StackEntry[] {
  return useSyncExternalStore(stacks.subscribe, stacks.list);
}

/** Why a new file name won't work, or null if it's fine. */
export function stackNameProblem(name: string): string | null {
  if (!name.endsWith(".py")) return "End the name with .py";
  const stem = name.slice(0, -3);
  if (!/^[A-Za-z_][A-Za-z0-9_]*$/.test(stem)) return "Use letters, digits and underscores, starting with a letter";
  if (stacks.has(name)) return `${name} already exists`;
  return null;
}

export const NEW_STACK_TEMPLATE = `"""My stack: perfect state feedback into the PID.

Double-click empty space in the graph to add sensors, then add an estimator
and point the controller's state at it.
"""

from pathlib import Path

from drone_sim.nodes import ControllerNode, PerfectState

CONTROLLERS = Path(__file__).resolve().parent.parent / "controllers"


def stack():
    return [
        PerfectState(),
        ControllerNode(CONTROLLERS / "pid.py"),
    ]
`;
