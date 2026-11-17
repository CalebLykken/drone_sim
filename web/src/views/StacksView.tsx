// Stacks tab: a stack's node graph (like `python -m drone_sim --graph`) that
// you edit by double-clicking. Double-click a node to see or change its
// properties; double-click empty space to add a sensor (or any node).
//
// The Python side (drone_sim/editor) parses the stack file, validates every
// change by building the graph, and rewrites the file's stack() function.
// Edits are saved in this browser, like controllers.

import {
  type FormEvent,
  type KeyboardEvent as ReactKeyboardEvent,
  type PointerEvent as ReactPointerEvent,
  useCallback,
  useEffect,
  useId,
  useLayoutEffect,
  useMemo,
  useRef,
  useState,
} from "react";
import { controllers } from "../controllers";
import { sim } from "../sim/client";
import type {
  CatalogEntry,
  GraphVertex,
  SimInfo,
  StackEdit,
  StackParam,
  StackState,
} from "../sim/protocol";
import { NEW_STACK_TEMPLATE, type StackEntry, stackNameProblem, stacks, useStackList } from "../stacks";
import { scenarioLabel } from "../theme";
import "./stacks.css";

const S = 72; // pixels per layout unit (see drone_sim/nodes/layout.py)
const BOX_W = 2.5 * S;
const BOX_H = 0.72 * S;
const MIN_FIT_ZOOM = 0.8; // node labels stay at least ~12 px when fitting
const KIND_ORDER = ["sensor", "estimator", "controller", "planner", "node"];
const KIND_LABEL: Record<string, string> = {
  sensor: "Sensors",
  estimator: "Estimators",
  controller: "Controllers",
  planner: "Planners",
  node: "Other nodes",
};

type Inspect =
  | { mode: "overview" }
  | { mode: "node"; name: string }
  | { mode: "topic"; name: string }
  | { mode: "add"; key: string }
  | { mode: "source" };

type Load = { state: "loading" } | { state: "ready"; data: StackState } | { state: "error"; error: string };

const errorText = (e: unknown) => (e instanceof Error ? e.message : String(e));

function download(name: string, source: string) {
  const url = URL.createObjectURL(new Blob([source], { type: "text/x-python" }));
  const a = document.createElement("a");
  a.href = url;
  a.download = name;
  document.body.appendChild(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

function freeName(stem: string): string {
  for (let i = 1; ; i++) {
    const name = `${stem}${i === 1 ? "" : i}.py`;
    if (!stacks.has(name)) return name;
  }
}

export function StacksView({
  info,
  selected,
  onSelect,
}: {
  info: SimInfo;
  selected: string;
  onSelect: (name: string) => void;
}) {
  const list = useStackList();
  const entry: StackEntry | undefined = list.find((s) => s.name === selected) ?? list[0];
  const name = entry?.name ?? "";

  const [load, setLoad] = useState<Load>({ state: "loading" });
  const [catalog, setCatalog] = useState<CatalogEntry[]>([]);
  const [inspect, setInspect] = useState<Inspect>({ mode: "overview" });
  const [busy, setBusy] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [picker, setPicker] = useState<{ x: number; y: number } | null>(null);
  const [scenario, setScenario] = useState("square");
  const [duration, setDuration] = useState(6);
  const [confirm, setConfirm] = useState<"revert" | "delete" | null>(null);
  const [creating, setCreating] = useState(false);
  const [openCount, setOpenCount] = useState(0); // bumps on every (re)open, to refit the view
  const requests = useRef(0);

  // The selected file is gone (deleted, or never existed): open the first one.
  useLayoutEffect(() => {
    if (entry && entry.name !== selected) onSelect(entry.name);
  }, [entry, selected, onSelect]);

  const open = useCallback((fileName: string) => {
    const id = ++requests.current;
    setLoad({ state: "loading" });
    setInspect({ mode: "overview" });
    setError(null);
    setPicker(null);
    const files = controllers.list().map((c) => controllers.file(c.name));
    sim.openStack(fileName, stacks.source(fileName), files).then(
      (reply) => {
        if (id !== requests.current) return;
        if (reply.ok) {
          setLoad({ state: "ready", data: reply });
          setCatalog(reply.catalog ?? []);
          setOpenCount((n) => n + 1);
        } else {
          setLoad({ state: "error", error: reply.error });
        }
      },
      (e: unknown) => id === requests.current && setLoad({ state: "error", error: errorText(e) }),
    );
  }, []);

  useEffect(() => {
    if (name) open(name);
  }, [name, open]);

  /** Send an edit; on success show the new graph and keep the new source. */
  const apply = useCallback(
    async (edit: StackEdit, label: string): Promise<boolean> => {
      setBusy(label);
      setError(null);
      try {
        const reply = await sim.editStack(edit);
        if (!reply.ok) {
          setError(reply.error);
          return false;
        }
        setLoad({ state: "ready", data: reply });
        stacks.save(name, reply.source);
        if (reply.selected) setInspect({ mode: "node", name: reply.selected });
        return true;
      } catch (e) {
        setError(errorText(e));
        return false;
      } finally {
        setBusy(null);
      }
    },
    [name],
  );

  const data = load.state === "ready" ? load.data : null;
  const vertexCount = data?.graph.vertices.length ?? 0;

  const openVertex = useCallback((v: GraphVertex) => {
    setPicker(null);
    setError(null);
    setInspect(v.type === "node" ? { mode: "node", name: v.name } : { mode: "topic", name: v.name });
  }, []);

  const create = (fileName: string) => {
    stacks.create(fileName, NEW_STACK_TEMPLATE);
    setCreating(false);
    onSelect(fileName);
  };

  const duplicate = () => {
    const copy = freeName(`${name.replace(/\.py$/, "").replace(/_copy\d*$/, "")}_copy`);
    stacks.create(copy, stacks.source(name));
    onSelect(copy);
  };

  const testFlight = () => {
    void apply({ op: "test", scenario, duration }, "Flying…").then((ok) => ok && setInspect({ mode: "overview" }));
  };

  const selectedId =
    inspect.mode === "node" ? `node:${inspect.name}` : inspect.mode === "topic" ? `topic:${inspect.name}` : null;

  return (
    <div className="stk">
      <section className="stk-files panel" aria-label="Stack files">
        <h2 className="stk-heading">Stack files</h2>
        <ul className="stk-list">
          {list.map((s) => (
            <li key={s.name}>
              <button
                type="button"
                className="stk-file"
                aria-current={s.name === name ? "true" : undefined}
                onClick={() => onSelect(s.name)}
              >
                <span className="stk-file-name">{s.name}</span>
                <Tags entry={s} />
              </button>
            </li>
          ))}
        </ul>
        {creating ? (
          <NewStackForm onCreate={create} onCancel={() => setCreating(false)} />
        ) : (
          <div className="stk-list-actions">
            <button type="button" onClick={() => setCreating(true)}>
              New stack
            </button>
            {entry && (
              <button type="button" className="quiet" onClick={duplicate}>
                Duplicate
              </button>
            )}
          </div>
        )}
      </section>

      <section className="stk-main panel" aria-label={entry ? `Stack ${name}` : "Stack"}>
        {entry && (
          <div className="stk-toolbar">
            <div className="stk-title">
              <h2>{name}</h2>
              <Tags entry={entry} />
            </div>
            <div className="stk-actions">
              <button
                type="button"
                disabled={!data || busy !== null}
                onClick={() => setPicker({ x: -1, y: -1 })}
                title="Add a sensor or another node (or double-click empty space)"
              >
                Add node
              </button>
              <label className="stk-inline">
                <span>Test flight</span>
                <select value={scenario} onChange={(e) => setScenario(e.target.value)} aria-label="Test flight scenario">
                  {info.scenarios.map((s) => (
                    <option key={s} value={s}>
                      {scenarioLabel(s)}
                    </option>
                  ))}
                </select>
              </label>
              <label className="stk-inline stk-duration">
                <span className="stk-sr">Duration</span>
                <span className="number">
                  <input
                    type="number"
                    min={1}
                    max={30}
                    step={1}
                    value={duration}
                    onChange={(e) => setDuration(Math.min(30, Math.max(1, Number(e.target.value) || 6)))}
                    aria-label="Test flight duration in seconds"
                  />
                  <span className="unit">s</span>
                </span>
              </label>
              <button
                type="button"
                className={busy === "Flying…" ? "primary stk-busy" : "primary"}
                disabled={!data || busy !== null}
                onClick={testFlight}
              >
                {busy === "Flying…" ? "Flying…" : "Fly"}
              </button>
              <button
                type="button"
                className="quiet"
                aria-pressed={inspect.mode === "source"}
                onClick={() => setInspect(inspect.mode === "source" ? { mode: "overview" } : { mode: "source" })}
              >
                Source
              </button>
              <button type="button" className="quiet" onClick={() => download(name, stacks.source(name))}>
                Download
              </button>
              {entry.builtin && entry.edited && (
                <button type="button" className="quiet stk-danger" onClick={() => setConfirm("revert")}>
                  Revert
                </button>
              )}
              {!entry.builtin && (
                <button type="button" className="quiet stk-danger" onClick={() => setConfirm("delete")}>
                  Delete
                </button>
              )}
            </div>
          </div>
        )}

        {confirm && (
          <div className="stk-confirm" role="group">
            <p>
              {confirm === "revert"
                ? `Go back to the original ${name}? Your changes will be lost.`
                : `Delete ${name} from this browser? Download it first if you want a copy.`}
            </p>
            <div className="stk-form-actions">
              <button
                type="button"
                className="stk-danger-strong"
                onClick={() => {
                  if (confirm === "revert") {
                    stacks.revert(name);
                    open(name);
                  } else {
                    stacks.remove(name);
                  }
                  setConfirm(null);
                }}
              >
                {confirm === "revert" ? "Discard my changes" : `Delete ${name}`}
              </button>
              <button type="button" className="quiet" onClick={() => setConfirm(null)} autoFocus>
                Cancel
              </button>
            </div>
          </div>
        )}

        <div className="stk-canvas-wrap">
          {load.state === "loading" && <p className="stk-canvas-msg">Loading {name}…</p>}
          {load.state === "error" && (
            <div className="stk-canvas-msg stk-load-error" role="alert">
              <p>{name} couldn't be opened.</p>
              <pre>{load.error}</pre>
            </div>
          )}
          {data && (
            <StackGraph
              data={data}
              selectedId={selectedId}
              fitKey={`${openCount}:${vertexCount}`}
              onOpen={openVertex}
              onBackgroundDoubleClick={(p) => {
                setError(null);
                setPicker(p);
              }}
              onBackgroundClick={() => setPicker(null)}
            />
          )}
          {data && data.warnings.length > 0 && (
            <ul className="stk-warnings" aria-label="Warnings">
              {data.warnings.map((w) => (
                <li key={w}>{w}</li>
              ))}
            </ul>
          )}
          {busy && busy !== "Flying…" && <p className="stk-busy-note">{busy}</p>}
          {data && picker && (
            <NodePicker
              at={picker}
              catalog={catalog}
              onPick={(c) => {
                setPicker(null);
                setInspect({ mode: "add", key: c.key });
              }}
              onClose={() => setPicker(null)}
            />
          )}
          {data && (
            <p className="stk-hint">
              Double-click a node to edit it · double-click empty space to add a sensor · drag to pan · scroll to zoom
            </p>
          )}
        </div>
      </section>

      <section className="stk-inspector panel" aria-label="Properties">
        {data ? (
          <Inspector
            key={inspect.mode === "node" || inspect.mode === "topic" ? `${inspect.mode}:${inspect.name}` : inspect.mode === "add" ? `add:${inspect.key}` : inspect.mode}
            inspect={inspect}
            data={data}
            catalog={catalog}
            busy={busy}
            error={error}
            stackName={name}
            onInspect={(i) => {
              setError(null);
              setInspect(i);
            }}
            apply={apply}
          />
        ) : (
          <p className="hint">Properties of the node you double-click show up here.</p>
        )}
      </section>
    </div>
  );
}

// --- File tags and new-file form ----------------------------------------------------

function Tags({ entry }: { entry: StackEntry }) {
  return (
    <span className="stk-tags">
      {entry.builtin ? <span className="stk-tag">example</span> : <span className="stk-tag stk-tag-yours">yours</span>}
      {entry.edited && <span className="stk-tag stk-tag-edited">edited</span>}
    </span>
  );
}

function NewStackForm({ onCreate, onCancel }: { onCreate: (name: string) => void; onCancel: () => void }) {
  const [draft, setDraft] = useState(() => freeName("my_stack"));
  const input = useRef<HTMLInputElement>(null);
  const problemId = useId();
  const fileName = draft.trim();
  const problem = stackNameProblem(fileName);

  useLayoutEffect(() => {
    const el = input.current;
    if (!el) return;
    el.focus();
    el.setSelectionRange(0, el.value.endsWith(".py") ? el.value.length - 3 : el.value.length);
  }, []);

  return (
    <form
      className="stk-new"
      aria-label="New stack"
      onSubmit={(e) => {
        e.preventDefault();
        if (!problem) onCreate(fileName);
      }}
      onKeyDown={(e) => {
        if (e.key !== "Escape") return;
        e.preventDefault();
        onCancel();
      }}
    >
      <label>
        <span>File name</span>
        <input
          ref={input}
          type="text"
          value={draft}
          onChange={(e) => setDraft(e.target.value)}
          spellCheck={false}
          autoComplete="off"
          aria-invalid={problem ? true : undefined}
          aria-describedby={problem ? problemId : undefined}
        />
      </label>
      {problem && (
        <p id={problemId} className="stk-problem">
          {problem}
        </p>
      )}
      <div className="stk-form-actions">
        <button type="submit" className="primary" disabled={problem !== null}>
          Create
        </button>
        <button type="button" className="quiet" onClick={onCancel}>
          Cancel
        </button>
      </div>
    </form>
  );
}

// --- The graph -------------------------------------------------------------------------

type View = { x: number; y: number; k: number }; // screen = layout * S * k + (x, y)

function StackGraph({
  data,
  selectedId,
  fitKey,
  onOpen,
  onBackgroundDoubleClick,
  onBackgroundClick,
}: {
  data: StackState;
  selectedId: string | null;
  fitKey: string;
  onOpen: (v: GraphVertex) => void;
  onBackgroundDoubleClick: (p: { x: number; y: number }) => void;
  onBackgroundClick: () => void;
}) {
  const wrap = useRef<HTMLDivElement>(null);
  const [view, setView] = useState<View>({ x: 0, y: 0, k: 1 });
  const drag = useRef<{ x: number; y: number; vx: number; vy: number; moved: boolean } | null>(null);
  const arrowId = useId();
  const feedbackId = useId();

  const bounds = useMemo(() => {
    let x0 = Infinity,
      y0 = Infinity,
      x1 = -Infinity,
      y1 = -Infinity;
    const grow = (x: number, y: number) => {
      x0 = Math.min(x0, x);
      x1 = Math.max(x1, x);
      y0 = Math.min(y0, y);
      y1 = Math.max(y1, y);
    };
    for (const v of data.graph.vertices) {
      grow(v.x * S - BOX_W / 2, -v.y * S - BOX_H / 2);
      grow(v.x * S + BOX_W / 2, -v.y * S + BOX_H / 2);
    }
    for (const e of data.graph.edges) for (const [x, y] of e.points) grow(x * S, -y * S);
    return { x0, y0, x1, y1 };
  }, [data]);

  const fit = useCallback(() => {
    const el = wrap.current;
    if (!el || !Number.isFinite(bounds.x0)) return;
    const pad = 40;
    const w = el.clientWidth - 2 * pad;
    const h = el.clientHeight - 2 * pad - 30; // leave room for the hint line
    const fitK = Math.min(1.25, w / (bounds.x1 - bounds.x0), h / (bounds.y1 - bounds.y0));
    // Never shrink text below readable: a wide stack starts at MIN_FIT_ZOOM,
    // showing its start (the sim and sensors); drag or scroll for the rest.
    const k = Math.max(fitK, Math.min(MIN_FIT_ZOOM, h / (bounds.y1 - bounds.y0)));
    const width = (bounds.x1 - bounds.x0) * k;
    setView({
      k,
      x: width <= w ? pad + (w - width) / 2 - bounds.x0 * k : pad - bounds.x0 * k,
      y: pad + (h - (bounds.y1 - bounds.y0) * k) / 2 - bounds.y0 * k,
    });
  }, [bounds]);

  // Refit when a stack opens or nodes come and go; keep the view for property edits.
  const fitRef = useRef(fit);
  fitRef.current = fit;
  useLayoutEffect(() => {
    fitRef.current();
  }, [fitKey]);

  // Wheel zoom needs a non-passive listener to stop the page scrolling.
  useEffect(() => {
    const el = wrap.current;
    if (!el) return;
    const onWheel = (e: WheelEvent) => {
      e.preventDefault();
      const r = el.getBoundingClientRect();
      const px = e.clientX - r.left;
      const py = e.clientY - r.top;
      setView((v) => {
        const k = Math.min(3, Math.max(0.25, v.k * Math.exp(-e.deltaY * 0.0015)));
        return { k, x: px - ((px - v.x) * k) / v.k, y: py - ((py - v.y) * k) / v.k };
      });
    };
    el.addEventListener("wheel", onWheel, { passive: false });
    return () => el.removeEventListener("wheel", onWheel);
  }, []);

  const onPointerDown = (e: ReactPointerEvent) => {
    if (e.button !== 0 || (e.target as Element).closest(".stk-v")) return;
    drag.current = { x: e.clientX, y: e.clientY, vx: view.x, vy: view.y, moved: false };
    (e.currentTarget as Element).setPointerCapture(e.pointerId);
  };
  const onPointerMove = (e: ReactPointerEvent) => {
    const d = drag.current;
    if (!d) return;
    const dx = e.clientX - d.x;
    const dy = e.clientY - d.y;
    if (Math.abs(dx) + Math.abs(dy) > 3) d.moved = true;
    setView((v) => ({ ...v, x: d.vx + dx, y: d.vy + dy }));
  };
  const onPointerUp = () => {
    const d = drag.current;
    drag.current = null;
    if (d && !d.moved) onBackgroundClick();
  };

  const vertexKey = (e: ReactKeyboardEvent, v: GraphVertex) => {
    if (e.key === "Enter" || e.key === " ") {
      e.preventDefault();
      onOpen(v);
    }
  };

  const tx = (x: number) => x * S;
  const ty = (y: number) => -y * S;

  return (
    <div
      ref={wrap}
      className="stk-canvas"
      onPointerDown={onPointerDown}
      onPointerMove={onPointerMove}
      onPointerUp={onPointerUp}
      onDoubleClick={(e) => {
        if ((e.target as Element).closest(".stk-v, .stk-picker, .stk-warnings, .stk-fit")) return;
        const r = e.currentTarget.getBoundingClientRect();
        onBackgroundDoubleClick({ x: e.clientX - r.left, y: e.clientY - r.top });
      }}
    >
      <svg className="stk-svg" role="group" aria-label="Node graph">
        <defs>
          <marker id={arrowId} viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto">
            <path d="M0 0L10 5L0 10z" className="stk-arrow" />
          </marker>
          <marker id={feedbackId} viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto">
            <path d="M0 0L10 5L0 10z" className="stk-arrow-fb" />
          </marker>
        </defs>
        <g transform={`translate(${view.x} ${view.y}) scale(${view.k})`}>
          {data.graph.edges.map((e, i) => (
            <path
              key={i}
              className={e.feedback ? "stk-edge stk-edge-fb" : "stk-edge"}
              d={e.points.map(([x, y], j) => `${j ? "L" : "M"}${tx(x)} ${ty(y)}`).join(" ")}
              markerEnd={`url(#${e.feedback ? feedbackId : arrowId})`}
            />
          ))}
          {data.graph.vertices.map((v) => (
            <g
              key={v.id}
              className={`stk-v stk-v-${v.type}${v.id === selectedId ? " stk-v-selected" : ""}`}
              transform={`translate(${tx(v.x)} ${ty(v.y)})`}
              role="button"
              tabIndex={0}
              aria-label={`${v.type === "node" ? "Node" : "Topic"} ${v.label} (${v.sub}). Press Enter to open.`}
              onDoubleClick={(e) => {
                e.stopPropagation();
                onOpen(v);
              }}
              onKeyDown={(e) => vertexKey(e, v)}
            >
              <rect
                x={-BOX_W / 2}
                y={-BOX_H / 2}
                width={BOX_W}
                height={BOX_H}
                rx={v.type === "node" ? 16 : 2}
                style={v.type === "node" ? { fill: v.color } : undefined}
              />
              <text className="stk-v-label" y={-3}>
                {v.label}
              </text>
              <text className="stk-v-sub" y={14}>
                {v.sub}
              </text>
            </g>
          ))}
        </g>
      </svg>
      <button type="button" className="quiet stk-fit" onClick={fit} title="Fit the graph to the view">
        Fit
      </button>
    </div>
  );
}

// --- Picking a node type ----------------------------------------------------------------

function NodePicker({
  at,
  catalog,
  onPick,
  onClose,
}: {
  at: { x: number; y: number };
  catalog: CatalogEntry[];
  onPick: (c: CatalogEntry) => void;
  onClose: () => void;
}) {
  const [query, setQuery] = useState("");
  const [showAll, setShowAll] = useState(false);
  const box = useRef<HTMLDivElement>(null);
  const [pos, setPos] = useState({ left: 0, top: 0 });

  // Open at the double-click (or centered for the toolbar button), kept on screen.
  useLayoutEffect(() => {
    const el = box.current;
    const parent = el?.parentElement;
    if (!el || !parent) return;
    const w = el.offsetWidth;
    const h = el.offsetHeight;
    const pw = parent.clientWidth;
    const ph = parent.clientHeight;
    const left = at.x < 0 ? (pw - w) / 2 : Math.min(Math.max(8, at.x), pw - w - 8);
    const top = at.y < 0 ? 48 : Math.min(Math.max(8, at.y), ph - h - 8);
    setPos({ left, top });
  }, [at, showAll, query]);

  const q = query.trim().toLowerCase();
  const matches = catalog.filter(
    (c) => (showAll || q || c.kind === "sensor") && (!q || `${c.class} ${c.kind} ${c.doc}`.toLowerCase().includes(q)),
  );
  const groups = KIND_ORDER.map((k) => [k, matches.filter((c) => (KIND_ORDER.includes(c.kind) ? c.kind : "node") === k)] as const).filter(
    ([, items]) => items.length,
  );

  return (
    <div
      ref={box}
      className="stk-picker"
      style={{ left: pos.left, top: pos.top }}
      role="dialog"
      aria-label="Add a node"
      onKeyDown={(e) => {
        if (e.key === "Escape") {
          e.preventDefault();
          onClose();
        }
      }}
      onPointerDown={(e) => e.stopPropagation()}
    >
      <div className="stk-picker-head">
        <strong>{showAll || q ? "Add a node" : "Add a sensor"}</strong>
        <button type="button" className="quiet stk-close" onClick={onClose} aria-label="Close">
          ×
        </button>
      </div>
      <input
        type="text"
        className="stk-search"
        placeholder="Search"
        value={query}
        onChange={(e) => setQuery(e.target.value)}
        autoFocus
        aria-label="Search node types"
      />
      <div className="stk-picker-list">
        {groups.map(([kind, items]) => (
          <div key={kind} className="stk-picker-group">
            <h3 className="stk-heading">{KIND_LABEL[kind] ?? kind}</h3>
            {items.map((c) => (
              <button key={c.key} type="button" className="stk-picker-item" onClick={() => onPick(c)}>
                <span className="stk-dot" style={{ background: KIND_SWATCH[c.kind] ?? KIND_SWATCH.node }} aria-hidden="true" />
                <span className="stk-picker-text">
                  <span className="stk-picker-name">
                    {c.class}
                    {c.local && <span className="stk-tag">in this file</span>}
                  </span>
                  {c.doc && <span className="stk-picker-doc">{c.doc}</span>}
                </span>
              </button>
            ))}
          </div>
        ))}
        {groups.length === 0 && <p className="hint">Nothing matches “{query}”.</p>}
      </div>
      {!showAll && !q && (
        <button type="button" className="quiet stk-more" onClick={() => setShowAll(true)}>
          Show estimators and controllers too
        </button>
      )}
    </div>
  );
}

// Same colors as the graph (drone_sim/nodes/layout.py KIND_COLORS).
const KIND_SWATCH: Record<string, string> = {
  sim: "#d9d9d9",
  sensor: "#a6cee3",
  estimator: "#b2df8a",
  controller: "#fdbf6f",
  planner: "#cab2d6",
  node: "#ffffb3",
};

// --- Inspector --------------------------------------------------------------------------

function Inspector({
  inspect,
  data,
  catalog,
  busy,
  error,
  stackName,
  onInspect,
  apply,
}: {
  inspect: Inspect;
  data: StackState;
  catalog: CatalogEntry[];
  busy: string | null;
  error: string | null;
  stackName: string;
  onInspect: (i: Inspect) => void;
  apply: (edit: StackEdit, label: string) => Promise<boolean>;
}) {
  const [confirmDelete, setConfirmDelete] = useState(false);

  if (inspect.mode === "source") {
    return (
      <div className="stk-pane">
        <h2 className="stk-pane-title">{stackName}</h2>
        <p className="hint">The file as it is now. Changes you make in the graph rewrite its stack() function.</p>
        <pre className="stk-source" tabIndex={0} aria-label={`Source of ${stackName}`}>
          {data.source}
        </pre>
      </div>
    );
  }

  if (inspect.mode === "topic") {
    const v = data.graph.vertices.find((x) => x.type === "topic" && x.name === inspect.name);
    if (!v) return <Overview data={data} stackName={stackName} />;
    const nodeLink = (n: string) =>
      n === "sim" ? (
        <code key={n}>sim</code>
      ) : (
        <button key={n} type="button" className="stk-link" onClick={() => onInspect({ mode: "node", name: n })}>
          {n}
        </button>
      );
    return (
      <div className="stk-pane">
        <p className="stk-eyebrow">Topic</p>
        <h2 className="stk-pane-title stk-mono">{v.name}</h2>
        <dl className="stk-facts">
          <dt>Message</dt>
          <dd>
            <code>{v.msg_type}</code>
          </dd>
          <dt>Rate</dt>
          <dd>{v.rate ? `${v.rate.toFixed(0)} Hz (measured in the last test flight)` : "Fly to measure"}</dd>
          <dt>Published by</dt>
          <dd className="stk-links">{v.publishers?.length ? v.publishers.map(nodeLink) : "nothing"}</dd>
          <dt>Read by</dt>
          <dd className="stk-links">{v.subscribers?.length ? v.subscribers.map(nodeLink) : "nothing"}</dd>
        </dl>
        <BackButton onClick={() => onInspect({ mode: "overview" })} />
      </div>
    );
  }

  if (inspect.mode === "node") {
    if (inspect.name === "sim") {
      return (
        <div className="stk-pane">
          <p className="stk-eyebrow">Simulator</p>
          <h2 className="stk-pane-title">sim</h2>
          <p>
            The simulator bridge. It publishes the true state (<code>ground_truth</code>, for sensors), the target
            (<code>setpoint</code>), the pilot's keys (<code>sticks</code>) and the other drones (<code>/swarm</code>),
            and flies the drone with the latest <code>motor_command</code>. It's part of every stack and has no
            settings.
          </p>
          <BackButton onClick={() => onInspect({ mode: "overview" })} />
        </div>
      );
    }
    const node = data.nodes.find((n) => n.name === inspect.name);
    if (!node) return <Overview data={data} stackName={stackName} />;
    return (
      <div className="stk-pane">
        <p className="stk-eyebrow">
          <KindChip kind={node.kind} /> {node.class}
        </p>
        <h2 className="stk-pane-title">{node.name}</h2>
        {node.doc && <p className="hint">{node.doc}</p>}
        <ParamsForm
          params={node.params}
          submitLabel={busy === "Saving…" ? "Saving…" : "Apply"}
          busy={busy !== null}
          error={error}
          onSubmit={(values) => apply({ op: "update", name: node.name, params: values }, "Saving…")}
          onCancel={() => onInspect({ mode: "overview" })}
        />
        {confirmDelete ? (
          <div className="stk-confirm" role="group">
            <p>Remove {node.name} from the stack?</p>
            <div className="stk-form-actions">
              <button
                type="button"
                className="stk-danger-strong"
                onClick={() =>
                  void apply({ op: "remove", name: node.name }, "Removing…").then((ok) => ok && onInspect({ mode: "overview" }))
                }
              >
                Remove {node.name}
              </button>
              <button type="button" className="quiet" onClick={() => setConfirmDelete(false)} autoFocus>
                Cancel
              </button>
            </div>
          </div>
        ) : (
          <button type="button" className="quiet stk-danger stk-remove" onClick={() => setConfirmDelete(true)}>
            Remove node
          </button>
        )}
      </div>
    );
  }

  if (inspect.mode === "add") {
    const c = catalog.find((x) => x.key === inspect.key);
    if (!c) return <Overview data={data} stackName={stackName} />;
    return (
      <div className="stk-pane">
        <p className="stk-eyebrow">
          <KindChip kind={c.kind} /> New {c.kind}
        </p>
        <h2 className="stk-pane-title">Add {c.class}</h2>
        {c.doc && <p className="hint">{c.doc}</p>}
        <ParamsForm
          params={c.params}
          submitLabel={busy === "Adding…" ? "Adding…" : "Add to stack"}
          busy={busy !== null}
          error={error}
          onSubmit={(values) => apply({ op: "add", key: c.key, params: values }, "Adding…")}
          onCancel={() => onInspect({ mode: "overview" })}
        />
      </div>
    );
  }

  return <Overview data={data} stackName={stackName} />;
}

function Overview({ data, stackName }: { data: StackState; stackName: string }) {
  const counts = new Map<string, number>();
  for (const n of data.nodes) counts.set(n.kind, (counts.get(n.kind) ?? 0) + 1);
  const m = data.flight?.metrics;
  const num = (x: unknown, digits: number, unit: string) => (typeof x === "number" ? `${x.toFixed(digits)} ${unit}` : "–");
  return (
    <div className="stk-pane">
      <p className="stk-eyebrow">Stack</p>
      <h2 className="stk-pane-title">{stackName}</h2>
      <p>
        {data.nodes.length} nodes:{" "}
        {[...counts.entries()].map(([k, n], i) => (
          <span key={k}>
            {i > 0 && ", "}
            {n} {k}
            {n === 1 ? "" : "s"}
          </span>
        ))}
        .
      </p>
      {data.warnings.length > 0 && (
        <div className="stk-warn-box">
          <h3 className="stk-heading">Loose ends</h3>
          <ul>
            {data.warnings.map((w) => (
              <li key={w}>{w}</li>
            ))}
          </ul>
        </div>
      )}
      {m ? (
        <div className="stk-flight">
          <h3 className="stk-heading">
            Test flight: {scenarioLabel(data.flight!.scenario)}, {data.flight!.duration} s
          </h3>
          <dl className="stk-facts">
            <dt>Final distance to target</dt>
            <dd>{num(m.final_pos_error_m, 2, "m")}</dd>
            <dt>Average distance (RMS)</dt>
            <dd>{num(m.rms_pos_error_m, 2, "m")}</dd>
            <dt>Biggest tilt</dt>
            <dd>{num(m.max_tilt_deg, 1, "°")}</dd>
            <dt>Flipped or crashed</dt>
            <dd className={m.flipped || m.crashed ? "stk-bad" : "stk-good"}>{m.flipped || m.crashed ? "Yes" : "No"}</dd>
          </dl>
          <p className="hint">Topic boxes now show the rates measured during the flight.</p>
        </div>
      ) : (
        <p className="hint">Fly runs the stack headless on the chosen scenario and measures every topic's rate.</p>
      )}
      <ul className="stk-tips hint">
        <li>Double-click a node to see or change its properties.</li>
        <li>Double-click a topic to see who publishes and reads it.</li>
        <li>Double-click empty space to add a sensor.</li>
      </ul>
    </div>
  );
}

function KindChip({ kind }: { kind: string }) {
  return (
    <span className="stk-kind" style={{ background: KIND_SWATCH[kind] ?? KIND_SWATCH.node }}>
      {kind}
    </span>
  );
}

function BackButton({ onClick }: { onClick: () => void }) {
  return (
    <button type="button" className="quiet stk-back" onClick={onClick}>
      Back to stack
    </button>
  );
}

// --- Property form ----------------------------------------------------------------------

type FieldValue = string | boolean | string[] | [string, string][];

const label = (name: string) => name.replace(/_/g, " ");

function initial(p: StackParam): FieldValue {
  const v = p.value ?? p.default;
  switch (p.type) {
    case "bool":
      return Boolean(v);
    case "vector": {
      const arr = Array.isArray(v) ? v : Array(p.length ?? 3).fill(0);
      return arr.map((x) => String(x));
    }
    case "mapping":
      return v && typeof v === "object"
        ? Object.entries(v as Record<string, string>).map(([a, b]): [string, string] => [a, String(b)])
        : [];
    default:
      return v === null || v === undefined ? "" : String(v);
  }
}

function toJson(p: StackParam, v: FieldValue): unknown {
  switch (p.type) {
    case "number":
    case "integer":
      return v === "" ? null : Number(v);
    case "bool":
      return v;
    case "vector":
      return (v as string[]).map((x) => Number(x) || 0);
    case "mapping":
      return Object.fromEntries((v as [string, string][]).filter(([a, b]) => a.trim() && b.trim()));
    default:
      return v === "" ? null : v;
  }
}

function describeDefault(p: StackParam): string | null {
  if (p.required) return "required";
  const d = p.default;
  if (d === null || d === undefined || (typeof d === "object" && !Array.isArray(d) && Object.keys(d as object).length === 0))
    return null;
  return `default ${Array.isArray(d) ? `(${d.join(", ")})` : String(d)}`;
}

function ParamsForm({
  params,
  submitLabel,
  busy,
  error,
  onSubmit,
  onCancel,
}: {
  params: StackParam[];
  submitLabel: string;
  busy: boolean;
  error: string | null;
  onSubmit: (values: Record<string, unknown>) => Promise<boolean>;
  onCancel: () => void;
}) {
  const [values, setValues] = useState<Record<string, FieldValue>>(() =>
    Object.fromEntries(params.map((p) => [p.name, initial(p)])),
  );
  const set = (name: string, v: FieldValue) => setValues((s) => ({ ...s, [name]: v }));
  const errorId = useId();

  const submit = (e: FormEvent) => {
    e.preventDefault();
    if (busy) return;
    void onSubmit(Object.fromEntries(params.map((p) => [p.name, toJson(p, values[p.name])])));
  };

  return (
    <form className="stk-form" onSubmit={submit} aria-describedby={error ? errorId : undefined}>
      {params.map((p) => (
        <Field key={p.name} param={p} value={values[p.name]} onChange={(v) => set(p.name, v)} />
      ))}
      {error && (
        <div id={errorId} className="stk-error" role="alert">
          {error.includes("\n") ? <pre>{error}</pre> : <p>{error}</p>}
        </div>
      )}
      <div className="stk-form-actions">
        <button type="submit" className="primary" disabled={busy}>
          {submitLabel}
        </button>
        <button type="button" className="quiet" onClick={onCancel}>
          Cancel
        </button>
      </div>
    </form>
  );
}

function Field({ param: p, value, onChange }: { param: StackParam; value: FieldValue; onChange: (v: FieldValue) => void }) {
  const hint = describeDefault(p);
  const id = useId();
  const head = (
    <span className="stk-field-head">
      <span className="stk-field-name">{label(p.name)}</span>
      {hint && <span className="stk-field-default">{hint}</span>}
    </span>
  );

  if (p.type === "bool") {
    return (
      <label className="check">
        <input type="checkbox" checked={value as boolean} onChange={(e) => onChange(e.target.checked)} />
        <span>{label(p.name)}</span>
      </label>
    );
  }

  if (p.type === "vector") {
    const parts = value as string[];
    return (
      <div className="stk-field" role="group" aria-labelledby={id}>
        <span id={id}>{head}</span>
        <div className="stk-vector">
          {parts.map((x, i) => (
            <input
              key={i}
              type="number"
              step="any"
              value={x}
              aria-label={`${label(p.name)} ${["x", "y", "z", "w"][i] ?? i + 1}`}
              onChange={(e) => onChange(parts.map((y, j) => (j === i ? e.target.value : y)))}
            />
          ))}
        </div>
      </div>
    );
  }

  if (p.type === "mapping") {
    const rows = value as [string, string][];
    return (
      <div className="stk-field" role="group" aria-labelledby={id}>
        <span id={id}>
          <span className="stk-field-head">
            <span className="stk-field-name">{label(p.name)}</span>
            <span className="stk-field-default">topic → new name</span>
          </span>
        </span>
        {rows.map(([a, b], i) => (
          <div key={i} className="stk-remap">
            <input
              type="text"
              value={a}
              placeholder="topic"
              aria-label="Topic"
              onChange={(e) => onChange(rows.map((r, j): [string, string] => (j === i ? [e.target.value, r[1]] : r)))}
            />
            <span aria-hidden="true">→</span>
            <input
              type="text"
              value={b}
              placeholder="new name"
              aria-label="Rename to"
              onChange={(e) => onChange(rows.map((r, j): [string, string] => (j === i ? [r[0], e.target.value] : r)))}
            />
            <button
              type="button"
              className="quiet stk-close"
              aria-label={`Remove remap of ${a || "topic"}`}
              onClick={() => onChange(rows.filter((_, j) => j !== i))}
            >
              ×
            </button>
          </div>
        ))}
        <button type="button" className="quiet stk-add-row" onClick={() => onChange([...rows, ["", ""]])}>
          Add a remap
        </button>
      </div>
    );
  }

  if (p.type === "controller") {
    const choices = p.choices ?? [];
    const v = value as string;
    return (
      <label>
        {head}
        <select value={v} onChange={(e) => onChange(e.target.value)}>
          {!choices.includes(v) && v && <option value={v}>{v}</option>}
          {choices.map((c) => (
            <option key={c} value={c}>
              {c.replace(/^controllers\//, "")}
            </option>
          ))}
        </select>
      </label>
    );
  }

  const numeric = p.type === "number" || p.type === "integer";
  return (
    <label>
      {head}
      <input
        type={numeric ? "number" : "text"}
        step={p.type === "integer" ? 1 : "any"}
        value={value as string}
        placeholder={p.default === null || p.default === undefined ? "" : String(p.default)}
        spellCheck={false}
        autoComplete="off"
        onChange={(e) => onChange(e.target.value)}
      />
    </label>
  );
}
