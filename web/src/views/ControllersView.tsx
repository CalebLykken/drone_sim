import { indentUnit } from "@codemirror/language";
import { python } from "@codemirror/lang-python";
import CodeMirror, { EditorView } from "@uiw/react-codemirror";
import { type ReactNode, useCallback, useId, useLayoutEffect, useMemo, useRef, useState } from "react";
import {
  type ControllerEntry,
  controllers,
  fileNameProblem,
  NEW_CONTROLLER_TEMPLATE,
  useControllerList,
  useControllerSource,
} from "../controllers";
import { sim, useSimOutput } from "../sim/client";
import type { CheckResult } from "../sim/protocol";
import { useDarkMode } from "../store";
import "./controllers.css";

type Check =
  | { id: number; state: "running" }
  | { id: number; state: "ok"; result: CheckResult }
  | { id: number; state: "error"; error: string };

type Confirm = "revert" | "delete";

/** The first of stem.py, stem2.py, stem3.py, ... that isn't taken. */
function freeName(stem: string): string {
  for (let i = 1; ; i++) {
    const name = `${stem}${i === 1 ? "" : i}.py`;
    if (!controllers.has(name)) return name;
  }
}

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

export function ControllersView({
  selected,
  onSelect,
  onFly,
  onRunScenario,
}: {
  selected: string; // file name currently open, e.g. "pid.py"
  onSelect: (name: string) => void;
  onFly: (name: string) => void;
  onRunScenario: (name: string) => void;
}) {
  const dark = useDarkMode();
  const list = useControllerList();
  const entry: ControllerEntry | undefined = list.find((c) => c.name === selected) ?? list[0];
  const name = entry?.name ?? "";
  const source = useControllerSource(name);

  const [check, setCheck] = useState<Check | null>(null);
  const [confirm, setConfirm] = useState<Confirm | null>(null);
  const [creating, setCreating] = useState(false);
  const runs = useRef(0);
  const focusEditorFor = useRef<string | null>(null);
  const checkButton = useRef<HTMLButtonElement>(null);
  const confirmTrigger = useRef<HTMLButtonElement>(null);
  const newButton = useRef<HTMLButtonElement>(null);

  // A different file is open: forget the last check and any pending confirmation.
  const [shownFor, setShownFor] = useState(name);
  if (shownFor !== name) {
    setShownFor(name);
    setCheck(null);
    setConfirm(null);
  }

  // The selected file is gone (deleted, or never existed): open the first one.
  useLayoutEffect(() => {
    if (entry && entry.name !== selected) onSelect(entry.name);
  }, [entry, selected, onSelect]);

  const extensions = useMemo(
    () => [python(), indentUnit.of("    "), EditorView.contentAttributes.of({ "aria-label": `Source of ${name}` })],
    [name],
  );
  const onChange = useCallback((value: string) => controllers.save(name, value), [name]);
  const onCreateEditor = useCallback(
    (view: EditorView) => {
      if (focusEditorFor.current !== name) return;
      focusEditorFor.current = null;
      view.focus();
    },
    [name],
  );

  const running = check?.state === "running";

  const runCheck = () => {
    if (running || !entry) return;
    const id = ++runs.current;
    setCheck({ id, state: "running" });
    // Results for a file that's no longer open (or an older click) are dropped.
    sim.check(controllers.file(name)).then(
      (result) => setCheck((c) => (c?.id === id ? { id, state: "ok", result } : c)),
      (e: unknown) =>
        setCheck((c) => (c?.id === id ? { id, state: "error", error: e instanceof Error ? e.message : String(e) } : c)),
    );
  };

  const create = (fileName: string) => {
    controllers.create(fileName, NEW_CONTROLLER_TEMPLATE);
    setCreating(false);
    focusEditorFor.current = fileName;
    onSelect(fileName);
  };

  const duplicate = () => {
    const stem = name.replace(/\.py$/, "").replace(/_copy\d*$/, "");
    const copy = freeName(`${stem}_copy`);
    controllers.create(copy, controllers.source(name));
    onSelect(copy);
  };

  const cancelConfirm = () => {
    setConfirm(null);
    confirmTrigger.current?.focus();
  };

  const revert = () => {
    controllers.revert(name);
    setConfirm(null);
    checkButton.current?.focus();
  };

  const remove = () => {
    controllers.remove(name);
    setConfirm(null);
    const first = controllers.list()[0];
    if (first) onSelect(first.name);
    checkButton.current?.focus();
  };

  return (
    <div className="ctl">
      <section className="ctl-files panel" aria-label="Controller files">
        <h2 className="ctl-heading">Controller files</h2>
        <ul className="ctl-list">
          {list.map((c) => (
            <li key={c.name}>
              <button
                type="button"
                className="ctl-file"
                aria-current={c.name === name ? "true" : undefined}
                onClick={() => onSelect(c.name)}
              >
                <span className="ctl-file-name">{c.name}</span>
                <Tags entry={c} />
              </button>
            </li>
          ))}
        </ul>
        {creating ? (
          <NewControllerForm
            onCreate={create}
            onCancel={() => {
              setCreating(false);
              requestAnimationFrame(() => newButton.current?.focus());
            }}
          />
        ) : (
          <div className="ctl-list-actions">
            <button ref={newButton} type="button" onClick={() => setCreating(true)}>
              New controller
            </button>
            {entry && (
              <button type="button" className="quiet" onClick={duplicate} aria-label={`Duplicate ${name}`}>
                Duplicate
              </button>
            )}
          </div>
        )}
      </section>

      <div className="hint ctl-help">
        <p>
          <code>update()</code> runs 250 times a second and returns four rotor thrusts in newtons: front-left,
          front-right, back-right, back-left. When you fly, <code>self.sticks</code> holds the pilot's keys.
        </p>
        <p>
          Edits are saved in this browser only and never change the repo's files. Use Download to keep a copy.
        </p>
        <p>In the editor, press Esc and then Tab to move focus out.</p>
      </div>

      <section className="ctl-editor panel" aria-label={entry ? `Editing ${name}` : "Editor"}>
        {entry ? (
          <>
            <div className="ctl-toolbar">
              <div className="ctl-title">
                <h2>{name}</h2>
                <Tags entry={entry} />
              </div>
              <div className="ctl-actions">
                <button
                  ref={checkButton}
                  type="button"
                  className={running ? "primary ctl-busy" : "primary"}
                  aria-disabled={running || undefined}
                  onClick={runCheck}
                >
                  {running ? "Checking…" : "Check"}
                </button>
                <button type="button" onClick={() => onFly(name)}>
                  Fly it
                </button>
                <button type="button" onClick={() => onRunScenario(name)}>
                  Run a scenario
                </button>
                <button type="button" className="quiet" onClick={() => download(name, controllers.source(name))}>
                  Download
                </button>
                {entry.builtin && entry.edited && (
                  <button ref={confirmTrigger} type="button" className="quiet ctl-danger" onClick={() => setConfirm("revert")}>
                    Revert to original
                  </button>
                )}
                {!entry.builtin && (
                  <button ref={confirmTrigger} type="button" className="quiet ctl-danger" onClick={() => setConfirm("delete")}>
                    Delete
                  </button>
                )}
              </div>
            </div>

            {confirm === "revert" && entry.edited && (
              <ConfirmBar
                message={`Go back to the original ${name}? Your changes to it will be lost.`}
                confirmLabel="Discard my changes"
                onConfirm={revert}
                onCancel={cancelConfirm}
              />
            )}
            {confirm === "delete" && !entry.builtin && (
              <ConfirmBar
                message={`Delete ${name} from this browser? This can't be undone, so download it first if you want a copy.`}
                confirmLabel={`Delete ${name}`}
                onConfirm={remove}
                onCancel={cancelConfirm}
              />
            )}

            <CodeMirror
              key={name}
              className="ctl-cm"
              value={source}
              height="100%"
              theme={dark ? "dark" : "light"}
              extensions={extensions}
              onChange={onChange}
              onCreateEditor={onCreateEditor}
            />
          </>
        ) : (
          <p className="hint ctl-empty">There are no controller files yet. Make one with New controller.</p>
        )}
      </section>

      <section className="ctl-output panel" aria-label="Output">
        <div className="ctl-section ctl-check">
          <h3 className="ctl-heading">Check result</h3>
          {entry && <CheckReport name={name} check={check} />}
        </div>
        <PythonConsole />
      </section>
    </div>
  );
}

// --- File tags ---------------------------------------------------------------------

function Tags({ entry }: { entry: ControllerEntry }) {
  return (
    <span className="ctl-tags">
      {entry.builtin ? <span className="ctl-tag">example</span> : <span className="ctl-tag ctl-tag-yours">yours</span>}
      {entry.edited && <span className="ctl-tag ctl-tag-edited">edited</span>}
    </span>
  );
}

// --- New controller form -------------------------------------------------------------

function NewControllerForm({ onCreate, onCancel }: { onCreate: (name: string) => void; onCancel: () => void }) {
  const [draft, setDraft] = useState(() => freeName("my_controller"));
  const input = useRef<HTMLInputElement>(null);
  const problemId = useId();
  const fileName = draft.trim();
  const problem = fileNameProblem(fileName);

  // Focus the name with the stem selected, so typing replaces "my_controller" but keeps ".py".
  useLayoutEffect(() => {
    const el = input.current;
    if (!el) return;
    el.focus();
    el.setSelectionRange(0, el.value.endsWith(".py") ? el.value.length - 3 : el.value.length);
  }, []);

  return (
    <form
      className="ctl-new"
      aria-label="New controller"
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
          value={draft}
          onChange={(e) => setDraft(e.target.value)}
          spellCheck={false}
          autoComplete="off"
          autoCapitalize="off"
          autoCorrect="off"
          aria-invalid={problem ? true : undefined}
          aria-describedby={problem ? problemId : undefined}
        />
      </label>
      {problem && (
        <p id={problemId} className="ctl-problem">
          {problem}
        </p>
      )}
      <div className="ctl-form-actions">
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

// --- Inline confirmation -------------------------------------------------------------

function ConfirmBar({
  message,
  confirmLabel,
  onConfirm,
  onCancel,
}: {
  message: string;
  confirmLabel: string;
  onConfirm: () => void;
  onCancel: () => void;
}) {
  const messageId = useId();
  return (
    <div
      className="ctl-confirm"
      role="group"
      aria-labelledby={messageId}
      onKeyDown={(e) => {
        if (e.key !== "Escape") return;
        e.preventDefault();
        onCancel();
      }}
    >
      <p id={messageId}>{message}</p>
      <div className="ctl-form-actions">
        <button type="button" className="ctl-danger-strong" onClick={onConfirm}>
          {confirmLabel}
        </button>
        <button type="button" className="quiet" onClick={onCancel} autoFocus>
          Cancel
        </button>
      </div>
    </div>
  );
}

// --- Check result --------------------------------------------------------------------

const LOAD_FRAMES = /\bin (_load|load_controller)\b/;

function CheckReport({ name, check }: { name: string; check: Check | null }) {
  let summary: ReactNode;
  let traceback: string | null = null;

  if (!check) {
    summary = (
      <p className="hint">Check loads {name} and flies it for 2 s on the hover scenario, so you find errors before you fly.</p>
    );
  } else if (check.state === "running") {
    summary = <p>Checking {name}…</p>;
  } else if (check.state === "ok") {
    const { metrics: m } = check.result;
    const dist = typeof m.final_pos_error_m === "number" ? `${m.final_pos_error_m.toFixed(2)} m` : "an unknown distance";
    const tilt = typeof m.max_tilt_deg === "number" ? `${m.max_tilt_deg.toFixed(1)}°` : "an unknown amount";
    summary = (
      <p className="ctl-ok">
        <span className="ctl-tick" aria-hidden="true">
          ✓{" "}
        </span>
        Loaded <code>{check.result.class}</code>. In a 2 s hover test it ended {dist} from the target with at most {tilt}{" "}
        of tilt.
        {m.flipped === true && <strong className="ctl-bad"> It flipped over.</strong>}
        {m.crashed === true && <strong className="ctl-bad"> It crashed.</strong>}
      </p>
    );
  } else {
    traceback = check.error;
    const line = LOAD_FRAMES.test(check.error)
      ? `${name} didn't load.`
      : check.error.includes("Traceback")
        ? `${name} loaded, but raised an error during the 2 s hover test.`
        : "The check stopped with an error.";
    summary = <p className="ctl-fail">{line}</p>;
  }

  return (
    <>
      <div className="ctl-status" aria-live="polite">
        {summary}
      </div>
      {traceback && (
        <pre className="ctl-traceback" tabIndex={0} aria-label="Python traceback">
          {traceback}
        </pre>
      )}
    </>
  );
}

// --- Python console ------------------------------------------------------------------

function PythonConsole() {
  const lines = useSimOutput();
  const log = useRef<HTMLPreElement>(null);
  const stick = useRef(true); // follow new lines unless the person scrolled up
  const hasText = lines.some((l) => l !== "");

  useLayoutEffect(() => {
    const el = log.current;
    if (el && stick.current) el.scrollTop = el.scrollHeight;
  }, [lines]);

  return (
    <div className="ctl-section ctl-console">
      <div className="ctl-section-head">
        <h3 className="ctl-heading">Python console</h3>
        <button
          type="button"
          className="quiet"
          onClick={() => {
            stick.current = true;
            sim.clearOutput();
          }}
        >
          Clear
        </button>
      </div>
      {hasText ? (
        <pre
          ref={log}
          className="ctl-log"
          tabIndex={0}
          aria-label="Python console output"
          onScroll={(e) => {
            const el = e.currentTarget;
            stick.current = el.scrollHeight - el.scrollTop - el.clientHeight < 24;
          }}
        >
          {lines.join("\n")}
        </pre>
      ) : (
        <p className="hint ctl-log-empty">print() output from your controllers shows up here.</p>
      )}
    </div>
  );
}
