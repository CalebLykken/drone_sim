import { useCallback, useEffect, useState } from "react";
import { controllers } from "./controllers";
import { type SimStatus, useSimNotice, useSimStatus } from "./sim/client";
import type { PlaybackResult } from "./sim/protocol";
import { ControllersView } from "./views/ControllersView";
import { FlyView } from "./views/FlyView";
import { ScenarioView } from "./views/ScenarioView";
import { StacksView } from "./views/StacksView";
import { stacks } from "./stacks";

const TABS = [
  { id: "fly", label: "Fly" },
  { id: "scenarios", label: "Scenarios" },
  { id: "controllers", label: "Controllers" },
  { id: "stacks", label: "Stacks" },
] as const;
type Tab = (typeof TABS)[number]["id"];

const tabFromHash = (): Tab => {
  const id = window.location.hash.slice(1);
  return TABS.some((t) => t.id === id) ? (id as Tab) : "fly";
};

const pick = (preferred: string) => (controllers.has(preferred) ? preferred : (controllers.list()[0]?.name ?? preferred));

export function App() {
  const status = useSimStatus();
  const [tab, setTabState] = useState<Tab>(tabFromHash);
  const [flyController, setFlyController] = useState(() => pick("manual.py"));
  const [scenarioController, setScenarioController] = useState(() => pick("pid.py"));
  const [editing, setEditing] = useState(() => pick("pid.py"));
  const [stack, setStack] = useState(() => (stacks.has("gps_imu_pid.py") ? "gps_imu_pid.py" : (stacks.list()[0]?.name ?? "")));
  const [result, setResult] = useState<PlaybackResult | null>(null);
  const [pendingRun, setPendingRun] = useState(false);

  const setTab = useCallback((t: Tab) => {
    setTabState(t);
    if (window.location.hash !== `#${t}`) history.replaceState(null, "", `#${t}`);
  }, []);

  useEffect(() => {
    const onHash = () => setTabState(tabFromHash());
    window.addEventListener("hashchange", onHash);
    return () => window.removeEventListener("hashchange", onHash);
  }, []);

  const edit = useCallback(
    (name: string) => {
      setEditing(name);
      setTab("controllers");
    },
    [setTab],
  );

  return (
    <div className="app">
      <header className="topbar">
        <a className="brand" href="#fly" onClick={() => setTab("fly")}>
          <svg viewBox="0 0 32 32" aria-hidden="true">
            <g className="brand-arms">
              <path d="M8 8l16 16M24 8L8 24" />
            </g>
            <circle className="brand-front" cx="8" cy="8" r="5" />
            <circle className="brand-front" cx="24" cy="8" r="5" />
            <circle className="brand-back" cx="8" cy="24" r="5" />
            <circle className="brand-back" cx="24" cy="24" r="5" />
            <rect className="brand-body" x="12" y="12" width="8" height="8" rx="2" />
          </svg>
          drone_sim
        </a>
        <nav className="tabs" role="tablist" aria-label="Views">
          {TABS.map((t) => (
            <button
              key={t.id}
              type="button"
              role="tab"
              className="tab"
              aria-selected={tab === t.id}
              onClick={() => setTab(t.id)}
            >
              {t.label}
            </button>
          ))}
        </nav>
        <PythonStatus status={status} />
      </header>

      <main className="app-main">
        {status.state !== "ready" ? (
          <Boot status={status} />
        ) : tab === "fly" ? (
          <FlyView
            info={status.info}
            controller={flyController}
            onControllerChange={setFlyController}
            onEditController={edit}
          />
        ) : tab === "scenarios" ? (
          <ScenarioView
            info={status.info}
            controller={scenarioController}
            onControllerChange={setScenarioController}
            onEditController={edit}
            result={result}
            onResult={setResult}
            autoRun={pendingRun}
            onAutoRun={() => setPendingRun(false)}
          />
        ) : tab === "stacks" ? (
          <StacksView info={status.info} selected={stack} onSelect={setStack} />
        ) : (
          <ControllersView
            selected={editing}
            onSelect={setEditing}
            onFly={(name) => {
              setFlyController(name);
              setTab("fly");
            }}
            onRunScenario={(name) => {
              setScenarioController(name);
              setPendingRun(true);
              setTab("scenarios");
            }}
          />
        )}
      </main>
    </div>
  );
}

function PythonStatus({ status }: { status: SimStatus }) {
  const notice = useSimNotice();
  if (status.state === "ready" && notice) return <span className="py-status loading">{notice}…</span>;
  if (status.state === "ready")
    return (
      <span className="py-status ready" title="drone_sim runs in your browser with Pyodide">
        Python {status.info.python}, numpy {status.info.numpy}
      </span>
    );
  if (status.state === "error") return <span className="py-status error">Python didn't start</span>;
  return <span className="py-status loading">{status.text}…</span>;
}

const BOOT_STEPS = ["Downloading Python", "Loading numpy", "Loading drone_sim"];

function Boot({ status }: { status: SimStatus }) {
  if (status.state === "error") {
    return (
      <div className="boot">
        <div className="boot-card panel" role="alert">
          <h1>Python didn't start</h1>
          <p>
            The simulator runs Python in your browser and downloads it from cdn.jsdelivr.net the first time. Check your
            connection, then reload.
          </p>
          <pre className="boot-error">{status.error}</pre>
          <button type="button" className="primary" onClick={() => window.location.reload()}>
            Reload
          </button>
        </div>
      </div>
    );
  }
  const current = status.state === "loading" ? BOOT_STEPS.indexOf(status.text) : BOOT_STEPS.length;
  return (
    <div className="boot">
      <div className="boot-card panel" aria-live="polite">
        <h1>Starting the simulator</h1>
        <p>
          drone_sim's Python code runs right here in your browser. The first visit downloads about 12 MB; after that
          it's cached.
        </p>
        <ol className="boot-steps">
          {BOOT_STEPS.map((step, i) => (
            <li key={step} className={i < current ? "done" : i === current ? "active" : ""}>
              {step}
            </li>
          ))}
        </ol>
      </div>
    </div>
  );
}
