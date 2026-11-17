import { Canvas, useFrame, useThree } from "@react-three/fiber";
import { forwardRef, useCallback, useEffect, useImperativeHandle, useMemo, useRef, useState } from "react";
import * as THREE from "three";
import { controllers, useControllerList } from "../controllers";
import { sim } from "../sim/client";
import type { LiveFrame, SimInfo, SimKey } from "../sim/protocol";
import { useDarkMode } from "../store";
import { droneColor, HIVIS, scenarioLabel } from "../theme";
import { DroneModel, GroundShadow, TargetMarker, Trail, type TrailHandle, World } from "../three/scene";

const KEYS: Record<string, SimKey> = {
  ArrowUp: "up",
  ArrowDown: "down",
  ArrowLeft: "left",
  ArrowRight: "right",
  w: "w",
  a: "a",
  s: "s",
  d: "d",
  r: "r",
};

const simKey = (e: KeyboardEvent): SimKey | null => KEYS[e.key] ?? KEYS[e.key.toLowerCase()] ?? null;

const isTyping = (t: EventTarget | null) =>
  t instanceof HTMLElement && (t.isContentEditable || ["INPUT", "TEXTAREA", "SELECT"].includes(t.tagName));

const WIND = [
  { label: "Calm", gust: 0 },
  { label: "Breezy", gust: 0.5 },
  { label: "Gusty", gust: 1.5 },
];

const TRAIL_POINTS = 2400; // ~40 s at 60 frames a second

export function FlyView({
  info,
  controller,
  onControllerChange,
  onEditController,
}: {
  info: SimInfo;
  controller: string;
  onControllerChange: (name: string) => void;
  onEditController: (name: string) => void;
}) {
  const dark = useDarkMode();
  const list = useControllerList();
  const [scenario, setScenario] = useState("square");
  const [drones, setDrones] = useState(1);
  const [gust, setGust] = useState(0);
  const [restarts, setRestarts] = useState(0);
  const [paused, setPausedState] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [crashed, setCrashed] = useState(false);
  const [held, setHeld] = useState<ReadonlySet<SimKey>>(new Set());

  const frame = useRef<LiveFrame | null>(null);
  const trails = useRef<(TrailHandle | null)[]>([]);
  const hud = useRef<HudHandle>(null);
  const lastT = useRef(0);
  const pausedRef = useRef(false);
  const crashedRef = useRef(false);

  const wingman = controllers.has("pid.py") ? "pid.py" : controller;

  const setPaused = useCallback((p: boolean) => {
    if (pausedRef.current === p) return;
    pausedRef.current = p;
    setPausedState(p);
    sim.pauseLive(p);
  }, []);

  // (Re)start the session whenever its settings change.
  useEffect(() => {
    let alive = true;
    setError(null);
    pausedRef.current = false;
    setPausedState(false);
    frame.current = null;
    lastT.current = 0;
    trails.current.forEach((t) => t?.clear());
    sim
      .startLive({
        scenario,
        drones,
        spacing: 1.0,
        gust,
        controller: controllers.file(controller),
        others: controllers.file(wingman),
      })
      .catch((e: Error) => alive && setError(e.message));
    return () => {
      alive = false;
      sim.stopLive();
    };
  }, [scenario, drones, gust, controller, wingman, restarts]);

  useEffect(() => {
    const offFrame = sim.onFrame((f) => {
      if (f.t < lastT.current) trails.current.forEach((t) => t?.clear()); // reset with R
      lastT.current = f.t;
      f.drones.forEach((d, i) => trails.current[i]?.push(d.pos[0], d.pos[1], d.pos[2]));
      frame.current = f;
      hud.current?.update(f);
      const isCrashed = f.drones[0]?.crashed ?? false;
      if (isCrashed !== crashedRef.current) {
        crashedRef.current = isCrashed;
        setCrashed(isCrashed);
      }
    });
    const offError = sim.onLiveError(setError);
    return () => {
      offFrame();
      offError();
    };
  }, []);

  const press = useCallback(
    (key: SimKey, down: boolean) => {
      if (down && pausedRef.current) setPaused(false);
      sim.key(key, down);
      if (key === "r") return;
      setHeld((h) => {
        const next = new Set(h);
        if (down) next.add(key);
        else next.delete(key);
        return next;
      });
    },
    [setPaused],
  );

  useEffect(() => {
    const onDown = (e: KeyboardEvent) => {
      if (isTyping(e.target) || e.metaKey || e.ctrlKey || e.altKey) return;
      if (e.key === "Escape") {
        e.preventDefault();
        setPaused(!pausedRef.current);
        return;
      }
      const key = simKey(e);
      if (!key) return;
      e.preventDefault();
      if (!e.repeat) press(key, true);
    };
    const onUp = (e: KeyboardEvent) => {
      const key = simKey(e);
      if (key) press(key, false);
    };
    const releaseAll = () => {
      sim.releaseKeys();
      setHeld(new Set());
    };
    const onVisibility = () => document.hidden && setPaused(true);
    window.addEventListener("keydown", onDown);
    window.addEventListener("keyup", onUp);
    window.addEventListener("blur", releaseAll);
    document.addEventListener("visibilitychange", onVisibility);
    return () => {
      window.removeEventListener("keydown", onDown);
      window.removeEventListener("keyup", onUp);
      window.removeEventListener("blur", releaseAll);
      document.removeEventListener("visibilitychange", onVisibility);
      releaseAll();
    };
  }, [press, setPaused]);

  const blurAfterChange = (e: React.ChangeEvent<HTMLSelectElement>) => e.currentTarget.blur(); // give keys back to the drone

  const droneIndexes = useMemo(() => Array.from({ length: drones }, (_, i) => i), [drones]);

  return (
    <div className="fly">
      <Canvas
        className="fly-canvas"
        dpr={[1, 2]}
        camera={{ fov: 55, near: 0.05, far: 300, up: [0, 0, 1], position: [-4.5, 0, 1.8] }}
      >
        <World dark={dark} />
        <ChaseCamera getFrame={() => frame.current} />
        {droneIndexes.map((i) => (
          <group key={i}>
            <DroneModel
              params={info.params}
              body={i === 0 ? HIVIS : droneColor(i, dark)}
              getPose={() => frame.current?.drones[i] ?? null}
            />
            <GroundShadow dark={dark} getPos={() => frame.current?.drones[i]?.pos ?? null} />
            <Trail
              ref={(t) => void (trails.current[i] = t)}
              capacity={TRAIL_POINTS}
              color={droneColor(i, dark)}
              opacity={i === 0 ? 0.9 : 0.6}
              width={i === 0 ? 2.5 : 2}
            />
          </group>
        ))}
        <TargetMarker getTarget={() => frame.current?.drones[0]?.target ?? null} />
      </Canvas>

      <Hud ref={hud} />

      <form className="fly-settings panel" onSubmit={(e) => e.preventDefault()}>
        <label>
          <span>Flight controller</span>
          <span className="row">
            <select
              value={controller}
              onChange={(e) => {
                onControllerChange(e.target.value);
                blurAfterChange(e);
              }}
            >
              {list.map((c) => (
                <option key={c.name} value={c.name}>
                  {c.name}
                  {c.edited ? " (edited)" : ""}
                </option>
              ))}
            </select>
            <button type="button" className="quiet" onClick={() => onEditController(controller)}>
              Edit
            </button>
          </span>
        </label>
        <label>
          <span>Target path</span>
          <select
            value={scenario}
            onChange={(e) => {
              setScenario(e.target.value);
              blurAfterChange(e);
            }}
          >
            {info.scenarios.map((s) => (
              <option key={s} value={s}>
                {scenarioLabel(s)}
              </option>
            ))}
          </select>
        </label>
        <div className="split">
          <label>
            <span>Drones</span>
            <select
              value={drones}
              onChange={(e) => {
                setDrones(Number(e.target.value));
                blurAfterChange(e);
              }}
            >
              {/* Each drone costs ~15% of a core in Pyodide; four stays real-time on modest laptops. */}
              {[1, 2, 3, 4].map((n) => (
                <option key={n} value={n}>
                  {n}
                </option>
              ))}
            </select>
          </label>
          <label>
            <span>Wind</span>
            <select
              value={gust}
              onChange={(e) => {
                setGust(Number(e.target.value));
                blurAfterChange(e);
              }}
            >
              {WIND.map((w) => (
                <option key={w.gust} value={w.gust}>
                  {w.label}
                </option>
              ))}
            </select>
          </label>
        </div>
        {drones > 1 && <p className="hint">You fly the yellow drone. The others fly {wingman} along the same path, 1 m apart.</p>}
        <button type="button" onClick={() => setRestarts((n) => n + 1)}>
          Restart
        </button>
      </form>

      <div className="fly-help">
        Hold <kbd>↑</kbd> to take off. <kbd>W</kbd> <kbd>A</kbd> <kbd>S</kbd> <kbd>D</kbd> move, <kbd>←</kbd> <kbd>→</kbd> turn.
        <br />
        <kbd>R</kbd> resets, <kbd>Esc</kbd> pauses.
      </div>

      <Keycaps held={held} onPress={press} />

      {crashed && !error && (
        <div className="fly-banner">
          <strong>Crashed.</strong> Two drones touched and lost their motors. Press <kbd>R</kbd> to reset.
        </div>
      )}

      {paused && !error && (
        <button type="button" className="fly-paused" onClick={() => setPaused(false)}>
          <strong>Paused</strong>
          <span>Press Esc or a flight key to keep flying</span>
        </button>
      )}

      {error && (
        <div className="fly-error panel" role="alert">
          <h2>{controller} stopped with an error</h2>
          <pre>{error}</pre>
          <div className="row">
            <button type="button" onClick={() => onEditController(controller)}>
              Edit {controller}
            </button>
            <button type="button" className="quiet" onClick={() => setRestarts((n) => n + 1)}>
              Try again
            </button>
          </div>
        </div>
      )}
    </div>
  );
}

// --- Camera: follows behind the drone, turning with its heading -------------------

function ChaseCamera({ getFrame }: { getFrame: () => LiveFrame | null }) {
  const camera = useThree((s) => s.camera);
  const look = useMemo(() => new THREE.Vector3(0, 0, 0.4), []);
  const want = useMemo(() => new THREE.Vector3(), []);
  const lookWant = useMemo(() => new THREE.Vector3(), []);
  useFrame((_, dt) => {
    const d = getFrame()?.drones[0];
    if (!d) return;
    const [x, y, z] = d.pos;
    const yaw = d.euler[2];
    want.set(x - 4.2 * Math.cos(yaw), y - 4.2 * Math.sin(yaw), Math.max(z + 1.7, 0.6));
    lookWant.set(x, y, z + 0.35);
    camera.position.lerp(want, 1 - Math.exp(-dt * 3.5));
    look.lerp(lookWant, 1 - Math.exp(-dt * 10));
    camera.lookAt(look);
  });
  return null;
}

// --- Heads-up display -------------------------------------------------------------

interface HudHandle {
  update(f: LiveFrame): void;
}

const Hud = forwardRef<HudHandle>(function Hud(_, ref) {
  const els = useRef<Record<string, HTMLElement | null>>({});
  const set = (key: string) => (el: HTMLElement | null) => void (els.current[key] = el);
  useImperativeHandle(ref, () => ({
    update(f) {
      const d = f.drones[0];
      if (!d) return;
      const e = els.current;
      const speed = Math.hypot(d.vel[0], d.vel[1], d.vel[2]);
      const deg = (r: number) => (r * 180) / Math.PI;
      const heading = (deg(d.euler[2]) + 360) % 360;
      const dist = Math.hypot(d.target[0] - d.pos[0], d.target[1] - d.pos[1], d.target[2] - d.pos[2]);
      if (e.alt) e.alt.textContent = d.pos[2].toFixed(2);
      if (e.speed) e.speed.textContent = speed.toFixed(1);
      if (e.heading) e.heading.textContent = heading.toFixed(0).padStart(3, "0");
      if (e.attitude)
        e.attitude.textContent = `Roll ${fmtSigned(deg(d.euler[0]))}°  Pitch ${fmtSigned(deg(d.euler[1]))}°`;
      if (e.target) {
        e.target.textContent = `${dist.toFixed(2)} m to the X`;
        e.target.parentElement?.classList.toggle("on-target", dist < 0.3);
      }
      if (e.time) e.time.textContent = `${f.t.toFixed(1)} s`;
    },
  }));
  return (
    <div className="hud" aria-live="off">
      <div className="hud-readouts">
        <div>
          <span className="hud-value" ref={set("alt")}>0.00</span>
          <span className="hud-unit">m altitude</span>
        </div>
        <div>
          <span className="hud-value" ref={set("speed")}>0.0</span>
          <span className="hud-unit">m/s</span>
        </div>
        <div>
          <span className="hud-value" ref={set("heading")}>000</span>
          <span className="hud-unit">° heading</span>
        </div>
      </div>
      <div className="hud-line">
        <span className="hud-target">
          <svg viewBox="0 0 10 10" aria-hidden="true">
            <path d="M2 2l6 6M8 2L2 8" />
          </svg>
          <span ref={set("target")}>–</span>
        </span>
      </div>
      <div className="hud-line hud-dim">
        <span ref={set("attitude")} />
        <span ref={set("time")} />
      </div>
    </div>
  );
});

const fmtSigned = (v: number) => `${v >= 0 ? "+" : "−"}${Math.abs(v).toFixed(0)}`;

// --- On-screen keys: show what's held, and work as touch controls ------------------

const CLUSTERS: { label: string; keys: { key: SimKey; glyph: string; area: string }[] }[] = [
  {
    label: "Move",
    keys: [
      { key: "w", glyph: "W", area: "up" },
      { key: "a", glyph: "A", area: "left" },
      { key: "s", glyph: "S", area: "down" },
      { key: "d", glyph: "D", area: "right" },
    ],
  },
  {
    label: "Climb and turn",
    keys: [
      { key: "up", glyph: "↑", area: "up" },
      { key: "left", glyph: "←", area: "left" },
      { key: "down", glyph: "↓", area: "down" },
      { key: "right", glyph: "→", area: "right" },
    ],
  },
];

function Keycaps({ held, onPress }: { held: ReadonlySet<SimKey>; onPress: (k: SimKey, down: boolean) => void }) {
  return (
    <div className="keycaps">
      {CLUSTERS.map((c) => (
        <div key={c.label} className="keycap-cluster" role="group" aria-label={c.label}>
          {c.keys.map(({ key, glyph, area }) => (
            <button
              key={key}
              type="button"
              className={`keycap${held.has(key) ? " held" : ""}`}
              style={{ gridArea: area }}
              aria-label={`${glyph} key`}
              onPointerDown={(e) => {
                e.currentTarget.setPointerCapture(e.pointerId);
                onPress(key, true);
              }}
              onPointerUp={() => onPress(key, false)}
              onPointerCancel={() => onPress(key, false)}
              onContextMenu={(e) => e.preventDefault()}
            >
              {glyph}
            </button>
          ))}
        </div>
      ))}
    </div>
  );
}
