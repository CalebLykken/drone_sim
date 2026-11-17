import { OrbitControls } from "@react-three/drei";
import { Canvas, useThree } from "@react-three/fiber";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import type { OrbitControls as OrbitControlsImpl } from "three-stdlib";
import { type ChartSeries, TimeChart } from "../components/TimeChart";
import { controllers, useControllerList } from "../controllers";
import { sim } from "../sim/client";
import type { PlaybackResult, SimInfo, SimOptions } from "../sim/protocol";
import { createStore, type Store, useDarkMode, useStore } from "../store";
import { droneColor, HIVIS, scenarioLabel, seriesColors } from "../theme";
import { DroneModel, GroundShadow, PathLine, TargetMarker, TargetPath, World } from "../three/scene";

interface Form {
  scenario: string;
  duration: number;
  drones: number;
  spacing: number;
  ghost: boolean;
  windX: number;
  windY: number;
  gust: number;
  noise: number;
  seed: number;
}

const DEFAULT_FORM: Form = {
  scenario: "helix",
  duration: 12,
  drones: 1,
  spacing: 1,
  ghost: false,
  windX: 0,
  windY: 0,
  gust: 0,
  noise: 0,
  seed: 0,
};

const SPEEDS = [0.25, 0.5, 1, 2, 4];

function formFrom(opts: SimOptions | undefined): Form {
  if (!opts) return DEFAULT_FORM;
  return {
    scenario: opts.scenario,
    duration: opts.duration ?? 12,
    drones: opts.drones ?? 1,
    spacing: opts.spacing ?? 1,
    ghost: opts.ghost ?? false,
    windX: opts.wind?.[0] ?? 0,
    windY: opts.wind?.[1] ?? 0,
    gust: opts.gust ?? 0,
    noise: opts.noise ?? 0,
    seed: opts.seed ?? 0,
  };
}

export function ScenarioView({
  info,
  controller,
  onControllerChange,
  onEditController,
  result,
  onResult,
  autoRun,
  onAutoRun,
}: {
  info: SimInfo;
  controller: string;
  onControllerChange: (name: string) => void;
  onEditController: (name: string) => void;
  result: PlaybackResult | null;
  onResult: (r: PlaybackResult) => void;
  autoRun: boolean; // set by "Run a scenario" in the Controllers tab
  onAutoRun: () => void;
}) {
  const list = useControllerList();
  const [form, setForm] = useState<Form>(() => formFrom(result?.options));
  const [progress, setProgress] = useState<{ fraction: number; cancel: () => void } | null>(null);
  const [error, setError] = useState<string | null>(null);
  const clock = useMemo(() => createStore(0), []);
  const [playing, setPlaying] = useState(false);
  const [speed, setSpeed] = useState(1);
  const [selected, setSelected] = useState(0);

  const update = <K extends keyof Form>(key: K, value: Form[K]) => setForm((f) => ({ ...f, [key]: value }));

  const run = useCallback(() => {
    setError(null);
    setPlaying(false);
    const opts: SimOptions = {
      scenario: form.scenario,
      duration: form.duration,
      drones: form.drones,
      spacing: form.spacing,
      ghost: form.ghost,
      wind: [form.windX, form.windY, 0],
      gust: form.gust,
      noise: form.noise,
      seed: form.seed,
      controller: controllers.file(controller),
    };
    const job = sim.runPlayback(opts, (fraction) => setProgress((p) => (p ? { ...p, fraction } : p)));
    setProgress({ fraction: 0, cancel: job.cancel });
    job.promise
      .then((r) => {
        onResult(r);
        setSelected(0);
        clock.set(0);
        setPlaying(true);
      })
      .catch((e: Error) => e.message !== "Cancelled" && setError(e.message))
      .finally(() => setProgress(null));
  }, [form, controller, onResult, clock]);

  // Run on mount if there's nothing to show yet, or the Controllers tab asked
  // for a run. (The ref keeps React's dev-mode double effects from running twice.)
  const mounted = useRef(false);
  useEffect(() => {
    if (mounted.current) return;
    mounted.current = true;
    if (autoRun || !result) {
      run();
      onAutoRun();
    }
  }, []); // mount only

  // Playback loop.
  useEffect(() => {
    if (!playing || !result) return;
    const end = result.t[result.t.length - 1];
    if (clock.get() >= end) clock.set(0);
    let last = performance.now();
    let raf = requestAnimationFrame(function tick(now) {
      const next = clock.get() + ((now - last) / 1000) * speed;
      last = now;
      if (next >= end) {
        clock.set(end);
        setPlaying(false);
        return;
      }
      clock.set(next);
      raf = requestAnimationFrame(tick);
    });
    return () => cancelAnimationFrame(raf);
  }, [playing, speed, result, clock]);

  // Space toggles playback.
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const el = e.target as HTMLElement;
      if (e.key !== " " || ["INPUT", "SELECT", "TEXTAREA", "BUTTON"].includes(el.tagName)) return;
      e.preventDefault();
      setPlaying((p) => !p);
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, []);

  const seek = useCallback(
    (t: number) => {
      setPlaying(false);
      if (result) clock.set(Math.min(Math.max(t, 0), result.t[result.t.length - 1]));
    },
    [clock, result],
  );

  return (
    <div className="scenarios">
      <form
        className="scenario-form panel"
        onSubmit={(e) => {
          e.preventDefault();
          if (!progress) run();
        }}
      >
        <label>
          <span>Flight controller</span>
          <span className="row">
            <select value={controller} onChange={(e) => onControllerChange(e.target.value)}>
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
          <span>Scenario</span>
          <select value={form.scenario} onChange={(e) => update("scenario", e.target.value)}>
            {info.scenarios.map((s) => (
              <option key={s} value={s}>
                {scenarioLabel(s)}
              </option>
            ))}
          </select>
        </label>
        <div className="split">
          <NumberField label="Duration" unit="s" value={form.duration} min={1} max={60} step={1} onChange={(v) => update("duration", v)} />
          <NumberField label="Drones" value={form.drones} min={1} max={6} step={1} onChange={(v) => update("drones", v)} />
        </div>
        {form.drones > 1 && (
          <>
            <NumberField label="Spacing" unit="m" value={form.spacing} min={0.2} max={3} step={0.1} onChange={(v) => update("spacing", v)} />
            <label className="check">
              <input type="checkbox" checked={form.ghost} onChange={(e) => update("ghost", e.target.checked)} />
              <span>Drones pass through each other</span>
            </label>
          </>
        )}
        <fieldset>
          <legend>Disturbances</legend>
          <div className="split">
            <NumberField label="Wind east" unit="N" value={form.windX} min={-3} max={3} step={0.1} onChange={(v) => update("windX", v)} />
            <NumberField label="Wind north" unit="N" value={form.windY} min={-3} max={3} step={0.1} onChange={(v) => update("windY", v)} />
          </div>
          <div className="split">
            <NumberField label="Gusts" unit="N" value={form.gust} min={0} max={3} step={0.1} onChange={(v) => update("gust", v)} />
            <NumberField label="Sensor noise" value={form.noise} min={0} max={0.2} step={0.005} onChange={(v) => update("noise", v)} />
          </div>
          <NumberField label="Random seed" value={form.seed} min={0} max={9999} step={1} onChange={(v) => update("seed", v)} />
        </fieldset>
        {progress ? (
          <div className="progress">
            <div className="progress-bar">
              <div style={{ width: `${(progress.fraction * 100).toFixed(1)}%` }} />
            </div>
            <button type="button" className="quiet" onClick={progress.cancel}>
              Cancel
            </button>
          </div>
        ) : (
          <button type="submit" className="primary">
            Run scenario
          </button>
        )}
        {error && (
          <div className="form-error" role="alert">
            <p>The run stopped with a Python error.</p>
            <pre>{error}</pre>
            <button type="button" className="quiet" onClick={() => onEditController(controller)}>
              Edit {controller}
            </button>
          </div>
        )}
      </form>

      <section className="stage">
        {result ? (
          <>
            <PlaybackScene result={result} clock={clock} />
            <Transport result={result} clock={clock} playing={playing} setPlaying={setPlaying} speed={speed} setSpeed={setSpeed} onSeek={seek} />
            <Metrics result={result} />
          </>
        ) : (
          <div className="stage-empty">{progress ? "Simulating…" : "Run a scenario to see it fly."}</div>
        )}
      </section>

      <section className="charts" aria-label="Flight data">
        {result && (
          <>
            {result.drones.length > 1 && (
              <label className="chart-drone">
                <span>Plots for</span>
                <select value={selected} onChange={(e) => setSelected(Number(e.target.value))}>
                  {result.drones.map((d, i) => (
                    <option key={d.name} value={i}>
                      {d.name}
                    </option>
                  ))}
                </select>
              </label>
            )}
            <Charts result={result} drone={Math.min(selected, result.drones.length - 1)} clock={clock} onSeek={seek} />
          </>
        )}
      </section>
    </div>
  );
}

// --- Form field -------------------------------------------------------------------

function NumberField({
  label,
  unit,
  value,
  min,
  max,
  step,
  onChange,
}: {
  label: string;
  unit?: string;
  value: number;
  min: number;
  max: number;
  step: number;
  onChange: (v: number) => void;
}) {
  const [text, setText] = useState(String(value));
  useEffect(() => setText(String(value)), [value]);
  return (
    <label>
      <span>{label}</span>
      <span className="number">
        <input
          type="number"
          inputMode="decimal"
          value={text}
          min={min}
          max={max}
          step={step}
          onChange={(e) => {
            setText(e.target.value);
            const v = Number(e.target.value);
            if (e.target.value !== "" && Number.isFinite(v)) onChange(Math.min(max, Math.max(min, v)));
          }}
          onBlur={() => setText(String(value))}
        />
        {unit && <span className="unit">{unit}</span>}
      </span>
    </label>
  );
}

// --- 3D playback -------------------------------------------------------------------

function indexAt(result: PlaybackResult, t: number) {
  return Math.min(result.t.length - 1, Math.max(0, Math.round(t / result.dt)));
}

function PlaybackScene({ result, clock }: { result: PlaybackResult; clock: Store<number> }) {
  const dark = useDarkMode();
  const at = () => indexAt(result, clock.get());
  const slice = (a: Float64Array, stride: number) => {
    const k = at();
    return a.subarray(k * stride, k * stride + stride);
  };
  return (
    <div className="stage-canvas">
      <Canvas dpr={[1, 2]} camera={{ fov: 45, near: 0.05, far: 400, up: [0, 0, 1], position: [6, -8, 5] }}>
        <World dark={dark} />
        <OrbitControls makeDefault enableDamping dampingFactor={0.12} maxPolarAngle={Math.PI * 0.495} />
        <FitCamera result={result} />
        {result.drones.map((d, i) => (
          <group key={d.name}>
            <TargetPath target={d.target} />
            <TargetMarker getTarget={() => slice(d.target, 4)} />
            <PathLine points={d.pos} getCount={() => at() + 1} color={droneColor(i, dark)} />
            <PathLine points={d.pos} getCount={() => at() + 1} color={dark ? "#000000" : "#56606a"} opacity={0.3} width={1.5} flat />
            <GroundShadow dark={dark} getPos={() => slice(d.pos, 3)} />
            <DroneModel
              params={result.params}
              body={i === 0 ? HIVIS : droneColor(i, dark)}
              getPose={() => ({ pos: slice(d.pos, 3), quat: slice(d.quat, 4), thrust: slice(d.thrust, 4) })}
            />
          </group>
        ))}
      </Canvas>
      <StageInfo result={result} clock={clock} />
    </div>
  );
}

/** Frame the whole flight when a new result arrives. */
function FitCamera({ result }: { result: PlaybackResult }) {
  const camera = useThree((s) => s.camera);
  const controls = useThree((s) => s.controls) as OrbitControlsImpl | null;
  useEffect(() => {
    const lo = [Infinity, Infinity, 0];
    const hi = [-Infinity, -Infinity, -Infinity];
    for (const d of result.drones) {
      for (const arr of [d.pos, d.target]) {
        const stride = arr === d.pos ? 3 : 4;
        for (let i = 0; i < arr.length; i += stride) {
          for (let j = 0; j < 3; j++) {
            lo[j] = Math.min(lo[j], arr[i + j]);
            hi[j] = Math.max(hi[j], arr[i + j]);
          }
        }
      }
    }
    const center = lo.map((v, j) => (v + hi[j]) / 2);
    const size = Math.max(hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2], 2);
    const dist = size * 1.5 + 2;
    camera.position.set(center[0] + dist * 0.55, center[1] - dist * 0.8, center[2] + dist * 0.55);
    if (controls) {
      controls.target.set(center[0], center[1], center[2]);
      controls.update();
    } else {
      camera.lookAt(center[0], center[1], center[2]);
    }
  }, [result, camera, controls]);
  return null;
}

function StageInfo({ result, clock }: { result: PlaybackResult; clock: Store<number> }) {
  const t = useStore(clock);
  const k = indexAt(result, t);
  const wx = result.wind[k * 3];
  const wy = result.wind[k * 3 + 1];
  const windy = result.wind.some((w) => w !== 0);
  return (
    <div className="stage-info">
      <span>{scenarioLabel(result.scenario)}</span>
      {windy && (
        <span>
          Wind {wx.toFixed(2)}, {wy.toFixed(2)} N
        </span>
      )}
      <span className="stage-hint">Drag to orbit, scroll to zoom</span>
    </div>
  );
}

// --- Transport bar -------------------------------------------------------------------

function Transport({
  result,
  clock,
  playing,
  setPlaying,
  speed,
  setSpeed,
  onSeek,
}: {
  result: PlaybackResult;
  clock: Store<number>;
  playing: boolean;
  setPlaying: (p: boolean) => void;
  speed: number;
  setSpeed: (s: number) => void;
  onSeek: (t: number) => void;
}) {
  const t = useStore(clock);
  const end = result.t[result.t.length - 1];
  return (
    <div className="transport">
      <button type="button" className="play" onClick={() => setPlaying(!playing)} aria-label={playing ? "Pause" : "Play"}>
        {playing ? (
          <svg viewBox="0 0 16 16" aria-hidden="true">
            <rect x="3" y="2" width="3.5" height="12" rx="1" />
            <rect x="9.5" y="2" width="3.5" height="12" rx="1" />
          </svg>
        ) : (
          <svg viewBox="0 0 16 16" aria-hidden="true">
            <path d="M4 2.5v11a.8.8 0 0 0 1.2.7l9-5.5a.8.8 0 0 0 0-1.4l-9-5.5A.8.8 0 0 0 4 2.5z" />
          </svg>
        )}
      </button>
      <input
        type="range"
        className="scrub"
        min={0}
        max={end}
        step={result.dt}
        value={t}
        aria-label="Time"
        onChange={(e) => onSeek(Number(e.target.value))}
      />
      <span className="time">
        {t.toFixed(2)} <span className="of">/ {end.toFixed(2)} s</span>
      </span>
      <select value={speed} onChange={(e) => setSpeed(Number(e.target.value))} aria-label="Playback speed">
        {SPEEDS.map((s) => (
          <option key={s} value={s}>
            {s}×
          </option>
        ))}
      </select>
    </div>
  );
}

// --- Metrics -------------------------------------------------------------------------

function Metrics({ result }: { result: PlaybackResult }) {
  const num = (v: unknown, digits = 2) => (typeof v === "number" ? v.toFixed(digits) : "–");
  return (
    <div className="metrics">
      <table>
        <thead>
          <tr>
            <th scope="col">Drone</th>
            <th scope="col">RMS error</th>
            <th scope="col">Final error</th>
            <th scope="col">Max tilt</th>
            <th scope="col">Outcome</th>
          </tr>
        </thead>
        <tbody>
          {result.drones.map((d) => {
            const m = d.metrics;
            const outcome = d.crashed_at != null ? `Crashed at ${d.crashed_at.toFixed(2)} s` : m.flipped ? "Flipped" : "Stayed upright";
            const bad = d.crashed_at != null || Boolean(m.flipped);
            return (
              <tr key={d.name}>
                <th scope="row">{d.name}</th>
                <td>{num(m.rms_pos_error_m)} m</td>
                <td>{num(m.final_pos_error_m)} m</td>
                <td>{num(m.max_tilt_deg, 1)}°</td>
                <td className={bad ? "bad" : "good"}>
                  <span className="status-icon" aria-hidden="true">
                    {bad ? "!" : "✓"}
                  </span>
                  {outcome}
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
      {result.collisions.length > 0 && (
        <ul className="collisions">
          {result.collisions.map((c) => (
            <li key={`${c.a}-${c.b}`}>
              {c.a} and {c.b} touched at {c.t.toFixed(2)} s ({c.distance.toFixed(2)} m apart)
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}

// --- Charts -------------------------------------------------------------------------

function column(a: Float64Array, stride: number, j: number, scale = 1): Float64Array {
  const n = a.length / stride;
  const out = new Float64Array(n);
  for (let i = 0; i < n; i++) out[i] = a[i * stride + j] * scale;
  return out;
}

function unwrapDegrees(rad: Float64Array): Float64Array {
  const out = new Float64Array(rad.length);
  let offset = 0;
  for (let i = 0; i < rad.length; i++) {
    if (i > 0) {
      const jump = rad[i] - rad[i - 1];
      if (jump > Math.PI) offset -= 2 * Math.PI;
      else if (jump < -Math.PI) offset += 2 * Math.PI;
    }
    out[i] = ((rad[i] + offset) * 180) / Math.PI;
  }
  return out;
}

function Charts({
  result,
  drone,
  clock,
  onSeek,
}: {
  result: PlaybackResult;
  drone: number;
  clock: Store<number>;
  onSeek: (t: number) => void;
}) {
  const dark = useDarkMode();
  const c = seriesColors(dark);
  const d = result.drones[drone];
  const charts = useMemo(() => {
    const deg = 180 / Math.PI;
    const n = result.t.length;
    const err = new Float64Array(n);
    for (let i = 0; i < n; i++) {
      err[i] = Math.hypot(
        d.pos[i * 3] - d.target[i * 4],
        d.pos[i * 3 + 1] - d.target[i * 4 + 1],
        d.pos[i * 3 + 2] - d.target[i * 4 + 2],
      );
    }
    const dashed = [6, 4];
    const position: ChartSeries[] = [
      { label: "x", data: column(d.pos, 3, 0), color: c[0] },
      { label: "y", data: column(d.pos, 3, 1), color: c[1] },
      { label: "z", data: column(d.pos, 3, 2), color: c[2] },
      { label: "x target", data: column(d.target, 4, 0), color: c[0], dash: dashed, endLabel: false },
      { label: "y target", data: column(d.target, 4, 1), color: c[1], dash: dashed, endLabel: false },
      { label: "z target", data: column(d.target, 4, 2), color: c[2], dash: dashed, endLabel: false },
    ];
    const error: ChartSeries[] = [{ label: "Distance", data: err, color: c[0], endLabel: false }];
    const attitude: ChartSeries[] = [
      { label: "roll", data: column(d.euler, 3, 0, deg), color: c[0] },
      { label: "pitch", data: column(d.euler, 3, 1, deg), color: c[1] },
      { label: "yaw", data: unwrapDegrees(column(d.euler, 3, 2)), color: c[2] },
      { label: "yaw target", data: unwrapDegrees(column(d.target, 4, 3)), color: c[2], dash: [2, 3], endLabel: false },
    ];
    const thrust: ChartSeries[] = ["FL", "FR", "BR", "BL"].map((label, j) => ({
      label,
      data: column(d.thrust, 4, j),
      color: c[j],
    }));
    return { position, error, attitude, thrust };
  }, [d, result.t, c]);

  const thrustRange = useMemo<[number, number]>(() => [0, result.params.max_thrust * 1.08], [result.params.max_thrust]);
  const thrustRef = useMemo(() => ({ value: result.params.max_thrust, label: `Max ${result.params.max_thrust} N` }), [result.params.max_thrust]);

  return (
    <>
      <TimeChart title="Position" unit="m (dashed: target)" t={result.t} series={charts.position} clock={clock} onSeek={onSeek} height={150} />
      <TimeChart title="Distance to target" unit="m" t={result.t} series={charts.error} clock={clock} onSeek={onSeek} />
      <TimeChart title="Attitude" unit="degrees (dotted: yaw target)" t={result.t} series={charts.attitude} clock={clock} onSeek={onSeek} height={150} />
      <TimeChart
        title="Rotor thrust"
        unit="N"
        t={result.t}
        series={charts.thrust}
        clock={clock}
        onSeek={onSeek}
        yRange={thrustRange}
        refLine={thrustRef}
        height={150}
      />
    </>
  );
}
