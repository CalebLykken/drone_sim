import { useProgress } from "@react-three/drei";
import type { Store } from "../store";
import { TIMES, timeStore, useStore, WORLDS, worldStore } from "../store";

/** World and time-of-day pickers for a 3D view's settings. The forest is a 6 MB download the first time. */
export function ScenerySelects({ blurOnChange = false }: { blurOnChange?: boolean }) {
  const world = useStore(worldStore);
  const loading = useProgress((s) => s.active);
  return (
    <>
      <div className="split">
        <Choice label="World" store={worldStore} options={WORLDS} blurOnChange={blurOnChange} />
        <Choice label="Time of day" store={timeStore} options={TIMES} blurOnChange={blurOnChange} />
      </div>
      {world === "forest" && loading && <p className="hint">Loading the forest (6 MB)…</p>}
    </>
  );
}

function Choice<T extends string>({
  label,
  store,
  options,
  blurOnChange,
}: {
  label: string;
  store: Store<T>;
  options: { name: T; label: string }[];
  blurOnChange: boolean;
}) {
  const value = useStore(store);
  return (
    <label>
      <span>{label}</span>
      <select
        value={value}
        onChange={(e) => {
          store.set(e.target.value as T);
          if (blurOnChange) e.currentTarget.blur(); // give keys back to the drone
        }}
      >
        {options.map((o) => (
          <option key={o.name} value={o.name}>
            {o.label}
          </option>
        ))}
      </select>
    </label>
  );
}
