// Colors that live in JavaScript (three.js materials, chart series). Page
// colors are CSS custom properties in styles.css.

export const HIVIS = "#f2b705"; // the piloted drone's body
export const SIGNAL_RED = "#d64545"; // front rotors and the target X

// Categorical chart series, validated for color-vision deficiency on the
// app's light (#f7f9fa) and dark (#1d242c) surfaces. Fixed order; never cycled.
const SERIES_LIGHT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"];
const SERIES_DARK = ["#3987e5", "#d95926", "#199e70", "#c98500"];
const OVERFLOW = "#8a949e"; // drones past the fourth share a neutral gray

export function seriesColors(dark: boolean): string[] {
  return dark ? SERIES_DARK : SERIES_LIGHT;
}

/** Trail/body color for drone i. Drone 0 gets slot 1, and so on. */
export function droneColor(i: number, dark: boolean): string {
  return seriesColors(dark)[i] ?? OVERFLOW;
}

export interface SceneColors {
  sky: string;
  ground: string;
  grid: string;
  gridMajor: string;
  pad: string;
  skyLight: string;
  groundLight: string;
  shadow: string;
}

export const SCENE: Record<"light" | "dark", SceneColors> = {
  light: {
    sky: "#dde5ec",
    ground: "#c9d1c6",
    grid: "#b4beb2",
    gridMajor: "#8f9b8d",
    pad: "#f7f9fa",
    skyLight: "#f3f7fb",
    groundLight: "#9aa596",
    shadow: "#27313a",
  },
  dark: {
    sky: "#141a21",
    ground: "#1b232b",
    grid: "#26313b",
    gridMajor: "#36434f",
    pad: "#2b3540",
    skyLight: "#9fb3c8",
    groundLight: "#1b232b",
    shadow: "#000000",
  },
};

export const SCENARIO_LABELS: Record<string, string> = {
  hover: "Hover",
  step: "Step",
  square: "Square",
  figure8: "Figure 8",
  helix: "Helix",
  spin: "Spin",
};

export const scenarioLabel = (name: string) => SCENARIO_LABELS[name] ?? name;
