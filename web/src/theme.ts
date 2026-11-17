// Colors that live in JavaScript (three.js materials, chart series). Page
// colors are CSS custom properties in styles.css.

import type { TimeOfDay } from "./store";

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

/** How the 3D scene looks at one time of day. Independent of the page's light
 * or dark theme: the sky is drawn either way. */
export interface SceneLook {
  zenith: string; // sky overhead
  horizon: string; // sky at the horizon, and the fog, so far-off ground melts into it
  sunGlow: string; // haze around the sun
  sunDisk: string;
  sunDirection: [number, number, number]; // toward the sun, in the sim's frame (x forward, z up)
  sunLight: string;
  sunIntensity: number;
  skyLight: string; // hemisphere light from above
  groundLight: string; // and bounced from below
  ambientIntensity: number;
  ground: string; // the grid world's floor
  grid: string;
  gridMajor: string;
  pad: string;
  shadow: string; // spot under each drone
  shadowOpacity: number;
  pathShadow: string; // flight path projected on the ground
}

export const SCENE: Record<TimeOfDay, SceneLook> = {
  day: {
    zenith: "#5f93cf",
    horizon: "#dde5ec",
    sunGlow: "#fff4dc",
    sunDisk: "#fffaf0",
    sunDirection: [5, -7, 12],
    sunLight: "#ffffff",
    sunIntensity: 1.8,
    skyLight: "#f3f7fb",
    groundLight: "#9aa596",
    ambientIntensity: 1.4,
    ground: "#c9d1c6",
    grid: "#b4beb2",
    gridMajor: "#8f9b8d",
    pad: "#f7f9fa",
    shadow: "#27313a",
    shadowOpacity: 0.28,
    pathShadow: "#56606a",
  },
  // Sun low, ahead and to the left of where the Fly camera starts (looking
  // along +x), so you take off toward the sunset with the trees lit from the side.
  evening: {
    zenith: "#2f4270",
    horizon: "#d9a68f",
    sunGlow: "#ffa057",
    sunDisk: "#fff1cf",
    sunDirection: [1, 0.7, 0.12],
    sunLight: "#ffb578",
    sunIntensity: 3.0,
    skyLight: "#c4c0e2",
    groundLight: "#86705a",
    ambientIntensity: 2.2,
    ground: "#8c8577",
    grid: "#7d766a",
    gridMajor: "#655f55",
    pad: "#f1e2cf",
    shadow: "#1d1410",
    shadowOpacity: 0.35,
    pathShadow: "#3a2e29",
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
