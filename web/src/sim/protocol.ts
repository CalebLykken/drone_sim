// Shapes exchanged with the Python bridge (src/python/bridge.py) through the
// worker. Flat arrays are row-major: pos is [x0, y0, z0, x1, y1, z1, ...].

export interface ControllerFile {
  name: string; // e.g. "pid.py"
  source: string;
}

export interface SimOptions {
  scenario: string;
  duration?: number;
  wind?: [number, number, number];
  gust?: number;
  noise?: number;
  seed?: number;
  drones?: number;
  spacing?: number;
  ghost?: boolean;
  controller: ControllerFile; // the first drone (the one you pilot in live mode)
  others?: ControllerFile | null; // every other drone; defaults to `controller`
}

export interface DroneParams {
  mass: number;
  arm_length: number;
  max_thrust: number;
  radius: number;
  yaw_signs: number[];
  rotor_positions: number[][];
}

export interface SimInfo {
  python: string;
  numpy: string;
  backend?: "local"; // set by the local Python server; absent in the browser
  scenarios: string[];
  params: DroneParams;
}

export interface Collision {
  t: number;
  a: string;
  b: string;
  distance: number;
}

export type Metrics = Record<string, number | boolean | null>;

export interface DroneTrack {
  name: string;
  pos: Float64Array; // N x 3
  quat: Float64Array; // N x 4, (w, x, y, z) body -> world
  euler: Float64Array; // N x 3, roll pitch yaw [rad]
  target: Float64Array; // N x 4, x y z yaw
  thrust: Float64Array; // N x 4, FL FR BR BL [N]
  metrics: Metrics;
  crashed_at: number | null;
}

export interface PlaybackResult {
  scenario: string;
  dt: number;
  t: Float64Array;
  wind: Float64Array; // N x 3
  params: DroneParams;
  collisions: Collision[];
  drones: DroneTrack[];
  options: SimOptions;
}

export interface LiveDrone {
  name: string;
  pos: number[];
  quat: number[];
  vel: number[];
  euler: number[];
  target: number[];
  thrust: number[];
  crashed: boolean;
}

export interface LiveFrame {
  t: number;
  sticks: { throttle: number; roll: number; pitch: number; yaw: number };
  wind: number[];
  collisions: Collision[];
  drones: LiveDrone[];
}

export interface CheckResult {
  class: string;
  metrics: Metrics;
}

// --- Stacks tab (drone_sim/editor/api.py) ---

export type StackParamType = "number" | "integer" | "bool" | "string" | "vector" | "mapping" | "controller" | "path";

/** One constructor argument of a node, as an editable field. */
export interface StackParam {
  name: string;
  type: StackParamType;
  value: unknown; // current value (null: the default)
  default: unknown;
  required: boolean;
  choices?: string[]; // controller files, for type "controller"
  length?: number; // for type "vector"
}

export interface StackNode {
  name: string;
  key: string; // node type, e.g. "drone_sim.nodes.Imu"
  class: string;
  kind: string; // sensor, estimator, controller, ...
  doc: string;
  params: StackParam[];
}

export interface CatalogEntry {
  key: string;
  class: string;
  kind: string;
  doc: string;
  local: boolean; // defined in the stack file itself
  params: StackParam[];
}

export interface GraphVertex {
  id: string;
  type: "node" | "topic";
  name: string;
  label: string;
  sub: string;
  x: number; // layout units, y up
  y: number;
  kind?: string;
  color?: string;
  editable?: boolean;
  msg_type?: string;
  rate?: number | null;
  publishers?: string[];
  subscribers?: string[];
}

export interface GraphEdge {
  points: [number, number][];
  feedback: boolean;
}

export interface StackState {
  ok: true;
  nodes: StackNode[];
  graph: { vertices: GraphVertex[]; edges: GraphEdge[] };
  warnings: string[];
  source: string; // the stack file after the latest change
  flight: { scenario: string; duration: number; metrics: Metrics } | null;
  selected?: string; // node added or renamed by the last edit
  catalog?: CatalogEntry[]; // sent when a stack is opened
}

export type StackReply = StackState | { ok: false; error: string };

export type StackEdit =
  | { op: "add"; key: string; params: Record<string, unknown> }
  | { op: "update"; name: string; params: Record<string, unknown> }
  | { op: "remove"; name: string }
  | { op: "test"; scenario: string; duration: number };

// Live-session key names follow drone_sim/session.py.
export type SimKey = "w" | "a" | "s" | "d" | "up" | "down" | "left" | "right" | "r";

export type ToWorker =
  | { id: number; type: "playback"; opts: SimOptions }
  | { id: number; type: "check"; controller: ControllerFile }
  | { id: number; type: "live:start"; opts: SimOptions }
  | { id: number; type: "stack:open"; name: string; source: string; controllers: ControllerFile[] }
  | { id: number; type: "stack:edit"; edit: StackEdit }
  | { type: "cancel"; id: number }
  | { type: "live:key"; key: SimKey; down: boolean }
  | { type: "live:release" }
  | { type: "live:reset" }
  | { type: "live:pause"; paused: boolean }
  | { type: "live:stop" };

export type FromWorker =
  | { type: "status"; text: string }
  | { type: "ready"; info: SimInfo }
  | { type: "fatal"; error: string }
  | { type: "reply"; id: number; ok: true; value: unknown }
  | { type: "reply"; id: number; ok: false; error: string }
  | { type: "progress"; id: number; fraction: number }
  | { type: "live:frame"; frame: LiveFrame }
  | { type: "live:error"; error: string }
  | { type: "stdout"; text: string }
  | { type: "notice"; text: string | null }; // e.g. "Loading cvxpy" while an optional package downloads
