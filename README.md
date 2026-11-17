# drone_sim

A small 3D quadrotor simulator for trying out flight controllers. You write a
controller as a Python class, point the sim at the file, and watch it fly.

- Full 3D rigid-body dynamics (quaternion attitude) for an X-configuration quad
- Motor lag and thrust saturation
- Optional steady wind, gusts, and sensor noise
- Rotatable 3D animation with time-series plots, or headless metrics for batch tuning
- Live mode: fly it yourself with the keyboard
- Multiple drones in one simulation, with drone-to-drone collisions
- ROS-style nodes and topics for full stacks (sensors → estimator → controller)
- A prosocial multi-agent planner (PHRI, ICRA 2024)
- A browser version that runs the same Python with no server

**Contents:**
[Quick start](#quick-start) ·
[Multiple drones](#multiple-drones) ·
[Stacks from nodes](#building-a-full-stack-from-nodes) ·
[PHRI](#prosocial-planning-phri) ·
[Flying it yourself](#flying-it-yourself) ·
[Web app](#web-app) ·
[Writing a controller](#writing-your-own-controller) ·
[Python API](#using-it-from-python) ·
[CLI options](#command-line-options) ·
[Tests](#running-the-tests) ·
[Layout](#layout)

## Quick start

Requires Python 3.10+, `numpy`, and `matplotlib`. The PHRI controller also
needs `cvxpy`.

```bash
pip install numpy matplotlib          # add cvxpy for PHRI, pytest for the tests
```

There is nothing to install for the package itself: run it from the repo root.

```bash
python -m drone_sim                                   # example PID controller, step scenario
python -m drone_sim -s helix                          # pick a scenario (see below)
python -m drone_sim -s square --wind 1 0.5 --gust 0.5 # steady wind force (x y [z], N) and gusts
python -m drone_sim --no-gui                          # just print metrics
python -m drone_sim --save flight.gif                 # export the animation (.mp4 needs ffmpeg)
```

Drag the 3D view with the mouse to rotate it. Every run prints the metrics
for each drone (RMS and final position error, max tilt, whether it flipped
or crashed). All the flags are listed under
[Command-line options](#command-line-options).

| Scenario | Target |
|---|---|
| `hover` | hold (0, 0, 1) |
| `step` (default) | hover at 1 m, then jump to (2, 1, 2) at t = 3 s |
| `square` | four corners at different heights, 3 s each |
| `figure8` | a figure eight that rises and falls, 8 s per loop |
| `helix` | climb in a circle while turning to face the direction of travel |
| `spin` | hover, then turn 90° at t = 3 s |

## Multiple drones

```bash
python -m drone_sim -n 3 -s square --no-gui           # 3 drones fly the square side by side
python -m drone_sim -n 3 --spacing 2                  # start/fly 2 m apart (default 1 m)
python -m drone_sim -c controllers/pid.py -c my.py    # one drone per --controller
python -m drone_sim -n 2 --spacing 0.4 --ghost        # let drones pass through each other
python -m drone_sim --live -n 3                       # you fly drone0; the others run pid.py
python -m drone_sim -n 4 --layout circle --radius 3   # 4 drones swap sides of a circle
```

With the default `--layout line`, drones start in a line along y and each
flies the scenario shifted by its starting offset. With `--layout circle`,
they start evenly spaced around a circle (`--radius`, default 4 m), climb to
1.5 m for 2 s, then each flies to the opposite side, ignoring `--scenario`.
It's a hard test for collision avoidance, because every drone heads for the
center at once. Metrics are printed per drone, followed by any collisions.
Every drone is drawn in its own color with a label; crashed drones fade out
and collisions are marked with a red star. In live mode the camera follows
the drone you fly.

All drones share the clock and the wind. Two drones collide when their
centers come closer than the sum of their `DroneParams.radius` (0.25 m by
default); both lose their motors and fall, unless `crash_on_collision=False`
(`--ghost`), in which case the collision is only recorded.

From Python, describe each drone with a `DroneSpec`:

```python
import numpy as np
from drone_sim import DroneParams, DroneSpec, SimConfig, State, load_controller, run_many

specs = [
    DroneSpec(load_controller("controllers/pid.py"), name="leader",
              initial_state=State(pos=np.array([0.0, 0.0, 0.0]))),
    DroneSpec(load_controller("my_follower.py"), name="follower",
              initial_state=State(pos=np.array([-1.0, 0.0, 0.0])),
              params=DroneParams(mass=1.3),                 # drones can differ
              target_fn=lambda t: (-1.0, 0.0, 1.0)),       # default: the scenario
]
results, collisions = run_many(specs, SimConfig(scenario="square", duration=10))
for r in results:
    print(r.name, r.metrics())                           # includes "crashed"
```

Each drone needs its own controller instance. Inside a controller,
`self.name` is the drone's name and `self.others` lists the other drones
(`Neighbor(name, state, crashed)`), so you can write formation-keeping or
collision-avoidance controllers. `Simulator(drones=[...])` steps them one
control period at a time; `sim.drones`, `sim.drone("leader")`,
`sim.results()`, and `sim.collisions` give per-drone access.

## Building a full stack from nodes

Beyond a single controller class, a drone can run a **stack**: ROS-style
nodes that talk over typed topics, e.g. sensors → estimator → controller.

```bash
python -m drone_sim -c stacks/gps_imu_pid.py -s square        # IMU + GPS -> filter -> PID
python -m drone_sim -c stacks/gps_imu_pid.py --graph          # show the node graph (like rqt_graph)
python -m drone_sim -c stacks/gps_imu_pid.py --graph-out stack.png   # or .svg, .dot, .mmd
python -m drone_sim -c stacks/mocap_phri.py --layout circle -n 4    # indoor swarm on mocap + PHRI
python -m drone_sim --live -c stacks/manual_gps.py            # fly it yourself on estimated state
```

A stack file is like a launch file: a `stack()` function returning nodes.
Pass it anywhere a controller goes (`-c`, `--live`, multiple drones).
`stacks/` has four examples:

| Stack | Nodes |
|---|---|
| `perfect_pid.py` | perfect state → PID (flies exactly like `-c controllers/pid.py`) |
| `gps_imu_pid.py` | IMU + GPS → complementary filter → PID (outdoor) |
| `mocap_phri.py` | IMU + motion capture → filter → PHRI planner (indoor swarm) |
| `manual_gps.py` | IMU + GPS → filter → manual controller, for `--live` |

A minimal stack file looks like this:

```python
from drone_sim.nodes import ControllerNode, Gps, GpsImuEstimator, Imu

def stack():
    return [
        Imu(rate=250, accel_noise=0.05, gyro_noise=0.002),
        Gps(rate=10, pos_noise=0.1),
        GpsImuEstimator(),
        ControllerNode("controllers/pid.py", rate=250),   # any existing controller
    ]
```

Each drone's stack runs in its own namespace (`/drone0/imu`, ...), plus the
global `/swarm`. A `sim` node (the simulator bridge, like a Gazebo plugin)
publishes `ground_truth`, `setpoint`, `sticks`, and `/swarm`, and applies
`motor_command`:

| Node | Subscribes | Publishes |
|---|---|---|
| `Imu`, `Gps`, `Mocap` (sensors, configurable rate/noise) | `ground_truth` | `imu`, `gps`, `mocap` |
| `PerfectState` (no sensors; same as a plain controller) | `ground_truth` | `state_estimate` |
| `GpsImuEstimator`, `MocapImuEstimator` (complementary filters) | `imu` + `gps` / `mocap` | `state_estimate` |
| `ControllerNode(controller)` | `state_estimate`, `setpoint`, `/swarm`, `sticks` | `motor_command` |

Both estimators build on `InertialEstimator`, which integrates the IMU
between position fixes. Messages are defined in `drone_sim/nodes/msgs.py`.
With `--graph`, a plain controller (`-c controllers/pid.py`) is drawn as
its equivalent stack: perfect state → controller.

### Writing a node

```python
from dataclasses import dataclass
from drone_sim.nodes import Node
from drone_sim.nodes.msgs import StateEstimate

@dataclass(frozen=True)
class Altitude:          # messages are dataclasses with a `stamp`
    stamp: float
    z: float

class AltitudeMonitor(Node):
    kind = "estimator"                      # colors it in the graph view
    def setup(self):                        # runs on every (re)start
        self.pub = self.create_publisher("altitude", Altitude)
        self.create_subscription("state_estimate", StateEstimate, self.on_state)
        self.create_timer(0.5, self.report) # every 0.5 s of sim time
        self.z = 0.0
    def on_state(self, msg):
        self.z = msg.state.z
        self.pub.publish(Altitude(self.now, self.z))
    def report(self):
        print(f"{self.namespace} at {self.now:.1f} s: {self.z:.2f} m")
```

Topic names are relative to the drone's namespace unless they start with
`/`; rename a node's topics with `remap={"imu": "imu_raw"}` (e.g. to put a
filter node between a sensor and the estimator). Parameters passed as
keyword arguments are available as `self.param(name, default)`.

Unlike ROS, everything runs in one process in **simulation time**: timers
fire in time order (same-time timers in the order nodes are listed, so list
stacks like a pipeline), and publishing delivers to subscribers immediately.
Runs are exactly reproducible, and sensor noise is seeded per drone and per
sensor. `graph.describe()` lists nodes and topics with their measured rates,
and `to_dot` / `to_mermaid` export the graph.

With real sensors every PHRI drone sees a slightly different world, so they
can no longer share game solves (see below): a 4-drone mocap swarm takes
about 3× longer than with perfect state.

## Prosocial planning (PHRI)

`controllers/phri.py` implements the planner from

> J. Geldenbott and K. Leung, *Legible and Proactive Robot Planning for
> Prosocial Human-Robot Interactions*, ICRA 2024.
> [arXiv:2404.03734](https://arxiv.org/abs/2404.03734) ·
> [paper code (PHRI.jl)](https://github.com/UW-CTRL/PHRI.jl) ·
> [multi-agent IBR (ProactiveHRI.jl, `multi_ibr_dev`)](https://github.com/UW-CTRL/ProactiveHRI.jl/tree/multi_ibr_dev)

extended to any number of drones with iterated best response, as on the
`multi_ibr_dev` branch.

```bash
python -m drone_sim -c controllers/phri.py --layout circle -n 4   # 4 drones cross a circle
python -m drone_sim -c controllers/pid.py  --layout circle -n 4   # same with PID: they crash
python -m drone_sim --live -n 2 -c controllers/manual.py -c controllers/phri.py --layout circle
```

Every 0.1 s each PHRI drone solves a game with every drone around it:

- **Ideal problem:** reach the goal ignoring everyone. Its cost c(τ_ideal)
  sets the **inconvenience budget** (1 + β)·c(τ_ideal), so no agent has to
  give way more than its share.
- **Inconvenience problem:** the same plus a **markup** μ^t on running costs
  (μ > 1 rewards avoiding early, making intent legible and the drone
  proactive), linearized **collision avoidance** with discounted slack, the
  budget constraint, and a trust region, solved by sequential convex
  programming.
- **Iterated best response:** every agent re-plans against the others'
  latest plans, 3 rounds by default (`ibr_iterations = 1` gives the paper's
  non-interactive baseline). The drone flies the first part of its plan with
  the usual attitude controller.

Other drones are modeled as running the same planner toward their current
targets (exposed to controllers as `Neighbor.target`, the "privileged
information" the paper assumes), whatever controller they really run, so a
PHRI drone also avoids PID drones or a human pilot. Settings are class
attributes on `Prosocial` (`params`, `other_params`, `ibr_iterations`,
`order`, `interaction_radius`); planner hyperparameters live in
`drone_sim.phri.PHRIParams` and default to the paper's values.

Differences from the Julia code, since these are drones: the planner uses a
3D double integrator instead of a unicycle, so drones can also pass over or
under each other; a soft altitude floor; and a 2 cm "keep right" nudge so
perfectly symmetric head-on encounters still resolve. Drones running the
planner share identical game solves (`GameCache`), so cost grows linearly
with the number of drones (about 0.45 s of compute per drone per simulated
second on an M1 Mac). Requires `cvxpy` (Clarabel by default; set
`PHRIParams(solver="ECOS")` for the paper's solver).

## Flying it yourself

```bash
python -m drone_sim --live                 # manual controller, try to reach the red X
python -m drone_sim --live -s square --gust 0.5
```

| Key | Action |
|---|---|
| `↑` / `↓` | climb / descend (start here to take off) |
| `W` / `S` | forward / back |
| `A` / `D` | left / right |
| `←` / `→` | turn left / right |
| `R` | reset |
| `Esc` | quit |

`controllers/manual.py` flies in velocity mode, like a camera drone: the
keys ask for a speed relative to where the drone is facing, and letting go
brakes to a hover. Click the window first if the keys don't respond.

Any controller can read the pilot's inputs from `self.sticks`
(`throttle`, `roll`, `pitch`, `yaw`, each in [-1, 1]), so you can write
your own flight mode (e.g. angle mode or acro) and fly it with
`--live --controller path/to/yours.py`. You can also fly an autonomous
controller live, e.g. `--live -c controllers/pid.py -s figure8`.

## Web app

`web/` runs the same simulator in a browser tab, with the Python running in
the page (Pyodide in a Web Worker) and no server:

- **Fly** tab: fly with the keyboard; the on-screen keys also work as touch controls
- **Scenarios** tab: run a scenario and replay it in 3D with plots
- **Controllers** tab: edit controllers in Python in the page

Requires Node 18+.

```bash
cd web
npm install
npm run dev        # open the localhost URL it prints
npm run build      # static site in web/dist/
```

The build uses relative paths, so `web/dist/` can be served by any static
host (e.g. GitHub Pages). The first load downloads about 12 MB of Python
(Pyodide 0.29.4 and numpy) from cdn.jsdelivr.net; it is cached afterwards.

The page bundles `drone_sim/` and `controllers/` straight from the repo at
build time (`graphics/` is left out), so changes to the simulator or a new
file in `controllers/` show up in the web app without touching `web/`. The
only web-specific Python is `web/src/python/bridge.py`, which sets up drones
the way `__main__.py` does. `tests/test_web_bridge.py` runs it under CPython,
so `python -m pytest tests/` catches core changes that would break the web app.

Controller edits made in the page are kept in that browser only
(localStorage) and never change the repo's files. To keep one, click Download
and move the file into `controllers/`; it then works from the command line too
and shows up in the web app's list for everyone.

## Writing your own controller

Make a `.py` file anywhere with one subclass of `drone_sim.Controller`:

```python
from drone_sim import Controller

class MyController(Controller):
    def reset(self, params):
        # Called once before the run. params is a DroneParams
        # (mass, inertia, arm_length, max_thrust, hover_thrust, mixer, ...)
        self.p = params

    def update(self, t, state, target, dt):
        # state:  State with .pos .vel (world, 3-vectors), .quat (w, x, y, z),
        #         .omega (body rates), plus .R (rotation matrix) and .euler
        # target: (x, y, z, yaw) desired position and heading
        # return: thrust commands in newtons for rotors FL, FR, BR, BL
        total_thrust = self.p.mass * self.p.gravity
        torque = [0.0, 0.0, 0.0]  # roll, pitch, yaw torque in the body frame
        return self.p.allocate(total_thrust, torque)
```

Then run it:

```bash
python -m drone_sim --controller path/to/my_controller.py
python -m drone_sim --controller path/to/file.py:MyController   # if the file has several
```

Start by copying `controllers/pid.py` (cascaded position PID → geometric
attitude control), `controllers/manual.py` (keyboard velocity mode), or
`controllers/hover_only.py` (the minimal template: holds altitude and stays
level, but drifts).

`drone_sim.control` has helpers for the usual last step: turn a desired
acceleration into a tilt-limited thrust vector (`thrust_vector`) and fly
along it while facing a heading (`track_thrust_vector`).

### Conventions

- **World frame:** z up, gravity along -z.
- **Body frame:** x forward, y left, z up through the rotors.
- **Euler angles** (`state.euler`) are ZYX `(roll, pitch, yaw)`, right-handed:
  positive roll lifts the left side, positive pitch dips the nose, positive yaw
  turns left (counter-clockwise from above).
- **Rotors** are numbered `0 FL, 1 FR, 2 BR, 3 BL`. FL and BR yaw the body
  counter-clockwise; FR and BL clockwise.
- `params.allocate(total_thrust, torque)` turns a total thrust (N) and body
  torque (N m) into the four rotor thrusts, so you don't have to work out
  the mixer by hand. `params.mixer` is the matrix going the other way.
- Commands are clipped to `[0, max_thrust]` and then pass through a first-order
  motor lag (`motor_tau`), so the drone doesn't get exactly what you ask for.
- The controller runs at `--control-hz` (default 250 Hz). Physics runs at 1 kHz.

## Using it from Python

For tuning sweeps, optimization, or custom target paths:

```python
from drone_sim import DroneParams, SimConfig, Simulator, Sticks, load_controller, run

ctrl = load_controller("controllers/pid.py")
result = run(ctrl, SimConfig(scenario="step", wind_gust=0.5),
             params=DroneParams(mass=1.2),
             target_fn=lambda t: (t * 0.5, 0.0, 1.5))   # optional (x, y, z[, yaw]) path
print(result.metrics())   # rms_pos_error_m, final_pos_error_m, max_tilt_deg, flipped, crashed
result.pos                # (N, 3) positions
result.euler              # (N, 3) roll, pitch, yaw
result.states             # (N, 13): pos, vel, quat, omega

# Step by step, e.g. to script stick inputs:
sim = Simulator(load_controller("controllers/manual.py"))
sim.controller.sticks = Sticks(throttle=1.0)
for _ in range(500):      # 2 s at 250 Hz
    sim.step()
print(sim.state.pos)
```

## Command-line options

`python -m drone_sim --help` prints the same list.

| Option | Default | |
|---|---|---|
| `-c`, `--controller FILE[:Class]` | `controllers/pid.py` (`manual.py` with `--live`) | controller or stack file; repeat for one drone each |
| `-n`, `--drones N` | one per `-c` | number of drones; with a single `-c`, each drone gets its own copy |
| `--layout line\|circle` | `line` | side by side flying the scenario, or circle swap |
| `--spacing M` | 1.0 | distance between drones in the line layout [m] |
| `--radius M` | 4.0 | circle layout radius [m] |
| `--ghost` | off | drones that touch pass through each other instead of crashing |
| `-s`, `--scenario NAME` | `step` | `hover`, `step`, `square`, `figure8`, `helix`, `spin` |
| `-d`, `--duration S` | 12 | simulated seconds (ignored with `--live`) |
| `--control-hz HZ` | 250 | controller rate (physics runs at 1 kHz) |
| `--wind X [Y [Z]]` | 0 | steady wind force [N] |
| `--gust N` | 0 | gust strength, std dev per horizontal axis [N] |
| `--noise SIGMA` | 0 | Gaussian noise added to every component of the state the controller is given |
| `--seed N` | 0 | random seed for gusts and noise |
| `--live` | off | fly in real time with the keyboard |
| `--speed X` | 1.0 | playback speed multiplier |
| `--save FILE` | | write the animation to `.gif`, or `.mp4` (needs ffmpeg) |
| `--no-gui` | off | print metrics only |
| `--graph` | off | show each drone's node graph instead of the flight |
| `--graph-out FILE` | | write the node graph: `.png`, `.svg`, `.pdf`, `.dot`, or `.mmd` |

`--live` can't be combined with `--save` or `--no-gui`.

## Running the tests

```bash
pip install pytest
python -m pytest tests/
```

The PHRI tests are skipped if `cvxpy` isn't installed.
`tests/test_web_bridge.py` runs the web app's Python bridge under CPython, so
the suite also catches simulator changes that would break the web app.

## Layout

```
drone_sim/
  dynamics.py     DroneParams, State, equations of motion, motor model, rotation helpers
  controller.py   Controller base class (the interface you implement) and Sticks
  control.py      helpers: thrust vector -> attitude -> rotor thrusts
  phri.py         prosocial multi-agent planner (PHRI paper, IBR)
  nodes/          ROS-style framework: Node/Graph (core), msgs, sensors,
                  estimators, bridge (stacks as controllers, ControllerNode)
  sim.py          Simulator, run(), scenarios, wind/noise, controller loader
  session.py      LiveSession: keys -> sticks, real-time stepping (no graphics)
  graphics/       everything that draws (the only code that imports matplotlib)
    drone.py        drone model shared by both views
    playback.py     animate a finished SimResult
    live.py         window for keyboard flight around a LiveSession
    node_graph.py   node graph view (rqt_graph-style) and image export
  __main__.py     command line
controllers/      example controllers
stacks/           example stacks (sensors + estimator + controller)
tests/            python -m pytest tests/
  test_web_bridge.py  runs web/src/python/bridge.py under CPython
web/              browser app (Vite + React + TypeScript, Python via Pyodide)
  src/python/bridge.py  the only web-specific Python
  src/sim/        Web Worker that loads Pyodide and the repo's Python, and its client
  src/three/      3D scene
  src/views/      the Fly, Scenarios and Controllers tabs
```

Graphics only read from the core (`SimResult`, `Simulator`, `LiveSession`);
nothing outside `graphics/` imports it, and a test enforces that controllers
and the simulator load without matplotlib. A new front end needs an
`animate(results, collisions=...)` and a `fly(session)`; see
`drone_sim/graphics/__init__.py`.

## License

MIT; see [LICENSE](LICENSE).
