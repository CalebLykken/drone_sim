# drone_sim
Drone Simulator

A small planar (2D) quadrotor simulator for trying out flight controllers. You
write a controller as a Python class, point the sim at the file, and watch it
fly.

- Rigid-body dynamics in the x-z plane with pitch (`x, z, theta` + velocities)
- Motor lag and thrust saturation
- Optional steady wind, gusts, and sensor noise
- Live animation with time-series plots, or headless metrics for batch tuning

## Quick start

Requires Python 3.10+, `numpy`, and `matplotlib` (`pip install numpy matplotlib`).

```bash
python -m drone_sim                                   # example cascaded PID, step scenario
python -m drone_sim -s figure8                        # other scenarios: hover, step, square, figure8
python -m drone_sim -s step --wind 1 --gust 0.5       # add a disturbance
python -m drone_sim --no-gui                          # just print metrics
python -m drone_sim --save flight.gif                 # export the animation
```

## Writing your own controller

Make a `.py` file anywhere with one subclass of `drone_sim.Controller`:

```python
from drone_sim import Controller

class MyController(Controller):
    def reset(self, params):
        # Called once before the run. params is a DroneParams
        # (mass, inertia, arm_length, gravity, max_thrust, hover_thrust, ...)
        self.p = params

    def update(self, t, state, target, dt):
        # state:  State with .x .z .theta .vx .vz .omega
        # target: (x, z) desired position
        # return: (left, right) rotor thrust commands in newtons
        hover = self.p.hover_thrust
        return hover, hover
```

Then run it:

```bash
python -m drone_sim --controller path/to/my_controller.py
python -m drone_sim --controller path/to/file.py:MyController   # if the file has several
```

Start by copying `controllers/pid.py` (cascaded position → attitude PID) or
`controllers/hover_only.py` (the minimal template).

### Conventions

- `theta` is pitch, counter-clockwise positive. More thrust on the **right**
  rotor rotates the drone CCW, which tilts thrust toward **-x**.
- Commands are clipped to `[0, max_thrust]` and then pass through a first-order
  motor lag (`motor_tau`), so the drone doesn't get exactly what you ask for.
- The controller runs at `--control-hz` (default 250 Hz). Physics runs at 1 kHz.

## Using it from Python

For tuning sweeps, optimization, or custom target paths:

```python
from drone_sim import DroneParams, SimConfig, load_controller, run

ctrl = load_controller("controllers/pid.py")
result = run(ctrl, SimConfig(scenario="step", wind_gust=0.5),
             params=DroneParams(mass=1.2),
             target_fn=lambda t: (t * 0.5, 1.5))   # optional custom path
print(result.metrics())   # rms_pos_error_m, final_pos_error_m, max_tilt_deg, flipped
result.states             # (N, 6) array: x, z, theta, vx, vz, omega
```

## Layout

```
drone_sim/
  dynamics.py     DroneParams, State, equations of motion, motor model
  controller.py   Controller base class (the interface you implement)
  sim.py          run loop, scenarios, wind/noise, controller loader
  viz.py          matplotlib animation
  __main__.py     command line
controllers/      example controllers
tests/            python -m pytest tests/
```
