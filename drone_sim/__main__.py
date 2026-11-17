"""Command-line entry point: python -m drone_sim --controller controllers/pid.py"""

import argparse
import sys
from pathlib import Path

import numpy as np

from .dynamics import State
from .sim import SCENARIOS, DroneSpec, SimConfig, Simulator, load_controller, offset_target

CONTROLLERS = Path(__file__).resolve().parent.parent / "controllers"


def main(argv=None):
    ap = argparse.ArgumentParser(prog="drone_sim", description=__doc__)
    ap.add_argument("-c", "--controller", action="append",
                    help="path/to/file.py or path/to/file.py:ClassName; repeat for one drone each "
                         "(default: controllers/pid.py, or controllers/manual.py with --live)")
    ap.add_argument("-n", "--drones", type=int,
                    help="number of drones (default: one per --controller); with a single "
                         "--controller, every drone flies a copy of it")
    ap.add_argument("--spacing", type=float, default=1.0,
                    help="drones start side by side this far apart [m] and fly the scenario "
                         "shifted by the same amount")
    ap.add_argument("--ghost", action="store_true",
                    help="drones that touch pass through each other instead of crashing")
    ap.add_argument("--live", action="store_true",
                    help="fly in real time with the keyboard (see controllers/manual.py)")
    ap.add_argument("-s", "--scenario", default="step", choices=sorted(SCENARIOS))
    ap.add_argument("-d", "--duration", type=float, default=12.0, help="seconds (ignored with --live)")
    ap.add_argument("--control-hz", type=float, default=250.0)
    ap.add_argument("--wind", type=float, nargs="+", default=[0.0], metavar="N",
                    help="steady wind force [N]: X, or X Y, or X Y Z")
    ap.add_argument("--gust", type=float, default=0.0, help="gust strength (std dev per axis) [N]")
    ap.add_argument("--noise", type=float, default=0.0, help="sensor noise std dev")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--speed", type=float, default=1.0, help="playback speed multiplier")
    ap.add_argument("--save", help="save the animation to a .gif or .mp4 instead of showing it")
    ap.add_argument("--no-gui", action="store_true", help="just print metrics")
    args = ap.parse_args(argv)
    if len(args.wind) > 3:
        ap.error("--wind takes at most 3 values (X Y Z)")
    if args.live and (args.save or args.no_gui):
        ap.error("--live can't be combined with --save or --no-gui")

    # One controller path per drone. Live mode: the first drone is piloted.
    paths = args.controller
    n = args.drones or (len(paths) if paths else 1)
    if n < 1:
        ap.error("--drones must be at least 1")
    if paths is None:
        paths = [str(CONTROLLERS / "manual.py")] if args.live else [str(CONTROLLERS / "pid.py")]
        paths += [str(CONTROLLERS / "pid.py")] * (n - 1)
    elif len(paths) == 1:
        paths = paths * n
    elif len(paths) != n:
        ap.error(f"got {len(paths)} --controller values but --drones {n}")

    sys.path.insert(0, str(Path.cwd()))  # let controllers import helpers next to them
    scenario = SCENARIOS[args.scenario]
    specs = []
    for i, path in enumerate(paths):
        y = (i - (n - 1) / 2) * args.spacing
        specs.append(DroneSpec(
            load_controller(path),  # a fresh instance per drone
            initial_state=State(pos=np.array([0.0, y, 0.0])),
            target_fn=offset_target(scenario, (0.0, y, 0.0)),
        ))
    config = SimConfig(
        duration=args.duration,
        control_hz=args.control_hz,
        scenario=args.scenario,
        wind_mean=tuple(args.wind + [0.0] * (3 - len(args.wind))),
        wind_gust=args.gust,
        sensor_noise=args.noise,
        seed=args.seed,
        crash_on_collision=not args.ghost,
    )
    try:
        sim = Simulator(drones=specs, config=config)
    except ValueError as e:
        ap.error(f"{e} (try a bigger --spacing, or --ghost)")
    show = not (args.no_gui or args.save)
    if n > 1 and (show or args.live or args.save):
        print("note: the graphics currently draw only the first drone", file=sys.stderr)

    if args.live:
        from .graphics.live import fly
        from .session import LiveSession

        fly(LiveSession(sim))
        if sim.k == 0:
            return
    else:
        for _ in range(int(round(config.duration / sim.control_dt))):
            sim.step()

    print(f"scenario: {args.scenario}")
    for d, r in zip(sim.drones, sim.results()):
        print(f"{d.name}  controller: {type(d.controller).__name__}")
        for k, v in r.metrics().items():
            print(f"  {k:20s} {v:.3f}" if isinstance(v, float) else f"  {k:20s} {v}")
    for c in sim.collisions:
        print(f"collision: {c.a} and {c.b} at t = {c.t:.2f} s")

    if args.no_gui or args.live:
        return
    result = sim.result()
    from .graphics.playback import animate

    if args.save:
        animate(result, speed=args.speed, save=args.save)
        print(f"saved {args.save}")
    else:
        import matplotlib.pyplot as plt

        anim = animate(result, speed=args.speed)  # noqa: F841 (keep a reference alive)
        plt.show()


if __name__ == "__main__":
    main()
