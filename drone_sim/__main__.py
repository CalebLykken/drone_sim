"""Command-line entry point: python -m drone_sim --controller controllers/pid.py"""

import argparse
import sys
from pathlib import Path

from .sim import SCENARIOS, SimConfig, load_controller, run

DEFAULT_CONTROLLER = Path(__file__).resolve().parent.parent / "controllers" / "pid.py"


def main(argv=None):
    ap = argparse.ArgumentParser(prog="drone_sim", description=__doc__)
    ap.add_argument("-c", "--controller", default=str(DEFAULT_CONTROLLER),
                    help="path/to/file.py or path/to/file.py:ClassName (default: controllers/pid.py)")
    ap.add_argument("-s", "--scenario", default="step", choices=sorted(SCENARIOS))
    ap.add_argument("-d", "--duration", type=float, default=12.0, help="seconds")
    ap.add_argument("--control-hz", type=float, default=250.0)
    ap.add_argument("--wind", type=float, default=0.0, help="steady sideways force [N]")
    ap.add_argument("--gust", type=float, default=0.0, help="gust strength (std dev) [N]")
    ap.add_argument("--noise", type=float, default=0.0, help="sensor noise std dev")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--speed", type=float, default=1.0, help="playback speed multiplier")
    ap.add_argument("--save", help="save the animation to a .gif or .mp4 instead of showing it")
    ap.add_argument("--no-gui", action="store_true", help="just print metrics")
    args = ap.parse_args(argv)

    sys.path.insert(0, str(Path.cwd()))  # let controllers import helpers next to them
    controller = load_controller(args.controller)
    config = SimConfig(
        duration=args.duration,
        control_hz=args.control_hz,
        scenario=args.scenario,
        wind_mean=args.wind,
        wind_gust=args.gust,
        sensor_noise=args.noise,
        seed=args.seed,
    )
    result = run(controller, config)

    print(f"controller: {type(controller).__name__}  scenario: {args.scenario}")
    for k, v in result.metrics().items():
        print(f"  {k:20s} {v:.3f}" if isinstance(v, float) else f"  {k:20s} {v}")

    if args.no_gui:
        return
    from .viz import animate

    if args.save:
        animate(result, speed=args.speed, save=args.save)
        print(f"saved {args.save}")
    else:
        import matplotlib.pyplot as plt

        anim = animate(result, speed=args.speed)  # noqa: F841 (keep a reference alive)
        plt.show()


if __name__ == "__main__":
    main()
