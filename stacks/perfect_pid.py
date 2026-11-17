"""The plain simulator as a stack: perfect state feedback into the PID.

Flies exactly like `-c controllers/pid.py`.
"""

from pathlib import Path

from drone_sim.nodes import ControllerNode, PerfectState

CONTROLLERS = Path(__file__).resolve().parent.parent / "controllers"


def stack():
    return [
        PerfectState(rate=250),
        ControllerNode(CONTROLLERS / "pid.py", rate=250),
    ]
