"""Everything that draws.

The rest of the package never imports this, so controllers and the
simulator run without matplotlib. Graphics only read from the core:

- playback.py  animates finished flights: one SimResult or a list of them
               (one per drone, from `Simulator.results()`) plus the
               simulator's collisions
- live.py      opens a window around a LiveSession (drone_sim/session.py),
               forwarding key events to it and drawing every drone in
               `session.sim.drones`; the camera follows the first (piloted) one
- drone.py     the drone model and per-drone colors shared by both

To swap in a different front end, implement the same two entry points
(`animate(results, ..., collisions=...)` and `fly(session, ...)`) and point
__main__.py at them.
"""
