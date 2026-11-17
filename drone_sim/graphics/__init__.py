"""Everything that draws.

The rest of the package never imports this, so controllers and the
simulator run without matplotlib. Graphics only read from the core:

- playback.py  animates a finished SimResult
- live.py      opens a window around a LiveSession (drone_sim/session.py),
               forwarding key events to it and drawing `session.sim`
- drone.py     the drone model shared by both

To swap in a different front end, implement the same two entry points
(`animate(result, ...)` and `fly(session, ...)`) and point __main__.py at them.
"""
