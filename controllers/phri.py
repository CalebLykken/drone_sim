"""Legible and proactive prosocial planning (Geldenbott & Leung, ICRA 2024).

    python -m drone_sim -c controllers/phri.py -n 2 ...
    python -m drone_sim --live -c controllers/manual.py -c controllers/phri.py

Every 0.1 s the drone solves a planning game with every drone around it
(see drone_sim/phri.py): each agent minimizes effort and distance to its
goal, with a markup that rewards avoiding early, collision avoidance, and an
inconvenience budget so nobody gives way more than their share. Agents are
coupled by iterated best response. The drone flies the first part of its
plan with the usual attitude controller until the next replan.

Other drones are modeled as running the same planner (with `other_params`)
toward their current targets, whatever controller they really run, just
like the paper models the human. Crashed drones, or ones without a known
target, are treated as moving at constant velocity.

Paper:  https://arxiv.org/abs/2404.03734
Code:   https://github.com/UW-CTRL/PHRI.jl
        https://github.com/UW-CTRL/ProactiveHRI.jl/tree/multi_ibr_dev (multi-agent IBR)
"""

import math

import numpy as np

from drone_sim import Controller
from drone_sim.control import thrust_vector, track_thrust_vector
from drone_sim.phri import POS, VEL, GameAgent, PHRIParams, PHRIPlanner


class Prosocial(Controller):
    params = PHRIParams()  # this drone's planner
    other_params = None  # how it models everyone else (None: same as params)
    ibr_iterations = 3  # 1 = non-interactive baseline (plan against others' ideal plans)
    order = "simultaneous"  # or "sequential" (this drone responds first)
    interaction_radius = None  # m; ignore drones farther than this (None: everyone)
    workers = None  # processes solving agents in parallel (None: one per performance core; 1: none)

    k_pos = 4.0  # tracking gains between replans
    k_vel = 3.0
    max_tilt = math.radians(35)

    def reset(self, params):
        self.p = params
        self.planner = PHRIPlanner(self.params, self.other_params, self.ibr_iterations, self.order, self.workers)
        self.plan = None  # (xs, us) from the last replan
        self.plan_t = -math.inf
        self.last_stats = None

    def update(self, t, state, target, dt):
        pp = self.params
        if self.plan is None or t >= self.plan_t + pp.dt - 1e-9:
            self._replan(t, state, target)

        # Follow the plan: exact double-integrator interpolation within a step.
        xs, us = self.plan
        tau = t - self.plan_t
        k = min(int(tau / pp.dt + 1e-9), pp.horizon - 1)
        s = tau - k * pp.dt
        a_ref = us[k]
        v_ref = xs[k, VEL] + a_ref * s
        p_ref = xs[k, POS] + xs[k, VEL] * s + 0.5 * a_ref * s * s

        acc = a_ref + self.k_pos * (p_ref - state.pos) + self.k_vel * (v_ref - state.vel)
        f = thrust_vector(acc, self.p, self.max_tilt)
        return track_thrust_vector(f, target[3], state, self.p)

    def _replan(self, t, state, target):
        others_params = self.planner.other_params
        me = GameAgent(self.name, np.r_[state.pos, state.vel], _goal(target), self.params)
        agents = [me]
        for n in self.others:
            if self.interaction_radius is not None and \
                    np.linalg.norm(n.state.pos - state.pos) > self.interaction_radius:
                continue
            known = n.target is not None and not n.crashed
            agents.append(GameAgent(
                n.name, np.r_[n.state.pos, n.state.vel],
                _goal(n.target) if known else np.r_[n.state.pos, 0.0, 0.0, 0.0],
                others_params, constant_velocity=not known,
            ))
        self.planner.plan(agents)
        self.plan = (me.xs, me.us)
        self.plan_t = t
        self.last_stats = self.planner.last_stats


def _goal(target):
    return np.r_[target[:3], 0.0, 0.0, 0.0]
