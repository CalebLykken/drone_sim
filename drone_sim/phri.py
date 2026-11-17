"""Legible and proactive prosocial planning (PHRI) for multiple agents.

A Python port of

    J. Geldenbott and K. Leung, "Legible and Proactive Robot Planning for
    Prosocial Human-Robot Interactions", ICRA 2024. arXiv:2404.03734
    Paper code:        https://github.com/UW-CTRL/PHRI.jl
    Multi-agent IBR:   https://github.com/UW-CTRL/ProactiveHRI.jl/tree/multi_ibr_dev

Every agent plans with the same two problems, solved by sequential convex
programming (SCP):

1. Ideal problem: reach the goal as if nobody else were around. Its
   "convenience" c(tau_ideal) sets the agent's inconvenience budget
   (1 + inconvenience_ratio) * c(tau_ideal).
2. Inconvenience problem: the same, plus
   - a markup factor mu^t on running costs (mu > 1 makes later effort more
     expensive, so agents start avoiding early: proactive and legible),
   - linearized collision avoidance ||p_t - q_t||^2 >= r^2 against every other
     agent, softened by a slack eps_t with discounted penalty
     gamma0 * gamma^t * eps_t^2,
   - the inconvenience budget c(tau) <= budget, so no agent gives way more
     than its fair share,
   - a trust region on the change from the previous iterate.

Agents are coupled by iterated best response (IBR): each round, every agent
re-solves its inconvenience problem against the others' latest plans
(simultaneously, as in ProactiveHRI.jl's multi-agent `ibr`, or one after the
other). With `ibr_iterations=1` agents instead plan against the others' ideal
plans only (the paper's non-interactive "OC" baseline).

Differences from the Julia code, because these are drones, not ground robots:
- Dynamics are a 3D double integrator (position, velocity; control is
  acceleration) instead of a unicycle, discretized exactly with dt.
- Convenience features use velocity change for "acceleration" and the full
  final-state error (position and velocity) for distance to goal. The Julia
  integrator models return 0 here; this follows the paper's definition.
- The altitude floor is soft, and the speed limit is relaxed to the current
  speed when an agent is already faster, so every problem stays feasible.
- A small absolute budget floor keeps the budget constraint numerically
  feasible when an agent is already at its goal (ideal cost ~ 0).
- Exactly symmetric encounters (a perfect head-on) give a collision gradient
  with no sideways part, so nobody steps aside. Each agent linearizes about
  a point nudged a couple of centimeters to its right (`pass_side_bias`), so
  such encounters resolve like traffic: everyone keeps right.
- Solves are memoized: when several drones run this planner they all solve
  the same game from the same inputs, so the first one to solve it shares
  the result (see `GameCache`). This changes speed, not results.
- Like the Julia code's `Threads.@threads`, each IBR round solves every
  agent's problem in parallel, here in a pool of worker processes (Python
  threads wouldn't help: Clarabel holds the GIL while it solves). See
  `WorkerPool`. This also changes speed, not results.

This module needs cvxpy (with ECOS, the paper's solver; Clarabel also works).
Nothing else in drone_sim imports it.
"""

from __future__ import annotations

import atexit
import multiprocessing
import os
import subprocess
import sys
import threading
import time
import traceback
import warnings
from collections import OrderedDict
from dataclasses import dataclass, field

import cvxpy as cp
import numpy as np

POS = slice(0, 3)
VEL = slice(3, 6)


@dataclass(frozen=True)
class PHRIParams:
    """Planner hyperparameters. Defaults follow the paper / ProactiveHRI.jl."""

    dt: float = 0.1  # s, planner step
    horizon: int = 25  # steps (2.5 s)
    markup: float = 1.05  # mu: running-cost markup per step
    collision_radius: float = 1.0  # m, keep-out distance between agents
    collision_slack: float = 150.0  # gamma0: slack penalty weight
    collision_markup: float = 0.98  # gamma: slack penalty discount per step
    trust_region_weight: float = 5.0
    inconvenience_ratio: float = 0.2  # beta: allowed fractional increase over ideal
    inconvenience_weights: tuple[float, float, float] = (1.0, 1.0, 0.01)  # path, accel, goal
    control_weight: float = 1.0  # R = control_weight * I
    terminal_weights: tuple[float, float] = (10.0, 1.0)  # Qt on (position, velocity)
    max_speed: float = 3.0  # m/s
    max_accel: float = 3.0  # m/s^2 per axis
    min_altitude: float = 0.3  # m, soft floor once airborne
    altitude_slack: float = 1000.0
    ideal_iterations: int = 3
    budget_floor: float = 1e-2  # absolute slack on the budget (see module docstring)
    pass_side_bias: float = 0.02  # m, nudge to the right when linearizing (see docstring)
    solver: str = "CLARABEL"  # or "ECOS", the paper's solver (slower here)


def double_integrator(dt: float) -> tuple[np.ndarray, np.ndarray]:
    """Exact discretization of p'' = u in 3D. State [p, v], control u."""
    A = np.eye(6)
    A[POS, VEL] = dt * np.eye(3)
    B = np.vstack([0.5 * dt**2 * np.eye(3), dt * np.eye(3)])
    return A, B


def straight_line(x0: np.ndarray, goal: np.ndarray, params: PHRIParams) -> tuple[np.ndarray, np.ndarray]:
    """Initial guess: coast toward the goal at the current speed (at least 1 m/s)."""
    N, dt = params.horizon, params.dt
    d = goal[POS] - x0[POS]
    dist = np.linalg.norm(d)
    speed = min(max(np.linalg.norm(x0[VEL]), 1.0), params.max_speed)
    direction = d / dist if dist > 1e-6 else np.zeros(3)
    xs = np.empty((N + 1, 6))
    for t in range(N + 1):
        travel = min(speed * t * dt, dist)
        xs[t, POS] = x0[POS] + direction * travel
        xs[t, VEL] = direction * speed if travel < dist else 0.0
    xs[0] = x0
    return xs, np.zeros((N, 3))


def convenience(xs: np.ndarray, goal: np.ndarray, weights) -> float:
    """c(tau): weighted squared path length, velocity change, and final error."""
    w1, w2, w3 = weights
    return float(
        w1 * np.sum(np.diff(xs[:, POS], axis=0) ** 2)
        + w2 * np.sum(np.diff(xs[:, VEL], axis=0) ** 2)
        + w3 * np.sum((xs[-1] - goal) ** 2)
    )


def _right_of(direction: np.ndarray) -> np.ndarray:
    """Unit horizontal vector to the right of `direction` (zero if vertical)."""
    right = np.array([direction[1], -direction[0], 0.0])
    n = np.linalg.norm(right)
    return right / n if n > 1e-6 else right


class AgentProblem:
    """The ideal and inconvenience problems for one agent, built once with
    cvxpy Parameters and re-solved with new data every planning step."""

    def __init__(self, params: PHRIParams):
        self.params = params
        self._ideal = self._build(n_others=None)
        self._incon: dict[int, tuple] = {}  # by number of other agents

    def _build(self, n_others: int | None):
        p = self.params
        N = p.horizon
        A, B = double_integrator(p.dt)
        x = cp.Variable((N + 1, 6))
        u = cp.Variable((N, 3))
        s = cp.Variable(N + 1, nonneg=True)  # altitude-floor slack
        par = {
            "x0": cp.Parameter(6),
            "goal": cp.Parameter(6),
            "xbar": cp.Parameter((N + 1, 6)),
            "ubar": cp.Parameter((N, 3)),
            "floor": cp.Parameter(),
            "vmax": cp.Parameter(nonneg=True),
        }
        markup = p.markup ** np.arange(1, N + 1)
        wp, wv = p.terminal_weights
        err = x[N] - par["goal"]
        cost = (
            p.control_weight * cp.sum(cp.multiply(markup, cp.sum(cp.square(u), axis=1)))
            + wp * cp.sum_squares(err[POS]) + wv * cp.sum_squares(err[VEL])
            + p.trust_region_weight * (cp.sum_squares(x - par["xbar"]) + cp.sum_squares(u - par["ubar"]))
            + p.altitude_slack * cp.sum_squares(s)
        )
        cons = [
            x[0] == par["x0"],
            x[1:].T == A @ x[:-1].T + B @ u.T,
            cp.abs(u) <= p.max_accel,
            cp.norm(x[1:, VEL], 2, axis=1) <= par["vmax"],
            x[:, 2] >= par["floor"] - s,
        ]
        eps = None
        if n_others is not None:
            eps = cp.Variable(N + 1, nonneg=True)
            par["budget"] = cp.Parameter(nonneg=True)
            par["G"] = [cp.Parameter((N + 1, 3)) for _ in range(n_others)]
            par["H"] = [cp.Parameter(N + 1) for _ in range(n_others)]
            discount = p.collision_markup ** np.arange(1, N + 2)
            cost += p.collision_slack * cp.sum(cp.multiply(discount, cp.square(eps)))
            for G, H in zip(par["G"], par["H"]):
                cons.append(cp.sum(cp.multiply(G, x[:, POS]), axis=1) + H >= -eps)
            w1, w2, w3 = p.inconvenience_weights
            c = (
                w1 * cp.sum_squares(x[1:, POS] - x[:-1, POS])
                + w2 * cp.sum_squares(x[1:, VEL] - x[:-1, VEL])
                + w3 * cp.sum_squares(x[N] - par["goal"])
            )
            cons.append(c <= par["budget"])
        return cp.Problem(cp.Minimize(cost), cons), x, u, eps, par

    def _set_common(self, par, x0, goal, xbar, ubar):
        p = self.params
        par["x0"].value = x0
        par["goal"].value = goal
        par["xbar"].value = xbar
        par["ubar"].value = ubar
        par["floor"].value = min(p.min_altitude, float(x0[2]))
        par["vmax"].value = max(p.max_speed, float(np.linalg.norm(x0[VEL])) + 1e-3)

    def _solve(self, prob) -> bool:
        try:
            with warnings.catch_warnings():
                # "Solution may be inaccurate": accepted on purpose, the next
                # replan (0.1 s later) starts from it anyway.
                warnings.simplefilter("ignore", UserWarning)
                prob.solve(solver=self.params.solver, warm_start=True)
        except cp.SolverError:
            return False
        return prob.status in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE)

    def solve_ideal(self, x0, goal, xbar, ubar):
        """SCP on the ideal problem. Returns (xs, us, ok)."""
        prob, x, u, _, par = self._ideal
        ok = True
        for _ in range(self.params.ideal_iterations):
            self._set_common(par, x0, goal, xbar, ubar)
            if not self._solve(prob):
                ok = False
                break
            xbar, ubar = x.value.copy(), u.value.copy()
        return xbar, ubar, ok

    def solve_incon(self, x0, goal, xbar, ubar, others_pos: list[np.ndarray], budget: float):
        """One SCP iteration of the inconvenience problem against fixed
        trajectories of the others ((N+1, 3) positions each).
        Returns (xs, us, max_slack, ok)."""
        n = len(others_pos)
        if n not in self._incon:
            self._incon[n] = self._build(n_others=n)
        prob, x, u, eps, par = self._incon[n]
        self._set_common(par, x0, goal, xbar, ubar)
        par["budget"].value = budget + self.params.budget_floor
        pbar = xbar[:, POS] + self.params.pass_side_bias * _right_of(goal[POS] - x0[POS])
        r2 = self.params.collision_radius ** 2
        for G, H, q in zip(par["G"], par["H"], others_pos):
            g = 2.0 * (pbar - q)  # gradient of ||p - q||^2 - r^2 at pbar
            G.value = g
            H.value = np.sum((pbar - q) ** 2, axis=1) - r2 - np.sum(g * pbar, axis=1)
        if not self._solve(prob):
            return xbar, ubar, float("nan"), False
        slack = float(eps.value.max()) if n else 0.0
        return x.value.copy(), u.value.copy(), slack, True


# --- Solving: in this process or in a pool of worker processes ---------------------
#
# A task is one agent's solve: ("ideal", name, params, x0, goal, xbar, ubar) or
# ("incon", name, params, x0, goal, xbar, ubar, others, budget, scp_iterations).
# run_task() is the same code in both places; only where it runs differs.

def run_task(problems: dict, task: tuple):
    """Run one task with the agent's (cached) AgentProblem.

    ideal -> (xs, us, ok, solves); incon -> (xs, us, slack, ok_any, solves, failures)
    """
    kind, name, params, x0, goal, xbar, ubar, *rest = task
    key = (name, params)
    problem = problems.get(key)
    if problem is None:
        problem = problems[key] = AgentProblem(params)
    if kind == "ideal":
        xs, us, ok = problem.solve_ideal(x0, goal, xbar, ubar)
        return xs, us, ok, params.ideal_iterations
    others, budget, iterations = rest
    xs, us, slack, any_ok, failures = xbar, ubar, None, False, 0
    for _ in range(iterations):
        nxs, nus, nslack, ok = problem.solve_incon(x0, goal, xs, us, others, budget)
        if ok:
            xs, us, slack, any_ok = nxs, nus, nslack, True
        else:
            failures += 1
    return xs, us, slack, any_ok, iterations, failures


def _worker_main(conn) -> None:
    """A worker process: keeps compiled problems for the agents assigned to
    it and runs batches of tasks until told to stop."""
    problems: dict = {}
    while True:
        try:
            tasks = conn.recv()
        except EOFError:
            return
        if tasks is None:
            return
        try:
            conn.send(("ok", [run_task(problems, t) for t in tasks]))
        except Exception:
            conn.send(("error", traceback.format_exc()))


def default_workers() -> int:
    """How many worker processes to use: DRONE_SIM_PHRI_WORKERS if set, 1 in
    the browser (Pyodide has no processes), else the number of performance
    cores (all cores where there's no such distinction), at most 8."""
    env = os.environ.get("DRONE_SIM_PHRI_WORKERS")
    if env:
        return max(1, int(env))
    if sys.platform == "emscripten":
        return 1
    cores = None
    if sys.platform == "darwin":  # Apple silicon: don't count efficiency cores
        try:
            out = subprocess.run(["sysctl", "-n", "hw.perflevel0.physicalcpu"],
                                 capture_output=True, text=True, timeout=2)
            cores = int(out.stdout.strip()) if out.returncode == 0 else None
        except (OSError, ValueError, subprocess.SubprocessError):
            cores = None
    return max(1, min(8, cores or os.cpu_count() or 1))


class WorkerPool:
    """Worker processes that solve agents' problems in parallel.

    Each agent (name, params) is pinned to one worker, so its cvxpy problems
    are compiled there once and reused; a round sends each worker one batch
    of small arrays. Started on first use and shared by every PHRIPlanner in
    the process (see `pool()`).
    """

    def __init__(self, size: int):
        ctx = multiprocessing.get_context("spawn")  # fork is unsafe with BLAS/Accelerate
        self.size = size
        self.conns = []
        self.procs = []
        for i in range(size):
            parent, child = ctx.Pipe()
            proc = ctx.Process(target=_worker_main, args=(child,), name=f"phri-worker-{i}", daemon=True)
            proc.start()
            child.close()
            self.conns.append(parent)
            self.procs.append(proc)
        self.assigned: dict[tuple, int] = {}
        self.load = [0] * size
        self.lock = threading.Lock()

    def _worker(self, key: tuple) -> int:
        if key not in self.assigned:
            i = min(range(self.size), key=lambda j: self.load[j])
            self.assigned[key] = i
            self.load[i] += 1
        return self.assigned[key]

    def run(self, tasks: list[tuple]) -> list:
        """Run tasks in parallel (one batch per worker); results in task order."""
        with self.lock:
            batches: dict[int, list[int]] = {}
            for index, task in enumerate(tasks):
                batches.setdefault(self._worker((task[1], task[2])), []).append(index)
            for w, indices in batches.items():
                self.conns[w].send([tasks[i] for i in indices])
            results: list = [None] * len(tasks)
            for w, indices in batches.items():
                try:
                    status, payload = self.conns[w].recv()
                except EOFError:
                    raise RuntimeError(f"PHRI worker {w} stopped unexpectedly") from None
                if status != "ok":
                    raise RuntimeError(f"PHRI worker {w} failed:\n{payload}")
                for i, result in zip(indices, payload):
                    results[i] = result
            return results

    def close(self) -> None:
        for conn in self.conns:
            try:
                conn.send(None)
                conn.close()
            except OSError:
                pass
        for proc in self.procs:
            proc.join(timeout=2)
            if proc.is_alive():
                proc.terminate()


_pool: WorkerPool | None = None
_pool_lock = threading.Lock()


def pool(size: int) -> WorkerPool:
    """The shared worker pool, started (or grown) on first use."""
    global _pool
    with _pool_lock:
        if _pool is None or _pool.size < size:
            if _pool is not None:
                _pool.close()
            _pool = WorkerPool(size)
        return _pool


def shutdown_pool() -> None:
    global _pool
    with _pool_lock:
        if _pool is not None:
            _pool.close()
            _pool = None


atexit.register(shutdown_pool)


@dataclass
class GameAgent:
    """One participant in the planning game, as the ego sees it."""

    name: str
    x0: np.ndarray  # [p, v]
    goal: np.ndarray  # [p, v] (goal velocity is zero)
    params: PHRIParams
    # Agents with no known goal (or crashed) just keep their current velocity.
    constant_velocity: bool = False
    xs: np.ndarray | None = None
    us: np.ndarray | None = None
    budget: float = 0.0
    ideal_xs: np.ndarray | None = None


@dataclass
class GameStats:
    solves: int = 0
    failures: int = 0
    seconds: float = 0.0
    max_slack: dict[str, float] = field(default_factory=dict)
    budgets: dict[str, float] = field(default_factory=dict)
    cached: bool = False  # reused another planner's identical solve
    wall_seconds: float = 0.0  # time spent in this call (tiny when cached)


class GameCache:
    """Recent game solutions, shared by every PHRIPlanner in the process.

    Keys contain every input of the solve (agents, states, goals, parameters,
    warm starts), so a hit returns exactly what solving would have returned.
    """

    capacity = 32
    _entries: "OrderedDict[tuple, tuple]" = OrderedDict()

    @classmethod
    def get(cls, key):
        hit = cls._entries.get(key)
        if hit is not None:
            cls._entries.move_to_end(key)
        return hit

    @classmethod
    def put(cls, key, value) -> None:
        cls._entries[key] = value
        while len(cls._entries) > cls.capacity:
            cls._entries.popitem(last=False)

    @classmethod
    def clear(cls) -> None:
        cls._entries.clear()


class PHRIPlanner:
    """Plans for the ego and everyone around it with multi-agent IBR.

    Keeps one AgentProblem per agent name (so cvxpy problems are compiled
    once) and warm-starts each agent from its previous plan shifted by one
    step.
    """

    def __init__(self, params: PHRIParams | None = None, other_params: PHRIParams | None = None,
                 ibr_iterations: int = 3, order: str = "simultaneous", workers: int | None = None):
        if order not in ("simultaneous", "sequential"):
            raise ValueError("order must be 'simultaneous' or 'sequential'")
        self.params = params or PHRIParams()
        self.other_params = other_params or self.params
        if self.other_params.horizon != self.params.horizon or self.other_params.dt != self.params.dt:
            raise ValueError("all agents must share dt and horizon")
        self.ibr_iterations = ibr_iterations
        self.order = order
        # Worker processes for parallel solves; 1 solves everything in this process.
        self.workers = default_workers() if workers is None else max(1, int(workers))
        self._problems: dict[tuple[str, PHRIParams], AgentProblem] = {}
        self._warm: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        self.last_stats = GameStats()
        self.last_game: list[GameAgent] = []

    def _run(self, tasks: list[tuple]) -> list:
        """Solve tasks, in parallel when there's more than one and workers > 1."""
        if self.workers > 1 and len(tasks) > 1:
            return pool(self.workers).run(tasks)
        return [run_task(self._problems, t) for t in tasks]

    def _initial_guess(self, agent: GameAgent):
        warm = self._warm.get(agent.name)
        if warm is None:
            return straight_line(agent.x0, agent.goal, agent.params)
        xs, us = warm
        A, B = double_integrator(agent.params.dt)
        xs = np.vstack([xs[1:], A @ xs[-1]])  # shift by one step, coast at the end
        us = np.vstack([us[1:], np.zeros((1, 3))])
        xs[0] = agent.x0
        return xs, us

    def _constant_velocity(self, agent: GameAgent) -> np.ndarray:
        p = agent.params
        t = np.arange(p.horizon + 1)[:, None] * p.dt
        return agent.x0[POS] + t * agent.x0[VEL]

    def plan(self, agents: list[GameAgent]) -> GameAgent:
        """Solve the game; agents[0] is the ego. Returns the ego with its plan.

        With simultaneous IBR the result doesn't depend on who the ego is, so
        agents are solved in name order and identical games (same agents,
        states, goals, parameters, and warm starts) are shared through
        GameCache instead of being solved again.
        """
        start = time.perf_counter()
        ego = agents[0]
        if self.order == "simultaneous":
            agents = sorted(agents, key=lambda a: a.name)
        planners = [a for a in agents if not a.constant_velocity]
        fixed = {a.name: self._constant_velocity(a) for a in agents if a.constant_velocity}
        guesses = {a.name: self._initial_guess(a) for a in planners}

        key = None
        if self.order == "simultaneous":
            key = (self.ibr_iterations, tuple(
                (a.name, a.params, a.constant_velocity, a.x0.tobytes(), a.goal.tobytes(),
                 *(g.tobytes() for g in guesses.get(a.name, ())))
                for a in agents
            ))
            hit = GameCache.get(key)
            if hit is not None:
                plans, stats = hit
                for a in planners:
                    a.xs, a.us, a.ideal_xs, a.budget = (v.copy() if isinstance(v, np.ndarray) else v
                                                        for v in plans[a.name])
                self._finish(planners, agents, GameStats(**{**vars(stats), "cached": True}), start)
                return ego

        stats = GameStats()

        # 1. Ideal plans set each agent's budget and the starting point for IBR.
        ideal = self._run([("ideal", a.name, a.params, a.x0, a.goal, *guesses[a.name]) for a in planners])
        for a, (xs, us, ok, solves) in zip(planners, ideal):
            a.xs, a.us = xs, us
            stats.solves += solves
            stats.failures += not ok
            a.ideal_xs = a.xs.copy()
            a.budget = (1 + a.params.inconvenience_ratio) * convenience(a.xs, a.goal, a.params.inconvenience_weights)
            stats.budgets[a.name] = a.budget

        def respond_task(a: GameAgent, plans: dict[str, np.ndarray], scp_iterations: int = 1) -> tuple:
            others = [plans[b.name] for b in agents if b is not a]
            return ("incon", a.name, a.params, a.x0, a.goal, a.xs, a.us, others, a.budget, scp_iterations)

        def apply(group: list[GameAgent], results: list) -> None:
            for a, (xs, us, slack, ok, solves, failures) in zip(group, results):
                stats.solves += solves
                stats.failures += failures
                if ok:
                    a.xs, a.us = xs, us
                    stats.max_slack[a.name] = slack

        # 2. Best responses. Simultaneous rounds solve every agent at once
        #    (in parallel); sequential rounds go one agent at a time. One
        #    iteration = non-interactive baseline: plan against the others'
        #    ideal plans with a few SCP iterations.
        if self.ibr_iterations <= 1:
            plans = {**fixed, **{a.name: a.ideal_xs[:, POS] for a in planners}}
            apply(planners, self._run([respond_task(a, plans, scp_iterations=5) for a in planners]))
        else:
            for _ in range(self.ibr_iterations):
                plans = {**fixed, **{a.name: a.xs[:, POS] for a in planners}}
                if self.order == "simultaneous":
                    apply(planners, self._run([respond_task(a, plans) for a in planners]))
                else:
                    for a in planners:
                        apply([a], self._run([respond_task(a, plans)]))
                        plans[a.name] = a.xs[:, POS]

        stats.seconds = time.perf_counter() - start
        if key is not None:
            GameCache.put(key, ({a.name: (a.xs, a.us, a.ideal_xs, a.budget) for a in planners}, stats))
        self._finish(planners, agents, stats, start)
        return ego

    def _finish(self, planners, agents, stats, start) -> None:
        for a in planners:
            self._warm[a.name] = (a.xs, a.us)
        stats.wall_seconds = time.perf_counter() - start
        self.last_stats = stats
        self.last_game = agents

    def forget(self, name: str) -> None:
        """Drop the warm start for an agent (e.g. after a reset)."""
        self._warm.pop(name, None)
