"""Measurement and growth of the qijt tree (spec/QIJ_qijt_spec.md
sections 2.3, 4, 5, 9.2): the batch contract shared by every stage, one
node's coefficient/energy, and the level-synchronous round loop.
"""

from __future__ import annotations

import time
from typing import List, Sequence, Tuple

import numpy as np

from ..parallel import call_T, fit_status
from .differences import step_parameter
from .tree_h import (
    TreeState, active_rows, is_degenerate, is_point_leaf, measured_child,
    node_mass, node_rows, open_cells,
)


def set_weights(omega0: np.ndarray, member_pos: np.ndarray, p_K: float, t: float) -> np.ndarray:
    """omega_r(t) = omega0_r * [(1-t) + t*1_K(r)/p_K] (spec 2.3), a new
    (R,) array over the compact row order; `member_pos` are K's rows."""
    omega = omega0 * (1.0 - t)
    omega[member_pos] += omega0[member_pos] * t / p_K
    return omega


def _eval_task(T, case, shared: np.ndarray, task: Tuple) -> Tuple:
    """One evaluation, run worker-side (spec 9.2): task = (key, omega,
    start, eta). Failure boundary (R7): an exception is caught and
    reported as a NaN result of T's own width."""
    key, omega, start, eta = task
    t0 = time.perf_counter()
    try:
        value = np.asarray(call_T(T, shared, omega, start, eta), dtype=float)
        failed = bool(np.any(np.isnan(value)))
        status = fit_status(T, value)
    except Exception:
        value = np.full(len(T.outputs), np.nan)
        failed = True
        status = fit_status(T, value, raised=True)
    return key, value, failed, status, time.perf_counter() - t0


def run_batch(T, counter, pool, shared: np.ndarray, tasks: Sequence[Tuple]) -> List[Tuple]:
    """`pool.map` over `tasks` = [(key, omega, start, eta), ...] (spec
    9.2), or the same tasks run serially when `pool is None`; consumed
    by position. `counter.add` is charged once per task in the parent
    either way, so accounting matches at any worker count."""
    if pool is not None:
        pool.share(shared)
        results = pool.map(_eval_task, tasks)
    else:
        results = [_eval_task(T, None, shared, task) for task in tasks]
    n_rows = len(shared)
    for _, _, failed, _, _ in results:
        counter.add(1, n_rows, int(failed))
    return results


def measure_nodes(state: TreeState, T, counter, pool, nodes: Sequence[int], round_: int) -> None:
    """Measure each node in `nodes`, in order given: one evaluation of
    U_A along its measured child (spec 4.1), then U_B, y, s, E and the
    pass rule (4.2). A failed evaluation gives y/s NaN, E 0, no pass."""
    row_ids, rows_x, omega0 = active_rows(state)
    tasks, info = [], []
    for c in nodes:
        A, B = measured_child(state, c)
        p_A, p_B = node_mass(state, A), node_mass(state, B)
        t_A = step_parameter(state.delta_f, p_A)
        pos_A = np.searchsorted(row_ids, node_rows(state, A))
        tasks.append((c, set_weights(omega0, pos_A, p_A, t_A), state.theta_Q, state.eta_full))
        info.append((A, B, p_A, p_B, t_A))
    meas = state.measured
    for (c, value, failed, status, _wall), (A, B, p_A, p_B, t_A) in zip(
        run_batch(T, counter, pool, rows_x, tasks), info,
    ):
        p_c = p_A + p_B
        if failed:
            U_A_full = np.full(state.q_full, np.nan)
            U_B_full = np.full(state.q_full, np.nan)
            y = np.full(meas.size, np.nan)
            s = np.full(meas.size, np.nan)
            E = np.zeros(meas.size)
            passes = np.zeros(meas.size, dtype=bool)
        else:
            U_A_full = (value - state.theta_Q) / t_A
            U_B_full = (p_c * state.U[c] - p_A * U_A_full) / p_B
            y = (U_A_full - U_B_full)[meas]
            s = np.sqrt(2.0) * state.eta_Q * np.abs(state.theta_Q[meas]) / t_A * (p_c / p_B)
            passes = np.abs(y) > state.z_n * s
            E = (1.0 / state.N) * (p_A * p_B / p_c) * y ** 2 * passes
        state.U[A], state.U[B] = U_A_full, U_B_full
        state.y[c, :], state.s[c, :], state.E[c, :], state.passes[c, :] = y, s, E, passes
        state.node_t[c], state.node_status[c], state.node_round[c] = t_A, status, round_
        state.meas_child[c] = 0 if A == state.node_child0[c] else 1
        state.node_measured[c] = True


def reevaluate_theta_Q(state: TreeState, T, counter) -> bool:
    """theta_Q <- T(rows, omega0, start=theta_Q, eta=eta_full) after
    openings (spec 5.3); False (theta_Q set NaN) means 'opening_failed'."""
    _, rows_x, omega0 = active_rows(state)
    value = np.asarray(counter(rows_x, omega0, start=state.theta_Q, eta=state.eta_full), dtype=float)
    if np.any(np.isnan(value)):
        state.theta_Q = np.full(state.q_full, np.nan)
        return False
    state.theta_Q = value
    return True


def _open_candidates(state: TreeState, prev_round: int) -> List[int]:
    """Children of nodes measured in `prev_round` that passed, less
    point leaves and degenerate nodes (spec 5.2 step 1)."""
    candidates = []
    for c in np.where(state.node_round == prev_round)[0]:
        if not state.passes[c].any():
            continue
        for child in (state.node_child0[c], state.node_child1[c]):
            if not is_point_leaf(state, child) and not is_degenerate(state, child):
                candidates.append(int(child))
    return candidates


def _priority(E_parent: np.ndarray, V_btw: np.ndarray) -> float:
    """max_o E_parent,o / V_btw,o, 0 where V_btw,o = 0 (spec 5.2 step 2)."""
    ratio = np.zeros_like(V_btw)
    mask = V_btw > 0.0
    ratio[mask] = E_parent[mask] / V_btw[mask]
    return float(np.max(ratio)) if ratio.size else 0.0


def _curve_row(round_: int, evals_tree: int, R: int, state: TreeState, this_round: Sequence[int]) -> dict:
    """One curve row (spec 11.4): V_btw sums every measured node's E in
    ascending id (boolean masking preserves index order, spec 9.2)."""
    n_open = sum(1 for c in this_round if state.passes[c].any())
    V_btw = state.E[state.node_measured].sum(axis=0)
    return {'round': round_, 'evals_tree': evals_tree, 'R': R, 'n_open': n_open, 'V_btw': V_btw}


def grow(state: TreeState, T, counter, pool, budget: int) -> Tuple[List[dict], str, int, int]:
    """Grow the tree in level-synchronous rounds (spec 5.2): round 0
    measures the root; each round opens its selected cell nodes,
    re-evaluates theta_Q once, then measures the selection in a batch.
    Returns (curve_rows, tree_status, n_rounds, evals_tree); the last
    counts node measurements only, not openings."""
    measure_nodes(state, T, counter, pool, [0], 0)
    evals_tree, n_rounds, round_ = 1, 1, 0
    row_ids, _, _ = active_rows(state)
    curve_rows = [_curve_row(0, evals_tree, len(row_ids), state, [0])]

    while True:
        candidates = _open_candidates(state, round_)
        remaining = budget - evals_tree
        if remaining <= 0:
            return curve_rows, 'budget', n_rounds, evals_tree
        if not candidates:
            return curve_rows, 'exhausted', n_rounds, evals_tree

        V_btw = curve_rows[-1]['V_btw']
        priority = [_priority(state.E[state.node_parent[c]], V_btw) for c in candidates]
        order = sorted(range(len(candidates)), key=lambda i: (-priority[i], candidates[i]))
        selected = [candidates[order[i]] for i in range(min(len(candidates), remaining))]

        cells = [int(state.node_cell[c]) for c in selected if state.node_cell[c] != -1]
        if cells:
            open_cells(state, cells)
            if not reevaluate_theta_Q(state, T, counter):
                return curve_rows, 'opening_failed', n_rounds, evals_tree

        round_ += 1
        measure_nodes(state, T, counter, pool, selected, round_)
        evals_tree += len(selected)
        n_rounds += 1
        row_ids, _, _ = active_rows(state)
        curve_rows.append(_curve_row(round_, evals_tree, len(row_ids), state, selected))
