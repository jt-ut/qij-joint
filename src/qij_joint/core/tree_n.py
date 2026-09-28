"""Anchors, bias and drift (spec/QIJ_qijt_spec.md section 6): the
full-data scale of the quantized tree's contrasts (6.1-6.2), the
root-children bias ingredient (6.4), and the final drift check (6.3).
"""

from __future__ import annotations

import numpy as np

from .differences import step_parameter
from .tree_g import measure_nodes, run_batch, set_weights
from .tree_h import active_rows, measured_child, node_mass, node_rows


def _anchor_nodes(state) -> list:
    """Root, its <=2 children and their <=4 children -- the cell-level
    tree's depth<=2 nodes that split (6.1); a node with no children is
    excluded, since it has no measured child to give it a y. BFS
    construction already orders these ascending; sorted for safety."""
    frontier = [0]
    found: list = []
    for _ in range(3):
        nxt = []
        for c in frontier:
            if state.node_child0[c] == -1:
                continue
            found.append(c)
            nxt.append(state.node_child0[c])
            nxt.append(state.node_child1[c])
        frontier = nxt
    return sorted(found)


def _full_member_mask(state, child: int) -> np.ndarray:
    """The native points of `child`'s cells (6.1's full-data member
    set): `child` is a cell-level node, its cells are `cell_perm`'s
    slice at (node_lo, node_hi)."""
    cells = state.cell_perm[state.node_lo[child]:state.node_hi[child]]
    return np.isin(state.bmu, cells)


def _batch_to_dict(results, q_full: int) -> dict:
    """(key, value, failed, status, wall) results, by key, a failed
    task's value replaced by a full-width NaN vector (10)."""
    out = {}
    for key, value, failed, _status, _wall in results:
        out[key] = np.full(q_full, np.nan) if failed else np.asarray(value, dtype=float)
    return out


def _scale(y_full: np.ndarray, y_Q: np.ndarray):
    """a_o, scatter_o (6.2) over anchor nodes with both y^full and y^Q
    finite; NaN where fewer than 2 such pairs on an output."""
    mask = np.isfinite(y_full) & np.isfinite(y_Q)
    count = mask.sum(axis=0)
    num = np.sum(np.where(mask, y_full * y_Q, 0.0), axis=0)
    den = np.sum(np.where(mask, y_Q ** 2, 0.0), axis=0)
    with np.errstate(divide='ignore', invalid='ignore'):
        a = num / den
    a = np.where((count >= 2) & (den > 0.0), a, np.nan)
    num2 = np.sum(np.where(mask, (y_full - a * y_Q) ** 2, 0.0), axis=0)
    den2 = np.sum(np.where(mask, y_full ** 2, 0.0), axis=0)
    with np.errstate(divide='ignore', invalid='ignore'):
        scatter = np.sqrt(num2 / den2)
    scatter = np.where(np.isfinite(a) & (den2 > 0.0), scatter, np.nan)
    return a, scatter


def _bias(state, T, counter, pool, info, full_raw) -> np.ndarray:
    """b_hat (6.4): the root's own second difference along each child's
    member set, reusing the root's anchor/half-step evaluations for the
    measured child A, two fresh full-data evaluations for the other
    child B (stage 'bias')."""
    q_meas = state.measured
    A, B, p_A, p_B, t_A = info[0]
    D2_A = (full_raw[('f', 0)] - 2.0 * full_raw[('h', 0)] + state.theta_hat) / (t_A / 2.0) ** 2

    t_B = step_parameter(state.delta_f, p_B)
    mask_B = _full_member_mask(state, B)
    ones_N = np.ones(state.N)
    tasks = [('f', set_weights(ones_N, mask_B, p_B, t_B), state.theta_hat, state.eta_full),
             ('h', set_weights(ones_N, mask_B, p_B, t_B / 2.0), state.theta_hat, state.eta_full)]
    raw = _batch_to_dict(run_batch(T, counter, pool, state.X, tasks), state.q_full)
    D2_B = (raw['f'] - 2.0 * raw['h'] + state.theta_hat) / (t_B / 2.0) ** 2

    return (p_A * D2_A[q_meas] + p_B * D2_B[q_meas]) / (2.0 * state.N)


def anchors(state, T, counter, pool) -> dict:
    """Full-data and quantized contrasts at the cell-level tree's
    depth<=2 nodes, giving a (q,) the scale, scatter (q,), b_hat (q,),
    the count of extra quantized evaluations spent, and the anchors
    table (spec 6.1, 6.2, 6.4; 11.5). One `pool.map` call shares `X`
    for the full-data batch, another for the bias batch; an anchor
    node the tree never measured is measured on the current rows by
    `measure_nodes`, depth by depth so a node's parent is always
    measured (its U known, 4.1) before the node itself is."""
    nodes = _anchor_nodes(state)
    q_meas = state.measured

    info = {}
    full_tasks = []
    for c in nodes:
        A, B = measured_child(state, c)
        p_A, p_B = node_mass(state, A), node_mass(state, B)
        t_A = step_parameter(state.delta_f, p_A)
        info[c] = (A, B, p_A, p_B, t_A)
        mask = _full_member_mask(state, A)
        full_tasks.append((('f', c), set_weights(np.ones(state.N), mask, p_A, t_A),
                            state.theta_hat, state.eta_full))
        full_tasks.append((('h', c), set_weights(np.ones(state.N), mask, p_A, t_A / 2.0),
                            state.theta_hat, state.eta_full))
    full_raw = _batch_to_dict(run_batch(T, counter, pool, state.X, full_tasks), state.q_full)

    evals_anchor_Q = 0
    for depth in (1, 2):
        todo = [c for c in nodes if state.node_depth[c] == depth and not state.node_measured[c]]
        if todo:
            measure_nodes(state, T, counter, pool, todo, round_=-1)
            evals_anchor_Q += len(todo)

    # Top-down (ascending id) cascade: a child's full/half U is only
    # known once its parent's own contrast is derived; the quantized
    # y is already in state.y, every anchor node now measured.
    U_full, U_half = {0: np.zeros(len(q_meas))}, {0: np.zeros(len(q_meas))}
    table, y_full_all, y_Q_all = [], [], []
    for c in nodes:
        A, B, p_A, p_B, t_A = info[c]
        p_c = p_A + p_B
        U_A_f = (full_raw[('f', c)][q_meas] - state.theta_hat[q_meas]) / t_A
        U_A_h = (full_raw[('h', c)][q_meas] - state.theta_hat[q_meas]) / (t_A / 2.0)
        U_B_f = (p_c * U_full[c] - p_A * U_A_f) / p_B
        U_B_h = (p_c * U_half[c] - p_A * U_A_h) / p_B
        y_full, y_half = U_A_f - U_B_f, U_A_h - U_B_h
        y_Q = np.asarray(state.y[c, :], dtype=float)
        U_full[A], U_full[B] = U_A_f, U_B_f
        U_half[A], U_half[B] = U_A_h, U_B_h

        with np.errstate(divide='ignore', invalid='ignore'):
            step_ratio = y_full / y_half
        table.append(dict(node=c, depth=int(state.node_depth[c]), p_A=p_A,
                           y_full=y_full, y_half=y_half, y_Q=y_Q, step_ratio=step_ratio))
        y_full_all.append(y_full)
        y_Q_all.append(y_Q)

    q = len(q_meas)
    a, scatter = _scale(np.array(y_full_all) if y_full_all else np.empty((0, q)),
                         np.array(y_Q_all) if y_Q_all else np.empty((0, q)))
    b_hat = _bias(state, T, counter, pool, info, full_raw)
    return dict(a=a, scatter=scatter, table=table, b_hat=b_hat, evals_anchor_Q=evals_anchor_Q)


def drift(state, T, counter, pool) -> np.ndarray:
    """root_drift (6.3): one quantized re-measurement of the root's
    contrast on the final rows, against its initial y (`state.y[0]`)."""
    q_meas = state.measured
    A, B = measured_child(state, 0)
    p_A, p_B = node_mass(state, A), node_mass(state, B)
    t_A = float(state.node_t[0])
    row_ids, rows_x, omega0 = active_rows(state)
    mask = np.isin(row_ids, node_rows(state, A))
    omega = set_weights(omega0, mask, p_A, t_A)
    results = run_batch(T, counter, pool, rows_x, [(0, omega, state.theta_Q, state.eta_full)])
    _key, value, failed, _status, _wall = results[0]
    if failed:
        return np.full(len(q_meas), np.nan)
    U_A = (np.asarray(value, dtype=float)[q_meas] - state.theta_Q[q_meas]) / t_A
    U_B = -(p_A / p_B) * U_A
    y_final = U_A - U_B
    with np.errstate(divide='ignore', invalid='ignore'):
        return y_final / state.y[0, :]
