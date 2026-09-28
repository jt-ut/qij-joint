"""The qijt tree (spec/QIJ_qijt_spec.md): the hierarchy and the evaluation rows (sections
2.1, 3), node measurement and level-synchronous growth (2.3, 4, 5, 9.2), the anchors, scale,
bias and drift (6), and the within-term and the per-point reconstruction (7, 8.1).
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, TYPE_CHECKING, Tuple
import numpy as np
import time

from ..parallel import call_T, fit_status
from .differences import step_parameter
from .xvq import _field_moments


@dataclass
class TreeState:
    X: np.ndarray
    Z: np.ndarray
    bmu: np.ndarray
    centers_x: np.ndarray
    centers_z: np.ndarray
    N: int
    M: int
    q_full: int
    measured: np.ndarray
    cell_order: np.ndarray
    cell_start: np.ndarray
    row_cap: int
    row_x: np.ndarray
    row_z: np.ndarray
    omega0: np.ndarray
    row_cell: np.ndarray
    row_point: np.ndarray
    row_active: np.ndarray
    n_rows: int
    cell_row: np.ndarray
    point_row: np.ndarray
    node_cap: int
    node_parent: np.ndarray
    node_child0: np.ndarray
    node_child1: np.ndarray
    node_depth: np.ndarray
    node_kind: np.ndarray
    node_cell: np.ndarray
    node_lo: np.ndarray
    node_hi: np.ndarray
    n_nodes: int
    cell_perm: np.ndarray
    point_perm: np.ndarray
    point_perm_used: int
    cell_node: np.ndarray
    node_measured: np.ndarray
    node_round: np.ndarray
    meas_child: np.ndarray
    node_t: np.ndarray
    node_status: np.ndarray
    U: np.ndarray
    y: np.ndarray
    s: np.ndarray
    E: np.ndarray
    passes: np.ndarray
    open_shift: np.ndarray
    theta_hat: np.ndarray
    theta_Q: np.ndarray
    eta_full: float
    eta_Q: float
    delta_f: float
    z_n: float


def bisect(z: np.ndarray, m: np.ndarray) -> np.ndarray:
    """The one bisection rule (spec 3.1): seeds at the extreme
    projections onto S's leading eigenvector, then Lloyd (Euclidean in
    Z, ties to the first seed, <=100 rounds). Returns `first` (n,) bool,
    True for the members joining the child of the first (min-projection)
    seed. Degenerate handling (S=0) is the caller's (spec 3.4)."""
    mu = (m[:, None] * z).sum(axis=0) / m.sum()
    d = z - mu
    S = np.einsum('i,ia,ib->ab', m, d, d) / m.sum()
    v = np.linalg.eigh(S)[1][:, -1]
    proj = d @ v
    c0, c1 = z[int(np.argmin(proj))].copy(), z[int(np.argmax(proj))].copy()
    assign = None
    for _ in range(100):
        near0 = np.sum((z - c0) ** 2, axis=1) <= np.sum((z - c1) ** 2, axis=1)
        if assign is not None and np.array_equal(near0, assign):
            break
        assign = near0
        w0, w1 = m * assign, m * (~assign)
        if w0.sum() > 0:
            c0 = (w0[:, None] * z).sum(axis=0) / w0.sum()
        if w1.sum() > 0:
            c1 = (w1[:, None] * z).sum(axis=0) / w1.sum()
    return assign


def _cell_mass(state: TreeState) -> np.ndarray:
    """p_j = n_j / N for every cell (spec 3.2), from `cell_start`."""
    return np.diff(state.cell_start).astype(float) / state.N


def _members_zm(state: TreeState, c: int):
    """Node c's member positions (in Z) and masses: cells above the
    cell level, native points below it (spec 3.1-3.3)."""
    lo, hi = int(state.node_lo[c]), int(state.node_hi[c])
    if state.node_kind[c] == 0:
        members = state.cell_perm[lo:hi]
        return state.centers_z[members], _cell_mass(state)[members]
    members = state.point_perm[lo:hi]
    return state.Z[members], np.full(members.shape[0], 1.0 / state.N)


def _single_cell_points(state: TreeState, c: int):
    """The native points' Z positions of a single-cell node, or None for
    any other node. Opening such a node makes these its members, so they
    decide whether it can split."""
    lo, hi = int(state.node_lo[c]), int(state.node_hi[c])
    if state.node_kind[c] != 0 or hi - lo != 1:
        return None
    j = int(state.cell_perm[lo])
    return state.Z[state.cell_order[state.cell_start[j]:state.cell_start[j + 1]]]


def is_degenerate(state: TreeState, c: int) -> bool:
    """True iff c's members (a single cell's native points, for a cell
    node) number >= 2 and all sit at one Z position (S = 0, spec 3.4)."""
    z = _single_cell_points(state, c)
    if z is None:
        if int(state.node_hi[c]) - int(state.node_lo[c]) <= 1:
            return False
        z, _ = _members_zm(state, c)
    return bool(z.shape[0] >= 2 and np.all(z == z[0]))


def is_point_leaf(state: TreeState, c: int) -> bool:
    """A node of one native point (spec 3.3), including a single-cell
    node whose cell holds one point."""
    z = _single_cell_points(state, c)
    if z is not None:
        return bool(z.shape[0] == 1)
    return bool(state.node_kind[c] == 1 and state.node_hi[c] - state.node_lo[c] == 1)


def node_mass(state: TreeState, c: int) -> float:
    """p of node c: sum of member cell masses, or point count / N."""
    _, m = _members_zm(state, c)
    return float(m.sum())


def measured_child(state: TreeState, c: int):
    """A, B: c's children ordered by mass, ties to the first child (spec 3.5)."""
    a0, a1 = int(state.node_child0[c]), int(state.node_child1[c])
    return (a0, a1) if node_mass(state, a0) >= node_mass(state, a1) else (a1, a0)


def _split_node(state: TreeState, c: int) -> bool:
    """Bisects node c in place if it has >= 2 members and is not
    degenerate, allocating two new nodes at the next free ids."""
    lo, hi = int(state.node_lo[c]), int(state.node_hi[c])
    if hi - lo <= 1 or is_degenerate(state, c):
        return False
    kind = int(state.node_kind[c])
    perm = state.cell_perm if kind == 0 else state.point_perm
    members = perm[lo:hi]
    z, m = _members_zm(state, c)
    first = bisect(z, m)
    perm[lo:hi] = np.concatenate([members[first], members[~first]])
    mid = lo + int(first.sum())
    i0, i1 = state.n_nodes, state.n_nodes + 1
    state.n_nodes += 2
    state.node_parent[i0] = state.node_parent[i1] = c
    state.node_child0[c], state.node_child1[c] = i0, i1
    state.node_depth[i0] = state.node_depth[i1] = state.node_depth[c] + 1
    state.node_kind[i0] = state.node_kind[i1] = kind
    state.node_cell[i0] = state.node_cell[i1] = state.node_cell[c]
    state.node_lo[i0], state.node_hi[i0] = lo, mid
    state.node_lo[i1], state.node_hi[i1] = mid, hi
    return True


def _bfs_split(state: TreeState, roots) -> None:
    """Recursive bisection from `roots`, breadth-first: ids are handed
    out in split order, which a FIFO queue makes breadth-first exactly
    (spec 3.2, 3.3)."""
    queue = deque(int(r) for r in roots)
    while queue:
        c = queue.popleft()
        if _split_node(state, c):
            queue.append(int(state.node_child0[c]))
            queue.append(int(state.node_child1[c]))


def node_rows(state: TreeState, c: int) -> np.ndarray:
    """Active row ids of node c's members, ascending: a cell's
    prototype row, or its native points' rows once opened (spec 4.1)."""
    lo, hi = int(state.node_lo[c]), int(state.node_hi[c])
    if state.node_kind[c] == 1:
        rows = state.point_row[state.point_perm[lo:hi]]
    else:
        cells = state.cell_perm[lo:hi]
        proto = state.cell_row[cells]
        rows = proto[proto >= 0]
        opened = cells[proto < 0]
        if opened.size:
            pts = np.nonzero(np.isin(state.bmu, opened))[0]
            rows = np.concatenate([rows, state.point_row[pts]])
    return np.sort(rows).astype(int)


def open_cells(state: TreeState, cells) -> None:
    """Spec 5.3's row part and 3.3's within-cell tree, per cell: drop
    the prototype row, append the cell's native points at omega0 = 1,
    then bisect them to single points (ids continuing breadth-first
    from the current maximum, spec 3.3). Does not re-evaluate theta_Q."""
    for j in cells:
        j = int(j)
        state.row_active[state.cell_row[j]] = False
        pts = state.cell_order[state.cell_start[j]:state.cell_start[j + 1]]
        n_j = pts.size
        r0, r1 = state.n_rows, state.n_rows + n_j
        state.row_x[r0:r1] = state.X[pts]
        state.row_z[r0:r1] = state.Z[pts]
        state.omega0[r0:r1] = 1.0
        state.row_cell[r0:r1] = j
        state.row_point[r0:r1] = pts
        state.row_active[r0:r1] = True
        state.point_row[pts] = np.arange(r0, r1)
        state.n_rows = r1
        state.cell_row[j] = -1
        c = int(state.cell_node[j])
        p0, p1 = state.point_perm_used, state.point_perm_used + n_j
        state.point_perm[p0:p1] = pts
        state.point_perm_used = p1
        state.node_kind[c] = 1
        state.node_cell[c] = j
        state.node_lo[c], state.node_hi[c] = p0, p1
        _bfs_split(state, [c])


def active_rows(state: TreeState):
    """The evaluation set: active rows in ascending row id (spec 2.1).
    Returns (row ids, native-coordinate rows, base weights)."""
    ids = np.nonzero(state.row_active[:state.n_rows])[0]
    return ids, state.row_x[ids], state.omega0[ids]


def build_state(X, Z, bmu, centers_x, centers_z, measured, q_full, node_cap) -> TreeState:
    """The initial rows (one per cell, omega0 = n_j) and the cell-level
    tree, built once (spec 2.1, 3.2), root id 0, breadth-first ids."""
    N = X.shape[0]
    d = Z.shape[1]
    M = centers_x.shape[0]
    bmu = np.asarray(bmu, dtype=int)
    measured = np.asarray(measured, dtype=int)
    q = measured.size
    row_cap = M + N
    order = np.argsort(bmu, kind='stable')
    cell_start = np.searchsorted(bmu[order], np.arange(M + 1)).astype(int)
    n_j = np.diff(cell_start).astype(float)
    # Rows keep T's own native shape ((N,) data stays 1-D).
    row_x = np.zeros((row_cap,) + X.shape[1:]); row_x[:M] = centers_x
    row_z = np.zeros((row_cap, d)); row_z[:M] = centers_z
    omega0 = np.zeros(row_cap); omega0[:M] = n_j
    row_cell = np.full(row_cap, -1, dtype=int); row_cell[:M] = np.arange(M)
    row_point = np.full(row_cap, -1, dtype=int)
    row_active = np.zeros(row_cap, dtype=bool); row_active[:M] = True
    U = np.full((node_cap, q_full), np.nan)
    U[0] = 0.0
    state = TreeState(
        X=X, Z=Z, bmu=bmu, centers_x=centers_x, centers_z=centers_z,
        N=N, M=M, q_full=q_full, measured=measured,
        cell_order=order.astype(int), cell_start=cell_start,
        row_cap=row_cap, row_x=row_x, row_z=row_z, omega0=omega0,
        row_cell=row_cell, row_point=row_point, row_active=row_active,
        n_rows=M, cell_row=np.arange(M, dtype=int), point_row=np.full(N, -1, dtype=int),
        node_cap=node_cap,
        node_parent=np.full(node_cap, -1, dtype=int),
        node_child0=np.full(node_cap, -1, dtype=int),
        node_child1=np.full(node_cap, -1, dtype=int),
        node_depth=np.zeros(node_cap, dtype=int),
        node_kind=np.zeros(node_cap, dtype=int),
        node_cell=np.full(node_cap, -1, dtype=int),
        node_lo=np.zeros(node_cap, dtype=int), node_hi=np.zeros(node_cap, dtype=int),
        n_nodes=1, cell_perm=np.arange(M, dtype=int), point_perm=np.zeros(N, dtype=int),
        point_perm_used=0, cell_node=np.full(M, -1, dtype=int),
        node_measured=np.zeros(node_cap, dtype=bool),
        node_round=np.full(node_cap, -1, dtype=int),
        meas_child=np.zeros(node_cap, dtype=int),
        node_t=np.full(node_cap, np.nan),
        node_status=np.full(node_cap, '', dtype=object),
        U=U, y=np.full((node_cap, q), np.nan), s=np.full((node_cap, q), np.nan),
        E=np.full((node_cap, q), np.nan), passes=np.zeros((node_cap, q), dtype=bool),
        open_shift=np.full((node_cap, q), np.nan),
        theta_hat=np.full(q_full, np.nan), theta_Q=np.full(q_full, np.nan),
        eta_full=float('nan'), eta_Q=float('nan'), delta_f=float('nan'), z_n=5.0,
    )
    state.node_hi[0] = M
    _bfs_split(state, [0])
    kind0 = state.node_kind[:state.n_nodes] == 0
    width = state.node_hi[:state.n_nodes] - state.node_lo[:state.n_nodes]
    leaves = np.nonzero(kind0 & (width == 1))[0]
    state.cell_node[state.cell_perm[state.node_lo[leaves]]] = leaves
    state.node_cell[leaves] = state.cell_perm[state.node_lo[leaves]]
    return state


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


def remeasure_opened(state: TreeState, T, counter, pool, cells: Sequence[int]) -> None:
    """Each opened cell's U re-measured on the new rows (spec 5.3): one
    evaluation per cell along its native points, base the new theta_Q.
    The new U replaces the parent-derived one for everything below the
    cell; the shift is recorded. A failed task (counted by run_batch's
    counter) leaves its cell's old U."""
    row_ids, rows_x, omega0 = active_rows(state)
    tasks, info = [], []
    for j in cells:
        X_node = int(state.cell_node[j])
        p_X = node_mass(state, X_node)
        t = step_parameter(state.delta_f, p_X)
        pos = np.searchsorted(row_ids, node_rows(state, X_node))
        tasks.append((X_node, set_weights(omega0, pos, p_X, t), state.theta_Q, state.eta_full))
        info.append((X_node, t))
    for (_, value, failed, _status, _wall), (X_node, t) in zip(
        run_batch(T, counter, pool, rows_x, tasks), info,
    ):
        if failed:
            continue
        U_new = (value - state.theta_Q) / t
        state.open_shift[X_node] = (U_new - state.U[X_node])[state.measured]
        state.U[X_node] = U_new


def leaf_V_btw(state: TreeState) -> np.ndarray:
    """(1/N) sum over the leaves of p_l U_l^2, leaves in ascending id
    (spec 8.2): the between-leaf variance from the leaf means, unscaled."""
    ids = leaves(state)
    p = np.array([node_mass(state, c) for c in ids])
    return (p[:, None] * state.U[ids][:, state.measured] ** 2).sum(axis=0) / state.N


def _curve_row(round_: int, evals_tree: int, R: int, state: TreeState, this_round: Sequence[int]) -> dict:
    """One curve row (spec 11.4): V_btw is the leaf sum of 8.2 for the
    leaves as they stand after the round."""
    n_open = sum(1 for c in this_round if state.passes[c].any())
    V_btw = leaf_V_btw(state)
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

        # Only still-unopened cell nodes open; within-cell nodes carry
        # their cell's id too.
        cells = [int(state.node_cell[c]) for c in selected
                 if state.node_kind[c] == 0 and state.node_cell[c] != -1
                 and state.cell_row[state.node_cell[c]] >= 0]
        if cells:
            open_cells(state, cells)
            if not reevaluate_theta_Q(state, T, counter):
                return curve_rows, 'opening_failed', n_rounds, evals_tree
            remeasure_opened(state, T, counter, pool, cells)

        round_ += 1
        measure_nodes(state, T, counter, pool, selected, round_)
        evals_tree += len(selected)
        n_rounds += 1
        row_ids, _, _ = active_rows(state)
        curve_rows.append(_curve_row(round_, evals_tree, len(row_ids), state, selected))


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


if TYPE_CHECKING:
    from ..parallel import Pool


def leaves(state: TreeState) -> np.ndarray:
    """Node ids whose parent is measured and that are not themselves
    measured (7.1), ascending; the root has no parent and is never one."""
    n = state.n_nodes
    parent = state.node_parent[:n]
    has_parent = parent >= 0
    parent_measured = np.zeros(n, dtype=bool)
    parent_measured[has_parent] = state.node_measured[parent[has_parent]]
    is_leaf = has_parent & parent_measured & ~state.node_measured[:n]
    return np.nonzero(is_leaf)[0]


def _sibling(state: TreeState, node: int) -> int:
    """The other child of `node`'s parent (7.1's ĝ needs both)."""
    p = state.node_parent[node]
    c0, c1 = state.node_child0[p], state.node_child1[p]
    return c1 if node == c0 else c0


def _frame(state: TreeState, node: int, row_z: np.ndarray, omega0: np.ndarray,
           pos: np.ndarray, fn: np.ndarray, fm: np.ndarray, fc: np.ndarray) -> Dict[str, Any]:
    """Mass, mean, and eigen-geometry of node c's members in Z (7.1):
    its rows as they stand, or -- for an unopened single cell, one row
    so far -- its native points via `_field_moments`."""
    d = state.Z.shape[1]
    cell = int(state.node_cell[node])
    single = state.node_kind[node] == 0 and cell != -1 and state.cell_row[cell] != -1
    p_l = node_mass(state, node)
    if single:
        n_l, mu, cov = float(fn[cell]), fm[cell], fc[cell]
        rows = np.array([state.cell_row[cell]])
    else:
        rows = node_rows(state, node)
        z = row_z[pos[rows]]
        m = omega0[pos[rows]] / state.N
        mu = (m @ z) / p_l
        diff = z - mu
        cov = (diff * m[:, None]).T @ diff / p_l
        n_l = len(rows)
    vals, vecs = np.linalg.eigh(cov)
    order = np.argsort(vals)[::-1]
    vals, vecs = vals[order], vecs[:, order]
    rank = int(np.sum(vals > d * vals[0] * np.finfo(float).eps)) if vals[0] > 0 else 0
    return dict(rows=rows, cell=cell, single=single, n=n_l, p=p_l, mu=mu,
                eigval=vals, eigvec=vecs, rank=rank)


def _pair_h(fr: Dict[str, Any], j: int, row_z: np.ndarray,
            pos: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """h_r for pair (leaf, j) (7.2): unit mass-weighted scatter along
    the leaf's j-th principal direction (1-indexed, j=1 largest)."""
    pos_l = pos[fr['rows']]
    v, lam = fr['eigvec'][:, j - 1], fr['eigval'][j - 1]
    return pos_l, (row_z[pos_l] - fr['mu']) @ v / np.sqrt(lam)


def _quad_h(fr: Dict[str, Any], row_z: np.ndarray, pos: np.ndarray,
            omega0: np.ndarray, N: int) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """h-tilde for the leaf's quadratic contrast (7.3): the Mahalanobis
    form under the rank-r pseudo-inverse of C_l (eigh, E3), centred and
    made orthogonal to every one of the leaf's r principal directions,
    then unit mass-weighted mean square. None when it is degenerate."""
    pos_l = pos[fr['rows']]
    z, m, p_l, r = row_z[pos_l], omega0[pos_l] / N, fr['p'], fr['rank']
    V, lam = fr['eigvec'][:, :r], fr['eigval'][:r]
    c_plus = (V / lam) @ V.T
    diff = z - fr['mu']
    m_vals = np.einsum('ij,jk,ik->i', diff, c_plus, diff)
    H = (diff @ V) / np.sqrt(lam)
    mean_m = (m @ m_vals) / p_l
    beta = (m * m_vals) @ H / p_l
    h_tilde = m_vals - mean_m - H @ beta
    scale = (m @ h_tilde ** 2) / p_l
    if scale <= 0.0:
        return pos_l, None
    return pos_l, h_tilde / np.sqrt(scale)


def _weight(omega0: np.ndarray, pos_l: np.ndarray, h: np.ndarray,
            delta_f: float) -> Tuple[np.ndarray, float]:
    """omega(t) for one contrast (7.2-7.3): t set so the leaf's own
    largest |h| takes the step delta_f; every other row is unchanged."""
    t = delta_f / np.max(np.abs(h))
    omega_t = omega0.copy()
    omega_t[pos_l] = omega0[pos_l] * (1.0 + t * h)
    return omega_t, t


def _response(state: TreeState, value: np.ndarray, failed: bool, t: float,
              p_l: float) -> Tuple[np.ndarray, np.ndarray]:
    """The response, noise and pass rule shared by a pair and a
    quadratic contrast (7.2-7.3): the unscaled contribution
    (1/N) resp^2 / p_l where |resp| clears z_n * s, else 0."""
    meas = state.measured
    resp = (value - state.theta_Q) / t
    s = np.sqrt(2.0) * state.eta_Q * np.abs(state.theta_Q[meas]) / t
    passed = (not failed) & (np.abs(resp[meas]) > state.z_n * s)
    contribution = np.where(passed, resp[meas] ** 2 / (state.N * p_l), 0.0)
    return resp, contribution, s, passed


def within(state: TreeState, T: Any, counter: Any, pool: Optional["Pool"],
           budget_win: int, budget_quad: int, V_btw_unscaled: np.ndarray) -> Dict[str, Any]:
    """Buy the top `budget_win` pair contrasts and `budget_quad`
    quadratic contrasts against the tree's current leaves (7.1-7.3)."""
    leaf_ids = leaves(state)
    row_ids, rows_x, omega0 = active_rows(state)
    row_z = state.row_z[row_ids]
    pos = np.full(state.n_rows, -1, dtype=int)
    pos[row_ids] = np.arange(len(row_ids))
    fn, fm, fc = _field_moments(state.Z, state.bmu, state.M)
    frames = {ell: _frame(state, ell, row_z, omega0, pos, fn, fm, fc) for ell in leaf_ids}

    meas = state.measured
    ghat = {}
    for ell in leaf_ids:
        sib = int(_sibling(state, ell))
        sib_fr = frames.get(sib) or _frame(state, sib, row_z, omega0, pos, fn, fm, fc)
        d_u = np.abs(state.U[ell, meas] - state.U[sib, meas])
        norm = np.linalg.norm(frames[ell]['mu'] - sib_fr['mu'])
        ghat[ell] = d_u / norm if norm > 0 else np.zeros_like(d_u)

    safe_v = np.where(V_btw_unscaled > 0, V_btw_unscaled, 1.0)
    scored = []
    for ell in leaf_ids:
        fr = frames[ell]
        if fr['rank'] == 0:
            continue
        term = np.where(V_btw_unscaled > 0, ghat[ell] ** 2 / (state.N * safe_v), 0.0)
        gmax = float(np.max(term))
        for j in range(1, min(fr['rank'], fr['n'] - 1) + 1):
            scored.append((fr['p'] * fr['eigval'][j - 1] * gmax, int(ell), j))
    scored.sort(key=lambda s: (-s[0], s[1], s[2]))
    bought = scored[:budget_win]

    opened = sorted({frames[ell]['cell'] for _, ell, _ in bought if frames[ell]['single']})
    if opened:
        open_cells(state, opened)
        if not reevaluate_theta_Q(state, T, counter):
            return dict(pairs=[], quads=[], opened=opened, leaf_rows=[], status='opening_failed')
        remeasure_opened(state, T, counter, pool, opened)
        row_ids, rows_x, omega0 = active_rows(state)
        row_z = state.row_z[row_ids]
        pos = np.full(state.n_rows, -1, dtype=int)
        pos[row_ids] = np.arange(len(row_ids))
        for ell in {ell for _, ell, _ in bought}:
            frames[ell] = _frame(state, ell, row_z, omega0, pos, fn, fm, fc)

    tasks, meta = [], []
    for _, ell, j in bought:
        pos_l, h = _pair_h(frames[ell], j, row_z, pos)
        omega_t, t = _weight(omega0, pos_l, h, state.delta_f)
        tasks.append(((ell, j), omega_t, state.theta_Q, state.eta_full))
        meta.append((ell, j, t))
    results = run_batch(T, counter, pool, rows_x, tasks) if tasks else []

    pairs: List[Dict[str, Any]] = []
    A: Dict[int, np.ndarray] = {}
    for (_, value, failed, status, _wall), (ell, j, t) in zip(results, meta):
        p_l = frames[ell]['p']
        D, W, s, passed = _response(state, value, failed, t, p_l)
        pairs.append(dict(leaf=ell, j=j, D=D, W=W, t=t, s=s, passed=passed, status=status))
        A[ell] = A.get(ell, np.zeros_like(V_btw_unscaled)) + W

    quad_scored = sorted(
        ((float(np.max(np.where(V_btw_unscaled > 0, A[ell] / safe_v, 0.0))), ell) for ell in A),
        key=lambda s: (-s[0], s[1]),
    )
    quad_leaves = [ell for _, ell in quad_scored[:budget_quad]]

    tasks, meta = [], []
    for ell in quad_leaves:
        pos_l, h = _quad_h(frames[ell], row_z, pos, omega0, state.N)
        if h is None:
            continue
        omega_t, t = _weight(omega0, pos_l, h, state.delta_f)
        tasks.append((ell, omega_t, state.theta_Q, state.eta_full))
        meta.append((ell, t))
    q_results = run_batch(T, counter, pool, rows_x, tasks) if tasks else []

    quads: List[Dict[str, Any]] = []
    for (_, value, failed, status, _wall), (ell, t) in zip(q_results, meta):
        p_l = frames[ell]['p']
        Qv, contribution, s, passed = _response(state, value, failed, t, p_l)
        quads.append(dict(leaf=ell, Q=Qv, contribution=contribution, t=t, s=s, passed=passed,
                          status=status))

    n_pairs: Dict[int, int] = {}
    for pr in pairs:
        n_pairs[pr['leaf']] = n_pairs.get(pr['leaf'], 0) + 1
    Q2 = {qd['leaf']: qd['contribution'] for qd in quads}
    leaf_rows = [dict(leaf=int(ell), n_points=frames[ell]['n'], p=frames[ell]['p'],
                       rank=frames[ell]['rank'], n_pairs_bought=n_pairs.get(ell, 0),
                       quad_bought=ell in Q2, U=state.U[ell, meas],
                       ghat=ghat[ell], A=A.get(ell, np.zeros_like(V_btw_unscaled)),
                       Q2=Q2.get(ell, np.zeros_like(V_btw_unscaled)))
                 for ell in leaf_ids]
    return dict(pairs=pairs, quads=quads, opened=list(opened), leaf_rows=leaf_rows)


def reconstruct(state: TreeState, pairs: List[Dict[str, Any]], quads: List[Dict[str, Any]],
                 a: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """psi per row and per native point (8.1): each leaf's own U plus
    its bought directions' response, scaled by the anchor `a`; a point
    in a still-unopened cell takes its cell's one row's value."""
    row_ids, rows_x, omega0 = active_rows(state)
    row_z = state.row_z[row_ids]
    R = len(row_ids)
    pos = np.full(state.n_rows, -1, dtype=int)
    pos[row_ids] = np.arange(R)
    fn, fm, fc = _field_moments(state.Z, state.bmu, state.M)
    meas = state.measured
    psi_rows = np.zeros((R, state.q_full))
    leaf_of_row = np.full(R, -1, dtype=int)
    frames: Dict[int, Dict[str, Any]] = {}
    for ell in leaves(state):
        fr = _frame(state, ell, row_z, omega0, pos, fn, fm, fc)
        frames[ell] = fr
        pos_l = pos[fr['rows']]
        leaf_of_row[pos_l] = ell
        psi_rows[np.ix_(pos_l, meas)] += a * state.U[ell, meas]
    for pr in pairs:
        fr = frames[pr['leaf']]
        pos_l, h = _pair_h(fr, pr['j'], row_z, pos)
        psi_rows[np.ix_(pos_l, meas)] += np.outer(h, np.where(pr['passed'], a * pr['D'][meas] / fr['p'], 0.0))
    for qd in quads:
        fr = frames[qd['leaf']]
        pos_l, h = _quad_h(fr, row_z, pos, omega0, state.N)
        psi_rows[np.ix_(pos_l, meas)] += np.outer(h, np.where(qd['passed'], a * qd['Q'][meas] / fr['p'], 0.0))

    point_to_row = np.full(state.N, -1, dtype=int)
    rp = state.row_point[row_ids]
    valid = rp >= 0
    point_to_row[rp[valid]] = row_ids[valid]
    opened_pt = state.cell_row[state.bmu] == -1
    point_row = np.where(opened_pt, point_to_row, state.cell_row[state.bmu])
    point_pos = pos[point_row]
    psi_points = psi_rows[point_pos][:, meas]
    leaf_of_point = leaf_of_row[point_pos]
    return psi_rows, psi_points, leaf_of_point
