"""
The hierarchy and the rows (spec/QIJ_qijt_spec.md 2.1, 3, 5.3's row and
within-cell-tree part). `TreeState` is the shared state every other
`tree_*` part reads and mutates; `build_state` builds the cell-level
tree once, `open_cells` grows a within-cell tree when a cell is opened.
No estimator evaluation happens here -- theta_hat/theta_Q/eta_full/
eta_Q/delta_f are allocated as NaN placeholders and filled by the
caller once known.

Naming note (flagged to the coordinator): the interfaces file names
BOTH the per-draw (q,) output-index array and the per-node measured-flag
array `measured`. `build_state`'s own parameter list fixes the former as
`state.measured`; the per-node flag is kept here as `state.node_measured`
to avoid the collision.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np


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


def is_degenerate(state: TreeState, c: int) -> bool:
    """True iff c has >= 2 members and they all sit at one Z position
    (S = 0, spec 3.4). A node of <= 1 member is a terminal node (a cell
    node awaiting opening, or a point leaf), not a degenerate split."""
    if int(state.node_hi[c]) - int(state.node_lo[c]) <= 1:
        return False
    z, _ = _members_zm(state, c)
    return bool(np.all(z == z[0]))


def is_point_leaf(state: TreeState, c: int) -> bool:
    """A node of one native point (spec 3.3)."""
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
    N, dx = X.shape
    d = Z.shape[1]
    M = centers_x.shape[0]
    bmu = np.asarray(bmu, dtype=int)
    measured = np.asarray(measured, dtype=int)
    q = measured.size
    row_cap = M + N
    order = np.argsort(bmu, kind='stable')
    cell_start = np.searchsorted(bmu[order], np.arange(M + 1)).astype(int)
    n_j = np.diff(cell_start).astype(float)
    row_x = np.zeros((row_cap, dx)); row_x[:M] = centers_x
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
        theta_hat=np.full(q_full, np.nan), theta_Q=np.full(q_full, np.nan),
        eta_full=float('nan'), eta_Q=float('nan'), delta_f=float('nan'), z_n=5.0,
    )
    state.node_hi[0] = M
    _bfs_split(state, [0])
    kind0 = state.node_kind[:state.n_nodes] == 0
    width = state.node_hi[:state.n_nodes] - state.node_lo[:state.n_nodes]
    leaves = np.nonzero(kind0 & (width == 1))[0]
    state.cell_node[state.cell_perm[state.node_lo[leaves]]] = leaves
    return state
