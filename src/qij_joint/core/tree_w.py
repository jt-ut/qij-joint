"""The within-term contrasts and the per-point reconstruction (spec
QIJ_qijt_spec.md sections 7, 8.1): pair and quadratic contrasts bought
against the leaves of the measured tree, and psi_hat assembled from
them plus each leaf's own known U.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

import numpy as np

from .tree_g import reevaluate_theta_Q, run_batch
from .tree_h import TreeState, active_rows, node_mass, node_rows, open_cells
from .xvq import _field_moments

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
    return resp, contribution


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
        D, W = _response(state, value, failed, t, p_l)
        pairs.append(dict(leaf=ell, j=j, D=D, W=W, t=t, status=status))
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
        Qv, contribution = _response(state, value, failed, t, p_l)
        quads.append(dict(leaf=ell, Q=Qv, contribution=contribution, status=status))

    n_pairs: Dict[int, int] = {}
    for pr in pairs:
        n_pairs[pr['leaf']] = n_pairs.get(pr['leaf'], 0) + 1
    quad_bought = {qd['leaf'] for qd in quads}
    leaf_rows = [dict(leaf=int(ell), n_points=frames[ell]['n'], p=frames[ell]['p'],
                       rank=frames[ell]['rank'], n_pairs_bought=n_pairs.get(ell, 0),
                       quad_bought=ell in quad_bought, U=state.U[ell, meas],
                       ghat=ghat[ell], A=A.get(ell, np.zeros_like(V_btw_unscaled)))
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
        psi_rows[np.ix_(pos_l, meas)] += np.outer(h, a * pr['D'][meas] / fr['p'])
    for qd in quads:
        fr = frames[qd['leaf']]
        pos_l, h = _quad_h(fr, row_z, pos, omega0, state.N)
        psi_rows[np.ix_(pos_l, meas)] += np.outer(h, a * qd['Q'][meas] / fr['p'])

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
