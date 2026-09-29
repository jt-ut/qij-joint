"""`QIJDT`: the data-tree method (spec/QIJ_qijdt_spec.md). Every
evaluation is on the full data from theta_hat -- no X-VQ, no quantized
rows, no openings, no anchors, no scale. `QIJDT(eps=0.01,
vq_transform=None).fit(X, T, pool=None)` runs the stage sequence (spec
9.1): full_fit, eta_full, tree, bias, within, quadratic, curvature,
assembly, returning a `QIJDTResult`.

The hierarchy is a LAZY point-level bisection in Z (spec 3.2): a node's
children are built only when first selected for measurement, so only
the measured part of the tree ever exists. `_init_state` builds a
root-only `core.tree.TreeState` (every node "point" kind, unit row
weights) so `core.tree`'s bisection, node-splitting, leaf-geometry and
contrast-direction primitives are reused unmodified -- `core.tree.py`
itself carries only two new, purely additive primitives: `fixed_point_nu`
(2.1's absolute noise needs the fixed point's final pair, which
`fixed_point`'s relative-residual-only return does not carry) and
`step_verify` (4.3's batched halving loop, shared by every contrast type
here). Node measurement (4.1-4.3), the round loop (5.2), the within-
term's buying (7) and the per-point reconstruction (8.1) have no qijt
analogue (qijt's own versions are budget- and cell/opening-based, with
no step verification) and so are written here rather than imported.
"""
from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from .core.abc import abc_interval as _abc_interval
from .core.abc import curvature as _curvature
from .core.counter import Counter
from .core.differences import central_step, difference, forward_step, step_parameter
from .core.eta import measure_eta_full
from .core.tree import (TreeState, active_rows, candidate_priority, frame, fixed_point_nu,
                        is_degenerate, is_point_leaf, leaf_V_btw, leaves, measured_child,
                        node_mass, node_rows, pair_h, quad_h, run_batch, set_weights, sibling,
                        split_node, step_verify)
from .core.xvq import SurveyRows, cost_rule_M
from .qij import _wrap
from .result import _normal_interval

_STAGES = ('full_fit', 'eta_full', 'tree', 'bias', 'within', 'quadratic', 'curvature')
_LEVEL = 0.95


@contextmanager
def _stage(evals: dict, wall: dict, counter, name: str):
    """Time one named stage and add its own evaluation count to `evals`/
    `wall` (spec 9.1); `+=` so a stage timed in two pieces (full_fit's
    base fit and its fixed point) accumulates correctly."""
    t0 = time.perf_counter()
    ev0, _ = counter.snapshot()
    yield
    wall[name] += time.perf_counter() - t0
    ev1, _ = counter.snapshot()
    evals[name] += ev1 - ev0


@dataclass
class QIJDTResult:
    """One qijdt draw's result (spec 11.1; per-output fields are (q,)
    arrays ordered like `outputs`)."""

    outputs: Tuple[str, ...]
    measured: np.ndarray
    N: int
    eps: float
    M_floor: int
    workers: int
    status: str
    theta_hat_status: str
    eta_full: float
    eta_full_failed: bool
    n_fp: int
    r_fp: float
    n_rounds: int
    n_measured: int
    n_leaves: int
    max_depth: int
    n_failed: int
    n_nonsmooth: int
    n_halvings_total: int
    tree_status: str
    n_pairs_bought: int
    n_quad_bought: int
    n_below_closed: int
    busy_time: float
    wall_time: float
    evals_by_stage: Dict[str, int]
    wall_time_by_stage: Dict[str, float]
    theta_hat: np.ndarray            # (q_full,)
    nu: np.ndarray                    # (q,) spec 2.1
    V_btw: np.ndarray                # (q,)
    V_win: np.ndarray                # (q,)
    V_tot: np.ndarray                # (q,)
    accel: np.ndarray                # (q,) spec 8.2
    b_hat: np.ndarray                # (q,) spec 6
    c_q: np.ndarray                  # (q,) spec 8.3
    P_unbought: np.ndarray           # (q,) spec 11.1 remainder diagnostic
    nodes: pd.DataFrame
    leaves: pd.DataFrame
    pairs: pd.DataFrame
    curve: pd.DataFrame
    psi_hat: np.ndarray               # (N, q)
    leaf_of_point: np.ndarray         # (N,)

    @property
    def variance(self) -> np.ndarray:
        """V_tot per output (spec 8.4)."""
        return self.V_tot

    def interval(self, level: float) -> np.ndarray:
        """(q, 2): the normal interval on V_tot (spec 8.4)."""
        return _normal_interval(self.theta_hat[self.measured], self.V_tot, level)

    def interval_btw(self, level: float) -> np.ndarray:
        """(q, 2): the same normal interval on V_btw alone (spec 8.4)."""
        return _normal_interval(self.theta_hat[self.measured], self.V_btw, level)

    def abc_interval(self, level: float) -> np.ndarray:
        """(q, 2): the ABC_q interval (spec 8.4) from `accel`/`b_hat`/`c_q`
        and sigma = sqrt(V_tot)."""
        sigma = np.sqrt(np.maximum(self.V_tot, 0.0))
        return _abc_interval(self.theta_hat[self.measured], sigma, self.accel,
                              self.b_hat, self.c_q, level)


def _nan_result(outputs, measured, N, eps, M_floor, workers, status, theta_hat_status, eta_full,
                 eta_full_failed, evals, wall, t_start, counter, theta_hat,
                 n_fp=0, r_fp=0.0) -> QIJDTResult:
    """A failed draw (spec 10): NaN variances/intervals, the status,
    whatever stage counts were made before the failure, empty tables."""
    q = len(outputs)
    nan_q = np.full(q, np.nan)
    wall_time = time.perf_counter() - t_start
    return QIJDTResult(
        outputs=outputs, measured=np.asarray(measured, dtype=int), N=N, eps=eps,
        M_floor=int(M_floor), workers=workers, status=status, theta_hat_status=theta_hat_status,
        eta_full=float(eta_full), eta_full_failed=bool(eta_full_failed),
        n_fp=int(n_fp), r_fp=float(r_fp), n_rounds=0, n_measured=0, n_leaves=0,
        max_depth=0, n_failed=int(counter.failed), n_nonsmooth=0, n_halvings_total=0,
        tree_status='', n_pairs_bought=0, n_quad_bought=0, n_below_closed=0,
        busy_time=float(wall_time), wall_time=float(wall_time),
        evals_by_stage=evals, wall_time_by_stage=wall,
        theta_hat=np.asarray(theta_hat, dtype=float), nu=nan_q, V_btw=nan_q, V_win=nan_q,
        V_tot=nan_q, accel=nan_q, b_hat=nan_q, c_q=nan_q, P_unbought=nan_q,
        nodes=pd.DataFrame(), leaves=pd.DataFrame(), pairs=pd.DataFrame(), curve=pd.DataFrame(),
        psi_hat=np.full((N, q), np.nan), leaf_of_point=np.full(N, -1, dtype=int),
    )


def _init_state(X, Z, theta_hat, eta_full, eta_f, measured, q_full, node_cap) -> TreeState:
    """The root-only `TreeState` for qijdt's lazy point tree (spec 3.2):
    one point-kind node (id 0) spanning all N points, unit row weights,
    no cells. `theta_Q`/`eta_Q` are set to `theta_hat`/`eta_full` (no
    quantization stage exists) purely so the dataclass is satisfied;
    qijdt's own node/pair/quadratic measurement (below) reads
    `state.theta_hat`/`state.eta_full` directly and never `theta_Q`. The
    dummy cell/opening fields are never read: node_kind is 1 (point)
    everywhere, so every `core.tree` branch keyed on kind == 0 is dead
    code for this state."""
    N, d = X.shape[0], Z.shape[1]
    q = len(measured)
    U = np.full((node_cap, q_full), np.nan)
    U[0] = 0.0
    node_hi = np.zeros(node_cap, dtype=int)
    node_hi[0] = N
    return TreeState(
        X=X, Z=Z, bmu=np.zeros(N, dtype=int), centers_x=np.zeros((0,) + X.shape[1:]),
        centers_z=np.zeros((0, d)), N=N, M=0, q_full=q_full, measured=measured,
        cell_order=np.zeros(0, dtype=int), cell_start=np.zeros(1, dtype=int),
        row_cap=N, row_x=X, row_z=Z, omega0=np.ones(N),
        row_cell=np.full(N, -1, dtype=int), row_point=np.arange(N),
        row_active=np.ones(N, dtype=bool), n_rows=N,
        cell_row=np.zeros(0, dtype=int), point_row=np.arange(N),
        node_cap=node_cap, node_parent=np.full(node_cap, -1, dtype=int),
        node_child0=np.full(node_cap, -1, dtype=int), node_child1=np.full(node_cap, -1, dtype=int),
        node_depth=np.zeros(node_cap, dtype=int), node_kind=np.ones(node_cap, dtype=int),
        node_cell=np.full(node_cap, -1, dtype=int), node_lo=np.zeros(node_cap, dtype=int),
        node_hi=node_hi, n_nodes=1, cell_perm=np.zeros(0, dtype=int), point_perm=np.arange(N),
        point_perm_used=N, cell_node=np.zeros(0, dtype=int),
        node_measured=np.zeros(node_cap, dtype=bool), node_round=np.full(node_cap, -1, dtype=int),
        meas_child=np.zeros(node_cap, dtype=int), node_t=np.full(node_cap, np.nan),
        node_status=np.full(node_cap, '', dtype=object),
        U=U, y=np.full((node_cap, q), np.nan), s=np.full((node_cap, q), np.nan),
        E=np.full((node_cap, q), np.nan), passes=np.zeros((node_cap, q), dtype=bool),
        open_shift=np.full((node_cap, q), np.nan),
        theta_hat=theta_hat, theta_Q=theta_hat, eta_full=eta_full, eta_Q=eta_full,
        delta_f=forward_step(eta_f), z_n=5.0, n_fp_opening=0, r_fp_opening_max=0.0,
    )


def _omega_h(omega0: np.ndarray, pos_l: np.ndarray, h: np.ndarray, t: float) -> np.ndarray:
    """omega(t) along h at positions pos_l (spec 7.2/7.3's omega_i(t) =
    1 + t*h_i), generalized to the arbitrary t values 4.3's halving
    needs beyond the initial delta_f-set step."""
    omega_t = omega0.copy()
    omega_t[pos_l] = omega0[pos_l] * (1.0 + t * h)
    return omega_t


def _measure_nodes_verified(state: TreeState, T, counter, pool, nodes, round_: int,
                             nu_full: np.ndarray, n_halv: np.ndarray, nonsmooth: np.ndarray
                             ) -> None:
    """Measure each node in `nodes` (spec 4.1, 4.3): the step-verified
    forward difference of U_A along its measured child, then U_B, y_c,
    E_c and the pass rule (4.2) at the accepted step. `n_halv`/
    `nonsmooth` are (node_cap,) arrays, written in place for the nodes
    table (11.2)."""
    row_ids, rows_x, omega0 = active_rows(state)
    specs, info = {}, {}
    for c in nodes:
        A, B = measured_child(state, c)
        p_A, p_B = node_mass(state, A), node_mass(state, B)
        t_A = step_parameter(state.delta_f, p_A)
        pos_A = np.searchsorted(row_ids, node_rows(state, A))
        specs[c] = dict(t=t_A,
                         omega=lambda t, pos_A=pos_A, p_A=p_A: set_weights(omega0, pos_A, p_A, t))
        info[c] = (A, B, p_A, p_B)
    verified = step_verify(specs, T, counter, pool, rows_x, state.theta_hat, state.eta_full,
                            nu_full, state.measured, state.z_n)
    meas = state.measured
    for c in nodes:
        v, (A, B, p_A, p_B) = verified[c], info[c]
        p_c = p_A + p_B
        if v['failed']:
            U_A_full = np.full(state.q_full, np.nan)
            U_B_full = np.full(state.q_full, np.nan)
            y = s = np.full(meas.size, np.nan)
            E = np.zeros(meas.size)
            passes = np.zeros(meas.size, dtype=bool)
        else:
            U_A_full = v['U']
            U_B_full = (p_c * state.U[c] - p_A * U_A_full) / p_B
            y = (U_A_full - U_B_full)[meas]
            s = np.sqrt(2.0) * nu_full[meas] / v['t'] * (p_c / p_B)
            passes = np.abs(y) > state.z_n * s
            E = (1.0 / state.N) * (p_A * p_B / p_c) * y ** 2 * passes
            n_halv[c], nonsmooth[c] = v['n_halvings'], v['nonsmooth']
        state.U[A], state.U[B] = U_A_full, U_B_full
        state.y[c, :], state.s[c, :], state.E[c, :], state.passes[c, :] = y, s, E, passes
        state.node_t[c], state.node_status[c], state.node_round[c] = v['t'], v['status'], round_
        state.meas_child[c] = 0 if A == state.node_child0[c] else 1
        state.node_measured[c] = True


def _grow(state: TreeState, T, counter, pool, eps: float, nu_full: np.ndarray, M_floor: int):
    """The round loop (spec 5.2): round 0 measures the root. Below
    M_floor, every non-leaf child of a round's measured nodes is a
    candidate and nothing closes (4.2's noise rule only gates energy);
    at or above it, a node with no passing output is closed and a
    lineage closes after two consecutive below-threshold splits, as
    qijt's own refinement rule. Stops with 'tolerance' once two
    consecutive rounds, at or above the floor, both have
    max_o G_r,o/V_btw,o < eps, or with 'exhausted' when there are no
    candidates. Returns (curve_rows, tree_status, n_rounds,
    n_below_closed, below, n_halv, nonsmooth)."""
    below = np.zeros(state.node_cap, dtype=bool)
    n_halv = np.zeros(state.node_cap, dtype=int)
    nonsmooth = np.zeros(state.node_cap, dtype=bool)

    split_node(state, 0)
    _measure_nodes_verified(state, T, counter, pool, [0], 0, nu_full, n_halv, nonsmooth)
    n_below_closed, evals_total, round_ = 0, 1, 0
    V0 = leaf_V_btw(state)
    curve_rows = [dict(round=0, evals_total=evals_total, L=len(leaves(state)), n_selected=1,
                        below_floor=True, V_btw=V0, tau=np.full_like(V0, np.nan), G=state.E[0])]
    # max_o G_o/V_o with 0 where V_o = 0: the same ratio-with-floor
    # `candidate_priority` already computes for spec 5.2 step 3.
    prev_ratio = candidate_priority(state.E[0], V0)

    while True:
        V_prev, L_prev = curve_rows[-1]['V_btw'], curve_rows[-1]['L']
        tau = eps * V_prev / L_prev
        measured_prev = np.nonzero(state.node_round[:state.n_nodes] == round_)[0]
        floor_round = L_prev < M_floor
        candidates: List[int] = []
        for c in measured_prev:
            if floor_round:
                kids = (int(state.node_child0[c]), int(state.node_child1[c]))
            else:
                below[c] = bool(np.all((~state.passes[c]) | (state.E[c] < tau)))
                if not state.passes[c].any():
                    continue
                parent = int(state.node_parent[c])
                if below[c] and parent >= 0 and below[parent]:
                    n_below_closed += 1
                    continue
                kids = (int(state.node_child0[c]), int(state.node_child1[c]))
            candidates.extend(k for k in kids
                               if not is_point_leaf(state, k) and not is_degenerate(state, k))
        if not candidates:
            return curve_rows, 'exhausted', round_ + 1, n_below_closed, below, n_halv, nonsmooth

        priority = [candidate_priority(state.E[state.node_parent[c]], V_prev) for c in candidates]
        order = sorted(range(len(candidates)), key=lambda i: (-priority[i], candidates[i]))
        selected = [candidates[i] for i in order]
        for c in selected:
            split_node(state, c)
        round_ += 1
        _measure_nodes_verified(state, T, counter, pool, selected, round_, nu_full, n_halv,
                                 nonsmooth)
        evals_total += len(selected)
        V_new, L_new = leaf_V_btw(state), len(leaves(state))
        G = np.sum(state.E[selected], axis=0)
        ratio = candidate_priority(G, V_new)
        curve_rows.append(dict(round=round_, evals_total=evals_total, L=L_new,
                                n_selected=len(selected), below_floor=floor_round,
                                V_btw=V_new, tau=tau, G=G))
        if L_new >= M_floor and ratio < eps and prev_ratio < eps:
            return curve_rows, 'tolerance', round_ + 1, n_below_closed, below, n_halv, nonsmooth
        prev_ratio = ratio


def _bias(state: TreeState, T, counter, pool, eta_f: float) -> np.ndarray:
    """b_hat (spec 6): the central second-difference stencil on the
    root's own two children, one batch of four evaluations."""
    A, B = int(state.node_child0[0]), int(state.node_child1[0])
    delta = central_step(eta_f)
    ones_N = np.ones(state.N)
    tasks = []
    for K in (A, B):
        p_K, t = node_mass(state, K), step_parameter(delta, node_mass(state, K))
        mask = node_rows(state, K)
        tasks.append(((K, 1), set_weights(ones_N, mask, p_K, t), state.theta_hat, state.eta_full))
        tasks.append(((K, -1), set_weights(ones_N, mask, p_K, -t), state.theta_hat, state.eta_full))
    raw = {key: (np.full(state.q_full, np.nan) if failed else np.asarray(value, dtype=float))
           for key, value, failed, _status, _wall in run_batch(T, counter, pool, state.X, tasks)}
    D2 = {K: difference(node_mass(state, K), delta,
                         lambda t, K=K: raw[(K, 1 if t > 0 else -1)], state.theta_hat)[1]
          for K in (A, B)}
    meas, p_A, p_B = state.measured, node_mass(state, A), node_mass(state, B)
    return (p_A * D2[A][meas] + p_B * D2[B][meas]) / (2.0 * state.N)


def _leaf_geometry(state: TreeState):
    """Every current leaf's geometry (spec 7.1) and sibling derivative
    ĝ_ℓ,o (ranking only), plus the row arrays every contrast below
    needs: every point is always an active row for qijdt (no
    compaction), so `pos` is `core.tree.frame`'s general row-position
    map, computed once and reused."""
    row_ids, rows_x, omega0 = active_rows(state)
    row_z = state.row_z[row_ids]
    pos = np.full(state.n_rows, -1, dtype=int)
    pos[row_ids] = np.arange(len(row_ids))
    dummy, meas = np.zeros(0), state.measured
    leaf_ids = leaves(state)
    frames = {int(ell): frame(state, int(ell), row_z, omega0, pos, dummy, dummy, dummy)
              for ell in leaf_ids}
    ghat = {}
    for ell in leaf_ids:
        sib = int(sibling(state, ell))
        sib_fr = frames.get(sib) or frame(state, sib, row_z, omega0, pos, dummy, dummy, dummy)
        d = np.abs(state.U[ell, meas] - state.U[sib, meas])
        norm = np.linalg.norm(frames[ell]['mu'] - sib_fr['mu'])
        ghat[ell] = d / norm if norm > 0 else np.zeros_like(d)
    return leaf_ids, frames, ghat, row_z, pos, omega0, rows_x


def _buy_pairs(state: TreeState, T, counter, pool, leaf_ids, frames, ghat, tau, row_z, pos,
               omega0, rows_x, nu_full: np.ndarray):
    """Score every (leaf, j) candidate (spec 7.2) and buy the descending
    prefix with score >= 1, each contrast step-verified (4.3); the
    remainder's predicted terms sum to `P_unbought` (spec 11.1). Returns
    (pairs, A, P_unbought, n_nonsmooth, n_halvings)."""
    safe_tau = np.where(tau > 0, tau, 1.0)
    scored = []
    for ell in leaf_ids:
        fr = frames[ell]
        if fr['rank'] == 0:
            continue
        base = fr['p'] * ghat[ell] ** 2 / state.N
        for j in range(1, min(fr['rank'], fr['n'] - 1) + 1):
            P = fr['eigval'][j - 1] * base
            score = float(np.max(np.where(tau > 0, P / safe_tau, 0.0)))
            scored.append((score, int(ell), j, P))
    scored.sort(key=lambda s: (-s[0], s[1], s[2]))
    n_bought = sum(1 for s in scored if s[0] >= 1.0)

    specs, meta = {}, {}
    for _, ell, j, _ in scored[:n_bought]:
        pos_l, h = pair_h(frames[ell], j, row_z, pos)
        t0 = state.delta_f / np.max(np.abs(h))
        specs[(ell, j)] = dict(
            t=t0, omega=lambda t, pos_l=pos_l, h=h: _omega_h(omega0, pos_l, h, t))
        meta[(ell, j)] = frames[ell]['p']
    verified = step_verify(specs, T, counter, pool, rows_x, state.theta_hat, state.eta_full,
                            nu_full, state.measured, state.z_n) if specs else {}

    meas = state.measured
    pairs, A = [], {}
    n_nonsmooth = n_halvings = 0
    for (ell, j), p_l in meta.items():
        v = verified[(ell, j)]
        D = v['U']
        if v['failed']:
            W = np.zeros(meas.size)
            s = np.full(meas.size, np.nan)
            passed = np.zeros(meas.size, dtype=bool)
        else:
            s = np.sqrt(2.0) * nu_full[meas] / v['t']
            passed = np.abs(D[meas]) > state.z_n * s
            W = np.where(passed, D[meas] ** 2 / (state.N * p_l), 0.0)
            n_nonsmooth += int(v['nonsmooth'])
            n_halvings += v['n_halvings']
        pairs.append(dict(leaf=ell, j=j, D=D, W=W, t=v['t'], s=s, passed=passed,
                           status=v['status'], n_halvings=v['n_halvings'],
                           nonsmooth=v['nonsmooth']))
        A[ell] = A.get(ell, np.zeros_like(W)) + W
    zero = np.zeros(len(state.measured))
    P_unbought = np.sum([P for _, _, _, P in scored[n_bought:]], axis=0) \
        if len(scored) > n_bought else zero
    return pairs, A, P_unbought, n_nonsmooth, n_halvings


def _buy_quadratic(state: TreeState, T, counter, pool, frames, A, tau, row_z, pos, omega0,
                    rows_x, nu_full: np.ndarray):
    """Score every leaf with a bought pair and n >= rank + 2 (spec 7.3)
    and buy the descending prefix with score >= 1, step-verified (4.3).
    Returns (quads, n_nonsmooth, n_halvings)."""
    safe_tau = np.where(tau > 0, tau, 1.0)
    scored = []
    for ell, A_ell in A.items():
        fr = frames[ell]
        if fr['n'] < fr['rank'] + 2:
            continue
        ratio = np.where(tau > 0, A_ell / safe_tau, 0.0)
        scored.append((float(np.max(ratio)), ell))
    scored.sort(key=lambda s: (-s[0], s[1]))
    quad_leaves = [ell for score, ell in scored if score >= 1.0]

    specs, meta = {}, []
    for ell in quad_leaves:
        pos_l, h = quad_h(frames[ell], row_z, pos, omega0, state.N)
        if h is None:
            continue
        t0 = state.delta_f / np.max(np.abs(h))
        specs[ell] = dict(t=t0, omega=lambda t, pos_l=pos_l, h=h: _omega_h(omega0, pos_l, h, t))
        meta.append(ell)
    verified = step_verify(specs, T, counter, pool, rows_x, state.theta_hat, state.eta_full,
                            nu_full, state.measured, state.z_n) if specs else {}

    meas = state.measured
    quads = []
    n_nonsmooth = n_halvings = 0
    for ell in meta:
        v, p_l = verified[ell], frames[ell]['p']
        Q = v['U']
        if v['failed']:
            contribution = np.zeros(meas.size)
            s = np.full(meas.size, np.nan)
            passed = np.zeros(meas.size, dtype=bool)
        else:
            s = np.sqrt(2.0) * nu_full[meas] / v['t']
            passed = np.abs(Q[meas]) > state.z_n * s
            contribution = np.where(passed, Q[meas] ** 2 / (state.N * p_l), 0.0)
            n_nonsmooth += int(v['nonsmooth'])
            n_halvings += v['n_halvings']
        quads.append(dict(leaf=ell, Q=Q, contribution=contribution, t=v['t'], s=s, passed=passed,
                           status=v['status'], n_halvings=v['n_halvings'],
                           nonsmooth=v['nonsmooth']))
    return quads, n_nonsmooth, n_halvings


def _leaf_rows(state: TreeState, leaf_ids, frames, ghat, A, pairs, quads) -> List[dict]:
    """§11.3's leaf rows (unscaled -- qijdt has no anchor scale)."""
    n_pairs: Dict[int, int] = {}
    for pr in pairs:
        n_pairs[pr['leaf']] = n_pairs.get(pr['leaf'], 0) + 1
    Q2 = {qd['leaf']: qd['contribution'] for qd in quads}
    zero = np.zeros(len(state.measured))
    meas = state.measured
    return [dict(leaf=int(ell), n_points=frames[ell]['n'], p=frames[ell]['p'],
                 rank=frames[ell]['rank'], n_pairs_bought=n_pairs.get(ell, 0),
                 quad_bought=ell in Q2, U=state.U[ell, meas], ghat=ghat[ell],
                 A=A.get(ell, zero), Q2=Q2.get(ell, zero))
            for ell in leaf_ids]


def _reconstruct(state: TreeState, leaf_ids, frames, row_z, pos, omega0, pairs, quads):
    """psi_hat per point (spec 8.1): each leaf's own U plus its bought
    and passed directions' response; no anchor scale."""
    meas = state.measured
    psi = np.zeros((state.N, state.q_full))
    leaf_of_point = np.full(state.N, -1, dtype=int)
    for ell in leaf_ids:
        fr = frames[ell]
        leaf_of_point[fr['rows']] = ell
        psi[np.ix_(fr['rows'], meas)] += state.U[ell, meas]
    for pr in pairs:
        fr = frames[pr['leaf']]
        _, h = pair_h(fr, pr['j'], row_z, pos)
        psi[np.ix_(fr['rows'], meas)] += np.outer(
            h, np.where(pr['passed'], pr['D'][meas] / fr['p'], 0.0))
    for qd in quads:
        fr = frames[qd['leaf']]
        _, h = quad_h(fr, row_z, pos, omega0, state.N)
        psi[np.ix_(fr['rows'], meas)] += np.outer(
            h, np.where(qd['passed'], qd['Q'][meas] / fr['p'], 0.0))
    return psi, leaf_of_point


def _acceleration(psi_meas: np.ndarray) -> np.ndarray:
    """a_o (spec 8.2): the point-mass-(1/N) skewness ratio; NaN where
    the scale (mean psi^2) is 0."""
    N = psi_meas.shape[0]
    m2 = np.mean(psi_meas ** 2, axis=0)
    m3 = np.mean(psi_meas ** 3, axis=0)
    out = np.full(psi_meas.shape[1], np.nan)
    valid = m2 > 0.0
    out[valid] = m3[valid] / (6.0 * np.sqrt(N) * m2[valid] ** 1.5)
    return out


def _nodes_table(state: TreeState, outputs, below, n_halv, nonsmooth) -> pd.DataFrame:
    """§11.2: one row per measured node and per leaf."""
    measured_mask = state.node_measured[:state.n_nodes]
    parent = state.node_parent[:state.n_nodes]
    has_parent = parent >= 0
    is_child = np.zeros(state.n_nodes, dtype=bool)
    is_child[has_parent] = measured_mask[parent[has_parent]]
    rows = []
    for c in np.nonzero(measured_mask | is_child)[0]:
        row = dict(node=int(c), parent=int(parent[c]), depth=int(state.node_depth[c]),
                   n_points=int(state.node_hi[c] - state.node_lo[c]),
                   p=float(node_mass(state, c)), node_measured=bool(measured_mask[c]),
                   round=int(state.node_round[c]), below=bool(below[c]),
                   measured_child=int(state.meas_child[c]), t=float(state.node_t[c]),
                   n_halvings=int(n_halv[c]), nonsmooth=bool(nonsmooth[c]),
                   status=str(state.node_status[c]))
        for j, o in enumerate(outputs):
            row[f'U_{o}'] = float(state.U[c, state.measured[j]])
            row[f'y_{o}'] = float(state.y[c, j])
            row[f's_{o}'] = float(state.s[c, j])
            row[f'E_{o}'] = float(state.E[c, j])
            row[f'pass_{o}'] = bool(state.passes[c, j])
        rows.append(row)
    return pd.DataFrame(rows)


def _leaves_table(leaf_rows, outputs) -> pd.DataFrame:
    """§11.3."""
    rows = []
    for lr in leaf_rows:
        row = dict(leaf=int(lr['leaf']), n_points=int(lr['n_points']), p=float(lr['p']),
                   rank=int(lr['rank']), n_pairs_bought=int(lr['n_pairs_bought']),
                   quad_bought=bool(lr['quad_bought']))
        for j, o in enumerate(outputs):
            row[f'U_{o}'] = float(lr['U'][j])
            row[f'ghat_{o}'] = float(lr['ghat'][j])
            row[f'A_{o}'] = float(lr['A'][j])
            row[f'Q2_{o}'] = float(lr['Q2'][j])
        rows.append(row)
    return pd.DataFrame(rows)


def _pairs_table(pairs, quads, outputs, measured) -> pd.DataFrame:
    """§11.4: j = 0 for the quadratic, Q in the D column."""
    rows = []
    for rec, j, resp in ([(p, p['j'], p['D']) for p in pairs]
                         + [(qd, 0, qd['Q']) for qd in quads]):
        row = dict(leaf=int(rec['leaf']), j=int(j), t=float(rec['t']),
                   n_halvings=int(rec['n_halvings']), nonsmooth=bool(rec['nonsmooth']))
        for k, o in enumerate(outputs):
            row[f'D_{o}'] = float(resp[measured[k]])
            row[f's_{o}'] = float(rec['s'][k])
            row[f'pass_{o}'] = bool(rec['passed'][k])
        rows.append(row)
    return pd.DataFrame(rows)


def _curve_table(curve_rows, outputs) -> pd.DataFrame:
    """§11.5: one row per round."""
    rows = []
    for r in curve_rows:
        row = dict(round=int(r['round']), evals_total=int(r['evals_total']),
                   L=int(r['L']), n_selected=int(r['n_selected']),
                   below_floor=bool(r['below_floor']))
        for j, o in enumerate(outputs):
            row[f'V_btw_{o}'] = float(r['V_btw'][j])
            row[f'tau_{o}'] = float(r['tau'][j])
            row[f'G_{o}'] = float(r['G'][j])
        rows.append(row)
    return pd.DataFrame(rows)


class QIJDT:
    """The data-tree method (spec/QIJ_qijdt_spec.md)."""

    def __init__(self, eps: float = 0.01, vq_transform=None) -> None:
        self.eps = eps
        self.vq_transform = vq_transform

    def fit(self, X: np.ndarray, T, pool=None) -> QIJDTResult:
        """Run the draw's stage sequence (spec 9.1) and return a
        `QIJDTResult`."""
        t_start = time.perf_counter()
        X = np.asarray(X, dtype=float)
        N = len(X)
        T = _wrap(T)
        outputs_full = T.outputs
        q_full = len(outputs_full)
        measured = np.asarray(list(getattr(T, 'measured', range(q_full))), dtype=int)
        outputs = tuple(outputs_full[i] for i in measured)
        M_floor = cost_rule_M(N, len(measured), self.eps)
        workers = pool.workers if pool is not None else 1
        counter = Counter(T, N)
        evals = {s: 0 for s in _STAGES}
        wall = {s: 0.0 for s in _STAGES}

        # --- full_fit (spec 2 item 1): theta_hat = T(X, 1_N) via
        # `run_batch` (spec 9.2's own task shape), not `_theta_hat_task`
        # + a parent-side `fit_status` call -- the latter reads
        # `T.last_fit_info` in the PARENT process, which a pool worker's
        # fit never updates, so the status would silently depend on
        # worker count (found by this build's own V3 run; qijt.py's
        # full_fit uses `run_batch` for exactly this reason).
        with _stage(evals, wall, counter, 'full_fit'):
            [(_key, result, _failed, theta_hat_status, _wall)] = run_batch(
                T, counter, pool, X, [('theta_hat', np.ones(N), None, None)])
            theta_hat = np.asarray(result, dtype=float)
        if np.any(np.isnan(theta_hat)):
            return _nan_result(outputs, measured, N, self.eps, M_floor, workers,
                                'theta_hat_failed', theta_hat_status, float('nan'), False,
                                evals, wall, t_start, counter, theta_hat)

        # --- eta_full (spec 2 item 2) ---
        with _stage(evals, wall, counter, 'eta_full'):
            eta_full, _n_eta, eta_full_failed = measure_eta_full(counter, X, theta_hat)

        # --- the base fit's own fixed point (spec 2.1), folded into
        # full_fit; `fixed_point_nu` also returns the last iteration's
        # pre-image theta, so nu (2.1) can use the exact final pair.
        with _stage(evals, wall, counter, 'full_fit'):
            if counter.takes_start:
                theta_hat, n_fp, r_fp, ok, theta_prev = fixed_point_nu(
                    lambda th: counter(X, np.ones(N), start=th, eta=eta_full), theta_hat,
                    eta_full)
            else:
                n_fp, r_fp, ok, theta_prev = 0, 0.0, True, theta_hat
        if np.any(np.isnan(theta_hat)):
            return _nan_result(outputs, measured, N, self.eps, M_floor, workers,
                                'theta_hat_failed', theta_hat_status, eta_full, eta_full_failed,
                                evals, wall, t_start, counter, theta_hat, n_fp=n_fp, r_fp=r_fp)
        if not ok:
            return _nan_result(outputs, measured, N, self.eps, M_floor, workers,
                                'base_unconverged', theta_hat_status, eta_full, eta_full_failed,
                                evals, wall, t_start, counter, theta_hat, n_fp=n_fp, r_fp=r_fp)
        # nu_o = max(eta_f*|theta_hat_o|, |theta'_o - theta_o|) (spec
        # 2.1); theta_prev == theta_hat with no iteration, giving
        # nu = eta_f*|theta_hat| for a start-less T, as the spec's own
        # start-less case (eta_full == T's declared eta there already).
        nu_full = np.maximum(eta_full * np.abs(theta_hat), np.abs(theta_hat - theta_prev))

        # --- tree (spec 3, 5) ---
        eta_f = eta_full  # spec 2.2's eta_f; always eta_full here (measure_eta_full already
                          # returns T's declared eta in the start-less case)
        Z, _inv = self.vq_transform(X) if self.vq_transform is not None else (X, None)
        Z = np.asarray(Z, dtype=float)
        if Z.ndim == 1:
            Z = Z.reshape(-1, 1)
        state = _init_state(X, Z, theta_hat, eta_full, eta_f, measured, q_full, 2 * N)
        with _stage(evals, wall, counter, 'tree'):
            curve_rows, tree_status, n_rounds, n_below_closed, below, n_halv, nonsmooth = _grow(
                state, T, counter, pool, self.eps, nu_full, M_floor)

        # --- bias (spec 6) ---
        with _stage(evals, wall, counter, 'bias'):
            b_hat = _bias(state, T, counter, pool, eta_f)

        # --- within (spec 7.2) + quadratic (spec 7.3) ---
        leaf_ids, frames, ghat, row_z, pos, omega0, rows_x = _leaf_geometry(state)
        L_final, V_btw_final = curve_rows[-1]['L'], curve_rows[-1]['V_btw']
        tau_final = self.eps * V_btw_final / L_final
        with _stage(evals, wall, counter, 'within'):
            pairs, A, P_unbought, ns_pairs, nh_pairs = _buy_pairs(
                state, T, counter, pool, leaf_ids, frames, ghat, tau_final, row_z, pos, omega0,
                rows_x, nu_full)
        with _stage(evals, wall, counter, 'quadratic'):
            quads, ns_quad, nh_quad = _buy_quadratic(
                state, T, counter, pool, frames, A, tau_final, row_z, pos, omega0, rows_x,
                nu_full)

        # --- reconstruction (8.1), acceleration (8.2) ---
        psi, leaf_of_point = _reconstruct(state, leaf_ids, frames, row_z, pos, omega0, pairs,
                                          quads)
        accel = _acceleration(psi[:, measured])
        q = len(measured)
        W_sum = np.sum([p['W'] for p in pairs], axis=0) if pairs else np.zeros(q)
        Q_sum = np.sum([qd['contribution'] for qd in quads], axis=0) if quads else np.zeros(q)
        V_btw = leaf_V_btw(state)
        V_win = W_sum + Q_sum
        V_tot = V_btw + V_win

        # --- curvature (spec 8.3) ---
        curv_busy = 0.0
        with _stage(evals, wall, counter, 'curvature'):
            sv = SurveyRows(rows=X, row_field=np.arange(N), omega0=np.ones(N), eta_Q=eta_full,
                             step_ratio=np.full((5, q_full), np.nan), quantized_start='full-data',
                             eta_rows=eta_full)
            c_q, _eps_c, curv_busy, _one_sided = _curvature(
                counter, sv, theta_hat, psi, np.full(N, 1.0 / N), N, pool, outputs=measured)

        # --- assembly ---
        wall_time = time.perf_counter() - t_start
        busy_time = wall_time + curv_busy
        n_measured = int(np.sum(state.node_measured[:state.n_nodes]))
        max_depth = int(state.node_depth[:state.n_nodes].max()) if state.n_nodes else 0
        leaf_rows = _leaf_rows(state, leaf_ids, frames, ghat, A, pairs, quads)
        n_nonsmooth = int(nonsmooth[:state.n_nodes].sum()) + ns_pairs + ns_quad
        n_halvings_total = int(n_halv[:state.n_nodes].sum()) + nh_pairs + nh_quad
        nu = nu_full[measured]

        return QIJDTResult(
            outputs=outputs, measured=measured, N=N, eps=self.eps, M_floor=M_floor,
            workers=workers, status='ok', theta_hat_status=theta_hat_status,
            eta_full=float(eta_full), eta_full_failed=bool(eta_full_failed), n_fp=int(n_fp),
            r_fp=float(r_fp), n_rounds=int(n_rounds), n_measured=n_measured,
            n_leaves=len(leaf_ids), max_depth=max_depth, n_failed=int(counter.failed),
            n_nonsmooth=n_nonsmooth, n_halvings_total=n_halvings_total, tree_status=tree_status,
            n_pairs_bought=len(pairs), n_quad_bought=len(quads),
            n_below_closed=int(n_below_closed), busy_time=float(busy_time),
            wall_time=float(wall_time), evals_by_stage=evals, wall_time_by_stage=wall,
            theta_hat=theta_hat, nu=nu, V_btw=V_btw, V_win=V_win, V_tot=V_tot, accel=accel,
            b_hat=b_hat, c_q=c_q, P_unbought=P_unbought,
            nodes=_nodes_table(state, outputs, below, n_halv, nonsmooth),
            leaves=_leaves_table(leaf_rows, outputs),
            pairs=_pairs_table(pairs, quads, outputs, measured),
            curve=_curve_table(curve_rows, outputs),
            psi_hat=psi[:, measured], leaf_of_point=leaf_of_point,
        )
