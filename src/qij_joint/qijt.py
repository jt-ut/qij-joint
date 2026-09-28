"""`QIJT`: the tree method (spec/QIJ_qijt_spec.md). `QIJT(M_X, budget,
budget_win, budget_quad, seed=0, vq_transform=None).fit(X, T,
pool=None)` runs the draw's stage sequence (spec 9.1): full_fit,
eta_full, xvq, base_Q, eta_Q, tree, anchors, bias, within, quadratic,
drift, curvature, assembly, returning a `QIJTResult`. `M_X` is the
prototype count, `budget` the tree contrast budget (spec 5),
`budget_win`/`budget_quad` the within-term pair/quadratic budgets
(spec 7.2-7.3); all four are required, no cost rule and no other
option (spec 1.1).

The hierarchy, growth, anchors/bias/drift and within-term/reconstruction
(spec 3-8.1) live in `core.tree` (agents H/G/N/W); this module owns the
draw's orchestration, stage timing (spec 9.1), the failures of spec 10,
the acceleration of spec 8.2, and the four result tables of spec 11.

Two named quantities share the letter `a` in the spec and are kept
distinct here: `a_scale` (spec 6.2, the anchor-derived full-data scale
that turns a quantized quantity into full-data units) and `accel`
(spec 8.2, the BCa acceleration `abc_interval` takes as its own `a`).
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import pandas as pd

from .core.abc import abc_interval as _abc_interval
from .core.abc import curvature as _curvature
from .core.counter import Counter
from .core.differences import forward_step
from .core.eta import measure_eta_full
from .core.tree import (active_rows, anchors, build_state, drift, grow, leaf_V_btw, leaves, node_mass,
                        reconstruct, run_batch, within)
from .core.xvq import SurveyRows, _measure_eta_Q, fit_xvq
from .parallel import fit_status
from .qij import _wrap
from .result import _normal_interval

_STAGES = ('full_fit', 'eta_full', 'xvq', 'base_Q', 'eta_Q', 'tree', 'opening',
           'anchors', 'bias', 'within', 'quadratic', 'drift', 'curvature')
_LEVEL = 0.95


@dataclass
class QIJTResult:
    """One draw's QIJT result (spec 11.1, as attributes). `theta_hat`
    is FULL width (q_full,); `measured` (q,) are the absolute indices
    every other per-output array is ordered by. `nodes`/`leaves`/
    `curve`/`anchors` are the tables of spec 11.2-11.5."""

    outputs: Tuple[str, ...]
    measured: np.ndarray            # (q,) int, absolute indices into theta_hat
    N: int
    M_X: int
    R_final: int
    budget: int
    budget_win: int
    budget_quad: int
    workers: int
    status: str
    theta_hat_status: str
    theta_Q_status: str
    eta_full: float
    eta_Q: float
    eta_full_failed: bool
    eta_Q_failed: bool
    n_rounds: int
    n_measured: int
    n_opened: int
    n_leaves: int
    max_depth: int
    n_failed: int
    tree_status: str
    n_pairs_bought: int
    n_quad_bought: int
    busy_time: float
    wall_time: float
    evals_by_stage: Dict[str, int]
    rows_by_stage: Dict[str, int]
    wall_time_by_stage: Dict[str, float]
    theta_hat: np.ndarray            # (q_full,)
    V_btw: np.ndarray                # (q,)
    V_win: np.ndarray                # (q,)
    V_tot: np.ndarray                # (q,)
    a_scale: np.ndarray              # (q,) spec 6.2
    a_scatter: np.ndarray            # (q,) spec 6.2
    root_drift: np.ndarray           # (q,) spec 6.3
    accel: np.ndarray                # (q,) spec 8.2
    b_hat: np.ndarray                # (q,) spec 6.4
    c_q: np.ndarray                  # (q,) spec 8.3
    nodes: pd.DataFrame
    leaves: pd.DataFrame
    curve: pd.DataFrame
    anchors: pd.DataFrame
    pairs: pd.DataFrame
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
        """(q, 2): the same normal interval on V_btw alone (spec 8.4,
        for comparison against `interval`)."""
        return _normal_interval(self.theta_hat[self.measured], self.V_btw, level)

    def abc_interval(self, level: float) -> np.ndarray:
        """(q, 2): the ABC_q interval (spec 8.4) from `accel`/`b_hat`/`c_q`
        and sigma = sqrt(V_tot)."""
        sigma = np.sqrt(np.maximum(self.V_tot, 0.0))
        return _abc_interval(self.theta_hat[self.measured], sigma, self.accel,
                              self.b_hat, self.c_q, level)


def _acceleration(psi_points: np.ndarray) -> np.ndarray:
    """a_o (spec 8.2): `ivq.bias_and_acceleration`'s acceleration
    formula at point masses 1/N, from the reconstructed per-point
    psi_hat; NaN where the scale (mean psi^2) is 0."""
    N = psi_points.shape[0]
    m2 = np.mean(psi_points ** 2, axis=0)
    m3 = np.mean(psi_points ** 3, axis=0)
    out = np.full(psi_points.shape[1], np.nan)
    valid = m2 > 0.0
    out[valid] = m3[valid] / (6.0 * np.sqrt(N) * m2[valid] ** 1.5)
    return out


def _tree_rows(curve_rows) -> int:
    """Rows of the tree's own node measurements: each round's new
    evaluations times that round's (post-opening) row count."""
    total, prev = 0, 0
    for row in curve_rows:
        total += (row['evals_tree'] - prev) * row['R']
        prev = row['evals_tree']
    return total


def _pairs_table(pairs, quads, outputs: Tuple[str, ...], measured: np.ndarray) -> pd.DataFrame:
    """§11.2a: one row per bought contrast, j = 0 for the quadratic
    (D unscaled; Q in the D column)."""
    rows = []
    for rec, j, resp in ([(p, p['j'], p['D']) for p in pairs]
                         + [(qd, 0, qd['Q']) for qd in quads]):
        row = dict(leaf=int(rec['leaf']), j=int(j), t=float(rec['t']))
        for k, o in enumerate(outputs):
            row[f'D_{o}'] = float(resp[measured[k]])
            row[f's_{o}'] = float(rec['s'][k])
            row[f'pass_{o}'] = bool(rec['passed'][k])
        rows.append(row)
    return pd.DataFrame(rows)


def _nodes_table(state, outputs: Tuple[str, ...], curve_rows) -> pd.DataFrame:
    """§11.2: one row per node of the measured tree and its children."""
    round_to_R = {r['round']: r['R'] for r in curve_rows}
    measured_mask = state.node_measured[:state.n_nodes]
    parent = state.node_parent[:state.n_nodes]
    is_child = np.zeros(state.n_nodes, dtype=bool)
    has_parent = parent >= 0
    is_child[has_parent] = measured_mask[parent[has_parent]]
    keep = np.nonzero(measured_mask | is_child)[0]
    rows = []
    for c in keep:
        lo, hi = int(state.node_lo[c]), int(state.node_hi[c])
        if state.node_kind[c] == 0:
            cells = state.cell_perm[lo:hi]
            n_cells = len(cells)
            n_points = int(np.sum(state.cell_start[cells + 1] - state.cell_start[cells]))
        else:
            n_cells, n_points = 1, hi - lo
        row = dict(node=int(c), parent=int(parent[c]), depth=int(state.node_depth[c]),
                   kind=('cells' if state.node_kind[c] == 0 else 'points'),
                   cell=int(state.node_cell[c]), n_cells=n_cells, n_points=n_points,
                   p=float(node_mass(state, c)), node_measured=bool(measured_mask[c]),
                   round=int(state.node_round[c]), measured_child=int(state.meas_child[c]),
                   t=float(state.node_t[c]),
                   evals_rows=int(round_to_R.get(int(state.node_round[c]), 0)),
                   status=str(state.node_status[c]))
        for j, o in enumerate(outputs):
            row[f'U_{o}'] = float(state.U[c, state.measured[j]])
            row[f'y_{o}'] = float(state.y[c, j])
            row[f's_{o}'] = float(state.s[c, j])
            row[f'E_{o}'] = float(state.E[c, j])
            row[f'pass_{o}'] = bool(state.passes[c, j])
            row[f'open_shift_{o}'] = float(state.open_shift[c, j])
        rows.append(row)
    return pd.DataFrame(rows)


def _leaves_table(leaf_rows, outputs: Tuple[str, ...], a_scale: np.ndarray) -> pd.DataFrame:
    """§11.3: `within`'s leaf rows (unscaled); A_o/Q2_o scaled by
    a_scale**2 here, as the affine and quadratic terms are throughout."""
    rows = []
    for lr in leaf_rows:
        row = dict(leaf=int(lr['leaf']), n_points=int(lr['n_points']), p=float(lr['p']),
                   rank=int(lr['rank']), n_pairs_bought=int(lr['n_pairs_bought']),
                   quad_bought=bool(lr['quad_bought']))
        for j, o in enumerate(outputs):
            row[f'U_{o}'] = float(lr['U'][j])
            row[f'ghat_{o}'] = float(lr['ghat'][j])
            row[f'A_{o}'] = float(lr['A'][j] * a_scale[j] ** 2)
            row[f'Q2_{o}'] = float(lr['Q2'][j] * a_scale[j] ** 2)
        rows.append(row)
    return pd.DataFrame(rows)


def _curve_table(curve_rows, outputs: Tuple[str, ...], a_scale: np.ndarray) -> pd.DataFrame:
    """§11.4: one row per round; V_btw_o scaled a posteriori by a_scale**2."""
    rows = []
    for r in curve_rows:
        row = dict(round=int(r['round']), evals_tree=int(r['evals_tree']),
                   R=int(r['R']), n_open=int(r['n_open']))
        for j, o in enumerate(outputs):
            row[f'V_btw_{o}'] = float(r['V_btw'][j] * a_scale[j] ** 2)
        rows.append(row)
    return pd.DataFrame(rows)


def _anchors_table(table_rows, outputs: Tuple[str, ...]) -> pd.DataFrame:
    """§11.5: one row per anchor node."""
    rows = []
    for r in table_rows:
        row = dict(node=int(r['node']), depth=int(r['depth']), p_A=float(r['p_A']))
        for j, o in enumerate(outputs):
            row[f'y_full_{o}'] = float(r['y_full'][j])
            row[f'y_half_{o}'] = float(r['y_half'][j])
            row[f'y_Q_{o}'] = float(r['y_Q'][j])
            row[f'step_ratio_{o}'] = float(r['step_ratio'][j])
        rows.append(row)
    return pd.DataFrame(rows)


def _nan_result(outputs, measured, N, method, workers, status, theta_hat_status,
                 theta_Q_status, eta_full, eta_Q, eta_full_failed, eta_Q_failed,
                 evals, rows, wall, t_start, counter, theta_hat) -> QIJTResult:
    """A failed draw (spec 10): NaN variances/intervals, the status,
    whatever stage counts were made before the failure, empty tables."""
    q = len(outputs)
    nan_q = np.full(q, np.nan)
    wall_time = time.perf_counter() - t_start
    for s in _STAGES:
        evals.setdefault(s, 0)
        rows.setdefault(s, 0)
        wall.setdefault(s, 0.0)
    evals['total'] = sum(evals[s] for s in _STAGES)
    rows['total'] = sum(rows[s] for s in _STAGES)
    wall['total'] = sum(wall[s] for s in _STAGES)
    return QIJTResult(
        outputs=outputs, measured=np.asarray(measured, dtype=int), N=N, M_X=0, R_final=0,
        budget=method.budget, budget_win=method.budget_win, budget_quad=method.budget_quad,
        workers=workers, status=status, theta_hat_status=theta_hat_status,
        theta_Q_status=theta_Q_status, eta_full=float(eta_full), eta_Q=float(eta_Q),
        eta_full_failed=bool(eta_full_failed), eta_Q_failed=bool(eta_Q_failed),
        n_rounds=0, n_measured=0, n_opened=0, n_leaves=0, max_depth=0,
        n_failed=int(counter.failed), tree_status='', n_pairs_bought=0, n_quad_bought=0,
        busy_time=float(wall_time), wall_time=float(wall_time),
        evals_by_stage=evals, rows_by_stage=rows, wall_time_by_stage=wall,
        theta_hat=np.asarray(theta_hat, dtype=float), V_btw=nan_q, V_win=nan_q, V_tot=nan_q,
        a_scale=nan_q, a_scatter=nan_q, root_drift=nan_q, accel=nan_q, b_hat=nan_q, c_q=nan_q,
        nodes=pd.DataFrame(), leaves=pd.DataFrame(), curve=pd.DataFrame(), anchors=pd.DataFrame(),
        pairs=pd.DataFrame(),
        psi_hat=np.full((N, q), np.nan), leaf_of_point=np.full(N, -1, dtype=int),
    )


class QIJT:
    """The tree method (spec/QIJ_qijt_spec.md)."""

    def __init__(self, M_X: int, budget: int, budget_win: int, budget_quad: int,
                 seed: int = 0, vq_transform=None) -> None:
        self.M_X = M_X
        self.budget = budget
        self.budget_win = budget_win
        self.budget_quad = budget_quad
        self.seed = seed
        self.vq_transform = vq_transform

    def fit(self, X: np.ndarray, T, pool=None) -> QIJTResult:
        """Run the draw's stage sequence (spec 9.1) and return a
        `QIJTResult`. `budget` >= 1, `budget_win`/`budget_quad` >= 0."""
        t_start = time.perf_counter()
        X = np.asarray(X, dtype=float)
        N = len(X)
        T = _wrap(T)
        outputs_full = T.outputs
        q_full = len(outputs_full)
        measured = np.asarray(list(getattr(T, 'measured', range(q_full))), dtype=int)
        outputs = tuple(outputs_full[i] for i in measured)
        q = len(measured)
        workers = pool.workers if pool is not None else 1
        counter = Counter(T, N)
        evals = {s: 0 for s in _STAGES}
        rows = {s: 0 for s in _STAGES}
        wall = {s: 0.0 for s in _STAGES}

        # --- full_fit (spec 2.2 item 1) ---
        t0 = time.perf_counter()
        # run_batch reads the fit status in the process that ran the fit
        # (a pool worker's estimator, not the parent's).
        [(_, result, _failed, theta_hat_status, _)] = run_batch(
            T, counter, pool, X, [('theta_hat', np.ones(N), None, None)])
        theta_hat = np.asarray(result, dtype=float)
        wall['full_fit'] = time.perf_counter() - t0
        evals['full_fit'], rows['full_fit'] = counter.snapshot()
        if np.any(np.isnan(theta_hat)):
            return _nan_result(outputs, measured, N, self, workers, 'theta_hat_failed',
                                theta_hat_status, '', float('nan'), float('nan'), False, False,
                                evals, rows, wall, t_start, counter, theta_hat)

        # --- eta_full (spec 2.2 item 2) ---
        t0 = time.perf_counter()
        ev0, rw0 = counter.snapshot()
        eta_full, _n_eta, eta_full_failed = measure_eta_full(counter, X, theta_hat)
        wall['eta_full'] = time.perf_counter() - t0
        ev1, rw1 = counter.snapshot()
        evals['eta_full'], rows['eta_full'] = ev1 - ev0, rw1 - rw0

        # --- xvq (spec 9.1; Z/inverse exactly as qij.py, incl. 1-D promotion) ---
        t0 = time.perf_counter()
        Z, inverse = self.vq_transform(X) if self.vq_transform is not None else (X, lambda A: A)
        Z = np.asarray(Z, dtype=float)
        if Z.ndim == 1:
            Z = Z.reshape(-1, 1)
            inverse = lambda A, _inv=inverse: _inv(A.reshape(-1))
        xvq = fit_xvq(Z, self.M_X, self.seed, workers)
        centers_x = np.asarray(inverse(xvq.centers), dtype=float)
        node_cap = xvq.M_used + N + 1
        state = build_state(X, Z, xvq.bmu, centers_x, xvq.centers, measured, q_full, node_cap)
        state.theta_hat = theta_hat
        wall['xvq'] = time.perf_counter() - t0
        ev2, rw2 = counter.snapshot()
        evals['xvq'], rows['xvq'] = ev2 - ev1, rw2 - rw1  # 0: fit_xvq spends no evaluation

        # --- base_Q (spec 2.2 item 3) ---
        t0 = time.perf_counter()
        _row_ids, rows_x, omega0 = active_rows(state)
        theta_Q = np.asarray(counter(rows_x, omega0, start=theta_hat, eta=eta_full), dtype=float)
        theta_Q_status = fit_status(T, theta_Q)
        state.theta_Q = theta_Q
        wall['base_Q'] = time.perf_counter() - t0
        ev3, rw3 = counter.snapshot()
        evals['base_Q'], rows['base_Q'] = ev3 - ev2, rw3 - rw2
        if np.any(np.isnan(theta_Q)):
            return _nan_result(outputs, measured, N, self, workers, 'theta_Q_failed',
                                theta_hat_status, theta_Q_status, eta_full, float('nan'),
                                eta_full_failed, False, evals, rows, wall, t_start, counter,
                                theta_hat)

        # --- eta_Q (spec 2.2 item 4, 2.3) ---
        t0 = time.perf_counter()
        R0 = len(omega0)
        m0 = omega0 / N
        if counter.takes_start:
            eta_Q, eta_Q_failed = _measure_eta_Q(counter, rows_x, omega0, np.arange(R0),
                                                  m0, theta_Q, eta=eta_full)
            eta_f = eta_full
        else:
            eta_Q, eta_Q_failed = float(counter.eta), False
            eta_f = float(counter.eta)
        delta_f = forward_step(eta_f)
        state.eta_full, state.eta_Q, state.delta_f = eta_full, eta_Q, delta_f
        wall['eta_Q'] = time.perf_counter() - t0
        ev4, rw4 = counter.snapshot()
        evals['eta_Q'], rows['eta_Q'] = ev4 - ev3, rw4 - rw3

        # --- tree (spec 5); `opening` split from the curve table ---
        t0 = time.perf_counter()
        curve_rows, tree_status, n_rounds, evals_tree = grow(state, T, counter, pool, self.budget)
        wall['tree'] = time.perf_counter() - t0
        ev5, rw5 = counter.snapshot()
        rw_tree_window = rw5 - rw4
        if tree_status == 'opening_failed':
            return _nan_result(outputs, measured, N, self, workers, 'opening_failed',
                                theta_hat_status, theta_Q_status, eta_full, eta_Q,
                                eta_full_failed, eta_Q_failed, evals, rows, wall, t_start,
                                counter, theta_hat)
        # The window also holds the openings' theta_Q re-evaluations and U
        # re-measurements (spec 5.3), counted in `opening`.
        evals['tree'], evals['opening'] = evals_tree, (ev5 - ev4) - evals_tree
        rows['tree'] = _tree_rows(curve_rows)
        rows['opening'] = rw_tree_window - rows['tree']

        # --- anchors + bias (spec 6.1-6.2, 6.4); one N call, split by count ---
        t0 = time.perf_counter()
        R_tree_final = len(active_rows(state)[0])
        anc = anchors(state, T, counter, pool)
        wall['anchors'] = time.perf_counter() - t0  # bias's 2 full evals share this wall time
        n_anchor = len(anc['table'])
        evals['bias'], rows['bias'] = 4, 4 * N
        evals['anchors'] = 2 * n_anchor
        rows['anchors'] = 2 * n_anchor * N + anc['evals_anchor_Q'] * R_tree_final
        a_scale, a_scatter, b_hat = anc['a'], anc['scatter'], anc['b_hat']

        # --- within + quadratic (spec 7.2-7.3); `opening`'s within share ---
        t0 = time.perf_counter()
        ev_w0, rw_w0 = counter.snapshot()
        wr = within(state, T, counter, pool, self.budget_win, self.budget_quad, leaf_V_btw(state))
        wall['within'] = time.perf_counter() - t0  # quadratic shares this wall time
        if wr.get('status') == 'opening_failed':
            return _nan_result(outputs, measured, N, self, workers, 'opening_failed',
                                theta_hat_status, theta_Q_status, eta_full, eta_Q,
                                eta_full_failed, eta_Q_failed, evals, rows, wall,
                                t_start, counter, theta_hat)
        R_final = len(active_rows(state)[0])
        n_pairs, n_quads = len(wr['pairs']), len(wr['quads'])
        evals['within'], rows['within'] = n_pairs, n_pairs * R_final
        evals['quadratic'], rows['quadratic'] = n_quads, n_quads * R_final
        ev_w1, rw_w1 = counter.snapshot()
        evals['opening'] += (ev_w1 - ev_w0) - n_pairs - n_quads
        rows['opening'] += (rw_w1 - rw_w0) - (n_pairs + n_quads) * R_final

        W_sum = np.sum([p['W'] for p in wr['pairs']], axis=0) if wr['pairs'] else np.zeros(q)
        Q_sum = np.sum([qd['contribution'] for qd in wr['quads']], axis=0) if wr['quads'] else np.zeros(q)
        V_win = a_scale ** 2 * (W_sum + Q_sum)
        V_btw = a_scale ** 2 * leaf_V_btw(state)
        V_tot = V_btw + V_win

        # --- drift (spec 6.3) ---
        t0 = time.perf_counter()
        ev6, rw6 = counter.snapshot()
        root_drift = drift(state, T, counter, pool)
        wall['drift'] = time.perf_counter() - t0
        ev7, rw7 = counter.snapshot()
        evals['drift'], rows['drift'] = ev7 - ev6, rw7 - rw6

        # --- reconstruction (spec 8.1; no evaluation) and acceleration (8.2) ---
        psi_rows, psi_points, leaf_of_point = reconstruct(state, wr['pairs'], wr['quads'], a_scale)
        accel = _acceleration(psi_points)

        # --- curvature (spec 8.3) ---
        t0 = time.perf_counter()
        ev8, rw8 = counter.snapshot()
        row_ids_f, rows_x_f, omega0_f = active_rows(state)
        sv = SurveyRows(rows=rows_x_f, row_field=np.arange(len(row_ids_f)), omega0=omega0_f,
                         eta_Q=eta_Q,
                         step_ratio=np.full((5, q_full), np.nan),
                         quantized_start='full-data', eta_rows=eta_full)
        c_q, _eps, curv_busy, _one_sided = _curvature(
            counter, sv, state.theta_Q, psi_rows, omega0_f / N, N, pool, outputs=measured)
        wall['curvature'] = time.perf_counter() - t0
        ev9, rw9 = counter.snapshot()
        evals['curvature'], rows['curvature'] = ev9 - ev8, rw9 - rw8

        # --- assembly ---
        wall_time = time.perf_counter() - t_start
        # Only `curvature` returns its own pool busy time; the other pool
        # stages' task walls are not yet surfaced by core.tree (reported).
        busy_time = wall_time + curv_busy
        evals['total'] = sum(evals[s] for s in _STAGES)
        rows['total'] = sum(rows[s] for s in _STAGES)
        wall['total'] = sum(wall[s] for s in _STAGES)

        n_measured_nodes = int(np.sum(state.node_measured[:state.n_nodes]))
        max_depth = int(state.node_depth[:state.n_nodes].max()) if state.n_nodes else 0
        n_opened = int(np.sum(state.cell_row[:state.M] < 0))
        leaf_ids = leaves(state)

        return QIJTResult(
            outputs=outputs, measured=measured, N=N, M_X=xvq.M_used, R_final=R_final,
            budget=self.budget, budget_win=self.budget_win, budget_quad=self.budget_quad,
            workers=workers, status='ok', theta_hat_status=theta_hat_status,
            theta_Q_status=theta_Q_status, eta_full=float(eta_full), eta_Q=float(eta_Q),
            eta_full_failed=bool(eta_full_failed), eta_Q_failed=bool(eta_Q_failed),
            n_rounds=int(n_rounds), n_measured=n_measured_nodes, n_opened=n_opened,
            n_leaves=len(leaf_ids), max_depth=max_depth, n_failed=int(counter.failed),
            tree_status=tree_status, n_pairs_bought=n_pairs, n_quad_bought=n_quads,
            busy_time=float(busy_time), wall_time=float(wall_time),
            evals_by_stage=evals, rows_by_stage=rows, wall_time_by_stage=wall,
            theta_hat=theta_hat, V_btw=V_btw, V_win=V_win, V_tot=V_tot,
            a_scale=a_scale, a_scatter=a_scatter, root_drift=root_drift, accel=accel,
            b_hat=b_hat, c_q=c_q,
            nodes=_nodes_table(state, outputs, curve_rows),
            leaves=_leaves_table(wr['leaf_rows'], outputs, a_scale),
            curve=_curve_table(curve_rows, outputs, a_scale),
            anchors=_anchors_table(anc['table'], outputs),
            pairs=_pairs_table(wr['pairs'], wr['quads'], outputs, measured),
            psi_hat=psi_points, leaf_of_point=leaf_of_point,
        )
