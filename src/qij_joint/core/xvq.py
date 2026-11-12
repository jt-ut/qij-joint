"""
Quantized data. The method itself (method.py) makes no quantized evaluation: measured 2026-11-07, the quantized refit returns the
influence of a different problem (V-hat off by 4 to 40 % on 16 of 16 shearmix draws, spec/METHOD_as_built.md section 4) and
costs the same wall time as a full refit. What remains here serves the diagnostics: `Representation` (end of the module) is the
quantized representation R of `scripts/quant_check.py` (units as point masses, their loss mismatch at theta-hat, and their
refinement); the cell-means quantized evaluation is what `scripts/pilot_diag.py` and `scripts/scale_check.py` use; and
`_measure_eta_Q` is what `core/eta.py` uses for the full data's reproducibility.

The cell-means quantized evaluation (the cells are `core/pilot.py`'s): the data are
replaced by one row per pilot cell (its native-coordinate mean, base weight M_used * p_j), theta_Q = T(rows, base weights) once,
and the influence at cell j is one forward difference at cell j's weight (`prototype_influences`), every cell's step run through
`pool` when given, mass-centred over the finite cells. Each such refit is on M_used rows, so it costs about M_used / N of a
full-data refit. With `T.takes_start` the rows' reproducibility eta_Q is measured first (`_measure_eta_Q`, three evaluations on
the rows) and the step-doubling self-check runs on the five largest-mass cells (`_step_doubling`, ten evaluations on the rows).
`_measure_eta_Q` is also the full data's reproducibility measure (`core/eta.py`).

The quantizer itself is no longer here: `core/pilot.py` builds the cells with sklearn's k-means++ (no FAISS, no vqlp).
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from ..parallel import call_T
from .differences import forward_step, perturbed_weights, step_parameter


@dataclass
class SurveyRows:
    """The pilot's rows and their self-checks; `eta_Q` and `step_ratio` are NaN when T lacks `takes_start`."""
    rows: np.ndarray          # (M_used, d_x) the rows, one per cell (1-D when the data are 1-D)
    row_field: np.ndarray     # (M_used,) cell index of each row
    omega0: np.ndarray        # (M_used,) base weights M_used * p_j, summing to M_used
    eta_Q: float
    step_ratio: np.ndarray    # (5, q)
    quantized_start: str
    eta_rows: Optional[float] = None  # polish tolerance of every evaluation on `rows`; None = T's own


def _field_step_weights(
    omega0: np.ndarray, row_field: np.ndarray, j: int, p_j: float, t: float,
) -> np.ndarray:
    """Every row of cell j scaled by (1-t) + t/p_j, every other row by (1-t), p_j the cell's own mass fraction."""
    member = (row_field == j).astype(float)
    return (1.0 - t) * omega0 + t * omega0 * member / p_j


def _survey_task(T, case, rows: np.ndarray, task):
    """One task: T at cell j's perturbed weights on the shared rows, through `parallel.call_T`. A failing evaluation is caught
    here. Returns (j, raw evaluation, failure flag, this call's own wall time)."""
    j, omega, start, eta = task
    t0 = time.perf_counter()
    try:
        result = np.asarray(call_T(T, rows, omega, start, eta), dtype=float)
        failed = bool(np.any(np.isnan(result)))
    except Exception:
        result = np.full(len(T.outputs), np.nan)
        failed = True
    return j, result, failed, time.perf_counter() - t0


def _step_prototype(omega0: np.ndarray, p: np.ndarray, delta_f: float, j: int) -> Tuple[float, np.ndarray]:
    """t_j = step_parameter(delta_f, p_j) and omega0 perturbed at cell j alone."""
    member = np.zeros(len(p), dtype=bool)
    member[j] = True
    t_j = step_parameter(delta_f, float(p[j]))
    omega = perturbed_weights(omega0, member, t_j)
    return t_j, omega


def _measure_eta_Q(
    counter, rows: np.ndarray, omega0: np.ndarray, row_field: np.ndarray,
    p: np.ndarray, theta_Q: np.ndarray, eta: Optional[float] = None,
) -> Tuple[float, bool]:
    """The rows' reproducibility of a continuation from theta_Q: two fits at the base weights and one at the largest-mass cell's
    weights perturbed by a relative 1e-6. Returns (eta_Q, failed); on failure eta_Q is the floor (`eta` when given, else
    `counter.eta`), never NaN. On success eta_Q is the largest relative difference among the three fits, floored at that floor."""
    f_a = counter(rows, omega0, start=theta_Q, eta=eta)
    f_b = counter(rows, omega0, start=theta_Q, eta=eta)
    j_max = int(np.argmax(p))
    p_max = float(p[j_max])
    omega_pert = _field_step_weights(omega0, row_field, j_max, p_max,
                                      step_parameter(1e-6, p_max))
    f_c = counter(rows, omega_pert, start=theta_Q, eta=eta)
    floor = counter.eta if eta is None else eta
    if not (np.all(np.isfinite(f_a)) and np.all(np.isfinite(f_b))
            and np.all(np.isfinite(f_c))):
        return float(floor), True
    scale_ab = np.maximum(np.abs(f_a), np.abs(f_b))
    scale_ac = np.maximum(np.abs(f_a), np.abs(f_c))
    diffs = np.concatenate([
        np.abs(f_a - f_b) / np.where(scale_ab > 0, scale_ab, 1.0),
        np.abs(f_a - f_c) / np.where(scale_ac > 0, scale_ac, 1.0),
    ])
    return float(max(np.max(diffs), floor)), False


def _step_doubling(
    counter, rows: np.ndarray, omega0: np.ndarray, row_field: np.ndarray,
    p: np.ndarray, theta_Q: np.ndarray, delta_f: float, q: int,
    eta: Optional[float] = None,
) -> np.ndarray:
    """On the five largest-mass cells, raw responses at step delta_f and 2*delta_f (ten evaluations), ratio per output, expected
    near 2 for a differentiable T."""
    order = np.argsort(-p)[:5]
    step_ratio = np.full((5, q), np.nan)
    for row, j in enumerate(order):
        p_j = float(p[j])
        w1 = _field_step_weights(omega0, row_field, j, p_j, step_parameter(delta_f, p_j))
        w2 = _field_step_weights(omega0, row_field, j, p_j, step_parameter(2.0 * delta_f, p_j))
        r1 = counter(rows, w1, start=theta_Q, eta=eta) - theta_Q
        r2 = counter(rows, w2, start=theta_Q, eta=eta) - theta_Q
        step_ratio[row] = r2 / r1
    return step_ratio


def prototype_influences(
    W_X: np.ndarray, counter, p: np.ndarray, eta: float, pool=None,
    quantized_start: str = 'multistart', theta_hat: Optional[np.ndarray] = None,
    eta_rows: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray, float, SurveyRows]:
    """theta_Q = T(rows, base weights) once (the continuation of theta_hat under `quantized_start='full-data'`, else a fit from
    T's own start), then one forward difference per cell j at t_j = step_parameter(delta_f, p_j), on `pool` when given (the pool's
    shared array becomes `rows`), mass-centred over the finite cells. `rows` = W_X, one row per cell, base weight M_used * p_j.
    With `takes_start` the step uses eta_Q (measured here first) and the step-doubling self-check runs; without it neither extra
    evaluation is spent. Returns (theta_Q, I_proto (M_used, q), busy_delta, SurveyRows); `busy_delta` is 0 with `pool=None`, else
    the tasks' summed wall time less `pool.map`'s own wall clock."""
    M_used = len(p)
    rows = W_X
    row_field = np.arange(M_used)
    omega0 = M_used * p

    if quantized_start == 'full-data':
        theta_Q = np.asarray(counter(rows, omega0, start=theta_hat, eta=eta_rows), dtype=float)
    else:
        theta_Q = np.asarray(counter(rows, omega0), dtype=float)
    q = theta_Q.shape[0]

    if counter.takes_start:
        eta_Q, _ = _measure_eta_Q(counter, rows, omega0, row_field, p, theta_Q, eta_rows)
        delta_f = forward_step(eta_Q)
        step_ratio = _step_doubling(counter, rows, omega0, row_field, p, theta_Q, delta_f, q, eta_rows)
    else:
        eta_Q = float('nan')
        delta_f = forward_step(eta)
        step_ratio = np.full((5, q), np.nan)

    sv = SurveyRows(rows=rows, row_field=row_field, omega0=omega0, eta_Q=eta_Q,
                    step_ratio=step_ratio, quantized_start=quantized_start, eta_rows=eta_rows)

    I_proto = np.empty((M_used, q), dtype=float)
    busy_delta = 0.0
    if pool is None:
        for j in range(M_used):
            t_j, omega = _step_prototype(omega0, p, delta_f, j)
            I_proto[j] = (counter(rows, omega, start=theta_Q, eta=eta_rows) - theta_Q) / t_j
    else:
        pool.share(rows)
        t_j = np.empty(M_used, dtype=float)
        tasks = []
        for j in range(M_used):
            t_j[j], omega = _step_prototype(omega0, p, delta_f, j)
            tasks.append((j, omega, theta_Q, eta_rows))
        t_map0 = time.perf_counter()
        results = pool.map(_survey_task, tasks)
        busy_delta = -(time.perf_counter() - t_map0)
        row_count = rows.shape[0]
        for j, result, failed, wall in results:
            I_proto[j] = (result - theta_Q) / t_j[j]
            counter.add(1, row_count, int(failed))
            busy_delta += wall

    # A failed evaluation at a cell is a missing response: centre over the finite cells only, mass-weighted, per output.
    finite = np.isfinite(I_proto)
    mass = np.sum(np.where(finite, p[:, None], 0.0), axis=0)
    psi_bar = np.sum(np.where(finite, p[:, None] * I_proto, 0.0), axis=0) / mass
    I_proto -= psi_bar[None, :]
    return theta_Q, I_proto, busy_delta, sv


# ---- the quantized representation R of method.py ----
def _two_means_points(Y: np.ndarray):
    """Two-means split of the rows Y (n, dc): initial partition by the sign of the projection on the leading principal
    direction, then Lloyd's iteration until no point moves (a point moves only when strictly closer to the other centre).
    Returns a boolean mask (True = one side), or None when the rows are all equal or one side is empty."""
    if len(Y) < 2 or np.all(Y.max(0) == Y.min(0)): return None
    Z = Y - Y.mean(0); _, V = np.linalg.eigh(Z.T @ Z); lb = (Z @ V[:, -1]) > 0
    if lb.all() or not lb.any(): return None
    while True:
        c1, c0 = Y[lb].mean(0), Y[~lb].mean(0)
        d1, d0 = ((Y - c1) ** 2).sum(1), ((Y - c0) ** 2).sum(1)
        nl = np.where(d1 < d0, True, np.where(d0 < d1, False, lb))
        if np.array_equal(nl, lb): break
        lb = nl
    if lb.all() or not lb.any(): return None
    return lb


class Representation:
    """R: the data as units, each unit a point mass of weight n_u at the mean of its points in the statistic's coordinates.
    It starts from the pilot cells. The distortion of a unit is its loss mismatch at the full-data fit theta-hat,
    Delta_u = sum_{i in u} rho(x_i; theta-hat) - n_u rho(m_u; theta-hat), rho = T.per_point_loss; a one-point unit is exact
    (Delta = 0). `refine()` splits units by two-means on the clustering coordinates, largest |Delta| first, until the total
    sum_u |Delta_u| has dropped to half its value at the start of the call. `labels` (N,) gives each point's unit."""

    def __init__(self, X, Xc, labels, loss, theta):
        X = np.asarray(X, float); self.flat = X.ndim == 1; self.X2 = X[:, None] if self.flat else X
        self.Xc = np.asarray(Xc, float); self.loss = loss; self.theta = np.asarray(theta, float)
        self.labels = np.asarray(labels, np.int64).copy(); U = int(self.labels.max()) + 1
        self.lp = np.asarray(loss(X, self.theta), float)
        order = np.argsort(self.labels, kind='stable'); b = np.r_[0, np.cumsum(np.bincount(self.labels, minlength=U))]
        self.members = [order[b[u]:b[u + 1]] for u in range(U)]
        n = np.diff(b).astype(float)
        means = np.stack([np.bincount(self.labels, self.X2[:, a], U) for a in range(self.X2.shape[1])], 1) / n[:, None]
        lm = np.asarray(loss(means[:, 0] if self.flat else means, self.theta), float)
        self.delta = np.bincount(self.labels, self.lp, U) - n * lm
        self.delta[n <= 1] = 0.0

    @property
    def size(self):
        return len(self.members)

    def total_mismatch(self):
        return float(np.abs(self.delta).sum())

    def _unit_delta(self, pts):
        if len(pts) <= 1: return 0.0
        m = self.X2[pts].mean(0)[None, :]
        return float(self.lp[pts].sum() - len(pts) * float(np.asarray(self.loss(m[:, 0] if self.flat else m, self.theta))[0]))

    def refine(self):
        """Split units, largest |Delta| first, until sum |Delta| has halved. Returns the number of splits."""
        import heapq
        D0 = self.total_mismatch(); D = D0; heap = [(-abs(d), u) for u, d in enumerate(self.delta) if d != 0.0]; heapq.heapify(heap)
        delta = list(self.delta); n_split = 0
        while D > 0.5 * D0 and heap:
            negd, u = heapq.heappop(heap)
            pts = self.members[u]
            side = _two_means_points(self.Xc[pts])
            if side is None:
                continue
            a, b = pts[side], pts[~side]; v = len(self.members)
            self.members[u] = a; self.members.append(b); self.labels[b] = v
            da, db = self._unit_delta(a), self._unit_delta(b)
            D += abs(da) + abs(db) - abs(delta[u]); delta[u] = da; delta.append(db); n_split += 1
            if da != 0.0: heapq.heappush(heap, (-abs(da), u))
            if db != 0.0: heapq.heappush(heap, (-abs(db), v))
        self.delta = np.asarray(delta)
        return n_split
