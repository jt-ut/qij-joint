"""QIJ: derivative-free estimation of the influence variance V with a certified error (the method of the test bed's
bed/METHOD_qij.md, ported from bed/method.py).

INTERFACE (shared with pipeline.py; keep it exactly)

  qij_joint.method.run(T, X, *, eps=0.01, level=0.84, M=None, k=None, pool=None, evaluator=None, seed=0) -> MethodResult

  - T: a statistic object as registry/estimators provide (T(X, w) -> outputs; `outputs`, `eta`, optionally `takes_start`,
    `measured`, `periodic`, `prepare`). The method works on the outputs `T.measured` when T has it, else on all of T.outputs.
    X: (N, d) data (or (N,) for a 1-D sample).
  - evaluation(u), for a weight vector u on the N points, returns sum_i u_i psi_i for every output: the forward difference of
    T along the weights omega = omega0 (1 - t sum u) + t N u, t = delta / max|N u - sum u|, delta =
    differences.forward_step(eta_full), eta_full = core.eta.measure_eta_full at theta-hat = T(X, ones), every refit continued
    from theta-hat. Every evaluation is made on the full data: base weights 1, the base fit theta-hat itself, one refit per
    evaluation. It equals the exact mean influence of the group only up to the step term (the second-order term of the forward
    difference plus the refit tolerance, of relative order sqrt(eta_full)). `evaluator` overrides it: a callable u -> (J,) array
    (the test-bed equivalence check with exact influences; the step term is zero then); T may then be any object with `outputs`.
  - pilot: M = the budget rule unless given, M(eps) = 3 (kappa_r / eps)^(r/2), r = min(d, q) with d the dimension of the
    clustering coordinates (core.pilot.pilot_budget; spec/METHOD_pilot_budget.md; 300 at r = 2, eps = 0.01; independent of N);
    cells by sklearn k-means++ (KMeans(M, n_init=1, random_state=seed)) on the
    clustering coordinates (vq_transform(X)[0] when given, else the standardized X); kernel coordinates = standardized X; CONN
    cell scales as bed/data.geometry (core.pilot). The pilot evaluation of cell c is the evaluation at u = 1{cell c}/n_c, all M
    submitted to the pool at once.
  - k: the cap on splits per round (and on repairs per round); default the pool size (pool.workers), else os.cpu_count(). Each
    batch of evaluations (the pilot; a round's test evaluations; a round's child-mean evaluations; a round's repairs; the J
    report evaluations) is submitted together to the pool. There are no environment switches: parameters only.
  - The bound of a row, per output (scaled units): a measured row (pilot cell, measured child, repaired leaf, test row) has
    bound sqrt(eta_full) on every output (0 with `evaluator`); an inferred sibling has bound (n_p bound_p + n_a bound_a) / n_b
    (the root's bound is sum_c p_c bound_c). The offset Omega_j = sum_leaves p_l bound_l,j^2.
  - The entry criterion and the stop are relative: pred + Omega <= eps q-tilde V_lb and U + Omega <= eps V_lb, output by
    output, V_lb = max(B, F/(1 + sqrt eps)^2), B = sum_l p_l m_l^2 over the leaves, F = |psi|^2 / N of the model (psi-hat
    before the test, the reported model psi-tilde at the test). Nothing assumes V = 1.
  - MethodResult (a dataclass): Vhat (J,), U (J,), delta (J,), V_lb (J,), E (every evaluation, once: theta-hat and
    eta_full's three, M pilot, one per split (the smaller child; the sibling is inferred), the test evaluations, one per repair,
    the J report evaluations), E_full (= E: every evaluation is on the full data), M, K (leaves), rounds, rounds_table (list of
    dicts: round, phase, n_taken, n_repaired, E, K, R_hat, U, delta, offset), leaves_table (list of dicts at the stop: leaf, n,
    mean per output, term per output, determined, bound per output, inferred), timings (dict: pilot_cells_s, pilot_eval_s,
    fit_s, eval_s, total_s, full_refit_mean_s), intervals (None: core.abc's ABC needs a, b, c, not measured), outputs (names);
    eval_error (the redundant rows: count by kind, max and rms per output of value minus the value the active rows imply, in
    units of the output's scale); offset (J,) Omega at the stop; n_inferred (leaves whose mean is inferred at the stop);
    n_repaired; repairs (one dict per repaired leaf: round, leaf, n, inferred, measured, discrepancy = inferred - measured in
    units of s, ratio = |discrepancy| / bound per output, entered); bound_eval (J,) sqrt(eta_full) x s; eta_full.
    Offset, U, delta, V_lb and bound_eval are in the statistic's units.

  Optional keywords beyond the interface (defaults reproduce it): cells=None ((N,) precomputed cell labels, used as they are),
  vq_transform=None (the case's clustering transform), scale=None ((J,) the per-output unit of the ranking and the cut: the
  outputs of a statistic come in different units, and the ranking sums p_l Var_l(psi-hat) over outputs and the two-means cut
  measures distance across them, so each output is divided by its unit first; default sqrt(B_pilot), B_pilot = sum_c p_c m_c^2
  from the pilot evaluations, fixed for the run; the test bed's influences are already in units with V = 1 and the bed
  equivalence check passes ones; the criteria themselves are relative and do not depend on it), record=None (a dict; when given, run stores in it the arrays a check needs: 'cells' (N,),
  'pilot_values' (M, J) the pilot, 'psi_tilde' (N, J) the reported model, 'scale' (J,), 'redundant'),
  pilot_only=False (stop after the pilot: a MethodResult with Vhat None), progress=None (a callable (row, s^2), called with each
  rounds_table entry as it is made, for a log written while the run goes). Each rounds_table entry also carries 'split', the
  leaves split in that round in order, and 'repaired', the leaves repaired in that round in order.

Statement at the stop: Vhat_j <= V_j <= Vhat_j + U_j, one-sided, level `level`, every output, with U_j + Omega_j <= eps V_lb,j.

Rows of psi-hat. One child per split is evaluated (the smaller child a); the sibling b's mean is inferred,
mean_b = (n_p mean_p - n_a mean_a) / n_b. The parent's row leaves; a's row takes the parent's position; b's row is the
combination (n_p row_p - n_a row_a) / n_b with the inferred value. A new row whose weight pattern lies in the span of the active
rows (core/span.py) does not enter psi-hat: its value is compared with the value the active rows imply, and the discrepancy is
reported as eval_error (the free measurement of the evaluation error).

Repair. In the test phase, when the test fails U <= delta - Omega on some output but U <= delta holds on every output (the
offset alone blocks the stop), the round makes no splits: it evaluates directly the inferred leaves of largest
max_outputs p_l bound_l^2 / delta, largest first, as many as it takes to remove enough of Omega (at least one, at most k). A
repaired leaf's direct row (its points, weights 1/n_l, the measured value) takes the position of its combination row; its
bound is then sqrt(eta_full) and it is no longer inferred.

Units. Inside the loop every evaluation is divided by the fixed per-output scale; the ranking (sum over outputs of p_l Var_l of
psi-hat) and the two-means cut (Euclidean in the J outputs) are computed in those units. Every reported quantity is in the
statistic's own units.

One round (bed/METHOD_qij.md section 2). Fit psi-hat if stale. Rank the splittable leaves (n_l >= 2, not unsplittable) by
sum_outputs p_l Var_l(psi-hat). Before the test has begun: the entry criterion (pred <= eps q-tilde V_lb on every output). In
the test phase (every round once it has begun): every leaf lacking a test evaluation gets one centred-Gaussian evaluation, one
refit, cross-fitted misfits by the leave-one-out identity, determined-leaf accounting, Imhof quantiles, U; stop if U <= delta on
every output (U <= delta - Omega); else the repair when it applies. Then the batch: take leaves in ranking order, cut each by the
two-means cut of psi-hat, at least one, at most k; the count is the forecast (before the test: the taken leaves' weights removed
until the remaining sum <= eps q-tilde V_lb - Omega; in the test phase: the number of measured terms that must be removed,
largest max_outputs term/delta first, until it holds against delta - Omega). Each taken leaf costs one child-mean evaluation;
the next round refits once."""
from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import scipy.linalg as sl
import scipy.sparse as sp

from .core.differences import evaluation_weights, forward_step
from .core.interpolant import Interpolant
from .core.pilot import build_cells, cell_weights
from .core.span import Span
from .parallel import call_T

LEV_TOL = 1e-8          # a point is determined when 1 - h_i <= LEV_TOL (h_i its leverage in the row space of the constraints)
TOL_CDF = 1e-9          # the integration tolerance of the distribution function of the quantile


@dataclass
class MethodResult:
    Vhat: np.ndarray
    U: np.ndarray
    delta: np.ndarray
    V_lb: np.ndarray
    E: int
    E_full: float
    M: int
    K: int
    rounds: int
    rounds_table: List[dict] = field(default_factory=list)
    leaves_table: List[dict] = field(default_factory=list)
    timings: Dict[str, float] = field(default_factory=dict)
    intervals: Any = None
    outputs: List[str] = field(default_factory=list)
    eval_error: Any = None
    offset: Any = None
    n_inferred: int = 0
    n_repaired: int = 0
    repairs: List[dict] = field(default_factory=list)
    bound_eval: Any = None
    eta_full: Any = None


# ---- per-leaf sums ----
def _within(A, lab, K, N):
    """(K, J): p_l Var_l(A_j) (population variance about the leaf mean), every leaf in one bincount per output."""
    n = np.maximum(np.bincount(lab, minlength=K), 1).astype(float); q = A.shape[1]
    mean = np.stack([np.bincount(lab, A[:, k], K) for k in range(q)], 1) / n[:, None]
    D = A - mean[lab]
    return np.stack([np.bincount(lab, D[:, k] ** 2, K) for k in range(q)], 1) / N


def _two_means(rows, idx):
    """The two-means partition of the leaf's psi-hat rows (|idx|, J). Deterministic: the initial partition is the sign of the
    projection of the centred rows on their leading principal direction; then Lloyd's iteration (a point moves only when strictly
    closer to the other centre) until no point moves. Returns (a_idx, b_idx), a the smaller child (on equal sizes the child
    holding the leaf's first point), or None if the rows are all equal or the principal direction leaves one side empty."""
    if len(idx) < 2 or np.all(rows.max(0) == rows.min(0)): return None
    Y = rows - rows.mean(0); _, V = np.linalg.eigh(Y.T @ Y); lb = (Y @ V[:, -1]) > 0
    if lb.all() or not lb.any(): return None
    while True:
        c1, c0 = rows[lb].mean(0), rows[~lb].mean(0)
        d1, d0 = ((rows - c1) ** 2).sum(1), ((rows - c0) ** 2).sum(1)
        nl = np.where(d1 < d0, True, np.where(d0 < d1, False, lb))
        if np.array_equal(nl, lb): break
        lb = nl
    a, b = idx[lb], idx[~lb]
    if len(a) > len(b) or (len(a) == len(b) and b[0] < a[0]): a, b = b, a
    return a, b


# ---- the quantile (bed/METHOD_qij.md section 6) ----
def _imhof_cdf(a, x_lo):
    """F(x) = P(sum a_l chi2_1 <= x) (a_l > 0, sum a_l = 1) for x >= x_lo, by Imhof (1961), on composite 8-point Gauss-Legendre
    panels of width 2 up to U, plus the tail term after one integration by parts; U = the smallest 16 * 2^j with theta'(U) < 0 and
    the remainder bound <= TOL_CDF at x = x_lo."""
    def tail(U, x):
        t = a * U; den = 1.0 + t * t
        th1 = 0.5 * np.sum(a / den) - 0.5 * x; th2 = -np.sum(a * a * t / den ** 2)
        g = np.exp(-0.25 * np.log1p(t * t).sum()) / U; gl = -1.0 / U - 0.5 * np.sum(a * t / den)
        h = g / th1; return th1, h, h * (gl - th2 / th1) / th1
    U = 16.0
    while True:
        th1, h, k = tail(U, x_lo)
        if th1 < 0 and abs(k) / np.pi <= TOL_CDF: break
        U *= 2
    npan = int(np.ceil(U / 2.0)); gx, gw = np.polynomial.legendre.leggauss(8)
    u = ((np.arange(npan)[:, None] * 2.0) + (gx[None, :] + 1.0)).ravel(); w = np.tile(gw, npan)
    S = np.zeros_like(u); lr = np.zeros_like(u); t = np.empty_like(u); f_ = np.empty_like(u)
    for al in a:
        np.multiply(u, al, out=t); S += np.arctan(t, out=f_); t *= t; lr += np.log1p(t, out=t)
    S *= 0.5; lr *= 0.25; wt = w / (u * np.exp(lr)); SU = 0.5 * np.arctan(a * U).sum()

    def F(x):
        th1, h, k = tail(U, x); thU = SU - 0.5 * x * U
        return 0.5 - (np.dot(wt, np.sin(S - 0.5 * x * u)) + h * np.cos(thU)) / np.pi
    return F


def _imhof_quantile(a, alpha):
    """alpha-quantile of Q = sum a_l chi2_1 (sum a_l = 1, alpha <= 1/2): brentq on brackets of relative width 0.02, 0.08, 0.32
    around the Welch-Satterthwaite value, then on [a_max q1, 1], which always holds the quantile."""
    from scipy.optimize import brentq
    from scipy.stats import chi2
    nu = 1.0 / np.sum(a * a); x0 = float(chi2.ppf(alpha, nu) / nu); built = None; F = None
    lo_all = float(a.max() * chi2.ppf(alpha, 1))
    for lo, hi in [(x0 * (1.0 - wd), min(x0 * (1.0 + wd), 1.0)) for wd in (0.02, 0.08, 0.32)] + [(lo_all, 1.0)]:
        lo = max(lo, lo_all)
        if not lo < hi: continue
        if built is None or lo < built: F = _imhof_cdf(a, lo); built = lo
        if F(lo) <= alpha <= F(hi): return brentq(lambda x: F(x) - alpha, lo, hi, xtol=1e-12, rtol=1e-10)
    raise AssertionError(('no bracket holds the quantile', len(a), alpha))


def wchi2_quantile(w, alpha):
    """alpha-quantile of Q = sum_l w_l rho_l / sum_l w_l, rho_l independent chi2_1. w (L,) or (L, J) -> scalar or (J,).
    Non-positive weights are dropped; one positive weight -> chi2_1 quantile; none -> 1.0."""
    from scipy.stats import chi2
    w = np.asarray(w, float)
    if w.ndim == 2: return np.array([wchi2_quantile(w[:, j], alpha) for j in range(w.shape[1])])
    w = w[w > 0]
    if len(w) == 0: return 1.0
    if len(w) == 1: return float(chi2.ppf(alpha, 1))
    return float(_imhof_quantile(w / w.sum(), alpha))


def wchi2_cdf(w, x):
    """P(sum_l w_l rho_l / sum_l w_l <= x), the law of wchi2_quantile (none -> the point mass at 1)."""
    from scipy.stats import chi2
    w = np.asarray(w, float); w = w[w > 0]
    if len(w) == 0: return float(x >= 1.0)
    if len(w) == 1: return float(chi2.cdf(x, 1))
    return float(_imhof_cdf(w / w.sum(), x)(x))


def _holds(Wf, S, lim, alpha):
    """Does q_j >= S_j / lim_j hold on every output (q_j the alpha-quantile of the weights Wf[:, j])? S = Wf.sum(0), lim (J,).
    q <= 1, so S_j > lim_j fails without computing anything; otherwise q_j >= x iff F_j(x) <= alpha, one distribution-function
    evaluation per output, stopping at the first failure; x <= a_max q1 holds without it."""
    from scipy.stats import chi2
    q1 = float(chi2.ppf(alpha, 1))
    if np.any(S > lim): return False
    for k in np.argsort(-(S / lim), kind='stable'):
        x = float(S[k] / lim[k])
        if S[k] <= 0 or x <= Wf[:, k].max() / S[k] * q1: continue
        if wchi2_cdf(Wf[:, k], x) > alpha: return False
    return True


def _leverage(A, P0, inv, ncell, pts):
    """h_i, the squared norm of the projection of the indicator e_i onto the row space of A, for the points pts (model-free). The
    first P0 rows are the cell means (their span is the cell indicators, contributing 1/n_c); the other rows are replaced by their
    residuals after the cell-mean projection, Rr, whose basis is found by Cholesky with complete pivoting of Rr Rr' (LAPACK dpstrf),
    rank r; h_i = 1/n_c(i) + |L_r^-1 Rr[piv_r, i]|^2."""
    from scipy.linalg.lapack import dpstrf
    N = A.shape[1]; C = len(ncell); At = A[P0:]
    h = 1.0 / ncell[inv[pts]]
    if At.shape[0] == 0: return h
    Ind = sp.csr_matrix((np.ones(N), (np.arange(N), inv)), shape=(N, C))
    Rr = (At - (At @ Ind) @ sp.diags(1.0 / ncell) @ Ind.T).tocsr()
    G = (Rr @ Rr.T).toarray()
    c, piv, r, info = dpstrf(G.T, lower=1, overwrite_a=1); assert info >= 0, info
    if r == 0: return h
    Lr = np.tril(c[:r, :r]); Z = sl.solve_triangular(Lr, Rr[piv[:r] - 1][:, pts].toarray(), lower=True, check_finite=False)
    return h + np.einsum('ij,ij->j', Z, Z)


class _Rows:
    """The constraint rows: permanent id -> (sorted indices, weights, value, combination or None); `order` = the active ids."""

    def __init__(self, N):
        self.N = N; self.idx = []; self.w = []; self.y = []; self.combo = {}; self.order = []; self.active = set(); self._A = None

    def add(self, idx, w, y, at=None, combo=None, active=True):
        """Add a row; at = an active id whose position it takes (that row is deactivated), else it is appended. active=False: a
        redundant row, kept with its value but not a constraint of psi-hat."""
        i = len(self.idx); self.idx.append(np.asarray(idx, np.int64)); self.w.append(np.asarray(w, np.float64)); self.y.append(np.asarray(y, float).reshape(-1))
        if not active: return i
        if combo is not None: self.combo[i] = combo
        if at is None: self.order.append(i)
        else: self.order[self.order.index(at)] = i; self.active.discard(at)
        self.active.add(i); self._A = None; return i

    def A(self):
        if self._A is None:
            o = self.order; ptr = np.concatenate([[0], np.cumsum([len(self.idx[i]) for i in o])])
            self._A = sp.csr_matrix((np.concatenate([self.w[i] for i in o]), np.concatenate([self.idx[i] for i in o]).astype(np.int32), ptr), shape=(len(o), self.N))
            self._A.has_sorted_indices = True
        return self._A

    def Y(self):
        return np.stack([self.y[i] for i in self.order])


# ---- evaluations ----
def _dense(u, N):
    idx, w = u; out = np.zeros(N); out[idx] = w
    return out


def _evaluation_task(T, case, XA, task):
    """One evaluation (on the pool or in this process), on the full data. XA (N, d): the data (a 1-D sample as one column).
    The weight vector u = (indices, weights) has the step t = delta / max|N u - sum u| (`differences.evaluation_weights`); the
    weights are 1 - t sum u, plus t N u on u's support; the base fit is theta-hat itself (`start`) and the perturbed fit is one
    refit continued from it. Returns (key, base fit, perturbed fit, failure flag, t, wall time, rows, refits)."""
    key, u, delta, start, eta, flat = task
    t0 = time.perf_counter()
    idx, w = u
    Xn = np.asarray(XA); N = len(Xn)
    su = float(w.sum()); g = np.abs(N * w - su).max()
    if len(idx) < N: g = max(g, abs(su))
    t = float(delta / g)
    omega = np.full(N, 1.0 - t * su); omega[idx] += t * N * w
    rows_T = Xn[:, 0] if flat else Xn
    nfit = 0
    try:
        base = np.asarray(start, float)
        pert = np.asarray(call_T(T, rows_T, omega, start, eta), dtype=float); nfit += 1
        failed = bool(np.any(~np.isfinite(base)) or np.any(~np.isfinite(pert)))
    except Exception:
        base = pert = np.full(len(T.outputs), np.nan); failed = True
    return key, base, pert, failed, t, time.perf_counter() - t0, N, nfit


class _Evaluations:
    """evaluation(u) = sum_i u_i psi_i for a batch of weight vectors: the exact evaluator when given, else forward differences of T
    on the full data (on the pool when given). Counts evaluations, refits and their wall time."""

    def __init__(self, T, X, evaluator, pool, measured):
        self.T, self.X, self.evaluator, self.pool, self.measured = T, X, evaluator, pool, np.asarray(measured, int)
        self.n = 0; self.n_base = 0; self.wall = 0.0; self.counter = None; self.theta = None; self.eta_full = None; self.delta = None
        self.flat = np.asarray(X).ndim == 1
        X2 = np.asarray(X, float); self.XA = X2[:, None] if self.flat else X2
        self.stats = dict(full_refits=0, full_s=0.0)
        per = getattr(T, 'periodic', {}) or {}
        outs = list(T.outputs)
        self.period = np.array([per.get(outs[i], 0.0) for i in self.measured], float) if evaluator is None else None
        if evaluator is None and pool is not None: pool.share(self.XA)

    def base(self):
        """theta-hat = T(X, ones) and eta_full (core.eta), once, in this process."""
        if self.evaluator is not None: return
        from .core.counter import Counter
        from .core.eta import measure_eta_full
        t0 = time.perf_counter(); N = len(self.X)
        self.counter = Counter(self.T, N)
        self.theta = np.asarray(self.counter(self.X, np.ones(N)), float)
        if not np.all(np.isfinite(self.theta)): raise RuntimeError('the full-data fit theta-hat failed')
        self.eta_full, _, _ = measure_eta_full(self.counter, self.X, self.theta)
        self.delta = forward_step(self.eta_full)
        self.n_base = self.counter.evaluations
        self.wall += time.perf_counter() - t0

    def __call__(self, us):
        """(len(us), J) evaluations of the weight vectors us, each given as (sorted point indices, weights), on the full data."""
        if not len(us): return np.zeros((0, len(self.measured)))
        t0 = time.perf_counter(); self.n += len(us); N = len(self.X)
        if self.evaluator is not None:
            out = np.stack([np.asarray(self.evaluator(_dense(u, N)), float) for u in us])
            self.wall += time.perf_counter() - t0; return out
        tasks = [(i, u, self.delta, self.theta, self.eta_full, self.flat) for i, u in enumerate(us)]
        if self.pool is not None:
            res = self.pool.map(_evaluation_task, tasks)
        else:
            res = [_evaluation_task(self.T, None, self.XA, task) for task in tasks]
        for r in res:
            self.counter.add(r[7], r[6] * r[7], int(r[3]))
            self.stats['full_refits'] += r[7]; self.stats['full_s'] += r[5]
        raw = np.stack([r[2] for r in res]); base = np.stack([r[1] for r in res])
        if not (np.all(np.isfinite(raw)) and np.all(np.isfinite(base))):
            raise RuntimeError('an evaluation of the statistic failed (%d of %d)' % (int((~np.isfinite(raw)).any(1).sum()), len(raw)))
        d = (raw - base)[:, self.measured]
        per = self.period > 0
        if per.any():                                         # a periodic output's difference wraps into [-P/2, P/2)
            P = self.period[per]; d[:, per] = (d[:, per] + 0.5 * P) % P - 0.5 * P
        out = d / np.array([r[4] for r in res])[:, None]
        self.wall += time.perf_counter() - t0
        return out


def run(T, X, *, eps=0.01, level=0.84, M=None, k=None, pool=None, evaluator=None, seed=0,
        cells=None, vq_transform=None, scale=None, record=None, pilot_only=False, progress=None) -> MethodResult:
    """The method on one data set (module docstring)."""
    t_start = time.perf_counter()
    alpha = 1.0 - float(level); assert 0.0 < alpha <= 0.5, level
    k_cap = int(k if k is not None else (pool.workers if pool is not None else os.cpu_count())); assert k_cap >= 1
    X = np.asarray(X, float); N = X.shape[0]
    measured = list(getattr(T, 'measured', None) or range(len(T.outputs)))
    outputs = [str(T.outputs[i]) for i in measured]; J = len(measured)
    timings = dict(pilot_cells_s=0.0, pilot_eval_s=0.0, fit_s=0.0, eval_s=0.0, total_s=0.0)

    ev = _Evaluations(T, X, evaluator, pool, measured)
    ev.base()

    # ---- step 1: the pilot ----
    t0 = time.perf_counter()
    pil = build_cells(X, M, vq_transform, cells, seed, eps=eps, q=J)
    timings['pilot_cells_s'] = time.perf_counter() - t0
    C = pil.C; inv = pil.inv; ncell = pil.ncell; pc = ncell / N
    t0 = time.perf_counter(); w_eval0 = ev.wall
    pilot_u = cell_weights(pil)
    Yp = ev(pilot_u)
    timings['pilot_eval_s'] = time.perf_counter() - t0; pilot_wall = ev.wall - w_eval0
    s = np.sqrt(np.maximum((pc[:, None] * Yp ** 2).sum(0), 1e-300)) if scale is None else np.asarray(scale, float).reshape(J)
    s2 = s ** 2
    # the bound of a measured row, per output, in scaled units: sqrt(eta_full) (the step term); 0 with the exact evaluator
    b_eval = np.zeros(J) if evaluator is not None else np.full(J, float(np.sqrt(ev.eta_full)))
    if record is not None:                                    # stored before the loop, so a run that fails still leaves them
        record.update(cells=inv.copy(), pilot_values=Yp.copy(), scale=s.copy())
    if pilot_only:                                            # the pilot only (no tree)
        timings['total_s'] = time.perf_counter() - t_start
        st = ev.stats; timings['full_refit_mean_s'] = st['full_s'] / max(st['full_refits'], 1)
        return MethodResult(Vhat=None, U=None, delta=None, V_lb=None, E=int(ev.n_base + ev.n), E_full=float(ev.n_base + ev.n), M=int(C), K=1,
                            rounds=0, timings=timings, outputs=outputs, bound_eval=b_eval * s, eta_full=ev.eta_full)

    rows = _Rows(N)
    y0 = Yp / s
    for c, (idx, w) in enumerate(pilot_u):
        rows.add(idx, w, y0[c])
    P0 = C
    est = Interpolant(pil.Xs, pil.sigma_pt)

    # leaves: points, label, mean (scaled units), mean row (-1 = the root, whose row is sum_c (n_c / N) row_c), unsplittable, test row,
    # the bound of the mean's row (J,), whether the mean is inferred
    lab = np.zeros(N, int); leaves = [np.arange(N)]; lmean = [(pc[:, None] * y0).sum(0)]; con_of = [-1]; unsp = [False]; t_row = [-1]
    lbound = [(pc[:, None] * np.tile(b_eval, (C, 1))).sum(0)]; linf = [False]
    n_split = 0; n_test = 0; n_repair = 0; repairs = []; rng = np.random.RandomState(seed)
    looked_once = False; rounds_table = []; last = None; passed = False; trunc = False

    # the redundancy rule (core/span.py): a row in the span of the active rows does not enter psi-hat; its value is compared with
    # the value the active rows imply (row . psi-hat after the next fit)
    span = Span(inv, ncell)
    recipe = {-1: [(c, ncell[c] / N) for c in range(C)]}      # rows that are exact combinations of other rows (the root; siblings)
    fitted = set(); direct_new = set()                        # ids stored at the last fit; direct rows added since
    pending = []; redundant = []; disc_of = {}; rel_accepted = []

    def resolve(psi):
        for rid, kind, rel in pending:
            d = rows.y[rid] - rows.w[rid] @ psi[rows.idx[rid]]
            redundant.append((rid, kind, rel, d)); disc_of[rid] = d
        pending.clear()

    A_fit = None                                              # the CSR of the rows of the last fit, for the kernel-metric test

    def fit():
        nonlocal A_fit
        A = rows.A(); y = rows.Y(); t = time.perf_counter()
        psi = est.fit(list(rows.order), A, y, rows.combo); timings['fit_s'] += time.perf_counter() - t
        A_fit = A
        fitted.clear(); fitted.update(rows.order); direct_new.clear()
        resolve(psi)
        return psi

    def expand(rid, coef, out):
        """rid as a combination of rows whose columns are stored or new direct rows (into out); False if it is not."""
        if rid in fitted or rid in direct_new:
            out[rid] = out.get(rid, 0.0) + coef; return True
        rec = recipe.get(rid)
        if rec is None: return False
        return all(expand(j, coef * c, out) for j, c in rec)

    def enter(idx, w, y, kind, at=None, rec=None):
        """Test a new row against the span of the active rows (core/span.py) and, if it passes, in the kernel's metric
        (Interpolant.kernel_test); enter it (taking position `at` when given) or keep it as redundant. Returns (id, entered)."""
        red, rel, data = span.test(idx, w)
        if red:
            rid = rows.add(idx, w, y, active=False); pending.append((rid, kind, rel))
            if rec is not None: recipe[rid] = rec
            return rid, False
        combo = None
        if rec is not None:
            out = {}
            if all(expand(j, c, out) for j, c in rec):
                combo = (list(out.keys()), list(out.values()))
        rid_next = len(rows.idx)                              # the id the row will get
        kred, krel = est.kernel_test(rid_next, idx, w, A_fit, combo=combo)
        if kred:
            rid = rows.add(idx, w, y, active=False); pending.append((rid, kind + '/kernel', krel))
            if rec is not None: recipe[rid] = rec
            return rid, False
        rid = rows.add(idx, w, y, at=at, combo=combo); assert rid == rid_next
        if rec is not None: recipe[rid] = rec
        if combo is None: direct_new.add(rid)
        span.add(rid, data); rel_accepted.append(rel)
        return rid, True

    def V_lb_of(F, p):
        B = (p[:, None] * np.stack(lmean) ** 2).sum(0)
        return np.maximum(B, F / (1.0 + np.sqrt(eps)) ** 2)

    def offset(p):
        """Omega (J,) = sum_leaves p_l bound_l^2 (scaled units)."""
        return (p[:, None] * np.stack(lbound) ** 2).sum(0)

    def test(K, n, p, psi, Om):
        """The test at K leaves (Om: the offset): returns (psi-hat after the refit, info)."""
        nonlocal n_test
        new = []
        for l in range(K):                                    # one centred-Gaussian evaluation for every leaf lacking one (leaf order)
            if n[l] < 2 or t_row[l] >= 0: continue
            idl = leaves[l]; nl = len(idl); v = rng.standard_normal(nl); v = v - v.mean(); new.append((l, idl, v / nl))
        if new:
            vals = ev([(idl, u) for _, idl, u in new]) / s
            entered = False
            for (l, idl, u), y in zip(new, vals):
                t_row[l], ok = enter(idl, u, y, 'test'); entered |= ok
            n_test += len(new)
            if entered: psi = fit()
            else: resolve(psi)
        rl = np.flatnonzero(n[:K] >= 2); L = len(rl); assert L
        # cross-fitted misfits: e_j = alpha_j / (G^-1)_jj for each leaf's test row in psi-hat; a redundant test row is not in
        # psi-hat, so its misfit is its value less the value the active rows imply
        posof = {i: r for r, i in enumerate(rows.order)}
        acc = np.array([t_row[l] in rows.active for l in rl]); ja = np.flatnonzero(acc)
        pos = np.array([posof[t_row[l]] for l in rl[ja]], int)
        Xi = est.inverse_columns(pos) if len(pos) else np.zeros((len(posof), 0))
        e = np.empty((L, J)); col_of = {}
        if len(pos): e[ja] = est.alpha[pos] / Xi[pos, np.arange(len(pos))][:, None]
        for c_, j in enumerate(ja): col_of[int(j)] = c_
        for j in np.flatnonzero(~acc): e[j] = disc_of[t_row[rl[j]]]
        cR = (p[rl] * n[rl])[:, None] * e ** 2
        # determined leaves: met by at least n_l rows, and every point of the leaf in the row space of the active rows
        A = rows.A()
        nrow = np.asarray((((A != 0).astype(np.float64) @ sp.csr_matrix((np.ones(N), (np.arange(N), lab)), shape=(N, K))) > 0).sum(0)).ravel()
        cand = np.flatnonzero(nrow[rl] >= n[rl]); det = np.zeros(L, bool)
        if len(cand):
            sizes = n[rl[cand]].astype(int); hv = _leverage(A, P0, inv, ncell, np.concatenate([leaves[rl[j]] for j in cand]))
            gap = np.maximum.reduceat(1.0 - hv, np.concatenate([[0], np.cumsum(sizes)[:-1]])); det[cand] = gap <= LEV_TOL
        cR_q = cR.copy(); cR_q[det] = 0.0; WhR = cR_q.sum(0)
        # the reported model psi-tilde: on each tested, undetermined leaf the model fitted without that leaf's test row
        psit = psi.copy()
        for j in np.flatnonzero(~det & acc):
            idl = leaves[rl[j]]; psit[idl] -= np.outer(est.leaf_dot(Xi[:, col_of[int(j)]], idl), e[j])
        F_til = (psit ** 2).sum(0) / N
        delta = eps * V_lb_of(F_til, p)
        U = np.full(J, np.inf)
        for kk in np.flatnonzero(WhR <= delta):
            qk = wchi2_quantile(cR_q[:, kk], alpha); U[kk] = WhR[kk] / max(qk, 1e-300) if WhR[kk] > 0 else 0.0
        if det.all(): U = np.zeros(J)
        ok = bool(np.all(U <= delta - Om))                    # the stop: U + Omega <= delta on every output
        return psi, dict(rl=rl, cR=cR, cR_q=cR_q, det=det, delta=delta, passed=ok, psit=psit, R_hat=WhR, U=U, e=e, active=acc)

    def evals_so_far():
        return ev.n_base + ev.n

    def log_row(row):
        rounds_table.append(row)
        if progress is not None: progress(row, s2)

    psi = None; stale = True; n_round = 0
    while True:
        K0 = len(leaves); n = np.bincount(lab, minlength=K0).astype(float); p = n / N
        if stale: psi = fit(); stale = False
        can = (n >= 2) & ~np.array(unsp)
        if not can.any(): trunc = True; break
        Om = offset(p)                                         # the offset over the current leaves
        Wl = _within(psi, lab, K0, N)                         # (K0, J): p_l Var_l(psi-hat): ranking, entry forecast, batch count
        pred = np.where(can, Wl.sum(1), -np.inf)
        tinfo = None; phase = 'test' if looked_once else 'pre'; lim_pre = None
        if not looked_once:                                    # the entry criterion: pred + Omega <= lim_pre
            pw = Wl[n >= 2]; pred6 = pw.sum(0)
            lim_pre = np.maximum(eps * V_lb_of((psi ** 2).sum(0) / N, p) - Om, 0.0)
            trigger = False if (np.any(pred6 > lim_pre) or not np.all(lim_pre > 0)) else _holds(pw, pred6, lim_pre, alpha)
        else: trigger = True
        if trigger:
            looked_once = True; phase = 'test'
            psi, tinfo = test(K0, n, p, psi, Om); last = tinfo
        psi_cut = psi                                          # cuts use psi-hat after this round's refit
        row = dict(round=n_round, phase=phase, n_taken=0, n_repaired=0, E=evals_so_far(), K=K0,
                   R_hat=None if tinfo is None else tinfo['R_hat'] * s2, U=None if tinfo is None else tinfo['U'] * s2,
                   delta=None if tinfo is None else tinfo['delta'] * s2, offset=Om * s2)
        if tinfo is not None and tinfo['passed']:
            log_row(row); passed = True; n_round += 1; break
        # ---- the repair: U <= delta on every output, but U <= delta - Omega fails on some (the offset alone blocks the stop) ----
        cand = [l for l in range(K0) if linf[l]] if tinfo is not None and np.all(tinfo['U'] <= tinfo['delta']) else []
        if cand:
            dl = tinfo['delta']; Ut = tinfo['U']
            tm = {l: p[l] * lbound[l] ** 2 for l in cand}
            cand.sort(key=lambda l: -float((tm[l] / dl).max()))   # largest max_j p_l bound_l,j^2 / delta_j first (stable)
            Om_r = Om.copy(); take = []
            for l in cand:                                     # remove their terms from Omega until U <= delta - Omega holds
                if len(take) >= k_cap: break
                Om_r = Om_r - tm[l]; take.append(l)
                if np.all(Ut <= dl - Om_r): break
            vals = ev([(leaves[l], np.full(len(leaves[l]), 1.0 / len(leaves[l]))) for l in take]) / s
            for l in take:                                     # every row that leaves this round, before any repair row is tested
                if con_of[l] in rows.active: est.leaving(con_of[l])
            for l, y in zip(take, vals):
                idl = leaves[l]; nl = len(idl); old = con_of[l]; o_act = old in rows.active
                if o_act: span.remove(old)                     # the direct row takes the position of the combination row
                rid, r_in = enter(idl, np.full(nl, 1.0 / nl), y, 'repair', at=(old if o_act else None))
                disc = np.asarray(lmean[l]) - y                # inferred - measured, in units of s
                with np.errstate(divide='ignore', invalid='ignore'):
                    ratio = np.abs(disc) / lbound[l]
                repairs.append(dict(round=n_round, leaf=int(l), n=int(nl), inferred=np.asarray(lmean[l]) * s, measured=y * s,
                                    discrepancy=disc, bound=lbound[l].copy(), ratio=ratio, entered=bool(r_in)))
                lmean[l] = y; lbound[l] = b_eval.copy(); linf[l] = False; con_of[l] = rid
            n_repair += len(take); stale = True
            row.update(n_repaired=len(take), E=evals_so_far(), repaired=[int(l) for l in take])
            log_row(row); n_round += 1; continue
        # ---- the batch ----
        order = [int(l) for l in np.argsort(-pred, kind='stable') if can[l]]
        if phase == 'pre':
            fl = np.flatnonzero(n >= 2); Wf = Wl[fl].copy(); frow = {int(l): r for r, l in enumerate(fl)}
        else:
            Wd = np.array(tinfo['cR_q'], float).reshape(-1, J).copy(); j_need = 0
            lim_t = np.maximum(tinfo['delta'] - Om, 0.0)       # the stop's limit: delta - Omega
            for jr in np.argsort(-(Wd / tinfo['delta']).max(1), kind='stable'):
                if j_need >= k_cap: break
                Wd[jr] = 0.0; j_need += 1
                if _holds(Wd, Wd.sum(0), lim_t, alpha): break
            j_need = max(1, min(j_need, k_cap))
        cuts = []; nxt = 0
        while nxt < len(order):
            ls = order[nxt]; nxt += 1
            idx = leaves[ls]; cut = _two_means(psi_cut[idx], idx)
            if cut is None:
                unsp[ls] = True
                if not cuts:                                   # re-rank from the current psi-hat
                    can = (n >= 2) & ~np.array(unsp[:K0])
                    if not can.any(): trunc = True; break
                    pred_sel = np.where(can, _within(psi_cut, lab, K0, N).sum(1), -np.inf)
                    order = [int(l) for l in np.argsort(-pred_sel, kind='stable') if can[l]]; nxt = 0
                continue
            cuts.append((ls,) + cut)
            if len(cuts) >= k_cap: break
            if phase == 'pre':
                Wf[frow[ls]] = 0.0
                hold = _holds(Wf, Wf.sum(0), lim_pre, alpha)
            else:
                hold = len(cuts) >= j_need
            if hold: break
        if trunc or not cuts:
            log_row(row); trunc = True; n_round += 1; break
        # one child per split: the smaller child a is evaluated, all cuts at once; the sibling b's mean is inferred,
        # mean_b = (n_p mean_p - n_a mean_a) / n_b, and its row is the combination (n_p row_p - n_a row_a) / n_b. The parent's
        # row leaves when the children's rows enter. b's column is built as the combination of stored columns when its recipe
        # reaches them (an identity of the weights).
        vals = ev([(a, np.full(len(a), 1.0 / len(a))) for _, a, _ in cuts]) / s
        for ls, _, _ in cuts:                                  # every parent row that leaves this round, before any child is tested
            if con_of[ls] in rows.active: est.leaving(con_of[ls])
        for i_, (ls, a, b) in enumerate(cuts):
            nl = float(len(leaves[ls])); na, nb = len(a), len(b)
            ma = vals[i_]; mb = (nl * np.asarray(lmean[ls]) - na * ma) / nb
            bb = (nl * lbound[ls] + na * b_eval) / nb          # the bound of the inferred sibling
            lbound[ls] = b_eval.copy(); lbound.append(bb); linf[ls] = False; linf.append(True)
            par = con_of[ls]; p_act = par in rows.active        # the parent's mean row leaves when its children's rows enter
            if p_act: span.remove(par)
            ca, a_in = enter(a, np.full(na, 1.0 / na), ma, 'leaf mean', at=(par if p_act else None))
            cb, b_in = enter(b, np.full(nb, 1.0 / nb), mb, 'sibling', at=(par if (p_act and not a_in) else None),
                             rec=[(par, nl / nb), (ca, -na / nb)])
            assert a_in or b_in or not p_act, 'both children of an active row are redundant'
            lmean[ls] = ma; lmean.append(mb); con_of[ls] = ca; con_of.append(cb); t_row[ls] = -1; t_row.append(-1)
            leaves[ls] = a; leaves.append(b); unsp.append(False); lab[b] = len(leaves) - 1
        n_split += len(cuts); stale = True
        row.update(n_taken=len(cuts), E=evals_so_far(), split=[int(c[0]) for c in cuts])
        log_row(row); n_round += 1

    # ---- step 8: the report ----
    K = len(leaves); n = np.bincount(lab, minlength=K).astype(float); p = n / N
    Om = offset(p)
    psit = (last['psit'] if last is not None else psi) * s
    allpts = np.arange(N)
    P = np.diag(ev([(allpts, psit[:, j] / N) for j in range(J)])).copy()   # its support is every point
    F_til = (psit ** 2).sum(0) / N
    Vhat = 2.0 * P - F_til
    if last is not None:
        U = last['U'] * s2; delta = last['delta'] * s2
    else:
        U = np.full(J, np.inf); delta = eps * V_lb_of(F_til / s2, p) * s2
    V_lb = delta / eps
    leaves_table = []
    term = {}; detd = {}
    if last is not None:
        for j, l in enumerate(last['rl']): term[int(l)] = last['cR_q'][j] * s2; detd[int(l)] = bool(last['det'][j])
    for l in range(K):
        leaves_table.append(dict(leaf=l, n=int(n[l]), mean=np.asarray(lmean[l]) * s, term=term.get(l, np.zeros(J)), determined=detd.get(l, False),
                                 bound=lbound[l] * s, inferred=bool(linf[l])))
    E = ev.n_base + ev.n                                      # every evaluation, once: base, pilot, 1 per split, tests, repairs, J
    assert E == ev.n_base + C + n_split + n_test + n_repair + J, (E, ev.n_base, ev.n)
    E_full = E                                                # every evaluation is on the full data
    timings['eval_s'] = ev.wall - pilot_wall                 # theta-hat, eta_full, splits, tests, repairs, report (the pilot is pilot_eval_s)
    st = ev.stats
    timings['full_refit_mean_s'] = st['full_s'] / max(st['full_refits'], 1)
    resolve(psi)
    kinds = {}
    for _, kd, _, _ in redundant: kinds[kd] = kinds.get(kd, 0) + 1
    D = np.array([d for _, _, _, d in redundant]).reshape(-1, J)
    eval_error = dict(n=len(redundant), kinds=kinds,
                      max=(np.abs(D).max(0) if len(D) else np.zeros(J)), rms=(np.sqrt((D ** 2).mean(0)) if len(D) else np.zeros(J)),
                      rel_residual_redundant_max=(max(r[2] for r in redundant) if redundant else None),
                      rel_residual_entered_min=(min(rel_accepted) if rel_accepted else None))
    if record is not None:
        # the last test, per tested leaf: the test row (points, weights, value in scaled units), whether it is an active row of
        # psi-hat, and the misfit e the test assigned (scaled units); for a check of the test's accounting
        if last is not None:
            record['test_rows'] = {int(l): dict(idx=rows.idx[t_row[l]], w=rows.w[t_row[l]], y=np.asarray(rows.y[t_row[l]]), active=bool(a_),
                                                e=np.asarray(last['e'][j]), term=np.asarray(last['cR'][j]), determined=bool(last['det'][j]))
                                   for j, (l, a_) in enumerate(zip(last['rl'], last['active'])) if t_row[l] >= 0}
        record.update(psi_tilde=psit, leaves=lab.copy(), redundant=[(rid, kd, rel, d.copy()) for rid, kd, rel, d in redundant],
                      counts=dict(base=int(ev.n_base), pilot=int(C), splits=int(n_split), tests=int(n_test), repairs=int(n_repair), q=int(J)))
    timings['total_s'] = time.perf_counter() - t_start
    return MethodResult(Vhat=Vhat, U=U, delta=delta, V_lb=V_lb, E=int(E), E_full=float(E_full), M=int(C), K=int(K), rounds=int(n_round),
                        rounds_table=rounds_table, leaves_table=leaves_table, timings=timings, intervals=None, outputs=outputs,
                        eval_error=eval_error, offset=Om * s2, n_inferred=int(sum(linf)), n_repaired=int(n_repair), repairs=repairs,
                        bound_eval=b_eval * s, eta_full=ev.eta_full)
