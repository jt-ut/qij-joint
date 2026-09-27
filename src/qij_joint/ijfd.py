"""
`IJFD` (the infinitesimal jackknife by finite differences): the exact
infinitesimal jackknife computed the only way a black box allows, one
perturbed fit per observation -- the reference `QIJ` approximates and
the cost it avoids (spec A13). A method like the others: `IJFD(
point_curvature=False).fit(X, T, pool=None)` runs one draw.

GOVERNING SPEC. The scratch copy's `spec/QIJ_mods_waves.md` has no A13
(checked: no such heading, no occurrence of "IJFD" or "infinitesimal
jackknife" anywhere in it -- stale, per the coordinator). The real A13
is `/Users/jtaylor/Dropbox/Software/JT_Py_Pkgs/qij_joint/spec/
QIJ_mods_waves.md` lines 595-642 (read-only; not part of this working
copy). This module implements that real A13, plus the coordinator's
phase-2 `point_curvature` extension and the planner's ruling on it
(below); A9/A10, which the scratch copy DOES carry correctly, supply
the shared machinery A13 itself points to.

The draw (A13's own five steps):

  1. theta_hat = T(X, ones(N)), cold (no `start`).
  2. eta_full: the full data's own reproducibility, measured by
     `core.eta.measure_eta_full` (A13 step 2 / A15), every point its own
     "field", p_i = 1/N, rounded up to a power of ten; `(T.eta, 0)` with
     no evaluation for an estimator without `takes_start`.
  3. delta_f = forward_step(eta_full) (A9's ported rule) when
     `point_curvature` is off, else central_step(eta_full) (module
     docstring's `point_curvature` section explains why), then one
     fit per point at that step, `start=theta_hat` (A9 item 2), through
     `pool` when given (`core.xvq._step_prototype`'s own weight
     construction; every point takes the same step since every p_i is
     1/N). This gives T(+t_i); mass-centred per output the way
     `xvq.prototype_influences` centres I_proto.
  4. V_c = sum_i psi_ic^2 / N^2, and the ABC ingredients (A10's a/b/c)
     at point-level masses, the curvature evaluations along psi on the
     full data, two per output (A13's own words), restricted to the
     MEASURED outputs when T carries `measured` (spec A11, qij.py's own
     restriction of `core.abc.curvature`) -- see `point_curvature` below
     for how this module actually spends (or skips) that cost.
  5. Step-doubling (A9 item 4) on the ten heaviest points; every point
     carries the same mass here, so this is the first ten indices, as
     the coordinator's build prompt directs.

`point_curvature` (the coordinator's phase-2 addition; A13 itself does
not name it -- amended by the planner's ruling of the same day):

  OFF (the package default). `psi` is the forward-difference influence
  of step 3, `V` from it, exactly A13's base draw. `a`, `b_hat`, `c_q`
  are ALL NaN: none of the ABC ingredients' curvature evaluations run
  (a fabricated value is worse than NaN -- the planner's ruling, and
  the reason the OLD version of this module, which broadcast a single
  full-data curvature to every point as a `b_hat` stand-in, is gone).
  `abc_interval` is therefore NaN throughout; `normal_interval` still
  works (it only needs `theta_hat`/`V`). Evaluations: N + 1 + the
  check's (`evals_by_stage['abc']` is 0).

  ON. Step 3 itself switches to central_step(eta_full) (see below), and
  one backward-step fit per point is added (N more evaluations, through
  the pool, `start=theta_hat`, at -t_i, the SAME per-point step t_i
  step 3 now uses): together with step 3's T(+t_i) this is the
  three-point stencil `core/differences.py` uses everywhere else in the
  package (`T(+t) - 2*t0 + T(-t)` for the second difference,
  `(T(+t)-T(-t))/(2t)` for the first) -- at central_step's own step
  size, `ivq.bin_differences`'s choice for exactly this central-stencil
  role: forward_step balances a ONE-SIDED difference's O(t) truncation
  against O(eta/t) noise (t ~ sqrt(eta)), far too small once the SECOND
  difference divides by t^2, which amplifies noise as O(eta/t^2) and so
  needs the larger, eta^(1/3)-scaled step. The STORED `psi` is
  this central difference (replacing the forward one, not stored
  alongside it -- the planner's ruling); `V` is computed from it. `a`
  (Sum psi^3, A10's table) and `b_hat` (A10's ported B_hat, now a REAL
  per-point Δ²T_i/t_i^2 from the stencil, not an approximation) both
  reuse `ivq.bias_and_acceleration` at point-level masses (p_i=1/N, U_i =
  the central psi); `c_q` is the two full-data curvature evaluations
  per MEASURED output along that same psi (`core.abc.curvature`'s
  construction, its survey M_X/sigma_Q replaced by the full data's
  N/sqrt(V), eta_Q by eta_full). Evaluations: 2N + 1 + the check's +
  the curvature's 2*len(measured) (`evals_by_stage['backward']` and
  `['abc']` both nonzero, broken out so both costs -- the stencil's and
  the curvature's -- can be read separately from the base N). `psi`,
  `D2` are still computed at every output's own width -- one fit already
  returns them all, spec A11 -- only the curvature's own evaluation loop
  and the reported `V`/`a`/`b_hat`/`c_q` are restricted to `measured`.

A NaN at either of a point's fits (forward always; backward too when
`point_curvature`) leaves that point's psi (and, ON, its Δ²T/t^2) NaN for
the outputs the failing fit returned NaN for; summing into V then NaNs
only those outputs (numpy's NaN propagation through the column sum),
so one point's failure never voids an output it did not touch. A13's
own words: "A NaN at any point fails the output's V ... the draw's
other outputs stand." `nan_fraction` is the fraction of perturbed fits
(forward, and backward when run) that returned NaN.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np

from .core import abc as core_abc
from .core.abc import _curvature_task
from .core.counter import Counter
from .core.differences import central_step, forward_step
from .core.eta import measure_eta_full
from .core.ivq import BinSet, bias_and_acceleration
from .core.xvq import _step_prototype
from .parallel import call_T
from .result import _normal_interval

_N_CHECK_POINTS = 10


@dataclass
class IJFDResult:
    """One draw's IJFD fit. `outputs` (spec A11's `measured` subset,
    identity when T carries none) is what every per-output array below
    is ordered by and sized to, EXCEPT `psi`, which stays at
    `outputs_full`'s full T-output width: the comparison against `ij`'s
    own full-width `psi` needs every output, and centering/differencing
    it costs nothing extra at that width, unlike the ABC curvature's own
    two evaluations per reported output. `psi` (N, q_full) is the
    forward-difference influence of A13's base draw when
    `point_curvature` was off, or the central-difference influence when
    it was on (module docstring) -- never both; `V` is computed from
    whichever `psi` this is, then restricted to `outputs`."""

    outputs: Tuple[str, ...]
    outputs_full: Tuple[str, ...]
    N: int
    point_curvature: bool
    theta_hat: np.ndarray          # (q,) q = len(outputs)
    V: np.ndarray                  # (q,) sum psi^2 / N^2, from THIS draw's psi
    a: np.ndarray                  # (q,) ABC acceleration (A10); NaN unless point_curvature
    b_hat: np.ndarray              # (q,) ABC second-order bias (A10); NaN unless point_curvature
    c_q: np.ndarray                # (q,) ABC curvature-along-psi (A10); NaN unless point_curvature
    eta_full: float                # step 2
    evals_by_stage: Dict[str, int]  # 'cold','eta_full','forward','backward','abc','check','total'
    rows_by_stage: Dict[str, int]
    wall_time_total: float
    busy_time_total: float
    workers: int
    step_ratio: np.ndarray         # (10, q); step 5
    nan_fraction: float            # perturbed fits that returned NaN, over all such fits run
    psi: np.ndarray                # (N, q_full); see the class docstring

    @property
    def variance(self) -> np.ndarray:
        """V per output (q,): the point-level finite-difference
        variance the interval rests on."""
        return self.V

    def normal_interval(self, level: float) -> np.ndarray:
        """(q, 2) [lo, hi]: theta_hat +/- z*sqrt(V) (mirrors
        `QIJResult.interval`); needs neither `a`/`b_hat`/`c_q` nor
        `point_curvature`."""
        return _normal_interval(self.theta_hat, self.V, level)

    def abc_interval(self, level: float) -> np.ndarray:
        """(q, 2) [lo, hi]: the ABC_q interval from the stored
        `a`/`b_hat`/`c_q` and sigma = sqrt(V) (mirrors
        `QIJResult.abc_interval`, spec A10). All-NaN when this draw ran
        with `point_curvature=False` (module docstring)."""
        sigma = np.sqrt(np.maximum(self.V, 0.0))
        return core_abc.abc_interval(self.theta_hat, sigma, self.a, self.b_hat, self.c_q, level)


def _point_task(T, case, X: np.ndarray, task):
    """One point's +/-step fit against the shared full data (task =
    (i, omega, start)); `parallel.call_T` applies the estimator's
    `start` rule. This task's own failure boundary. Returns (i, result,
    failure flag, this call's own wall time). Shared by the forward
    pass (step 3) and the backward pass (`point_curvature`)."""
    i, omega, start = task
    t0 = time.perf_counter()
    try:
        result = np.asarray(call_T(T, X, omega, start), dtype=float)
        failed = bool(np.any(np.isnan(result)))
    except Exception:
        result = np.full(len(T.outputs), np.nan)
        failed = True
    return i, result, failed, time.perf_counter() - t0


def _failed_result(outputs: Tuple[str, ...], outputs_full: Tuple[str, ...], N: int, workers: int,
                    counter: Counter, t_start: float, point_curvature: bool) -> IJFDResult:
    """The cold fit (step 1) itself came back NaN: nothing past it ran,
    so every downstream quantity is NaN/empty and only step 1's own
    evaluation is counted. `outputs` (spec A11's measured subset) sizes
    every per-output field except `psi`, which stays at `outputs_full`'s
    width (class docstring)."""
    q, q_full = len(outputs), len(outputs_full)
    ev, rows = counter.snapshot()
    wall = time.perf_counter() - t_start
    evals = {'cold': ev, 'eta_full': 0, 'forward': 0, 'backward': 0, 'abc': 0, 'check': 0, 'total': ev}
    rows_by = {'cold': rows, 'eta_full': 0, 'forward': 0, 'backward': 0, 'abc': 0, 'check': 0, 'total': rows}
    return IJFDResult(
        outputs=outputs, outputs_full=outputs_full, N=N, point_curvature=point_curvature,
        theta_hat=np.full(q, np.nan), V=np.full(q, np.nan),
        a=np.full(q, np.nan), b_hat=np.full(q, np.nan), c_q=np.full(q, np.nan),
        eta_full=float('nan'), evals_by_stage=evals, rows_by_stage=rows_by,
        wall_time_total=wall, busy_time_total=wall, workers=workers,
        step_ratio=np.full((_N_CHECK_POINTS, q), np.nan), nan_fraction=1.0,
        psi=np.full((N, q_full), np.nan),
    )


def _perturbed_fits(counter: Counter, X: np.ndarray, omega0: np.ndarray, p: np.ndarray,
                     theta_hat: np.ndarray, signed_delta: float, pool) -> Tuple[np.ndarray, np.ndarray, int, float]:
    """One weight-perturbed fit per point at `signed_delta` (positive
    for step 3's forward pass, negative for `point_curvature`'s
    backward pass): `core.xvq._step_prototype`'s weight construction,
    `start=theta_hat` (A9 item 2), through `pool` when given. Returns
    (T(omega(i)) for every point (N, q), the per-point step actually
    used t_i (N,) -- t_i is negative when `signed_delta` is --, points
    failed, busy_delta)."""
    N = len(X)
    q = len(theta_hat)
    T_signed = np.empty((N, q))
    t_of = np.empty(N)
    n_failed = 0
    busy = 0.0
    if pool is None:
        for i in range(N):
            t_of[i], omega = _step_prototype(omega0, p, signed_delta, i)
            result = np.asarray(counter(X, omega, start=theta_hat), dtype=float)
            T_signed[i] = result
            n_failed += int(np.any(np.isnan(result)))
    else:
        pool.share(X)
        tasks = []
        for i in range(N):
            t_of[i], omega = _step_prototype(omega0, p, signed_delta, i)
            tasks.append((i, omega, theta_hat))
        t_map0 = time.perf_counter()
        results = pool.map(_point_task, tasks)
        busy = -(time.perf_counter() - t_map0)
        for i, result, failed, wall in results:
            T_signed[i] = result
            counter.add(1, N, int(failed))
            n_failed += int(failed)
            busy += wall
    return T_signed, t_of, n_failed, busy


def _center_psi(psi_raw: np.ndarray, p: np.ndarray) -> np.ndarray:
    """Mass-centre psi per output, over the finite points only (mirrors
    `xvq.prototype_influences`'s centering of I_proto): a failed point
    is excluded from the centering mass for the outputs it failed, not
    filled or dropped."""
    finite = np.isfinite(psi_raw)
    mass = np.sum(np.where(finite, p[:, None], 0.0), axis=0)
    psi_bar = np.sum(np.where(finite, p[:, None] * psi_raw, 0.0), axis=0) / mass
    return psi_raw - psi_bar[None, :]


def _curvature(counter: Counter, X: np.ndarray, theta_hat: np.ndarray, psi: np.ndarray,
               V: np.ndarray, eta_full: float, N: int, pool, measured) -> Tuple[np.ndarray, float]:
    """
    `c_q`: two full-data evaluations per MEASURED output (spec A11,
    qij.py's own restriction of `core.abc.curvature`'s evaluation loop)
    along psi's own direction (`core.abc.curvature`'s construction, with
    the survey's M_X-cell family replaced by the point family: n = N,
    mass p_i = 1/N, field value I_i = psi_ic): u_i = psi_ic /
    (sqrt(N)*sigma_c) (eq. 4.9's direction, sigma_c = sqrt(V_c) this
    family's own delta-method sigma-hat at n = N, eq. 4.7), eps_c = min(
    eta_full^(1/4), 0.5/max_i|u_ic|), c_q,c = raw_c / (2*N*sigma_c),
    raw_c the central second difference T_c(1+eps*u) - 2*theta_hat_c +
    T_c(1-eps*u), divided by eps^2 (`abc.curvature`'s own
    normalization, its eq. 4.10). A MEASURED output whose sigma_c is NaN
    or 0 (a failed or degenerate V_c), or an output not in `measured`, is
    left NaN and spends no evaluation. Returns (c_q, busy), `c_q` at
    `psi`'s own (full) width -- the caller restricts it to `measured`.
    """
    q = psi.shape[1]
    sigma = np.sqrt(np.maximum(V, 0.0))
    is_measured = np.zeros(q, dtype=bool)
    is_measured[measured] = True
    valid = np.isfinite(sigma) & (sigma > 0) & is_measured
    u = np.zeros((N, q))
    u[:, valid] = psi[:, valid] / (np.sqrt(N) * sigma[valid])
    max_abs_u = np.max(np.abs(u), axis=0)
    eps = np.full(q, np.nan)
    safe_max = np.where(max_abs_u > 0, max_abs_u, 1.0)
    eps[valid] = np.minimum(eta_full ** 0.25, 0.5 / safe_max[valid])

    c_q = np.full(q, np.nan)
    busy = 0.0
    cols = np.where(valid)[0]
    if pool is None:
        for c in cols:
            omega_plus = 1.0 + eps[c] * u[:, c]
            omega_minus = 1.0 - eps[c] * u[:, c]
            t_plus = np.asarray(counter(X, omega_plus, start=theta_hat), dtype=float)
            t_minus = np.asarray(counter(X, omega_minus, start=theta_hat), dtype=float)
            raw = (t_plus[c] - 2.0 * theta_hat[c] + t_minus[c]) / eps[c] ** 2
            c_q[c] = raw / (2.0 * N * sigma[c])
    else:
        pool.share(X)
        tasks = []
        for c in cols:
            omega_plus = 1.0 + eps[c] * u[:, c]
            omega_minus = 1.0 - eps[c] * u[:, c]
            tasks.append((('+', c), omega_plus, theta_hat))
            tasks.append((('-', c), omega_minus, theta_hat))
        t_map0 = time.perf_counter()
        results = pool.map(_curvature_task, tasks)
        busy = sum(r[3] for r in results) - (time.perf_counter() - t_map0)
        by_key = {}
        for key, result, failed, _ in results:
            counter.add(1, N, int(failed))
            by_key[key] = result
        for c in cols:
            t_plus, t_minus = by_key[('+', c)], by_key[('-', c)]
            raw = (t_plus[c] - 2.0 * theta_hat[c] + t_minus[c]) / eps[c] ** 2
            c_q[c] = raw / (2.0 * N * sigma[c])
    return c_q, busy


def _acceleration(psi: np.ndarray, N: int) -> np.ndarray:
    """`a` (A10's table, point level: p_i=1/N, U_i=psi_ic), read off
    `ivq.bias_and_acceleration` (a dummy zero `D2`; `a` never reads
    it, so the formula still has exactly one home)."""
    q = psi.shape[1]
    binset = BinSet(labels=np.arange(N), n=np.ones(N, dtype=int), U=psi,
                     centering_residual=np.zeros(q), D2=np.zeros((N, q)),
                     M_init=N, M_used=N, within_share=float('nan'), failed=False)
    _, a = bias_and_acceleration(binset, N)
    return a


def _bias(psi: np.ndarray, D2: np.ndarray, N: int) -> np.ndarray:
    """`b_hat` (A10's ported B_hat, point level), from a REAL per-point
    `D2` = d2T_i/t_i^2 (the `point_curvature` stencil's own second
    difference, rescaled to its own step t_i -- the same t_i step 3's
    `psi` was built from), via `ivq.bias_and_acceleration`."""
    q = psi.shape[1]
    binset = BinSet(labels=np.arange(N), n=np.ones(N, dtype=int), U=psi,
                     centering_residual=np.zeros(q), D2=D2,
                     M_init=N, M_used=N, within_share=float('nan'), failed=False)
    b_hat, _ = bias_and_acceleration(binset, N)
    return b_hat


def _step_doubling_points(counter: Counter, X: np.ndarray, omega0: np.ndarray, p: np.ndarray,
                           theta_hat: np.ndarray, delta_f: float, q: int) -> np.ndarray:
    """Step 5: on the ten heaviest points -- with every mass equal, the
    first ten indices, as the coordinator's build prompt directs --
    fresh responses at delta_f and 2*delta_f (`core.xvq._step_prototype`
    at each step), ratio per output (mirrors `xvq._step_doubling`, A9
    item 4)."""
    n = min(_N_CHECK_POINTS, len(p))
    step_ratio = np.full((_N_CHECK_POINTS, q), np.nan)
    for i in range(n):
        _, w1 = _step_prototype(omega0, p, delta_f, i)
        _, w2 = _step_prototype(omega0, p, 2.0 * delta_f, i)
        r1 = np.asarray(counter(X, w1, start=theta_hat), dtype=float) - theta_hat
        r2 = np.asarray(counter(X, w2, start=theta_hat), dtype=float) - theta_hat
        step_ratio[i] = r2 / r1
    return step_ratio


class IJFD:
    """The infinitesimal jackknife by finite differences (spec A13;
    module docstring for `point_curvature`)."""

    def __init__(self, point_curvature: bool = False) -> None:
        self.point_curvature = point_curvature

    def fit(self, X: np.ndarray, T, pool=None) -> IJFDResult:
        """Run A13's draw on one draw; the backward pass and the ABC
        ingredients only when `point_curvature` (module docstring)."""
        t_start = time.perf_counter()
        X = np.asarray(X)
        N = len(X)
        outputs_full = T.outputs
        q_full = len(outputs_full)
        # `measured` (spec A11, qij.py's own convention): the indices of
        # `outputs_full` this draw REPORTS at output width; identity when
        # T carries no `measured` attribute, so every `[measured]` slice
        # below is a no-op and the draw is bit-identical to before A11.
        # `psi` is the one field kept at `outputs_full`'s own width
        # regardless (class docstring).
        measured = list(getattr(T, 'measured', range(q_full)))
        outputs = tuple(outputs_full[i] for i in measured)
        q = len(measured)
        workers = pool.workers if pool is not None else 1
        counter = Counter(T, N)
        omega0 = np.ones(N)
        p = np.full(N, 1.0 / N)

        # Step 1.
        theta_hat = np.asarray(counter(X, omega0), dtype=float)
        if np.any(np.isnan(theta_hat)):
            return _failed_result(outputs, outputs_full, N, workers, counter, t_start,
                                   self.point_curvature)
        ev_cold, rows_cold = counter.snapshot()

        # Step 2 (core/eta.py::measure_eta_full, A13 step 2 / A15):
        # every point its own "field" (row_field = its own index) when
        # T takes a start; an estimator without one spends nothing here.
        eta_full, _ = measure_eta_full(counter, X, theta_hat)
        ev_eta, rows_eta = counter.snapshot()

        # Step 3: the step size depends on how this pass will be used.
        # Off, it is the base one-sided influence estimate, sized by
        # forward_step's own O(t)-truncation/O(eta/t)-noise balance. On,
        # it becomes one half of a central stencil whose SECOND
        # difference (D2, the ABC bias) divides by t^2 -- reusing
        # forward_step's much smaller step there amplifies floating-
        # point noise by O(eta/t^2), swamping the true curvature (an
        # estimator as reproducible as a closed-form MLE has eta_full
        # near machine precision, driving t_forward down to ~1e-9);
        # central_step's larger, eta^(1/3)-scaled step is
        # `ivq.bin_differences`'s own choice for exactly this role.
        delta_f = central_step(eta_full) if self.point_curvature else forward_step(eta_full)
        T_plus, t_of, n_fwd_failed, busy_forward = _perturbed_fits(
            counter, X, omega0, p, theta_hat, delta_f, pool)
        ev_fwd, rows_fwd = counter.snapshot()

        if self.point_curvature:
            # The backward half of the three-point stencil, at the SAME
            # per-point step magnitude (negating `delta_f` negates
            # every t_i alike, since every p_i is 1/N).
            T_minus, _t_of_neg, n_bwd_failed, busy_backward = _perturbed_fits(
                counter, X, omega0, p, theta_hat, -delta_f, pool)
            ev_bwd, rows_bwd = counter.snapshot()
            # `_t_of_neg` (== -t_of, step_parameter is linear in delta)
            # is unused: the central difference's denominator, and D2's
            # own t^2 below, use the forward pass's own t_of directly,
            # exact by construction (t_of^2 == (-t_of)^2 regardless).
            d2T = T_plus - 2.0 * theta_hat[None, :] + T_minus
            D2 = d2T / t_of[:, None] ** 2
            psi = _center_psi((T_plus - T_minus) / (2.0 * t_of[:, None]), p)
            V_full = np.sum(psi ** 2, axis=0) / N ** 2
            # The curvature's own evaluation cost (two per output) is
            # spent only on `measured` (spec A11); `a`/`b_hat` are pure
            # arithmetic on the already-computed psi/D2 (no evaluation
            # cost either width), sliced to `measured` below to match.
            c_q_full, busy_curvature = _curvature(
                counter, X, theta_hat, psi, V_full, eta_full, N, pool, measured)
            ev_abc, rows_abc = counter.snapshot()
            a_full = _acceleration(psi, N)
            b_hat_full = _bias(psi, D2, N)
            V, a, b_hat, c_q = V_full[measured], a_full[measured], b_hat_full[measured], c_q_full[measured]
            n_perturbed_failed = n_fwd_failed + n_bwd_failed
            n_perturbed_total = 2 * N
        else:
            ev_bwd, rows_bwd = ev_fwd, rows_fwd
            busy_backward = 0.0
            psi = _center_psi((T_plus - theta_hat[None, :]) / t_of[:, None], p)
            V = np.sum(psi ** 2, axis=0)[measured] / N ** 2
            ev_abc, rows_abc = ev_fwd, rows_fwd
            busy_curvature = 0.0
            a = np.full(q, np.nan)
            b_hat = np.full(q, np.nan)
            c_q = np.full(q, np.nan)
            n_perturbed_failed = n_fwd_failed
            n_perturbed_total = N

        # Step 5, at outputs_full's own width (T's evaluation returns
        # every output regardless), then restricted to `measured` like
        # every other reported per-output field (`psi` stays the
        # exception, class docstring).
        step_ratio_full = _step_doubling_points(counter, X, omega0, p, theta_hat, delta_f, q_full)
        step_ratio = step_ratio_full[:, measured]
        ev_check, rows_check = counter.snapshot()

        wall_time_total = time.perf_counter() - t_start
        busy_time_total = wall_time_total + busy_forward + busy_backward + busy_curvature

        evals_by_stage = {
            'cold': ev_cold, 'eta_full': ev_eta - ev_cold, 'forward': ev_fwd - ev_eta,
            'backward': ev_bwd - ev_fwd, 'abc': ev_abc - ev_bwd,
            'check': ev_check - ev_abc, 'total': ev_check,
        }
        rows_by_stage = {
            'cold': rows_cold, 'eta_full': rows_eta - rows_cold, 'forward': rows_fwd - rows_eta,
            'backward': rows_bwd - rows_fwd, 'abc': rows_abc - rows_bwd,
            'check': rows_check - rows_abc, 'total': rows_check,
        }

        return IJFDResult(
            outputs=outputs, outputs_full=outputs_full, N=N, point_curvature=self.point_curvature,
            theta_hat=theta_hat[measured], V=V, a=a, b_hat=b_hat, c_q=c_q,
            eta_full=eta_full, evals_by_stage=evals_by_stage, rows_by_stage=rows_by_stage,
            wall_time_total=wall_time_total, busy_time_total=busy_time_total, workers=workers,
            step_ratio=step_ratio, nan_fraction=n_perturbed_failed / n_perturbed_total, psi=psi,
        )
