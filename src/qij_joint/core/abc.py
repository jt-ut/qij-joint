"""
The ABC interval (spec/QIJ_mods_waves.md A10): the BCa interval's
second-order correction from a handful of derivatives of T at the data,
in place of B bootstrap replicates. Formulas from DiCiccio & Efron
(1992, JASA 87:421, "More accurate confidence intervals in exponential
families"), DiCiccio & Efron (1996, Statistical Science 11:189,
"Bootstrap confidence intervals" -- the same algorithm restated for a
general reader, eq. numbers below are this paper's), and Efron &
Tibshirani (1993, An Introduction to the Bootstrap, ch. 22: the
nonparametric ABC and its quadratic form ABC_q).

QIJ <-> book, once and for all. The book's resampling vector P sums to
1 (mass p_i on point/cell i); every ABC quantity is built from T_i, the
directional derivative of T at P along the mass-centred direction
e_i - P (DiCiccio & Efron 1996 eq. 6.5 for equal masses; eq. 4.9-4.11's
general exponential-family form, direction Sigma-hat*tdot, for unequal
masses -- a multinomial with cell probabilities P is itself such a
family). QIJ's weights sum to N (or, on the survey rows, to M_X or R)
and psi_i is defined as N*dT/dw_i (method_notes.md). Moving weight i by
N*dw in QIJ's convention traces the SAME curve of distributions as
moving mass i by dw in the book's, so d(book T)/deps along e_i - P
equals N*dT/dw_i - N*sum_k p_k*dT/dw_k = psi_i - sum_k p_k*psi_i(k):
QIJ's mass-centred psi (or I_j, or bin U_k, all already centred where
they are built) IS the book's T_i, exactly, no extra factor, for ANY
partition and ANY masses -- this is what lets `ivq.bias_and_acceleration`
read a and b_hat straight off the bin-level U_k/D2_k at the full data's
N. `curvature` below is different: its LEAST-FAVORABLE DIRECTION is
read off the SURVEY's own M_X-cell problem (a cheap, second-order-only
stand-in for a full-data direction, spec A10: an error of a few percent
here is below what the interval resolves), but its own "n" -- the
exponential family whose delta-method sigma-hat sets the curvature's
scale -- is the DATA's N, not M_X: the direction u_j (below) turns out
not to depend on which n the family is taken at, so building it from
the survey's I_j/p costs nothing when the family itself is the
full-data one.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Optional, Sequence, Tuple

import numpy as np
from scipy.stats import norm

from ..parallel import call_T

if TYPE_CHECKING:
    from ..parallel import Pool
    from .xvq import SurveyRows


def _curvature_task(T, case, rows: np.ndarray, task):
    """One +/- evaluation of T on the shared survey rows, for one
    output's curvature direction: `task` = (key, omega, start).
    `parallel.call_T` applies the estimator-protocol rule for `start`
    (interface sheet); this is this task's own failure boundary (R7).
    Returns (key, result, failure flag, this call's own wall time)."""
    key, omega, start, eta = task
    t0 = time.perf_counter()
    try:
        result = np.asarray(call_T(T, rows, omega, start, eta), dtype=float)
        failed = bool(np.any(np.isnan(result)))
    except Exception:
        result = np.full(len(T.outputs), np.nan)
        failed = True
    return key, result, failed, time.perf_counter() - t0


def _cq_from_pair(
    t_plus_c: float, t_minus_c: float, t_far_c: float,
    theta_c: float, eps_i: float, sigma_c: float, N: float,
) -> Tuple[float, bool]:
    """c_q for one output: the central second difference (DiCiccio &
    Efron 1996 eq. 4.10) when both `t_plus_c`/`t_minus_c` are finite,
    else the one-sided fallback (A10 amendment, spec A15) on whichever
    one survived, using `t_far_c` (T at 2*eps on that side; ignored,
    may be NaN, when no fallback is needed). Returns (c_q, one_sided);
    `one_sided` is True whenever the fallback was attempted, whether or
    not `t_far_c` itself was finite."""
    plus_ok = np.isfinite(t_plus_c)
    minus_ok = np.isfinite(t_minus_c)
    if plus_ok and minus_ok:
        raw = (t_plus_c - 2.0 * theta_c + t_minus_c) / eps_i ** 2
        return raw / (2.0 * N * sigma_c), False
    if not (plus_ok or minus_ok):
        return np.nan, False
    if not np.isfinite(t_far_c):
        return np.nan, True
    near = t_plus_c if plus_ok else t_minus_c
    raw = (t_far_c - 2.0 * near + theta_c) / eps_i ** 2
    return raw / (2.0 * N * sigma_c), True


def curvature(
    counter, sv: "SurveyRows", theta_Q: np.ndarray, I_proto: np.ndarray,
    p: np.ndarray, N: int, pool: Optional["Pool"] = None, step_scale: float = 1.0,
    outputs: Optional[Sequence[int]] = None,
) -> Tuple[np.ndarray, np.ndarray, float, np.ndarray]:
    """
    c_q (len(outputs),): the ABC curvature (DiCiccio & Efron 1996 eq.
    4.10), its LEAST-FAVORABLE DIRECTION read off the survey's own
    M_X-cell family (prototype masses `p`, I_jc the book's mass-centred
    T_i for this cell system -- module docstring) but its SCALE the
    DATA's own N, as the book's eq. 4.10 asks for at the full data's
    sample size: u_j = I_jc / sqrt(sum_k p_k*I_kc^2) is eq. 4.9's
    Sigma-hat*tdot/sigma-hat direction made dimensionless -- since
    sqrt(n)*sigma-hat(n) = sqrt(sum_k p_k*I_kc^2) for ANY family size n
    (eq. 4.7's sigma-hat(n) = sqrt(sum_k p_k*I_kc^2/n)), u_j does not
    depend on n at all, so building it from the survey (M_X) costs
    nothing when the target family size is N. sigma_Q,c = sqrt( (1/N) *
    sum_j p_j*I_jc^2 ) is that SAME sum's delta-method sigma-hat at n=N
    (NOT sqrt(V_btw,c), which is not yet available: curvature runs
    right after stage 1). Moving weights multiplicatively,
    omega0*(1 +/- eps*u_j) (the interface sheet's construction), traces
    the SAME curve as moving mass by eps*sqrt(N)*v, v = p_j*I_jc/
    (N*sigma_Q,c) -- so the raw central second difference at step eps,
    divided by (eps^2 * N), equals 2*sigma_Q,c*c_q (eq. 4.10
    rearranged); hence c_q = raw_second_difference / (2*N*sigma_Q,c). A
    prototype j whose I_jc is non-finite (a failed survey fit) is
    excluded from column c's sum_pI2, mass renormalised over the finite
    prototypes -- `core.influence_model.fit_influence_model`'s own
    finite/mass_c convention -- and its own u_j is 0, so it draws no
    perturbation and drops out of every downstream sum.

    `eps` (len(outputs),) = min(eta**0.25, 0.5/max_j|u_j|) * `step_scale`,
    eta the measured eta_Q when finite, else the estimator's declared
    eta: `step_scale=1.0` for the survey's own use, `step_scale=2.0` (a
    function argument, not a second function) for the acceptance run's
    step-doubling check, so both calls share every other line of code.

    `outputs` (spec/QIJ_mods_waves.md A11): the absolute indices of
    `theta_Q`/`I_proto` to measure; every column by default. `theta_Q`/
    `I_proto` stay full width regardless -- `sigma_Q`/`u` are a cheap
    vectorized pass over every output, and `theta_Q` also stands in for
    a pool task's own `start` continuation, which needs every raw
    parameter -- only the loop below, the two evaluations per requested
    output, is restricted. Two evaluations per requested output on
    `sv.rows`, run through `pool` (its tasks carrying `start=theta_Q`,
    the A9 continuation) when given; `theta_Q[c]` stands in for the
    base evaluation (never re-evaluated, E6).

    Failed-curvature rule (A10 amendment, spec A15): when exactly one
    of an output's +/-eps pair is non-finite, c_q falls back to the
    one-sided second difference on the surviving side, with ONE extra
    evaluation at 2*eps on that same side: raw = T(1 +/- 2*eps*u) -
    2*T(1 +/- eps*u) + T(1), same sign as the side that survived. This
    is the SAME `raw / eps**2 / (2*N*sigma_Q,c)` normalisation as the
    central form, with no extra scale factor: writing h = eps*u and
    expanding T around the surviving base point, T(1+2h) - 2*T(1+h) +
    T(1) = h^2*T'' + h^3*T''' + O(h^4), i.e. raw/eps**2 = T'' + O(eps)
    -- the same leading coefficient the central difference gives
    (T(1+h) - 2*T(1) + T(1-h) = h^2*T'' + O(h^4), one order of eps
    better since the odd term cancels between the two sides). The extra
    evaluation runs through `pool` like the others (serial when
    `pool=None`), `start=theta_Q` under the same rule; nothing is
    retried beyond it. If both sides fail, or the extra evaluation is
    itself non-finite, c_q is NaN for that output (the failure counted
    as any other evaluation's is). Returns (c_q, eps, busy, one_sided),
    `c_q`/`eps`/`one_sided` each of length len(outputs), in `outputs`'
    own order; `busy` is 0.0 with `pool=None`. `one_sided` (bool) marks
    every output where this fallback was attempted (whether or not the
    extra evaluation itself succeeded); False where both sides agreed
    or both failed outright.
    """
    theta_Q = np.asarray(theta_Q, dtype=float)
    I_proto = np.asarray(I_proto, dtype=float)
    p = np.asarray(p, dtype=float)
    N = float(N)
    if outputs is None:
        outputs = range(theta_Q.shape[0])
    outputs = list(outputs)
    n = len(outputs)

    # A non-finite I_proto row (a failed survey fit) is excluded from
    # sum_pI2/u per column, mass renormalised over the finite prototypes
    # -- `core.influence_model.fit_influence_model`'s own per-column
    # finite/mass_c convention, matched here so one failed survey fit no
    # longer NaNs c_q on every output. Only a column that actually HAS a
    # non-finite row takes this path: `p.sum()` need not be exactly 1.0
    # in floating point, so dividing every column by its own mass (even
    # a column with nothing to exclude) would perturb c_q in the last
    # bit everywhere, not just where a survey fit failed.
    finite = np.isfinite(I_proto)
    needs_fix = ~np.all(finite, axis=0)
    I_safe = np.where(finite, I_proto, 0.0)
    sum_pI2_raw = np.sum(p[:, None] * I_proto ** 2, axis=0)
    sum_pI2_safe = np.sum(p[:, None] * I_safe ** 2, axis=0)
    mass = np.sum(np.where(finite, p[:, None], 0.0), axis=0)
    safe_mass = np.where(mass > 0.0, mass, 1.0)
    sum_pI2 = np.where(needs_fix, sum_pI2_safe / safe_mass, sum_pI2_raw)
    sigma_Q = np.sqrt(sum_pI2 / N)
    safe_sum_pI2 = np.where(sum_pI2 > 0.0, sum_pI2, 1.0)
    u = I_safe / np.sqrt(safe_sum_pI2)[None, :]  # 0 on a non-finite row: no perturbation there
    # eta_Q is measured only for an estimator with restarts; a
    # deterministic optimizer's accuracy on the survey rows is its
    # declared eta.
    eta = sv.eta_Q if np.isfinite(sv.eta_Q) else counter.eta
    eps_full = np.minimum(eta ** 0.25, 0.5 / np.max(np.abs(u), axis=0)) * step_scale
    eps = eps_full[outputs]
    row_u = u[sv.row_field]

    c_q = np.empty(n)
    one_sided = np.zeros(n, dtype=bool)
    busy = 0.0
    if pool is None:
        for i, c in enumerate(outputs):
            omega_plus = sv.omega0 * (1.0 + eps[i] * row_u[:, c])
            omega_minus = sv.omega0 * (1.0 - eps[i] * row_u[:, c])
            t_plus = np.asarray(counter(sv.rows, omega_plus, start=theta_Q, eta=sv.eta_rows), dtype=float)
            t_minus = np.asarray(counter(sv.rows, omega_minus, start=theta_Q, eta=sv.eta_rows), dtype=float)
            plus_ok = np.isfinite(t_plus[c])
            minus_ok = np.isfinite(t_minus[c])
            t_far_c = np.nan
            if plus_ok != minus_ok:
                sign = 1.0 if plus_ok else -1.0
                omega_far = sv.omega0 * (1.0 + sign * 2.0 * eps[i] * row_u[:, c])
                t_far = np.asarray(counter(sv.rows, omega_far, start=theta_Q, eta=sv.eta_rows), dtype=float)
                t_far_c = t_far[c]
            c_q[i], one_sided[i] = _cq_from_pair(
                t_plus[c], t_minus[c], t_far_c, theta_Q[c], eps[i], sigma_Q[c], N,
            )
    else:
        pool.share(sv.rows)
        tasks = []
        for i, c in enumerate(outputs):
            omega_plus = sv.omega0 * (1.0 + eps[i] * row_u[:, c])
            omega_minus = sv.omega0 * (1.0 - eps[i] * row_u[:, c])
            tasks.append((('+', c), omega_plus, theta_Q, sv.eta_rows))
            tasks.append((('-', c), omega_minus, theta_Q, sv.eta_rows))
        t_map0 = time.perf_counter()
        results = pool.map(_curvature_task, tasks)
        busy = sum(r[3] for r in results) - (time.perf_counter() - t_map0)
        row_count = sv.rows.shape[0]
        by_key = {}
        for key, result, failed, _ in results:
            counter.add(1, row_count, int(failed))
            by_key[key] = result

        # One extra 2*eps evaluation per output whose +/-eps pair split
        # (one side finite, one not) -- the failed-curvature rule above;
        # batched into its own pool.map so an output with no split costs
        # nothing beyond the two evaluations already made.
        extra_tasks = []
        for i, c in enumerate(outputs):
            plus_ok = np.isfinite(by_key[('+', c)][c])
            minus_ok = np.isfinite(by_key[('-', c)][c])
            if plus_ok != minus_ok:
                sign = 1.0 if plus_ok else -1.0
                omega_far = sv.omega0 * (1.0 + sign * 2.0 * eps[i] * row_u[:, c])
                extra_tasks.append((('2', c), omega_far, theta_Q, sv.eta_rows))
        if extra_tasks:
            t_map1 = time.perf_counter()
            extra_results = pool.map(_curvature_task, extra_tasks)
            busy += sum(r[3] for r in extra_results) - (time.perf_counter() - t_map1)
            for key, result, failed, _ in extra_results:
                counter.add(1, row_count, int(failed))
                by_key[key] = result

        for i, c in enumerate(outputs):
            far = by_key.get(('2', c))
            t_far_c = far[c] if far is not None else np.nan
            c_q[i], one_sided[i] = _cq_from_pair(
                by_key[('+', c)][c], by_key[('-', c)][c], t_far_c,
                theta_Q[c], eps[i], sigma_Q[c], N,
            )

    return c_q, eps, busy, one_sided


def _tail_z(level: float) -> Tuple[float, float]:
    """The two-sided level's tail standard-normal quantiles
    (z_{alpha/2}, z_{1-alpha/2}), alpha = 1-level, the package's level
    convention (`result._normal_interval`)."""
    alpha = 1.0 - level
    return float(norm.ppf(alpha / 2.0)), float(norm.ppf(1.0 - alpha / 2.0))


def abc_interval(
    theta_hat: np.ndarray, sigma: np.ndarray, a: np.ndarray, b: np.ndarray,
    c: np.ndarray, level: float,
) -> np.ndarray:
    """
    (q, 2) [lo, hi]: the ABC_q interval (DiCiccio & Efron 1996 eq.
    4.12-4.13, the quadratic ABC), vectorized over q. z0 = a + c - b/sigma
    is eq. 4.12's own linear form (not the exact Phi^-1(2*Phi(a)*Phi(c) -
    b/sigma): the paper gives that as the computational definition of
    z0, and it is what the nonparametric ABC algorithm evaluates). At
    each tail's z_alpha: w = z0+z_alpha, lambda = w/(1-a*w)^2,
    xi = lambda + c*lambda^2, endpoint = theta_hat + sigma*xi (eq.
    4.13); `a`, `b`, `c` are `ivq.bias_and_acceleration`'s `a`/`b_hat`
    and this module's `c_q`, `sigma` = sqrt(V_btw).
    """
    theta_hat = np.asarray(theta_hat, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    c = np.asarray(c, dtype=float)
    z0 = a + c - b / sigma
    z_lo, z_hi = _tail_z(level)

    def endpoint(z_alpha: float) -> np.ndarray:
        w = z0 + z_alpha
        lam = w / (1.0 - a * w) ** 2
        xi = lam + c * lam ** 2
        return theta_hat + sigma * xi

    return np.stack([endpoint(z_lo), endpoint(z_hi)], axis=-1)


def _z0_from_replicates(replicates: np.ndarray, theta_hat: np.ndarray) -> np.ndarray:
    """z0 = Phi^-1(#{replicate < theta_hat}/B) (eq. 2.8), per output,
    over the non-NaN replicates."""
    replicates = np.asarray(replicates, dtype=float)
    theta_hat = np.asarray(theta_hat, dtype=float)
    B = np.sum(~np.isnan(replicates), axis=0)
    prop = np.nansum(replicates < theta_hat[None, :], axis=0) / B
    return norm.ppf(prop)


def _replicate_quantiles(replicates: np.ndarray, p_lo: np.ndarray, p_hi: np.ndarray) -> np.ndarray:
    """(q, 2) [lo, hi]: `replicates`' own (p_lo, p_hi) quantiles per
    output; one `nanquantile` call per output since each output's
    probability differs (not batchable in one numpy call, q is small)."""
    replicates = np.asarray(replicates, dtype=float)
    q = replicates.shape[1]
    lo = np.array([np.nanquantile(replicates[:, i], p_lo[i]) for i in range(q)])
    hi = np.array([np.nanquantile(replicates[:, i], p_hi[i]) for i in range(q)])
    return np.stack([lo, hi], axis=-1)


def bc_interval(replicates: np.ndarray, theta_hat: np.ndarray, level: float) -> np.ndarray:
    """(q, 2) [lo, hi]: the bias-corrected percentile interval (DiCiccio
    & Efron 1996 eq. 2.3 with a=0, z0 from eq. 2.8): the endpoint's
    percentile in the replicate distribution is Phi(2*z0 + z_alpha)."""
    z0 = _z0_from_replicates(replicates, theta_hat)
    z_lo, z_hi = _tail_z(level)
    p_lo = norm.cdf(2.0 * z0 + z_lo)
    p_hi = norm.cdf(2.0 * z0 + z_hi)
    return _replicate_quantiles(replicates, p_lo, p_hi)


def bca_interval(
    replicates: np.ndarray, theta_hat: np.ndarray, a: np.ndarray, level: float,
) -> np.ndarray:
    """(q, 2) [lo, hi]: the full BCa interval (DiCiccio & Efron 1996 eq.
    2.3), z0 as in `bc_interval` (eq. 2.8) but with the GIVEN
    acceleration `a` (e.g. an analytic influence's) in place of 0: the
    endpoint's percentile is Phi(z0 + (z0+z_alpha)/(1-a*(z0+z_alpha))).
    Never estimates `a` itself (no jackknife runs here)."""
    a = np.asarray(a, dtype=float)
    z0 = _z0_from_replicates(replicates, theta_hat)
    z_lo, z_hi = _tail_z(level)

    def pctile(z_alpha: float) -> np.ndarray:
        w = z0 + z_alpha
        return norm.cdf(z0 + w / (1.0 - a * w))

    return _replicate_quantiles(replicates, pctile(z_lo), pctile(z_hi))
