"""
The I-VQ bin machinery shared by the joint second stage and `ijfd`
(spec/method_notes.md sections 1 and 4; spec/QIJ_mods_waves.md A21):
`BinSet` holds one partition's full-data measurement, `bin_differences`
measures a `BinSet`'s bins on the full data, serially or as pool tasks
(method_notes section 4), giving each bin's finite-differenced
influence U_k and its stencil curvature D2_k = d2T_k/t_k^2, t_k the
stencil's own step (spec/QIJ_mods_waves.md A10). `between_terms`
reduces U to V_btw; `bias_and_acceleration` reduces U/D2 to the ABC
interval's bias and acceleration ingredients (A10).

A21 removed this module's own quantizer (`kmeans_1d`, `build_bins`,
`within_share`): both built the marginal path's per-coordinate initial
partition (`refine.py`, removed in the same wave); the joint path's own
partition is grown by `core.joint.grow` directly on psi0_all, never
through this module.

A failed evaluation of a bin (spec section 5) makes the whole draw's
result a write-off: `bin_differences` stops at the bin that failed and
returns a `BinSet` with `failed=True` and `U`/`D2` all NaN.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, replace
from typing import Tuple

import numpy as np

from ..parallel import call_T
from .differences import central_step, difference, perturbed_weights, step_parameter

__all__ = [
    "BinSet", "bin_differences", "between_terms", "bias_and_acceleration",
]


@dataclass
class BinSet:
    """
    One estimand coordinate's I-VQ: the initial bins over psi0_c and
    their full-data measurement.

    labels    (N,) int, bin index per point, contiguous 0..M_used-1.
    n         (M_used,) int, bin sizes, all > 0.
    U         (M_used, q) centered bin influences, (M_used, 0) before
              `bin_differences` has run.
    centering_residual  (q,) the bin-mass-weighted residual sum_k p_k
              U_k subtracted out of U; a later split (`core.joint`)
              subtracts this same residual from every split it
              measures, so a split's U stays on the initial bins'
              centered scale. zeros(0) before `bin_differences` has run.
    D2        (M_used, q) each bin's stencil curvature d2T_k/t_k^2,
              d2T_k = T(+t) - 2*theta_hat + T(-t) the central-stencil
              second difference and t_k = differences.step_parameter(
              delta, p_k) the same step U was built from (spec A10);
              uncentered; (M_used, 0) before `bin_differences` has run.
    M_init    the initial bin count from the count rule.
    M_used    bins actually used (<= M_init; empty bins are dropped).
    within_share  achieved share of Var(psi0_c) left inside the bins.
    failed    True when a bin's full-data evaluation returned any NaN
              (spec section 5.2): U/centering_residual/D2 are then all
              NaN and MUST NOT be read by `between_terms` or
              `bias_and_acceleration`.
    """

    labels: np.ndarray
    n: np.ndarray
    U: np.ndarray
    centering_residual: np.ndarray
    D2: np.ndarray
    M_init: int
    M_used: int
    within_share: float
    failed: bool = False

    @property
    def p(self) -> np.ndarray:
        """Bin masses n / N."""
        return self.n / self.labels.size


def _bin_failed(binset: BinSet, M_used: int, q: int) -> BinSet:
    """Failed measurement: U/centering_residual/D2 NaN, failed=True."""
    return replace(
        binset, U=np.full((M_used, q), np.nan), centering_residual=np.full(q, np.nan),
        D2=np.full((M_used, q), np.nan), failed=True,
    )


def _bin_done(binset: BinSet, U: np.ndarray, D2: np.ndarray, p: np.ndarray) -> BinSet:
    """Completed measurement: U centered by sum_k p_k U_k; D2 stored as
    measured (a curvature needs no centering)."""
    cr = p @ U
    return replace(binset, U=U - cr, centering_residual=cr, D2=D2, failed=False)


def _bin_task(T, case, X: np.ndarray, task):
    """One stencil evaluation on the pool (spec section 4): `task` is
    (bin k, signed step t, member mask, start, eta); `call_T` applies
    the estimator's prepared state, the start-continuation rule (A9)
    and the per-call eta override (spec/QIJ_mods_waves.md A15). Returns
    (k, t, evaluation, failure flag, this call's own wall time)."""
    k, t, mask, start, eta = task
    omega = perturbed_weights(np.ones(len(X)), mask, t)
    t0 = time.perf_counter()
    try:
        result = call_T(T, X, omega, start, eta)
        result = np.asarray(result, dtype=float)
        failed = bool(np.any(np.isnan(result)))
    except Exception:
        result = np.full(len(T.outputs), np.nan)
        failed = True
    return k, t, result, failed, time.perf_counter() - t0


def bin_differences(
    X: np.ndarray, counter, theta_hat: np.ndarray, binset: BinSet, eta: float, pool=None,
    start: np.ndarray = None,
) -> Tuple[BinSet, float]:
    """
    Measure one coordinate's I-VQ on the full data (spec section 4):
    for each bin k, in order, the central stencil against the bin's own
    mass. A NaN in any U_k stops the loop at that bin, returning
    `failed=True` with U/centering_residual/D2 all NaN; otherwise U is
    centered by centering_residual = sum_k p_k U_k, kept on the `BinSet`
    for a later split to reuse; D2 = d2T/t_k^2 (spec A10) is
    stored uncentered (a curvature needs no centering). `start` (A9)
    passes through unchanged to every evaluation below (`counter`/
    `call_T` gate it on `takes_start`); `start=None` reproduces today's
    evaluations exactly. `eta` sets both the stencil step
    (`central_step(eta)`, below) and, via the same per-call override
    `counter`/`call_T` apply for `start` (spec/QIJ_mods_waves.md A15),
    the polish's own acceptance tolerance for every evaluation below --
    `eta_full` on every full-data path that reaches here.

    With a `pool`, every bin's +t/-t evaluations run as pool tasks
    against X (shared by the caller); `pool.map` returns them in task
    order, so `results[2k]`/`results[2k+1]` are bin k's own pair,
    assembled into the same U/D2 above. Counted evaluations match the
    serial loop exactly (every bin before the first failure, plus that
    bin's own two -- `difference` always evaluates both); a later bin's
    task still ran and its wall time is in `busy_delta` regardless.
    Returns (binset, busy_delta); busy_delta is 0.0 with `pool=None`.
    """
    N = binset.labels.size
    theta_hat = np.asarray(theta_hat, dtype=float)
    q = theta_hat.size
    M_used = binset.M_used
    delta = central_step(eta)
    p = binset.p

    if pool is None:
        U = np.empty((M_used, q))
        D2 = np.empty((M_used, q))
        for k in range(M_used):
            mask = binset.labels == k

            def evaluate(t: float, _mask=mask) -> np.ndarray:
                return counter(X, perturbed_weights(np.ones(N), _mask, t), start=start, eta=eta)

            U_k, D2_k = difference(float(p[k]), delta, evaluate, theta_hat)
            if np.any(np.isnan(U_k)):
                return _bin_failed(binset, M_used, q), 0.0
            U[k] = U_k
            D2[k] = D2_k
        return _bin_done(binset, U, D2, p), 0.0

    t_bin = [step_parameter(delta, float(pk)) for pk in p]
    tasks = [(k, sign * t_bin[k], binset.labels == k, start, eta)
             for k in range(M_used) for sign in (+1.0, -1.0)]
    t_map0 = time.perf_counter()
    results = pool.map(_bin_task, tasks)
    busy_delta = sum(r[4] for r in results) - (time.perf_counter() - t_map0)

    U = np.empty((M_used, q))
    D2 = np.empty((M_used, q))
    for k in range(M_used):
        _, _, result_p, failed_p, _ = results[2 * k]
        _, _, result_m, failed_m, _ = results[2 * k + 1]
        counter.add(1, N, int(failed_p))
        counter.add(1, N, int(failed_m))
        U_k = (result_p - result_m) / (2.0 * t_bin[k])
        if np.any(np.isnan(U_k)):
            return _bin_failed(binset, M_used, q), busy_delta
        U[k] = U_k
        D2[k] = (result_p - 2.0 * theta_hat + result_m) / t_bin[k] ** 2

    return _bin_done(binset, U, D2, p), busy_delta


def between_terms(binset: BinSet, coordinate: int) -> float:
    """V_btw = (1/N) sum_k p_k U[k, coordinate]^2 (spec section 4).
    Callers must not call this on a `BinSet` with `failed=True`."""
    N = binset.labels.size
    p = binset.p
    U_c = binset.U[:, coordinate]
    return float(np.sum(p * U_c ** 2)) / N


def bias_and_acceleration(binset: BinSet, N: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    ABC interval bias/acceleration per output (spec A10), from one
    measured `BinSet`: B_hat_c = (1/(2N)) * sum_k p_k*D2[k,c], D2 the
    stencil's own curvature (`BinSet.D2`, `differences.difference`) --
    along the stencil direction D2_k is the bin-mass Hessian's diagonal
    H_kk, and the plug-in bias (1/2)*tr(H*Cov(p_hat)) under the
    multinomial covariance of the bin masses reduces to this sum (at
    point-level masses p_i=1/N this is the book's b = (1/(2N^2)) *
    sum_i T_i_ddot). a_c = (1/(6*sqrt(N))) * sum_k p_k*U[k,c]^3 /
    (sum_k p_k*U[k,c]^2)^1.5, U the centered bin influence -- NaN where
    that denominator is 0. Callers must not call this on `failed=True`.
    """
    p = binset.p
    U = binset.U
    D2 = binset.D2
    q = U.shape[1]

    B_hat = np.sum(p[:, None] * D2, axis=0) / (2.0 * N)

    num = np.sum(p[:, None] * U ** 3, axis=0)
    den = np.sum(p[:, None] * U ** 2, axis=0)
    a = np.full(q, np.nan)
    valid = den > 0.0
    a[valid] = num[valid] / (6.0 * np.sqrt(N) * den[valid] ** 1.5)

    return B_hat, a
