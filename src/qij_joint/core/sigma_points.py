"""Sigma points: an optional interval stage (spec/QIJ_sigma_points_spec.md).

Runs after refinement and the ABC curvature stage, on the full data,
from a fixed point of the base fit's own continuation (spec 2.2): 2n
one-sided evaluations along the eigenvectors of the refined influence
estimate's own covariance (spec 2.3-2.4), whose unscented mean and
covariance give `QIJResult.sigma_interval` (spec 2.5).
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Sequence, Tuple

import numpy as np

from .abc import _curvature_task

_FP_CAP = 5
_EIG_TOL = 1e-12


@dataclass
class SigmaResult:
    """One draw's sigma-points stage (spec section 3). `status` is 'ok',
    'base_unconverged' (the base fixed point missed the cap), or
    'eval_failed' (a NaN/exception among the 2n evaluations); `mean`/
    `sd`/`bias`/`k`/`sign`/`response` are all-NaN or empty under either
    failure. `k`/`sign`/`response` are the raw per-evaluation array
    product: direction index, +-1, and R^(k+-) (q,) per row."""

    status: str
    n_fp: int
    r_fp: float
    n_dirs: int
    max_abs_d: float
    n_failed: int
    busy: float
    mean: np.ndarray
    sd: np.ndarray
    bias: np.ndarray
    k: np.ndarray
    sign: np.ndarray
    response: np.ndarray


def _fixed_point(counter, X: np.ndarray, theta: np.ndarray, eta_full: float,
                  N: int) -> Tuple[np.ndarray, int, float, bool]:
    """theta -> a fixed point of T(X, 1_N, start=theta, eta=eta_full)
    (spec 2.2): theta' = T(theta); r = max_o |theta'_o-theta_o| /
    max(|theta'_o|,|theta_o|) (0/0 -> 0); theta <- theta' until
    r <= eta_full, at most 5 iterations. An estimator without a start
    never iterates (r = 0, no evaluation). A non-finite theta' ends the
    loop at once, reported the same as reaching the cap (both
    'base_unconverged' to the caller). Returns (theta, n_iter, r,
    converged)."""
    if not getattr(counter, 'takes_start', False):
        return theta, 0, 0.0, True
    ones = np.ones(N)
    r = float('nan')
    for n in range(1, _FP_CAP + 1):
        theta_prime = np.asarray(counter(X, ones, start=theta, eta=eta_full), dtype=float)
        if np.any(np.isnan(theta_prime)):
            return theta_prime, n, float('nan'), False
        scale = np.maximum(np.abs(theta_prime), np.abs(theta))
        diff = np.abs(theta_prime - theta)
        r = float(np.max(np.where(scale > 0.0, diff / scale, 0.0)))
        theta = theta_prime
        if r <= eta_full:
            return theta, n, r, True
    return theta, _FP_CAP, r, False


def _periods(counter, outputs: Sequence[str]) -> np.ndarray:
    """One period per column of `outputs` (an estimator's measured
    output names, in `response`'s own column order): `counter.periodic`
    (a name -> period map; `{}` on an estimator declaring none) gives
    the period for a periodic column, 0.0 marking a non-periodic one."""
    periodic = getattr(counter, 'periodic', {})
    return np.array([periodic.get(name, 0.0) for name in outputs], dtype=float)


def _wrap_deviation(response: np.ndarray, periods: np.ndarray) -> np.ndarray:
    """`response`'s (2n, q) deviations from theta_base, with its
    periodic columns (`periods[j] > 0`) wrapped into [-P/2, P/2):
    d = ((r + P/2) mod P) - P/2, so a response that crossed the
    output's branch edge reads as a small deviation instead of one near
    a full period P. Non-periodic columns (`periods[j] == 0`) pass
    `response` through unchanged, bit for bit."""
    is_periodic = (periods > 0.0)[None, :]
    safe_periods = np.where(periods > 0.0, periods, 1.0)[None, :]
    wrapped = np.mod(response + 0.5 * safe_periods, safe_periods) - 0.5 * safe_periods
    return np.where(is_periodic, wrapped, response)


def _unscented_stats(response: np.ndarray, theta_base_measured: np.ndarray,
                      periods: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The unscented mean/sd/bias of the 2n sigma-point responses (spec
    2.5): the plain (equally-weighted) average of their deviations --
    wrapped per `_wrap_deviation` on a periodic column, unchanged on a
    non-periodic one -- added back onto `theta_base_measured` for the
    mean, and its own centering for S. `bias` is that average deviation
    (the wrapped mean deviation on a periodic column, spec 2.5's
    `m - theta_hat` on the rest); `sd` is `sqrt(diag(S))`. A periodic
    column's `mean` therefore lands within [theta_base - P/2, theta_base
    + P/2) of its own base fixed point, which need not be the output's
    declared reporting branch -- that branch's edge is not available
    here, only the period, so this is reported as the closest
    equivalent, within one half-period of theta_base."""
    d = _wrap_deviation(response, periods)
    bias = d.mean(axis=0)
    centered = d - bias
    S = (centered.T @ centered) / d.shape[0]
    return theta_base_measured + bias, np.sqrt(np.diag(S)), bias


def _directions(psi_hat: np.ndarray, N: int) -> Tuple[np.ndarray, np.ndarray]:
    """The kept eigenpairs of C = mean_i(psi_hat_i psi_hat_i^T) / N (spec
    2.3), lambda_k > 1e-12*lambda_max, sorted by decreasing lambda_k.
    Returns (lam (n,), V (q, n))."""
    C = (psi_hat.T @ psi_hat) / (N ** 2)
    lam, V = np.linalg.eigh(C)
    order = np.argsort(lam)[::-1]
    lam, V = lam[order], V[:, order]
    keep = lam > _EIG_TOL * lam[0]
    return lam[keep], V[:, keep]


def fit(counter, X: np.ndarray, theta_hat: np.ndarray, psi_hat: np.ndarray,
        eta_full: float, N: int, measured: Sequence[int], pool=None) -> SigmaResult:
    """The sigma-points stage (spec section 2): a fixed point of the
    base fit (2.2), 2n evaluations along the refined influence's own
    covariance directions (2.3-2.4), and their unscented mean/covariance
    (2.5). `theta_hat` is the FULL (q_full,) fit, the continuation's own
    layout; `psi_hat` is (N, q) at the MEASURED outputs; `measured` the
    absolute indices of those q columns in `theta_hat`. A measured
    output periodic on `counter.periodic` has its mean/S formed from
    wrapped deviations (`_unscented_stats`); the rest are unaffected.
    `pool.share(X)` is assumed already done for this draw (spec 5)."""
    measured = np.asarray(measured, dtype=int)
    q = measured.size
    periods = _periods(counter, [counter.outputs[i] for i in measured])
    theta_base, n_fp, r_fp, converged = _fixed_point(
        counter, X, np.asarray(theta_hat, dtype=float), eta_full, N)
    empty = np.zeros(0, dtype=int)
    if not converged:
        nan_q = np.full(q, np.nan)
        return SigmaResult('base_unconverged', n_fp, r_fp, 0, float('nan'), 0, 0.0,
                            nan_q, nan_q, nan_q, empty, empty, np.zeros((0, q)))

    lam, V = _directions(psi_hat, N)
    n = lam.size
    c = np.sqrt(n)
    psi_k = psi_hat @ V                                    # (N, n), spec 2.3
    d = c * psi_k / (N * np.sqrt(lam))[None, :]             # (N, n), d_i^(k+)
    max_abs_d = float(np.max(np.abs(d))) if n > 0 else float('nan')

    k_idx = np.repeat(np.arange(n), 2)
    sign_idx = np.tile(np.array([1, -1], dtype=int), n)
    weights = np.empty((N, 2 * n))
    for j in range(n):
        for pos, sign in ((0, 1.0), (1, -1.0)):
            expo = sign * d[:, j]
            weights[:, 2 * j + pos] = N * np.exp(expo) / np.sum(np.exp(expo))

    theta_base_measured = theta_base[measured]
    response = np.full((2 * n, q), np.nan)
    n_failed = 0
    busy = 0.0
    if pool is None:
        for col in range(2 * n):
            result = np.asarray(
                counter(X, weights[:, col], start=theta_base, eta=eta_full), dtype=float)
            if np.any(np.isnan(result)):
                n_failed += 1
            else:
                response[col] = result[measured] - theta_base_measured
    else:
        tasks = [((int(k_idx[col]), int(sign_idx[col])), weights[:, col], theta_base, eta_full)
                 for col in range(2 * n)]
        t_map0 = time.perf_counter()
        results = pool.map(_curvature_task, tasks)
        busy = sum(r[3] for r in results) - (time.perf_counter() - t_map0)
        for col, (_key, result, failed, _wall) in enumerate(results):
            counter.add(1, N, int(failed))
            if failed:
                n_failed += 1
            else:
                response[col] = result[measured] - theta_base_measured

    if n_failed > 0:
        nan_q = np.full(q, np.nan)
        return SigmaResult('eval_failed', n_fp, r_fp, n, max_abs_d, n_failed, busy,
                            nan_q, nan_q, nan_q, k_idx, sign_idx, response)

    mean, sd, bias = _unscented_stats(response, theta_base_measured, periods)  # spec 2.5
    return SigmaResult('ok', n_fp, r_fp, n, max_abs_d, 0, busy,
                        mean, sd, bias, k_idx, sign_idx, response)
