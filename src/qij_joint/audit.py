"""The cold search's audit: a diagnostic outside every method.

A cold fit (no `start`) searches for the global maximum and can settle on
a lower one. On a draw from a known truth, one continuation from the true
parameters on the same draw tells the two apart: `search_gap` is the cold
fit's penalized log-likelihood per unit weight minus the continuation's,
and `search_failed` marks a gap below zero by more than the estimator's
`eta`. The truth never enters any estimate; it only judges the search.
The audit's evaluation is not counted in any method's evaluations or wall
time; its own wall time is reported beside it. An estimator without a
continuation (`takes_start`) or a likelihood (`loglik`) is not audited.
"""
from __future__ import annotations

import time

import numpy as np

from . import registry
from .parallel import fit_status

NO_AUDIT = dict(search_gap=float('nan'), search_failed=False,
                search_status='', search_wall_time=0.0)


def search_audit(T, X: np.ndarray, theta_hat: np.ndarray, dataset: str, estimator: str) -> dict:
    """The audit of one cold fit `theta_hat` (T's full output layout) on
    draw `X` at unit weights. Call it after the cold fit's own status has
    been read: the continuation overwrites `T`'s `last_fit_info`."""
    theta_hat = np.asarray(theta_hat, dtype=float)
    if not (getattr(T, 'takes_start', False) and hasattr(T, 'loglik')
            and np.all(np.isfinite(theta_hat))):
        return dict(NO_AUDIT)
    t0 = time.perf_counter()
    w = np.ones(len(X))
    try:
        cont = np.asarray(T(X, w, start=registry.truth(dataset, estimator)), dtype=float)
        status = fit_status(T, cont)
    except Exception:
        cont = np.full_like(theta_hat, np.nan)
        status = fit_status(T, cont, raised=True)
    gap = float('nan')
    if np.all(np.isfinite(cont)):
        gap = T.loglik(X, w, theta_hat) - T.loglik(X, w, cont)
    return dict(search_gap=gap, search_failed=bool(gap < -T.eta),
                search_status=status, search_wall_time=time.perf_counter() - t0)
