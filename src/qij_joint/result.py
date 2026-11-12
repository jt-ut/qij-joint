"""`BootstrapResult`: the bootstrap comparator's replicates, theta_hat,
the percentile interval and `.bc_interval(level)` (the bias-corrected
percentile interval). The ported method returns its own result
(`qij_joint.method.MethodResult`); the result classes of the superseded
variance-estimation loops were removed with them (CLEANUP_NOTES.md).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
from scipy.stats import norm

from .core import abc as _abc


@dataclass
class BootstrapResult:
    """The bootstrap comparator's result: `outputs`, the full-data point
    estimate `theta_hat` (spec/archive/QIJ_mods_waves.md A10, one evaluation
    outside the resample loop's own RNG stream), the (B, q) replicate
    array, the degenerate-replicate count, and timing. `n_degenerate`
    (spec/archive/QIJ_mods_waves.md A11) is the number of replicates whose fit
    failed the acceptance rule (NaN) -- for an estimator without
    `takes_start` this is the same replicates today's `n_failed` counted,
    under its new name. `eta_full` (A15/A13 step 2) is measured once,
    after `theta_hat`, and used by every replicate in place of the
    declared eta (`bootstrap.py`); == the declared eta for an estimator
    without `takes_start`."""

    outputs: Tuple[str, ...]
    theta_hat: np.ndarray    # (q,)
    replicates: np.ndarray   # (B, q)
    n_degenerate: int
    wall_time: float
    busy_time: float
    workers: int
    eta_full: float
    theta_hat_status: str        # `parallel.fit_status` of theta_hat's own fit
    replicate_status: list       # (B,) `parallel.fit_status` of each replicate

    @property
    def variance(self) -> np.ndarray:
        """Var(theta*) per output (q,), over the non-failed replicates."""
        return np.nanvar(self.replicates, axis=0, ddof=1)

    def interval(self, level: float) -> np.ndarray:
        """(q, 2) [lo, hi]: the percentile interval of `.replicates`,
        ignoring failed (NaN) replicates -- any prefix of the B
        replicates may be passed in."""
        replicates = np.asarray(self.replicates, dtype=float)
        alpha = 1.0 - level
        lo = np.nanquantile(replicates, alpha / 2.0, axis=0)
        hi = np.nanquantile(replicates, 1.0 - alpha / 2.0, axis=0)
        return np.stack([lo, hi], axis=-1)

    def bc_interval(self, level: float) -> np.ndarray:
        """(q, 2) [lo, hi]: the bias-corrected percentile interval
        (Efron & Tibshirani 1993 ch. 14), z0 from `.replicates` against
        `theta_hat`, at no cost beyond the one extra evaluation
        `theta_hat` already is (spec/archive/QIJ_mods_waves.md A10)."""
        return _abc.bc_interval(self.replicates, self.theta_hat, level)


def _normal_interval(theta_hat: np.ndarray, V_btw: np.ndarray, level: float) -> np.ndarray:
    theta_hat = np.asarray(theta_hat, dtype=float)
    V_btw = np.asarray(V_btw, dtype=float)
    z = float(norm.ppf(0.5 * (1.0 + level)))
    half = z * np.sqrt(np.maximum(V_btw, 0.0))
    return np.stack([theta_hat - half, theta_hat + half], axis=-1)
