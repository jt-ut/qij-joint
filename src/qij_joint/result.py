"""
`QIJResult` (spec/method_notes.md section 4): one draw's QIJ fit, and
the pinned user surface `.variance` (V_btw) / `.interval(level)` (the
normal interval on V_btw). `BootstrapResult` carries the bootstrap
comparator's replicates and the percentile interval.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
from scipy.stats import norm


@dataclass
class QIJResult:
    """One draw's QIJ result (spec/method_notes.md section 4).
    Per-output arrays are ordered like `outputs`; diagnostic arrays are
    the method's own points and prototypes, in native coordinates."""

    outputs: Tuple[str, ...]
    N: int
    theta_hat: np.ndarray          # (q,)
    V_btw: np.ndarray              # (q,)
    V_win_hat: np.ndarray          # (q,)
    V_tot_hat: np.ndarray          # (q,)
    L: np.ndarray                  # (q,) int
    n_level_splits: np.ndarray     # (q,) int
    n_adjacency_splits: np.ndarray  # (q,) int
    rho: np.ndarray                # (q,)
    gain_ratio: np.ndarray         # (q,)
    n_refine_evals: np.ndarray     # (q,) int
    ell: np.ndarray                # (q,)
    lam: np.ndarray                # (q,)
    ell_bound: np.ndarray          # (q,) bool
    lam_bound: np.ndarray          # (q,) bool
    M_X: int
    n_failed: int
    evals_by_stage: Dict[str, int]
    rows_by_stage: Dict[str, int]
    wall_time_by_stage: Dict[str, float]
    busy_time_total: float
    workers: int
    psi0: np.ndarray               # (N, q)
    sigma: np.ndarray              # (N, q)
    psi_hat: np.ndarray            # (N, q)
    bin_label: np.ndarray          # (N, q) int
    bmu: np.ndarray                # (N,) int
    prototype_p: np.ndarray        # (M,)
    prototype_w: np.ndarray        # (M,) or (M, d), native coordinates
    prototype_I: np.ndarray        # (M, q)

    @property
    def variance(self) -> np.ndarray:
        """V_btw per output (q,): the measured between-bin variance
        the interval rests on."""
        return self.V_btw

    def interval(self, level: float) -> np.ndarray:
        """(q, 2) [lo, hi]: theta_hat +/- z_{(1+level)/2}*sqrt(V_btw),
        the normal interval on the measured between-bin variance
        (spec/method_notes.md section 4). A negative V_btw is floored
        at 0; NaN in theta_hat or V_btw propagates to NaN."""
        return _normal_interval(self.theta_hat, self.V_btw, level)


@dataclass
class BootstrapResult:
    """The bootstrap comparator's result: `outputs`, the (B, q)
    replicate array, the failed-replicate count, and timing."""

    outputs: Tuple[str, ...]
    replicates: np.ndarray   # (B, q)
    n_failed: int
    wall_time: float
    busy_time: float
    workers: int

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


def _normal_interval(theta_hat: np.ndarray, V_btw: np.ndarray, level: float) -> np.ndarray:
    theta_hat = np.asarray(theta_hat, dtype=float)
    V_btw = np.asarray(V_btw, dtype=float)
    z = float(norm.ppf(0.5 * (1.0 + level)))
    half = z * np.sqrt(np.maximum(V_btw, 0.0))
    return np.stack([theta_hat - half, theta_hat + half], axis=-1)
