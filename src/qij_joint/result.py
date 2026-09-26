"""
`QIJResult` (spec/method_notes.md section 4): one draw's QIJ fit, and
the pinned user surface `.variance` (V_btw) / `.interval(level)` (the
normal interval on V_btw). `BootstrapResult` carries the bootstrap
comparator's replicates and the percentile interval.

`ivqbins` picks the second stage: `'marginal'` (a per-output refinement
queue, one partition per output) or `'joint'` (one partition shared by
every output, method_notes joint section). `V_btw`/`V_win_hat`/`V_tot_hat`/`gain_ratio`
are populated by whichever stage ran; the marginal-only fields (`L`,
`rho`, `psi_hat`, `bin_label`, the refinement counts) are NaN/0/-1
under `'joint'`; the `joint_*` fields are NaN/0/False/empty under
`'marginal'`.
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
    L: np.ndarray                  # (q,) int; 0 under ivqbins='joint'
    n_level_splits: np.ndarray     # (q,) int; 0 under ivqbins='joint'
    n_adjacency_splits: np.ndarray  # (q,) int; 0 under ivqbins='joint'
    rho: np.ndarray                # (q,); NaN under ivqbins='joint'
    gain_ratio: np.ndarray         # (q,); populated by whichever stage ran
    n_refine_evals: np.ndarray     # (q,) int; 0 under ivqbins='joint'
    ell: np.ndarray                # (q,); NaN under gpwidth='local'
    lam: np.ndarray                # (q,)
    ell_bound: np.ndarray          # (q,) bool; False under gpwidth='local'
    lam_bound: np.ndarray          # (q,) bool
    gptrend: str                   # 'affine' or 'quadratic' (method_notes section 3)
    gpwidth: str                   # 'global' or 'local' (method_notes section 3)
    c: np.ndarray                  # (q,) fitted local-width factor; NaN under gpwidth='global'
    c_bound: np.ndarray            # (q,) bool; False under gpwidth='global'
    M_X: int
    M_X_source: str                # 'rule' or 'argument' (method_notes section 2)
    n_failed: int
    evals_by_stage: Dict[str, int]
    rows_by_stage: Dict[str, int]
    wall_time_by_stage: Dict[str, float]
    busy_time_total: float
    workers: int
    psi0: np.ndarray               # (N, q)
    sigma: np.ndarray              # (N, q)
    psi_hat: np.ndarray            # (N, q); NaN under ivqbins='joint'
    bin_label: np.ndarray          # (N, q) int; -1 under ivqbins='joint'
    bmu: np.ndarray                # (N,) int
    prototype_p: np.ndarray        # (M,)
    prototype_w: np.ndarray        # (M,) or (M, d), native coordinates
    prototype_I: np.ndarray        # (M, q)
    prototype_h: np.ndarray        # (M,) local CONN spacing; NaN under gpwidth='global'
    ivqbins: str                   # 'marginal' or 'joint' (method_notes joint section)
    joint_S_pred: np.ndarray       # (q,) predicted within share at end of growth; NaN under 'marginal'
    joint_a: np.ndarray            # (q,) the check's fitted scale factor; NaN under 'marginal'
    joint_S_pred_pre_lloyd: np.ndarray  # (q,); NaN under 'marginal' or when the Lloyd pass did not run
    joint_L0: int                  # bins after growth; 0 under 'marginal'
    joint_L: int                   # final joint bin count after the check; 0 under 'marginal'
    joint_n_growth_rounds: int
    joint_growth_capped: bool
    joint_n_flagged: int
    joint_n_check_rounds: int
    joint_n_check_evals: int
    joint_n_level_splits: int
    joint_n_adjacency_splits: int
    joint_check_capped: bool
    joint_failed: bool             # a failed output, or a failed initial bin measurement before any check ran
    joint_bin_mass: np.ndarray     # (L,); empty under 'marginal'
    joint_bin_U: np.ndarray        # (L, q); empty under 'marginal'
    joint_bin_m: np.ndarray        # (L, q); empty under 'marginal'
    joint_bin_flagged: np.ndarray  # (L,) bool; empty under 'marginal'
    joint_bin_label: np.ndarray    # (N,) int; -1 under 'marginal'
    joint_busy_delta: float        # 0.0 under 'marginal'

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
