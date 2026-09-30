"""
`QIJResult` (spec/method_notes.md section 4): one draw's QIJ fit, and
the pinned user surface `.variance` (V_btw) / `.interval(level)` (the
normal interval on V_btw) / `.abc_interval(level)` (the ABC_q interval,
spec/QIJ_mods_waves.md A10). `BootstrapResult` carries the bootstrap
comparator's replicates, theta_hat, the percentile interval and
`.bc_interval(level)` (the bias-corrected percentile interval, A10).

`ivqbins` picks the second stage: `'marginal'` (a per-output refinement,
one partition per output) or `'joint'` (one partition shared by every
output, method_notes joint section). `V_btw`/`V_win_hat`/`V_tot_hat`/`gain_ratio`
are populated by whichever stage ran; the marginal-only fields (`L`,
`bin_label`, `bin_U`, the refinement counts) are NaN/0/-1/empty under
`'joint'`; the `joint_*` fields are NaN/0/False/empty under `'marginal'`.
`rho` and `psi_hat` are populated under both: under `'joint'` they are
`core.joint.joint_psi_hat`'s per-output rho_c and the refined field on
the joint path's own shared final bins, in place of the marginal path's
per-coordinate partition.

`refine_schedule` picks the marginal path's own schedule
(spec/QIJ_mods_waves.md A14): `'queue'` (ported, the default) or
`'rounds'` (every measured output refined together in synchronized
rounds); inert under `ivqbins='joint'`, whose own check already
advances in rounds. `n_rounds` is the round count `'rounds'` actually
ran; 0 under `'queue'` or `'joint'`.

`quantized_start` picks the quantized base fit theta_Q's starting point
(spec/QIJ_mods_waves.md A9 item 5): `'multistart'` (ported) or
`'full-data'` (theta_Q continues theta_hat onto the survey rows). `a`,
`b_hat`, `c_q` are the ABC interval's per-output acceleration, bias and
curvature ingredients (A10); `eta_Q` and `survey_step_ratio` are the
survey's own reproducibility diagnostics (A9 items 3-4). All five are
populated for every estimator; for one without `takes_start` the
curvature and bias/acceleration ingredients still cost their declared
evaluations (a and b_hat from the deterministic optimizer's own bins),
only `eta_Q`/`survey_step_ratio` stay NaN (no continuation to check).

`pilot` (spec/QIJ_affine_pilot_spec.md) picks stage 1's initial
influence estimate: `'affine'` or `'gp'`. Under `'affine'`, `ell`,
`lam`, `ell_bound`, `lam_bound`, `c`, `c_bound`, `sigma` and
`prototype_h` are NaN/False (no posterior variance or width/local-scale
search of any kind), and `bridge_j`/`bridge_k`/`bridge_m`/`bridge_delta`
(the bridge score, one row per CADJ pair with a non-empty second-order
cell) and `cell_p`/`cell_mu`/`cell_g` (the affine field's own per-cell
mass, mean and gradient) are populated; both are empty under `'gp'`,
whose every other field stays exactly as before this option existed.

`check_rule` (spec/QIJ_joint_check_measured_spec.md), inert (but
recorded) under `ivqbins='marginal'`, picks the joint check's
continuation rule: `'predicted'` (the default, today's rule, every
product byte-identical) or `'measured'` (`pilot='gp'` only), which also
populates `joint_n_closed_unpaid`/`joint_n_closed_unflagged`/
`joint_n_noise_floored`/`joint_sum_b_delta` and changes `V_win_hat`/
`V_tot_hat`'s own definition (no posterior variance).

`fit_weights` (spec/QIJ_mass_weighted_fit_spec.md), unused under
`pilot='affine'` the way `gptrend`/`gpwidth` are, picks the pilot's own
kernel-regression noise: `'none'` (the default, every product
byte-identical) or `'mass'` (a per-observation noise diagonal
1/m_tilde_j in place of the identity, in both the profiled marginal
likelihood and the posterior).

`sigma_points` (spec/QIJ_sigma_points_spec.md) picks the optional
interval stage that runs after refinement and the curvature stage, on
the full data: `.sigma_interval(level)` from the stored `sigma_mean`/
`sigma_sd`, beside `.interval`/`.abc_interval`, which it changes
nothing about. `sigma_status` is `None` under `sigma_points=False`;
`'ok'`, `'base_unconverged'` (the base fixed point missed its cap) or
`'eval_failed'` (a NaN/exception among the 2n evaluations) under
`True`, both failures leaving `sigma_mean`/`sigma_sd`/`sigma_bias`/
`sigma_k`/`sigma_sign`/`sigma_response` NaN or empty.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
from scipy.stats import norm

from .core import abc as _abc


@dataclass
class QIJResult:
    """One draw's QIJ result (spec/method_notes.md section 4).
    Per-output arrays are ordered like `outputs`; diagnostic arrays are
    the method's own points and prototypes, in native coordinates."""

    outputs: Tuple[str, ...]
    N: int
    theta_hat: np.ndarray          # (q,)
    theta_hat_full: np.ndarray     # (q_full,) every T output; the search audit's input
    V_btw: np.ndarray              # (q,)
    V_win_hat: np.ndarray          # (q,) model quantity: under ivqbins='joint', the pilot's
                                    # within-bin spread including the posterior variance v
                                    # (check_rule='predicted') or plain psi0 variance with no v
                                    # (check_rule='measured'; spec/QIJ_joint_check_measured_
                                    # spec.md 2.5); unaffected under ivqbins='marginal'
    V_tot_hat: np.ndarray          # (q,) model quantity: V_btw + V_win_hat
    L: np.ndarray                  # (q,) int; 0 under ivqbins='joint'
    n_level_splits: np.ndarray     # (q,) int; 0 under ivqbins='joint'
    n_adjacency_splits: np.ndarray  # (q,) int; 0 under ivqbins='joint'
    rho: np.ndarray                # (q,); under ivqbins='joint', core.joint.joint_psi_hat's rho_c
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
    psi_hat: np.ndarray            # (N, q); under ivqbins='joint', core.joint.joint_psi_hat's field
    bin_label: np.ndarray          # (N, q) int; -1 under ivqbins='joint'
    bin_U: Tuple[np.ndarray, ...]  # per output (L_c, q); every measured output's
                                    # derivative on this output's own final bins
                                    # (spec/QIJ_mods_waves.md A14); empty under 'joint'
    refine_schedule: str            # 'queue' or 'rounds' (A14); inert under 'joint'
    n_rounds: int                   # rounds run under 'rounds'; 0 under 'queue' or 'joint'
    bmu: np.ndarray                # (N,) int
    prototype_p: np.ndarray        # (M,)
    prototype_w: np.ndarray        # (M,) or (M, d), native coordinates
    prototype_I: np.ndarray        # (M, q)
    prototype_h: np.ndarray        # (M,) local CONN spacing; NaN under gpwidth='global'
    ivqbins: str                   # 'marginal' or 'joint' (method_notes joint section)
    survey: str                    # 'points' or 'moments' (method_notes section 2)
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
    joint_n_closed_unpaid: int     # check_rule='measured' only, 0 under 'predicted' (spec/
                                    # QIJ_joint_check_measured_spec.md 2.4): splits whose two
                                    # children were closed because the split did not pay
    joint_n_closed_unflagged: int  # check_rule='measured' only, 0 under 'predicted' (spec
                                    # 2.4): children of a paying split closed by the pilot's
                                    # own flag alone
    joint_n_noise_floored: int     # check_rule='measured' only, 0 under 'predicted' (spec
                                    # 2.4): splits whose deciding tau' was the noise floor
    joint_sum_b_delta: np.ndarray  # (q,) check_rule='measured' only, NaN under 'predicted'
                                    # (spec 2.4-2.5): summed finite-difference noise-bias
                                    # floor over every split measured, the residual bias left
                                    # in V_btw (never subtracted from it)
    joint_failed: bool             # a failed output, or a failed initial bin measurement before any check ran
    joint_bin_mass: np.ndarray     # (L,); empty under 'marginal'
    joint_bin_U: np.ndarray        # (L, q); empty under 'marginal'
    joint_bin_m: np.ndarray        # (L, q); empty under 'marginal'
    joint_bin_flagged: np.ndarray  # (L,) bool; empty under 'marginal'
    joint_bin_label: np.ndarray    # (N,) int; -1 under 'marginal'
    joint_busy_delta: float        # 0.0 under 'marginal'
    a: np.ndarray                  # (q,) ABC acceleration (spec/QIJ_mods_waves.md A10)
    b_hat: np.ndarray              # (q,) ABC second-order bias (A10)
    c_q: np.ndarray                # (q,) ABC curvature-along-influence, survey-row evaluated (A10)
    c_q_one_sided: np.ndarray      # (q,) bool: c_q fell back to the one-sided form (A10/A15)
    eta_Q: float                   # A9 item 3; NaN for an estimator without `takes_start`
    eta_full: float                # A15/A13 step 2; == the declared eta for an estimator
                                    # without `takes_start`
    survey_step_ratio: np.ndarray  # (5, q) A9 item 4; NaN rows/columns as eta_Q is
    quantized_start: str           # 'multistart' or 'full-data' (A9 item 5)
    pilot: str                     # 'affine' or 'gp' (spec/QIJ_affine_pilot_spec.md 1)
    check_rule: str                # 'predicted' or 'measured' (spec/QIJ_joint_check_measured_
                                    # spec.md); inert under ivqbins='marginal'
    fit_weights: str               # 'none' or 'mass' (spec/QIJ_mass_weighted_fit_spec.md);
                                    # unused under pilot='affine'
    bridge_j: np.ndarray           # (P,) int; empty under pilot='gp'
    bridge_k: np.ndarray           # (P,) int; empty under pilot='gp'
    bridge_m: np.ndarray           # (P,) m_jk; empty under pilot='gp'
    bridge_delta: np.ndarray       # (P, q) Delta_jk; empty under pilot='gp'
    cell_p: np.ndarray             # (M,) p_j; empty under pilot='gp'
    cell_mu: np.ndarray            # (M, d_z) mu_j; empty under pilot='gp'
    cell_g: np.ndarray             # (M, d_z, q) g_j; empty under pilot='gp'
    sigma_points: bool              # spec/QIJ_sigma_points_spec.md 1
    sigma_status: Optional[str]     # None under sigma_points=False; else 'ok'/
                                     # 'base_unconverged'/'eval_failed'
    n_fp_sigma: int                 # base fixed-point iterations (2.2); 0 without a start
    r_fp_sigma: float               # the base fixed point's own residual (2.2)
    n_failed_sigma: int             # of the 2n direction evaluations (2.4)
    n_dirs_sigma: int               # n, the kept influence-covariance eigendirections (2.3)
    max_abs_d_sigma: float          # max_i |d_i| over every kept direction (2.3)
    sigma_mean: np.ndarray          # (q,) m, the unscented mean (2.5)
    sigma_sd: np.ndarray            # (q,) sqrt(S_oo) (2.5)
    sigma_bias: np.ndarray          # (q,) m - theta_hat at the base fixed point (2.5)
    sigma_k: np.ndarray             # (2n,) int, the direction index of each evaluation
    sigma_sign: np.ndarray          # (2n,) int, +-1
    sigma_response: np.ndarray      # (2n, q) R^(k+-), raw (spec section 3)

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

    def abc_interval(self, level: float) -> np.ndarray:
        """(q, 2) [lo, hi]: the ABC_q interval (DiCiccio & Efron 1992;
        Efron & Tibshirani 1993 ch. 22) from the stored `a`/`b_hat`/
        `c_q` ingredients and sigma = sqrt(V_btw) (spec/QIJ_mods_waves.md
        A10)."""
        sigma = np.sqrt(np.maximum(self.V_btw, 0.0))
        return _abc.abc_interval(self.theta_hat, sigma, self.a, self.b_hat, self.c_q, level)

    def sigma_interval(self, level: float) -> np.ndarray:
        """(q, 2) [lo, hi]: sigma_mean +/- z_{(1+level)/2}*sigma_sd, the
        unscented interval of the sigma-points stage
        (spec/QIJ_sigma_points_spec.md 2.5); `interval`/`abc_interval`
        are unchanged by this stage (`QIJResult` has no `interval_btw`,
        that is `QIJTResult`'s own method). NaN when `sigma_points` is
        False or the stage failed."""
        return _normal_interval(self.sigma_mean, self.sigma_sd ** 2, level)


@dataclass
class BootstrapResult:
    """The bootstrap comparator's result: `outputs`, the full-data point
    estimate `theta_hat` (spec/QIJ_mods_waves.md A10, one evaluation
    outside the resample loop's own RNG stream), the (B, q) replicate
    array, the degenerate-replicate count, and timing. `n_degenerate`
    (spec/QIJ_mods_waves.md A11) is the number of replicates whose fit
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
        `theta_hat` already is (spec/QIJ_mods_waves.md A10)."""
        return _abc.bc_interval(self.replicates, self.theta_hat, level)


def _normal_interval(theta_hat: np.ndarray, V_btw: np.ndarray, level: float) -> np.ndarray:
    theta_hat = np.asarray(theta_hat, dtype=float)
    V_btw = np.asarray(V_btw, dtype=float)
    z = float(norm.ppf(0.5 * (1.0 + level)))
    half = z * np.sqrt(np.maximum(V_btw, 0.0))
    return np.stack([theta_hat - half, theta_hat + half], axis=-1)
