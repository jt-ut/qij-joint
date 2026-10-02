"""
`QIJResult` (spec/method_notes.md section 4): one draw's QIJ fit, and
the pinned user surface `.variance` (V_btw) / `.interval(level)` (the
normal interval on V_btw) / `.abc_interval(level)` (the ABC_q interval,
spec/QIJ_mods_waves.md A10). `BootstrapResult` carries the bootstrap
comparator's replicates, theta_hat, the percentile interval and
`.bc_interval(level)` (the bias-corrected percentile interval, A10).

A21 (spec/QIJ_mods_waves.md A21, "one path: the removals") removed
every switch this result used to carry beside the one surviving
method: `ivqbins` (the per-output marginal second stage, and `'joint'`
as an alternative to it), `pilot` (the affine pilot, an alternative to
the GP), `check_rule` (the per-bin check's continuation rule) and
`tree_rule` (the per-bin growth/share rule, an alternative to the
total-share one) are gone as both switches and fields, along with
every product that existed only to report a choice among them or the
marginal/affine/per-bin-check machinery itself: `L`, `n_level_splits`,
`n_adjacency_splits`, `rho`, `gain_ratio`, `n_refine_evals`,
`bin_label`, `bin_U` (the marginal path's own per-output partition and
counts), `refine_schedule`/`n_rounds` (the marginal path's own
schedule), `bridge_j`/`bridge_k`/`bridge_m`/`bridge_delta`/`cell_p`/
`cell_mu`/`cell_g` (the affine pilot's own products), and
`joint_n_flagged`/`joint_n_adjacency_splits`/`joint_bin_flagged`/
`joint_n_closed_unpaid`/`joint_n_closed_unflagged`/
`joint_n_noise_floored`/`joint_sum_b_delta` (the per-bin check's own
flag/pay bookkeeping).

The unified measured loop (spec/QIJ_unified_loop_spec.md) then replaced
A20/A21's own growth/check second stage with one loop from one bin,
every split measured: gone as both switch-era and growth/check-era
fields are `joint_S_pred`/`joint_S_pred_pre_lloyd` (the growth stage's
predicted within share), `joint_a` (A18's separate survey-to-full-data
scale step -- the correction now enters bin by bin through the first
splits' own update, spec 4.0, so `psi0`/`model.offset`/`sigma` are
never rebound by `qij.py` after the fact any more), `joint_L0`/
`joint_n_growth_rounds`/`joint_growth_capped` (growth's own counters),
`joint_n_check_rounds`/`joint_n_check_evals`/`joint_n_level_splits`/
`joint_check_capped` (the check continuation's own counters, replaced
below by `joint_n_splits`/`joint_n_rounds`/`joint_n_evals`/
`joint_capped`/`joint_stop_met`), `joint_share_final` (replaced by the
calibrated stop's own `margin`) and `joint_pilot_err_btw` (the fixed-
pilot diagnostic the old growth stage read; the unified loop has no
fixed pilot to compare against once bin 0 is the whole cloud).

`V_btw`/`V_win_hat`/`se_V_win`/`V_tot_hat`/`kappa`/`se_kappa`/`margin`
are the loop's own report (spec 4.4, 4.5): `kappa`/`se_kappa` are the
calibration ratio of realized to predicted gain over every split made
so far; `se_V_win` is V_win_hat's own standard error (sampling scatter
and the splits' chi-square variability -- NOT the structural under-
read below the final leaves, which `kappa` cannot see, spec "On z");
`margin` is the stop's excess X_c/(eps*V_tot) at termination, the
binding output's value when per-output values differ. `z` (the stop's
margin against both that scatter and the structural under-read),
`n_min` (splits required before the stop may fire) and `L_max` (the
cap on leaves, always resolved -- never None, spec 4.4) are the loop's
own user-exposed levers, recorded here alongside `survey`/
`quantized_start`/`fit_weights`. `psi_hat` is the loop's final
per-point STATE vector (THE ARCHITECTURE RULE, spec/QIJ_mods_waves.md
A20, kept verbatim by the unified loop), rewritten in place after
every measurement; `joint_n_update_scale`/`joint_n_update_shift`/
`joint_n_update_negative` are its own products. Which leaf is split
next reads the leaf's own measured error against the update's noise
term, falling back to the current vector's within-variance (the A20
ranking rule, kept verbatim); `joint_rank_rule` (always
`'measured_error'`) and `joint_bin_ebar` (each final leaf's own error,
beside `joint_bin_U`/`joint_bin_m`/`joint_bin_W`/`joint_bin_n`) record
it. `joint_bin_m` is the leaf's own m_pre (the state vector's mean over
the leaf's points at measurement time), never a fixed copy of the
pilot (THE ARCHITECTURE RULE again; A21's own fix, spec/QIJ_mods_
waves.md A20's validation note). `joint_split_parent`/
`joint_split_child_a`/`joint_split_child_b`/`joint_split_G`/
`joint_split_D`/`joint_split_round` are the per-split record (spec 3,
4.3: G the predicted gain, D the realized gain, both read before the
children's own update) and `joint_round_kappa`/`joint_round_se_kappa`/
`joint_round_V_win`/`joint_round_margin`/`joint_round_n_splits` are the
per-round trajectory of the calibrated stop (spec 4.4), one row per
completed round -- both populated on every draw, failed or not (empty
on failure).

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

`fit_weights` (spec/QIJ_mass_weighted_fit_spec.md), beside `gptrend`/
`gpwidth` the GP pilot's own settings (A21: "they shape the vector's
population"), picks the pilot's own kernel-regression noise: `'none'`
(the default, every product byte-identical) or `'mass'` (a
per-observation noise diagonal 1/m_tilde_j in place of the identity, in
both the profiled marginal likelihood and the posterior). `sigma` is
the GP's per-point posterior sd: under A20/A21's architecture rule it
is read by no decision and is carried here only as a stored survey
diagnostic, never as a per-point companion of `psi_hat`.

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
    V_win_hat: np.ndarray          # (q,) model quantity: kappa * sum of leaf within-bin W
                                    # (spec/QIJ_unified_loop_spec.md 4.4)
    se_V_win: np.ndarray           # (q,) V_win_hat's own standard error (sampling scatter and
                                    # the splits' chi-square variability; NOT the structural
                                    # under-read below the final leaves, spec "On z")
    V_tot_hat: np.ndarray          # (q,) model quantity: V_btw + V_win_hat
    kappa: np.ndarray              # (q,) the calibration ratio, realized/predicted gain summed
                                    # over every split so far (spec 4.4); 1.0 if no split yet
    se_kappa: np.ndarray           # (q,) kappa's own sandwich standard error; 0.0 if no split yet
    margin: np.ndarray             # (q,) the stop's excess X_c/(eps*V_tot) at termination
                                    # (spec 4.4), the binding output's value
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
    sigma: np.ndarray              # (N, q) the GP posterior sd; a stored survey diagnostic
                                    # only (THE ARCHITECTURE RULE, spec/QIJ_mods_waves.md
                                    # A20/A21) -- read by no decision
    psi_hat: np.ndarray            # (N, q); the joint path's final per-point STATE vector
                                    # (A20), rewritten in place after every measurement
    bmu: np.ndarray                # (N,) int
    prototype_p: np.ndarray        # (M,)
    prototype_w: np.ndarray        # (M,) or (M, d), native coordinates
    prototype_I: np.ndarray        # (M, q)
    prototype_h: np.ndarray        # (M,) local CONN spacing; NaN under gpwidth='global'
    survey: str                    # 'points' or 'moments' (method_notes section 2)
    joint_L: int                   # final leaf count (spec/QIJ_unified_loop_spec.md 4.4)
    joint_n_splits: int             # total splits made
    joint_n_rounds: int             # total rounds (one pool batch each, spec 4.2)
    joint_n_evals: int               # total evaluations the loop itself spent (2 per split)
    joint_capped: bool               # L_max bound (spec 4.4); reported whether it binds
    joint_stop_met: bool             # the calibrated stop (4.4) actually fired, vs. the loop
                                      # exhausting every candidate or hitting L_max first
    joint_failed: bool             # a failed output, or a failed initial bin measurement
                                    # before any continuation ran
    joint_bin_mass: np.ndarray     # (L,)
    joint_bin_U: np.ndarray        # (L, q) the measured bin mean
    joint_bin_m: np.ndarray        # (L, q) the leaf's own m_pre (THE ARCHITECTURE RULE,
                                    # spec/QIJ_mods_waves.md A20/A21): psi_hat's mean over the
                                    # leaf's points at measurement time, never a fixed copy of
                                    # the pilot (A21's own fix to the A20 build)
    joint_bin_ebar: np.ndarray           # (L,q) A20 amendment: each final leaf's own measured
                                          # error at creation, beside joint_bin_U/joint_bin_m
    joint_bin_W: np.ndarray               # (L,q) each final leaf's own within contribution
                                           # W_l = p_l*Var_l(psi_hat)/N (spec 3)
    joint_bin_n: np.ndarray               # (L,) int, each final leaf's own point count
    joint_bin_label: np.ndarray    # (N,) int
    joint_busy_delta: float
    joint_split_parent: np.ndarray        # (S,) int, the split's parent leaf id
    joint_split_child_a: np.ndarray       # (S,) int, the smaller (measured) child's id
    joint_split_child_b: np.ndarray       # (S,) int, the larger (conservation) child's id
    joint_split_G: np.ndarray             # (S,q) the split's predicted gain (spec 3, 4.3)
    joint_split_D: np.ndarray             # (S,q) the split's realized gain (spec 3, 4.3)
    joint_split_round: np.ndarray         # (S,) int, the round the split was made in
    joint_round_kappa: np.ndarray         # (R,q) kappa after each round (spec 4.4)
    joint_round_se_kappa: np.ndarray      # (R,q) se(kappa) after each round
    joint_round_V_win: np.ndarray         # (R,q) V_win_hat after each round
    joint_round_margin: np.ndarray        # (R,q) the stop's margin after each round
    joint_round_n_splits: np.ndarray      # (R,) int, cumulative splits after each round
    joint_n_update_scale: np.ndarray     # (q,) int; A20 (spec/QIJ_mods_waves.md): bins whose
                                          # state-vector update scaled (`core.joint.
                                          # _apply_update`'s own count, every measured bin)
    joint_n_update_shift: np.ndarray     # (q,) int; ditto, shifted instead of scaled
    joint_n_update_negative: np.ndarray  # (q,) int; ditto, of n_update_scale, a negative ratio
    joint_rank_rule: str                 # A20 amendment (2 October): always 'measured_error'
                                          # (the method, no switch); 'n/a' on a failed draw
    z: float                       # the stop's margin (spec 0, 4.4); user-exposed, default 2.0
    n_min: int                     # splits required before the stop may fire; default 30
    L_max: int                     # the cap on leaves, always resolved (never None); default
                                    # M_X_used
    a: np.ndarray                  # (q,) ABC acceleration (spec/QIJ_mods_waves.md A10)
    b_hat: np.ndarray              # (q,) ABC second-order bias (A10)
    c_q: np.ndarray                # (q,) ABC curvature-along-influence, survey-row evaluated (A10)
    c_q_one_sided: np.ndarray      # (q,) bool: c_q fell back to the one-sided form (A10/A15)
    eta_Q: float                   # A9 item 3; NaN for an estimator without `takes_start`
    eta_full: float                # A15/A13 step 2; == the declared eta for an estimator
                                    # without `takes_start`
    survey_step_ratio: np.ndarray  # (5, q) A9 item 4; NaN rows/columns as eta_Q is
    quantized_start: str           # 'multistart' or 'full-data' (A9 item 5)
    fit_weights: str               # 'none' or 'mass' (spec/QIJ_mass_weighted_fit_spec.md);
                                    # the GP pilot's own setting, beside gptrend/gpwidth
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
