"""`QIJ`: the influence-aligned variance method (spec/method_notes.md
section 4).

`QIJ(eps=0.01, seed=0, vq_transform=None, gptrend='affine',
gpwidth='global', M_X=None, survey='points', quantized_start=
'multistart').fit(X, T, pool=None)` runs stage 1 (the X-VQ, prototype
influences, the GP pilot's initial influence estimate -- THE
ARCHITECTURE RULE, spec/QIJ_mods_waves.md A20, populates the one
per-point vector psi_hat here, in survey units) and stage 2
(`core.joint.run_joint`'s shared partition, grown on that vector,
measured on the full data, then the state-vector continuation that
carries every measurement into psi_hat in place, method_notes joint
section; A21 removed every other second-stage path -- there is no
longer a switch). `gptrend`/`gpwidth`/`fit_weights` pass straight to
`fit_influence_model` (method_notes section 3; `fit_weights`,
spec/QIJ_mass_weighted_fit_spec.md); `M_X` overrides the prototype-count
rule when given (method_notes section 2); `survey` picks the prototype
survey's receptive-field representation, `'points'` (one row per
prototype) or `'moments'` (spec/method_notes.md section 2), passed to
`run_xvq`.

Under the unified measured loop (spec/QIJ_unified_loop_spec.md,
replacing A20/A21's growth/check split), `run_joint` carries psi_hat
forward from one bin, rewritten in place after every measurement;
`QIJResult.psi_hat` is that final vector (`rho` stays NaN, since there
is no field+rho reconstruction left to build it from) and
`joint_n_update_scale`/`joint_n_update_shift`/`joint_n_update_negative`
are its own products. A18's separate survey-to-full-data scale step is
gone: the correction now enters bin by bin through the first splits'
own update (spec 4.0), so `psi0_all`/`offset`/`sigma_all` are never
rebound here after the fact. A leaf's ranking (which leaf to split
next) reads its own measured error against the update's noise term,
falling back to the current vector's own within-variance (the A20
ranking rule, kept verbatim); `joint_rank_rule` (always
`'measured_error'`) and `joint_bin_ebar` (each final bin's own error)
record it. The stop is calibrated by `kappa`/`se_kappa` (the ratio of
realized to predicted gain over every split so far) and reported via
`se_V_win`/`margin`, each a top-level `QIJResult` field beside
`V_btw`/`V_win_hat`/`V_tot_hat`; `z` (the stop's margin against both
sampling scatter and the within estimate's structural under-read --
V_win_hat/true ~0.6-0.9 on draw 1, NOT a pure confidence level),
`n_min` (splits required before the stop may fire) and `L_max` (the
cap on leaves, default `M_X_used`) are QIJ arguments and CLI flags,
recorded on `QIJResult` as given (z, n_min) or as resolved
(`L_max`, never None there).

`quantized_start`
(spec/QIJ_mods_waves.md A9 item 5) picks theta_Q's starting point:
`'multistart'` (ported) fits theta_Q from scratch on the survey rows;
`'full-data'` runs theta_hat first (on `pool` when given) and continues
it onto the survey rows via `start` (a no-op for an estimator without
`takes_start`). After stage 1, the
ABC interval's curvature ingredient (`core.abc.curvature`,
spec/QIJ_mods_waves.md A10) is measured on the survey rows for every
estimator, in its own `'curvature'` evaluation stage. With a `pool`:
the survey and stage 2's full-data stencils run on it, the full-data
base evaluation is submitted at the start and collected either before
the survey (`quantized_start='full-data'`) or after stage 1, the
influence model's width-grid candidates run on it (method_notes section
3), and the codebook fit uses `pool.workers` FAISS threads
(method_notes sections 2 and 4); a split's own evaluation
stays serial. A plain callable T is wrapped with outputs=('theta',) and
machine-precision eta; the method sees T only through `Counter`, never
an analytic influence.

An estimator carrying `measured` (a list of indices into `T.outputs`,
spec/QIJ_mods_waves.md A11) is measured on that subset only: the
prototype-count rule reads q = len(measured); the per-output stages
(the influence model, the joint partition, and the ABC curvature) still
fit every output, since one evaluation of T returns them all at no
extra cost, but only the measured indices are reported, so `QIJResult`'s
per-output fields and the prototype survey's `I_proto` carry the
measured outputs under their names. `theta_hat`/`theta_Q` passed as
`start` stay the full T output throughout, so a continuation always
sees every raw parameter; only the stored copy is restricted. Without
`measured` every index is its own position (0..q-1) and nothing changes
bit for bit.

`eta_full` (spec/QIJ_mods_waves.md A15, A13 step 2) is measured once
per draw, right after `theta_hat`, by `core.eta.measure_eta_full`, and
used in place of `T`'s own declared `eta` by every full-data evaluation
of stage 2 -- the joint path's initial bins and continuation splits
alike -- both for their stencil step and, for a restart estimator, the
polish's own acceptance tolerance (the per-call `eta` override,
interface sheet); its own evaluations are counted in a dedicated
`'eta_full'` stage, outside `'total'`'s audited meaning. Zero
evaluations, and `eta_full == T.eta` exactly, for an estimator without
`takes_start`, so every audited path stays bit-identical.

`sigma_points` (spec/QIJ_sigma_points_spec.md), off by default, runs an
optional stage after refinement and the curvature stage, on the full
data: the base fit is first brought to a fixed point of its own
continuation, then evaluated at 2n resample-scale perturbations along
the eigendirections of the refined influence estimate's own covariance,
in one pool batch (`core.sigma_points`, its own `'sigma'` stage, outside
`'total'`'s audited meaning). It adds `QIJResult.sigma_interval`; every
other product is exactly as before this option existed when it is
False.
"""
from __future__ import annotations

import time
from dataclasses import replace

import numpy as np
from numpy.linalg import LinAlgError

from .core import abc as core_abc
from .core.counter import Counter
from .core.eta import measure_eta_full
from .core.influence_model import fit_influence_model
from .core.influence_model import psi0 as _psi0
from .core.influence_model import uncertainty as _uncertainty
from .core.joint import run_joint
from .core.sigma_points import fit as fit_sigma_points
from .core.xvq import cost_rule_M, run_xvq
from .parallel import prepared
from .result import QIJResult


def _joint_defaults(N: int, q: int) -> dict:
    """The `joint_*` `QIJResult` fields, used by a draw that failed
    before stage 2 (`_failed_draw`) -- the unified loop never ran, so
    every count is zero/empty and `psi_hat` (populated here too, the
    one exception to the `joint_` prefix) is NaN."""
    return dict(
        joint_L=0, joint_n_splits=0, joint_n_rounds=0, joint_n_evals=0,
        joint_capped=False, joint_stop_met=False,
        joint_failed=False, joint_bin_mass=np.zeros(0), joint_bin_U=np.zeros((0, q)),
        joint_bin_m=np.zeros((0, q)), joint_bin_ebar=np.zeros((0, q)),
        joint_bin_W=np.zeros((0, q)), joint_bin_n=np.zeros(0, dtype=int),
        joint_bin_label=np.full(N, -1, dtype=int), joint_busy_delta=0.0,
        joint_split_parent=np.zeros(0, dtype=int),
        joint_split_child_a=np.zeros(0, dtype=int),
        joint_split_child_b=np.zeros(0, dtype=int),
        joint_split_G=np.zeros((0, q)), joint_split_D=np.zeros((0, q)),
        joint_split_round=np.zeros(0, dtype=int),
        joint_round_kappa=np.zeros((0, q)), joint_round_se_kappa=np.zeros((0, q)),
        joint_round_V_win=np.zeros((0, q)), joint_round_margin=np.zeros((0, q)),
        joint_round_n_splits=np.zeros(0, dtype=int),
        joint_n_update_scale=np.zeros(q, dtype=int), joint_n_update_shift=np.zeros(q, dtype=int),
        joint_n_update_negative=np.zeros(q, dtype=int),
        joint_rank_rule='n/a',
        psi_hat=np.full((N, q), np.nan),
    )


def _sigma_defaults(q: int) -> dict:
    """The `sigma_*` `QIJResult` fields when the stage did not run
    (spec/QIJ_sigma_points_spec.md 3), either `sigma_points=False` or a
    draw that failed before the stage was reached; `sigma_points` itself
    is set by the caller, not here."""
    return dict(
        sigma_status=None, n_fp_sigma=0, r_fp_sigma=float('nan'), n_failed_sigma=0,
        n_dirs_sigma=0, max_abs_d_sigma=float('nan'),
        sigma_mean=np.full(q, np.nan), sigma_sd=np.full(q, np.nan),
        sigma_bias=np.full(q, np.nan), sigma_k=np.zeros(0, dtype=int),
        sigma_sign=np.zeros(0, dtype=int), sigma_response=np.zeros((0, q)),
    )


def _wrap(T):
    """A plain callable gains outputs=('theta',) and machine-precision
    eta; a T already carrying `outputs` is used as given."""
    if hasattr(T, 'outputs'):
        return T
    T.outputs = ('theta',)
    T.eta = float(np.finfo(float).eps)
    T.name = getattr(T, '__name__', 'theta')
    return T


def _theta_hat_task(T, case, X: np.ndarray, _task):
    """theta_hat = T(X, ones(N)) on the pool, submitted at draw start
    (method_notes section 2). Returns (evaluation, failure, wall time)."""
    prep = prepared(T, X)
    t0 = time.perf_counter()
    try:
        w = np.ones(len(X))
        result = T(X, w, prep=prep) if prep is not None else T(X, w)
        result = np.asarray(result, dtype=float)
        failed = bool(np.any(np.isnan(result)))
    except Exception:
        result, failed = np.full(len(T.outputs), np.nan), True
    return result, failed, time.perf_counter() - t0


def _failed_draw(outputs, measured, N, xvq, W_X, I_proto, counter, t_start,
                  gptrend, gpwidth, M_X_source, workers, survey,
                  sv, quantized_start, fit_weights, sigma_points,
                  z, n_min, L_max) -> QIJResult:
    """The result of a draw whose influence model could not be fitted:
    every variance and model quantity NaN, every count after stage 1
    zero, the X-VQ and prototype survey diagnostics kept. `eta_Q` and
    `survey_step_ratio` come from `sv`, since the survey (stage 1) ran
    to completion before the influence model's own fit failed; the ABC
    interval's ingredients (`a`, `b_hat`, `c_q`, `c_q_one_sided`) and
    `eta_full` (spec/QIJ_mods_waves.md A15), which need stage 2 and the
    curvature/eta_full stages none of which ever ran, are NaN/False.
    The unified loop (spec/QIJ_unified_loop_spec.md) never ran, so
    every `joint_*` field is inert and `se_V_win`/`kappa`/`se_kappa`/
    `margin` are NaN, and the sigma-points stage (spec/QIJ_sigma_
    points_spec.md), which needs the loop's own refined influence,
    never ran either. `z`/`n_min` are recorded as given; `L_max` as
    resolved by the caller (`xvq.M_used` when the argument was None,
    since the loop itself never ran to resolve it). `I_proto` and
    `sv.step_ratio` are the full-width survey products; `measured`
    restricts them to the reported outputs, exactly as the successful
    path does (spec QIJ_mods_waves.md A11)."""
    q = len(outputs)
    nan_q, false_q = np.full(q, np.nan), np.zeros(q, dtype=bool)
    nan_Nq = np.full((N, q), np.nan)
    ev, rows = counter.snapshot()
    wall = time.perf_counter() - t_start
    return QIJResult(
        outputs=outputs, N=N, theta_hat=nan_q,
        theta_hat_full=np.full(np.asarray(I_proto).shape[1], np.nan),
        V_btw=nan_q, V_win_hat=nan_q, se_V_win=nan_q, V_tot_hat=nan_q,
        kappa=nan_q, se_kappa=nan_q, margin=nan_q,
        ell=nan_q, lam=nan_q, ell_bound=false_q, lam_bound=false_q,
        gptrend=gptrend, gpwidth=gpwidth, c=nan_q, c_bound=false_q,
        M_X=xvq.M_used, M_X_source=M_X_source, n_failed=counter.failed,
        evals_by_stage={'prototype': ev, 'full_data': 0, 'refinement': 0,
                        'curvature': 0, 'eta_full': 0, 'sigma': 0, 'total': ev},
        rows_by_stage={'prototype': rows, 'full_data': 0, 'refinement': 0,
                       'curvature': 0, 'eta_full': 0, 'sigma': 0, 'total': rows},
        wall_time_by_stage={'prototype': wall, 'full_data': 0.0, 'refinement': 0.0,
                            'curvature': 0.0, 'eta_full': 0.0, 'sigma': 0.0, 'total': wall},
        busy_time_total=wall, workers=workers,
        psi0=nan_Nq, sigma=nan_Nq,
        bmu=xvq.bmu, prototype_p=xvq.p,
        prototype_w=W_X, prototype_I=np.asarray(I_proto, dtype=float)[:, measured],
        prototype_h=np.full(xvq.M_used, np.nan), survey=survey,
        a=nan_q, b_hat=nan_q, c_q=nan_q, c_q_one_sided=false_q, eta_Q=sv.eta_Q,
        eta_full=float('nan'),
        survey_step_ratio=np.asarray(sv.step_ratio, dtype=float)[:, measured],
        quantized_start=quantized_start, fit_weights=fit_weights,
        sigma_points=sigma_points,
        z=z, n_min=n_min, L_max=(xvq.M_used if L_max is None else L_max),
        **_joint_defaults(N, q), **_sigma_defaults(q),
    )


class QIJ:
    """The QIJ method (spec/method_notes.md section 4)."""

    def __init__(self, eps: float = 0.01, seed: int = 0, vq_transform=None,
                 gptrend: str = 'affine', gpwidth: str = 'global', M_X: int = None,
                 survey: str = 'points', quantized_start: str = 'multistart',
                 sigma_points: bool = False, fit_weights: str = 'none',
                 z: float = 2.0, n_min: int = 30, L_max: int = None) -> None:
        """`z`, `n_min`, `L_max` are the unified loop's own user-exposed
        levers (spec/QIJ_unified_loop_spec.md 0, 4.4, 5), passed straight
        to `core.joint.run_joint`. `z` (default 2.0) is the stop's
        margin against BOTH the within estimate's sampling scatter AND
        its structural under-read (V_win_hat/true ~0.6-0.9 on draw 1)
        -- NOT a pure confidence level: lowering it lowers that margin
        too. `n_min` (default 30) is the minimum number of splits made
        before the stop may fire (a ratio's standard error read from
        fewer splits is itself unreliable). `L_max` (default None ->
        `M_X_used`, resolved inside `run_joint`) caps the number of
        leaves; `QIJResult.L_max` always records the resolved value,
        never None."""
        if not z > 0:
            raise ValueError(f'z must be > 0 (got {z})')
        if n_min < 1:
            raise ValueError(f'n_min must be >= 1 (got {n_min})')
        if L_max is not None and L_max < 2:
            raise ValueError(f'L_max must be >= 2 or None (got {L_max})')
        self.eps = eps
        self.seed = seed
        self.vq_transform = vq_transform
        self.gptrend = gptrend
        self.gpwidth = gpwidth
        self.M_X = M_X
        self.survey = survey
        self.quantized_start = quantized_start
        self.sigma_points = sigma_points
        self.fit_weights = fit_weights
        self.z = z
        self.n_min = n_min
        self.L_max = L_max

    def fit(self, X: np.ndarray, T, pool=None) -> QIJResult:
        """Run the method on one draw: stage 1 (X-VQ, prototype
        influences, the GP pilot), the shared full-data base evaluation,
        the ABC curvature stage, then the joint second stage
        (`core.joint.run_joint`), then, under `sigma_points=True`, the
        optional sigma-points interval stage (spec/QIJ_sigma_points_
        spec.md), on `pool` per the module docstring."""
        t_start = time.perf_counter()
        X = np.asarray(X)
        N = len(X)
        T = _wrap(T)
        eta = T.eta
        outputs_full = T.outputs
        # `measured` (spec/QIJ_mods_waves.md A11): the indices of
        # `outputs_full` QIJ reports; identity when T carries no
        # `measured` attribute, so every downstream `[measured]` slice is
        # a no-op and the draw is bit-identical to before A11.
        measured = list(getattr(T, 'measured', range(len(outputs_full))))
        outputs = tuple(outputs_full[i] for i in measured)
        q_full = len(outputs_full)
        q = len(measured)
        workers = pool.workers if pool is not None else 1

        counter = Counter(T, N)

        theta_future = None
        if pool is not None:
            pool.share(X)
            theta_future = pool.submit(_theta_hat_task, None)

        # `vq_transform` returns (Z, inverse) fitted to this draw;
        # identity when `vq_transform` is None. A 1-D Z is promoted to
        # a column for the quantizer and un-promoted on the way out.
        Z, inverse = self.vq_transform(X) if self.vq_transform is not None else (X, lambda A: A)
        Z = np.asarray(Z, dtype=float)
        if Z.ndim == 1:
            Z = Z.reshape(-1, 1)
            inverse = lambda A, _inv=inverse: _inv(A.reshape(-1))

        # `M_X` given overrides the cost rule (method_notes section 2).
        if self.M_X is not None:
            M_requested, M_X_source = self.M_X, 'argument'
        else:
            M_requested, M_X_source = cost_rule_M(N, q, self.eps), 'rule'

        # `quantized_start='full-data'` (A9 item 5) needs theta_hat before
        # theta_Q's own fit, so the shared full-data evaluation is
        # collected here rather than after stage 1; under 'multistart'
        # (the default) this block is skipped and theta_hat is collected
        # in its usual place below, bit-identical to before.
        theta_hat_pre = None
        full_data_busy = 0.0
        wall_time_full_data = 0.0
        if self.quantized_start == 'full-data':
            t0 = time.perf_counter()
            if theta_future is None:
                theta_hat_pre = np.asarray(counter(X, np.ones(N)), dtype=float)
            else:
                result, failed, full_data_busy = theta_future.result()
                theta_future = None
                counter.add(1, N, int(failed))
                theta_hat_pre = np.asarray(result, dtype=float)
            wall_time_full_data = time.perf_counter() - t0
        # With theta_hat in hand before the survey, eta_full (below) is
        # measured now and is also the polish tolerance of every
        # evaluation on the survey rows: T's declared eta can sit below
        # what a fit reproduces (cloudfil draw 1: 1e-12 against a polish
        # floor of 2e-12), which fails every survey evaluation.
        eta_rows = None
        eta_full_pre = None
        if theta_hat_pre is not None:
            t0 = time.perf_counter()
            ev_e0, rows_e0 = counter.snapshot()
            eta_full_pre, _n_eta_full, _ = measure_eta_full(counter, X, theta_hat_pre)
            wall_time_eta_full = time.perf_counter() - t0
            ev_e1, rows_e1 = counter.snapshot()
            evals_eta_full = ev_e1 - ev_e0
            rows_eta_full = rows_e1 - rows_e0
            if getattr(counter, 'takes_start', False):
                eta_rows = eta_full_pre
        ev0, rows0 = counter.snapshot()  # the eta_full evaluations above are excluded

        t0 = time.perf_counter()
        xvq, theta_Q, I_proto, xvq_busy, sv = run_xvq(
            Z, inverse, counter, eta, M_requested, self.seed, pool, workers,
            survey=self.survey, X=X, quantized_start=self.quantized_start,
            theta_hat=theta_hat_pre, eta_rows=eta_rows)
        # Prototype positions in T's native coordinates, for the
        # result's diagnostics -- the same `inverse(xvq.centers)`
        # `run_xvq` already applied, recovered rather than re-derived.
        W_X = np.asarray(inverse(xvq.centers), dtype=float)
        try:
            # Fit only the MEASURED outputs' GPs (spec/QIJ_mods_waves.md
            # A11): `model` (and, below, `psi0_all`/`sigma_all`) are then
            # indexed by LOCAL position (0..q-1, `measured`'s own order),
            # never by the absolute index into `theta_hat`/`I_proto`,
            # which stay full width for T's own continuation and for the
            # bin measurements that reuse one shared evaluation across
            # every output. Identity `measured` reproduces today's
            # full-width model bit for bit.
            model, model_busy = fit_influence_model(
                Z, xvq, I_proto[:, measured], theta_Q[measured], eta,
                gptrend=self.gptrend, gpwidth=self.gpwidth, pool=pool,
                fit_weights=self.fit_weights)
        except (RuntimeError, LinAlgError):
            # A failed stage 1 is recorded as failed, never retried; a
            # submitted `theta_future` is left uncollected and uncounted,
            # as the serial code never reaches its own call here either.
            return _failed_draw(outputs, measured, N, xvq, W_X, I_proto, counter, t_start,
                                 self.gptrend, self.gpwidth, M_X_source, workers,
                                 self.survey, sv, self.quantized_start,
                                 self.fit_weights, self.sigma_points,
                                 self.z, self.n_min, self.L_max)
        # THE ARCHITECTURE RULE (spec/QIJ_mods_waves.md A20): `psi0_all`
        # is populated ONCE here, in survey units -- the GP's point
        # predictions -- and from this point on it IS the one per-point
        # vector psi_hat the rest of the draw reads and (from stage 2's
        # first measurement on) rewrites in place. `sigma_all` (the GP
        # posterior sd) is a stored survey diagnostic only (A21's
        # architecture rule: it feeds no decision and is never carried
        # alongside psi_hat as a second vector).
        psi0_all = _psi0(model, Z)
        sigma_all = _uncertainty(model, Z)
        offset = np.asarray(model.offset, dtype=float)
        wall_time_prototype = time.perf_counter() - t0
        ev1, rows1 = counter.snapshot()

        # The ABC interval's curvature ingredient (A10): 2q evaluations
        # on the SURVEY rows, along each output's field-level influence
        # direction, `start=theta_Q` inside `core_abc.curvature`. Run
        # here, before the survey's shared rows are replaced by X below,
        # and counted in its own stage so 'prototype' keeps its meaning
        # and 'total' (below) excludes it, keeping every audited count
        # unchanged.
        t0 = time.perf_counter()
        # `outputs=measured` (A11): `theta_Q`/`I_proto` stay full width
        # (needed for the continuation and for the cheap sigma_Q/u pass
        # over every output), but the two evaluations per output only
        # run for the measured indices; `c_q` comes back already at
        # length q, in `measured`'s order.
        c_q, _curvature_eps, curvature_busy, c_q_one_sided = core_abc.curvature(
            counter, sv, theta_Q, I_proto, xvq.p, N, pool=pool, outputs=measured)
        wall_time_curvature = time.perf_counter() - t0
        ev1c, rows1c = counter.snapshot()
        evals_curvature = ev1c - ev1
        rows_curvature = rows1c - rows1

        t0 = time.perf_counter()
        if theta_hat_pre is not None:
            theta_hat = theta_hat_pre
        elif theta_future is None:
            theta_hat = np.asarray(counter(X, np.ones(N)), dtype=float)
            wall_time_full_data = time.perf_counter() - t0
        else:
            result, failed, full_data_busy = theta_future.result()
            theta_hat = np.asarray(result, dtype=float)
            counter.add(1, N, int(failed))
            wall_time_full_data = time.perf_counter() - t0

        # eta_full (spec/QIJ_mods_waves.md A15, A13 step 2): measured
        # once, right after theta_hat, and used in place of T's own
        # declared eta by every full-data evaluation of stage 2 below
        # (module docstring); its own evaluations are counted in their
        # own stage, never in 'full_data' or 'total'. Zero evaluations,
        # eta_full == eta exactly, for an estimator without
        # `takes_start` (`measure_eta_full`), so this is a no-op there.
        if eta_full_pre is not None:
            eta_full = eta_full_pre
        else:
            t0 = time.perf_counter()
            ev_e0, rows_e0 = counter.snapshot()
            eta_full, _n_eta_full, _ = measure_eta_full(counter, X, theta_hat)
            wall_time_eta_full = time.perf_counter() - t0
            ev_e1, rows_e1 = counter.snapshot()
            evals_eta_full = ev_e1 - ev_e0
            rows_eta_full = rows_e1 - rows_e0

        if pool is not None:
            pool.share(X)  # re-share X: the survey shared its own rows on `pool`

        # A9 item 2: the shared full-data base value continues into
        # stage 2's own evaluations only for an estimator with restarts;
        # None is a no-op for every other estimator (bit-identical).
        start_second_stage = theta_hat if getattr(T, 'takes_start', False) else None

        t0 = time.perf_counter()
        # `psi0_all`/`model` are already the measured subset (this
        # call's own `fit_influence_model` above), so `run_joint`'s own
        # psi0-driven fields (V_btw, V_win_hat, se_V_win, V_tot_hat,
        # kappa, se_kappa, margin, the bin/split/round constituents)
        # come back at that same measured width already, no further
        # slicing. `B_hat`/`a_bca` are `ivq.bias_and_acceleration`'s
        # result against the FULL `theta_hat` (one shared evaluation
        # covers every output, as above), so they stay q_full wide and
        # are restricted to `measured` here (spec/QIJ_mods_waves.md
        # A11).
        jr = run_joint(X, counter, theta_hat, psi0_all, model, xvq, eta_full,
                       self.eps, offset, pool=pool, start=start_second_stage,
                       measured=measured, z=self.z, n_min=self.n_min, L_max=self.L_max)
        # A failed output, or a failed initial bin measurement before
        # any continuation ran, voids every output's variance quantities.
        if jr.failed:
            jr = replace(jr, V_btw=np.full(q, np.nan), V_win_hat=np.full(q, np.nan),
                          se_V_win=np.full(q, np.nan), V_tot_hat=np.full(q, np.nan),
                          kappa=np.full(q, np.nan), se_kappa=np.full(q, np.nan),
                          margin=np.full(q, np.nan),
                          B_hat=np.full(q_full, np.nan), a_bca=np.full(q_full, np.nan))
        second_stage_evals = int(jr.n_evals)
        second_stage_busy = float(jr.busy_delta)
        second_stage_fields = dict(V_btw=jr.V_btw, V_win_hat=jr.V_win_hat,
                                    se_V_win=jr.se_V_win, V_tot_hat=jr.V_tot_hat,
                                    kappa=jr.kappa, se_kappa=jr.se_kappa, margin=jr.margin,
                                    a=jr.a_bca[measured], b_hat=jr.B_hat[measured])
        if not jr.failed:
            # spec/QIJ_unified_loop_spec.md 4.0: A18's separate survey-
            # to-full-data scale step is GONE -- the correction now
            # enters bin by bin through the first splits' own state
            # update inside `run_joint`, so `psi0_all`/`offset`/
            # `sigma_all` are never rebound here after the fact (unlike
            # the old A18 a_c step this replaces). `jr.state_psi_hat`
            # IS the method's own final per-point estimate -- the A18
            # pilot, rewritten in place by every measurement -- taken
            # directly; rho is NOT produced (stays NaN), since there is
            # no field+rho reconstruction left to do (A21 removed it).
            second_stage_fields['psi_hat'] = jr.state_psi_hat
        else:
            second_stage_fields['psi_hat'] = np.full((N, q), np.nan)
        joint_fields = dict(
            joint_L=jr.L, joint_n_splits=jr.n_splits, joint_n_rounds=jr.n_rounds,
            joint_n_evals=jr.n_evals, joint_capped=jr.capped, joint_stop_met=jr.stop_met,
            joint_failed=jr.failed,
            joint_bin_mass=jr.bin_mass, joint_bin_U=jr.bin_U, joint_bin_m=jr.bin_m,
            joint_bin_ebar=jr.bin_ebar, joint_bin_W=jr.bin_W, joint_bin_n=jr.bin_n,
            joint_bin_label=jr.bin_label,
            joint_busy_delta=jr.busy_delta,
            joint_split_parent=jr.split_parent, joint_split_child_a=jr.split_child_a,
            joint_split_child_b=jr.split_child_b, joint_split_G=jr.split_G,
            joint_split_D=jr.split_D, joint_split_round=jr.split_round,
            joint_round_kappa=jr.round_kappa, joint_round_se_kappa=jr.round_se_kappa,
            joint_round_V_win=jr.round_V_win, joint_round_margin=jr.round_margin,
            joint_round_n_splits=jr.round_n_splits,
            joint_n_update_scale=jr.n_update_scale, joint_n_update_shift=jr.n_update_shift,
            joint_n_update_negative=jr.n_update_negative,
            joint_rank_rule=jr.rank_rule,
        )
        wall_time_refinement = time.perf_counter() - t0
        ev3, rows3 = counter.snapshot()

        # The sigma-points stage (spec/QIJ_sigma_points_spec.md 2): after
        # refinement and the curvature stage, on the full data, from the
        # method's own refined influence estimate (`psi_hat`, already at
        # the measured width). A no-op, at zero cost, under
        # `sigma_points=False` (module docstring); `pool` already shares
        # `X` (re-shared above stage 2, never replaced by stage 2 itself).
        t0 = time.perf_counter()
        if self.sigma_points:
            sr = fit_sigma_points(counter, X, theta_hat, second_stage_fields['psi_hat'],
                                   eta_full, N, measured, pool=pool)
            sigma_busy = sr.busy
            sigma_fields = dict(
                sigma_points=True, sigma_status=sr.status, n_fp_sigma=sr.n_fp,
                r_fp_sigma=sr.r_fp, n_failed_sigma=sr.n_failed, n_dirs_sigma=sr.n_dirs,
                max_abs_d_sigma=sr.max_abs_d, sigma_mean=sr.mean, sigma_sd=sr.sd,
                sigma_bias=sr.bias, sigma_k=sr.k, sigma_sign=sr.sign,
                sigma_response=sr.response)
        else:
            sigma_busy = 0.0
            sigma_fields = dict(sigma_points=False, **_sigma_defaults(q))
        wall_sigma = time.perf_counter() - t0
        ev3b, rows3b = counter.snapshot()
        evals_sigma, rows_sigma = ev3b - ev3, rows3b - rows3

        # 'prototype' is stage 1's own work, net of the early theta_hat
        # evaluation `quantized_start='full-data'` may have spent before
        # it (ev0/rows0, 0 under 'multistart'); that evaluation's cost
        # joins 'full_data' below instead, alongside the late theta_hat
        # call this draw makes instead when it was NOT spent early.
        # eta_full's own evaluations (spec/QIJ_mods_waves.md A15) are
        # counted in their own stage, excluded from 'full_data' and
        # 'total' (module docstring); 0 for an estimator without
        # `takes_start`, so both stay bit-identical there.
        evals_by_stage = {'prototype': ev1 - ev0,
                           'full_data': ev0 + (ev3 - ev1c) - second_stage_evals - evals_eta_full,
                           'refinement': second_stage_evals, 'curvature': evals_curvature,
                           'eta_full': evals_eta_full, 'sigma': evals_sigma,
                           'total': ev3 - evals_curvature - evals_eta_full}
        rows_by_stage = {'prototype': rows1 - rows0,
                          'full_data': rows0 + (rows3 - rows1c) - second_stage_evals * N - rows_eta_full,
                          'refinement': second_stage_evals * N, 'curvature': rows_curvature,
                          'eta_full': rows_eta_full, 'sigma': rows_sigma,
                          'total': rows3 - rows_curvature - rows_eta_full}
        wall_time_total = time.perf_counter() - t_start
        wall_time_by_stage = {'prototype': wall_time_prototype, 'full_data': wall_time_full_data,
                               'refinement': wall_time_refinement, 'curvature': wall_time_curvature,
                               'eta_full': wall_time_eta_full, 'sigma': wall_sigma,
                               'total': wall_time_total}
        # Parent work plus every pool task's own time, in place of each
        # parallel stage's share of the elapsed total.
        busy_time_total = (wall_time_total + xvq_busy + model_busy + full_data_busy
                            + second_stage_busy + curvature_busy + sigma_busy)

        # at_bound[:, 0] is whichever width parameter gpwidth fits
        # (method_notes section 3); `model` is already the measured
        # subset (spec/QIJ_mods_waves.md A11), so every model-derived
        # field below is used as-is, at `model`'s own (measured) width
        # -- only the T-output arrays (`theta_hat`, `I_proto`, `c_q`,
        # `survey_step_ratio`) still need `[measured]`.
        local = self.gpwidth == 'local'
        ell_bound = np.zeros(q, dtype=bool) if local else model.at_bound[:, 0].copy()
        c_bound = model.at_bound[:, 0].copy() if local else np.zeros(q, dtype=bool)
        ell, lam = np.array(model.width), np.array(model.lam)
        lam_bound, c_arr = model.at_bound[:, 1].copy(), np.array(model.c)
        prototype_h = np.array(model.h)

        return QIJResult(
            outputs=outputs, N=N, theta_hat=theta_hat[measured], theta_hat_full=theta_hat,
            ell=ell, lam=lam, ell_bound=ell_bound, lam_bound=lam_bound,
            gptrend=self.gptrend, gpwidth=self.gpwidth,
            c=c_arr, c_bound=c_bound,
            M_X=xvq.M_used, M_X_source=M_X_source, n_failed=counter.failed,
            evals_by_stage=evals_by_stage, rows_by_stage=rows_by_stage,
            wall_time_by_stage=wall_time_by_stage,
            busy_time_total=busy_time_total, workers=workers,
            psi0=psi0_all, sigma=sigma_all,
            bmu=xvq.bmu, prototype_p=xvq.p, prototype_w=W_X, prototype_h=prototype_h,
            prototype_I=np.asarray(I_proto, dtype=float)[:, measured],
            survey=self.survey,
            c_q=c_q, c_q_one_sided=c_q_one_sided, eta_Q=sv.eta_Q, eta_full=float(eta_full),
            survey_step_ratio=np.asarray(sv.step_ratio, dtype=float)[:, measured],
            z=jr.z, n_min=jr.n_min, L_max=jr.L_max,
            quantized_start=self.quantized_start,
            fit_weights=self.fit_weights,
            **second_stage_fields, **joint_fields, **sigma_fields,
        )
