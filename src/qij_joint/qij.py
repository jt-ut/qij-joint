"""`QIJ`: the influence-aligned variance method (spec/method_notes.md
section 4).

`QIJ(eps=0.01, seed=0, vq_transform=None, gptrend='affine',
gpwidth='global', M_X=None, ivqbins='marginal', survey='points',
quantized_start='multistart', pilot='affine').fit(X, T, pool=None)` runs
stage 1 (the X-VQ, prototype influences, initial influence estimate)
and stage 2 (the shared full-data evaluation, then `ivqbins`'s
`'marginal'` per-output refinement or `'joint'` shared partition,
method_notes joint section), returning a `QIJResult`. `pilot`
(spec/QIJ_affine_pilot_spec.md) picks stage 1's initial influence
estimate: `'affine'` (the default), the mean-preserving per-cell affine
field of `core.affine_pilot`, with no posterior variance and the
refinement's adjacency proposals priced by the bridge score
(spec/QIJ_affine_pilot_spec.md 3.2); or `'gp'`, today's Gaussian-process
pilot, bit-identical to before this option existed. `gptrend`/`gpwidth`
pass straight to `fit_influence_model` (method_notes section 3) and are
unused under `pilot='affine'`; `M_X` overrides the prototype-count rule
when given (method_notes section 2); `survey` picks the prototype
survey's receptive-field representation, `'points'` (one row per
prototype) or `'moments'` (spec/method_notes.md section 2), passed to
`run_xvq`. `ivqbins='joint'` runs under either pilot
(spec/QIJ_affine_pilot_spec.md 3.3): `core.joint.run_joint`'s own flag
and kind pricing follows `pilot` the same way the marginal path's
`core.refine.prepare_coordinate` does, choosing once, at setup, between
the GP posterior's rule and the bridge score's. `quantized_start`
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
(method_notes sections 2 and 4); a marginal split's own evaluation
stays serial. A plain callable T is wrapped with outputs=('theta',) and
machine-precision eta; the method sees T only through `Counter`, never
an analytic influence.

An estimator carrying `measured` (a list of indices into `T.outputs`,
spec/QIJ_mods_waves.md A11) is measured on that subset only: the
prototype-count rule reads q = len(measured); the per-output stages
(the influence model, the marginal refinement or the joint partition,
and the ABC curvature) still fit every output, since one evaluation of
T returns them all at no extra cost, but only the measured indices are
reported, so `QIJResult`'s per-output fields and the prototype survey's
`I_proto` carry the measured outputs under their names; the marginal
refinement itself runs only for the measured outputs, since that
stage's own cost is one evaluation loop per output. `theta_hat`/
`theta_Q` passed as `start` stay the full T output throughout, so a
continuation always sees every raw parameter; only the stored copy is
restricted. Without `measured` every index is its own position
(0..q-1) and nothing changes bit for bit.

`refine_schedule` picks the marginal path's own schedule
(spec/QIJ_mods_waves.md A14): `'queue'` (ported, the default) runs
`core.refine.run_refinement` once per measured output, its evaluations
always serial; `'rounds'` runs every measured output's refinement
together through `core.rounds.run_refinement_rounds`, which shares one
pool batch per round across outputs. Under `ivqbins='joint'` the switch
is inert -- the joint path's own check already advances in rounds, not
a queue.

`eta_full` (spec/QIJ_mods_waves.md A15, A13 step 2) is measured once
per draw, right after `theta_hat`, by `core.eta.measure_eta_full`, and
used in place of `T`'s own declared `eta` by every full-data evaluation
of stage 2 -- the marginal refinement's and the joint check's initial
bins and splits alike -- both for their stencil step and, for a
restart estimator, the polish's own acceptance tolerance (the per-call
`eta` override, interface sheet); its own evaluations are counted in a
dedicated `'eta_full'` stage, outside `'total'`'s audited meaning. Zero
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
from .core.affine_pilot import fit as fit_affine_pilot
from .core.counter import Counter
from .core.eta import measure_eta_full
from .core.influence_model import fit_influence_model
from .core.influence_model import psi0 as _psi0
from .core.influence_model import uncertainty as _uncertainty
from .core.joint import joint_psi_hat, run_joint
from .core.refine import run_refinement
from .core.rounds import run_refinement_rounds
from .core.sigma_points import fit as fit_sigma_points
from .core.xvq import cost_rule_M, run_xvq
from .parallel import prepared
from .result import QIJResult


def _joint_defaults(N: int, q: int) -> dict:
    """The `joint_*` `QIJResult` fields, inert under `ivqbins='marginal'`."""
    return dict(
        joint_S_pred=np.full(q, np.nan), joint_a=np.full(q, np.nan),
        joint_S_pred_pre_lloyd=np.full(q, np.nan),
        joint_L0=0, joint_L=0, joint_n_growth_rounds=0, joint_growth_capped=False,
        joint_n_flagged=0, joint_n_check_rounds=0, joint_n_check_evals=0,
        joint_n_level_splits=0, joint_n_adjacency_splits=0, joint_check_capped=False,
        joint_failed=False, joint_bin_mass=np.zeros(0), joint_bin_U=np.zeros((0, q)),
        joint_bin_m=np.zeros((0, q)), joint_bin_flagged=np.zeros(0, dtype=bool),
        joint_bin_label=np.full(N, -1, dtype=int), joint_busy_delta=0.0,
    )


def _marginal_defaults(N: int, q: int) -> dict:
    """The marginal-only `QIJResult` fields, inert under `ivqbins='joint'` --
    except `psi_hat`/`rho`, which the joint branch of `QIJ.fit` overrides
    with `core.joint.joint_psi_hat`'s own field/rho_c on a draw that did
    not fail; these NaN defaults are what a failed joint draw keeps."""
    return dict(
        L=np.zeros(q, dtype=int), n_level_splits=np.zeros(q, dtype=int),
        n_adjacency_splits=np.zeros(q, dtype=int), rho=np.full(q, np.nan),
        n_refine_evals=np.zeros(q, dtype=int),
        psi_hat=np.full((N, q), np.nan), bin_label=np.full((N, q), -1, dtype=int),
        bin_U=(),
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


def _empty_pilot_products(q: int) -> dict:
    """Empty `bridge`/`cells` array products (spec/QIJ_affine_pilot_spec.md
    2.3, 4): under `pilot='gp'`, or a draw that never reached the affine
    pilot's own fit, both are empty."""
    return dict(
        bridge_j=np.zeros(0, dtype=int), bridge_k=np.zeros(0, dtype=int),
        bridge_m=np.zeros(0, dtype=float), bridge_delta=np.zeros((0, q), dtype=float),
        cell_p=np.zeros(0, dtype=float), cell_mu=np.zeros((0, 0), dtype=float),
        cell_g=np.zeros((0, 0, q), dtype=float),
    )


def _failed_draw(outputs, measured, N, xvq, W_X, I_proto, counter, t_start,
                  gptrend, gpwidth, M_X_source, workers, ivqbins, survey,
                  sv, quantized_start, refine_schedule, pilot, sigma_points) -> QIJResult:
    """The result of a draw whose influence model could not be fitted:
    every variance and model quantity NaN, every count after stage 1
    zero, the X-VQ and prototype survey diagnostics kept. `eta_Q` and
    `survey_step_ratio` come from `sv`, since the survey (stage 1) ran
    to completion before the influence model's own fit failed; the ABC
    interval's ingredients (`a`, `b_hat`, `c_q`, `c_q_one_sided`) and
    `eta_full` (spec/QIJ_mods_waves.md A15), which need stage 2 and the
    curvature/eta_full stages none of which ever ran, are NaN/False. The
    second stage never ran under either value of `ivqbins`, so both the
    marginal and the joint fields are inert, and the sigma-points stage
    (spec/QIJ_sigma_points_spec.md), which needs stage 2's own refined
    influence, never ran either. `I_proto` and `sv.step_ratio`
    are the full-width survey products; `measured` restricts them to the
    reported outputs, exactly as the successful path does (spec
    QIJ_mods_waves.md A11)."""
    q = len(outputs)
    nan_q, false_q = np.full(q, np.nan), np.zeros(q, dtype=bool)
    nan_Nq = np.full((N, q), np.nan)
    ev, rows = counter.snapshot()
    wall = time.perf_counter() - t_start
    return QIJResult(
        outputs=outputs, N=N, theta_hat=nan_q,
        theta_hat_full=np.full(np.asarray(I_proto).shape[1], np.nan),
        V_btw=nan_q, V_win_hat=nan_q, V_tot_hat=nan_q, gain_ratio=nan_q,
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
        prototype_h=np.full(xvq.M_used, np.nan), ivqbins=ivqbins, survey=survey,
        a=nan_q, b_hat=nan_q, c_q=nan_q, c_q_one_sided=false_q, eta_Q=sv.eta_Q,
        eta_full=float('nan'),
        survey_step_ratio=np.asarray(sv.step_ratio, dtype=float)[:, measured],
        quantized_start=quantized_start, refine_schedule=refine_schedule, n_rounds=0,
        pilot=pilot, sigma_points=sigma_points,
        **_marginal_defaults(N, q), **_joint_defaults(N, q), **_empty_pilot_products(q),
        **_sigma_defaults(q),
    )


class QIJ:
    """The QIJ method (spec/method_notes.md section 4)."""

    def __init__(self, eps: float = 0.01, seed: int = 0, vq_transform=None,
                 gptrend: str = 'affine', gpwidth: str = 'global', M_X: int = None,
                 ivqbins: str = 'marginal', survey: str = 'points',
                 quantized_start: str = 'multistart',
                 refine_schedule: str = 'queue', pilot: str = 'affine',
                 sigma_points: bool = False) -> None:
        if pilot not in ('affine', 'gp'):
            raise ValueError(f"unknown pilot {pilot!r}")
        self.eps = eps
        self.seed = seed
        self.vq_transform = vq_transform
        self.gptrend = gptrend
        self.gpwidth = gpwidth
        self.M_X = M_X
        self.ivqbins = ivqbins
        self.survey = survey
        self.refine_schedule = refine_schedule
        self.quantized_start = quantized_start
        self.pilot = pilot
        self.sigma_points = sigma_points

    def fit(self, X: np.ndarray, T, pool=None) -> QIJResult:
        """Run the method on one draw: stage 1 (X-VQ, prototype
        influences), the shared full-data base evaluation, the ABC
        curvature stage, then per-output refinement, then, under
        `sigma_points=True`, the optional sigma-points interval stage
        (spec/QIJ_sigma_points_spec.md), on `pool` per the module
        docstring."""
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
        pilot_result = None
        if self.pilot == 'gp':
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
                    gptrend=self.gptrend, gpwidth=self.gpwidth, pool=pool)
            except (RuntimeError, LinAlgError):
                # A failed stage 1 is recorded as failed, never retried; a
                # submitted `theta_future` is left uncollected and uncounted,
                # as the serial code never reaches its own call here either.
                return _failed_draw(outputs, measured, N, xvq, W_X, I_proto, counter, t_start,
                                     self.gptrend, self.gpwidth, M_X_source, workers, self.ivqbins,
                                     self.survey, sv, self.quantized_start, self.refine_schedule,
                                     self.pilot, self.sigma_points)
            psi0_all = _psi0(model, Z)
            sigma_all = _uncertainty(model, Z)
            offset = np.asarray(model.offset, dtype=float)
            constant_path_arr = np.asarray(model.constant_path, dtype=bool)
        else:
            # The affine pilot (spec/QIJ_affine_pilot_spec.md 2): no
            # posterior variance (`sigma_all` all NaN), no width search
            # (`model_busy` 0), `I_proto` at FULL width (section 6, cheap
            # for every output at once) then sliced to `measured` here,
            # where `qij.fit` slices every other stage-1 product.
            # `constant_path` is never raised for this pilot: a genuinely
            # flat output is still caught downstream by the initial I-VQ's
            # own `M_used <= 1` (`refine.prepare_coordinate`).
            model, model_busy = None, 0.0
            pilot_result = fit_affine_pilot(Z, xvq, I_proto)
            psi0_all = pilot_result.psi0[:, measured]
            sigma_all = np.full((N, q), np.nan)
            offset = psi0_all.mean(axis=0)
            constant_path_arr = np.zeros(q, dtype=bool)
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
        n_rounds = 0
        if self.ivqbins == 'joint':
            # `psi0_all`/`sigma_all`/`model` are already the measured
            # subset (this call's own `fit_influence_model` above), so
            # `run_joint`'s own psi0-driven fields (V_btw, V_win_hat,
            # V_tot_hat, gain_ratio, S_pred, the check's own scale `a`,
            # the bin constituents) come back at that same measured
            # width already, no further slicing. `B_hat`/`a_bca` are
            # `ivq.bias_and_acceleration`'s result against the FULL
            # `theta_hat` (one shared evaluation covers every output, as
            # in the marginal path below), so they stay q_full wide and
            # are restricted to `measured` here (spec/QIJ_mods_waves.md
            # A11; inert for the demo, which never runs this path).
            # `pilot` picks `run_joint`'s own flag/kind pricing the same
            # way it picks the marginal path's below; under `'affine'`
            # the bridge pair geometry and this draw's own m_jk*Delta_jk^2
            # over every measured output (`bridge_value`, spec/QIJ_affine_
            # pilot_spec.md 3.2) come from the same `pilot_result` the
            # marginal path's `bridge_value_cols` reads per output.
            if self.pilot == 'affine':
                bridge_pair_id = pilot_result.pair_id
                bridge_pair_n = pilot_result.pair_n
                bridge_value = pilot_result.pair_mass[:, None] * pilot_result.pair_delta[:, measured] ** 2
            else:
                bridge_pair_id = bridge_pair_n = bridge_value = None
            jr = run_joint(X, counter, theta_hat, psi0_all, sigma_all, model, Z, xvq, eta_full,
                           self.eps, offset, pool=pool, I_proto=I_proto[:, measured],
                           start=start_second_stage, measured=measured, pilot=self.pilot,
                           bridge_pair_id=bridge_pair_id, bridge_pair_n=bridge_pair_n,
                           bridge_value=bridge_value)
            # A failed output, or a failed initial bin measurement before
            # any check ran, voids every output's variance quantities, as
            # the marginal path voids them on any one coordinate's failure.
            if jr.failed:
                jr = replace(jr, V_btw=np.full(q, np.nan), V_win_hat=np.full(q, np.nan),
                              V_tot_hat=np.full(q, np.nan), B_hat=np.full(q_full, np.nan),
                              a_bca=np.full(q_full, np.nan))
            second_stage_evals = int(jr.n_check_evals)
            second_stage_busy = float(jr.busy_delta)
            second_stage_fields = dict(V_btw=jr.V_btw, V_win_hat=jr.V_win_hat,
                                        V_tot_hat=jr.V_tot_hat, gain_ratio=jr.gain_ratio,
                                        a=jr.a_bca[measured], b_hat=jr.B_hat[measured],
                                        **_marginal_defaults(N, q))
            if not jr.failed:
                # psi_hat/rho (spec/method_notes.md section 6): the
                # marginal path's own per-coordinate field, evaluated on
                # the joint path's shared final bins instead of a private
                # partition per output; NaN (the `_marginal_defaults`
                # above) when the draw failed, since `bin_label`/`bin_U`
                # may then be unusable (`joint_psi_hat`'s own docstring).
                psi_hat, rho = joint_psi_hat(psi0_all, offset, jr.bin_label, jr.bin_U,
                                              jr.bin_mass, jr.V_btw)
                second_stage_fields['psi_hat'] = psi_hat
                second_stage_fields['rho'] = rho
            joint_fields = dict(
                joint_S_pred=jr.S_pred, joint_a=jr.a,
                joint_S_pred_pre_lloyd=jr.S_pred_pre_lloyd,
                joint_L0=jr.L0, joint_L=jr.L, joint_n_growth_rounds=jr.n_growth_rounds,
                joint_growth_capped=jr.growth_capped, joint_n_flagged=jr.n_flagged,
                joint_n_check_rounds=jr.n_check_rounds, joint_n_check_evals=jr.n_check_evals,
                joint_n_level_splits=jr.n_level_splits, joint_n_adjacency_splits=jr.n_adjacency_splits,
                joint_check_capped=jr.check_capped, joint_failed=jr.failed,
                joint_bin_mass=jr.bin_mass, joint_bin_U=jr.bin_U, joint_bin_m=jr.bin_m,
                joint_bin_flagged=jr.bin_flagged, joint_bin_label=jr.bin_label,
                joint_busy_delta=jr.busy_delta,
            )
        else:
            # One coordinate per MEASURED output only (its own cost is an
            # evaluation loop, unlike the shared full-data bin measurement
            # below). `c` (absolute) indexes `theta_hat`/`I_proto`, which
            # stay full width; `j` (local) indexes `psi0_all`/`sigma_all`/
            # `model`, which are already the measured subset -- passed as
            # `model_index` so `bin_posterior_variance` reads `model`'s
            # own coordinate, not the absolute one (spec/QIJ_mods_waves.md
            # A11). Identity `measured` reproduces `enumerate(outputs)` bit
            # for bit (c == j == its own name's position). `refine_schedule`
            # (A14) picks how the q measured outputs' evaluations are
            # scheduled: `'queue'` runs each output's own queue in turn,
            # serial evaluations only; `'rounds'` runs them together,
            # batching every round's evaluations across outputs onto `pool`.
            # Under `pilot='affine'` the bridge pair geometry (`pair_id`/
            # `pair_n`) is shared across every output; only each output's
            # own m_jk*Delta_jk^2 (`bridge_value_c`, spec/QIJ_affine_pilot_
            # spec.md 3.2) differs, at output c's ABSOLUTE index -- the
            # pilot's own tables are full width (section 6).
            if self.pilot == 'affine':
                bridge_pair_id = pilot_result.pair_id
                bridge_pair_n = pilot_result.pair_n
                bridge_value_cols = [
                    pilot_result.pair_mass * pilot_result.pair_delta[:, c] ** 2 for c in measured
                ]
            else:
                bridge_pair_id = bridge_pair_n = bridge_value_cols = None
            if self.refine_schedule == 'rounds':
                coordinates, n_rounds = run_refinement_rounds(
                    X, counter, theta_hat, list(measured), list(outputs),
                    [psi0_all[:, j] for j in range(q)], [float(offset[j]) for j in range(q)],
                    [sigma_all[:, j] for j in range(q)], [I_proto[:, c] for c in measured],
                    xvq.bmu, xvq.bmu2, eta_full, self.eps, xvq.M_used,
                    [bool(constant_path_arr[j]) for j in range(q)], Z, model, list(range(q)),
                    pool, start=start_second_stage,
                    pilot=self.pilot, bridge_pair_id=bridge_pair_id, bridge_pair_n=bridge_pair_n,
                    bridge_value_cols=bridge_value_cols,
                )
            else:
                coordinates = [
                    run_refinement(
                        X, counter, theta_hat, c, name,
                        psi0_all[:, j], float(offset[j]), sigma_all[:, j],
                        I_proto[:, c], xvq.bmu, xvq.bmu2,
                        eta_full, self.eps, xvq.M_used, bool(constant_path_arr[j]), Z, model, pool,
                        start=start_second_stage, model_index=j,
                        pilot=self.pilot, bridge_pair_id=bridge_pair_id, bridge_pair_n=bridge_pair_n,
                        bridge_value_c=(bridge_value_cols[j] if bridge_value_cols is not None else None),
                    )
                    for j, (c, name) in enumerate(zip(measured, outputs))
                ]
            # A failed initial-bin evaluation NaNs the whole draw, not just
            # its own coordinate: every V_btw/V_win_hat/V_tot_hat (and the
            # ABC ingredients derived from the same bins) is voided.
            # `B_hat`/`a_bca` stay (q_full,) arrays: they are indexed per
            # output below.
            if any(cr.failed for cr in coordinates):
                nan_q_full = np.full(q_full, np.nan)
                coordinates = [
                    replace(cr, V_btw=float('nan'), V_win_hat=float('nan'), V_tot_hat=float('nan'),
                            B_hat=nan_q_full, a_bca=nan_q_full)
                    for cr in coordinates
                ]
            second_stage_evals = sum(cr.n_refine_evals for cr in coordinates)
            second_stage_busy = sum(cr.busy_delta for cr in coordinates)
            second_stage_fields = dict(
                V_btw=np.array([cr.V_btw for cr in coordinates]),
                V_win_hat=np.array([cr.V_win_hat for cr in coordinates]),
                V_tot_hat=np.array([cr.V_tot_hat for cr in coordinates]),
                L=np.array([cr.L for cr in coordinates]),
                n_level_splits=np.array([cr.n_level_splits for cr in coordinates]),
                n_adjacency_splits=np.array([cr.n_adjacency_splits for cr in coordinates]),
                rho=np.array([cr.rho for cr in coordinates]),
                gain_ratio=np.array([cr.gain_ratio for cr in coordinates]),
                n_refine_evals=np.array([cr.n_refine_evals for cr in coordinates]),
                psi_hat=np.stack([cr.field for cr in coordinates], axis=1),
                bin_label=np.stack([cr.labels for cr in coordinates], axis=1),
                # `a_bca`/`B_hat` are each coordinate's OWN (q_full,) bin
                # measurement (every output, from the one shared
                # evaluation); take this coordinate's own entry, at its
                # absolute index, not position `local_c` in `coordinates`.
                a=np.array([cr.a_bca[c] for cr, c in zip(coordinates, measured)]),
                b_hat=np.array([cr.B_hat[c] for cr, c in zip(coordinates, measured)]),
                # Every final bin's derivative on every OTHER measured
                # output too (spec/QIJ_mods_waves.md A14's bin_U product),
                # sliced from `cr.bin_U`'s full T-output width down to the
                # measured columns; one (L_c, q) array per output, L_c its
                # own final bin count.
                bin_U=tuple(cr.bin_U[:, measured] for cr in coordinates),
            )
            joint_fields = _joint_defaults(N, q)
        wall_time_refinement = time.perf_counter() - t0
        ev3, rows3 = counter.snapshot()

        # The sigma-points stage (spec/QIJ_sigma_points_spec.md 2): after
        # refinement and the curvature stage, on the full data, from the
        # method's own refined influence estimate (`psi_hat`, already at
        # the measured width regardless of `ivqbins`). A no-op, at zero
        # cost, under `sigma_points=False` (module docstring); `pool`
        # already shares `X` (re-shared above stage 2, never replaced by
        # stage 2 itself).
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

        if self.pilot == 'gp':
            # at_bound[:, 0] is whichever width parameter gpwidth fits
            # (method_notes section 3); `model` is already the measured
            # subset (spec/QIJ_mods_waves.md A11), so every model-derived
            # field below is used as-is, at `model`'s own (measured)
            # width -- only the T-output arrays (`theta_hat`, `I_proto`,
            # `c_q`, `survey_step_ratio`) still need `[measured]`.
            local = self.gpwidth == 'local'
            ell_bound = np.zeros(q, dtype=bool) if local else model.at_bound[:, 0].copy()
            c_bound = model.at_bound[:, 0].copy() if local else np.zeros(q, dtype=bool)
            ell, lam = np.array(model.width), np.array(model.lam)
            lam_bound, c_arr = model.at_bound[:, 1].copy(), np.array(model.c)
            prototype_h = np.array(model.h)
            pilot_products = _empty_pilot_products(q)
        else:
            # No width/lam/c of any kind under 'affine' (spec/QIJ_affine_
            # pilot_spec.md 4), `prototype_h` NaN as under a failed GP fit
            # today (module docstring); `bridge`/`cells` sliced to
            # `measured` from the pilot's own full-width tables.
            nan_q, false_q = np.full(q, np.nan), np.zeros(q, dtype=bool)
            ell = lam = c_arr = nan_q
            ell_bound = lam_bound = c_bound = false_q
            prototype_h = np.full(xvq.M_used, np.nan)
            pilot_products = dict(
                bridge_j=pilot_result.pair_j, bridge_k=pilot_result.pair_k,
                bridge_m=pilot_result.pair_mass,
                bridge_delta=pilot_result.pair_delta[:, measured],
                cell_p=xvq.p, cell_mu=pilot_result.cell_mu,
                cell_g=pilot_result.cell_g[:, :, measured],
            )

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
            ivqbins=self.ivqbins, survey=self.survey,
            c_q=c_q, c_q_one_sided=c_q_one_sided, eta_Q=sv.eta_Q, eta_full=float(eta_full),
            survey_step_ratio=np.asarray(sv.step_ratio, dtype=float)[:, measured],
            quantized_start=self.quantized_start,
            refine_schedule=self.refine_schedule, n_rounds=n_rounds,
            pilot=self.pilot,
            **second_stage_fields, **joint_fields, **pilot_products, **sigma_fields,
        )
