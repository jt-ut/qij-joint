"""`QIJ`: the influence-aligned variance method (spec/method_notes.md
section 4).

`QIJ(eps=0.01, seed=0, vq_transform=None, gptrend='affine',
gpwidth='global', M_X=None, ivqbins='marginal').fit(X, T, pool=None)`
runs stage 1 (the X-VQ, prototype influences, initial influence
estimate) and stage 2 (the shared full-data evaluation, then
`ivqbins`'s `'marginal'` per-output refinement or `'joint'` shared
partition, method_notes joint section), returning a `QIJResult`.
`gptrend`/`gpwidth` pass straight to `fit_influence_model` (method_notes
section 3); `M_X` overrides the prototype-count rule when given
(method_notes section 2). With a `pool`: the survey and stage 2's
full-data stencils run on it, the full-data base evaluation is
submitted at the start and collected after stage 1, the influence
model's width-grid candidates run on it (method_notes section 3), and
the codebook fit uses `pool.workers` FAISS threads (method_notes
sections 2 and 4); a marginal split's own evaluation stays serial. A
plain callable T is wrapped with outputs=('theta',) and machine-
precision eta; the method sees T only through `Counter`, never an
analytic influence.
"""
from __future__ import annotations

import time
from dataclasses import replace

import numpy as np
from numpy.linalg import LinAlgError

from .core.counter import Counter
from .core.influence_model import fit_influence_model
from .core.influence_model import psi0 as _psi0
from .core.influence_model import uncertainty as _uncertainty
from .core.joint import run_joint
from .core.refine import run_refinement
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
    """The marginal-only `QIJResult` fields, inert under `ivqbins='joint'`."""
    return dict(
        L=np.zeros(q, dtype=int), n_level_splits=np.zeros(q, dtype=int),
        n_adjacency_splits=np.zeros(q, dtype=int), rho=np.full(q, np.nan),
        n_refine_evals=np.zeros(q, dtype=int),
        psi_hat=np.full((N, q), np.nan), bin_label=np.full((N, q), -1, dtype=int),
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


def _failed_draw(outputs, N, xvq, W_X, I_proto, counter, t_start,
                  gptrend, gpwidth, M_X_source, workers, ivqbins) -> QIJResult:
    """The result of a draw whose influence model could not be fitted:
    every variance and model quantity NaN, every count after stage 1
    zero, the X-VQ and prototype survey diagnostics kept. The second
    stage never ran under either value of `ivqbins`, so both the
    marginal and the joint fields are inert."""
    q = len(outputs)
    nan_q, false_q = np.full(q, np.nan), np.zeros(q, dtype=bool)
    nan_Nq = np.full((N, q), np.nan)
    ev, rows = counter.snapshot()
    wall = time.perf_counter() - t_start
    return QIJResult(
        outputs=outputs, N=N, theta_hat=nan_q,
        V_btw=nan_q, V_win_hat=nan_q, V_tot_hat=nan_q, gain_ratio=nan_q,
        ell=nan_q, lam=nan_q, ell_bound=false_q, lam_bound=false_q,
        gptrend=gptrend, gpwidth=gpwidth, c=nan_q, c_bound=false_q,
        M_X=xvq.M_used, M_X_source=M_X_source, n_failed=counter.failed,
        evals_by_stage={'prototype': ev, 'full_data': 0, 'refinement': 0, 'total': ev},
        rows_by_stage={'prototype': rows, 'full_data': 0, 'refinement': 0, 'total': rows},
        wall_time_by_stage={'prototype': wall, 'full_data': 0.0, 'refinement': 0.0, 'total': wall},
        busy_time_total=wall, workers=workers,
        psi0=nan_Nq, sigma=nan_Nq,
        bmu=xvq.bmu, prototype_p=xvq.p,
        prototype_w=W_X, prototype_I=np.asarray(I_proto, dtype=float),
        prototype_h=np.full(xvq.M_used, np.nan), ivqbins=ivqbins,
        **_marginal_defaults(N, q), **_joint_defaults(N, q),
    )


class QIJ:
    """The QIJ method (spec/method_notes.md section 4)."""

    def __init__(self, eps: float = 0.01, seed: int = 0, vq_transform=None,
                 gptrend: str = 'affine', gpwidth: str = 'global', M_X: int = None,
                 ivqbins: str = 'marginal') -> None:
        self.eps = eps
        self.seed = seed
        self.vq_transform = vq_transform
        self.gptrend = gptrend
        self.gpwidth = gpwidth
        self.M_X = M_X
        self.ivqbins = ivqbins

    def fit(self, X: np.ndarray, T, pool=None) -> QIJResult:
        """Run the method on one draw: stage 1 (X-VQ, prototype
        influences), the shared full-data base evaluation, then
        per-output refinement, on `pool` per the module docstring."""
        t_start = time.perf_counter()
        X = np.asarray(X)
        N = len(X)
        T = _wrap(T)
        eta = T.eta
        outputs = T.outputs
        q = len(outputs)
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

        t0 = time.perf_counter()
        xvq, theta_Q, I_proto, xvq_busy = run_xvq(
            Z, inverse, counter, eta, M_requested, self.seed, pool, workers)
        # Prototype positions in T's native coordinates, for the
        # result's diagnostics -- the same `inverse(xvq.centers)`
        # `run_xvq` already applied, recovered rather than re-derived.
        W_X = np.asarray(inverse(xvq.centers), dtype=float)
        try:
            model, model_busy = fit_influence_model(
                Z, xvq, I_proto, theta_Q, eta, gptrend=self.gptrend, gpwidth=self.gpwidth,
                pool=pool)
        except (RuntimeError, LinAlgError):
            # A failed stage 1 is recorded as failed, never retried; a
            # submitted `theta_future` is left uncollected and uncounted,
            # as the serial code never reaches its own call here either.
            return _failed_draw(outputs, N, xvq, W_X, I_proto, counter, t_start,
                                 self.gptrend, self.gpwidth, M_X_source, workers, self.ivqbins)
        psi0_all = _psi0(model, Z)
        sigma_all = _uncertainty(model, Z)
        wall_time_prototype = time.perf_counter() - t0
        ev1, rows1 = counter.snapshot()

        t0 = time.perf_counter()
        if theta_future is None:
            theta_hat = np.asarray(counter(X, np.ones(N)), dtype=float)
            full_data_busy = 0.0
        else:
            result, failed, full_data_busy = theta_future.result()
            theta_hat = np.asarray(result, dtype=float)
            counter.add(1, N, int(failed))
        wall_time_full_data = time.perf_counter() - t0

        if pool is not None:
            pool.share(X)  # re-share X: the survey shared W_X on `pool`

        t0 = time.perf_counter()
        if self.ivqbins == 'joint':
            jr = run_joint(X, counter, theta_hat, psi0_all, sigma_all, model, Z, xvq, eta, self.eps,
                           pool=pool, I_proto=I_proto)
            # A failed output, or a failed initial bin measurement before
            # any check ran, voids every output's variance quantities, as
            # the marginal path voids them on any one coordinate's failure.
            if jr.failed:
                jr = replace(jr, V_btw=np.full(q, np.nan), V_win_hat=np.full(q, np.nan),
                              V_tot_hat=np.full(q, np.nan))
            second_stage_evals = int(jr.n_check_evals)
            second_stage_busy = float(jr.busy_delta)
            second_stage_fields = dict(V_btw=jr.V_btw, V_win_hat=jr.V_win_hat, V_tot_hat=jr.V_tot_hat,
                                        gain_ratio=jr.gain_ratio, **_marginal_defaults(N, q))
            joint_fields = dict(
                joint_S_pred=jr.S_pred, joint_a=jr.a, joint_S_pred_pre_lloyd=jr.S_pred_pre_lloyd,
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
            coordinates = [
                run_refinement(
                    X, counter, theta_hat, c, name,
                    psi0_all[:, c], float(model.offset[c]), sigma_all[:, c],
                    I_proto[:, c], xvq.bmu, xvq.bmu2,
                    eta, self.eps, xvq.M_used, bool(model.constant_path[c]), Z, model, pool,
                )
                for c, name in enumerate(outputs)
            ]
            # A failed initial-bin evaluation NaNs the whole draw, not just
            # its own coordinate: every V_btw/V_win_hat/V_tot_hat is voided.
            if any(cr.failed for cr in coordinates):
                coordinates = [
                    replace(cr, V_btw=float('nan'), V_win_hat=float('nan'), V_tot_hat=float('nan'))
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
            )
            joint_fields = _joint_defaults(N, q)
        wall_time_refinement = time.perf_counter() - t0
        ev3, rows3 = counter.snapshot()

        evals_by_stage = {'prototype': ev1, 'full_data': (ev3 - ev1) - second_stage_evals,
                           'refinement': second_stage_evals, 'total': ev3}
        rows_by_stage = {'prototype': rows1, 'full_data': (rows3 - rows1) - second_stage_evals * N,
                          'refinement': second_stage_evals * N, 'total': rows3}
        wall_time_total = time.perf_counter() - t_start
        wall_time_by_stage = {'prototype': wall_time_prototype, 'full_data': wall_time_full_data,
                               'refinement': wall_time_refinement, 'total': wall_time_total}
        # Parent work plus every pool task's own time, in place of each
        # parallel stage's share of the elapsed total.
        busy_time_total = wall_time_total + xvq_busy + model_busy + full_data_busy + second_stage_busy

        # at_bound[:, 0] is whichever width parameter gpwidth fits
        # (method_notes section 3).
        local = self.gpwidth == 'local'
        ell_bound = np.zeros(q, dtype=bool) if local else model.at_bound[:, 0].copy()
        c_bound = model.at_bound[:, 0].copy() if local else np.zeros(q, dtype=bool)

        return QIJResult(
            outputs=outputs, N=N, theta_hat=theta_hat,
            ell=np.array(model.width), lam=np.array(model.lam),
            ell_bound=ell_bound, lam_bound=model.at_bound[:, 1].copy(),
            gptrend=self.gptrend, gpwidth=self.gpwidth,
            c=np.array(model.c), c_bound=c_bound,
            M_X=xvq.M_used, M_X_source=M_X_source, n_failed=counter.failed,
            evals_by_stage=evals_by_stage, rows_by_stage=rows_by_stage,
            wall_time_by_stage=wall_time_by_stage,
            busy_time_total=busy_time_total, workers=workers,
            psi0=psi0_all, sigma=sigma_all,
            bmu=xvq.bmu, prototype_p=xvq.p, prototype_w=W_X, prototype_h=np.array(model.h),
            prototype_I=np.asarray(I_proto, dtype=float),
            ivqbins=self.ivqbins, **second_stage_fields, **joint_fields,
        )
