"""The five methods' per-draw loops (Sections 2 and 4; spec A13 for
`ijfd`). `oracle` and `ij` run their draws across a process pool; `boot`,
`qij` and `ijfd` loop over draws in sequence, each parallel within a
draw -- `boot` over replicate chunks, `qij` over its prototype survey
(method_notes section 2), `ijfd` over its N (or 2N under
`point_curvature`) point-perturbed fits.
"""
from __future__ import annotations

import time
from typing import Iterable, Optional, Tuple

import numpy as np
import pandas as pd

from . import products, registry
from .bootstrap import Bootstrap
from .ijfd import IJFD
from .audit import search_audit
from .parallel import Pool, fit_status
from .qij import QIJ
from .qijdt import QIJDT
from .qijt import QIJT


def _oracle_task(T, case, X, task):
    """One draw's full-data estimate: `theta_hat = T(X, ones)`; `X` is
    drawn fresh in this worker since draws are shared only by (N, seed),
    not by value, when the pool runs across draws. `failed` is also set
    on a NaN return that raised no exception (a gate failing quietly),
    not only on an exception."""
    s, N, base_seed = task
    seed = base_seed + s
    t0 = time.perf_counter()
    Xd = case.draw(N, seed)
    try:
        theta_hat = np.asarray(T(Xd, np.ones(N)), dtype=float)
        failed = bool(np.any(np.isnan(theta_hat)))
        status = fit_status(T, theta_hat)
    except Exception:
        theta_hat = np.full(len(T.outputs), np.nan)
        failed = True
        status = fit_status(T, theta_hat, raised=True)
    wall = time.perf_counter() - t0
    info = getattr(T, 'last_fit_info', None)
    search = dict(getattr(info, 'search', None) or {})
    audit = search_audit(T, Xd, theta_hat, case.dataset, case.estimator)
    return s, seed, theta_hat, failed, status, wall, {**search, **audit}


def _ij_task(T, case, X, task):
    """One draw's analytic variance: `psi = T.influence(X, ones)`,
    `V_ij = mean(psi**2, axis=0) / N`, the infinitesimal-jackknife
    variance of theta_hat. `failed` is also set on a NaN return that
    raised no exception (a gate failing quietly), not only on an
    exception."""
    s, N, base_seed = task
    seed = base_seed + s
    t0 = time.perf_counter()
    Xd = case.draw(N, seed)
    try:
        psi = np.asarray(T.influence(Xd, np.ones(N)), dtype=float)
        failed = bool(np.any(np.isnan(psi)))
        status = fit_status(T, psi)
    except Exception:
        psi = np.full((N, len(T.outputs)), np.nan)
        failed = True
        status = fit_status(T, psi, raised=True)
    V_ij = np.mean(psi ** 2, axis=0) / N
    return s, seed, V_ij, psi, failed, status, time.perf_counter() - t0


def run_oracle(dataset: str, estimator: str, N: int, draws: Iterable[int], seed: int,
               out_dir: str, workers: int, force: bool, tag: str = '') -> Tuple[int, int]:
    """`oracle`: theta_true (the registry) and theta_hat = T(X, ones),
    parallel across draws."""
    draws = list(draws)
    md = products.method_dir(out_dir, dataset, estimator, N, 'oracle', tag)
    theta_true = registry.truth(dataset, estimator)
    outputs = registry.case(dataset, estimator).make_T().outputs
    pending = [s for s in draws if force or not products.is_done(md, s)]
    if pending:
        pool = Pool(workers, case=(dataset, estimator))
        tasks = [(s, N, seed) for s in pending]
        for s, dseed, theta_hat, failed, status, wall, audit in pool.map(_oracle_task, tasks):
            row = {'dataset': dataset, 'estimator': estimator, 'N': N,
                   's': s, 'seed': dseed, 'n_failed': int(failed), 'fit_status': status,
                   'wall_time': wall, 'busy_time': wall, 'workers': workers, **audit}
            for j, o in enumerate(outputs):
                row[f'theta_true_{o}'] = float(theta_true[j])
                row[f'theta_hat_{o}'] = float(theta_hat[j])
            products.write_draw(md, s, row)
        pool.close()
    return len(pending), len(draws) - len(pending)


def run_ij(dataset: str, estimator: str, N: int, draws: Iterable[int], seed: int,
           out_dir: str, workers: int, force: bool, tag: str = '') -> Tuple[int, int]:
    """`ij`: analytic influence and V_ij, parallel across draws; the
    `psi` array is stored for every draw written; `ijfd` stores its own
    under the same kind, mass-centred, so a comparison against it centres
    this one first."""
    draws = list(draws)
    md = products.method_dir(out_dir, dataset, estimator, N, 'ij', tag)
    outputs = registry.case(dataset, estimator).make_T().outputs
    pending = [s for s in draws if force or not products.is_done(md, s)]
    if pending:
        pool = Pool(workers, case=(dataset, estimator))
        tasks = [(s, N, seed) for s in pending]
        for s, dseed, V_ij, psi, failed, status, wall in pool.map(_ij_task, tasks):
            row = {'dataset': dataset, 'estimator': estimator, 'N': N,
                   's': s, 'seed': dseed, 'n_failed': int(failed), 'fit_status': status,
                   'wall_time': wall, 'busy_time': wall, 'workers': workers}
            for j, o in enumerate(outputs):
                row[f'V_ij_{o}'] = float(V_ij[j])
            psi_df = pd.DataFrame({f'psi_{o}': psi[:, j] for j, o in enumerate(outputs)})
            products.write_draw(md, s, row, {'psi': psi_df})
        pool.close()
    return len(pending), len(draws) - len(pending)


def run_boot(dataset: str, estimator: str, N: int, draws: Iterable[int], seed: int,
             out_dir: str, workers: int, B: int, force: bool, tag: str = '') -> Tuple[int, int]:
    """`boot`: a sequential draw loop, each draw's B replicates spread
    across one persistent pool for the whole run."""
    draws = list(draws)
    md = products.method_dir(out_dir, dataset, estimator, N, 'boot', tag)
    case = registry.case(dataset, estimator)
    T = case.make_T()
    pool = Pool(workers, T=T)
    written = 0
    for s in draws:
        if not force and products.is_done(md, s):
            continue
        dseed = seed + s
        X = case.draw(N, dseed)
        res = Bootstrap(B=B, seed=dseed).fit(X, T, pool=pool)
        row = {'dataset': dataset, 'estimator': estimator, 'N': N,
               's': s, 'seed': dseed, 'n_degenerate': res.n_degenerate,
               'wall_time': res.wall_time, 'busy_time': res.busy_time,
               'workers': res.workers, 'eta_full': float(res.eta_full),
               'fit_status': res.theta_hat_status,
               **search_audit(T, X, res.theta_hat, dataset, estimator)}
        rep_df = pd.DataFrame({f'theta_{o}': res.replicates[:, j]
                                for j, o in enumerate(res.outputs)})
        rep_df['fit_status'] = res.replicate_status
        products.write_draw(md, s, row, {'replicates': rep_df})
        written += 1
    pool.close()
    return written, len(draws) - written


def _qij_row(dataset: str, estimator: str, N: int, s: int, seed: int, res) -> dict:
    """One `qij` scalar row: `QIJResult`'s scalars flattened to columns
    per stage and per output, plus `survey`, `quantized_start`,
    `eta_full` (A15), the ABC interval's ingredients (`a`, `b_hat`,
    `c_q`, `c_q_one_sided`, `eta_Q`, spec/QIJ_mods_waves.md A10) and the
    unified loop's own scalars (spec/QIJ_unified_loop_spec.md 4.4, 4.5):
    `L`, `n_splits`, `n_rounds`, `n_evals` (two per split ATTEMPT,
    successful or not), `capped`, `stop_met`, and
    the loop's own user-exposed levers `z`, `n_min`, `L_max` (`L_max`
    always resolved, never None). Per output: `V_win_hat_<o>` (as
    before), plus the calibrated stop's own `se_V_win_<o>`, `kappa_<o>`,
    `se_kappa_<o>`, `margin_<o>`. `fit_weights`
    (spec/QIJ_mass_weighted_fit_spec.md) is recorded; `rank_rule` (A20
    amendment, spec/QIJ_mods_waves.md) is always 'measured_error';
    `n_update_scaled_<o>`/`n_update_shifted_<o>`/`n_update_negative_<o>`
    (A20) are its own per-output products. The unified loop (spec/
    QIJ_unified_loop_spec.md) removed `S_pred_<o>`, `joint_a_<o>` (A18's
    separate scale step, folded into the splits' own update), `L0`,
    `n_growth_rounds`, `growth_capped`, `n_check_rounds`,
    `n_check_evals`, `check_capped`, `share_final_<o>` and
    `pilot_err_btw_<o>` as columns: the growth/check second stage that
    produced them no longer exists. A21 (spec/QIJ_mods_waves.md A21)
    earlier removed `ivqbins`/`pilot`/`check_rule`/`tree_rule`/
    `refine_schedule`/`n_rounds` (the pre-A20 switches) as both
    switches and columns, along with every column that only ever
    reported the marginal path, the affine pilot or the per-bin check's
    own flag/pay bookkeeping (`gain_ratio_<o>`, `n_flagged`, `n_closed_
    unpaid`, `n_closed_unflagged`, `n_noise_floored`, `sum_b_delta_<o>`,
    the per-output `L_<o>`/`n_level_splits_<o>`/`n_adjacency_splits_<o>`/
    `n_refine_evals_<o>`/`rho_<o>`, and the `bridge`/`cells` array
    products): the code that produced them no longer exists."""
    row = {'dataset': dataset, 'estimator': estimator, 'N': N,
           's': s, 'seed': seed,
           'fit_weights': res.fit_weights,
           'rank_rule': res.joint_rank_rule,
           'gptrend': res.gptrend, 'gpwidth': res.gpwidth,
           'M_X': int(res.M_X), 'M_X_source': res.M_X_source, 'n_failed': int(res.n_failed),
           'survey': res.survey,
           'quantized_start': res.quantized_start, 'eta_Q': float(res.eta_Q),
           'eta_full': float(res.eta_full)}
    for stage in ('prototype', 'full_data', 'refinement', 'curvature', 'eta_full', 'total'):
        row[f'evals_{stage}'] = int(res.evals_by_stage[stage])
        row[f'rows_{stage}'] = int(res.rows_by_stage[stage])
        row[f'wall_time_{stage}'] = float(res.wall_time_by_stage[stage])
    row['normalized_rows'] = float(res.rows_by_stage['total']) / res.N
    row['busy_time_total'] = float(res.busy_time_total)
    row['workers'] = int(res.workers)
    row['L'] = int(res.joint_L)
    row['n_splits'] = int(res.joint_n_splits)
    row['n_rounds'] = int(res.joint_n_rounds)
    row['n_evals'] = int(res.joint_n_evals)
    row['capped'] = bool(res.joint_capped)
    row['stop_met'] = bool(res.joint_stop_met)
    row['z'] = float(res.z)
    row['n_min'] = int(res.n_min)
    row['L_max'] = int(res.L_max)
    # The sigma-points stage (spec/QIJ_sigma_points_spec.md 3): its own
    # 'sigma' stage entry (outside the loop above, whose own product
    # names -- 'wall_sigma', not 'wall_time_sigma' -- differ from the
    # other stages'), all NaN/0/None under `sigma_points=False`.
    row['sigma_points'] = bool(res.sigma_points)
    row['sigma_status'] = res.sigma_status
    row['n_fp_sigma'] = int(res.n_fp_sigma)
    row['r_fp_sigma'] = float(res.r_fp_sigma)
    row['evals_sigma'] = int(res.evals_by_stage['sigma'])
    row['rows_sigma'] = int(res.rows_by_stage['sigma'])
    row['wall_sigma'] = float(res.wall_time_by_stage['sigma'])
    row['n_failed_sigma'] = int(res.n_failed_sigma)
    row['n_dirs_sigma'] = int(res.n_dirs_sigma)
    row['max_abs_d_sigma'] = float(res.max_abs_d_sigma)
    lo_sigma, hi_sigma = res.sigma_interval(0.95).T
    for j, o in enumerate(res.outputs):
        row[f'V_btw_{o}'] = float(res.V_btw[j])
        row[f'V_win_hat_{o}'] = float(res.V_win_hat[j])
        row[f'se_V_win_{o}'] = float(res.se_V_win[j])
        row[f'V_tot_hat_{o}'] = float(res.V_tot_hat[j])
        row[f'kappa_{o}'] = float(res.kappa[j])
        row[f'se_kappa_{o}'] = float(res.se_kappa[j])
        row[f'margin_{o}'] = float(res.margin[j])
        row[f'ell_{o}'] = float(res.ell[j])
        row[f'lam_{o}'] = float(res.lam[j])
        row[f'ell_bound_{o}'] = bool(res.ell_bound[j])
        row[f'lam_bound_{o}'] = bool(res.lam_bound[j])
        row[f'c_{o}'] = float(res.c[j])
        row[f'c_bound_{o}'] = bool(res.c_bound[j])
        row[f'a_{o}'] = float(res.a[j])
        row[f'b_hat_{o}'] = float(res.b_hat[j])
        row[f'c_q_{o}'] = float(res.c_q[j])
        row[f'c_q_one_sided_{o}'] = bool(res.c_q_one_sided[j])
        row[f'sigma_mean_{o}'] = float(res.sigma_mean[j])
        row[f'sigma_sd_{o}'] = float(res.sigma_sd[j])
        row[f'sigma_bias_{o}'] = float(res.sigma_bias[j])
        row[f'lo_sigma_{o}'] = float(lo_sigma[j])
        row[f'hi_sigma_{o}'] = float(hi_sigma[j])
        # A20 (spec/QIJ_mods_waves.md), the unified loop's own update
        # rule, kept verbatim.
        row[f'n_update_scaled_{o}'] = int(res.joint_n_update_scale[j])
        row[f'n_update_shifted_{o}'] = int(res.joint_n_update_shift[j])
        row[f'n_update_negative_{o}'] = int(res.joint_n_update_negative[j])
    return row


def _qij_step_ratio(res) -> pd.DataFrame:
    """`step_ratio` for every `qij` draw (not gated behind `--diag-draws`):
    the survey's step-doubling self-check on its five largest-mass
    prototypes, per output (spec/QIJ_mods_waves.md A9 item 4); all-NaN
    rows for an estimator without `takes_start`."""
    step_ratio = np.asarray(res.survey_step_ratio, dtype=float)
    data = {'i': np.arange(step_ratio.shape[0])}
    for j, o in enumerate(res.outputs):
        data[f'step_ratio_{o}'] = step_ratio[:, j]
    return pd.DataFrame(data)


def _qij_points(res) -> pd.DataFrame:
    """`points` for `--diag-draws`: `i, bmu, psi0_<o>, sigma_<o>,
    psi_hat_<o>, bin_label` -- a single shared `bin_label` (not one per
    output), since every output shares one partition
    (spec/method_notes.md section 6; A21, spec/QIJ_mods_waves.md,
    removed the per-output marginal path this used to also serve)."""
    N = res.psi0.shape[0]
    data = {'i': np.arange(N), 'bmu': np.asarray(res.bmu, dtype=np.int32),
            'bin_label': np.asarray(res.joint_bin_label, dtype=np.int32)}
    for j, o in enumerate(res.outputs):
        data[f'psi0_{o}'] = res.psi0[:, j]
        data[f'sigma_{o}'] = res.sigma[:, j]
        data[f'psi_hat_{o}'] = res.psi_hat[:, j]
    return pd.DataFrame(data)


def _qij_bins(res) -> pd.DataFrame:
    """`bins` for every draw (spec/QIJ_unified_loop_spec.md 4.5, "the
    per-leaf table (n, U, e-bar, W)"): `k, bin_mass, n, U_<o>, m_<o>,
    ebar_<o>, W_<o>`, the unified loop's own final leaf constituents.
    `m_<o>` is the leaf's own m_pre (THE ARCHITECTURE RULE, spec/
    QIJ_mods_waves.md A20/A21), not a fixed copy of the pilot; `ebar_<o>`
    is the A20 ranking amendment's own per-leaf diagnostic, each final
    leaf's own measured error at creation; `n`/`W_<o>` are the leaf's
    own point count and within contribution (spec 3), new with the
    unified loop. A21 removed the per-bin check's own flag concept
    (`bin_flagged`), which no longer applies to any bin."""
    L = res.joint_bin_mass.shape[0]
    data = {'k': np.arange(L), 'bin_mass': res.joint_bin_mass, 'n': res.joint_bin_n}
    for j, o in enumerate(res.outputs):
        data[f'U_{o}'] = res.joint_bin_U[:, j]
        data[f'm_{o}'] = res.joint_bin_m[:, j]
        data[f'ebar_{o}'] = res.joint_bin_ebar[:, j]
        data[f'W_{o}'] = res.joint_bin_W[:, j]
    return pd.DataFrame(data)


def _qij_splits(res) -> pd.DataFrame:
    """`splits` for every draw (spec/QIJ_unified_loop_spec.md 4.5, "the
    per-split table (parent, children, G, D)"): `id, parent, child_a,
    child_b, round, G_<o>, D_<o>` -- `G` the predicted gain, `D` the
    realized gain, both read before the children's own update (spec
    3, 4.3); `child_a` is the smaller (measured) child, `child_b` the
    larger (conservation) child."""
    S = res.joint_split_parent.shape[0]
    data = {'id': np.arange(S), 'parent': res.joint_split_parent,
            'child_a': res.joint_split_child_a, 'child_b': res.joint_split_child_b,
            'round': res.joint_split_round}
    for j, o in enumerate(res.outputs):
        data[f'G_{o}'] = res.joint_split_G[:, j]
        data[f'D_{o}'] = res.joint_split_D[:, j]
    return pd.DataFrame(data)


def _qij_rounds(res) -> pd.DataFrame:
    """`rounds` for every draw (spec/QIJ_unified_loop_spec.md 4.4): the
    calibrated stop's own trajectory, one row per completed round --
    `round, n_splits, kappa_<o>, se_kappa_<o>, V_win_<o>, margin_<o>`."""
    R = res.joint_round_n_splits.shape[0]
    data = {'round': np.arange(R), 'n_splits': res.joint_round_n_splits}
    for j, o in enumerate(res.outputs):
        data[f'kappa_{o}'] = res.joint_round_kappa[:, j]
        data[f'se_kappa_{o}'] = res.joint_round_se_kappa[:, j]
        data[f'V_win_{o}'] = res.joint_round_V_win[:, j]
        data[f'margin_{o}'] = res.joint_round_margin[:, j]
    return pd.DataFrame(data)


def _qij_prototypes(res) -> pd.DataFrame:
    """`prototypes` for `--diag-draws`: `j, p, w_<d>, I_<o>, h`
    (method_notes section 3)."""
    M = res.prototype_p.shape[0]
    W = np.atleast_2d(np.asarray(res.prototype_w, dtype=float).reshape(M, -1))
    data = {'j': np.arange(M), 'p': res.prototype_p}
    for d in range(W.shape[1]):
        data[f'w_{d}'] = W[:, d]
    for j, o in enumerate(res.outputs):
        data[f'I_{o}'] = res.prototype_I[:, j]
    data['h'] = res.prototype_h
    return pd.DataFrame(data)


def _qij_sigma_points(res) -> pd.DataFrame:
    """`sigma_points` for every draw under `sigma_points=True`
    (spec/QIJ_sigma_points_spec.md 3): one row per evaluation, `k`
    (direction index), `sign`, `response_<o>` per measured output, raw
    (R^(k+-), section 2.4-2.5)."""
    data = {'k': res.sigma_k, 'sign': res.sigma_sign}
    for j, o in enumerate(res.outputs):
        data[f'response_{o}'] = res.sigma_response[:, j]
    return pd.DataFrame(data)


def run_qij(dataset: str, estimator: str, N: int, draws: Iterable[int], seed: int,
            out_dir: str, eps: float, diag_draws: Optional[Iterable[int]], force: bool,
            workers: int = 1, gptrend: str = 'affine', gpwidth: str = 'global',
            M_X: Optional[int] = None,
            survey: str = 'points', quantized_start: str = 'multistart',
            sigma_points: bool = False,
            fit_weights: str = 'none',
            z: float = 2.0, n_min: int = 30, L_max: Optional[int] = None,
            tag: str = '') -> Tuple[int, int]:
    """`qij`: a sequential draw loop; with `workers > 1` one pool is
    created for the run and passed to every draw's fit, so only the
    prototype survey (method_notes section 2) and the unified loop's
    own shared bins' full-data stencils run in parallel -- the rest of
    a draw is serial regardless of `workers`. `survey` picks the
    prototype survey's receptive-field representation (spec/method_
    notes.md section 2); `quantized_start` picks theta_Q's starting
    point (spec/QIJ_mods_waves.md A9 item 5). `z`/`n_min`/`L_max`
    (spec/QIJ_unified_loop_spec.md 0, 4.4, 5) are the unified loop's
    own user-exposed levers, passed straight to `QIJ`. Every draw also
    writes `step_ratio` (A9 item 4), `splits` and `rounds` (the
    unified loop's own per-split record and per-round calibration
    trajectory, spec 4.5) and `bins` (the per-leaf table, spec 4.5),
    none of them gated behind `--diag-draws`, unlike `points`/
    `prototypes`. `sigma_points` (spec/QIJ_sigma_points_spec.md), off
    by default, runs the optional sigma-points interval stage after
    refinement; when True, every draw also writes the `sigma_points`
    array product. `fit_weights` (spec/QIJ_mass_weighted_fit_spec.md)
    picks the GP pilot's own kernel-regression noise. A21
    (spec/QIJ_mods_waves.md A21) removed `ivqbins`/`pilot`/
    `refine_schedule`/`check_rule`/`tree_rule` as both parameters and
    products, and the unified loop (spec/QIJ_unified_loop_spec.md) then
    replaced A20/A21's own growth/check second stage entirely: the
    `bridge`/`cells` array products (the affine pilot) and the `bin_U`
    array product (the marginal path's own per-output bins) no longer
    exist."""
    draws = list(draws)
    md = products.method_dir(out_dir, dataset, estimator, N, 'qij', tag)
    diag = set(diag_draws) if diag_draws is not None else set()
    case = registry.case(dataset, estimator)
    T = case.make_T()
    pool = Pool(workers, T=T) if workers > 1 else None
    written = 0
    for s in draws:
        if not force and products.is_done(md, s):
            continue
        dseed = seed + s
        X = case.draw(N, dseed)
        res = QIJ(eps=eps, seed=dseed, vq_transform=case.vq_transform,
                  gptrend=gptrend, gpwidth=gpwidth, M_X=M_X,
                  survey=survey, quantized_start=quantized_start,
                  sigma_points=sigma_points,
                  fit_weights=fit_weights,
                  z=z, n_min=n_min, L_max=L_max).fit(X, T, pool=pool)
        row = _qij_row(dataset, estimator, N, s, dseed, res)
        row.update(search_audit(T, X, res.theta_hat_full, dataset, estimator))
        arrays = {'step_ratio': _qij_step_ratio(res),
                  'splits': _qij_splits(res), 'rounds': _qij_rounds(res),
                  'bins': _qij_bins(res)}
        if res.sigma_points:
            arrays['sigma_points'] = _qij_sigma_points(res)
        if s in diag:
            arrays['points'] = _qij_points(res)
            arrays['prototypes'] = _qij_prototypes(res)
        products.write_draw(md, s, row, arrays)
        written += 1
    if pool is not None:
        pool.close()
    return written, len(draws) - written


def _ijfd_row(dataset: str, estimator: str, N: int, s: int, seed: int, res) -> dict:
    """One `ijfd` scalar row (spec A13's product list): `eta_full`,
    `nan_fraction` (under `point_curvature=False`, n_dropped/N: the
    fraction lost even after `ijfd.py`'s backward retry; `n_retried`/
    `n_dropped` are that retry's own counts, both 0 under
    `point_curvature=True`, which has no retry), timing/workers, the
    evaluation/row breakdown by stage ('abc' is 0 under
    `point_curvature=False`, the planner's ruling), and per output
    `V_ijfd`/`a`/`b_hat`/`c_q` (the latter three NaN under
    `point_curvature=False`), one column per `res.outputs` -- the
    measured subset (spec A11), mirroring `qij`'s own `_qij_row`;
    identity `measured` reproduces every output."""
    row = {'dataset': dataset, 'estimator': estimator, 'N': N,
           's': s, 'seed': seed, 'point_curvature': bool(res.point_curvature),
           'eta_full': float(res.eta_full), 'nan_fraction': float(res.nan_fraction),
           'n_retried': int(res.n_retried), 'n_dropped': int(res.n_dropped),
           'wall_time_total': float(res.wall_time_total),
           'busy_time_total': float(res.busy_time_total), 'workers': int(res.workers)}
    for stage in ('cold', 'eta_full', 'forward', 'backward', 'abc', 'check', 'total'):
        row[f'evals_{stage}'] = int(res.evals_by_stage[stage])
        row[f'rows_{stage}'] = int(res.rows_by_stage[stage])
    for j, o in enumerate(res.outputs):
        row[f'V_ijfd_{o}'] = float(res.V[j])
        row[f'a_{o}'] = float(res.a[j])
        row[f'b_hat_{o}'] = float(res.b_hat[j])
        row[f'c_q_{o}'] = float(res.c_q[j])
    return row


def _ijfd_step_ratio(res) -> pd.DataFrame:
    """`step_ratio` for every `ijfd` draw (mirrors `_qij_step_ratio`):
    A13's step-doubling self-check on the ten heaviest points, per
    output (spec A9 item 4)."""
    step_ratio = np.asarray(res.step_ratio, dtype=float)
    data = {'i': np.arange(step_ratio.shape[0])}
    for j, o in enumerate(res.outputs):
        data[f'step_ratio_{o}'] = step_ratio[:, j]
    return pd.DataFrame(data)


def _ijfd_points(res) -> pd.DataFrame:
    """`i, psi_<o>` for every point (mirrors `_qij_points`'s shape;
    named like `ij`'s own stored `psi` array -- both use kind='psi' --
    so the comparison layer reads the two side by side without a
    rerun, spec A13). `psi` is at `res.outputs_full`'s width, not the
    measured subset `res.outputs` (spec A11, `IJFDResult` docstring):
    the comparison needs every output, not only the measured ones."""
    N = res.psi.shape[0]
    data = {'i': np.arange(N)}
    for j, o in enumerate(res.outputs_full):
        data[f'psi_{o}'] = res.psi[:, j]
    return pd.DataFrame(data)


def run_ijfd(dataset: str, estimator: str, N: int, draws: Iterable[int], seed: int,
             out_dir: str, workers: int, point_curvature: bool, force: bool,
             tag: str = '') -> Tuple[int, int]:
    """`ijfd` (spec A13): a sequential draw loop, one pool shared across
    the run (mirrors `run_qij`'s pattern) for each draw's N (or 2N under
    `point_curvature`) point-perturbed fits."""
    draws = list(draws)
    md = products.method_dir(out_dir, dataset, estimator, N, 'ijfd', tag)
    case = registry.case(dataset, estimator)
    T = case.make_T()
    pool = Pool(workers, T=T) if workers > 1 else None
    written = 0
    for s in draws:
        if not force and products.is_done(md, s):
            continue
        dseed = seed + s
        X = case.draw(N, dseed)
        res = IJFD(point_curvature=point_curvature).fit(X, T, pool=pool)
        row = _ijfd_row(dataset, estimator, N, s, dseed, res)
        row.update(search_audit(T, X, res.theta_hat_full, dataset, estimator))
        arrays = {'step_ratio': _ijfd_step_ratio(res), 'psi': _ijfd_points(res)}
        products.write_draw(md, s, row, arrays)
        written += 1
    if pool is not None:
        pool.close()
    return written, len(draws) - written


_QIJT_LEVEL = 0.95
# Stage order of spec/QIJ_qijt_spec.md section 9.1, with `opening`
# (section 5.3/7.2's re-evaluations of theta_Q) as its own stage key,
# counted separately from `tree` and `within` as the spec directs.
_QIJT_STAGES = ('full_fit', 'eta_full', 'xvq', 'base_Q', 'eta_Q', 'tree', 'opening',
                'anchors', 'bias', 'within', 'quadratic', 'drift', 'curvature')


def _qijt_row(dataset: str, estimator: str, N: int, s: int, seed: int, res) -> dict:
    """One `qijt` scalar row (spec section 11.1): the run identifiers,
    `QIJTResult`'s scalars, the per-stage evaluation/row/wall-time
    breakdown (section 9.1), and per output the coefficients and the
    three level-0.95 intervals (section 8.4)."""
    row = {'dataset': dataset, 'estimator': estimator, 'N': N, 's': s, 'seed': seed,
           'M_X': int(res.M_X), 'R_final': int(res.R_final),
           'budget': int(res.budget), 'budget_win': int(res.budget_win),
           'budget_quad': int(res.budget_quad), 'workers': int(res.workers),
           'status': res.status, 'theta_hat_status': res.theta_hat_status,
           'theta_Q_status': res.theta_Q_status, 'eta_full': float(res.eta_full),
           'eta_Q': float(res.eta_Q), 'eta_full_failed': bool(res.eta_full_failed),
           'eta_Q_failed': bool(res.eta_Q_failed),
           'n_fp_full': int(res.n_fp_full), 'r_fp_full': float(res.r_fp_full),
           'n_fp_Q': int(res.n_fp_Q), 'r_fp_Q': float(res.r_fp_Q),
           'n_fp_opening': int(res.n_fp_opening),
           'r_fp_opening_max': float(res.r_fp_opening_max),
           'n_rounds': int(res.n_rounds),
           'n_measured': int(res.n_measured), 'n_opened': int(res.n_opened),
           'n_leaves': int(res.n_leaves), 'max_depth': int(res.max_depth),
           'n_failed': int(res.n_failed), 'tree_status': res.tree_status,
           'n_pairs_bought': int(res.n_pairs_bought), 'n_quad_bought': int(res.n_quad_bought),
           'busy_time': float(res.busy_time), 'wall_time': float(res.wall_time)}
    for stage in _QIJT_STAGES:
        row[f'evals_{stage}'] = int(res.evals_by_stage[stage])
        row[f'rows_{stage}'] = int(res.rows_by_stage[stage])
        row[f'wall_{stage}'] = float(res.wall_time_by_stage[stage])
    lo, hi = res.interval(_QIJT_LEVEL).T
    lo_btw, hi_btw = res.interval_btw(_QIJT_LEVEL).T
    lo_abc, hi_abc = res.abc_interval(_QIJT_LEVEL).T
    for j, o in enumerate(res.outputs):
        row[f'theta_hat_{o}'] = float(res.theta_hat[res.measured[j]])
        row[f'V_btw_{o}'] = float(res.V_btw[j])
        row[f'V_win_{o}'] = float(res.V_win[j])
        row[f'V_tot_{o}'] = float(res.V_tot[j])
        row[f'a_scale_{o}'] = float(res.a_scale[j])
        row[f'a_scatter_{o}'] = float(res.a_scatter[j])
        row[f'root_drift_{o}'] = float(res.root_drift[j])
        row[f'accel_{o}'] = float(res.accel[j])
        row[f'b_hat_{o}'] = float(res.b_hat[j])
        row[f'c_q_{o}'] = float(res.c_q[j])
        row[f'lo_{o}'] = float(lo[j])
        row[f'hi_{o}'] = float(hi[j])
        row[f'lo_btw_{o}'] = float(lo_btw[j])
        row[f'hi_btw_{o}'] = float(hi_btw[j])
        row[f'lo_abc_{o}'] = float(lo_abc[j])
        row[f'hi_abc_{o}'] = float(hi_abc[j])
    return row


def _qijt_points(res) -> pd.DataFrame:
    """`points` for `--diag-draws` (spec section 11.6): `i, leaf`, per
    output `psi_hat_o`, from `res.psi_hat` (N, q) and `res.leaf_of_point`
    (N,)."""
    N = res.psi_hat.shape[0]
    data = {'i': np.arange(N), 'leaf': np.asarray(res.leaf_of_point, dtype=np.int64)}
    for j, o in enumerate(res.outputs):
        data[f'psi_hat_{o}'] = res.psi_hat[:, j]
    return pd.DataFrame(data)


def run_qijt(dataset: str, estimator: str, N: int, draws: Iterable[int], seed: int,
             out_dir: str, M_X: int, budget: int, budget_win: int, budget_quad: int,
             diag_draws: Optional[Iterable[int]], force: bool, workers: int = 1,
             tag: str = '') -> Tuple[int, int]:
    """`qijt` (spec/QIJ_qijt_spec.md): a sequential draw loop, one pool
    shared across the run when `workers > 1` (mirrors `run_qij`'s
    pattern). Every draw writes `nodes`, `leaves`, `curve`, `anchors`
    (sections 11.2-11.5, `QIJTResult`'s own DataFrames) and, on
    `--diag-draws`, `points` (section 11.6)."""
    draws = list(draws)
    md = products.method_dir(out_dir, dataset, estimator, N, 'qijt', tag)
    diag = set(diag_draws) if diag_draws is not None else set()
    case = registry.case(dataset, estimator)
    T = case.make_T()
    pool = Pool(workers, T=T) if workers > 1 else None
    written = 0
    for s in draws:
        if not force and products.is_done(md, s):
            continue
        dseed = seed + s
        X = case.draw(N, dseed)
        res = QIJT(M_X=M_X, budget=budget, budget_win=budget_win, budget_quad=budget_quad,
                   seed=dseed, vq_transform=case.vq_transform).fit(X, T, pool=pool)
        row = _qijt_row(dataset, estimator, N, s, dseed, res)
        row.update(search_audit(T, X, res.theta_hat, dataset, estimator))
        arrays = {'nodes': res.nodes, 'leaves': res.leaves, 'curve': res.curve,
                  'anchors': res.anchors, 'pairs': res.pairs}
        if s in diag:
            arrays['points'] = _qijt_points(res)
        products.write_draw(md, s, row, arrays)
        written += 1
    if pool is not None:
        pool.close()
    return written, len(draws) - written


_QIJDT_STAGES = ('full_fit', 'eta_full', 'tree', 'bias', 'within', 'quadratic', 'curvature')


def _qijdt_row(dataset: str, estimator: str, N: int, s: int, seed: int, res) -> dict:
    """One `qijdt` scalar row (spec/QIJ_qijdt_spec.md section 11.1):
    `QIJDTResult`'s scalars, the per-stage evaluation/wall-time
    breakdown (section 9.1), and per output the coefficients, the
    remainder diagnostic `P_unbought`, and the three level-0.95
    intervals (section 8.4)."""
    row = {'dataset': dataset, 'estimator': estimator, 'N': N, 's': s, 'seed': seed,
           'eps': float(res.eps), 'M_floor': int(res.M_floor), 'workers': int(res.workers),
           'status': res.status, 'theta_hat_status': res.theta_hat_status,
           'eta_full': float(res.eta_full), 'eta_full_failed': bool(res.eta_full_failed),
           'n_fp': int(res.n_fp), 'r_fp': float(res.r_fp), 'n_rounds': int(res.n_rounds),
           'n_measured': int(res.n_measured), 'n_leaves': int(res.n_leaves),
           'max_depth': int(res.max_depth), 'n_failed': int(res.n_failed),
           'n_nonsmooth': int(res.n_nonsmooth), 'n_halvings_total': int(res.n_halvings_total),
           'tree_status': res.tree_status, 'n_pairs_bought': int(res.n_pairs_bought),
           'n_quad_bought': int(res.n_quad_bought), 'n_below_closed': int(res.n_below_closed),
           'evals_total': int(sum(res.evals_by_stage[st] for st in _QIJDT_STAGES)),
           'busy_time': float(res.busy_time), 'wall_time': float(res.wall_time)}
    for stage in _QIJDT_STAGES:
        row[f'evals_{stage}'] = int(res.evals_by_stage[stage])
        row[f'wall_{stage}'] = float(res.wall_time_by_stage[stage])
    lo, hi = res.interval(_QIJT_LEVEL).T
    lo_btw, hi_btw = res.interval_btw(_QIJT_LEVEL).T
    lo_abc, hi_abc = res.abc_interval(_QIJT_LEVEL).T
    for j, o in enumerate(res.outputs):
        row[f'theta_hat_{o}'] = float(res.theta_hat[res.measured[j]])
        row[f'nu_{o}'] = float(res.nu[j])
        row[f'V_btw_{o}'] = float(res.V_btw[j])
        row[f'V_win_{o}'] = float(res.V_win[j])
        row[f'V_tot_{o}'] = float(res.V_tot[j])
        row[f'accel_{o}'] = float(res.accel[j])
        row[f'b_hat_{o}'] = float(res.b_hat[j])
        row[f'c_q_{o}'] = float(res.c_q[j])
        row[f'P_unbought_{o}'] = float(res.P_unbought[j])
        row[f'lo_{o}'] = float(lo[j])
        row[f'hi_{o}'] = float(hi[j])
        row[f'lo_btw_{o}'] = float(lo_btw[j])
        row[f'hi_btw_{o}'] = float(hi_btw[j])
        row[f'lo_abc_{o}'] = float(lo_abc[j])
        row[f'hi_abc_{o}'] = float(hi_abc[j])
    return row


def _qijdt_points(res) -> pd.DataFrame:
    """`points` for `--diag-draws` (spec section 11.6): `i, leaf`, per
    output `psi_hat_o`."""
    N = res.psi_hat.shape[0]
    data = {'i': np.arange(N), 'leaf': np.asarray(res.leaf_of_point, dtype=np.int64)}
    for j, o in enumerate(res.outputs):
        data[f'psi_hat_{o}'] = res.psi_hat[:, j]
    return pd.DataFrame(data)


def run_qijdt(dataset: str, estimator: str, N: int, draws: Iterable[int], seed: int,
              out_dir: str, eps: float, diag_draws: Optional[Iterable[int]], force: bool,
              workers: int = 1, tag: str = '') -> Tuple[int, int]:
    """`qijdt` (spec/QIJ_qijdt_spec.md): a sequential draw loop, one pool
    shared across the run when `workers > 1` (mirrors `run_qijt`'s
    pattern). Every draw writes `nodes`, `leaves`, `pairs`, `curve`
    (sections 11.2-11.5) and, on `--diag-draws`, `points` (section
    11.6)."""
    draws = list(draws)
    md = products.method_dir(out_dir, dataset, estimator, N, 'qijdt', tag)
    diag = set(diag_draws) if diag_draws is not None else set()
    case = registry.case(dataset, estimator)
    T = case.make_T()
    pool = Pool(workers, T=T) if workers > 1 else None
    written = 0
    for s in draws:
        if not force and products.is_done(md, s):
            continue
        dseed = seed + s
        X = case.draw(N, dseed)
        res = QIJDT(eps=eps, vq_transform=case.vq_transform).fit(X, T, pool=pool)
        row = _qijdt_row(dataset, estimator, N, s, dseed, res)
        row.update(search_audit(T, X, res.theta_hat, dataset, estimator))
        arrays = {'nodes': res.nodes, 'leaves': res.leaves, 'pairs': res.pairs,
                  'curve': res.curve}
        if s in diag:
            arrays['points'] = _qijdt_points(res)
        products.write_draw(md, s, row, arrays)
        written += 1
    if pool is not None:
        pool.close()
    return written, len(draws) - written
