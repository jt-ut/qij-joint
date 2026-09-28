"""
The stencils update the influence model, `refine_update='stencils'`
(spec/QIJ_A17_stencil_update.md, section 10, final -- supersedes
sections 3, 8 and 9 where they differ). Under 'stencils', stage 1
ITSELF is the set-observation model (section 10.1, built by `qij.py`'s
own call to `influence_model.fit_rows_model` with no extra rows): the
prototype survey enters as an observation of the mean of psi_c over a
receptive field's own points, never a value at the prototype's
position, and the initial bins are built from that model.

This module picks up from there: measures the initial bins
(`core.refine.prepare_coordinate`, unpriced -- `skip_pricing=True`,
since stage 1 here is not an `InfluenceModel` `bin_posterior_variance`
could price against), fits a_c, then does 10.1's own 'one full refit':
ONE more call to `fit_rows_model`, now with every output's own current
leaves folded in as extra rows at their declared noise (each output's
own largest leaf excluded -- section 10.2b, the other half of the
structural redundancy removal section 8 item 1a already gives the
split-parent/children half of). From there, every later round only
extends the design with the round's own new leaves and re-solves at
the FIXED hyperparameters the one full refit chose
(`influence_model.rows_solve`): no further hyperparameter search, no
eigendecomposition rank cut, a plain Cholesky with the existing ridge
ladder (`influence_model._cholesky_ridge_step`). A receptive field's
own K.A column is a per-draw invariant, formed once per kernel group
right after the one full refit and reused every round after (section
10.3); a leaf's column is formed the moment it is created and dropped
the moment it is split.

Shares `core.refine`'s per-output setup (`prepare_coordinate`) and
per-split update (`apply_split`, called here with `price=False`: a
split's two children are priced only after this round's whole batch of
new sets has been folded into the model) and `core.rounds`'s selection
rule (`_select_round`, reused unchanged) and pool-evaluation task
(`_split_task`). The outer loop's `active` set is rebuilt fresh every
round (rather than the queue/rounds fixed point) and stops when a round
selects nothing across every output, or every output is at its guard.
"""

from __future__ import annotations

import math
import time
from dataclasses import replace
from typing import Dict, List, Optional, Sequence, Tuple

import logging

import numpy as np

from .differences import central_step, perturbed_weights, step_parameter
from .influence_model import (
    _set_basis_row, fit_rows_model, rows_bin_posterior_variance,
    rows_kernel_columns, rows_point_terms, rows_solve,
)
from .joint import _fit_scale
from .refine import CoordinateResult, apply_split, compute_rho2, finalize_coordinate, prepare_coordinate, propose
from .rounds import _select_round, _split_task

_log = logging.getLogger(__name__)

__all__ = ["run_refinement_stencils"]


def run_refinement_stencils(
    X: np.ndarray,
    counter,
    theta_hat: np.ndarray,
    coordinates: Sequence[int],
    names: Sequence[str],
    psi0_cols: Sequence[np.ndarray],
    m_vals: Sequence[float],
    sigma_cols: Sequence[np.ndarray],
    I_proto_cols: Sequence[np.ndarray],
    bmu: np.ndarray,
    bmu2: np.ndarray,
    eta: float,
    eps: float,
    M_X_used: int,
    constant_paths: Sequence[bool],
    Z: np.ndarray,
    model_indices: Sequence[int],
    xvq,
    theta_Q_vec: np.ndarray,
    gptrend: str,
    gpwidth: str,
    stage1_per_coord: Dict[int, dict],
    stage1_aux: dict,
    pool=None,
    start: np.ndarray = None,
    _debug: Optional[dict] = None,
) -> Tuple[List[CoordinateResult], int, int, float, Dict[str, object]]:
    """
    Refine every measured output together under `refine_update=
    'stencils'` (module docstring). `stage1_per_coord`/`stage1_aux` are
    stage 1's own set-observation fit (`qij.py`'s call to `fit_rows_
    model`, receptive fields only), reused here for a_c's own
    reference s2_c and the shared whitening/CONN geometry -- not
    re-derived. `theta_Q_vec`/`I_proto_cols` are already at the
    measured width (position c matches `coordinates[c]`/`psi0_cols[c]`'s
    own output; `theta_Q_vec[c]`/`I_proto_cols[c]` line up the same way).
    Returns (this call's `CoordinateResult` per output, the number of
    rounds run, the number of model-update rounds, their total wall
    time `update_wall_time`, and a `products` dict: `lambda_c_stage1`/
    `lambda_c_update` (dicts keyed by output), `ridge_step_max` and
    `first_formation_wall_time` (shared, spec section 10.3)).
    """
    n = len(coordinates)
    results: List[Optional[CoordinateResult]] = [None] * n
    states = {}
    for i in range(n):
        setup = prepare_coordinate(
            X, counter, theta_hat, coordinates[i], names[i], psi0_cols[i], m_vals[i],
            sigma_cols[i], I_proto_cols[i], bmu, bmu2, eta, eps, M_X_used,
            constant_paths[i], Z, None, pool, start, model_indices[i],
            skip_pricing=True,
        )
        if isinstance(setup, CoordinateResult):
            results[i] = setup
        else:
            states[i] = setup

    products: Dict[str, object] = dict(
        lambda_c_stage1={i: float(stage1_per_coord[i]['lam']) for i in states},
        lambda_c_update={i: float('nan') for i in states},
        ridge_step_max=0, first_formation_wall_time=0.0,
    )

    if not states:
        return results, 0, 0, 0.0, products

    mean, transform = stage1_aux['whitening']
    h_full = stage1_aux['h_full']
    bmu_full = stage1_aux['bmu_full']
    d_z = stage1_aux['d_z']
    proto_idx_all = stage1_aux['proto_idx']
    Za = np.asarray(Z, dtype=float)
    if Za.ndim == 1:
        Za = Za.reshape(-1, 1)
    Zw = (Za - mean) @ transform.T
    N = Zw.shape[0]
    abs_coord = {i: states[i].coordinate for i in states}

    # a_c (spec section 3): B4's scale factor from each output's own
    # initial bins against the FROZEN stage-1 model, fitted once and
    # never refit; 1 when non-finite or <= 0.
    a_c_by_i: Dict[int, float] = {}
    for i, state in states.items():
        leaves = list(state.leaves.values())
        p = np.array([leaf['n'] for leaf in leaves], dtype=float) / N
        U_c = np.array([leaf['U'][state.coordinate] for leaf in leaves])
        m_kc = np.array([leaf['ubar'] + m_vals[i] for leaf in leaves])
        a_c = float(_fit_scale(p, U_c[:, None], m_kc[:, None])[0])
        if not (np.isfinite(a_c) and a_c > 0.0):
            a_c = 1.0
        a_c_by_i[i] = a_c
        state.a_c = a_c

    # A leaf's own declared noise-to-signal ratio at coordinate c is
    # (sd_scale * theta_hat[abs_coord[c]])**2 / s2_ref, floored at
    # 1e-12 (spec section 8 item 1b; section 10.2's own positivity
    # floor): sd_scale = sqrt(2)*eta_full/t_S for a directly-measured
    # leaf (an initial bin at its own central-stencil t_k, or a split's
    # smaller child at its own t_small), and (p_small/p_large)*the
    # smaller child's own sd_scale for the larger child, which is never
    # itself measured. `s2_ref` is stage 1's OWN s2_c: fixed, so the
    # ratio needed to BUILD the one full refit does not depend
    # circularly on that refit's own (not yet known) s2_c.
    sd_scale: Dict[Tuple[int, int], float] = {}
    delta_c = central_step(eta)
    for i, state in states.items():
        for lid, leaf in state.leaves.items():
            p_k = leaf['n'] / N
            t_k = step_parameter(delta_c, p_k)
            sd_scale[(i, lid)] = math.sqrt(2.0) * eta / t_k

    def _noise_ratio(owner: Tuple[int, int], c: int) -> float:
        theta_c = float(theta_hat[abs_coord[c]])
        s2_ref = float(stage1_per_coord[c]['s2'])
        ratio = (sd_scale[owner] * theta_c) ** 2 / s2_ref if s2_ref > 0.0 else 1e-12
        return max(ratio, 1e-12)

    def _leaf_response(owner: Tuple[int, int], absolute_c: int) -> float:
        i, lid = owner
        return float(states[i].leaves[lid]['U'][absolute_c])

    def _excluded_largest_leaf() -> Dict[int, int]:
        """Section 10.2b: each output's own leaf partition implies the
        same global-mean functional every partition does, so its own
        largest CURRENT leaf's row is dropped from the shared design --
        recomputed every call, since the partition evolves as leaves
        split."""
        excluded = {}
        for i, state in states.items():
            if state.leaves:
                excluded[i] = max(state.leaves.items(), key=lambda kv: kv[1]['n'])[0]
        return excluded

    def _current_leaf_rows() -> Tuple[List[Tuple[int, int]], List[np.ndarray]]:
        excluded = _excluded_largest_leaf()
        owners, idxs = [], []
        for i, state in states.items():
            for lid, leaf in state.leaves.items():
                if lid == excluded.get(i):
                    continue
                owners.append((i, lid))
                idxs.append(leaf['indices'])
        return owners, idxs

    # Every currently-existing leaf's own column is cached per group,
    # keyed (group, output, leaf id); formed at creation, dropped at
    # split (section 10.3) -- independent of whether that leaf happens
    # to be the CURRENT largest (excluded from the design this round):
    # its column is still needed the moment it stops being the largest.
    leaf_Kx: Dict[Tuple[int, int, int], np.ndarray] = {}
    leaf_H: Dict[Tuple[int, int, int], np.ndarray] = {}
    n_groups = 0  # set once the one full refit fixes the group count

    def _cache_leaves(new: List[Tuple[int, int, np.ndarray]], group_ctx: Dict[int, dict]) -> None:
        # One K.A pass per group for every leaf created this round.
        if not new:
            return
        for gi, ctx in group_ctx.items():
            Kx_new = rows_kernel_columns(Zw, [idx for _, _, idx in new], gpwidth, ctx['param'],
                                         h_full, bmu_full, d_z)
            for col, (i, lid, idx) in enumerate(new):
                leaf_Kx[(gi, i, lid)] = Kx_new[:, col]
                leaf_H[(gi, i, lid)] = _set_basis_row(Zw, idx, ctx['m'])

    def _drop_leaf(i: int, lid: int, group_ctx: Dict[int, dict]) -> None:
        for gi in group_ctx:
            del leaf_Kx[(gi, i, lid)]
            del leaf_H[(gi, i, lid)]

    # --- The 'one full refit' (spec section 10.1): a_c applied to the
    # survey responses, every output's own current (initial) leaves
    # folded in (its own largest dropped, section 10.2b), the FULL
    # hyperparameter search run once more, over every state at once (a
    # single `fit_rows_model` call re-groups by the rescaled I_proto's
    # own finite pattern, identical to stage 1's).
    t0 = time.perf_counter()
    leaf_owner, leaf_idx = _current_leaf_rows()
    I_rescaled = np.stack([I_proto_cols[c] / a_c_by_i[c] for c in states], axis=1)
    extra_response = (
        np.array([[_leaf_response(owner, abs_coord[c]) for c in states] for owner in leaf_owner])
        if leaf_owner else np.zeros((0, len(states)))
    )
    extra_noise = (
        np.array([[_noise_ratio(owner, c) for c in states] for owner in leaf_owner])
        if leaf_owner else np.zeros((0, len(states)))
    )
    per_coord, shared_by_group, _const2, _aux2 = fit_rows_model(
        Z, xvq, I_rescaled, theta_Q_vec, eta, gptrend=gptrend, gpwidth=gpwidth,
        extra_idx=leaf_idx, extra_response=extra_response, extra_noise_ratio=extra_noise,
    )
    # `fit_rows_model`'s own coordinates are local to the (M_X_used, n)
    # matrices just built, in the SAME order as `states`' own keys
    # (`I_rescaled`'s columns): local position i IS state key i here,
    # since every one of `states` was passed, none held back.
    for i in states:
        products['lambda_c_stage1'][i] = float(stage1_per_coord[i]['lam'])
        products['lambda_c_update'][i] = float(per_coord[i]['lam'])
    products['ridge_step_max'] = max((per_coord[i]['ridge_step'] for i in states), default=0)

    group_ctx = {gi: dict(param=sh['param'], m=sh['m'], cols=sh['cols'], ids_reduced=sh['ids_reduced'])
                 for gi, sh in shared_by_group.items()}
    n_groups = len(group_ctx)

    # Cache each group's own receptive-field columns (formed once,
    # reused every round after -- section 10.3) and every CURRENT
    # leaf's own column (formed now, at whatever round it happens to
    # already exist).
    proto_Kx: Dict[int, np.ndarray] = {}
    proto_H: Dict[int, np.ndarray] = {}
    for gi, ctx in group_ctx.items():
        ids_reduced = ctx['ids_reduced']
        proto_Kx[gi] = (
            rows_kernel_columns(Zw, [proto_idx_all[j] for j in ids_reduced], gpwidth, ctx['param'],
                                h_full, bmu_full, d_z)
            if ids_reduced else np.zeros((N, 0))
        )
        proto_H[gi] = (
            np.stack([_set_basis_row(Zw, proto_idx_all[j], ctx['m']) for j in ids_reduced], axis=0)
            if ids_reduced else np.zeros((0, ctx['m']))
        )
    _cache_leaves([(i, lid, leaf['indices']) for i, state in states.items()
                   for lid, leaf in state.leaves.items()], group_ctx)

    products['first_formation_wall_time'] = time.perf_counter() - t0

    def _assemble(gi: int, ctx: dict, owners: List[Tuple[int, int]]) -> Tuple[np.ndarray, np.ndarray, List[np.ndarray]]:
        if owners:
            leaf_Kx_block = np.stack([leaf_Kx[(gi,) + o] for o in owners], axis=1)
            leaf_H_block = np.stack([leaf_H[(gi,) + o] for o in owners], axis=0)
        else:
            leaf_Kx_block = np.zeros((N, 0))
            leaf_H_block = np.zeros((0, ctx['m']))
        Kx = np.hstack([proto_Kx[gi], leaf_Kx_block])
        Hb = np.vstack([proto_H[gi], leaf_H_block])
        all_idx = [proto_idx_all[j] for j in ctx['ids_reduced']] + \
            [states[o[0]].leaves[o[1]]['indices'] for o in owners]
        return Kx, Hb, all_idx

    lam_by_c = {i: per_coord[i]['lam'] for i in states}

    def _reprice_and_repropose(per_coord_now: Dict[int, dict]) -> None:
        for gi, ctx in group_ctx.items():
            cols_g = ctx['cols']
            owners, _idx = _current_leaf_rows()
            Kx, _Hb, _all_idx = _assemble(gi, ctx, owners)
            psi0_new, sigma_new, R_new = rows_point_terms(per_coord_now, cols_g, Kx, Zw, ctx['m'])
            for c in cols_g:
                state = states[c]
                state.psi0_c = psi0_new[c]
                state.sigma_c = sigma_new[c]
                # spec section 9 item 3 (not superseded by section 10):
                # re-derived, not the frozen stage-1 offset -- Sigma_k
                # p_k ubar_k^2 is not shift-invariant.
                m_c_new = float(psi0_new[c].mean())
                state.psi_centered = psi0_new[c] - m_c_new
                open_leaves = [leaf for leaf in state.leaves.values() if leaf['open']]
                for leaf in open_leaves:
                    idx = leaf['indices']
                    leaf['var_k'] = float(np.var(state.psi0_c[idx]))
                    leaf['ubar'] = float(state.psi_centered[idx].mean())
                qualifying = [leaf for leaf in open_leaves if leaf['n'] > 1]
                if qualifying:
                    idx_groups = [leaf['indices'] for leaf in qualifying]
                    v_vals = rows_bin_posterior_variance(
                        per_coord_now, c, idx_groups, state.sigma_c, Zw, R_new[c], Kx,
                        gpwidth, ctx['param'], h_full, bmu_full,
                    )
                    for leaf, v in zip(qualifying, v_vals):
                        leaf['v'] = float(v)
                state.sum_pubar2 = sum(
                    (leaf['n'] / N) * leaf['ubar'] ** 2 for leaf in state.leaves.values()
                )
                rho2 = compute_rho2(state.V_btw, state.sum_pubar2, N)
                for leaf in open_leaves:
                    propose(leaf, rho2, state.psi0_c, state.psi_centered,
                            state.I_proto_c, state.bmu, state.bmu2, N)
                products['ridge_step_max'] = max(products['ridge_step_max'], per_coord_now[c]['ridge_step'])

    _reprice_and_repropose(per_coord)
    update_wall_time = time.perf_counter() - t0
    n_update_rounds = 1

    def _do_round_update() -> None:
        nonlocal update_wall_time
        t0r = time.perf_counter()
        owners, _idx = _current_leaf_rows()
        per_coord_now = {}
        for gi, ctx in group_ctx.items():
            cols_g = ctx['cols']
            Kx, Hb, all_idx = _assemble(gi, ctx, owners)
            response = {c: np.concatenate([
                np.array([I_proto_cols[c][j] / a_c_by_i[c] for j in ctx['ids_reduced']]),
                np.array([_leaf_response(o, abs_coord[c]) for o in owners]),
            ]) for c in cols_g}
            leaf_noise_by_c = {c: np.array([_noise_ratio(o, c) for o in owners]) for c in cols_g}
            pc = rows_solve(Kx, all_idx, Hb, len(ctx['ids_reduced']), ctx['m'], cols_g,
                             response, {c: lam_by_c[c] for c in cols_g}, leaf_noise_by_c)
            for c in cols_g:
                per_coord_now[c] = pc[c]
        _reprice_and_repropose(per_coord_now)
        update_wall_time += time.perf_counter() - t0r

    n_rounds = 0
    total_busy = 0.0
    while True:
        active = {i for i in states if any(leaf['open'] for leaf in states[i].leaves.values())}
        selections = _select_round(states, active)
        if not selections:
            break
        n_rounds += 1

        tasks = []
        task_meta = []
        for i, (tau, leaves_sel) in selections.items():
            state = states[i]
            for leaf in leaves_sel:
                kind, idx_a, idx_b = leaf['split']
                if idx_a.size <= idx_b.size:
                    idx_small, idx_large = idx_a, idx_b
                else:
                    idx_small, idx_large = idx_b, idx_a
                p_small = idx_small.size / state.N
                p_large = idx_large.size / state.N
                t_small = step_parameter(state.delta_f, p_small)
                mask_small = np.zeros(state.N, dtype=bool)
                mask_small[idx_small] = True
                tasks.append((len(tasks), t_small, mask_small, start, state.eta))
                task_meta.append((i, leaf, kind, idx_small, idx_large, p_small, p_large,
                                   t_small, tau))

        if pool is None:
            eval_results = []
            for (idx, t_small, mask_small, _start, _eta) in tasks:
                omega = perturbed_weights(np.ones(len(X)), mask_small, t_small)
                T_val = np.asarray(counter(X, omega, start=start, eta=_eta), dtype=float)
                failed = bool(np.any(np.isnan(T_val)))
                eval_results.append((idx, T_val, failed, 0.0))
        else:
            t_map0 = time.perf_counter()
            eval_results = pool.map(_split_task, tasks)
            elapsed = time.perf_counter() - t_map0
            total_busy += sum(r[3] for r in eval_results) - elapsed
            for (_, _, failed, _wt) in eval_results:
                counter.add(1, len(X), int(failed))

        any_applied = False
        new_leaves: List[Tuple[int, int, np.ndarray]] = []
        for (_, T_val, failed, _wt), (i, leaf, kind, idx_small, idx_large, p_small, p_large,
                                       t_small, tau) in zip(eval_results, task_meta):
            state = states[i]
            small_id = state.next_id
            large_id = state.next_id + 1
            parent_id = leaf['id']
            # Section 4 step 3 / section 8 item 1a: the two children are
            # created, closed or left open by the ported two-strike
            # rule, but NOT priced (v/g) here -- `_do_round_update`
            # below prices every open leaf, against the design these
            # new leaves (replacing the parent's own row) are about to
            # join.
            apply_split(state, leaf, kind, idx_small, idx_large, p_small, p_large,
                        t_small, T_val, failed, tau, price=False)
            if failed:
                continue
            any_applied = True
            sd_small = math.sqrt(2.0) * state.eta / t_small
            sd_scale[(i, small_id)] = sd_small
            sd_scale[(i, large_id)] = (p_small / p_large) * sd_small
            del sd_scale[(i, parent_id)]
            new_leaves.append((i, small_id, state.leaves[small_id]['indices']))
            new_leaves.append((i, large_id, state.leaves[large_id]['indices']))
            _drop_leaf(i, parent_id, group_ctx)

        if any_applied:
            n_update_rounds += 1
            t_cache = time.perf_counter()
            _cache_leaves(new_leaves, group_ctx)
            update_wall_time += time.perf_counter() - t_cache
            _do_round_update()
            _log.info('round %d: %d splits, %d leaves, update %.1fs cumulative',
                      n_rounds, len(new_leaves) // 2, sum(len(st.leaves) for st in states.values()),
                      update_wall_time)

    for i, state in states.items():
        cr = finalize_coordinate(state)
        results[i] = replace(cr, a_c=a_c_by_i[i], lambda_c_update=products['lambda_c_update'][i])

    # The round loop's own pool-busy time beyond its wall clock (mirrors
    # `core.rounds.run_refinement_rounds`): folded into the first
    # coordinate's own `busy_delta` since `qij.py` only ever sums this
    # field across outputs.
    if total_busy != 0.0:
        results[0] = replace(results[0], busy_delta=results[0].busy_delta + total_busy)

    return results, n_rounds, n_update_rounds, update_wall_time, products
