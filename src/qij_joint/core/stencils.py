"""
The stencils update the influence model, `refine_update='stencils'`
(spec/QIJ_A17_stencil_update.md, sections 3-4, 8 and 9): every full-data
stencil -- the marginal path's own initial bins and every refinement
split -- is an exact-ish observation of the mean of the influence over
its own set of points, for every measured output at once, and enters
that output's influence model as such. A prototype's own receptive
field is a set too (section 8 item 2): the survey response I_jc/a_c is
an observation of the mean of psi_c over the receptive field's points,
not of psi_c at the prototype's position. Every open leaf of every
output is then re-proposed from the updated model, in synchronized
rounds; `refine_schedule='rounds'` is required (`QIJ.__init__` raises
`ValueError` otherwise -- a refit per split under `'queue'` is not paid
for).

**A failed prototype is a missing row for that output only** (section 9
item 1): output c's own design is every receptive field whose survey
value is finite FOR c, plus every output's current leaves (section 2's
sharing rule, unchanged) -- never a compacted array a failure would
shift. Every coordinate within one kernel GROUP (section 8 item 4:
outputs sharing the same finite prototype design) shares the identical
finite pattern by construction (`influence_model._coordinate_groups`),
so a group's own live receptive-field id list is computed once, from
any one of its coordinates, and used for every coordinate in it.

**Caching** (section 9 item 2, a code-standards ruling, not a
performance claim): a receptive field's own column (its kernel-to-every-
point row `_set_kernel_column`, and its mean-basis row) is a per-draw
invariant -- formed once per kernel group, at the first update, and
reused every round after. A leaf's own column is formed the moment the
leaf is created (an initial bin, or a split's child) and dropped the
moment the leaf is split; nothing about the augmented design is ever
recomputed for a row that already has a cached column. `K_ss` itself
(the small n x n matrix `Kx`'s own columns imply) is still reassembled
every round (`influence_model.assemble_K_ss`) -- pure numpy indexing
over the already-cached `Kx`, no kernel evaluation, so rebuilding it is
not the caching ruling's concern.

**The centring offset follows the model** (section 9 item 3): m_c is
re-derived at every update as the mean of the just-updated psi0_c over
the N points (not the frozen stage-1 offset `prepare_coordinate` was
given), so the centred field has mean zero on the same scale the
stencils' own U's are centred on; Sigma_k p_k ubar_k^2 is not
shift-invariant, so a frozen offset would bias rho^2 and every level
gain after the first update.

A leaf row's own noise is declared from its stencil's step (section 8
item 1b, section 9 item 4: unchanged); a receptive-field row's is
lam_c, re-estimated once, before round 1, by a one-dimensional REML
search (section 8 item 3, section 9 item 4: unchanged) and held fixed
after. The posterior solve is a truncated-eigendecomposition pseudo-
inverse, never a bare Cholesky (section 8 item 1c,
`influence_model.augmented_solve`/`_eigh_pinv`).

Shares `core.refine`'s per-output setup (`prepare_coordinate`) and
per-split update (`apply_split`, called here with `price=False`: a
split's two children are priced only after this round's whole batch of
new sets has been folded into the model) and `core.rounds`'s selection
rule (`_select_round`, reused unchanged) and pool-evaluation task
(`_split_task`). The schedule differs from `core.rounds.
run_refinement_rounds`: a "before round 1" step fits a_c and lam_c and
refits every output's model against the current (initial) design; every
round folds in the round's own new leaves and re-predicts/re-proposes
every OPEN leaf of EVERY output, not only the ones just split; the
outer loop's `active` set is rebuilt fresh every round (rather than the
queue/rounds fixed point) and stops when a round selects nothing across
every output, or every output is at its guard.
"""

from __future__ import annotations

import math
import time
from dataclasses import replace
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .differences import central_step, perturbed_weights, step_parameter
from .influence_model import (
    InfluenceModel, _coordinate_groups, _set_basis_row, _set_kernel_column,
    assemble_K_ss, augmented_bin_posterior_variance, augmented_point_terms,
    augmented_solve, fit_lambda_c,
)
from .joint import _fit_scale
from .refine import CoordinateResult, apply_split, compute_rho2, finalize_coordinate, prepare_coordinate, propose
from .rounds import _select_round, _split_task

__all__ = ["run_refinement_stencils"]


def _group_ctx(model: InfluenceModel, c0: int) -> dict:
    """One coordinate group's shared kernel width (spec section 8 item
    4: the factor c, or ell, is fitted per group; width, c and lam stay
    at their stage-1 REML values, lam_c re-estimated once by `fit_
    lambda_c` before round 1). No prototype-specific geometry is kept
    here any more -- a prototype is itself a set (module docstring)."""
    gpwidth = model.gpwidth
    if gpwidth == 'global':
        ell_or_c = float(model.width[c0])
    else:
        ell_or_c = float(model.c[c0])
    m_g = int(model.m[c0])
    return dict(gpwidth=gpwidth, ell_or_c=ell_or_c, m_g=m_g)


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
    model: InfluenceModel,
    model_indices: Sequence[int],
    pool=None,
    start: np.ndarray = None,
    _debug: Optional[dict] = None,
) -> Tuple[List[CoordinateResult], int, int, float]:
    """
    Refine every measured output together under `refine_update=
    'stencils'` (module docstring). Arguments mirror `core.rounds.
    run_refinement_rounds`'s own. `_debug`, when given a dict, is
    populated with this call's own group structure and per-group live
    receptive-field id lists (`{'groups': ..., 'proto_ids': ...}`) --
    an inert testing hook (section 9 item 1's acceptance check), never
    read by `qij.py`. Returns (this call's `CoordinateResult` per
    output, in `coordinates`' order, the number of rounds run, the
    number of model-update rounds, and their total wall time,
    `update_wall_time`).
    """
    n = len(coordinates)
    results: List[Optional[CoordinateResult]] = [None] * n
    states = {}
    for i in range(n):
        setup = prepare_coordinate(
            X, counter, theta_hat, coordinates[i], names[i], psi0_cols[i], m_vals[i],
            sigma_cols[i], I_proto_cols[i], bmu, bmu2, eta, eps, M_X_used,
            constant_paths[i], Z, model, pool, start, model_indices[i],
        )
        if isinstance(setup, CoordinateResult):
            results[i] = setup
        else:
            states[i] = setup

    if not states:
        return results, 0, 0, 0.0

    mean, transform = model.whitening
    Za = np.asarray(Z, dtype=float)
    if Za.ndim == 1:
        Za = Za.reshape(-1, 1)
    Zw = (Za - mean) @ transform.T
    N = Zw.shape[0]
    d_z = Zw.shape[1]
    abs_coord = {i: states[i].coordinate for i in states}

    raw_groups = _coordinate_groups(model)
    groups = [[c for c in g if c in states] for g in raw_groups]
    groups = [g for g in groups if g]
    group_ctx: Dict[int, dict] = {gi: _group_ctx(model, cols_g[0]) for gi, cols_g in enumerate(groups)}

    # Every live prototype's own receptive field is a candidate row (spec
    # section 8 item 2), keyed by its OWN id (never a compacted position,
    # section 9 item 1); an empty one (no data point has that BMU)
    # contributes no row to any output's design, as built.
    proto_idx: Dict[int, np.ndarray] = {j: np.where(bmu == j)[0] for j in range(M_X_used)}
    proto_idx = {j: idx for j, idx in proto_idx.items() if idx.size > 0}

    # Persistent per-group prototype columns (section 9 item 2: formed
    # once per group, at that group's first update) and the group's own
    # live receptive-field id list (section 9 item 1: every coordinate
    # in a group shares the identical finite pattern, since that is
    # exactly what `_coordinate_groups` groups on).
    proto_ids: Dict[int, List[int]] = {}
    proto_Kx: Dict[int, np.ndarray] = {}
    proto_H: Dict[int, np.ndarray] = {}

    def _init_group_protos(gi: int, cols_g: List[int]) -> None:
        ctx = group_ctx[gi]
        finite = np.isfinite(I_proto_cols[cols_g[0]])
        ids_g = sorted(j for j in proto_idx if finite[j])
        proto_ids[gi] = ids_g
        if ids_g:
            proto_Kx[gi] = np.stack(
                [_set_kernel_column(Zw, proto_idx[j], ctx['gpwidth'], ctx['ell_or_c'], model.h,
                                     model.bmu, d_z) for j in ids_g],
                axis=1,
            )
            proto_H[gi] = np.stack(
                [_set_basis_row(Zw, proto_idx[j], ctx['m_g']) for j in ids_g], axis=0,
            )
        else:
            proto_Kx[gi] = np.zeros((N, 0))
            proto_H[gi] = np.zeros((0, ctx['m_g']))

    if _debug is not None:
        _debug['groups'] = groups

    # a_c (spec section 3): B4's scale factor from each output's own
    # initial bins against the FROZEN model, fitted once and never
    # refit; 1 when non-finite or <= 0.
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
    # (sd_scale * theta_hat[abs_coord[c]])**2 / s2_c (spec section 8
    # item 1b, section 9 item 4: unchanged): sd_scale = sqrt(2)*eta_full
    # /t_S for a directly-measured leaf (an initial bin at its own
    # central-stencil t_k, or a split's smaller child at its own
    # t_small), and (p_small/p_large)*the smaller child's own sd_scale
    # for the larger child, which is never itself measured. Kept in a
    # plain dict, keyed by (output, leaf id) -- `core.refine`'s shared
    # leaf dict is not touched, so `refine_update='none'` stays
    # bit-identical.
    sd_scale: Dict[Tuple[int, int], float] = {}
    delta_c = central_step(eta)
    for i, state in states.items():
        for lid, leaf in state.leaves.items():
            p_k = leaf['n'] / N
            t_k = step_parameter(delta_c, p_k)
            sd_scale[(i, lid)] = math.sqrt(2.0) * eta / t_k

    # A leaf's own column, per group, cached from the moment it is
    # created to the moment it is split (spec section 9 item 2).
    leaf_Kx: Dict[Tuple[int, int, int], np.ndarray] = {}
    leaf_H: Dict[Tuple[int, int, int], np.ndarray] = {}

    def _cache_leaf(i: int, lid: int, idx: np.ndarray) -> None:
        for gi, cols_g in enumerate(groups):
            ctx = group_ctx[gi]
            key = (gi, i, lid)
            leaf_Kx[key] = _set_kernel_column(Zw, idx, ctx['gpwidth'], ctx['ell_or_c'], model.h,
                                               model.bmu, d_z)
            leaf_H[key] = _set_basis_row(Zw, idx, ctx['m_g'])

    def _drop_leaf(i: int, lid: int) -> None:
        for gi in range(len(groups)):
            del leaf_Kx[(gi, i, lid)]
            del leaf_H[(gi, i, lid)]

    for i, state in states.items():
        for lid, leaf in state.leaves.items():
            _cache_leaf(i, lid, leaf['indices'])

    update_wall_time = 0.0
    n_update_rounds = 0
    lam_c_by_i: Dict[int, Optional[float]] = {i: None for i in states}

    def _leaf_response(owner: Tuple[int, int], absolute_c: int) -> float:
        i, lid = owner
        return float(states[i].leaves[lid]['U'][absolute_c])

    def _do_update() -> None:
        nonlocal update_wall_time, n_update_rounds
        t0 = time.perf_counter()
        leaf_owner: List[Tuple[int, int]] = [
            (i, lid) for i, state in states.items() for lid in state.leaves
        ]

        for gi, cols_g in enumerate(groups):
            ctx = group_ctx[gi]
            if gi not in proto_Kx:
                _init_group_protos(gi, cols_g)
            ids_g = proto_ids[gi]
            M_proto_g = len(ids_g)

            if leaf_owner:
                leaf_Kx_block = np.stack([leaf_Kx[(gi,) + owner] for owner in leaf_owner], axis=1)
                leaf_H_block = np.stack([leaf_H[(gi,) + owner] for owner in leaf_owner], axis=0)
            else:
                leaf_Kx_block = np.zeros((N, 0))
                leaf_H_block = np.zeros((0, ctx['m_g']))
            Kx = np.hstack([proto_Kx[gi], leaf_Kx_block])
            H_all = np.vstack([proto_H[gi], leaf_H_block])
            all_idx = [proto_idx[j] for j in ids_g] + [
                states[i].leaves[lid]['indices'] for (i, lid) in leaf_owner
            ]
            K_ss = assemble_K_ss(Kx, all_idx)

            coord_state = {}
            for c in cols_g:
                theta_c = float(theta_hat[abs_coord[c]])
                s2_c = float(model.s2[c])
                leaf_noise_ratio = np.array([
                    (sd_scale[owner] * theta_c) ** 2 / s2_c for owner in leaf_owner
                ])
                response = np.concatenate([
                    np.array([I_proto_cols[c][j] for j in ids_g]) / a_c_by_i[c],
                    np.array([_leaf_response(owner, abs_coord[c]) for owner in leaf_owner]),
                ])
                if lam_c_by_i[c] is None:
                    lam_c_by_i[c] = fit_lambda_c(K_ss, H_all, M_proto_g, leaf_noise_ratio, s2_c, response)
                noise = np.concatenate([np.full(M_proto_g, lam_c_by_i[c]), leaf_noise_ratio])
                coord_state[c] = augmented_solve(K_ss, H_all, noise, response)

            psi0_new, sigma_new, R_new = augmented_point_terms(model, cols_g, coord_state, Kx, Zw, ctx['m_g'])

            for c in cols_g:
                state = states[c]
                state.psi0_c = psi0_new[c]
                state.sigma_c = sigma_new[c]
                # spec section 9 item 3: re-derived, not the frozen
                # stage-1 offset -- Sigma_k p_k ubar_k^2 is not
                # shift-invariant.
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
                    v_vals = augmented_bin_posterior_variance(
                        model, c, coord_state[c], idx_groups, state.sigma_c, Zw, R_new[c], Kx,
                        ctx['gpwidth'], ctx['ell_or_c'], model.h, model.bmu,
                    )
                    for leaf, v in zip(qualifying, v_vals):
                        leaf['v'] = float(v)
                # rho^2 over every CURRENT leaf (open leaves at their
                # just-updated ubar, closed leaves at whatever ubar they
                # were last open with -- spec section 4 step 6 only
                # re-proposes OPEN leaves).
                state.sum_pubar2 = sum(
                    (leaf['n'] / N) * leaf['ubar'] ** 2 for leaf in state.leaves.values()
                )
                rho2 = compute_rho2(state.V_btw, state.sum_pubar2, N)
                for leaf in open_leaves:
                    propose(leaf, rho2, state.psi0_c, state.psi_centered,
                            state.I_proto_c, state.bmu, state.bmu2, N)

        n_update_rounds += 1
        update_wall_time += time.perf_counter() - t0

    _do_update()

    if _debug is not None:
        _debug['proto_ids'] = dict(proto_ids)

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
        for (_, T_val, failed, _wt), (i, leaf, kind, idx_small, idx_large, p_small, p_large,
                                       t_small, tau) in zip(eval_results, task_meta):
            state = states[i]
            small_id = state.next_id
            large_id = state.next_id + 1
            parent_id = leaf['id']
            # Section 4 step 3 / section 8 item 1a: the two children are
            # created, closed or left open by the ported two-strike
            # rule, but NOT priced (v/g) here -- `_do_update` below
            # prices every open leaf, against the design these new
            # leaves (replacing the parent's own row) are about to join.
            apply_split(state, leaf, kind, idx_small, idx_large, p_small, p_large,
                        t_small, T_val, failed, tau, price=False)
            if failed:
                continue
            any_applied = True
            sd_small = math.sqrt(2.0) * state.eta / t_small
            sd_scale[(i, small_id)] = sd_small
            sd_scale[(i, large_id)] = (p_small / p_large) * sd_small
            del sd_scale[(i, parent_id)]  # the split parent's own row leaves the design
            _cache_leaf(i, small_id, state.leaves[small_id]['indices'])
            _cache_leaf(i, large_id, state.leaves[large_id]['indices'])
            _drop_leaf(i, parent_id)

        if any_applied:
            _do_update()

    for i, state in states.items():
        results[i] = finalize_coordinate(state)

    # The round loop's own pool-busy time beyond its wall clock (mirrors
    # `core.rounds.run_refinement_rounds`): folded into the first
    # coordinate's own `busy_delta` since `qij.py` only ever sums this
    # field across outputs.
    if total_busy != 0.0:
        results[0] = replace(results[0], busy_delta=results[0].busy_delta + total_busy)

    return results, n_rounds, n_update_rounds, update_wall_time
