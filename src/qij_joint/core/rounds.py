"""
Synchronized-round scheduling for the marginal path's per-output
refinement, `refine_schedule='rounds'` (spec/QIJ_mods_waves.md A14):
every measured output's leaves at or above ITS OWN current tau are
split together, all those evaluations -- across every output -- running
through the pool (or, serially, through `counter`) in one batch per
round, before any output's tau is recomputed.

`refine.prepare_coordinate` builds each output's initial I-VQ and first
proposals exactly as the queue (`refine.run_refinement`) does;
`refine.apply_split` applies one realized evaluation to an output's
`refine.RefineState` exactly as the queue does. The two schedules
differ only here: which leaves are selected together, against a tau
frozen at the round's start rather than recomputed after every split,
so a leaf late in a round is judged against a slightly larger tau than
the queue would have used -- `rounds` is therefore not bit-identical to
`queue` (an E8-class difference), even though every formula is the
ported one.

An output stops for good once a round finds nothing to select for it:
its own tau, leaves and V_btw only change when one of ITS OWN leaves
splits, so a round that selects nothing for an output can never later
select something for it either -- the same fixed point the queue's own
stopping rule reaches, just checked once per round instead of once per
split. The evaluation guard (1 + M_X_used refinement evaluations,
shared across outputs since M_X_used is) caps each output's selection
to its own remaining budget, in expected-gain order (ties: lower leaf
id, the queue's own tie-break), so the guard is never overshot.
"""

from __future__ import annotations

import time
from dataclasses import replace
from typing import List, Optional, Sequence, Tuple

import numpy as np

from ..parallel import call_T
from .differences import perturbed_weights, step_parameter
from .refine import CoordinateResult, apply_split, finalize_coordinate, prepare_coordinate

__all__ = ["run_refinement_rounds"]


def _split_task(T, case, X: np.ndarray, task):
    """One round's split evaluation on the pool (spec/QIJ_mods_waves.md
    A14) -- the same evaluation `refine.apply_split` would be given
    serially: `task` is (task index, signed weight parameter t, member
    mask, start, eta); `call_T` applies the estimator's prepared state,
    the start-continuation rule (A9) and the per-call eta override
    (spec/QIJ_mods_waves.md A15). Returns (task index, evaluation,
    failure flag, this call's own wall time)."""
    i, t, mask, start, eta = task
    omega = perturbed_weights(np.ones(len(X)), mask, t)
    t0 = time.perf_counter()
    try:
        result = call_T(T, X, omega, start, eta)
        result = np.asarray(result, dtype=float)
        failed = bool(np.any(np.isnan(result)))
    except Exception:
        result = np.full(len(T.outputs), np.nan)
        failed = True
    return i, result, failed, time.perf_counter() - t0


def _select_round(states: dict, active: set) -> dict:
    """This round's selection, per still-active output: every open leaf
    at or above that output's current tau, sorted by (-g, id) and
    capped at its remaining evaluation budget (spec/QIJ_mods_waves.md
    A14). An output with nothing to select is dropped from `active` in
    place -- a fixed point it can never leave under `refine_update=
    'none'` (module docstring); `core.stencils` calls this same
    function but rebuilds `active` fresh every round instead of relying
    on that fixed point, since an update can make an output newly
    qualify again (spec/QIJ_A17_stencil_update.md section 4)."""
    selections = {}
    for i in list(active):
        state = states[i]
        open_leaves = [l for l in state.leaves.values() if l['open']]
        if not open_leaves:
            active.discard(i)
            continue
        tau = state.eps * state.V_btw / len(state.leaves)
        qualifying = sorted(
            (l for l in open_leaves if l['g'] >= tau),
            key=lambda l: (-l['g'], l['id']),
        )
        remaining = state.evals_cap - state.n_refine_evals
        if not qualifying or remaining <= 0:
            active.discard(i)
            continue
        selections[i] = (tau, qualifying[:remaining])
    return selections


def run_refinement_rounds(
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
    model,
    model_indices: Sequence[int],
    pool=None,
    start: np.ndarray = None,
) -> Tuple[List[CoordinateResult], int]:
    """
    Refine every measured output together, in synchronized rounds
    (spec/QIJ_mods_waves.md A14). `refine.prepare_coordinate` sets up
    each output exactly as `refine.run_refinement` does; `_select_round`
    picks, from every output still active, the leaves this round
    evaluates; those evaluations run in one `pool.map` batch (or,
    serially, one `counter` call at a time, in the same order); each
    result is applied with `refine.apply_split` before the next round's
    selection is made. Every other argument mirrors
    `refine.run_refinement`'s own, one entry per measured output in the
    per-output sequences, same as `eta`/`eps`/`M_X_used`. Returns (this
    call's `CoordinateResult` per output, in `coordinates`' order, and
    the number of rounds run).
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

    n_rounds = 0
    total_busy = 0.0
    active = set(states.keys())
    while active:
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

        for (_, T_val, failed, _wt), (i, leaf, kind, idx_small, idx_large, p_small, p_large,
                                       t_small, tau) in zip(eval_results, task_meta):
            apply_split(states[i], leaf, kind, idx_small, idx_large, p_small, p_large,
                        t_small, T_val, failed, tau)

    for i, state in states.items():
        results[i] = finalize_coordinate(state)

    # The round loop's own pool-busy time beyond its wall clock (mirrors
    # `ivq.bin_differences`) mixes evaluations from every output in one
    # batch and cannot be split among them meaningfully; it is folded
    # into the first coordinate's own `busy_delta` since `qij.py` only
    # ever sums this field across outputs.
    if total_busy != 0.0:
        results[0] = replace(results[0], busy_delta=results[0].busy_delta + total_busy)

    return results, n_rounds
