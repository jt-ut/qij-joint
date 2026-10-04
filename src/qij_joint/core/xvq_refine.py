"""
Pilot feedback, section 2 (spec/QIJ_pilot_feedback_interface.md): rebuild
the codebook adjacency from (bmu, bmu2) pairs, refine the selected cells
by k-means, and re-survey only the refined prototypes. Reuses `xvq.py`'s
own helpers (the moments-survey row builder, the field step-weight
constructor, the forward-step/step-parameter rule, the per-task survey
evaluation, and `_resolve_bmu2`'s distance routine) rather than
duplicating them.
"""

from __future__ import annotations

import time
from typing import List, Optional, Tuple

import numpy as np
from scipy.sparse import csr_matrix
from vqlp import VQFitter

from .xvq import (
    XVQ,
    SurveyRows,
    _field_step_weights,
    _moments_survey_rows,
    _resolve_bmu2,
    _survey_task,
    forward_step,
    step_parameter,
)


def rebuild_conn(bmu: np.ndarray, bmu2: np.ndarray, M: int):
    """(M, M) csr adjacency built from the (bmu, bmu2) pairs exactly as
    vqlp's CADJ is defined (`vector-quantizer-lp/src/vqlp/vq.py`,
    `_compute_connectivity_matrix`): CADJ[i, j] = #{points : 1st BMU =
    i, 2nd BMU = j}, rows = bmu (1st BMU), cols = bmu2 (2nd BMU), data =
    ones(N, dtype=int), via `csr_matrix((data, (rows, cols)))` (which
    sums duplicate (row, col) pairs on construction -- no `sum_duplicates`
    call is needed, but one is made anyway so the returned matrix's CSR
    structure is already canonical)."""
    bmu = np.asarray(bmu, dtype=np.intp)
    bmu2 = np.asarray(bmu2, dtype=np.intp)
    data = np.ones(bmu.shape[0], dtype=int)
    conn = csr_matrix((data, (bmu, bmu2)), shape=(int(M), int(M)))
    conn.sum_duplicates()
    return conn.tocsr()


def refine_cells(
    Z: np.ndarray, xvq: XVQ, plan, seed: int,
) -> Tuple[XVQ, List[Tuple[int, list]], list]:
    """Section 6 rule 3' (spec/QIJ_pilot_feedback_interface.md). `plan`:
    list of (j, Q_j int array, k_j) -- `Q_j`/`k_j` are the caller's own
    (`qij.py`'s `_FeedbackRefiner`), built from the WHERE rule's
    selection and the current leaf sizes (rule 3'); this function only
    fits and assigns.

    For each (j, Q_j, k_j) (processed in ascending j, so new ids are
    appended in that order regardless of `plan`'s own order): k_j-means
    on `Z[Q_j]` via `vqlp.VQFitter` (M=k_j, p=2, max_bmu=2,
    `random_state` derived from `seed` and j, so the result is
    deterministic in `seed`), then EVERY point currently at bmu j
    (`idx_cell`, which may be a strict superset of `Q_j` -- rule 3'
    carves `Q_j` out of leaves with >= 2 points, leaving any singleton
    leaf at cell j out of the k-means fit but still needing a new bmu)
    is assigned to the nearest of the k_j fitted sub-centers via the
    same `VQFitter.recall` routine. Sub-cell 0 keeps prototype id j;
    sub-cells 1..k_j-1 are APPENDED (ids ascending in j, then sub-cell
    order). `k_j < 2`, an empty sub-cell, or any exception from the fit
    skips j entirely (no mutation to `bmu`/`centers` for that cell;
    reported in the third return value).

    bmu2 is recomputed for EVERY point (not only the refined cells'),
    since a point outside a refined cell can have its old bmu2 be a
    prototype id that no longer denotes the same cell (cell j's old
    single center is replaced by one of its sub-centers); `conn` is
    then `rebuild_conn` of the new (bmu, bmu2)."""
    Z = np.asarray(Z, dtype=float)
    N = Z.shape[0]
    M_old = xvq.M_used

    centers = np.array(xvq.centers, dtype=float, copy=True)
    bmu = np.array(xvq.bmu, dtype=np.intp, copy=True)

    plan_sorted = sorted(plan, key=lambda t: int(t[0]))
    refined: List[Tuple[int, list]] = []
    skipped: list = []
    new_center_rows: List[np.ndarray] = []
    next_id = M_old

    for j, Q_j, k_j in plan_sorted:
        j = int(j)
        k_j = int(k_j)
        if k_j < 2:
            skipped.append(j)
            continue
        Q_j = np.asarray(Q_j, dtype=np.intp)
        Zq = Z[Q_j]
        idx_cell = np.where(bmu == j)[0]
        # B4: a cell can have >= k_j distinct points in Q_j yet still
        # yield an empty k-means sub-cell (e.g. near-duplicates), and
        # VQFitter can raise; either way this cell is skipped (prototype
        # unchanged), never failing the whole pass.
        try:
            fitter = VQFitter(M=k_j, p=2, max_bmu=2, random_state=seed + j, verbose=False)
            fitter.fit(Zq)
            fitter.recall(Z[idx_cell])
            sub_bmu = fitter.recaller.BMU[:, 0]
            sub_centers = fitter.W
        except Exception:
            skipped.append(j)
            continue

        masks = [sub_bmu == k for k in range(k_j)]
        if any(not m.any() for m in masks):
            skipped.append(j)
            continue

        new_ids = [j]
        centers[j] = sub_centers[0]
        # mask0's points stay at bmu j -- no assignment needed, but set
        # explicitly for clarity (a no-op: bmu[idx_cell[masks[0]]] == j already).
        bmu[idx_cell[masks[0]]] = j
        for k in range(1, k_j):
            j_new = next_id
            new_center_rows.append(sub_centers[k])
            bmu[idx_cell[masks[k]]] = j_new
            new_ids.append(j_new)
            next_id += 1
        refined.append((j, new_ids))

    if new_center_rows:
        centers = np.vstack([centers, np.array(new_center_rows)])
    M_new = centers.shape[0]

    p = np.bincount(bmu, minlength=M_new).astype(float) / N

    # bmu2 recomputed for every point: pass bmu2 = -1 for all of them so
    # `_resolve_bmu2` treats every point as "missing" and resolves it to
    # the nearest LIVE prototype other than its own bmu, using its own
    # batched distance routine (the only distance logic reused here).
    bmu2 = _resolve_bmu2(Z, centers, bmu, np.full(N, -1, dtype=np.intp))

    conn = rebuild_conn(bmu, bmu2, M_new)

    xvq_new = XVQ(
        centers=centers, labels=bmu, p=p, bmu=bmu, bmu2=bmu2,
        conn=conn, M_requested=xvq.M_requested, M_used=M_new,
    )
    return xvq_new, refined, skipped


def resurvey(
    counter, X: np.ndarray, xvq_new: XVQ, refined, I_proto_old: np.ndarray,
    theta_Q_old: np.ndarray, sv_old: SurveyRows, eta: float, pool,
    eta_rows: Optional[float],
) -> Tuple[np.ndarray, np.ndarray, SurveyRows, int, int, float]:
    """Section 0 step 4 (measurement/centring unchanged by section 6).
    One new quantized fit on the new codebook's moments-survey rows
    (continued from `theta_Q_old`), then one forward-difference
    measurement per prototype of EVERY refined cell -- `refined`'s own
    (j, [j, j_new1, ...]) form (section 6 rule 3'): ids j and every
    appended sibling, cost sum of k_j evaluations over the refined
    cells -- exactly as `xvq.prototype_influences`'s own per-prototype
    step: same `step_parameter`/`_field_step_weights`, `start=
    theta_Q_new`, `eta=eta_rows`. `delta_f` reuses `sv_old.eta_Q` (not
    re-measured here) when `counter.takes_start`, else
    `forward_step(eta)` -- the A9 self-checks (eta_Q, step_ratio) are
    not re-run by a refinement pass; `sv_new.eta_Q` carries `sv_old.
    eta_Q` forward and `sv_new.step_ratio` is all-NaN (ASSUMPTION,
    stated in the report). Every other prototype keeps its old
    `I_proto_old` row; the whole array is then re-centred mass-weighted
    over finite entries with the NEW `xvq_new.p`, as
    `prototype_influences` does at its end."""
    X = np.asarray(X, dtype=float)
    M_new = xvq_new.M_used
    p_new = xvq_new.p

    rows, row_field, omega0 = _moments_survey_rows(X, xvq_new.bmu, p_new, M_new)

    ev0, rows0 = counter.snapshot()
    theta_Q_new = np.asarray(
        counter(rows, omega0, start=theta_Q_old, eta=eta_rows), dtype=float
    )
    q_full = theta_Q_new.shape[0]

    if getattr(counter, 'takes_start', False):
        delta_f = forward_step(sv_old.eta_Q)
    else:
        delta_f = forward_step(eta)

    M_old = I_proto_old.shape[0]
    refined_js = {int(j) for j, _ids in refined}
    refine_ids = [int(i) for _j, ids in refined for i in ids]

    I_proto_new = np.full((M_new, q_full), np.nan)
    for k in range(M_old):
        if k not in refined_js:
            I_proto_new[k] = I_proto_old[k]

    busy_delta = 0.0
    if pool is None:
        for j in refine_ids:
            p_j = float(p_new[j])
            t_j = step_parameter(delta_f, p_j)
            omega = _field_step_weights(omega0, row_field, j, p_j, t_j)
            I_proto_new[j] = (
                counter(rows, omega, start=theta_Q_new, eta=eta_rows) - theta_Q_new
            ) / t_j
    else:
        pool.share(rows)
        try:
            t_j_of = {}
            tasks = []
            for j in refine_ids:
                p_j = float(p_new[j])
                t_j_of[j] = step_parameter(delta_f, p_j)
                omega = _field_step_weights(omega0, row_field, j, p_j, t_j_of[j])
                tasks.append((j, omega, theta_Q_new, eta_rows))
            t_map0 = time.perf_counter()
            results = pool.map(_survey_task, tasks)
            busy_delta = -(time.perf_counter() - t_map0)
            row_count = rows.shape[0]
            for j, result, failed, wall in results:
                I_proto_new[j] = (result - theta_Q_new) / t_j_of[j]
                counter.add(1, row_count, int(failed))
                busy_delta += wall
        finally:
            # B1: restore the pool's shared data to X -- without this,
            # every later pooled evaluation in this run (the loop's own
            # splits) would run T against the survey's `rows` instead of
            # the full data. Mirrors qij.py's own re-share after the
            # stage-1 survey (`pool.share(X)` before `run_joint`); done
            # in `finally` so it also happens if a task raised above.
            pool.share(X)

    # M3: `I_proto_old` is already centred (stage 1's own mass-weighted
    # centring, `xvq.prototype_influences`); re-applying that same
    # centring step to the COMBINED array here would subtract psi_bar
    # twice from every kept row. Un-centre the kept rows back to raw
    # values using the centring vector stage 1 stored on `sv_old`
    # (`sv_old.psi_bar`), so the whole array re-centred below is RAW,
    # exactly as `prototype_influences` centres once at the end on a raw
    # array -- never touching stage 1's own arithmetic itself.
    psi_bar_old = getattr(sv_old, 'psi_bar', None)
    if psi_bar_old is None:
        raise ValueError("resurvey: sv_old.psi_bar is missing; the kept prototypes' centring is unknown")
    for k in range(M_old):
        if k not in refined_js:
            I_proto_new[k] = I_proto_new[k] + psi_bar_old

    finite = np.isfinite(I_proto_new)
    mass = np.sum(np.where(finite, p_new[:, None], 0.0), axis=0)
    psi_bar = np.sum(np.where(finite, p_new[:, None] * I_proto_new, 0.0), axis=0) / mass
    I_proto_new -= psi_bar[None, :]

    ev1, rows1 = counter.snapshot()
    n_evals = ev1 - ev0
    n_rows = rows1 - rows0

    sv_new = SurveyRows(
        rows=rows, row_field=row_field, omega0=omega0, eta_Q=sv_old.eta_Q,
        step_ratio=np.full((5, q_full), np.nan),
        quantized_start=sv_old.quantized_start, eta_rows=eta_rows,
        psi_bar=psi_bar.copy(),  # M3: carried forward for a later pass's un-centring
    )

    return theta_Q_new, I_proto_new, sv_new, n_evals, n_rows, busy_delta
