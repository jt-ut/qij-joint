"""
`joint_mode='seeded'` (spec/QIJ_seeded_measured_tree_spec.md): the
second stage's other code path, dispatched once by `core.joint.
run_joint` before `'staged'`'s own growth call (`'staged'` is never
reached under this mode). A Zador-sized minimax k-means seed on the
pilot's own standardized influence (section 3), forward-measured in one
pool batch, then a measured round loop that grows and prunes on
realized gains alone (section 4) -- no flag reads the pilot after the
seed; `check_rule` is inert here (the loop below IS the check).

Two state arrays, `psi_hat`/`sd_hat` (section 2), are updated in place
by `_apply_update` (3.3) as every bin -- the seed's own and every
split's two children -- is measured; every other quantity (V_btw,
V_win_hat, the open test, a split's realized gain) is a function of
these two arrays and the current bin labels, recomputed fresh every
round, never accumulated across rounds (section 4, 'Centering').

Requires `pilot='gp'` (validated by `run_joint`, R2): `sd_hat` is
seeded from the pilot's own posterior sd (`sigma_all`, already measured
by `qij.py` before this module is ever reached), never recomputed here.
"""
from __future__ import annotations

import math
import time
from typing import Dict, List, Sequence, Tuple

import numpy as np

from ..parallel import call_T
from .differences import forward_step, perturbed_weights, step_parameter
from .joint import JointResult, _group_stats, _lloyd_all, _predicted_share, grow, two_means_split

__all__ = ["run_seeded"]

_C1 = 1.0 / 12.0               # Zador's 1-D scalar-quantizer constant
_C2 = 5.0 / (36.0 * math.sqrt(3.0))  # Zador's 2-D hexagonal-lattice constant
_REWEIGHT_CAP = 10             # spec 3.2: a convergence backstop, not a target


def _zador_k(lam: np.ndarray, d_eff: int, eps: float) -> int:
    """K_Z, the unclipped Zador count at per-coordinate distortion
    `eps` (spec 3.1): C_1 (d_eff=1) or C_2 (d_eff=2); d_eff > 2 is not
    built."""
    if d_eff == 1:
        C = _C1
    elif d_eff == 2:
        C = _C2
    else:
        raise ValueError(f"seed: d_eff={d_eff} > 2 is not built (spec 3.1)")
    d = float(d_eff)
    lam_prod = float(np.prod(lam[:d_eff]))
    inner = (C * 2.0 * math.pi * ((d + 2.0) / d) ** ((d + 2.0) / 2.0)
             * lam_prod ** (1.0 / d) / (d * eps))
    return int(math.ceil(inner ** (d / 2.0)))


def _seed_count(psi0_all: np.ndarray, std: np.ndarray, d_x: int, eps: float,
                 L_pilot: int, M_X_used: int) -> Tuple[int, int, np.ndarray, np.ndarray]:
    """K_Z, K, lambda (d_eff,), v1 (spec 3.1-3.2): d_eff = min(d_x, q);
    lambda the top d_eff eigenvalues of Psi_tilde's own q x q
    covariance, which IS the standardized pilot's correlation matrix
    (every column already has unit variance via `std`) -- one `eigh`
    serves both 3.1's eigenvalues and 3.2's first principal axis v1,
    sign fixed by its largest-magnitude loading positive. K = clip(K_Z,
    1, min(L_pilot, M_X_used)); K must not exceed the number of
    distinct Psi_tilde rows (assert, spec 3.1)."""
    N, q = psi0_all.shape
    psi_tilde = psi0_all / std
    centered = psi_tilde - psi_tilde.mean(axis=0)
    cov = np.atleast_2d((centered.T @ centered) / N)
    eigval, eigvec = np.linalg.eigh(cov)

    d_eff = min(d_x, q)
    lam = eigval[::-1][:d_eff]
    K_Z = _zador_k(lam, d_eff, eps)
    K = int(np.clip(K_Z, 1, min(L_pilot, M_X_used)))
    n_distinct = int(np.unique(psi_tilde, axis=0).shape[0])
    assert K <= n_distinct, "seed: K exceeds the distinct Psi_tilde rows (spec 3.1)"

    v1 = eigvec[:, -1].copy()
    v1 *= float(np.sign(v1[np.argmax(np.abs(v1))]))
    return K_Z, K, lam, v1


def _seed_partition(psi0_all: np.ndarray, std: np.ndarray, V_hat: np.ndarray,
                     v1: np.ndarray, K: int) -> Tuple[np.ndarray, int, int, np.ndarray]:
    """The minimax reweighted k-means seed partition (spec 3.2): init
    at the mid-quantiles of the distinct projections of Psi_tilde on
    `v1` (`ivq.kmeans_1d`'s own quantile/searchsorted init rule), then
    Lloyd on Psi_tilde*sqrt(omega) from the current labels
    (`core.joint._lloyd_all`), capped at `_REWEIGHT_CAP` passes,
    stopping (keeping the PREVIOUS labels) the first time max_c S_c
    (the pilot's predicted within share on the UNWEIGHTED Psi_tilde)
    fails to decrease. Returns (labels, L, passes, S_pred)."""
    psi_tilde = psi0_all / std
    proj = psi_tilde @ v1
    prototypes = np.quantile(np.unique(proj), (np.arange(K) + 0.5) / K)
    mids = (prototypes[:-1] + prototypes[1:]) / 2.0
    labels = np.searchsorted(mids, proj, side='left')
    L = K

    omega = np.ones(psi_tilde.shape[1])
    s_prev = None
    acc_labels, acc_L, acc_S = labels, L, None
    passes = 0
    for _ in range(_REWEIGHT_CAP):
        weighted = psi_tilde * np.sqrt(omega)
        new_labels, new_L = _lloyd_all(weighted, labels, L)
        passes += 1
        _, S = _predicted_share(psi0_all, V_hat, new_labels, new_L)
        s_max = float(S.max())
        if s_prev is not None and s_max >= s_prev:
            break
        acc_labels, acc_L, acc_S = new_labels, new_L, S
        s_prev = s_max
        labels, L = new_labels, new_L
        omega = omega * S / S.mean()
        omega = omega / omega.mean()
    return acc_labels, acc_L, passes, acc_S


def _eval_task(T, case, X: np.ndarray, task):
    """One seeded-mode forward evaluation (spec 3.3/section 4): `task`
    is (id, signed step t, member mask, start, eta_full); mirrors
    `ivq._bin_task`'s own failure boundary, forward only (one call, not
    a central pair). `eta_full` is passed as the per-call polish-
    tolerance override (spec/QIJ_mods_waves.md A15), as every other
    full-data stencil in this package does."""
    tid, t, mask, start, eta = task
    omega = perturbed_weights(np.ones(len(X)), mask, t)
    t0 = time.perf_counter()
    try:
        result = np.asarray(call_T(T, X, omega, start, eta), dtype=float)
        failed = bool(np.any(np.isnan(result)))
    except Exception:
        result = np.full(len(T.outputs), np.nan)
        failed = True
    return tid, result, failed, time.perf_counter() - t0


def _measure_batch(X: np.ndarray, counter, pool, tasks: List[tuple]):
    """Run `tasks` (id, t, mask, start, eta) as one pool batch, or
    serially with no pool (spec section 6): `pool.map` returns task-
    order results (`parallel.Pool.map`'s own contract), so results are
    read off by the id each task carries, never by completion order --
    bit-identical at any worker count (spec section 4, 'Bit identity
    across workers'). Returns (id -> (value, failed), busy_delta)."""
    if pool is None:
        out = {}
        for tid, t, mask, start, eta in tasks:
            val = np.asarray(counter(X, perturbed_weights(np.ones(len(X)), mask, t),
                                      start=start, eta=eta), dtype=float)
            out[tid] = (val, bool(np.any(np.isnan(val))))
        return out, 0.0
    t0 = time.perf_counter()
    raw = pool.map(_eval_task, tasks)
    busy = sum(r[3] for r in raw) - (time.perf_counter() - t0)
    out = {}
    for tid, val, failed, _wall in raw:
        counter.add(1, len(X), int(failed))
        out[tid] = (val, failed)
    return out, busy


def _apply_update(psi_hat: np.ndarray, sd_hat: np.ndarray, idx: np.ndarray, U: np.ndarray,
                   t_k: float, eta_full: float, theta_abs: np.ndarray,
                   n_scale: np.ndarray, n_shift: np.ndarray, n_neg: np.ndarray) -> None:
    """3.3's scale/shift update, one bin's points (spec 3.3, section 4
    'The update'): m/s are psi_hat's CURRENT mean/within-sd over `idx`
    (population variance), read fresh since psi_hat may already carry
    an earlier update at these same points. delta_U_c = eta_full*
    |theta_hat_c|/t_k is the measurement's own error (3.3); scale when
    |m_c| > s_c + delta_U_c, shift otherwise. After the update the bin
    mean of psi_hat equals U_c exactly, in both branches. A negative
    scale ratio is allowed and counted (`n_neg`, spec section 8)."""
    rows = psi_hat[idx]
    m = rows.mean(axis=0)
    s = np.sqrt(np.maximum(rows.var(axis=0), 0.0))
    delta_U = eta_full * theta_abs / t_k
    scale_mask = np.abs(m) > (s + delta_U)
    for c in range(m.size):
        if scale_mask[c]:
            ratio = float(U[c] / m[c])
            psi_hat[idx, c] *= ratio
            sd_hat[idx, c] *= abs(ratio)
            n_scale[c] += 1
            if ratio < 0.0:
                n_neg[c] += 1
        else:
            psi_hat[idx, c] += (U[c] - m[c])
            n_shift[c] += 1


def _centered_between(leaves: Dict[int, dict], ids: Sequence[int], N: int):
    """ubar_c and V_btw,c over `ids`' own stored (raw) U (spec section
    4, 'Centering'): the current mass-weighted mean and the
    between-bin variance about it, recomputed fresh from `ids`, never
    accumulated across rounds."""
    p = np.array([leaves[i]['n'] for i in ids], dtype=float) / N
    U = np.array([leaves[i]['U'] for i in ids])
    ubar = np.sum(p[:, None] * U, axis=0)
    v_btw = np.sum(p[:, None] * (U - ubar[None, :]) ** 2, axis=0) / N
    return ubar, v_btw


def _run_seed(X: np.ndarray, counter, theta_hat: np.ndarray, psi0_all: np.ndarray,
              sigma_all: np.ndarray, offset: np.ndarray, growth, eps: float,
              eta_full: float, M_X_used: int, pool, start, measured) -> dict:
    """Sections 3.1-3.3: the seed's count, partition, measurement and
    first update. Returns a dict of everything `_run_rounds`/
    `run_seeded` need; `state['failed']` is True when any seed bin's
    evaluation returned NaN (spec 3.3: fails the whole draw's stage, as
    a failed initial stencil does under `'staged'`)."""
    N, q = psi0_all.shape
    std = growth.std
    V_hat = std ** 2
    K_Z, K, lam, v1 = _seed_count(psi0_all, std, X.shape[1], eps, growth.L0, M_X_used)
    labels, L, passes, S_pred = _seed_partition(psi0_all, std, V_hat, v1, K)

    theta_abs = np.maximum(np.abs(theta_hat[measured]), np.finfo(float).eps)
    delta_f = forward_step(eta_full)
    p0 = np.bincount(labels, minlength=L).astype(float) / N
    t_bin = np.array([step_parameter(delta_f, float(pk)) for pk in p0])
    tasks = [(k, t_bin[k], labels == k, start, eta_full) for k in range(L)]
    results, busy = _measure_batch(X, counter, pool, tasks)

    state = dict(std=std, K_Z=K_Z, K=K, L=L, passes=passes, S_pred=S_pred, lam=lam,
                 evals_seed=L, busy=busy, failed=any(f for _v, f in results.values()))
    if state['failed']:
        return state

    U0 = np.array([(results[k][0][measured] - theta_hat[measured]) / t_bin[k]
                    for k in range(L)])
    cr = np.sum(p0[:, None] * U0, axis=0)
    v_btw0 = np.sum(p0[:, None] * (U0 - cr[None, :]) ** 2, axis=0)
    state['seed_centering_residual'] = cr / np.sqrt(np.where(v_btw0 > 0.0, v_btw0, np.nan))

    psi_hat = (psi0_all - offset[None, :]).copy()
    sd_hat = sigma_all.copy()
    n_scale, n_shift, n_neg = np.zeros(q, dtype=int), np.zeros(q, dtype=int), np.zeros(q, dtype=int)
    leaves: Dict[int, dict] = {}
    for k in range(L):
        idx = np.where(labels == k)[0]
        leaves[k] = dict(indices=idx, n=idx.size, U=U0[k], t=float(t_bin[k]))
        _apply_update(psi_hat, sd_hat, idx, U0[k], t_bin[k], eta_full, theta_abs,
                      n_scale, n_shift, n_neg)
    state.update(leaves=leaves, next_id=L, psi_hat=psi_hat, sd_hat=sd_hat,
                 n_scale=n_scale, n_shift=n_shift, n_neg=n_neg, theta_abs=theta_abs,
                 delta_f=delta_f)
    return state


def _round_candidates(leaves: Dict[int, dict], closed: set, psi_hat: np.ndarray,
                       N: int, eps: float):
    """One round's start (spec section 4, spec section 8 Q&A): V_btw/
    V_win_hat/V_tot_hat over EVERY current bin (open or permanently
    closed), tau_round fixed for the round, then the open test (growth's
    own rule, on `psi_hat`) restricted to bins not in `closed`.

    The open test's numerator is bin k's own contribution to V_win_hat,c
    -- (p_k Var_k(psi_hat_c))/N, not the bare p_k Var_k(psi_hat_c)
    section 4 writes -- so it is on V_tot_hat's own O(1/N) scale
    (section 8 Q&A: the denominator is V_tot_hat, NOT the pilot's own
    O(1) V_hat growth itself compares against). Taken fully literally,
    the written formula compares an O(1) numerator to an O(1/N)
    denominator against an O(eps) threshold and is open almost always,
    for every bin, which is not a stopping rule; dividing by N here
    matches growth's own structural form (a bin's share of the total
    exceeding eps/L) at this stage's own units. Returns (ids, ubar,
    v_btw, tau_round, L_current, candidates, p, var, v_tot)."""
    ids = sorted(leaves.keys())
    L_current = len(ids)
    ubar, v_btw = _centered_between(leaves, ids, N)
    groups = [leaves[i]['indices'] for i in ids]
    _means, var = _group_stats(psi_hat, groups)
    p = np.array([leaves[i]['n'] for i in ids], dtype=float) / N
    v_win = np.sum(p[:, None] * var, axis=0) / N
    v_tot = v_btw + v_win
    tau_round = eps * v_btw / L_current
    own_share = (p[:, None] * var) / (N * v_tot[None, :])
    open_mask = np.any(own_share > (eps / L_current), axis=1)
    candidates = [ids[i] for i in range(L_current)
                  if open_mask[i] and ids[i] not in closed]
    return ids, ubar, v_btw, tau_round, L_current, candidates, p, var, v_tot


def _propose_splits(leaves: Dict[int, dict], candidates, psi_hat: np.ndarray,
                     std: np.ndarray, N: int, delta_f: float, closed_infeasible: set):
    """One round's proposals (spec section 4, 'Proposal'): two-means on
    the bin's own CURRENT Psi_tilde = psi_hat/std_pilot rows
    (`core.joint.two_means_split`); infeasible (fewer than two distinct
    rows) closes the bin immediately, no evaluation spent."""
    proposals = {}
    for cid in candidates:
        idx = leaves[cid]['indices']
        split = two_means_split(psi_hat[idx] / std[None, :])
        if split is None:
            closed_infeasible.add(cid)
            continue
        idx_a, idx_b = idx[split[0]], idx[split[1]]
        idx_small, idx_large = (idx_a, idx_b) if idx_a.size <= idx_b.size else (idx_b, idx_a)
        p_small, p_large = idx_small.size / N, idx_large.size / N
        proposals[cid] = dict(
            idx_small=idx_small, idx_large=idx_large, p_small=p_small, p_large=p_large,
            t_small=step_parameter(delta_f, p_small), t_large=step_parameter(delta_f, p_large))
    return proposals


def _measure_proposals(X: np.ndarray, counter, pool, proposals: dict, start, eta_full, N: int):
    """One round's own evaluation batch (spec section 4, 'Measurement'):
    both children of every proposal, one pool batch (`id` = ('s'|'l',
    parent id))."""
    tasks = []
    for cid, meta in proposals.items():
        mask_s = np.zeros(N, dtype=bool)
        mask_s[meta['idx_small']] = True
        mask_l = np.zeros(N, dtype=bool)
        mask_l[meta['idx_large']] = True
        tasks.append((('s', cid), meta['t_small'], mask_s, start, eta_full))
        tasks.append((('l', cid), meta['t_large'], mask_l, start, eta_full))
    return _measure_batch(X, counter, pool, tasks)


def _process_split(cid: int, meta: dict, leaf: dict, results: dict, theta_hat: np.ndarray,
                    measured, ubar: np.ndarray, tau_round: np.ndarray, eta_full: float,
                    theta_abs: np.ndarray, N: int, next_id: int):
    """One proposal's realized gain and pay decision (spec section 4,
    'The per-split identity' and 'The realized gain and the decision'):
    Delta_c formed from the re-centred parent and children values
    (section 4, 'Centering'); the floor n_Delta+b_Delta uses
    THIS split's own t_small (spec section 4). Returns None on a failed
    evaluation (the split is cancelled, the parent stays, closed,
    "its evaluations counted" -- folded into `closed_infeasible` by the
    caller, spec section 4 'Measurement'; no separate counter is named
    in section 5 for this case) or (U_small, U_large, Delta, paid,
    delta_split, id_small, id_large)."""
    val_s, fail_s = results[('s', cid)]
    val_l, fail_l = results[('l', cid)]
    if fail_s or fail_l:
        return None
    U_small = (val_s[measured] - theta_hat[measured]) / meta['t_small']
    U_large = (val_l[measured] - theta_hat[measured]) / meta['t_large']
    p_parent, U_parent, t_parent = leaf['n'] / N, leaf['U'], leaf['t']
    delta_split = U_parent - (meta['p_small'] * U_small + meta['p_large'] * U_large) / p_parent
    Delta = (meta['p_small'] * (U_small - ubar) ** 2 + meta['p_large'] * (U_large - ubar) ** 2
             - p_parent * (U_parent - ubar) ** 2) / N
    delta_U = eta_full * theta_abs / meta['t_small']
    n_delta = (2.0 / N) * (meta['p_small'] * np.abs(U_small)
                           + meta['p_large'] * np.abs(U_large)) * delta_U
    b_delta = ((meta['p_small'] + meta['p_large']) / N) * delta_U ** 2
    tau_prime = np.maximum(tau_round, n_delta + b_delta)
    margin = Delta - tau_prime
    paid = bool(margin[int(np.argmax(margin))] >= 0.0)
    return dict(U_small=U_small, U_large=U_large, Delta=Delta, paid=paid,
                delta=delta_split, t_parent=t_parent, p_parent=p_parent)


def _run_rounds(state: dict, X: np.ndarray, counter, theta_hat: np.ndarray, measured,
                 eps: float, eta_full: float, M_X_used: int, pool, start) -> dict:
    """Section 4's round loop: until no bin is open or the cap binds
    (tested at the top of every round, spec 'The cap'). Mutates
    `state['psi_hat']`/`state['sd_hat']` in place via `_apply_update`;
    returns `state` with the loop's own products added."""
    leaves, next_id = state['leaves'], state['next_id']
    psi_hat, sd_hat = state['psi_hat'], state['sd_hat']
    n_scale, n_shift, n_neg = state['n_scale'], state['n_shift'], state['n_neg']
    theta_abs, delta_f, std = state['theta_abs'], state['delta_f'], state['std']
    N = psi_hat.shape[0]
    cap_tree = 1 + 2 * M_X_used

    closed_unpaid: set = set()
    closed_infeasible: set = set()
    n_open_per_round: List[int] = []
    split_rows: List[dict] = []
    n_splits = 0
    n_rounds = 0
    tree_capped = False
    busy = 0.0

    while True:
        if state['evals_seed'] + 2 * n_splits >= cap_tree:
            tree_capped = True
            break
        ids, ubar, v_btw, tau_round, L_current, candidates, p, var, v_tot = _round_candidates(
            leaves, closed_unpaid | closed_infeasible, psi_hat, N, eps)
        if not candidates:
            break
        n_rounds += 1
        n_open_per_round.append(len(candidates))

        proposals = _propose_splits(leaves, candidates, psi_hat, std, N, delta_f, closed_infeasible)
        if not proposals:
            continue
        results, busy_r = _measure_proposals(X, counter, pool, proposals, start, eta_full, N)
        busy += busy_r
        n_splits += len(proposals)

        for cid, meta in proposals.items():
            leaf = leaves[cid]
            outcome = _process_split(cid, meta, leaf, results, theta_hat, measured, ubar,
                                      tau_round, eta_full, theta_abs, N, next_id)
            if outcome is None:
                closed_infeasible.add(cid)
                continue
            del leaves[cid]
            id_small, id_large = next_id, next_id + 1
            next_id += 2
            leaves[id_small] = dict(indices=meta['idx_small'], n=meta['idx_small'].size,
                                     U=outcome['U_small'], t=meta['t_small'])
            leaves[id_large] = dict(indices=meta['idx_large'], n=meta['idx_large'].size,
                                     U=outcome['U_large'], t=meta['t_large'])
            _apply_update(psi_hat, sd_hat, meta['idx_small'], outcome['U_small'], meta['t_small'],
                          eta_full, theta_abs, n_scale, n_shift, n_neg)
            _apply_update(psi_hat, sd_hat, meta['idx_large'], outcome['U_large'], meta['t_large'],
                          eta_full, theta_abs, n_scale, n_shift, n_neg)
            if not outcome['paid']:
                closed_unpaid.add(id_small)
                closed_unpaid.add(id_large)
            split_rows.append(dict(
                split_id=len(split_rows), parent_mass=outcome['p_parent'],
                mass_small=meta['p_small'], mass_large=meta['p_large'],
                t_parent=outcome['t_parent'], t_small=meta['t_small'], t_large=meta['t_large'],
                delta=outcome['delta']))

    state.update(leaves=leaves, closed_unpaid=closed_unpaid, closed_infeasible=closed_infeasible,
                 n_open_per_round=np.array(n_open_per_round, dtype=int), split_rows=split_rows,
                 n_splits=n_splits, n_rounds=n_rounds, tree_capped=tree_capped,
                 busy=state['busy'] + busy)
    return state


def _split_noise_arrays(rows: List[dict], q: int) -> Dict[str, np.ndarray]:
    """`split_noise` (spec section 4/5): one row per split, assembled
    once at termination from the per-split records the round loop
    collected."""
    n = len(rows)
    return dict(
        split_id=np.arange(n, dtype=int),
        parent_mass=np.array([r['parent_mass'] for r in rows], dtype=float),
        mass_small=np.array([r['mass_small'] for r in rows], dtype=float),
        mass_large=np.array([r['mass_large'] for r in rows], dtype=float),
        t_parent=np.array([r['t_parent'] for r in rows], dtype=float),
        t_small=np.array([r['t_small'] for r in rows], dtype=float),
        t_large=np.array([r['t_large'] for r in rows], dtype=float),
        delta=(np.array([r['delta'] for r in rows], dtype=float) if n else np.zeros((0, q))),
    )


def _seeded_failed(N: int, q: int, q_full: int, growth, state: dict) -> JointResult:
    """A failed seed measurement (spec 3.3): the whole draw's joint
    stage fails, as a failed initial stencil does under `'staged'`
    (`core.joint._failed_result`); growth's and the seed's own (free,
    no-evaluation) products are kept. Every `joint_mode='seeded'`-only
    field is given its own (q,)-shaped neutral value here (never
    `JointResult`'s own shape-(0,) class default), so a caller reading
    this mode never sees a shape that depends on whether the draw
    failed."""
    nan_q = np.full(q, np.nan)
    lam_padded = np.full(2, np.nan)
    lam_padded[:state['lam'].size] = state['lam']
    return JointResult(
        V_btw=nan_q, V_win_hat=nan_q, V_tot_hat=nan_q, S_pred=growth.S_pred,
        a=nan_q, gain_ratio=nan_q, B_hat=np.full(q_full, np.nan), a_bca=np.full(q_full, np.nan),
        L0=growth.L0, L=state['L'], n_growth_rounds=growth.n_growth_rounds,
        growth_capped=growth.growth_capped, S_pred_pre_lloyd=growth.S_pred_pre_lloyd,
        n_flagged=0, n_check_rounds=0, n_check_evals=state['evals_seed'], n_level_splits=0,
        n_adjacency_splits=0, check_capped=False, n_closed_unpaid=0,
        n_closed_unflagged=0, n_noise_floored=0, sum_b_delta=nan_q, failed=True,
        bin_mass=np.zeros(0), bin_U=np.zeros((0, q)), bin_m=np.zeros((0, q)),
        bin_flagged=np.zeros(0, dtype=bool), bin_label=np.full(N, -1, dtype=int),
        busy_delta=state['busy'],
        psi_hat=np.full((N, q), np.nan), sd_hat=np.full((N, q), np.nan),
        seed_K_zador=state['K_Z'], seed_K=state['K'], seed_L=state['L'],
        seed_reweight_passes=state['passes'], seed_S_pred=state['S_pred'],
        seed_centering_residual=nan_q, seed_lambda=lam_padded, evals_seed=state['evals_seed'],
        evals_tree=0, n_closed_infeasible=0,
        n_update_scale=np.zeros(q, dtype=int), n_update_shift=np.zeros(q, dtype=int),
        n_update_negative_scale=np.zeros(q, dtype=int), tree_centering_drift=nan_q,
        n_closed_hot=0, V_win_closed=nan_q, n_open_per_round=np.zeros(0, dtype=int),
        split_noise=dict(),
    )


def _terminate(state: dict, psi0_all: np.ndarray, growth, eps: float, q_full: int) -> JointResult:
    """Section 5: the final `JointResult`, from `_run_rounds`'s own
    final `leaves`/closed sets and the updated `psi_hat`/`sd_hat` --
    V_btw/V_win_hat/V_tot_hat re-centred and recomputed one last time
    (never carried from the last round), `tree_centering_drift`,
    `n_closed_hot`/`V_win_closed` (one-miss-closed bins still over
    eps/L here), and `bin_label`/`bin_m`."""
    N, q = psi0_all.shape
    leaves = state['leaves']
    psi_hat, sd_hat = state['psi_hat'], state['sd_hat']
    closed_unpaid = state['closed_unpaid']
    ids = sorted(leaves.keys())
    L_final = len(ids)
    p = np.array([leaves[i]['n'] for i in ids], dtype=float) / N
    U = np.array([leaves[i]['U'] for i in ids])
    ubar = np.sum(p[:, None] * U, axis=0)
    V_btw = np.sum(p[:, None] * (U - ubar[None, :]) ** 2, axis=0) / N
    groups = [leaves[i]['indices'] for i in ids]
    _means, var = _group_stats(psi_hat, groups)
    V_win = np.sum(p[:, None] * var, axis=0) / N
    V_tot = V_btw + V_win
    drift = ubar / np.sqrt(np.where(V_btw > 0.0, V_btw, np.nan))

    bin_label = np.empty(N, dtype=int)
    for new_id, old_id in enumerate(ids):
        bin_label[leaves[old_id]['indices']] = new_id
    bin_m, _ = _group_stats(psi0_all, groups)

    # Bin k's own contribution to V_win_hat,c, on V_tot's own O(1/N)
    # scale (the `_round_candidates` open test's same correction, same
    # reasoning there).
    share = (p[:, None] * var) / (N * V_tot[None, :])
    bin_flagged = np.any(share > eps / L_final, axis=1)
    hot_mask = np.array([old_id in closed_unpaid for old_id in ids], dtype=bool) & bin_flagged
    n_closed_hot = int(hot_mask.sum())
    V_win_closed = np.sum(share[hot_mask], axis=0) if n_closed_hot else np.zeros(q)

    lam_padded = np.full(2, np.nan)
    lam_padded[:state['lam'].size] = state['lam']

    return JointResult(
        V_btw=V_btw, V_win_hat=V_win, V_tot_hat=V_tot, S_pred=growth.S_pred,
        a=np.full(q, np.nan), gain_ratio=np.full(q, np.nan),
        B_hat=np.full(q_full, np.nan), a_bca=np.full(q_full, np.nan),
        L0=growth.L0, L=L_final, n_growth_rounds=growth.n_growth_rounds,
        growth_capped=growth.growth_capped, S_pred_pre_lloyd=growth.S_pred_pre_lloyd,
        n_flagged=0, n_check_rounds=state['n_rounds'],
        n_check_evals=state['evals_seed'] + 2 * state['n_splits'],
        n_level_splits=state['n_splits'], n_adjacency_splits=0, check_capped=state['tree_capped'],
        n_closed_unpaid=len(closed_unpaid), n_closed_unflagged=0, n_noise_floored=0,
        sum_b_delta=np.full(q, np.nan), failed=False,
        bin_mass=p, bin_U=U, bin_m=bin_m, bin_flagged=bin_flagged, bin_label=bin_label,
        busy_delta=state['busy'],
        psi_hat=psi_hat, sd_hat=sd_hat,
        seed_K_zador=state['K_Z'], seed_K=state['K'], seed_L=state['L'],
        seed_reweight_passes=state['passes'], seed_S_pred=state['S_pred'],
        seed_centering_residual=state['seed_centering_residual'], seed_lambda=lam_padded,
        evals_seed=state['evals_seed'], evals_tree=2 * state['n_splits'],
        n_closed_infeasible=len(state['closed_infeasible']),
        n_update_scale=state['n_scale'], n_update_shift=state['n_shift'],
        n_update_negative_scale=state['n_neg'], tree_centering_drift=drift,
        n_closed_hot=n_closed_hot, V_win_closed=V_win_closed,
        n_open_per_round=state['n_open_per_round'],
        split_noise=_split_noise_arrays(state['split_rows'], q),
    )


def run_seeded(X: np.ndarray, counter, theta_hat: np.ndarray, psi0_all: np.ndarray,
               sigma_all: np.ndarray, offset: np.ndarray, xvq, eta_full: float, eps: float,
               pool, start, measured: Sequence[int]) -> JointResult:
    """`joint_mode='seeded'` (spec/QIJ_seeded_measured_tree_spec.md):
    the seed (3.1-3.3, `_run_seed`), the measured round loop (section 4,
    `_run_rounds`), then termination's reported quantities (section 5,
    `_terminate`). `run_joint` dispatches here once, before `'staged'`'s
    own growth call, so `'staged'` is never reached under this mode;
    requires `pilot='gp'` (validated by the caller -- `sd_hat` is seeded
    from `sigma_all`, the pilot's own posterior sd, never recomputed
    here). Every `'staged'`-only field (gain_ratio, the check's scale
    `a`, `B_hat`/`a_bca`, the flag counters) is NaN/0, per section 5."""
    N, q = psi0_all.shape
    q_full = theta_hat.size
    M_X_used = xvq.M_used
    growth = grow(psi0_all, eps, M_X_used)

    state = _run_seed(X, counter, theta_hat, psi0_all, sigma_all, offset, growth, eps,
                       eta_full, M_X_used, pool, start, measured)
    if state['failed']:
        return _seeded_failed(N, q, q_full, growth, state)

    state = _run_rounds(state, X, counter, theta_hat, measured, eps, eta_full, M_X_used,
                        pool, start)
    return _terminate(state, psi0_all, growth, eps, q_full)
