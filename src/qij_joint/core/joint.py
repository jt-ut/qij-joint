"""
The joint second stage, `ivqbins='joint'` (spec/method_notes.md section
6): one partition grown on psi0_all to the tolerance (`grow`), measured
on the full data through `ivq.bin_differences`, then a measured check
that splits a bin where its measurement contradicts the model
(`run_joint`). Growth runs in-process numpy only; the initial
measurement and every check round's evaluations run through `pool`.

`I_proto` (M_X_used, q) is the prototype influence surveyed in stage 1;
the check's adjacency split reads it (`refine.adjacency_split_gain`) to
score a split along a prototype's own gradient direction. `QIJ.fit`
passes it in on every call.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple

import numpy as np

from ..parallel import prepared
from .differences import forward_step, perturbed_weights, step_parameter
from .influence_model import bin_posterior_variance
from .ivq import BinSet, between_terms, bin_differences
from .refine import adjacency_gain_value, adjacency_split_gain, level_gain_value

__all__ = ["Growth", "JointResult", "grow", "run_joint", "two_means_split"]

_CHUNK = 2048  # row chunk for the full-partition Lloyd reassignment (E4)


@dataclass
class Growth:
    """The shared partition built by growth, before any evaluation.
    labels (N,) bin index per point, contiguous 0..L0-1; L0 the bin
    count after growth (and the Lloyd pass, if kept); std (q,) =
    sqrt(V_hat), so that Ψ̃ = psi0_all/std throughout; S_pred,
    S_pred_pre_lloyd (q,) the predicted within share after / before the
    optional Lloyd pass (S_pred_pre_lloyd all NaN when the pass was not
    run)."""

    labels: np.ndarray
    L0: int
    n_growth_rounds: int
    growth_capped: bool
    S_pred: np.ndarray
    S_pred_pre_lloyd: np.ndarray
    std: np.ndarray


@dataclass
class JointResult:
    """The joint second stage's complete result (spec/method_notes.md
    section 6). Per output (q,): V_btw, V_win_hat, V_tot_hat, S_pred, a,
    gain_ratio. Shared: L0, L (final), n_growth_rounds, growth_capped,
    S_pred_pre_lloyd (q,), n_flagged (bins flagged in the first check),
    n_check_rounds, n_check_evals, n_level_splits, n_adjacency_splits,
    check_capped, failed (a failed output, or a failed initial
    measurement before any check ran, voids every output). Bin
    constituents: bin_mass (L,), bin_U (L,q), bin_m (L,q, predicted
    means), bin_flagged (L,); bin_label (N,); busy_delta as the other
    pool stages report it."""

    V_btw: np.ndarray
    V_win_hat: np.ndarray
    V_tot_hat: np.ndarray
    S_pred: np.ndarray
    a: np.ndarray
    gain_ratio: np.ndarray
    L0: int
    L: int
    n_growth_rounds: int
    growth_capped: bool
    S_pred_pre_lloyd: np.ndarray
    n_flagged: int
    n_check_rounds: int
    n_check_evals: int
    n_level_splits: int
    n_adjacency_splits: int
    check_capped: bool
    failed: bool
    bin_mass: np.ndarray
    bin_U: np.ndarray
    bin_m: np.ndarray
    bin_flagged: np.ndarray
    bin_label: np.ndarray
    busy_delta: float


def two_means_split(rows: np.ndarray) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Deterministic two-means on `rows` (n, q), the split rule shared
    by growth and the measured check (spec/method_notes.md section 6):
    initial centroids are the means of the two halves of `rows`
    split at the median of the projection onto the first principal axis
    (an eigh of the rows' own q x q covariance, q the output count,
    never a prototype-sized array), then Lloyd iterations (nearest
    centroid, ties to the lower one) until assignments stop changing or
    100 iterations. None if `rows` holds fewer than two distinct rows,
    or a Lloyd update would empty one side. Returns local (idx_a,
    idx_b) into `rows`' own 0..n-1."""
    n = rows.shape[0]
    if np.unique(rows, axis=0).shape[0] < 2:
        return None
    centered = rows - rows.mean(axis=0)
    _, eigvecs = np.linalg.eigh(centered.T @ centered)
    proj = centered @ eigvecs[:, -1]
    order = np.argsort(proj, kind='stable')
    half = n // 2
    labels = np.zeros(n, dtype=int)
    labels[order[half:]] = 1
    c0 = rows[order[:half]].mean(axis=0)
    c1 = rows[order[half:]].mean(axis=0)
    for _ in range(100):
        d0 = np.sum((rows - c0) ** 2, axis=1)
        d1 = np.sum((rows - c1) ** 2, axis=1)
        new_labels = (d1 < d0).astype(int)
        if np.array_equal(new_labels, labels):
            break
        labels = new_labels
        m0, m1 = labels == 0, labels == 1
        if not m0.any() or not m1.any():
            return None
        c0 = rows[m0].mean(axis=0)
        c1 = rows[m1].mean(axis=0)
    return np.where(labels == 0)[0], np.where(labels == 1)[0]


def _bin_stats(psi0_all: np.ndarray, labels: np.ndarray, L: int):
    """Per-bin count, mean and population variance of every output over
    the FULL labelling, via two `bincount` passes per output (E2): the
    loop is over q (outputs), never over bins or points."""
    q = psi0_all.shape[1]
    counts = np.bincount(labels, minlength=L).astype(float)
    means = np.empty((L, q))
    var = np.empty((L, q))
    for c in range(q):
        col = psi0_all[:, c]
        s1 = np.bincount(labels, weights=col, minlength=L)
        s2 = np.bincount(labels, weights=col ** 2, minlength=L)
        means[:, c] = s1 / counts
        var[:, c] = np.maximum(s2 / counts - means[:, c] ** 2, 0.0)
    return counts, means, var


def _group_stats(psi0_all: np.ndarray, groups: Sequence[np.ndarray]):
    """Per-group mean and population variance of every output, one
    group at a time (a check round's own bounded new-leaf count, E1):
    no full relabelling is built for a handful of new bins."""
    q = psi0_all.shape[1]
    means = np.empty((len(groups), q))
    var = np.empty((len(groups), q))
    for i, idx in enumerate(groups):
        rows = psi0_all[idx]
        means[i] = rows.mean(axis=0)
        var[i] = np.maximum(rows.var(axis=0), 0.0)
    return means, var


def _predicted_share(psi0_all: np.ndarray, V_hat: np.ndarray, labels: np.ndarray, L: int):
    """A growth round's predicted within share: w_kc = p_k*Var_k(psi0_c)/V_hat_c
    per bin and output, and S_c = sum_k w_kc, the total predicted within
    share for output c."""
    N = psi0_all.shape[0]
    counts, _, var = _bin_stats(psi0_all, labels, L)
    w = (counts[:, None] / N) * var / V_hat[None, :]
    return w, w.sum(axis=0)


def _split_growth_bins(psi_tilde: np.ndarray, labels: np.ndarray, to_split: np.ndarray, L: int):
    """Apply one growth round: every bin in `to_split` is replaced by
    `two_means_split` on its own standardized rows; a bin whose split is
    infeasible is kept whole. Relabelled contiguously in bin order."""
    to_split = set(to_split.tolist())
    new_labels = np.empty_like(labels)
    next_id = 0
    for k in range(L):
        idx = np.where(labels == k)[0]
        split = two_means_split(psi_tilde[idx]) if k in to_split else None
        if split is None:
            new_labels[idx] = next_id
            next_id += 1
        else:
            idx_a, idx_b = split
            new_labels[idx[idx_a]] = next_id
            new_labels[idx[idx_b]] = next_id + 1
            next_id += 2
    return new_labels, next_id


def _reassign(rows: np.ndarray, centroids: np.ndarray) -> np.ndarray:
    """Nearest centroid per row, chunked over N (E4: an N x L array, L
    the bin count, is exactly the prototype-sized shape E4 forbids);
    ties go to the lower centroid id (`argmin`'s own rule)."""
    N = rows.shape[0]
    csq = np.sum(centroids ** 2, axis=1)
    out = np.empty(N, dtype=int)
    for start in range(0, N, _CHUNK):
        sl = slice(start, start + _CHUNK)
        block = rows[sl]
        d2 = np.sum(block ** 2, axis=1)[:, None] + csq[None, :] - 2.0 * block @ centroids.T
        out[sl] = np.argmin(d2, axis=1)
    return out


def _lloyd_all(psi_tilde: np.ndarray, labels: np.ndarray, L: int):
    """One full Lloyd run of all L centroids over every row, growth's
    optional refinement pass: centroid = per-bin mean of psi_tilde, reassignment
    by `_reassign`, repeated until assignments stop changing or 100
    iterations; an emptied centroid is dropped and the rest relabelled
    (as `ivq.kmeans_1d`'s own empty-bin handling)."""
    q = psi_tilde.shape[1]
    labels = labels.copy()
    for _ in range(100):
        counts = np.bincount(labels, minlength=L).astype(float)
        centroids = np.empty((L, q))
        for c in range(q):
            centroids[:, c] = np.bincount(labels, weights=psi_tilde[:, c], minlength=L) / counts
        new_labels = _reassign(psi_tilde, centroids)
        counts = np.bincount(new_labels, minlength=L)
        if not counts.all():
            present = np.flatnonzero(counts)
            remap = np.full(L, -1, dtype=int)
            remap[present] = np.arange(present.size)
            new_labels = remap[new_labels]
            L = present.size
        converged = new_labels.shape == labels.shape and np.array_equal(new_labels, labels)
        labels = new_labels
        if converged:
            break
    return labels, L


def grow(psi0_all: np.ndarray, eps: float, M_X_used: int, lloyd: bool = False) -> Growth:
    """
    Grow one shared partition of the N points from a single bin to the
    tolerance (spec/method_notes.md section 6), no evaluations. A round splits,
    by `two_means_split` on Ψ̃ = psi0_all/std, every bin with
    max_c w_kc > eps/L; stops when every output's predicted within
    share S_c <= eps, or when L reaches M_X_used (`growth_capped`,
    checked only once tolerance is confirmed unmet, so a round that
    both converges and reaches the cap counts as converged). `lloyd`
    runs one extra Lloyd pass of all L centroids over every row
    afterward; S_pred is taken after that pass, S_pred_pre_lloyd before.
    """
    N, q = psi0_all.shape
    std = np.std(psi0_all, axis=0)
    V_hat = std ** 2
    psi_tilde = psi0_all / std
    labels = np.zeros(N, dtype=int)
    L = 1
    n_rounds = 0
    growth_capped = False

    while True:
        w, S = _predicted_share(psi0_all, V_hat, labels, L)
        if np.all(S <= eps):
            break
        if L >= M_X_used:
            growth_capped = True
            break
        to_split = np.where(w.max(axis=1) > eps / L)[0]
        if to_split.size == 0:
            break
        room = M_X_used - L  # each split adds one bin (1 -> 2)
        if to_split.size >= room:
            # Cap (B2 step 5): every flagged split is feasible (w_kc > 0
            # implies >= 2 distinct rows), so the `room` largest max_c
            # w_kc bins fill to M_X_used exactly, deterministically.
            order = np.argsort(-w[to_split].max(axis=1), kind='stable')
            to_split = to_split[order[:room]]
            growth_capped = True
        labels, L = _split_growth_bins(psi_tilde, labels, to_split, L)
        n_rounds += 1
        if growth_capped:
            break

    if lloyd:
        S_pre = S
        labels, L = _lloyd_all(psi_tilde, labels, L)
        _, S = _predicted_share(psi0_all, V_hat, labels, L)
    else:
        S_pre = np.full(q, np.nan)

    return Growth(labels=labels, L0=L, n_growth_rounds=n_rounds, growth_capped=growth_capped,
                  S_pred=S, S_pred_pre_lloyd=S_pre, std=std)


def _posterior_vu(model, Z: np.ndarray, sigma_all: np.ndarray, groups: Sequence[np.ndarray]):
    """v_kc, u_kc for every (bin, output), one `bin_posterior_variance`
    call per coordinate over all of `groups` at once (spec section 3)."""
    q = sigma_all.shape[1]
    v = np.empty((len(groups), q))
    u = np.empty((len(groups), q))
    for c in range(q):
        v[:, c], u[:, c] = bin_posterior_variance(model, Z, c, groups, sigma_all[:, c], with_mean=True)
    return v, u


def _fit_scale(p: np.ndarray, U: np.ndarray, m: np.ndarray) -> np.ndarray:
    """a_c = sum_k p_k*U_kc*m_kc / sum_k p_k*m_kc^2: the least-squares
    scale of the measured derivative on the predicted mean, fitted once
    and held fixed through the check so a uniform scale error in the
    model does not itself register as disagreement."""
    num = np.sum(p[:, None] * U * m, axis=0)
    den = np.sum(p[:, None] * m ** 2, axis=0)
    return num / den


def _flag_mask(p, U, m, a, V_hat, u, L: int, eps: float) -> np.ndarray:
    """Bin k is flagged when for some output c the rescaled measured
    derivative disagrees with the predicted mean by more than the bin's
    share of the tolerance plus what the posterior allows for the error
    of a bin mean: p_k*(U_kc-a_c*m_kc)^2 > eps*V_hat_c/L + p_k*a_c^2*u_kc."""
    lhs = p[:, None] * (U - a[None, :] * m) ** 2
    rhs = (eps * V_hat[None, :] / L) + p[:, None] * (a[None, :] ** 2) * u
    return np.any(lhs > rhs, axis=1)


def _split_task(T, case, X: np.ndarray, task):
    """One check-round split's one-shot forward evaluation, the
    same failure boundary as `ivq._bin_task`: `task` is (split id,
    signed step t, member mask); returns (id, evaluation, failure flag,
    this call's own wall time)."""
    sid, t, mask = task
    prep = prepared(T, X)
    omega = perturbed_weights(np.ones(len(X)), mask, t)
    t0 = time.perf_counter()
    try:
        result = T(X, omega, prep=prep) if prep is not None else T(X, omega)
        result = np.asarray(result, dtype=float)
        failed = bool(np.any(np.isnan(result)))
    except Exception:
        result = np.full(len(T.outputs), np.nan)
        failed = True
    return sid, result, failed, time.perf_counter() - t0


def _failed_result(N: int, q: int, growth: Optional[Growth] = None, busy_delta: float = 0.0) -> JointResult:
    """A failed draw: either the influence model's shared constant-path
    rule fails an output before growth ever runs, or the bin stencil's
    initial full-data measurement fails after a valid growth. Every
    variance and check quantity is void; growth's own products (needing
    no evaluation) are kept when growth ran."""
    nan_q = np.full(q, np.nan)
    if growth is None:
        L0, labels = 0, np.full(N, -1, dtype=int)
        n_rounds, capped, S_pred, S_pre = 0, False, nan_q, nan_q
        mass = np.zeros(0)
    else:
        L0, labels = growth.L0, growth.labels
        n_rounds, capped = growth.n_growth_rounds, growth.growth_capped
        S_pred, S_pre = growth.S_pred, growth.S_pred_pre_lloyd
        mass = np.bincount(labels, minlength=L0).astype(float) / N
    return JointResult(
        V_btw=nan_q, V_win_hat=nan_q, V_tot_hat=nan_q, S_pred=S_pred, a=nan_q, gain_ratio=nan_q,
        L0=L0, L=L0, n_growth_rounds=n_rounds, growth_capped=capped, S_pred_pre_lloyd=S_pre,
        n_flagged=0, n_check_rounds=0, n_check_evals=0, n_level_splits=0, n_adjacency_splits=0,
        check_capped=False, failed=True,
        bin_mass=mass, bin_U=np.full((L0, q), np.nan), bin_m=np.full((L0, q), np.nan),
        bin_flagged=np.zeros(L0, dtype=bool), bin_label=labels, busy_delta=busy_delta,
    )


def run_joint(
    X: np.ndarray, counter, theta_hat: np.ndarray, psi0_all: np.ndarray,
    sigma_all: np.ndarray, model, Z: np.ndarray, xvq, eta: float, eps: float,
    pool=None, lloyd: bool = False, I_proto: Optional[np.ndarray] = None,
) -> JointResult:
    """
    The joint second stage (spec/method_notes.md section 6): grow a
    shared partition (`grow`), measure every bin through
    `ivq.bin_differences`, then a measured check that splits a flagged
    bin, all a round's evaluations on `pool`. `I_proto` (M_X_used, q),
    the prototype influence from stage 1, feeds the check's adjacency
    split.
    """
    N, q = psi0_all.shape
    M_X_used = xvq.M_used

    if np.any(model.constant_path):
        return _failed_result(N, q)

    growth = grow(psi0_all, eps, M_X_used, lloyd=lloyd)
    L0 = growth.L0
    binset0 = BinSet(labels=growth.labels, n=np.bincount(growth.labels, minlength=L0),
                      U=np.zeros((L0, 0)), centering_residual=np.zeros(0),
                      M_init=L0, M_used=L0, within_share=0.0, failed=False)
    bins, busy_delta = bin_differences(X, counter, theta_hat, binset0, eta, pool)
    if bins.failed:
        return _failed_result(N, q, growth=growth, busy_delta=busy_delta)

    V_btw = np.array([between_terms(bins, c) for c in range(q)])
    V_hat = growth.std ** 2
    psi_tilde = psi0_all / growth.std
    _, m0, var0 = _bin_stats(psi0_all, bins.labels, L0)
    a = _fit_scale(bins.p, bins.U, m0)
    groups0 = [np.where(bins.labels == k)[0] for k in range(L0)]
    v0, u0 = _posterior_vu(model, Z, sigma_all, groups0)
    psi_centered = psi0_all - model.offset[None, :]

    leaves: Dict[int, dict] = {
        k: dict(indices=groups0[k], n=int(bins.n[k]), U=bins.U[k].copy(), m=m0[k].copy(),
                var=var0[k].copy(), v=v0[k].copy(), ubar=psi_centered[groups0[k]].mean(axis=0))
        for k in range(L0)
    }

    flagged_mask = _flag_mask(bins.p, bins.U, m0, a, V_hat, u0, L0, eps)
    n_flagged = int(flagged_mask.sum())
    current_flagged = set(np.where(flagged_mask)[0].tolist())

    n_check_rounds = n_check_evals = n_level_splits = n_adjacency_splits = 0
    sum_delta, sum_g = np.zeros(q), np.zeros(q)
    evals_cap = 1 + M_X_used
    check_capped = False
    delta_f = forward_step(eta)
    next_id = L0

    def decide_split(leaf):
        """The multi-output split-kind rule for a flagged bin: level
        when sum_c Var_k(psi0_c)/V_btw_c >= sum_c v_kc/V_btw_c, else
        adjacency; an infeasible kind falls back to the other."""
        idx = leaf['indices']
        p_k = leaf['n'] / N
        lhs = float(np.sum(leaf['var'] / V_btw))
        rhs = float(np.sum(leaf['v'] / V_btw))
        for kind in (('level', 'adjacency') if lhs >= rhs else ('adjacency', 'level')):
            if kind == 'level':
                split = two_means_split(psi_tilde[idx])
                if split is not None:
                    return kind, idx[split[0]], idx[split[1]]
            elif I_proto is not None:
                best, best_score = None, -np.inf
                for c in range(q):
                    result = adjacency_split_gain(idx, I_proto[:, c], xvq.bmu, xvq.bmu2,
                                                   N, p_k, float(leaf['v'][c]), 1.0)
                    if result is not None and result[2] / V_btw[c] > best_score:
                        best, best_score = (result[0], result[1]), result[2] / V_btw[c]
                if best is not None:
                    return (kind,) + best
        return None

    while current_flagged:
        if n_check_evals >= evals_cap:
            check_capped = True
            break
        n_check_rounds += 1
        proposals = {}
        for k in sorted(current_flagged):
            decided = decide_split(leaves[k])
            if decided is None:
                continue  # both kinds infeasible: the bin stays and is closed
            kind, idx_a, idx_b = decided
            p_k = leaves[k]['n'] / N
            if kind == 'adjacency':
                g = adjacency_gain_value(p_k, leaves[k]['v'], N, 1.0)
            else:
                ubar_a, ubar_b = psi_centered[idx_a].mean(axis=0), psi_centered[idx_b].mean(axis=0)
                g = level_gain_value(idx_a.size / N, ubar_a, idx_b.size / N, ubar_b,
                                      p_k, leaves[k]['ubar'], N, 1.0)
            idx_small, idx_large = (idx_a, idx_b) if idx_a.size <= idx_b.size else (idx_b, idx_a)
            p_small = idx_small.size / N
            p_large = idx_large.size / N
            t_small = step_parameter(delta_f, p_small)
            mask = np.zeros(N, dtype=bool)
            mask[idx_small] = True
            proposals[k] = dict(kind=kind, idx_small=idx_small, idx_large=idx_large,
                                 p_small=p_small, p_large=p_large, t_small=t_small,
                                 mask=mask, g=g)
        current_flagged.clear()
        if not proposals:
            continue

        tasks = [(k, meta['t_small'], meta['mask']) for k, meta in proposals.items()]
        if pool is None:
            evaluated = []
            for k, t, mask in tasks:
                val = np.asarray(counter(X, perturbed_weights(np.ones(N), mask, t)), dtype=float)
                evaluated.append((k, val, bool(np.any(np.isnan(val)))))
        else:
            t_map0 = time.perf_counter()
            raw = pool.map(_split_task, tasks)
            busy_delta += sum(r[3] for r in raw) - (time.perf_counter() - t_map0)
            evaluated = [(k, val, failed) for k, val, failed, _ in raw]
            for _, _, failed in evaluated:
                counter.add(1, N, int(failed))
        n_check_evals += len(tasks)

        new_leaves = []
        for k, val, failed in evaluated:
            if failed:
                continue  # cancelled: parent stays as a final bin, closed, counted above
            meta = proposals[k]
            leaf = leaves.pop(k)
            p_parent = leaf['n'] / N
            U_small = (val - theta_hat) / meta['t_small'] - bins.centering_residual
            U_large = (p_parent * leaf['U'] - meta['p_small'] * U_small) / meta['p_large']
            # /N: V_btw,c = (1/N) sum_k p_k U_kc^2 (ivq.between_terms).
            Delta = (meta['p_small'] * U_small ** 2 + meta['p_large'] * U_large ** 2
                     - p_parent * leaf['U'] ** 2) / N
            V_btw += Delta
            sum_delta += Delta
            sum_g += meta['g']
            if meta['kind'] == 'level':
                n_level_splits += 1
            else:
                n_adjacency_splits += 1
            new_leaves.append((next_id, meta['idx_small'], U_small))
            new_leaves.append((next_id + 1, meta['idx_large'], U_large))
            next_id += 2

        if not new_leaves:
            continue
        ids = [nl[0] for nl in new_leaves]
        idxs = [nl[1] for nl in new_leaves]
        Us = np.array([nl[2] for nl in new_leaves])
        means_new, var_new = _group_stats(psi0_all, idxs)
        v_new, u_new = _posterior_vu(model, Z, sigma_all, idxs)
        p_new = np.array([idx.size for idx in idxs], dtype=float) / N
        L_current = len(leaves) + len(new_leaves)
        flagged_new = _flag_mask(p_new, Us, means_new, a, V_hat, u_new, L_current, eps)
        for i, nid in enumerate(ids):
            leaves[nid] = dict(indices=idxs[i], n=idxs[i].size, U=Us[i], m=means_new[i],
                                var=var_new[i], v=v_new[i], ubar=psi_centered[idxs[i]].mean(axis=0))
            if flagged_new[i]:
                current_flagged.add(nid)

    ordered_ids = sorted(leaves.keys())
    L_final = len(ordered_ids)
    bin_mass = np.array([leaves[i]['n'] for i in ordered_ids], dtype=float) / N
    bin_U = np.array([leaves[i]['U'] for i in ordered_ids])
    bin_m = np.array([leaves[i]['m'] for i in ordered_ids])
    bin_flagged = np.array([i in current_flagged for i in ordered_ids], dtype=bool)
    bin_label = np.empty(N, dtype=int)
    for new_id, old_id in enumerate(ordered_ids):
        bin_label[leaves[old_id]['indices']] = new_id

    V_win_hat = np.zeros(q)
    for i in ordered_ids:
        leaf = leaves[i]
        if leaf['n'] > 1:
            V_win_hat += (leaf['n'] / N) * (leaf['var'] + leaf['v'])
    V_tot_hat = V_btw + V_win_hat

    gain_ratio = np.full(q, np.nan)
    nonzero = sum_g != 0.0
    gain_ratio[nonzero] = sum_delta[nonzero] / sum_g[nonzero]

    return JointResult(
        V_btw=V_btw, V_win_hat=V_win_hat, V_tot_hat=V_tot_hat,
        S_pred=growth.S_pred, a=a, gain_ratio=gain_ratio,
        L0=L0, L=L_final, n_growth_rounds=growth.n_growth_rounds,
        growth_capped=growth.growth_capped, S_pred_pre_lloyd=growth.S_pred_pre_lloyd,
        n_flagged=n_flagged, n_check_rounds=n_check_rounds, n_check_evals=n_check_evals,
        n_level_splits=n_level_splits, n_adjacency_splits=n_adjacency_splits,
        check_capped=check_capped, failed=False,
        bin_mass=bin_mass, bin_U=bin_U, bin_m=bin_m, bin_flagged=bin_flagged,
        bin_label=bin_label, busy_delta=busy_delta,
    )
