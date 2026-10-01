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

`joint_psi_hat`, called by `qij.py` after `run_joint`, builds the
per-point refined influence estimate on `run_joint`'s own final shared
bins, `refine.CoordinateResult.field`'s formula applied per output to
that shared partition in place of the marginal path's own private one
per output.

`run_joint`'s `pilot` argument (spec/QIJ_affine_pilot_spec.md 3.3) picks
the check's flag and kind pricing once, at setup, the same way
`refine.prepare_coordinate` picks it for the marginal path: `'gp'`
(today's rule, bit-identical) or `'affine'` (no posterior variance
anywhere; the adjacency proposal is priced from the bridge score
instead).

`check_rule` (spec/QIJ_joint_check_measured_spec.md) picks the check's
continuation rule once, at the same setup point as `pilot`: `'predicted'`
(today's re-flagging rule, byte-identical products) or `'measured'`
(`pilot='gp'` only; a ValueError otherwise) -- the pilot proposes,
measurement only closes (spec section 2.4): a child is open only if the
pilot's own flag holds for it AND its parent's split paid against a
noise-aware floor; no second chance once a split fails to pay.

`tree_rule` picks the joint tree's own growth/share rule, chosen once
at the same setup point as `pilot`/`check_rule`: `'perbin'` (today's
rule, byte-identical throughout, every existing code path untouched)
or `'total'` (`pilot='gp'` only; a ValueError otherwise). Under
`'total'` growth (`grow_total`, beside `grow`) and the continuation
that replaces the check both work against each measured output's own
TOTAL within share S_c = V_win_hat_c/(V_btw_c+V_win_hat_c): the bin or
leaf that does the most to bring the currently worst output's S_c
back under `eps` is split, one at a time, rather than every bin
independently against its own per-bin slice of the tolerance. There
is no flag, no pay test, no adjacency split under `'total'` -- every
split is a plain `two_means_split` on Ψ̃, and `check_rule` is accepted
but unused. Both rules fill `JointResult.share_final` (the final
V_win_hat/(V_btw+V_win_hat) per measured output, a cheap diagnostic
that changes no other product under `'perbin'`).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence, Tuple

import numpy as np

from ..parallel import call_T
from .differences import forward_step, perturbed_weights, step_parameter
from .influence_model import bin_posterior_variance
from .ivq import BinSet, between_terms, bias_and_acceleration, bin_differences
from .refine import (
    _try_adjacency_split, adjacency_gain_value, adjacency_split_gain, bridge_gain_value,
    compute_rho2, level_gain_value,
)

__all__ = ["Growth", "JointResult", "grow", "grow_total", "joint_psi_hat", "run_joint",
           "two_means_split"]

_CHUNK = 2048  # row chunk for the full-partition Lloyd reassignment: bounds memory


@dataclass
class Growth:
    """The shared partition built by growth, before any evaluation.
    labels (N,) bin index per point, contiguous 0..L0-1; L0 the bin
    count after growth and its Lloyd pass; std (q,) = sqrt(V_hat), so
    that Ψ̃ = psi0_all/std throughout; S_pred, S_pred_pre_lloyd (q,) the
    predicted within share after / before the Lloyd pass."""

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
    section 6). Per output (q,): V_btw, V_win_hat, V_tot_hat, S_pred, a
    (the check's scale factor, B4), gain_ratio, B_hat and a_bca
    (`ivq.bias_and_acceleration`'s ABC bias/acceleration, spec A10, from
    the shared bins as measured before any check split -- named `a_bca`
    to avoid colliding with the unrelated check scale factor `a`).
    Shared: L0, L (final), n_growth_rounds, growth_capped,
    S_pred_pre_lloyd (q,), n_flagged (bins flagged in the first check),
    n_check_rounds, n_check_evals, n_level_splits, n_adjacency_splits,
    check_capped, n_closed_unpaid, n_closed_unflagged, n_noise_floored,
    sum_b_delta (q,) (check_rule='measured' only, spec/QIJ_joint_check_
    measured_spec.md 2.4-2.5; 0/NaN under 'predicted'), failed (a failed
    output, or a failed initial
    measurement before any check ran, voids every output). Bin
    constituents: bin_mass (L,), bin_U (L,q), bin_m (L,q, predicted
    means), bin_flagged (L,); bin_label (N,); busy_delta as the other
    pool stages report it. share_final (q,) the final V_win_hat/
    (V_btw+V_win_hat) per measured output -- a cheap diagnostic filled
    under every tree_rule (NaN on a failed draw); it changes no other
    field under tree_rule='perbin'."""

    V_btw: np.ndarray
    V_win_hat: np.ndarray
    V_tot_hat: np.ndarray
    S_pred: np.ndarray
    a: np.ndarray
    gain_ratio: np.ndarray
    B_hat: np.ndarray
    a_bca: np.ndarray
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
    n_closed_unpaid: int
    n_closed_unflagged: int
    n_noise_floored: int
    sum_b_delta: np.ndarray
    failed: bool
    bin_mass: np.ndarray
    bin_U: np.ndarray
    bin_m: np.ndarray
    bin_flagged: np.ndarray
    bin_label: np.ndarray
    busy_delta: float
    share_final: np.ndarray = field(default_factory=lambda: np.full(0, np.nan))


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
    the FULL labelling, via two `bincount` passes per output: the
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


def _bridge_vectors(groups: Sequence[np.ndarray], pair_id: np.ndarray, pair_n: np.ndarray,
                     pair_value: np.ndarray) -> np.ndarray:
    """Per-group bridge-gain vector, every measured output at once
    (spec/QIJ_affine_pilot_spec.md 3.2, design (b)): for each group, the
    contained mask (`refine.batch_bridge`'s own containment test -- one
    `bincount` of the group's own `pair_id` against `pair_n`) dotted
    against `pair_value` = pair_mass[:, None] * pair_delta[:, measured]^2
    (built once, by `qij.py`), giving Sum_{(jk) subset group} m_jk*Delta_jk^2
    per output in one matrix product, never a loop over outputs or pairs."""
    n_pairs = pair_n.size
    q = pair_value.shape[1]
    out = np.empty((len(groups), q))
    for i, idx in enumerate(groups):
        counts = np.bincount(pair_id[idx], minlength=n_pairs + 1)[:n_pairs]
        contained = (counts == pair_n).astype(float)
        out[i] = contained @ pair_value
    return out


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


def grow(psi0_all: np.ndarray, eps: float, M_X_used: int) -> Growth:
    """
    Grow one shared partition of the N points from a single bin to the
    tolerance (spec/method_notes.md section 6), no evaluations. A round splits,
    by `two_means_split` on Ψ̃ = psi0_all/std, every bin with
    max_c w_kc > eps/L; stops when every output's predicted within
    share S_c <= eps, or when L reaches M_X_used (`growth_capped`,
    checked only once tolerance is confirmed unmet, so a round that
    both converges and reaches the cap counts as converged). Then one
    Lloyd pass of all L centroids over every row, which lowers the
    predicted within share at no evaluation cost; S_pred is taken after
    that pass, S_pred_pre_lloyd before.
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
            # The cap: every flagged split is feasible (w_kc > 0
            # implies >= 2 distinct rows), so the `room` largest max_c
            # w_kc bins fill to M_X_used exactly, deterministically.
            order = np.argsort(-w[to_split].max(axis=1), kind='stable')
            to_split = to_split[order[:room]]
            growth_capped = True
        labels, L = _split_growth_bins(psi_tilde, labels, to_split, L)
        n_rounds += 1
        if growth_capped:
            break

    S_pre = S
    labels, L = _lloyd_all(psi_tilde, labels, L)
    _, S = _predicted_share(psi0_all, V_hat, labels, L)

    return Growth(labels=labels, L0=L, n_growth_rounds=n_rounds, growth_capped=growth_capped,
                  S_pred=S, S_pred_pre_lloyd=S_pre, std=std)


def grow_total(psi0_all: np.ndarray, eps: float, M_X_used: int) -> Growth:
    """
    `grow`'s counterpart for `tree_rule='total'`: one bin is split per
    round, the one that does the most for whichever output's predicted
    TOTAL within share S_c = sum_k w_kc is currently worst against
    `eps`, rather than every over-threshold bin at once against its own
    per-bin slice of the tolerance (`grow`'s rule). A round: if every
    S_c <= eps, stop; if L has reached `M_X_used`, stop (growth_capped);
    otherwise c* = argmax_c S_c/eps (eps is one scalar across outputs,
    so this is argmax_c S_c; written as the ratio to match the
    tolerance it is tested against), and the bin with the largest
    w_kc* is split by `two_means_split` on its own Ψ̃ rows (ties -> the
    lowest bin id). A bin whose split comes back infeasible is skipped
    rather than split, and never tried again (nothing about the data
    changes by skipping it, so it would otherwise be picked again every
    round); if every remaining bin is infeasible, growth stops there
    (nothing splittable remains, not `growth_capped`). Each split adds
    exactly one bin, counted into `n_growth_rounds`. Then, as in
    `grow`, one `_lloyd_all` pass of all L centroids over every row;
    S_pred is taken after that pass, S_pred_pre_lloyd before.
    """
    N, q = psi0_all.shape
    std = np.std(psi0_all, axis=0)
    V_hat = std ** 2
    psi_tilde = psi0_all / std
    labels = np.zeros(N, dtype=int)
    L = 1
    n_rounds = 0
    growth_capped = False
    infeasible: set = set()

    while True:
        w, S = _predicted_share(psi0_all, V_hat, labels, L)
        if np.all(S <= eps):
            break
        if L >= M_X_used:
            growth_capped = True
            break
        c_star = int(np.argmax(S / eps))
        order = sorted((k for k in range(L) if k not in infeasible),
                        key=lambda k: (-w[k, c_star], k))
        chosen = None
        for k in order:
            idx = np.where(labels == k)[0]
            split = two_means_split(psi_tilde[idx])
            if split is None:
                infeasible.add(k)
                continue
            chosen = (idx, split)
            break
        if chosen is None:
            break  # every remaining bin is infeasible: nothing splittable remains
        idx, (idx_a, idx_b) = chosen
        labels = labels.copy()
        labels[idx[idx_b]] = L
        L += 1
        n_rounds += 1

    S_pre = S
    labels, L = _lloyd_all(psi_tilde, labels, L)
    _, S = _predicted_share(psi0_all, V_hat, labels, L)

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


def _flag_mask(p, U, m, V_hat, u, L: int, eps: float) -> np.ndarray:
    """Bin k is flagged when for some output c the measured derivative
    disagrees with the predicted mean by more than the bin's share of
    the tolerance plus what the posterior allows for the error of a
    bin mean: p_k*(U_kc-m_kc)^2 > eps*V_hat_c/L + p_k*u_kc. (A18,
    spec/QIJ_mods_waves.md: `m`, `u` and `V_hat` are already in the
    pilot's a_c-corrected units -- the same units as `U` -- by the time
    every caller of this function reaches it, so no `a_c`/`a_c^2`
    factor is carried here any more; carrying it was the earlier bug,
    comparing a measured-scale left side against a pilot-scale
    threshold.)"""
    lhs = p[:, None] * (U - m) ** 2
    rhs = (eps * V_hat[None, :] / L) + p[:, None] * u
    return np.any(lhs > rhs, axis=1)


def _split_task(T, case, X: np.ndarray, task):
    """One check-round split's one-shot forward evaluation, the
    same failure boundary as `ivq._bin_task`: `task` is (split id,
    signed step t, member mask, start); `call_T` applies the
    estimator's prepared state and the start-continuation rule (A9).
    Returns (id, evaluation, failure flag, this call's own wall time)."""
    sid, t, mask, start = task
    omega = perturbed_weights(np.ones(len(X)), mask, t)
    t0 = time.perf_counter()
    try:
        result = call_T(T, X, omega, start)
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
        B_hat=nan_q, a_bca=nan_q,
        L0=L0, L=L0, n_growth_rounds=n_rounds, growth_capped=capped, S_pred_pre_lloyd=S_pre,
        n_flagged=0, n_check_rounds=0, n_check_evals=0, n_level_splits=0, n_adjacency_splits=0,
        check_capped=False, n_closed_unpaid=0, n_closed_unflagged=0, n_noise_floored=0,
        sum_b_delta=nan_q, failed=True,
        bin_mass=mass, bin_U=np.full((L0, q), np.nan), bin_m=np.full((L0, q), np.nan),
        bin_flagged=np.zeros(L0, dtype=bool), bin_label=labels, busy_delta=busy_delta,
        share_final=nan_q,
    )


def _run_total(
    N: int, q: int, X: np.ndarray, counter, theta_hat: np.ndarray, psi0_all: np.ndarray,
    psi_tilde: np.ndarray,
    measured: Sequence[int], pool, start: Optional[np.ndarray], eta: float, eps: float,
    M_X_used: int, growth: Growth, bins: BinSet, V_btw: np.ndarray,
    groups0: Sequence[np.ndarray], U0: np.ndarray, m0: np.ndarray, var0: np.ndarray,
    B_hat: np.ndarray, a_bca: np.ndarray, a: np.ndarray, busy_delta: float,
) -> JointResult:
    """
    `run_joint`'s continuation under `tree_rule='total'`: there is no
    flag, no pay test, no adjacency split and no posterior v -- every
    round splits whichever open leaf (or group of leaves) does the
    most to bring the currently worst measured output's own TOTAL
    within share S_c = V_win_hat_c/(V_btw_c+V_win_hat_c) back under
    `eps`, measured exactly as a `check_rule='predicted'` level split
    is (one forward evaluation of the small child at its own step,
    `U_large` by conservation, `Delta` on `V_btw`'s own scale), all of
    a round's evaluations in one pool batch through `_split_task`.

    A round: V_win_hat is the plain within-bin spread over the CURRENT
    leaves, (1/N) sum_{k: n_k>1} p_k Var_k(psi0) (no v, the same form
    `check_rule='measured'` reports); if every S_c <= eps, stop. If the
    evaluation cap (`1 + M_X_used`) is spent, stop (`check_capped`).
    Otherwise c* = argmax_c S_c/eps (eps is one scalar across outputs,
    kept as the ratio to match what it is tested against) is the
    binding output, and E = V_win_hat_c* - eps/(1-eps)*V_btw_c* is the
    within that must come off it for S_c* to clear `eps`. Open leaves
    (n_k > 1, Var_k(psi0)[c*] > 0; a leaf with neither is never
    ranked, since it cannot change S_c* and cannot be split either)
    are ranked by their own contribution (1/N)*p_k*Var_k(psi0)[c*],
    descending, ties to the lowest id; the smallest prefix whose
    summed contribution reaches E (at least one leaf, the whole ranked
    list if E is never reached) is proposed for a plain
    `two_means_split` on its own Ψ̃ rows. A leaf whose split comes back
    infeasible is closed (kept, excluded from every later round's
    ranking -- the data under it does not change, so it would
    otherwise be picked again); a leaf whose evaluation fails is
    closed the same way, with no second chance. If nothing is proposed
    (every open leaf already closed, or ranking finds nothing with
    positive contribution), the continuation stops there, same as
    growth's own "nothing splittable remains".

    `V_btw` is carried in from `run_joint`'s own full-data measurement
    of the growth bins and updated by `+= Delta` each successful split,
    exactly as the check does. `a` (`_fit_scale`), `B_hat`/`a_bca`
    (`bias_and_acceleration`) are passed through unchanged, fitted once
    against the growth bins before any continuation round, as today.
    `gain_ratio` is NaN throughout: there is no predicted-gain geometry
    to size a split under this rule (no flag, no v, no adjacency), so
    there is nothing to compare a realized `Delta` against.

    `psi0_all`, `m0`, `var0` arrive already in A18's a_c-corrected
    units (spec/QIJ_mods_waves.md A18; `run_joint`'s own correction,
    applied once right after `a` is fitted, before this function is
    called), so `win_hat`, `candidates` and `E` below read the
    corrected within-bin spread with no further factor. `psi_tilde` is
    passed in SEPARATE from `psi0_all` and is NOT recomputed here from
    `psi0_all`/`growth.std`: it is the one array `run_joint` built
    before that correction, from the pre-correction `psi0_all` and
    `growth.std`, so every `two_means_split` cut drawn from it stays
    byte-identical to the pre-A18 code (scaling `psi0_all` and
    `growth.std` by the same `a_c` and then dividing is not
    bit-identical to the pre-correction ratio).
    """
    L0 = growth.L0
    centering_residual = bins.centering_residual[measured]
    delta_f = forward_step(eta)
    evals_cap = 1 + M_X_used
    n_check_rounds = n_check_evals = n_level_splits = 0
    check_capped = False
    next_id = L0
    closed: set = set()

    leaves: Dict[int, dict] = {
        k: dict(indices=groups0[k], n=int(bins.n[k]), U=U0[k].copy(), m=m0[k].copy(),
                var=var0[k].copy())
        for k in range(L0)
    }

    def win_hat() -> np.ndarray:
        out = np.zeros(q)
        for leaf in leaves.values():
            if leaf['n'] > 1:
                out += (leaf['n'] / N) * leaf['var']
        return out / N  # (1/N) sum_k p_k Var_k(psi0), no v (method_notes section 6)

    while True:
        V_win_hat = win_hat()
        S = V_win_hat / (V_btw + V_win_hat)
        if np.all(S <= eps):
            break
        if n_check_evals >= evals_cap:
            check_capped = True
            break
        c_star = int(np.argmax(S / eps))
        E = V_win_hat[c_star] - (eps / (1.0 - eps)) * V_btw[c_star]
        candidates = [
            (k, (leaf['n'] / N) * leaf['var'][c_star] / N)
            for k, leaf in sorted(leaves.items())
            if k not in closed and leaf['n'] > 1 and leaf['var'][c_star] > 0.0
        ]
        candidates.sort(key=lambda t: (-t[1], t[0]))
        chosen = []
        cum = 0.0
        for k, contrib in candidates:
            chosen.append(k)
            cum += contrib
            if cum >= E:
                break
        if not chosen:
            break  # nothing left can move S_c*: nothing splittable remains

        n_check_rounds += 1
        proposals = {}
        for k in chosen:
            idx = leaves[k]['indices']
            split = two_means_split(psi_tilde[idx])
            if split is None:
                closed.add(k)
                continue
            idx_a, idx_b = idx[split[0]], idx[split[1]]
            idx_small, idx_large = (idx_a, idx_b) if idx_a.size <= idx_b.size else (idx_b, idx_a)
            p_small = idx_small.size / N
            p_large = idx_large.size / N
            t_small = step_parameter(delta_f, p_small)
            mask = np.zeros(N, dtype=bool)
            mask[idx_small] = True
            proposals[k] = dict(idx_small=idx_small, idx_large=idx_large,
                                 p_small=p_small, p_large=p_large, t_small=t_small, mask=mask)
        if not proposals:
            continue

        tasks = [(k, meta['t_small'], meta['mask'], start) for k, meta in proposals.items()]
        if pool is None:
            evaluated = []
            for k, t, mask, _start in tasks:
                val = np.asarray(counter(X, perturbed_weights(np.ones(N), mask, t), start=_start),
                                  dtype=float)
                evaluated.append((k, val, bool(np.any(np.isnan(val)))))
        else:
            t_map0 = time.perf_counter()
            raw = pool.map(_split_task, tasks)
            busy_delta += sum(r[3] for r in raw) - (time.perf_counter() - t_map0)
            evaluated = [(k, val, failed) for k, val, failed, _ in raw]
            for _, _, failed in evaluated:
                counter.add(1, N, int(failed))
        n_check_evals += len(tasks)

        for k, val, failed in evaluated:
            if failed:
                closed.add(k)  # cancelled: the parent stays as a final bin, closed
                continue
            meta = proposals[k]
            leaf = leaves.pop(k)
            p_parent = leaf['n'] / N
            U_small = (val[measured] - theta_hat[measured]) / meta['t_small'] - centering_residual
            U_large = (p_parent * leaf['U'] - meta['p_small'] * U_small) / meta['p_large']
            Delta = (meta['p_small'] * U_small ** 2 + meta['p_large'] * U_large ** 2
                     - p_parent * leaf['U'] ** 2) / N
            V_btw += Delta
            n_level_splits += 1
            means_c, var_c = _group_stats(psi0_all, [meta['idx_small'], meta['idx_large']])
            leaves[next_id] = dict(indices=meta['idx_small'], n=meta['idx_small'].size,
                                    U=U_small, m=means_c[0], var=var_c[0])
            leaves[next_id + 1] = dict(indices=meta['idx_large'], n=meta['idx_large'].size,
                                        U=U_large, m=means_c[1], var=var_c[1])
            next_id += 2

    ordered_ids = sorted(leaves.keys())
    L_final = len(ordered_ids)
    bin_mass = np.array([leaves[i]['n'] for i in ordered_ids], dtype=float) / N
    bin_U = np.array([leaves[i]['U'] for i in ordered_ids])
    bin_m = np.array([leaves[i]['m'] for i in ordered_ids])
    bin_flagged = np.zeros(L_final, dtype=bool)  # no flag concept under tree_rule='total'
    bin_label = np.empty(N, dtype=int)
    for new_id, old_id in enumerate(ordered_ids):
        bin_label[leaves[old_id]['indices']] = new_id

    V_win_hat = win_hat()
    V_tot_hat = V_btw + V_win_hat
    share_final = V_win_hat / V_tot_hat

    return JointResult(
        V_btw=V_btw, V_win_hat=V_win_hat, V_tot_hat=V_tot_hat,
        S_pred=growth.S_pred, a=a, gain_ratio=np.full(q, np.nan),
        B_hat=B_hat, a_bca=a_bca,
        L0=L0, L=L_final, n_growth_rounds=growth.n_growth_rounds,
        growth_capped=growth.growth_capped, S_pred_pre_lloyd=growth.S_pred_pre_lloyd,
        n_flagged=0, n_check_rounds=n_check_rounds, n_check_evals=n_check_evals,
        n_level_splits=n_level_splits, n_adjacency_splits=0,
        check_capped=check_capped, n_closed_unpaid=0, n_closed_unflagged=0, n_noise_floored=0,
        sum_b_delta=np.full(q, np.nan), failed=False,
        bin_mass=bin_mass, bin_U=bin_U, bin_m=bin_m, bin_flagged=bin_flagged,
        bin_label=bin_label, busy_delta=busy_delta, share_final=share_final,
    )


def run_joint(
    X: np.ndarray, counter, theta_hat: np.ndarray, psi0_all: np.ndarray,
    sigma_all: np.ndarray, model, Z: np.ndarray, xvq, eta: float, eps: float,
    offset: np.ndarray, pool=None, I_proto: Optional[np.ndarray] = None,
    start: np.ndarray = None, measured: Optional[Sequence[int]] = None,
    pilot: str = 'gp', bridge_pair_id: Optional[np.ndarray] = None,
    bridge_pair_n: Optional[np.ndarray] = None, bridge_value: Optional[np.ndarray] = None,
    check_rule: str = 'predicted', tree_rule: str = 'perbin',
) -> JointResult:
    """
    The joint second stage (spec/method_notes.md section 6): grow a
    shared partition (`grow`), measure every bin through
    `ivq.bin_differences`, then a measured check that splits a flagged
    bin, all a round's evaluations on `pool`. `I_proto` (M_X_used, q),
    the prototype influence from stage 1, feeds the check's adjacency
    split. `start` (A9's continuation rule, spec/QIJ_mods_waves.md A9)
    is passed to every full-data evaluation here -- the initial bin
    measurement and every check split; None reproduces today's
    evaluations bit for bit. `measured` (spec/QIJ_mods_waves.md A11) is
    the absolute indices, into T's full output, that `psi0_all`/
    `sigma_all`/`model`/`I_proto` already carry; `theta_hat`, and every
    raw evaluation this function makes (`bin_differences`'s stencils,
    a check split's own `val`), stay full width, one evaluation
    reporting every output at no extra cost, and are cut down to
    `measured`'s columns wherever they meet a psi0-driven quantity --
    `bias_and_acceleration`'s `B_hat`/`a_bca` are the one exception,
    read from the full bins and restricted to `measured` by the caller
    instead (`qij.py`). `measured=None` defaults to identity
    (0..q-1), reproducing today's evaluations bit for bit.

    `offset` (q,) is `psi0_all`'s own centering constant (`qij.py`'s
    local `offset`, `model.offset` under `pilot='gp'`, the pilot-free
    mean under `'affine'`), read here instead of `model.offset` so this
    function never touches `model` when `pilot='affine'` (`model` is
    then `None`). `pilot` picks the check's two flag/kind pricers
    (spec/QIJ_affine_pilot_spec.md 3.3, R2), chosen once below, never
    inside the check loop: `'gp'` (today's rule, bit-identical) reads
    `model`/`sigma_all` for the within-bin posterior v_kc/u_kc and its
    ported flag/kind rules; `'affine'` never calls `bin_posterior_
    variance` (v_kc/u_kc are held at 0.0 throughout, so the flag rule's
    posterior term vanishes and V_win_hat is the plain within-bin
    variance), and instead prices a flagged bin's adjacency proposal
    from the bridge score, `bridge_pair_id`/`bridge_pair_n`/
    `bridge_value` (`bridge_value` = pair_mass[:, None] *
    pair_delta[:, measured]^2, built once by `qij.py`) feeding
    `_bridge_vectors`.

    `check_rule` (spec/QIJ_joint_check_measured_spec.md, R2) picks the
    check's continuation rule once, at the same setup point as `pilot`:
    `'predicted'` (today's `_flag_mask` re-flagging rule, every product
    byte-identical) or `'measured'` (`pilot='gp'` only -- a ValueError
    otherwise -- spec section 2.4: a child is open only if the pilot's
    own flag holds for it AND its parent's split paid against a
    finite-difference-aware floor; both gates read no posterior
    variance beyond the flag's own `u`, and there is no second chance).

    `tree_rule` picks the joint tree's own growth/share rule, chosen
    once at the same setup point as `pilot`/`check_rule`: `'perbin'`
    (today's rule; every line below this point, and everything grown
    or checked, is byte-identical to today) or `'total'` (`pilot='gp'`
    only -- a ValueError otherwise; `_run_total`, module docstring):
    growth (`grow_total`) and the continuation that replaces the check
    both work against each measured output's own TOTAL within share
    S_c = V_win_hat_c/(V_btw_c+V_win_hat_c), splitting whichever single
    bin or leaf does the most for the currently worst output, rather
    than every bin independently against its own per-bin slice of the
    tolerance; there is no flag, no pay test, no adjacency split, and
    `check_rule` is accepted but unused. Both rules fill
    `JointResult.share_final` (the final V_win_hat/(V_btw+V_win_hat)
    per measured output), which changes no other product under
    `tree_rule='perbin'`.
    """
    if check_rule not in ('predicted', 'measured'):
        raise ValueError(f"unknown check_rule {check_rule!r}")
    if check_rule == 'measured' and pilot != 'gp':
        raise ValueError("check_rule='measured' requires pilot='gp'")
    if tree_rule not in ('perbin', 'total'):
        raise ValueError(f"unknown tree_rule {tree_rule!r}")
    if tree_rule == 'total' and pilot != 'gp':
        raise ValueError("tree_rule='total' requires pilot='gp'")

    N, q = psi0_all.shape
    if measured is None:
        measured = list(range(q))
    M_X_used = xvq.M_used

    # `constant_path` is never raised under `pilot='affine'` (no `model`
    # to read it from; a genuinely flat output is caught downstream by
    # growth's own bin machinery instead, as the marginal path documents
    # in `qij.py`).
    if pilot == 'gp' and np.any(model.constant_path):
        return _failed_result(N, q)

    growth = grow_total(psi0_all, eps, M_X_used) if tree_rule == 'total' else grow(psi0_all, eps, M_X_used)
    L0 = growth.L0
    binset0 = BinSet(labels=growth.labels, n=np.bincount(growth.labels, minlength=L0),
                      U=np.zeros((L0, 0)), centering_residual=np.zeros(0), D2=np.zeros((L0, 0)),
                      M_init=L0, M_used=L0, within_share=0.0, failed=False)
    bins, busy_delta = bin_differences(X, counter, theta_hat, binset0, eta, pool, start=start)
    if bins.failed:
        return _failed_result(N, q, growth=growth, busy_delta=busy_delta)

    B_hat, a_bca = bias_and_acceleration(bins, N)
    V_btw = np.array([between_terms(bins, c) for c in measured])
    V_hat = growth.std ** 2
    psi_tilde = psi0_all / growth.std
    U0 = bins.U[:, measured]
    centering_residual = bins.centering_residual[measured]
    _, m0, var0 = _bin_stats(psi0_all, bins.labels, L0)
    a = _fit_scale(bins.p, U0, m0)

    # A18 (spec/QIJ_mods_waves.md): a_c is the gap between SURVEY units
    # and FULL-DATA units on each output axis (the planner's sharpened
    # ruling), applied once here, right after `a` is fitted. Every
    # survey-unit quantity read from this point on -- `psi0_all` itself,
    # `offset`, the growth-bin means `m0` and `V_hat` (first and second
    # moments respectively), the bridge table's pair_delta (via
    # `bridge_value`, squared, so a_c^2), and the survey's own prototype
    # influences `I_proto` (first moment, wherever it meets a corrected
    # quantity below) -- is rebound to a NEW local array in its
    # a_c-corrected units, so the flag test, the reported variance and
    # the check's adjacency pricing below compare survey-unit and
    # full-data quantities on ONE scale instead of patching the
    # comparison (spec "the rule that answers every site"). These are
    # rebindings, never in-place writes: under `pilot='gp'`, `psi0_all`
    # is `influence_model.psi0`'s own cache ("THE RETURNED ARRAY IS THE
    # CACHE, not a copy: callers must not write into it"), `offset` may
    # alias `model.offset`, and `I_proto` is the caller's own array
    # (`qij.py`'s stage-1 product, left alone there -- spec "leave
    # alone... the marginal path"); `a[None, :] * x` (or `x * a[None, :]`)
    # always allocates a fresh array, so the cached/aliased object
    # itself is untouched. `psi_tilde` and `V_hat`'s role in the growth
    # tolerance were already fixed above (lines 815-816), from the
    # PRE-correction `psi0_all`/`growth.std`, and must stay that way:
    # every `two_means_split` cut below (and `_run_total`'s own) reads
    # that already-built `psi_tilde`, never a recomputation from the
    # now-corrected `psi0_all`, since scaling `psi0_all` and
    # `growth.std` by the same `a_c` and then dividing is NOT
    # bit-identical, in floating point, to the pre-correction ratio.
    psi0_all = psi0_all * a[None, :]
    offset = offset * a
    m0 = m0 * a[None, :]
    var0 = var0 * (a[None, :] ** 2)
    V_hat = V_hat * (a ** 2)
    if bridge_value is not None:
        bridge_value = bridge_value * (a[None, :] ** 2)
    if I_proto is not None:
        # Survey units, first moment (I_proto is the prototype's own
        # survey influence, what the pilot was fitted to): corrected
        # wherever the check's adjacency pricing below reads it
        # (`decide_split_gp`/`_affine`/`_measured`, via
        # `adjacency_split_gain`/`_try_adjacency_split`); this can
        # change which split kind the check chooses (spec, Identities).
        # Stage 1's own `I_proto` (`qij.py`'s GP/affine pilot fit, ABC
        # curvature, the stored `prototype_I`) is untouched -- this
        # rebinds only `run_joint`'s own local parameter.
        I_proto = I_proto * a[None, :]

    groups0 = [np.where(bins.labels == k)[0] for k in range(L0)]

    if tree_rule == 'total':
        # No flag, no pay test, no adjacency split, no posterior v
        # below this point (module docstring, `_run_total`'s own);
        # everything from here to the end of this function is the
        # `tree_rule='perbin'` path, untouched. `psi0_all`/`m0`/`var0`
        # passed in are already A18-corrected (above); `psi_tilde` is
        # the pre-correction array built at line 816, passed separately
        # so `_run_total`'s own cuts stay byte-identical.
        return _run_total(N, q, X, counter, theta_hat, psi0_all, psi_tilde, measured, pool,
                           start, eta, eps, M_X_used, growth, bins, V_btw, groups0, U0, m0, var0,
                           B_hat, a_bca, a, busy_delta)

    # `pilot` selects the within-bin pricing input ONCE here (R2): 'gp'
    # reads the GP posterior (v0/u0) via `_posterior_vu`; 'affine' never
    # calls `bin_posterior_variance` (design (f)) and holds v0/u0 at 0.0
    # (the flag rule's posterior term then vanishes, spec/QIJ_affine_
    # pilot_spec.md 3.2 design (a)), pricing the adjacency proposal from
    # each bin's own bridge vector instead (`_bridge_vectors`).
    if pilot == 'gp':
        v0, u0 = _posterior_vu(model, Z, sigma_all, groups0)
        # A18: `_posterior_vu` always reads the model against the
        # UNCORRECTED `sigma_all` (the GP fit itself does not change,
        # spec "what does not change"), so its v/u come back raw; the
        # explicit x a_c^2 here puts them in `psi0_all`'s own new units
        # (second moments), matching `var0`/`m0`/`V_hat` above.
        v0 = v0 * (a[None, :] ** 2)
        u0 = u0 * (a[None, :] ** 2)
        bridge0 = None
    else:
        v0 = np.zeros((L0, q))
        u0 = np.zeros((L0, q))
        bridge0 = _bridge_vectors(groups0, bridge_pair_id, bridge_pair_n, bridge_value)
    psi_centered = psi0_all - offset[None, :]

    leaves: Dict[int, dict] = {
        k: dict(indices=groups0[k], n=int(bins.n[k]), U=U0[k].copy(), m=m0[k].copy(),
                var=var0[k].copy(), v=v0[k].copy(), ubar=psi_centered[groups0[k]].mean(axis=0))
        for k in range(L0)
    }
    if bridge0 is not None:
        for k in range(L0):
            leaves[k]['bridge'] = bridge0[k]

    flagged_mask = _flag_mask(bins.p, U0, m0, V_hat, u0, L0, eps)
    n_flagged = int(flagged_mask.sum())
    current_flagged = set(np.where(flagged_mask)[0].tolist())

    n_check_rounds = n_check_evals = n_level_splits = n_adjacency_splits = 0
    # check_rule='measured' only (spec 2.4/2.5); the ints stay 0 and
    # sum_b_delta stays NaN under 'predicted', which never touches them.
    n_closed_unpaid = n_closed_unflagged = n_noise_floored = 0
    sum_b_delta = np.zeros(q) if check_rule == 'measured' else np.full(q, np.nan)
    sum_delta, sum_g = np.zeros(q), np.zeros(q)
    evals_cap = 1 + M_X_used
    check_capped = False
    delta_f = forward_step(eta)
    next_id = L0

    def decide_split_gp(leaf):
        """`pilot='gp'`'s split-kind rule for a flagged bin (ported,
        bit-identical): level when sum_c Var_k(psi0_c)/V_btw_c >=
        sum_c v_kc/V_btw_c, else adjacency on whichever output's own
        CADJ direction scores highest; an infeasible kind falls back to
        the other."""
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

    def decide_split_affine(leaf):
        """`pilot='affine'`'s split-kind rule for a flagged bin
        (spec/QIJ_affine_pilot_spec.md 3.2 design (b)/(c)): BOTH kinds
        are priced per output at rho2=1 -- the level split from
        `two_means_split` on the bin's own Ψ̃ rows, priced by
        `level_gain_value`; the adjacency gain from this leaf's own
        bridge vector, priced by `bridge_gain_value` -- and level is
        chosen when its summed normalized gain is at least the
        adjacency one; the adjacency geometry itself comes from
        `_try_adjacency_split` on whichever output c* has the largest
        g_adj,c/V_btw,c, trying the remaining outputs in descending
        order if c*'s split is infeasible. An infeasible chosen kind
        falls back to the other; both infeasible closes the bin."""
        idx = leaf['indices']
        p_k = leaf['n'] / N
        score_adj = bridge_gain_value(leaf['bridge'], 1.0) / V_btw

        level_result = None
        s_level = -np.inf
        level_split = two_means_split(psi_tilde[idx])
        if level_split is not None:
            idx_a, idx_b = idx[level_split[0]], idx[level_split[1]]
            ubar_a = psi_centered[idx_a].mean(axis=0)
            ubar_b = psi_centered[idx_b].mean(axis=0)
            g_level = level_gain_value(idx_a.size / N, ubar_a, idx_b.size / N, ubar_b,
                                        p_k, leaf['ubar'], N, 1.0)
            s_level = float(np.sum(g_level / V_btw))
            level_result = ('level', idx_a, idx_b)

        if level_result is not None and s_level >= float(np.sum(score_adj)):
            return level_result

        adjacency_result = None
        if I_proto is not None:
            for c in np.argsort(-score_adj):
                split = _try_adjacency_split(idx, I_proto[:, c], xvq.bmu, xvq.bmu2)
                if split is not None:
                    adjacency_result = ('adjacency', split[0], split[1])
                    break
        return adjacency_result if adjacency_result is not None else level_result

    def decide_split_measured(leaf):
        """`check_rule='measured'`'s split-kind rule for a flagged bin
        (spec/QIJ_joint_check_measured_spec.md 2.2): both kinds are
        priced per output at rho2=1 from the pilot's mean only, no v --
        level from `two_means_split` on the bin's own Ψ̃ rows, priced by
        `level_gain_value`; adjacency from whichever output's own
        `_try_adjacency_split` geometry gives the largest
        level_gain_value/V_btw,c among the feasible outputs, its price
        the SAME level_gain_value formula applied to that geometry's own
        two children, for every output. The larger summed normalized
        gain wins, level at a tie (as `decide_split_affine` does); an
        infeasible chosen kind falls back to the other, both infeasible
        closes the bin. `adjacency_gain_value` (v-based) is never
        called here."""
        idx = leaf['indices']
        p_k = leaf['n'] / N

        level_result = None
        s_level = -np.inf
        level_split = two_means_split(psi_tilde[idx])
        if level_split is not None:
            idx_a, idx_b = idx[level_split[0]], idx[level_split[1]]
            ubar_a = psi_centered[idx_a].mean(axis=0)
            ubar_b = psi_centered[idx_b].mean(axis=0)
            g_level = level_gain_value(idx_a.size / N, ubar_a, idx_b.size / N, ubar_b,
                                        p_k, leaf['ubar'], N, 1.0)
            s_level = float(np.sum(g_level / V_btw))
            level_result = ('level', idx_a, idx_b)

        adjacency_result = None
        s_adj = -np.inf
        if I_proto is not None:
            best_split, best_c_score = None, -np.inf
            for c in range(q):
                split_c = _try_adjacency_split(idx, I_proto[:, c], xvq.bmu, xvq.bmu2)
                if split_c is None:
                    continue
                idx_a_c, idx_b_c = split_c
                ubar_a_c = float(psi_centered[idx_a_c, c].mean())
                ubar_b_c = float(psi_centered[idx_b_c, c].mean())
                c_score = level_gain_value(idx_a_c.size / N, ubar_a_c, idx_b_c.size / N,
                                            ubar_b_c, p_k, float(leaf['ubar'][c]), N, 1.0) / V_btw[c]
                if c_score > best_c_score:
                    best_split, best_c_score = split_c, c_score
            if best_split is not None:
                idx_a, idx_b = best_split
                ubar_a = psi_centered[idx_a].mean(axis=0)
                ubar_b = psi_centered[idx_b].mean(axis=0)
                g_adj = level_gain_value(idx_a.size / N, ubar_a, idx_b.size / N, ubar_b,
                                          p_k, leaf['ubar'], N, 1.0)
                s_adj = float(np.sum(g_adj / V_btw))
                adjacency_result = ('adjacency', idx_a, idx_b)

        if level_result is not None and s_level >= s_adj:
            return level_result
        return adjacency_result if adjacency_result is not None else level_result

    # Chosen once (R2), never re-chosen inside the check loop below.
    if check_rule == 'measured':
        decide_split = decide_split_measured
    else:
        decide_split = decide_split_gp if pilot == 'gp' else decide_split_affine

    def _continue_predicted(ids, idxs, Us, means_new, var_new, v_new, u_new, bridge_new,
                             p_new, L_current, split_records):
        """`check_rule='predicted'`'s continuation rule (today's,
        unchanged): re-flag every new child against `_flag_mask`, the
        same pilot means that were already wrong for the parent."""
        flagged_new = _flag_mask(p_new, Us, means_new, V_hat, u_new, L_current, eps)
        for i, nid in enumerate(ids):
            leaves[nid] = dict(indices=idxs[i], n=idxs[i].size, U=Us[i], m=means_new[i],
                                var=var_new[i], v=v_new[i], ubar=psi_centered[idxs[i]].mean(axis=0))
            if bridge_new is not None:
                leaves[nid]['bridge'] = bridge_new[i]
            if flagged_new[i]:
                current_flagged.add(nid)

    def _continue_measured(ids, idxs, Us, means_new, var_new, v_new, u_new, bridge_new,
                            p_new, L_current, split_records):
        """`check_rule='measured'`'s continuation rule (spec/QIJ_joint_
        check_measured_spec.md 2.4): a child is OPEN iff (a) the pilot's
        own flag holds for it (`_flag_mask` at the current bin count,
        the same call the seeds and the `'predicted'` rule use) AND (b)
        its parent's split PAID (Delta_c >= tau'_c for some measured
        output c, Delta the parent's realized gain, already added to
        V_btw above). Both children of a split share (b); each has its
        own (a). tau'_c = max(tau_round_c, n_Delta,c + b_Delta,c):
        tau_round is this round's tau_c = eps*V_btw_c/L at its start,
        the same for every split of the round; n_Delta,c is the
        finite-difference scatter term, b_Delta,c the noise-bias floor
        (both carried through mass balance). `n_closed_unpaid` counts
        the SPLIT when (b) fails (both children close, no second
        chance); `n_closed_unflagged` counts a CHILD of a paying split
        closed by (a) alone. `n_noise_floored` counts a split whose
        deciding output (argmax_c(Delta_c - tau'_c)) had the floor
        n_Delta + b_Delta, not tau_round, set its tau'. `sum_b_delta`
        accumulates b_Delta over every split measured this round (it is
        never subtracted from V_btw, only reported against it)."""
        nonlocal n_closed_unpaid, n_closed_unflagged, n_noise_floored, sum_b_delta
        flagged_new = _flag_mask(p_new, Us, means_new, V_hat, u_new, L_current, eps)
        paid = {}
        for rec in split_records:
            delta_U = eta * theta_floor / rec['t_small']
            n_delta = ((2.0 * rec['p_small'] / N) * (np.abs(rec['U_small']) + np.abs(rec['U_large']))
                       * delta_U)
            b_delta = ((rec['p_small'] / N) * (1.0 + rec['p_small'] / rec['p_large']) * delta_U ** 2)
            sum_b_delta += b_delta
            tau_prime = np.maximum(tau_round, n_delta + b_delta)
            margin = rec['Delta'] - tau_prime
            c_star = int(np.argmax(margin))
            split_paid = bool(margin[c_star] >= 0.0)
            if (n_delta[c_star] + b_delta[c_star]) >= tau_round[c_star]:
                n_noise_floored += 1
            if not split_paid:
                n_closed_unpaid += 1
            paid[rec['child_small_id']] = split_paid
            paid[rec['child_large_id']] = split_paid
        for i, nid in enumerate(ids):
            # `bridge_new` is always None here: `check_rule='measured'`
            # requires `pilot='gp'` (raised above), which never builds a
            # bridge vector -- unlike `_continue_predicted`, which also
            # serves `pilot='affine'`.
            leaves[nid] = dict(indices=idxs[i], n=idxs[i].size, U=Us[i], m=means_new[i],
                                var=var_new[i], v=v_new[i], ubar=psi_centered[idxs[i]].mean(axis=0))
            if not paid[nid]:
                continue
            if flagged_new[i]:
                current_flagged.add(nid)
            else:
                n_closed_unflagged += 1

    # Chosen once (R2), never re-tested inside the loop below.
    continue_round = _continue_measured if check_rule == 'measured' else _continue_predicted
    # theta_hat at the measured columns, floored at machine epsilon only
    # (spec section 2.4): fixed for the whole check, computed once.
    theta_floor = (np.maximum(np.abs(theta_hat[measured]), np.finfo(float).eps)
                   if check_rule == 'measured' else None)

    while current_flagged:
        if n_check_evals >= evals_cap:
            check_capped = True
            break
        n_check_rounds += 1
        # tau_c = eps*V_btw_c/L at the START of the round, before any of
        # its own splits (spec section 2.4/6, as `core.rounds` does);
        # the same array serves every split of this round.
        tau_round = (eps * V_btw / len(leaves)) if check_rule == 'measured' else None
        proposals = {}
        for k in sorted(current_flagged):
            decided = decide_split(leaves[k])
            if decided is None:
                continue  # both kinds infeasible: the bin stays and is closed
            kind, idx_a, idx_b = decided
            p_k = leaves[k]['n'] / N
            if kind == 'adjacency' and check_rule != 'measured':
                g = (adjacency_gain_value(p_k, leaves[k]['v'], N, 1.0) if pilot == 'gp'
                     else bridge_gain_value(leaves[k]['bridge'], 1.0))
            else:
                # Under `check_rule='measured'` an adjacency proposal is
                # priced by this SAME formula, on its own geometry's two
                # children (spec section 2.2) -- the branch above is
                # never taken there.
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

        tasks = [(k, meta['t_small'], meta['mask'], start) for k, meta in proposals.items()]
        if pool is None:
            evaluated = []
            for k, t, mask, _start in tasks:
                val = np.asarray(counter(X, perturbed_weights(np.ones(N), mask, t), start=_start), dtype=float)
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
        split_records = []
        for k, val, failed in evaluated:
            if failed:
                continue  # cancelled: parent stays as a final bin, closed, counted above
            meta = proposals[k]
            leaf = leaves.pop(k)
            p_parent = leaf['n'] / N
            U_small = (val[measured] - theta_hat[measured]) / meta['t_small'] - centering_residual
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
            if check_rule == 'measured':
                # Recorded for `_continue_measured`'s pay test below;
                # never read under `check_rule='predicted'`.
                split_records.append(dict(
                    child_small_id=next_id, child_large_id=next_id + 1, t_small=meta['t_small'],
                    p_small=meta['p_small'], p_large=meta['p_large'],
                    Delta=Delta, U_small=U_small, U_large=U_large,
                ))
            new_leaves.append((next_id, meta['idx_small'], U_small))
            new_leaves.append((next_id + 1, meta['idx_large'], U_large))
            next_id += 2

        if not new_leaves:
            continue
        ids = [nl[0] for nl in new_leaves]
        idxs = [nl[1] for nl in new_leaves]
        Us = np.array([nl[2] for nl in new_leaves])
        means_new, var_new = _group_stats(psi0_all, idxs)
        if pilot == 'gp':
            # `'measured'` needs u_kc for its own flag test (a) at the
            # new leaves, exactly like `'predicted'` (spec section 2.4);
            # v_new comes back from the same call but no split decision
            # reads it (spec section 1).
            v_new, u_new = _posterior_vu(model, Z, sigma_all, idxs)
            # A18: same explicit x a_c^2 as the first `_posterior_vu`
            # call above (`sigma_all` itself never changes).
            v_new = v_new * (a[None, :] ** 2)
            u_new = u_new * (a[None, :] ** 2)
            bridge_new = None
        else:
            v_new = np.zeros((len(idxs), q))
            u_new = np.zeros((len(idxs), q))
            bridge_new = _bridge_vectors(idxs, bridge_pair_id, bridge_pair_n, bridge_value)
        p_new = np.array([idx.size for idx in idxs], dtype=float) / N
        L_current = len(leaves) + len(new_leaves)
        continue_round(ids, idxs, Us, means_new, var_new, v_new, u_new, bridge_new,
                       p_new, L_current, split_records)

    ordered_ids = sorted(leaves.keys())
    L_final = len(ordered_ids)
    bin_mass = np.array([leaves[i]['n'] for i in ordered_ids], dtype=float) / N
    bin_U = np.array([leaves[i]['U'] for i in ordered_ids])
    bin_m = np.array([leaves[i]['m'] for i in ordered_ids])
    bin_flagged = np.array([i in current_flagged for i in ordered_ids], dtype=bool)
    bin_label = np.empty(N, dtype=int)
    for new_id, old_id in enumerate(ordered_ids):
        bin_label[leaves[old_id]['indices']] = new_id

    # check_rule='measured': the pilot's own within-bin spread, no v
    # (spec/QIJ_joint_check_measured_spec.md 2.5); 'predicted' keeps
    # today's Var_k + v_k for byte identity.
    win_includes_v = check_rule != 'measured'
    V_win_hat = np.zeros(q)
    for i in ordered_ids:
        leaf = leaves[i]
        if leaf['n'] > 1:
            spread = leaf['var'] + leaf['v'] if win_includes_v else leaf['var']
            V_win_hat += (leaf['n'] / N) * spread
    V_win_hat /= N  # (1/N) sum_k p_k (Var_k [+ v_k]), on V_btw's scale (method_notes section 6)
    V_tot_hat = V_btw + V_win_hat

    gain_ratio = np.full(q, np.nan)
    nonzero = sum_g != 0.0
    gain_ratio[nonzero] = sum_delta[nonzero] / sum_g[nonzero]
    # `share_final` (module docstring): a cheap diagnostic added to every
    # rule's result; it reads V_win_hat/V_tot_hat already computed above
    # and changes no other field here.
    share_final = V_win_hat / V_tot_hat

    return JointResult(
        V_btw=V_btw, V_win_hat=V_win_hat, V_tot_hat=V_tot_hat,
        S_pred=growth.S_pred, a=a, gain_ratio=gain_ratio,
        B_hat=B_hat, a_bca=a_bca,
        L0=L0, L=L_final, n_growth_rounds=growth.n_growth_rounds,
        growth_capped=growth.growth_capped, S_pred_pre_lloyd=growth.S_pred_pre_lloyd,
        n_flagged=n_flagged, n_check_rounds=n_check_rounds, n_check_evals=n_check_evals,
        n_level_splits=n_level_splits, n_adjacency_splits=n_adjacency_splits,
        check_capped=check_capped, n_closed_unpaid=n_closed_unpaid,
        n_closed_unflagged=n_closed_unflagged, n_noise_floored=n_noise_floored,
        sum_b_delta=sum_b_delta, failed=False, share_final=share_final,
        bin_mass=bin_mass, bin_U=bin_U, bin_m=bin_m, bin_flagged=bin_flagged,
        bin_label=bin_label, busy_delta=busy_delta,
    )


def joint_psi_hat(
    psi0_all: np.ndarray, offset: np.ndarray, bin_label: np.ndarray,
    bin_U: np.ndarray, bin_mass: np.ndarray, V_btw: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    The joint path's own per-point refined influence estimate, on
    `run_joint`'s final shared bins (`bin_label`/`bin_U`/`bin_mass`/
    `V_btw`): `refine.CoordinateResult.field`'s formula, per measured
    output c, evaluated on the shared partition in place of that
    output's own private one --

        psi_hat_i,c = U_{k(i),c} + rho_c*(psi0c_i,c - ubar_{k(i),c})

    psi0c = psi0_all - offset, the same centering
    `refine.prepare_coordinate` builds `psi_centered` with; ubar_{k,c}
    is psi0c's plain per-bin mean (points weighted 1/N), by one grouped
    `bincount` sum per output -- never a loop over points; rho_c =
    sqrt(V_btw_c / ((1/N) sum_k p_k ubar_kc^2)) via `refine.compute_rho2`,
    NaN for output c when that denominator is 0. Callers must not call
    this when `JointResult.failed` is True (`bin_label` may then hold
    -1, or `bin_U` be all NaN).
    """
    N, q = psi0_all.shape
    L = bin_U.shape[0]
    psi0c = psi0_all - offset[None, :]
    counts = np.bincount(bin_label, minlength=L).astype(float)
    sums = np.column_stack(
        [np.bincount(bin_label, weights=psi0c[:, c], minlength=L) for c in range(q)])
    ubar = sums / counts[:, None]
    sum_pubar2 = np.sum(bin_mass[:, None] * ubar ** 2, axis=0)
    rho2 = np.array([compute_rho2(float(V_btw[c]), float(sum_pubar2[c]), N) for c in range(q)])
    rho = np.where(np.isfinite(rho2) & (rho2 >= 0.0), np.sqrt(np.maximum(rho2, 0.0)), np.nan)
    psi_hat = bin_U[bin_label] + rho[None, :] * (psi0c - ubar[bin_label])
    psi_hat[:, ~np.isfinite(rho)] = np.nan
    return psi_hat, rho
