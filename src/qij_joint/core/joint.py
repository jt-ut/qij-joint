"""
The joint second stage (spec/method_notes.md section 6; spec/QIJ_mods_
waves.md A20/A21): `grow` builds one shared partition of the N points
from psi0_all to the tolerance, growing one bin at a time against
whichever measured output's own TOTAL within share S_c =
V_win_hat_c/(V_btw_c+V_win_hat_c) is currently worst; `run_joint`
measures that partition on the full data through `ivq.bin_differences`,
then carries the per-point state vector psi_hat forward, splitting
leaves the same way, until every S_c clears `eps`. Growth runs in-
process numpy only; the initial measurement and every later round's
evaluations run through `pool`.

A21 (spec/QIJ_mods_waves.md A21, "one path: the removals") removed the
per-bin joint second stage this module used to offer beside the total-
share one (`grow`'s own per-bin growth and the check's flag/pay/split-
kind machinery, `tree_rule`/`pilot`/`check_rule` as switches): the
total-share rule, under THE ARCHITECTURE RULE's one per-point vector
psi_hat (A20), is now the only method. What A20 called `grow_total`/
`_run_total` is this module's only growth and continuation; what A20
called `tree_rule='perbin'` no longer exists anywhere in this file.

THE ARCHITECTURE RULE (spec/QIJ_mods_waves.md A20, governs this module):
ONE per-point vector psi_hat exists. It is populated once, in survey
units, by the GP pilot (`qij.py`); growth below reads that populated
vector (growth is invariant to a per-output scale, so the partition is
identical whether or not the vector is yet a_c-corrected); the FIRST
measurement-driven update, after the stage-1 stencils, is the global
x a_c correction (A18) applied to the whole vector by `run_joint`
before `_run_total` is ever called, followed immediately by the per-
leaf `_apply_update` pass; every later reader -- the ranking, a cut,
V_win_hat, E, the leaf's own `ebar`, the end-of-run per-point estimate
-- reads that one vector, never a fixed copy of the pilot. Two sites
that still read something else were A21's own removals: `bin_m` now
reads the leaf's own `ebar`-moment reference value m_pre (the vector's
mean over the leaf's points at measurement time), not the fixed pilot
mean; `sd_hat`, the GP posterior sd carried alongside psi_hat purely
because `archive/seeded-tree`'s `_apply_update` wrote it, fed no
decision and is dropped from `_apply_update`'s own signature and the
state entirely (the GP posterior sd survives only as `QIJResult.sigma`,
a stored survey diagnostic `qij.py` builds on its own, never read back
in here).

`run_joint`'s own A18 correction: `a` (`_fit_scale`) is the least-
squares scale between each measured output's survey-unit pilot mean
and its measured bin derivative, fitted once against the growth bins
and applied to `psi0_all`/`offset`/the growth bins' own `m0`/`V_hat`
before `_run_total` ever runs -- the gap between survey units and
full-data units on each output axis (spec/QIJ_mods_waves.md A18).
`B_hat`/`a_bca` (`ivq.bias_and_acceleration`'s ABC bias/acceleration,
spec A10) are fitted from the same growth bins, unaffected by A20/A21.

`JointResult.state_psi_hat` carries the final vector out; `qij.py`
takes it directly as the method's own `psi_hat` (there is no field+rho
reconstruction left to build it from). `n_update_scale`/
`n_update_shift`/`n_update_negative` and `pilot_err_btw` (the stage-1
leaves' own between-leaf pilot-error variance) are its own products.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence, Tuple

import numpy as np

from ..parallel import call_T
from .differences import central_step, forward_step, perturbed_weights, step_parameter
from .ivq import BinSet, between_terms, bias_and_acceleration, bin_differences

__all__ = ["Growth", "JointResult", "grow", "run_joint", "two_means_split"]

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
    section 6; spec/QIJ_mods_waves.md A20/A21). Per output (q,): V_btw,
    V_win_hat, V_tot_hat, S_pred, a (the A18 scale factor), B_hat and
    a_bca (`ivq.bias_and_acceleration`'s ABC bias/acceleration, spec
    A10, from the growth bins as measured before any continuation
    split). Shared: L0, L (final), n_growth_rounds, growth_capped,
    S_pred_pre_lloyd (q,), n_check_rounds, n_check_evals,
    n_level_splits, check_capped, failed (a failed output, or a failed
    initial measurement before any continuation ran, voids every
    output). Bin constituents: bin_mass (L,), bin_U (L,q, the measured
    bin mean), bin_m (L,q, the leaf's own ebar reference value m_pre --
    THE ARCHITECTURE RULE, module docstring -- not the fixed pilot
    mean); bin_label (N,); busy_delta as the other pool stages report
    it. share_final (q,) the final V_win_hat/(V_btw+V_win_hat) per
    measured output (NaN on a failed draw).

    n_update_scale, n_update_shift, n_update_negative (q,) int,
    `_apply_update`'s own per-output counts over every measured bin
    (stage-1 leaves plus every split's two children); pilot_err_btw
    (q,) = sum_k p_k*(U_k-m_k^pilot)^2/N over the stage-1 leaves only
    (the measured, between-leaf part of the PILOT's own error
    variance -- a diagnostic contrasting the pilot against the state
    vector, deliberately read from the fixed pilot mean, unlike
    bin_m); state_psi_hat (N,q) the final state vector psi_hat,
    `qij.py`'s own `psi_hat` -- (0,0) on a failed draw, never read
    there. rank_rule: 'measured_error' always (the method, no switch).
    bin_ebar (L,q): each FINAL leaf's own ebar_c = U_c - (psi_hat's
    mean over the leaf's points just before that leaf's own update) at
    the moment the leaf was created -- the ranking's own per-leaf
    diagnostic, beside bin_U/bin_m."""

    V_btw: np.ndarray
    V_win_hat: np.ndarray
    V_tot_hat: np.ndarray
    S_pred: np.ndarray
    a: np.ndarray
    B_hat: np.ndarray
    a_bca: np.ndarray
    L0: int
    L: int
    n_growth_rounds: int
    growth_capped: bool
    S_pred_pre_lloyd: np.ndarray
    n_check_rounds: int
    n_check_evals: int
    n_level_splits: int
    check_capped: bool
    failed: bool
    bin_mass: np.ndarray
    bin_U: np.ndarray
    bin_m: np.ndarray
    bin_label: np.ndarray
    busy_delta: float
    share_final: np.ndarray = field(default_factory=lambda: np.full(0, np.nan))
    n_update_scale: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=int))
    n_update_shift: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=int))
    n_update_negative: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=int))
    pilot_err_btw: np.ndarray = field(default_factory=lambda: np.full(0, np.nan))
    state_psi_hat: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))
    rank_rule: str = 'n/a'
    bin_ebar: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))


def two_means_split(rows: np.ndarray) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Deterministic two-means on `rows` (n, q), the split rule shared
    by growth and a continuation round (spec/method_notes.md section
    6): initial centroids are the means of the two halves of `rows`
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


def _predicted_share(psi0_all: np.ndarray, V_hat: np.ndarray, labels: np.ndarray, L: int):
    """A growth round's predicted within share: w_kc = p_k*Var_k(psi0_c)/V_hat_c
    per bin and output, and S_c = sum_k w_kc, the total predicted within
    share for output c."""
    N = psi0_all.shape[0]
    counts, _, var = _bin_stats(psi0_all, labels, L)
    w = (counts[:, None] / N) * var / V_hat[None, :]
    return w, w.sum(axis=0)


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
    (as `ivq.kmeans_1d`'s own empty-bin handling did before A21 removed
    that quantizer)."""
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
    Grow one shared partition of the N points from a single bin
    (spec/QIJ_mods_waves.md A20's `grow_total`, the method under A21,
    no switch): one bin is split per round, the one that does the most
    for whichever output's predicted TOTAL within share S_c =
    sum_k w_kc is currently worst against `eps`, rather than every
    over-threshold bin at once against its own per-bin slice of the
    tolerance. A round: if every S_c <= eps, stop; if L has reached
    `M_X_used`, stop (growth_capped); otherwise c* = argmax_c S_c/eps
    (eps is one scalar across outputs, so this is argmax_c S_c; written
    as the ratio to match the tolerance it is tested against), and the
    bin with the largest w_kc* is split by `two_means_split` on its own
    Ψ̃ rows (ties -> the lowest bin id). A bin whose split comes back
    infeasible is skipped rather than split, and never tried again
    (nothing about the data changes by skipping it, so it would
    otherwise be picked again every round); if every remaining bin is
    infeasible, growth stops there (nothing splittable remains, not
    `growth_capped`). Each split adds exactly one bin, counted into
    `n_growth_rounds`. Then one `_lloyd_all` pass of all L centroids
    over every row; S_pred is taken after that pass, S_pred_pre_lloyd
    before.
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


def _fit_scale(p: np.ndarray, U: np.ndarray, m: np.ndarray) -> np.ndarray:
    """a_c = sum_k p_k*U_kc*m_kc / sum_k p_k*m_kc^2: the least-squares
    scale of the measured derivative on the predicted mean, fitted once
    and held fixed through the continuation so a uniform scale error in
    the model does not itself register as disagreement."""
    num = np.sum(p[:, None] * U * m, axis=0)
    den = np.sum(p[:, None] * m ** 2, axis=0)
    return num / den


# A20 (spec/QIJ_mods_waves.md), amended by A21's architecture rule:
# `_apply_update` was copied VERBATIM from `archive/seeded-tree:
# src/qij_joint/core/seed.py` for A20's build, including its `sd_hat`
# parameter -- the GP posterior sd, carried alongside psi_hat and fed
# no decision. A21's architecture rule (module docstring) drops that
# second vector from the state entirely, so `sd_hat` is removed from
# this signature and body; the GP posterior sd survives only as
# `QIJResult.sigma`, a stored survey diagnostic `qij.py` builds on its
# own. `_run_total` is this function's only caller here.
def _apply_update(psi_hat: np.ndarray, idx: np.ndarray, U: np.ndarray,
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
            n_scale[c] += 1
            if ratio < 0.0:
                n_neg[c] += 1
        else:
            psi_hat[idx, c] += (U[c] - m[c])
            n_shift[c] += 1


def _split_task(T, case, X: np.ndarray, task):
    """One continuation round's one-shot forward evaluation, the
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
    variance quantity is void; growth's own products (needing no
    evaluation) are kept when growth ran."""
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
        V_btw=nan_q, V_win_hat=nan_q, V_tot_hat=nan_q, S_pred=S_pred, a=nan_q,
        B_hat=nan_q, a_bca=nan_q,
        L0=L0, L=L0, n_growth_rounds=n_rounds, growth_capped=capped, S_pred_pre_lloyd=S_pre,
        n_check_rounds=0, n_check_evals=0, n_level_splits=0, check_capped=False, failed=True,
        bin_mass=mass, bin_U=np.full((L0, q), np.nan), bin_m=np.full((L0, q), np.nan),
        bin_label=labels, busy_delta=busy_delta, share_final=nan_q,
        n_update_scale=np.zeros(q, dtype=int), n_update_shift=np.zeros(q, dtype=int),
        n_update_negative=np.zeros(q, dtype=int), pilot_err_btw=nan_q,
        state_psi_hat=np.full((N, q), np.nan),
    )


def _run_total(
    N: int, q: int, X: np.ndarray, counter, theta_hat: np.ndarray, psi0_all: np.ndarray,
    offset: np.ndarray, measured: Sequence[int], pool, start: Optional[np.ndarray],
    eta: float, eps: float, M_X_used: int, growth: Growth, bins: BinSet, V_btw: np.ndarray,
    groups0: Sequence[np.ndarray], U0: np.ndarray, m0: np.ndarray,
    B_hat: np.ndarray, a_bca: np.ndarray, a: np.ndarray, busy_delta: float,
) -> JointResult:
    """
    `run_joint`'s continuation (spec/QIJ_mods_waves.md A20's
    `_run_total`, the method under A21, no switch): every round splits
    whichever open leaf (or group of leaves) does the most to bring the
    currently worst measured output's own TOTAL within share S_c =
    V_win_hat_c/(V_btw_c+V_win_hat_c) back under `eps`, measured exactly
    as a forward evaluation of the small child at its own step
    (`U_large` by conservation, `Delta` on `V_btw`'s own scale), all of
    a round's evaluations in one pool batch through `_split_task`.

    THE STATE VECTOR (module docstring). `psi_hat` (N,q) is the
    per-point STATE the method reports and reads from here on: it
    starts as the A18-corrected pilot, centred (`psi0_all - offset`,
    both already in full-data units -- `run_joint`'s own correction
    runs before this function is called). Right after the stage-1
    stencils, one update pass applies `_apply_update` to every stage-1
    leaf (`groups0[k]`, `U0[k]`, at that leaf's own central-stencil step
    `t_k`); after every later split, both children (small from its own
    forward evaluation, large from conservation) get their own
    `_apply_update` call, at their own step. `theta_abs` (the update's
    own noise floor, `eta*|theta_hat|/t_k`) is `|theta_hat[measured]|`
    floored at machine eps, fixed for the whole continuation.
    `n_update_scale`/`n_update_shift`/`n_update_negative` (q,) are
    `_apply_update`'s own running counts.

    Every decision below reads `psi_hat`, never a fixed copy of the
    pilot: a leaf's own `var_hat` (population variance of `psi_hat`
    over the leaf's points, recomputed fresh the moment the leaf is
    created -- stage 1, or a split's own child -- and cached from then
    on, since nothing touches those points again until the leaf is
    itself split) drives `win_hat`/`E`; a split's own cut is
    `two_means_split` on `psi_hat[idx] / std_growth`, `std_growth =
    growth.std * a` stage 1's own FIXED, A18-scaled per-output std (a
    unit set once, never the current vector's own spread, so every
    bin's cut reads the measured values in the same units) --
    `two_means_split` demeans its own input, so the result is the same
    whether or not `psi_hat` is itself already centred. `bin_m`
    (`JointResult`'s own product) is the leaf's own m_pre -- `psi_hat`'s
    mean over the leaf's points at the moment it was measured, before
    its own update -- the SAME quantity `ebar` is computed from, never
    the fixed pilot's mean: THE ARCHITECTURE RULE (module docstring)
    leaves no fixed copy of the pilot for any reader, bin_m included.

    Ranking (spec/QIJ_mods_waves.md A20, amended 2 October -- the
    author's ruling, after an offline test that found the measured
    error recovers the within error up to 3x faster per evaluation
    than the pilot's own W): a leaf's `ebar` (q,), set once at the
    leaf's own creation (stage 1, or a split's own child), is its
    measured mean minus `psi_hat`'s mean over its own points at that
    moment, BEFORE the update below applies (`m_pre`); `delta_U` (q,),
    set at the same time, is the update's own noise term `eta*
    theta_abs/t_k` at the leaf's own step t_k. `rank_score(leaf, c)` is
    `(leaf['n']/N)*ebar[c]**2/N` when `abs(ebar[c]) > delta_U[c]`, else
    `(leaf['n']/N)*var_hat[c]/N` (== W_kc, the current vector's own
    contribution to V_win_hat,c) -- both forms divided by N so a
    leaf's score is in `win_hat`'s own units regardless of which
    branch fires.

    A round: V_win_hat is (1/N) sum_{k: n_k>1} p_k Var_k(psi_hat) over
    the CURRENT leaves; if every S_c <= eps, stop. If the evaluation
    cap (`1 + M_X_used`) is spent, stop (`check_capped`). Otherwise c* =
    argmax_c S_c/eps (eps is one scalar across outputs, kept as the
    ratio to match what it is tested against) is the binding output,
    and E = V_win_hat_c* - eps/(1-eps)*V_btw_c* is the within that must
    come off it for S_c* to clear `eps`. Open leaves (n_k > 1,
    rank_score(leaf, c*) > 0; a leaf with neither is never ranked,
    since it cannot change S_c* and cannot be split either) are ranked
    by `rank_score(leaf, c*)`, descending, ties to the lowest id; the
    smallest prefix whose summed score reaches E (at least one leaf,
    the whole ranked list if E is never reached) is proposed for the
    cut above. A leaf whose split comes back infeasible is closed
    (kept, excluded from every later round's ranking -- the data under
    it does not change, so it would otherwise be picked again); a leaf
    whose evaluation fails is closed the same way, with no second
    chance. If nothing is proposed (every open leaf already closed, or
    ranking finds nothing with positive score), the continuation stops
    there, same as growth's own "nothing splittable remains".
    `rank_rule` = 'measured_error' always (the method, no switch);
    `bin_ebar` (L,q) is every FINAL leaf's own `ebar`, beside `bin_U`/
    `bin_m`.

    `V_btw` is carried in from `run_joint`'s own full-data measurement
    of the growth bins and updated by `+= Delta` each successful split.
    `a` (`_fit_scale`), `B_hat`/`a_bca` (`bias_and_acceleration`) are
    passed through unchanged, fitted once against the growth bins
    before any continuation round, as in `run_joint`.

    `pilot_err_btw` (q,) = sum_k p_k*(U0_k-m0_k)^2/N over the stage-1
    leaves only (`bins.p`, `U0`, `m0`, all already A18-corrected): the
    measured, between-leaf part of the PILOT's own error variance, on
    V_btw's own O(1/N) scale -- a diagnostic read from the fixed pilot
    mean `m0` deliberately (it measures the pilot's own error, not the
    state vector's), unrelated to `bin_m`, which the ARCHITECTURE RULE
    requires read from the state vector instead (spec/QIJ_mods_waves.md
    A20 writes the sum with no explicit 1/N; this function divides by N
    to match V_btw's convention and the quantity it is compared
    against -- see the spec-vs-code note filed with the A20 build).
    """
    L0 = growth.L0
    centering_residual = bins.centering_residual[measured]
    delta_f = forward_step(eta)
    delta_c = central_step(eta)
    evals_cap = 1 + M_X_used
    n_check_rounds = n_check_evals = n_level_splits = 0
    check_capped = False
    next_id = L0
    closed: set = set()

    # THE STATE VECTOR (module docstring): starts as the A18-corrected
    # pilot, centred (`psi0_all`/`offset` are already a_c-corrected by
    # `run_joint` before this function is called). No second vector
    # (`sd_hat`) is carried alongside it (A21's architecture rule).
    psi_hat = psi0_all - offset[None, :]
    std_growth = growth.std * a  # stage 1's FIXED, A18-scaled growth std (a unit, set once)
    theta_abs = np.maximum(np.abs(theta_hat[measured]), np.finfo(float).eps)
    n_update_scale = np.zeros(q, dtype=int)
    n_update_shift = np.zeros(q, dtype=int)
    n_update_negative = np.zeros(q, dtype=int)

    # The stage-1 leaves' own between-leaf pilot-error variance (spec
    # "Products, added"), from the fixed pilot mean `m0` against the
    # stage-1 stencils' own `U0` -- computed once, before the update
    # pass below changes anything, since it is the PILOT's error, not
    # the state vector's.
    pilot_err_btw = np.sum(bins.p[:, None] * (U0 - m0) ** 2, axis=0) / N

    leaves: Dict[int, dict] = {
        k: dict(indices=groups0[k], n=int(bins.n[k]), U=U0[k].copy())
        for k in range(L0)
    }
    # One update pass over every stage-1 leaf (spec: "one batch, one
    # update pass"): order among leaves does not matter, each leaf's
    # own points are disjoint from every other leaf's. `ebar`/`delta_U`
    # are captured from the PRE-update mean (`m_pre`), for the ranking
    # amendment below, and ALSO become this leaf's `bin_m` -- THE
    # ARCHITECTURE RULE's own fix (module docstring): `_apply_update`
    # recomputes its own mean fresh and is not told `m_pre` (it is
    # `_apply_update`'s own local `m`, verbatim from the archive -- not
    # threaded through as a parameter).
    for k in range(L0):
        idx = groups0[k]
        t_k = step_parameter(delta_c, float(bins.p[k]))
        m_pre = psi_hat[idx].mean(axis=0)
        leaves[k]['m'] = m_pre
        leaves[k]['ebar'] = U0[k] - m_pre
        leaves[k]['delta_U'] = eta * theta_abs / t_k
        _apply_update(psi_hat, idx, U0[k], t_k, eta, theta_abs,
                      n_update_scale, n_update_shift, n_update_negative)
        leaves[k]['var_hat'] = psi_hat[idx].var(axis=0)

    def win_hat() -> np.ndarray:
        out = np.zeros(q)
        for leaf in leaves.values():
            if leaf['n'] > 1:
                out += (leaf['n'] / N) * leaf['var_hat']
        return out / N  # (1/N) sum_k p_k Var_k(psi_hat), no v (method_notes section 6)

    def rank_score(leaf: dict, c: int) -> float:
        """A20's ranking amendment (2 October): the leaf's own measured
        error when it exceeds the update's noise term, else W_kc on the
        current vector -- both in `win_hat`'s own units (divided by N)."""
        if abs(leaf['ebar'][c]) > leaf['delta_U'][c]:
            return (leaf['n'] / N) * leaf['ebar'][c] ** 2 / N
        return (leaf['n'] / N) * leaf['var_hat'][c] / N

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
        # Ranking amendment (A20, 2 October): rank_score falls back to
        # W_kc (== the old (leaf['n']/N)*var_hat[c_star]/N) when the
        # leaf's own measured error is at noise level, so a leaf that
        # was a candidate before still is; one that only has a large
        # ebar (e.g. a homogeneous leaf a SCALE update left at var_hat
        # == 0, its mean still visibly off) is now a candidate too --
        # filtering on `rank_score > 0` rather than `var_hat > 0`
        # (spec-vs-code note filed with the A20 build).
        candidates = []
        for k, leaf in sorted(leaves.items()):
            if k in closed or leaf['n'] <= 1:
                continue
            score = rank_score(leaf, c_star)
            if score > 0.0:
                candidates.append((k, score))
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
            split = two_means_split(psi_hat[idx] / std_growth)
            if split is None:
                closed.add(k)
                continue
            idx_a, idx_b = idx[split[0]], idx[split[1]]
            idx_small, idx_large = (idx_a, idx_b) if idx_a.size <= idx_b.size else (idx_b, idx_a)
            p_small = idx_small.size / N
            p_large = idx_large.size / N
            t_small = step_parameter(delta_f, p_small)
            t_large = step_parameter(delta_f, p_large)
            mask = np.zeros(N, dtype=bool)
            mask[idx_small] = True
            proposals[k] = dict(idx_small=idx_small, idx_large=idx_large,
                                 p_small=p_small, p_large=p_large, t_small=t_small,
                                 t_large=t_large, mask=mask)
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
            # ebar/delta_U/m (ranking amendment, and bin_m -- THE
            # ARCHITECTURE RULE): the child's own measured mean against
            # psi_hat's mean over its own points BEFORE its update, at
            # its own step -- captured before `_apply_update` touches
            # those points; this SAME m_pre is the child's `bin_m`.
            m_pre_small = psi_hat[meta['idx_small']].mean(axis=0)
            m_pre_large = psi_hat[meta['idx_large']].mean(axis=0)
            ebar_small = U_small - m_pre_small
            ebar_large = U_large - m_pre_large
            delta_U_small = eta * theta_abs / meta['t_small']
            delta_U_large = eta * theta_abs / meta['t_large']
            _apply_update(psi_hat, meta['idx_small'], U_small, meta['t_small'], eta,
                          theta_abs, n_update_scale, n_update_shift, n_update_negative)
            _apply_update(psi_hat, meta['idx_large'], U_large, meta['t_large'], eta,
                          theta_abs, n_update_scale, n_update_shift, n_update_negative)
            leaves[next_id] = dict(indices=meta['idx_small'], n=meta['idx_small'].size,
                                    U=U_small, m=m_pre_small, ebar=ebar_small,
                                    delta_U=delta_U_small,
                                    var_hat=psi_hat[meta['idx_small']].var(axis=0))
            leaves[next_id + 1] = dict(indices=meta['idx_large'], n=meta['idx_large'].size,
                                        U=U_large, m=m_pre_large, ebar=ebar_large,
                                        delta_U=delta_U_large,
                                        var_hat=psi_hat[meta['idx_large']].var(axis=0))
            next_id += 2

    ordered_ids = sorted(leaves.keys())
    L_final = len(ordered_ids)
    bin_mass = np.array([leaves[i]['n'] for i in ordered_ids], dtype=float) / N
    bin_U = np.array([leaves[i]['U'] for i in ordered_ids])
    bin_m = np.array([leaves[i]['m'] for i in ordered_ids])
    # Ranking amendment's own per-leaf diagnostic, beside bin_U/bin_m:
    # each FINAL leaf's own ebar at the moment it was created.
    bin_ebar = np.array([leaves[i]['ebar'] for i in ordered_ids])
    bin_label = np.empty(N, dtype=int)
    for new_id, old_id in enumerate(ordered_ids):
        bin_label[leaves[old_id]['indices']] = new_id

    V_win_hat = win_hat()
    V_tot_hat = V_btw + V_win_hat
    share_final = V_win_hat / V_tot_hat

    return JointResult(
        V_btw=V_btw, V_win_hat=V_win_hat, V_tot_hat=V_tot_hat,
        S_pred=growth.S_pred, a=a, B_hat=B_hat, a_bca=a_bca,
        L0=L0, L=L_final, n_growth_rounds=growth.n_growth_rounds,
        growth_capped=growth.growth_capped, S_pred_pre_lloyd=growth.S_pred_pre_lloyd,
        n_check_rounds=n_check_rounds, n_check_evals=n_check_evals,
        n_level_splits=n_level_splits, check_capped=check_capped, failed=False,
        bin_mass=bin_mass, bin_U=bin_U, bin_m=bin_m,
        bin_label=bin_label, busy_delta=busy_delta, share_final=share_final,
        n_update_scale=n_update_scale, n_update_shift=n_update_shift,
        n_update_negative=n_update_negative, pilot_err_btw=pilot_err_btw,
        state_psi_hat=psi_hat, rank_rule='measured_error', bin_ebar=bin_ebar,
    )


def run_joint(
    X: np.ndarray, counter, theta_hat: np.ndarray, psi0_all: np.ndarray,
    model, xvq, eta: float, eps: float, offset: np.ndarray, pool=None,
    start: np.ndarray = None, measured: Optional[Sequence[int]] = None,
) -> JointResult:
    """
    The joint second stage (spec/method_notes.md section 6;
    spec/QIJ_mods_waves.md A20/A21): grow a shared partition (`grow`),
    measure every bin through `ivq.bin_differences`, then a continuation
    that splits whichever leaf does the most for the currently worst
    output's total within share, carrying the per-point state vector
    forward throughout (module docstring), all a round's evaluations on
    `pool`. `start` (A9's continuation rule, spec/QIJ_mods_waves.md A9)
    is passed to every full-data evaluation here -- the initial bin
    measurement and every continuation split; None reproduces today's
    evaluations bit for bit. `measured` (spec/QIJ_mods_waves.md A11) is
    the absolute indices, into T's full output, that `psi0_all`/`model`
    already carry; `theta_hat`, and every raw evaluation this function
    makes (`bin_differences`'s stencils, a split's own `val`), stay
    full width, one evaluation reporting every output at no extra cost,
    and are cut down to `measured`'s columns wherever they meet a
    psi0-driven quantity -- `bias_and_acceleration`'s `B_hat`/`a_bca`
    are the one exception, read from the full bins and restricted to
    `measured` by the caller instead (`qij.py`). `measured=None`
    defaults to identity (0..q-1), reproducing today's evaluations bit
    for bit.

    `offset` (q,) is `psi0_all`'s own centering constant
    (`model.offset`, `qij.py`'s local copy), read here rather than
    `model.offset` directly so the correction below never needs to
    touch `model` itself.
    """
    N, q = psi0_all.shape
    if measured is None:
        measured = list(range(q))
    M_X_used = xvq.M_used

    if np.any(model.constant_path):
        return _failed_result(N, q)

    growth = grow(psi0_all, eps, M_X_used)
    L0 = growth.L0
    binset0 = BinSet(labels=growth.labels, n=np.bincount(growth.labels, minlength=L0),
                      U=np.zeros((L0, 0)), centering_residual=np.zeros(0), D2=np.zeros((L0, 0)),
                      M_init=L0, M_used=L0, within_share=0.0, failed=False)
    bins, busy_delta = bin_differences(X, counter, theta_hat, binset0, eta, pool, start=start)
    if bins.failed:
        return _failed_result(N, q, growth=growth, busy_delta=busy_delta)

    B_hat, a_bca = bias_and_acceleration(bins, N)
    V_btw = np.array([between_terms(bins, c) for c in measured])
    U0 = bins.U[:, measured]
    _, m0, _ = _bin_stats(psi0_all, bins.labels, L0)
    a = _fit_scale(bins.p, U0, m0)

    # A18 (spec/QIJ_mods_waves.md): a_c is the gap between SURVEY units
    # and FULL-DATA units on each output axis, applied once here, right
    # after `a` is fitted. Every survey-unit quantity read from this
    # point on -- `psi0_all` itself, `offset`, the growth-bin means
    # `m0` (the pilot's own error diagnostic, `pilot_err_btw`) -- is
    # rebound to a NEW local array in its a_c-corrected units, never
    # written in place: `psi0_all` is `influence_model.psi0`'s own
    # cache ("THE RETURNED ARRAY IS THE CACHE, not a copy: callers must
    # not write into it"), and `offset` may alias `model.offset`;
    # `a[None, :] * x` always allocates a fresh array, so the
    # cached/aliased object itself is untouched. `psi_tilde` (used only
    # by `grow` above, already run) and `growth.std` were already fixed
    # before this correction, from the PRE-correction `psi0_all` --
    # `_run_total`'s own cuts read `growth.std * a` (`std_growth`),
    # never a recomputation from the now-corrected `psi0_all`, since
    # scaling `psi0_all` and `growth.std` by the same `a_c` and then
    # dividing is NOT bit-identical, in floating point, to the
    # pre-correction ratio.
    psi0_all = psi0_all * a[None, :]
    offset = offset * a
    m0 = m0 * a[None, :]

    groups0 = [np.where(bins.labels == k)[0] for k in range(L0)]

    return _run_total(N, q, X, counter, theta_hat, psi0_all, offset, measured,
                       pool, start, eta, eps, M_X_used, growth, bins, V_btw, groups0, U0, m0,
                       B_hat, a_bca, a, busy_delta)
