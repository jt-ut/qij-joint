"""
Gain-driven refinement for one estimand coordinate (spec/method_notes.md
section 4): starting from the I-VQ's initial bins (`ivq.build_bins`,
measured on the full data by `ivq.bin_differences`), adaptively split
the leaf with the largest expected gain, spending one forward
evaluation per split on the smaller child and deriving the larger
child by mass balance, until the largest expected gain drops below a
tolerance eps*V_btw/L or a cost guard (at most 1 + M_X_used refinement
evaluations) binds. V_btw only ever rises through this; V_win_hat is a
predicted estimate from the FINAL bin set alone, not accumulated.

Two failure cases (spec section 5), neither retried: stage 2's
initial-bin measurement (`ivq.bin_differences`) fails, voiding this
coordinate (`_failed_result`); a refinement split's own evaluation
fails, cancelling that split (the parent stays a closed bin) while the
rest of the refinement proceeds. A third case, stage 1 itself
collapsing for this output (the constant path, or an initial I-VQ with
M_used <= 1), is `_degenerate_result`: a failed draw, never a
zero-variance one.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from .differences import forward_step, perturbed_weights, step_parameter
from .influence_model import bin_posterior_variance
from .ivq import BinSet, between_terms, bin_differences, build_bins, kmeans_1d

__all__ = [
    "CoordinateResult", "run_refinement",
    "level_gain_value", "adjacency_gain_value",
    "level_split_gain", "adjacency_split_gain",
]


@dataclass
class CoordinateResult:
    """
    One estimand coordinate's complete refinement result
    (spec/method_notes.md section 4).

    V_btw, V_win_hat, V_tot_hat  the finite-differenced between term,
                        the predicted within term, and their sum.
    field               (N,) psi_hat_i = U_{k(i)}[c] + rho*(psi0_i -
                        ubar_{k(i)}), the refined influence estimate.
    labels              (N,) int, the final I-VQ bin index per data
                        point (contiguous 0..L-1).
    L                   final bin count.
    n_level_splits, n_adjacency_splits
                        counts of each split kind actually taken.
    rho                 sqrt(V_btw / ((1/N) sum_k p_k ubar_k^2)) at the
                        final bin set (NaN if that denominator is 0).
    gain_ratio          sum of realized split gains over sum of
                        expected split gains (NaN if no split taken).
    M_X                 the first-stage prototype count used (an input).
    M_used              the initial I-VQ's bin count (before refinement
                        grows it).
    n_refine_evals      forward evaluations spent by the refinement
                        loop, including a cancelled split's evaluation.
    failed              True when this coordinate's own initial-bin
                        measurement failed, or stage 1 collapsed for
                        this output; `qij.py` voids every coordinate's
                        variance quantities on the draw when any one
                        coordinate's failed is True.
    busy_delta          wall time this coordinate's initial-bin pool
                        tasks spent beyond `bin_differences`'s own
                        elapsed time (method_notes section 4); 0.0
                        without a pool.
    """

    coordinate: int
    name: str
    V_btw: float
    V_win_hat: float
    V_tot_hat: float
    field: np.ndarray
    labels: np.ndarray
    L: int
    n_level_splits: int
    n_adjacency_splits: int
    rho: float
    gain_ratio: float
    M_X: int
    M_used: int
    n_refine_evals: int
    failed: bool = False
    busy_delta: float = 0.0


def _variance(values: np.ndarray) -> float:
    """Population variance (ddof=0); 0.0 for an empty array."""
    if values.size == 0:
        return 0.0
    return float(np.mean((values - values.mean()) ** 2))


def _split_gamma(delta_realized: float, g_expected: float) -> float:
    """gamma for the two children of a split: clip(delta_realized /
    g_expected, 0, 1) when g_expected is strictly positive and both
    quantities are finite; 1.0 otherwise -- no usable information, so
    the within term is not down-weighted (spec/method_notes.md
    section 4)."""
    if g_expected > 0.0 and math.isfinite(g_expected) and math.isfinite(delta_realized):
        return min(1.0, max(0.0, delta_realized / g_expected))
    return 1.0


def _try_level_split(idx: np.ndarray, psi_leaf: np.ndarray) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Level split: two-means (`ivq.kmeans_1d`) on the leaf's psi0
    values; None if the leaf has fewer than 2 distinct values, or if
    k-means or the split collapses to one non-empty side."""
    distinct = np.unique(psi_leaf)
    if distinct.size < 2:
        return None
    labels2, protos2 = kmeans_1d(psi_leaf, 2)
    if protos2.size < 2:
        return None
    idx_a = idx[labels2 == 0]
    idx_b = idx[labels2 == 1]
    if idx_a.size == 0 or idx_b.size == 0:
        return None
    return idx_a, idx_b


def _try_adjacency_split(
    idx: np.ndarray, I_proto_c: np.ndarray, bmu: np.ndarray, bmu2: np.ndarray,
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """
    The CADJ adjacency split (spec/method_notes.md section 4): the
    bin's points with I_proto[bmu2[i], c] > I_proto[bmu[i], c] against
    the rest; ties (equal, or a NaN prototype influence when a
    prototype's own evaluation failed) go to "the rest". None if one
    side is empty; the caller falls back to a level split.
    """
    diff = I_proto_c[bmu2[idx]] - I_proto_c[bmu[idx]]
    mask = diff > 0.0
    if not mask.any() or mask.all():
        return None
    return idx[mask], idx[~mask]


def level_gain_value(p_a, ubar_a, p_b, ubar_b, p_k, ubar_k, N, rho2) -> float:
    """rho2*(p_a*ubar_a^2 + p_b*ubar_b^2 - p_k*ubar_k^2)/N (spec section
    4) at a KNOWN partition: the marginal path passes rho2=rho^2, the
    joint check (spec/method_notes.md section 6) rho2=1 per output."""
    return (rho2 * (p_a * ubar_a ** 2 + p_b * ubar_b ** 2 - p_k * ubar_k ** 2)) / N


def adjacency_gain_value(p_k: float, v_k: float, N: int, rho2: float) -> float:
    """rho2*p_k*v_k/N (spec section 4): the marginal path passes
    rho2=rho^2, the joint check (method_notes section 6) rho2=1."""
    return (rho2 * p_k * v_k) / N


def level_split_gain(
    idx: np.ndarray, psi0_c: np.ndarray, psi_centered_c: np.ndarray,
    N: int, p_k: float, ubar_k: float, rho2: float,
) -> Optional[Tuple[np.ndarray, np.ndarray, float]]:
    """Propose a level split (two-means on psi0_c, `_try_level_split`)
    and price it with `level_gain_value` (spec section 4); None if
    fewer than two distinct psi0 values or the split collapses."""
    split = _try_level_split(idx, psi0_c[idx])
    if split is None:
        return None
    idx_a, idx_b = split
    ubar_a = float(psi_centered_c[idx_a].mean())
    ubar_b = float(psi_centered_c[idx_b].mean())
    p_a, p_b = idx_a.size / N, idx_b.size / N
    g = level_gain_value(p_a, ubar_a, p_b, ubar_b, p_k, ubar_k, N, rho2)
    return idx_a, idx_b, g


def adjacency_split_gain(
    idx: np.ndarray, I_proto_c: np.ndarray, bmu: np.ndarray, bmu2: np.ndarray,
    N: int, p_k: float, v_k: float, rho2: float,
) -> Optional[Tuple[np.ndarray, np.ndarray, float]]:
    """Propose the CADJ adjacency split (`_try_adjacency_split`) and
    price it with `adjacency_gain_value` (spec section 4); None if one
    side would be empty."""
    split = _try_adjacency_split(idx, I_proto_c, bmu, bmu2)
    if split is None:
        return None
    idx_a, idx_b = split
    g = adjacency_gain_value(p_k, v_k, N, rho2)
    return idx_a, idx_b, g


def _degenerate_result(coordinate: int, name: str, N: int, bins0: BinSet, M_X_used: int) -> CoordinateResult:
    """Stage 1 collapsed for this output (the constant path, or an
    initial I-VQ with M_used <= 1): one synthetic bin holding every
    point, every variance quantity NaN, failed=True -- a failed draw,
    never a zero-variance result, since nothing was measured."""
    field = np.full(N, np.nan, dtype=float)
    labels = np.zeros(N, dtype=int)
    return CoordinateResult(
        coordinate=coordinate, name=name,
        V_btw=float('nan'), V_win_hat=float('nan'), V_tot_hat=float('nan'),
        field=field, labels=labels,
        L=1, n_level_splits=0, n_adjacency_splits=0,
        rho=float('nan'), gain_ratio=float('nan'),
        M_X=M_X_used, M_used=bins0.M_used, n_refine_evals=0,
        failed=True,
    )


def _failed_result(coordinate: int, name: str, N: int, bins0: BinSet, M_X_used: int,
                    busy_delta: float = 0.0) -> CoordinateResult:
    """Stage 2's initial-bin measurement failed (`ivq.bin_differences`):
    every per-output variance quantity is NaN; the bins stage 2
    genuinely built before the failure (`bins0.labels`/`M_used`) are
    kept. No refinement is attempted."""
    field = np.full(N, np.nan, dtype=float)
    return CoordinateResult(
        coordinate=coordinate, name=name,
        V_btw=float('nan'), V_win_hat=float('nan'), V_tot_hat=float('nan'),
        field=field, labels=bins0.labels,
        L=bins0.M_used, n_level_splits=0, n_adjacency_splits=0,
        rho=float('nan'), gain_ratio=float('nan'),
        M_X=M_X_used, M_used=bins0.M_used, n_refine_evals=0,
        failed=True, busy_delta=busy_delta,
    )


def run_refinement(
    X: np.ndarray,
    counter,
    theta_hat: np.ndarray,
    coordinate: int,
    name: str,
    psi0_c: np.ndarray,
    m_c: float,
    sigma_c: np.ndarray,
    I_proto_c: np.ndarray,
    bmu: np.ndarray,
    bmu2: np.ndarray,
    eta: float,
    eps: float,
    M_X_used: int,
    constant_path: bool,
    Z: np.ndarray,
    model,
    pool=None,
) -> CoordinateResult:
    """
    Refinement for one estimand coordinate (spec/method_notes.md
    section 4). `counter` is the run's single `Counter`; every
    evaluation, initial-bin and refinement alike, goes through it.
    `theta_hat` is the shared full-data base value (all q coordinates);
    `psi0_c`/`sigma_c` are this coordinate's own initial-influence and
    posterior-sd arrays (N,); `m_c` is the data-mean offset. `I_proto_c`
    (M_X_used,) is this coordinate's own mass-centered prototype
    influence; `bmu`/`bmu2` (N,) are the per-point first- and
    (resolved) second-BMU indices, shared across every coordinate.
    `Z` and `model` (the fitted `InfluenceModel`) price v_k, the
    within-bin posterior variance, via
    `influence_model.bin_posterior_variance`. `pool`, when given, runs
    the initial-bin stencils of `ivq.bin_differences` as pool tasks
    (method_notes section 4); the refinement loop's own split
    evaluations below stay serial regardless.
    """
    N = len(X)

    bins0 = build_bins(psi0_c, eps)

    if constant_path or bins0.M_used <= 1:
        return _degenerate_result(coordinate, name, N, bins0, M_X_used)

    bins0, busy_delta = bin_differences(X, counter, theta_hat, bins0, eta, pool)
    if bins0.failed:
        return _failed_result(coordinate, name, N, bins0, M_X_used, busy_delta)

    V_btw0 = between_terms(bins0, coordinate)

    psi_centered = psi0_c - m_c
    delta_f = forward_step(eta)
    evals_cap = 1 + M_X_used

    def new_leaf(leaf_id: int, idx: np.ndarray, U: np.ndarray, gamma: float, strike: bool) -> dict:
        """A leaf's Var(psi0) and mean depend only on its own fixed
        indices, so both are computed once here, at creation, and read
        back everywhere else (spec/method_notes.md section 4)."""
        return dict(
            id=leaf_id, indices=idx, n=int(idx.size), U=U,
            var_k=_variance(psi0_c[idx]), ubar=float(psi_centered[idx].mean()),
            open=True, split=None, g=0.0, gamma=gamma, strike=strike,
        )

    leaves: Dict[int, dict] = {}
    next_id = 0
    for k in range(bins0.M_used):
        idx = np.where(bins0.labels == k)[0]
        leaves[next_id] = new_leaf(next_id, idx, np.asarray(bins0.U[k], dtype=float).copy(), 1.0, False)
        next_id += 1

    V_btw = float(V_btw0)
    n_refine_evals = 0
    n_level_splits = 0
    n_adjacency_splits = 0
    sum_measured_delta = 0.0
    sum_expected_g = 0.0
    # sum_p_ubar2 = sum_k p_k*ubar_k^2 over the current leaves, maintained
    # incrementally at each split (subtract the parent's term, add the
    # two children's) rather than re-summed over every leaf on every
    # call -- an E8-approved reordering (spec/method_notes.md section 4).
    sum_pubar2 = sum((leaf['n'] / N) * leaf['ubar'] ** 2 for leaf in leaves.values())

    def compute_rho2() -> float:
        denom = sum_pubar2 / N
        return (V_btw / denom) if denom != 0.0 else float('nan')

    def batch_v(leaf_list: List[dict]) -> None:
        """Set leaf['v'] -- the within-bin posterior variance v_k -- on
        every leaf in `leaf_list` that holds more than one point, in
        ONE call to `bin_posterior_variance` over all of them. A leaf
        with n_k <= 1 never has its `v` read (`propose` closes it
        first), so it is skipped here."""
        qualifying = [leaf for leaf in leaf_list if leaf['n'] > 1]
        if not qualifying:
            return
        groups = [leaf['indices'] for leaf in qualifying]
        v_vals = bin_posterior_variance(model, Z, coordinate, groups, sigma_c)
        for leaf, v in zip(qualifying, v_vals):
            leaf['v'] = float(v)

    def propose(leaf: dict, rho2_current: float) -> None:
        """The bin's proposed split and expected gain g
        (spec/method_notes.md section 4): a level split (two-means on
        psi0) when Var_k(psi0) > v_k, otherwise an adjacency split,
        falling back to a level split when the adjacency split is
        degenerate. `rho2_current` is the rho^2 in force when the leaf
        was created, frozen into the gain at proposal time."""
        idx = leaf['indices']
        n_k = leaf['n']
        if n_k <= 1:
            leaf['open'] = False
            leaf['split'] = None
            leaf['g'] = 0.0
            return

        var_k = leaf['var_k']
        v_k = leaf['v']
        p_k = n_k / N
        ubar_k = leaf['ubar']
        rho2_local = rho2_current if np.isfinite(rho2_current) else 0.0

        if var_k > v_k:
            result = level_split_gain(idx, psi0_c, psi_centered, N, p_k, ubar_k, rho2_local)
            kind = 'level'
        else:
            # Priced by the adjacency gain (v_k-based) whichever
            # geometry supplies the partition: the choice between level
            # and adjacency pricing follows Var_k(psi0) vs v_k above,
            # not which split happened to be feasible.
            result = adjacency_split_gain(idx, I_proto_c, bmu, bmu2, N, p_k, v_k, rho2_local)
            kind = 'adjacency'
            if result is None:
                split = _try_level_split(idx, psi0_c[idx])
                kind = 'level'
                result = (
                    None if split is None
                    else (split[0], split[1], adjacency_gain_value(p_k, v_k, N, rho2_local))
                )
        if result is None:
            leaf['open'] = False
            leaf['split'] = None
            leaf['g'] = 0.0
            return
        idx_a, idx_b, g = result
        leaf['split'] = (kind, idx_a, idx_b)
        leaf['g'] = g
        leaf['open'] = True

    rho2 = compute_rho2()
    batch_v(list(leaves.values()))
    for leaf in leaves.values():
        propose(leaf, rho2)

    while True:
        open_leaves = [l for l in leaves.values() if l['open']]
        if not open_leaves:
            break
        tau = eps * V_btw / len(leaves)
        best = min(open_leaves, key=lambda l: (-l['g'], l['id']))
        if not (best['g'] >= tau):
            break
        if n_refine_evals >= evals_cap:
            break

        tau_at_selection = tau
        kind, idx_a, idx_b = best['split']
        if idx_a.size <= idx_b.size:
            idx_small, idx_large = idx_a, idx_b
        else:
            idx_small, idx_large = idx_b, idx_a

        p_small = idx_small.size / N
        p_large = idx_large.size / N
        t_small = step_parameter(delta_f, p_small)

        mask_small = np.zeros(N, dtype=bool)
        mask_small[idx_small] = True
        omega = perturbed_weights(np.ones(N), mask_small, t_small)
        T_small = np.asarray(counter(X, omega), dtype=float)
        n_refine_evals += 1

        if np.any(np.isnan(T_small)):
            # This split's own evaluation failed (spec section 5.2):
            # cancel it -- the parent stays a bin and is closed, so
            # the loop does not try it again; `counter` already
            # counted the failure. Nothing is retried.
            best['open'] = False
            continue

        U_small = (T_small - theta_hat) / t_small - bins0.centering_residual
        p_parent = best['n'] / N
        U_parent = best['U']
        U_large = (p_parent * U_parent - p_small * U_small) / p_large

        Delta = (
            p_small * U_small[coordinate] ** 2
            + p_large * U_large[coordinate] ** 2
            - p_parent * U_parent[coordinate] ** 2
        ) / N

        V_btw += Delta
        sum_measured_delta += Delta
        sum_expected_g += best['g']
        if kind == 'level':
            n_level_splits += 1
        else:
            n_adjacency_splits += 1

        gamma_children = _split_gamma(Delta, best['g'])

        # Two-strike closing rule (spec section 4): a below-tolerance
        # split flags its children rather than closing them outright;
        # only a SECOND consecutive below-tolerance split in the same
        # lineage closes both.
        children_strike = Delta < tau_at_selection
        close_children = children_strike and best['strike']

        del leaves[best['id']]

        leaf_small = new_leaf(next_id, idx_small, U_small, gamma_children, children_strike)
        leaf_large = new_leaf(next_id + 1, idx_large, U_large, gamma_children, children_strike)
        next_id += 2
        leaves[leaf_small['id']] = leaf_small
        leaves[leaf_large['id']] = leaf_large
        sum_pubar2 += (
            (leaf_small['n'] / N) * leaf_small['ubar'] ** 2
            + (leaf_large['n'] / N) * leaf_large['ubar'] ** 2
            - p_parent * best['ubar'] ** 2
        )

        # Both children need v_k regardless of what happens next: a
        # child closed immediately below is still a final bin if it is
        # never split again, and the V_win_hat gather reuses this same
        # cached value. One call prices both.
        batch_v([leaf_small, leaf_large])

        if close_children:
            leaf_small['open'] = False
            leaf_large['open'] = False
        else:
            rho2 = compute_rho2()
            propose(leaf_small, rho2)
            propose(leaf_large, rho2)

    final_rho2 = compute_rho2()
    rho = math.sqrt(final_rho2) if (np.isfinite(final_rho2) and final_rho2 >= 0.0) else float('nan')

    V_win_hat = 0.0
    for leaf in leaves.values():
        if leaf['n'] > 1:
            V_win_hat += (leaf['n'] / N) * leaf['gamma'] * (leaf['var_k'] + leaf['v'])
    V_win_hat = (V_win_hat * final_rho2) / N if np.isfinite(final_rho2) else float('nan')
    V_tot_hat = V_btw + V_win_hat

    gain_ratio = (sum_measured_delta / sum_expected_g) if sum_expected_g != 0.0 else float('nan')

    ordered_ids = sorted(leaves.keys())
    L = len(ordered_ids)
    labels_final = np.empty(N, dtype=int)
    U_arr = np.empty((L, theta_hat.shape[0]), dtype=float)
    ubar_arr = np.empty(L, dtype=float)
    for new_id, old_id in enumerate(ordered_ids):
        leaf = leaves[old_id]
        labels_final[leaf['indices']] = new_id
        U_arr[new_id] = leaf['U']
        ubar_arr[new_id] = leaf['ubar']

    if np.isfinite(rho):
        field = U_arr[labels_final, coordinate] + rho * (psi_centered - ubar_arr[labels_final])
    else:
        field = np.full(N, np.nan, dtype=float)

    return CoordinateResult(
        coordinate=coordinate, name=name,
        V_btw=float(V_btw), V_win_hat=float(V_win_hat), V_tot_hat=float(V_tot_hat),
        field=field, labels=labels_final,
        L=L, n_level_splits=n_level_splits, n_adjacency_splits=n_adjacency_splits,
        rho=float(rho), gain_ratio=float(gain_ratio),
        M_X=M_X_used, M_used=bins0.M_used, n_refine_evals=n_refine_evals,
        busy_delta=busy_delta,
    )
