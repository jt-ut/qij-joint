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

`run_refinement` is one schedule over this machinery,
`refine_schedule='queue'` (the default): the single open leaf with the
largest expected gain is split, one evaluation, before the next leaf is
even considered. `core.rounds.run_refinement_rounds`
(spec/QIJ_mods_waves.md A14) is the other, `'rounds'`: every measured
output's qualifying leaves are split together, all their evaluations
sharing one pool batch. The two share this module's per-output setup
(`prepare_coordinate`), per-leaf proposal (`new_leaf`, `propose`,
`batch_v`, `compute_rho2`) and per-split update (`apply_split`) --
everything but the selection and batching is the ported rule either
way.

`refine_trigger` (spec/QIJ_mods_waves.md A15) switches the marginal
path's own leaf-selection rule: `'gain'` (default, bit-identical) is
the ported rule above; `'measured'` is a FLAGGED LINEAGE, started at an
initial bin found to disagree with the model, run entirely differently
from an unflagged one.

1. **Entry** (item 1, unchanged since round 1): the initial bins alone
   (never a child -- `flag_leaf` is called only from
   `prepare_coordinate`) are tested against B4's flag test
   (`joint._flag_mask`, reused here at one output); a flagged initial
   bin gets priority at the front of the queue/round, ordered by
   measured discrepancy, bypassing the tau gate, until it is split.
2. **A flagged lineage ALWAYS splits GEOMETRICALLY** (item 2, round 4):
   `propose_geometric` -- a LEVEL split of the mass-weighted
   least-squares plane U = alpha + g.c fit over the cell's nearest
   ancestor with at least three currently live descendant cells
   (`_ancestor_gradient`/`_fit_plane_gradient`, c the cells' own mean
   whitened positions, no extra evaluation), split at the median of the
   cell's own members' projection onto g (`_median_projection_split`);
   `joint.two_means_split` on the whitened data coordinates
   (`whiten_columns`), deterministic principal-axis two-means, when no
   such ancestor exists (including a flagged lineage's own first split,
   which has no ancestor at all) or either step above is degenerate.
   Neither rule reads a posterior quantity (`leaf['v']`/`u`) at all.
   `from_flagged_lineage` (set True at a flagged initial bin's own
   creation, propagated parent to both children by `apply_split`,
   forever) is what marks a leaf as belonging to one; `_propose_dispatch`
   reads it to route to `propose_geometric` instead of the ported
   `propose`. `leaf['direction_rule']` records which of the two rules a
   cell's own split used, for the trace (not a product).
3. **Priority in a flagged lineage is MEASURED** (item 3): a flagged
   initial bin's own priority is its measured discrepancy (item 1,
   above, unaffected by this item); each of ITS CHILDREN, and every
   descendant after, enters with `leaf['g'] = Delta/2`, Delta the split
   that created it (`apply_split`, set directly -- never priced by a
   `propose`-style call, since there is no posterior-based expected gain
   to price). This 'g' is then compared to tau exactly as an unflagged
   leaf's predicted 'g' is (`_run_queue`/`core.rounds._select_round`
   make no distinction once past the front-of-queue initial bins).
4. **Closing** (item 4): the ported two-strike rule, everywhere,
   flagged lineage or not -- no zero-level shortcut (removed).
5. **Unflagged bins** (item 5): the ported refinement, unchanged --
   kind rule, priority, two strikes, exactly as under 'gain'.

`flag_leaf` and `propose_geometric` are `_run_queue`'s and
`core.rounds`'s shared implementation of this, exactly as
`propose`/`apply_split` are shared for the ported rule.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

from .differences import forward_step, perturbed_weights, step_parameter
from .influence_model import bin_posterior_variance
from .ivq import BinSet, between_terms, bias_and_acceleration, bin_differences, build_bins, kmeans_1d

# `core.joint` imports this module at its own top level (the CADJ gain
# helpers below); importing it back here at the top would be circular,
# so every use below (`flag_leaf`, `propose_geometric`, `prepare_coordinate`)
# imports it locally instead, by which point both modules are loaded.

__all__ = [
    "CoordinateResult", "RefineState", "run_refinement",
    "prepare_coordinate", "finalize_coordinate", "apply_split",
    "new_leaf", "propose", "propose_geometric", "flag_leaf", "batch_v",
    "compute_rho2", "split_gamma", "whiten_columns",
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
    B_hat, a_bca        (q,) `ivq.bias_and_acceleration`'s ABC bias and
                        acceleration (spec A10), from this coordinate's
                        own initial `BinSet` (NaN when unmeasured).
    M_X                 the first-stage prototype count used (an input).
    M_used              the initial I-VQ's bin count (before refinement
                        grows it).
    n_refine_evals      forward evaluations spent by the refinement
                        loop, including a cancelled split's evaluation.
    bin_U               (L, q) every T output's centered derivative on
                        this coordinate's own FINAL bins -- the same U
                        `field` is built from, kept rather than
                        discarded (spec/QIJ_mods_waves.md A14's bin_U
                        product); a caller slices this to the measured
                        columns.
    failed              True when this coordinate's own initial-bin
                        measurement failed, or stage 1 collapsed for
                        this output; `qij.py` voids every coordinate's
                        variance quantities on the draw when any one
                        coordinate's failed is True.
    busy_delta          wall time this coordinate's initial-bin pool
                        tasks spent beyond `bin_differences`'s own
                        elapsed time (method_notes section 4); 0.0
                        without a pool.
    a_c                 (spec/QIJ_mods_waves.md A15 item 6) the measured
                        trigger's fitted scale factor (B4's `a_c`), from
                        this coordinate's own initial bins; NaN under
                        `refine_trigger='gain'`.
    n_flagged, n_flag_evals, n_unflag_evals, n_geom_splits
                        (A15 item 6, round 3) initial bins flagged by the
                        measured trigger's flag test, evaluations spent
                        splitting a flagged lineage, evaluations spent
                        splitting an unflagged one (n_flag_evals +
                        n_unflag_evals == n_refine_evals), and GEOMETRIC
                        splits actually taken (every one of them in a
                        flagged lineage, item 2); all 0 under
                        `refine_trigger='gain'`.
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
    B_hat: np.ndarray
    a_bca: np.ndarray
    M_X: int
    M_used: int
    n_refine_evals: int
    bin_U: np.ndarray
    failed: bool = False
    busy_delta: float = 0.0
    a_c: float = float('nan')
    n_flagged: int = 0
    n_flag_evals: int = 0
    n_unflag_evals: int = 0
    n_geom_splits: int = 0


@dataclass
class RefineState:
    """
    One measured output's mutable refinement state, between
    `prepare_coordinate`'s initial proposals and `finalize_coordinate`'s
    `CoordinateResult` (spec/method_notes.md section 4): the shared
    handoff `run_refinement`'s own queue and `core.rounds`'s
    synchronized rounds both read and write, so the two schedules'
    leaf mechanics (`new_leaf`/`propose`/`batch_v`/`apply_split`) stay
    one implementation.

    X, counter, theta_hat, start   the shared full-data evaluation
                        inputs every split evaluation below reuses.
    coordinate, name, N, q         this output's absolute index into
                        `theta_hat`, its name, the point count, and the
                        full output width (every T output).
    psi0_c, psi_centered, sigma_c  this output's own initial-influence,
                        centered-influence and posterior-sd arrays (N,).
    I_proto_c, bmu, bmu2           this output's mass-centered
                        prototype influence, and the shared per-point
                        first/second BMU indices.
    eta, eps, M_X_used             the estimator's own step scale, the
                        cost tolerance, and the first-stage prototype
                        count the evaluation guard is built from.
    Z, model, model_index          the fitted `InfluenceModel` and this
                        output's own position in it.
    delta_f, evals_cap, centering_residual
                        the forward step scale, 1 + M_X_used, and the
                        initial bins' own bin-mass-weighted residual
                        every split's U is corrected by.
    leaves, next_id                the open-leaf pool, keyed by a
                        strictly increasing id.
    V_btw, n_refine_evals, n_level_splits, n_adjacency_splits,
    sum_measured_delta, sum_expected_g, sum_pubar2
                        the running totals `run_refinement`'s docstring
                        and spec section 4 describe.
    B_hat, a_bca, M_used, busy_delta
                        carried through from the initial-bin measurement
                        into the final `CoordinateResult`.
    refine_trigger      'gain' (ported) or 'measured' (A15 item 1:
                        entry-only flagging, always-GEOMETRIC flagged
                        lineages, measured priority -- see `flag_leaf`/
                        `propose_geometric`/`apply_split` below).
    a_c                 the measured trigger's fitted scale factor (B4),
                        from the initial bins, kept only for the final
                        `CoordinateResult` product (A15 item 6); NaN
                        under 'gain'. Never read after `prepare_coordinate`
                        sets it -- the entry-only flag test that needed it
                        runs once, there.
    Z_white             Z divided by its own per-column std
                        (`whiten_columns`), a flagged lineage's ALWAYS
                        split (A15 item 1); None under 'gain'.
    parent_id, children_of
                        A flagged lineage's own tree, id to parent id and
                        id to its two child ids (A15 item 1, round 4):
                        `apply_split` populates both, only for a flagged
                        lineage's splits, so `propose_geometric` can walk
                        a cell's ancestors and gather a nearby ancestor's
                        currently live descendant cells for its plane
                        fit. Empty dicts under 'gain'; unused entirely
                        for an unflagged leaf either way.
    n_flagged, n_flag_evals, n_unflag_evals, n_geom_splits
                        the measured trigger's own running counts (A15
                        item 6); always 0 under 'gain'.
    """

    X: np.ndarray
    counter: object
    theta_hat: np.ndarray
    start: Optional[np.ndarray]
    coordinate: int
    name: str
    N: int
    q: int
    psi0_c: np.ndarray
    psi_centered: np.ndarray
    sigma_c: np.ndarray
    I_proto_c: np.ndarray
    bmu: np.ndarray
    bmu2: np.ndarray
    eta: float
    eps: float
    M_X_used: int
    Z: np.ndarray
    model: object
    model_index: int
    delta_f: float
    evals_cap: int
    centering_residual: np.ndarray
    leaves: Dict[int, dict]
    next_id: int
    V_btw: float
    n_refine_evals: int
    n_level_splits: int
    n_adjacency_splits: int
    sum_measured_delta: float
    sum_expected_g: float
    sum_pubar2: float
    B_hat: np.ndarray
    a_bca: np.ndarray
    M_used: int
    busy_delta: float
    refine_trigger: str
    a_c: float
    Z_white: Optional[np.ndarray]
    parent_id: Dict[int, int]
    children_of: Dict[int, Tuple[int, int]]
    n_flagged: int
    n_flag_evals: int
    n_unflag_evals: int
    n_geom_splits: int


def _variance(values: np.ndarray) -> float:
    """Population variance (ddof=0); 0.0 for an empty array."""
    if values.size == 0:
        return 0.0
    return float(np.mean((values - values.mean()) ** 2))


def split_gamma(delta_realized: float, g_expected: float) -> float:
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


def new_leaf(leaf_id: int, idx: np.ndarray, U: np.ndarray, gamma: float, strike: bool,
             psi0_c: np.ndarray, psi_centered: np.ndarray) -> dict:
    """A leaf's Var(psi0) and mean depend only on its own fixed
    indices, so both are computed once here, at creation, and read
    back everywhere else (spec/method_notes.md section 4). The measured
    trigger's own state (A15 item 1, round 3), always at its
    'gain'-inert default here: `flagged`/`discrepancy` (the one-time
    entry flag test's own priority marker and ordering key -- an
    initial bin only, `flag_leaf`; never set for a child, so a child
    never re-enters the front of the queue) and `from_flagged_lineage`
    (whether this leaf descends from -- or is -- a flagged initial bin:
    ALWAYS split geometrically, priced by measured Delta/2, never a
    posterior quantity; propagated parent to child by `apply_split`,
    never by `new_leaf`, since a fresh leaf starts as its own,
    unflagged, lineage of one until proven otherwise) and
    `direction_rule` (A15 item 1, round 4: 'gradient' or
    'principal_axis', which of `propose_geometric`'s two direction
    rules this leaf's own proposed split used; None until proposed, and
    for every non-flagged-lineage leaf -- kept for the trace, not a
    product)."""
    return dict(
        id=leaf_id, indices=idx, n=int(idx.size), U=U,
        var_k=_variance(psi0_c[idx]), ubar=float(psi_centered[idx].mean()),
        open=True, split=None, g=0.0, gamma=gamma, strike=strike,
        flagged=False, discrepancy=0.0, from_flagged_lineage=False, direction_rule=None,
    )


def batch_v(leaf_list: List[dict], model, Z: np.ndarray, model_index: int,
            sigma_c: np.ndarray, with_mean: bool = False) -> None:
    """Set leaf['v'] -- the within-bin posterior variance v_k -- on
    every leaf in `leaf_list` that holds more than one point, in ONE
    call to `bin_posterior_variance` over all of them. A leaf with
    n_k <= 1 never has its `v` read (`propose` closes it first), so it
    is skipped here. `with_mean=True` (the measured trigger, A15 item 1)
    also sets leaf['u'], the posterior variance of the bin mean B4's
    flag test needs -- the same call's own second return, no second
    pass; the default path's own call and result are unchanged."""
    qualifying = [leaf for leaf in leaf_list if leaf['n'] > 1]
    if not qualifying:
        return
    groups = [leaf['indices'] for leaf in qualifying]
    if with_mean:
        v_vals, u_vals = bin_posterior_variance(model, Z, model_index, groups, sigma_c,
                                                 with_mean=True)
        for leaf, v, u in zip(qualifying, v_vals, u_vals):
            leaf['v'] = float(v)
            leaf['u'] = float(u)
    else:
        v_vals = bin_posterior_variance(model, Z, model_index, groups, sigma_c)
        for leaf, v in zip(qualifying, v_vals):
            leaf['v'] = float(v)


def compute_rho2(V_btw: float, sum_pubar2: float, N: int) -> float:
    """rho^2 = V_btw / ((1/N) sum_k p_k ubar_k^2) (spec section 4); NaN
    when that denominator is 0."""
    denom = sum_pubar2 / N
    return (V_btw / denom) if denom != 0.0 else float('nan')


def whiten_columns(Z: np.ndarray) -> np.ndarray:
    """Z divided by each column's own standard deviation over all N
    rows (spec/QIJ_mods_waves.md A15 item 1's GEOMETRIC split: 'the
    members' whitened data coordinates'); a zero-std (constant) column
    passes through unscaled rather than dividing by zero."""
    std = np.std(Z, axis=0)
    safe = np.where(std > 0.0, std, 1.0)
    return Z / safe


def _ported_kind_split(
    idx: np.ndarray, psi0_c: np.ndarray, psi_centered: np.ndarray,
    I_proto_c: np.ndarray, bmu: np.ndarray, bmu2: np.ndarray, N: int,
    p_k: float, ubar_k: float, var_k: float, v_k: float, rho2_local: float,
) -> Optional[Tuple[str, np.ndarray, np.ndarray, float]]:
    """The ported split-kind rule (spec/method_notes.md section 4): a
    level split (two-means on psi0) when Var_k(psi0) > v_k, else an
    adjacency split, falling back to level when adjacency is
    infeasible. None when no split is feasible. Used by `propose`
    (the rule as ported, for every unflagged leaf under either
    `refine_trigger` -- a flagged lineage never calls this, item 2:
    `propose_geometric` is its own, separate, always-GEOMETRIC rule)."""
    if var_k > v_k:
        result = level_split_gain(idx, psi0_c, psi_centered, N, p_k, ubar_k, rho2_local)
        kind = 'level'
    else:
        # Priced by the adjacency gain (v_k-based) whichever geometry
        # supplies the partition: the choice between level and
        # adjacency pricing follows Var_k(psi0) vs v_k above, not
        # which split happened to be feasible.
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
        return None
    idx_a, idx_b, g = result
    return kind, idx_a, idx_b, g


def propose(leaf: dict, rho2_current: float, psi0_c: np.ndarray, psi_centered: np.ndarray,
            I_proto_c: np.ndarray, bmu: np.ndarray, bmu2: np.ndarray, N: int) -> None:
    """The bin's proposed split and expected gain g, the ported rule
    (spec/method_notes.md section 4, `_ported_kind_split`).
    `rho2_current` is the rho^2 in force when the leaf was created,
    frozen into the gain at proposal time."""
    idx = leaf['indices']
    n_k = leaf['n']
    if n_k <= 1:
        leaf['open'] = False
        leaf['split'] = None
        leaf['g'] = 0.0
        return

    p_k = n_k / N
    rho2_local = rho2_current if np.isfinite(rho2_current) else 0.0
    result = _ported_kind_split(idx, psi0_c, psi_centered, I_proto_c, bmu, bmu2, N,
                                 p_k, leaf['ubar'], leaf['var_k'], leaf['v'], rho2_local)
    if result is None:
        leaf['open'] = False
        leaf['split'] = None
        leaf['g'] = 0.0
        return
    kind, idx_a, idx_b, g = result
    leaf['split'] = (kind, idx_a, idx_b)
    leaf['g'] = g
    leaf['open'] = True


def flag_leaf(leaf: dict, coordinate: int, N: int, a_c: float, V_hat_c: float,
              L: int, eps: float, m_c: float) -> bool:
    """The measured trigger's ENTRY-ONLY flag test (A15 item 1: the flag
    test runs exactly ONCE, on an initial bin, and never again on a
    child), B4's own formula reused at this single output:
    `joint._flag_mask`, called with this coordinate's own p_k/U_kc/
    m_kc/u_kc at q=1. m_kc, the mean of psi0_c over the leaf (B4), is
    leaf['ubar'] + m_c rather than a second array, since `ubar` is
    already the mean of psi0_c - m_c. Sets leaf['flagged'] and, when
    flagged, leaf['discrepancy'] = p_k*(U_kc-a_c*m_kc)^2, the queue's
    own ordering key for an as-yet-unsplit flagged initial bin; returns
    the flag. Called only from `prepare_coordinate`, on the initial bins
    -- `apply_split` never calls this on a child (item 1's whole point:
    once split, a flagged bin starts a flagged LINEAGE, tracked by
    `from_flagged_lineage`, not re-flagged). A leaf with n_k <= 1 is
    never flagged (`batch_v` never gives it a `u`, and `propose`/
    `propose_geometric` close it outright regardless, the ported rule)."""
    if leaf['n'] <= 1:
        leaf['flagged'] = False
        return False
    from .joint import _flag_mask  # local: `joint` imports this module (module docstring)
    p_k = leaf['n'] / N
    m_kc = leaf['ubar'] + m_c
    U_c = float(leaf['U'][coordinate])
    flagged = bool(_flag_mask(
        np.array([p_k]), np.array([[U_c]]), np.array([[m_kc]]),
        np.array([a_c]), np.array([V_hat_c]), np.array([[leaf['u']]]), L, eps,
    )[0])
    leaf['flagged'] = flagged
    if flagged:
        leaf['discrepancy'] = p_k * (U_c - a_c * m_kc) ** 2
    return flagged


def _live_descendants(state: "RefineState", node_id: int) -> List[int]:
    """Every CURRENTLY LIVE leaf id (open or closed, i.e. still present
    in `state.leaves`) descending from `node_id`, via `state.children_of`
    (child ids recorded at the split that created them, A15 item 1 round
    4 -- flagged-lineage bookkeeping only). An id not yet split has no
    entry in `children_of` and is itself the live leaf reached there. An
    explicit stack, not recursion: the flagged lineage's own depth is
    unbounded in principle."""
    result: List[int] = []
    stack = list(state.children_of.get(node_id, ()))
    while stack:
        nid = stack.pop()
        kids = state.children_of.get(nid)
        if kids is None:
            result.append(nid)
        else:
            stack.extend(kids)
    return result


def _fit_plane_gradient(state: "RefineState", leaf_ids: List[int]) -> Optional[np.ndarray]:
    """The mass-weighted least-squares plane U = alpha + g.c over the
    cells named by `leaf_ids` (A15 item 1, round 4): c_j each cell's own
    mean whitened position (`state.Z_white` over its members), U_j its
    own measured derivative on `state.coordinate`, p_j its mass -- no
    extra evaluation, every quantity already on the cell. Solved by
    weighted `lstsq` (rows scaled by sqrt(p_j)); returns g (d,), or None
    when the fit is degenerate (a non-finite or exactly-zero gradient --
    the cells' centroids carry no usable direction)."""
    N = state.N
    coordinate = state.coordinate
    Z_white = state.Z_white
    rows = [state.leaves[lid] for lid in leaf_ids]
    C = np.array([Z_white[leaf['indices']].mean(axis=0) for leaf in rows])
    U = np.array([leaf['U'][coordinate] for leaf in rows])
    p = np.array([leaf['n'] for leaf in rows], dtype=float) / N
    sw = np.sqrt(p)
    A = np.concatenate([np.ones((len(rows), 1)), C], axis=1) * sw[:, None]
    b = U * sw
    try:
        beta, *_ = np.linalg.lstsq(A, b, rcond=None)
    except np.linalg.LinAlgError:
        return None
    g = beta[1:]
    if not np.all(np.isfinite(g)) or not np.any(g):
        return None
    return g


def _ancestor_gradient(state: "RefineState", leaf: dict) -> Optional[np.ndarray]:
    """The plane-fit gradient at `leaf`'s NEAREST ancestor with at least
    three currently live descendant cells (A15 item 1, round 4), walking
    `state.parent_id` up from `leaf`'s own parent; None all the way to
    the lineage's root (including when `leaf` IS that root, which has no
    parent at all -- its own first split always falls back to principal
    axis, `propose_geometric`)."""
    ancestor = state.parent_id.get(leaf['id'])
    while ancestor is not None:
        descendants = _live_descendants(state, ancestor)
        if len(descendants) >= 3:
            return _fit_plane_gradient(state, descendants)
        ancestor = state.parent_id.get(ancestor)
    return None


def _median_projection_split(
    idx: np.ndarray, Z_white: np.ndarray, g: np.ndarray,
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """A level split of the measured linear model (A15 item 1, round 4):
    `idx`'s own members split at the MEDIAN of their projection onto
    `g` (`Z_white[idx] @ g`); None if that leaves one side empty (every
    member's projection ties at the median)."""
    proj = Z_white[idx] @ g
    med = np.median(proj)
    mask_a = proj <= med
    idx_a, idx_b = idx[mask_a], idx[~mask_a]
    if idx_a.size == 0 or idx_b.size == 0:
        return None
    return idx_a, idx_b


def propose_geometric(state: "RefineState", leaf: dict) -> None:
    """A flagged lineage's proposed split (A15 item 1, round 4): a LEVEL
    split of the measured linear model at the cell's nearest ancestor
    with at least three live descendant cells (`_ancestor_gradient`,
    `_median_projection_split`) when one is found; otherwise (including
    a flagged lineage's own first split, which has no ancestor at all,
    or a degenerate plane fit, or a median split that ties at one side)
    the principal-axis two-means split, as round 3 -- no posterior
    quantity (`leaf['v']`) enters either rule. Sets leaf['split'] and
    leaf['direction_rule'] ('gradient' or 'principal_axis', for the
    trace, not a product); leaf['g'] is `apply_split`'s own concern
    (item 3). No ported fallback when GEOMETRIC is itself infeasible
    either way -- the leaf simply closes, per item 2's 'no level/
    geometric choice'."""
    idx = leaf['indices']
    if leaf['n'] <= 1:
        leaf['open'] = False
        leaf['split'] = None
        return

    g = _ancestor_gradient(state, leaf)
    if g is not None:
        split = _median_projection_split(idx, state.Z_white, g)
        if split is not None:
            idx_a, idx_b = split
            leaf['split'] = ('geometric', idx_a, idx_b)
            leaf['direction_rule'] = 'gradient'
            leaf['open'] = True
            return

    from .joint import two_means_split  # local: see `flag_leaf`
    leaf['direction_rule'] = 'principal_axis'
    geo = two_means_split(state.Z_white[idx])
    if geo is None:
        leaf['open'] = False
        leaf['split'] = None
        return
    idx_a, idx_b = idx[geo[0]], idx[geo[1]]
    leaf['split'] = ('geometric', idx_a, idx_b)
    leaf['open'] = True


def _propose_dispatch(state: "RefineState", leaf: dict, rho2_current: float) -> None:
    """Propose one leaf's split: `propose_geometric` for a flagged
    lineage under `refine_trigger='measured'` (item 1), else the ported
    `propose`, EXACTLY as under 'gain' (item 5: unflagged bins are the
    ported refinement, unchanged) -- the single dispatch
    `prepare_coordinate` and `apply_split` both use, so a leaf's
    proposal never depends on which of the two called it (module
    docstring)."""
    if state.refine_trigger == 'measured' and leaf['from_flagged_lineage']:
        propose_geometric(state, leaf)
    else:
        propose(leaf, rho2_current, state.psi0_c, state.psi_centered,
                state.I_proto_c, state.bmu, state.bmu2, state.N)


def apply_split(
    state: RefineState, leaf: dict, kind: str,
    idx_small: np.ndarray, idx_large: np.ndarray, p_small: float, p_large: float,
    t_small: float, T_small: np.ndarray, failed: bool, tau_at_selection: float,
) -> None:
    """
    Apply one split's realized evaluation to `state`
    (spec/method_notes.md section 4): count the evaluation; on failure
    (spec section 5.2) close the parent and stop -- `counter`/the pool
    caller already counted it. Otherwise derive U_small/U_large by mass
    balance, update V_btw and the split counts, price gamma for the two
    children, close or propose them, and insert them (proposed with the
    rho^2 in force after this split, unless closed).

    The ported two-strike closing rule applies EVERYWHERE, flagged
    lineage or not, under either trigger (A15 item 1, item 4: zero-level
    closing is removed). The two children's priority differs
    by lineage under `refine_trigger='measured'` (item 3): a child of a
    flagged-lineage split gets leaf['g'] = Delta/2, THIS split's own
    realized gain -- a MEASURED priority, no posterior quantity, set
    here directly rather than by a `propose`-style pricing call; an
    unflagged child's g is priced by `_propose_dispatch`'s ported
    `propose` call, as under 'gain'. `from_flagged_lineage` propagates
    from parent to both children (never reset by `new_leaf`, whose
    default is unflagged) -- `n_flag_evals`/`n_unflag_evals` count every
    split by which lineage the LEAF BEING SPLIT belongs to, however many
    splits deep. `leaf['v']` (posterior within-bin variance) is priced
    for every leaf regardless of lineage -- `finalize_coordinate`'s
    V_win_hat gather needs it for every FINAL bin; item 2's 'no
    posterior quantity' is about the split-kind/pricing decision, not
    this unavoidable within-variance estimate. Shared by the queue's
    one-split-at-a-time loop (`run_refinement`) and the
    synchronized-round algorithm (`core.rounds`,
    spec/QIJ_mods_waves.md A14), which differ only in how many of these
    are applied, and in what order, between one tau recomputation and
    the next.
    """
    state.n_refine_evals += 1
    from_flagged_lineage = bool(leaf['from_flagged_lineage'])
    if from_flagged_lineage:
        state.n_flag_evals += 1
    else:
        state.n_unflag_evals += 1
    if failed:
        leaf['open'] = False
        return

    N = state.N
    U_small = (T_small - state.theta_hat) / t_small - state.centering_residual
    p_parent = leaf['n'] / N
    U_parent = leaf['U']
    U_large = (p_parent * U_parent - p_small * U_small) / p_large

    Delta = (
        p_small * U_small[state.coordinate] ** 2
        + p_large * U_large[state.coordinate] ** 2
        - p_parent * U_parent[state.coordinate] ** 2
    ) / N

    state.V_btw += Delta
    state.sum_measured_delta += Delta
    state.sum_expected_g += leaf['g']
    if kind == 'level':
        state.n_level_splits += 1
    elif kind == 'adjacency':
        state.n_adjacency_splits += 1
    else:
        state.n_geom_splits += 1

    gamma_children = split_gamma(Delta, leaf['g'])

    # Ported two-strike closing rule (spec section 4), everywhere (A15
    # item 4): a below-tolerance split flags its children rather than
    # closing them outright; only a SECOND consecutive below-tolerance
    # split in the same lineage closes both.
    children_strike = Delta < tau_at_selection
    close_children = children_strike and leaf['strike']

    del state.leaves[leaf['id']]

    leaf_small = new_leaf(state.next_id, idx_small, U_small, gamma_children, children_strike,
                           state.psi0_c, state.psi_centered)
    leaf_large = new_leaf(state.next_id + 1, idx_large, U_large, gamma_children, children_strike,
                           state.psi0_c, state.psi_centered)
    leaf_small['from_flagged_lineage'] = from_flagged_lineage
    leaf_large['from_flagged_lineage'] = from_flagged_lineage
    state.next_id += 2
    state.leaves[leaf_small['id']] = leaf_small
    state.leaves[leaf_large['id']] = leaf_large
    state.sum_pubar2 += (
        (leaf_small['n'] / N) * leaf_small['ubar'] ** 2
        + (leaf_large['n'] / N) * leaf_large['ubar'] ** 2
        - p_parent * leaf['ubar'] ** 2
    )

    # Both children need v_k regardless of what happens next: a child
    # closed immediately below is still a final bin if it is never
    # split again, and the V_win_hat gather reuses this same cached
    # value. One call prices both, every lineage alike (docstring).
    batch_v([leaf_small, leaf_large], state.model, state.Z, state.model_index, state.sigma_c)

    if from_flagged_lineage:
        # Item 3: measured priority, this split's own realized gain
        # halved -- no posterior-priced expected gain anywhere in a
        # flagged lineage.
        leaf_small['g'] = Delta / 2.0
        leaf_large['g'] = Delta / 2.0
        # Round 4: this lineage's own tree, for `propose_geometric`'s
        # ancestor search -- the parent leaf's id is already gone from
        # `state.leaves` (just deleted above), but its place in the tree
        # lives on here.
        state.children_of[leaf['id']] = (leaf_small['id'], leaf_large['id'])
        state.parent_id[leaf_small['id']] = leaf['id']
        state.parent_id[leaf_large['id']] = leaf['id']

    if close_children:
        leaf_small['open'] = False
        leaf_large['open'] = False
    else:
        rho2 = compute_rho2(state.V_btw, state.sum_pubar2, N)
        _propose_dispatch(state, leaf_small, rho2)
        _propose_dispatch(state, leaf_large, rho2)


def _degenerate_result(coordinate: int, name: str, N: int, bins0: BinSet, M_X_used: int,
                        q: int) -> CoordinateResult:
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
        B_hat=np.full(q, np.nan), a_bca=np.full(q, np.nan),
        M_X=M_X_used, M_used=bins0.M_used, n_refine_evals=0,
        bin_U=np.full((1, q), np.nan),
        failed=True,
    )


def _failed_result(coordinate: int, name: str, N: int, bins0: BinSet, M_X_used: int,
                    q: int, busy_delta: float = 0.0) -> CoordinateResult:
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
        B_hat=np.full(q, np.nan), a_bca=np.full(q, np.nan),
        M_X=M_X_used, M_used=bins0.M_used, n_refine_evals=0,
        bin_U=np.full((bins0.M_used, q), np.nan),
        failed=True, busy_delta=busy_delta,
    )


def prepare_coordinate(
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
    start: np.ndarray = None,
    model_index: int = None,
    refine_trigger: str = 'gain',
    Z_white: Optional[np.ndarray] = None,
) -> Union[CoordinateResult, RefineState]:
    """
    One estimand coordinate's initial I-VQ, its full-data measurement,
    and its leaves' first proposals (spec/method_notes.md section 4):
    the setup shared by the queue (`run_refinement`, below) and the
    synchronized-round algorithm (`core.rounds`, spec/QIJ_mods_waves.md
    A14), which differ only in how they select and batch what this
    setup proposes. Returns a terminal `CoordinateResult` when stage 1
    collapsed or the initial-bin measurement failed (spec section 5);
    otherwise a `RefineState`, its every open leaf already proposed a
    split. Arguments mirror `run_refinement`'s own. `refine_trigger`/
    `Z_white` are A15's own (module docstring); under 'gain' this
    function is exactly the ported setup.
    """
    if model_index is None:
        model_index = coordinate
    N = len(X)
    q = theta_hat.shape[0]

    bins0 = build_bins(psi0_c, eps)

    if constant_path or bins0.M_used <= 1:
        return _degenerate_result(coordinate, name, N, bins0, M_X_used, q)

    bins0, busy_delta = bin_differences(X, counter, theta_hat, bins0, eta, pool, start=start)
    if bins0.failed:
        return _failed_result(coordinate, name, N, bins0, M_X_used, q, busy_delta)

    B_hat, a_bca = bias_and_acceleration(bins0, N)
    V_btw0 = between_terms(bins0, coordinate)

    psi_centered = psi0_c - m_c
    delta_f = forward_step(eta)
    evals_cap = 1 + M_X_used

    leaves: Dict[int, dict] = {}
    next_id = 0
    for k in range(bins0.M_used):
        idx = np.where(bins0.labels == k)[0]
        leaves[next_id] = new_leaf(next_id, idx, np.asarray(bins0.U[k], dtype=float).copy(),
                                    1.0, False, psi0_c, psi_centered)
        next_id += 1

    # sum_k p_k*ubar_k^2 over the current leaves, maintained incrementally
    # at each split (subtract the parent's term, add the two children's)
    # rather than re-summed over every leaf on every call -- an
    # E8-approved reordering (spec/method_notes.md section 4).
    sum_pubar2 = sum((leaf['n'] / N) * leaf['ubar'] ** 2 for leaf in leaves.values())

    measured = refine_trigger == 'measured'
    a_c = float('nan')
    V_hat_c = float('nan')
    n_flagged = 0
    if measured:
        # B4's scale factor at this coordinate's own initial bins
        # (`joint._fit_scale`, local import: `joint` imports this
        # module, module docstring): m0 is the RAW (uncentered) bin
        # mean of psi0_c B4 itself is built from, from the same
        # `bincount`s `joint._bin_stats` uses.
        from .joint import _fit_scale
        counts0 = bins0.n.astype(float)
        sums0 = np.bincount(bins0.labels, weights=psi0_c, minlength=bins0.M_used)
        m0 = sums0 / counts0
        a_c = float(_fit_scale(bins0.p, bins0.U[:, coordinate:coordinate + 1], m0[:, None])[0])
        V_hat_c = float(np.var(psi0_c))

    state = RefineState(
        X=X, counter=counter, theta_hat=theta_hat, start=start,
        coordinate=coordinate, name=name, N=N, q=q,
        psi0_c=psi0_c, psi_centered=psi_centered, sigma_c=sigma_c,
        I_proto_c=I_proto_c, bmu=bmu, bmu2=bmu2, eta=eta, eps=eps,
        M_X_used=M_X_used, Z=Z, model=model, model_index=model_index,
        delta_f=delta_f, evals_cap=evals_cap, centering_residual=bins0.centering_residual,
        leaves=leaves, next_id=next_id, V_btw=float(V_btw0),
        n_refine_evals=0, n_level_splits=0, n_adjacency_splits=0,
        sum_measured_delta=0.0, sum_expected_g=0.0, sum_pubar2=sum_pubar2,
        B_hat=B_hat, a_bca=a_bca, M_used=bins0.M_used, busy_delta=busy_delta,
        refine_trigger=refine_trigger, a_c=a_c,
        Z_white=Z_white, parent_id={}, children_of={},
        n_flagged=0, n_flag_evals=0, n_unflag_evals=0, n_geom_splits=0,
    )
    # `with_mean=True` under 'measured' only: the entry-only flag test
    # (item 1) needs every initial leaf's own `u`; a child never needs
    # it (`apply_split`'s own `batch_v` call never asks for it), since a
    # child is never flag-tested.
    batch_v(list(leaves.values()), model, Z, model_index, sigma_c, with_mean=measured)
    if measured:
        L0 = bins0.M_used
        for leaf in leaves.values():
            # Entry-only (item 1): a flagged initial bin starts its own
            # flagged lineage right here -- it is itself split
            # geometrically (`_propose_dispatch` reads
            # `from_flagged_lineage`, set below), counted into
            # n_flag_evals from its very first split on. An unflagged
            # one is the ported refinement, unchanged (item 5).
            flagged = flag_leaf(leaf, coordinate, N, a_c, V_hat_c, L0, eps, float(m_c))
            leaf['from_flagged_lineage'] = flagged
            if flagged:
                n_flagged += 1
        state.n_flagged = n_flagged
    rho2 = compute_rho2(state.V_btw, state.sum_pubar2, N)
    for leaf in leaves.values():
        _propose_dispatch(state, leaf, rho2)
    return state


def finalize_coordinate(state: RefineState) -> CoordinateResult:
    """
    Build the final `CoordinateResult` from `state`'s open leaves
    (spec/method_notes.md section 4): rho and V_win_hat at the final
    bin set, the refined field, and the final bins' own (L, q)
    derivative array `bin_U` (spec/QIJ_mods_waves.md A14) -- every T
    output's derivative on THIS coordinate's own partition; a caller
    slices it to the measured columns.
    """
    leaves = state.leaves
    N = state.N
    final_rho2 = compute_rho2(state.V_btw, state.sum_pubar2, N)
    rho = math.sqrt(final_rho2) if (np.isfinite(final_rho2) and final_rho2 >= 0.0) else float('nan')

    V_win_hat = 0.0
    for leaf in leaves.values():
        if leaf['n'] > 1:
            V_win_hat += (leaf['n'] / N) * leaf['gamma'] * (leaf['var_k'] + leaf['v'])
    V_win_hat = (V_win_hat * final_rho2) / N if np.isfinite(final_rho2) else float('nan')
    V_tot_hat = state.V_btw + V_win_hat

    gain_ratio = (
        state.sum_measured_delta / state.sum_expected_g if state.sum_expected_g != 0.0
        else float('nan')
    )

    ordered_ids = sorted(leaves.keys())
    L = len(ordered_ids)
    labels_final = np.empty(N, dtype=int)
    U_arr = np.empty((L, state.q), dtype=float)
    ubar_arr = np.empty(L, dtype=float)
    for new_id, old_id in enumerate(ordered_ids):
        leaf = leaves[old_id]
        labels_final[leaf['indices']] = new_id
        U_arr[new_id] = leaf['U']
        ubar_arr[new_id] = leaf['ubar']

    if np.isfinite(rho):
        field = U_arr[labels_final, state.coordinate] + rho * (state.psi_centered - ubar_arr[labels_final])
    else:
        field = np.full(N, np.nan, dtype=float)

    return CoordinateResult(
        coordinate=state.coordinate, name=state.name,
        V_btw=float(state.V_btw), V_win_hat=float(V_win_hat), V_tot_hat=float(V_tot_hat),
        field=field, labels=labels_final,
        L=L, n_level_splits=state.n_level_splits, n_adjacency_splits=state.n_adjacency_splits,
        rho=float(rho), gain_ratio=float(gain_ratio),
        B_hat=state.B_hat, a_bca=state.a_bca,
        M_X=state.M_X_used, M_used=state.M_used, n_refine_evals=state.n_refine_evals,
        bin_U=U_arr, busy_delta=state.busy_delta,
        a_c=state.a_c, n_flagged=state.n_flagged, n_flag_evals=state.n_flag_evals,
        n_unflag_evals=state.n_unflag_evals, n_geom_splits=state.n_geom_splits,
    )


def _run_queue(state: RefineState) -> None:
    """
    The queue algorithm, `refine_schedule='queue'`: under
    `refine_trigger='gain'` (the default and the ported behaviour,
    spec/method_notes.md section 4) the single open leaf with the
    largest expected gain is split, one evaluation, tau recomputed
    before the next leaf is even considered. Under `'measured'` (A15
    item 1: flagging is entry-only, so `l['flagged']` is only ever true
    for an as-yet-unsplit, initial bin) every flagged open leaf is split
    BEFORE any gain-queued one, in descending measured-discrepancy order
    (ties: lower leaf id, the ported tie-break) -- a flagged initial bin
    bypasses the tau gate entirely, since it still disagrees with the
    model. Once no flagged initial bin remains, the gain queue above
    runs exactly as ported, for every OTHER leaf alike: an unflagged
    leaf's 'g' is the ported predicted gain, a flagged lineage's is the
    measured Delta/2 `apply_split` set at its own creation (item 3) --
    both compared to tau the same way, neither given priority over the
    other here.
    """
    leaves = state.leaves
    N = state.N
    measured = state.refine_trigger == 'measured'
    while True:
        open_leaves = [l for l in leaves.values() if l['open']]
        if not open_leaves:
            break
        tau = state.eps * state.V_btw / len(leaves)

        flagged_open = [l for l in open_leaves if l['flagged']] if measured else []
        if flagged_open:
            best = max(flagged_open, key=lambda l: (l['discrepancy'], -l['id']))
            if state.n_refine_evals >= state.evals_cap:
                break
        else:
            best = min(open_leaves, key=lambda l: (-l['g'], l['id']))
            if not (best['g'] >= tau):
                break
            if state.n_refine_evals >= state.evals_cap:
                break

        kind, idx_a, idx_b = best['split']
        if idx_a.size <= idx_b.size:
            idx_small, idx_large = idx_a, idx_b
        else:
            idx_small, idx_large = idx_b, idx_a

        p_small = idx_small.size / N
        p_large = idx_large.size / N
        t_small = step_parameter(state.delta_f, p_small)

        mask_small = np.zeros(N, dtype=bool)
        mask_small[idx_small] = True
        omega = perturbed_weights(np.ones(N), mask_small, t_small)
        T_small = np.asarray(
            state.counter(state.X, omega, start=state.start, eta=state.eta), dtype=float)
        failed = bool(np.any(np.isnan(T_small)))

        apply_split(state, best, kind, idx_small, idx_large, p_small, p_large, t_small,
                    T_small, failed, tau)


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
    start: np.ndarray = None,
    model_index: int = None,
    refine_trigger: str = 'gain',
    Z_white: Optional[np.ndarray] = None,
) -> CoordinateResult:
    """
    Refinement for one estimand coordinate, `refine_schedule='queue'`
    (spec/method_notes.md section 4). `counter` is the run's single
    `Counter`; every evaluation, initial-bin and refinement alike, goes
    through it. `theta_hat` is the shared full-data base value (every T
    output, spec QIJ_mods_waves.md A11); `coordinate` indexes it and
    every OTHER T-output array (the bins' own U/D2 from
    `ivq.bin_differences`, which measures every output from one shared
    evaluation). `psi0_c`/`sigma_c` are this coordinate's own
    initial-influence and posterior-sd arrays (N,); `m_c` is the
    data-mean offset. `I_proto_c` (M_X_used,) is this coordinate's own
    mass-centered prototype influence; `bmu`/`bmu2` (N,) are the
    per-point first- and (resolved) second-BMU indices, shared across
    every coordinate. `Z` and `model` (the fitted `InfluenceModel`,
    possibly fit on the measured outputs only) price v_k, the
    within-bin posterior variance, via
    `influence_model.bin_posterior_variance`, which `model_index`
    indexes into -- `model`'s OWN coordinate position, equal to
    `coordinate` only when `model` was fit on every output; defaults to
    `coordinate` for a caller that never restricts `model`. `pool`,
    when given, runs the initial-bin stencils of `ivq.bin_differences`
    as pool tasks (method_notes section 4); the queue's own split
    evaluations below always stay serial regardless -- `core.rounds`
    (spec/QIJ_mods_waves.md A14) is the schedule that batches them onto
    a pool. `start` (A9's continuation rule, spec/QIJ_mods_waves.md A9)
    and `eta` (A15's eta_full, the caller's own value) are passed to
    every full-data evaluation here -- the initial-bin stencils and
    every refinement split; `start=None` reproduces today's evaluations
    bit for bit. `refine_trigger`/`Z_white` select the leaf-selection
    rule (module docstring, A15); 'gain' with `Z_white=None` is the
    ported, bit-identical path.
    """
    setup = prepare_coordinate(
        X, counter, theta_hat, coordinate, name, psi0_c, m_c, sigma_c, I_proto_c, bmu, bmu2,
        eta, eps, M_X_used, constant_path, Z, model, pool, start, model_index,
        refine_trigger, Z_white,
    )
    if isinstance(setup, CoordinateResult):
        return setup
    _run_queue(setup)
    return finalize_coordinate(setup)
