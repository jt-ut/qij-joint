"""
The unified measured loop (spec/QIJ_unified_loop_spec.md; spec/QIJ_mods_
waves.md A20's state vector/update rule, A21's one-path removals): ONE
tree, grown from ONE leaf holding every point, every split measured,
one per-point state vector psi_hat, one calibrated stop. This replaces
everything this module used to do between the pilot and the report --
stage-1 growth (`grow`), the stencil pass over growth bins
(`ivq.bin_differences`), A18's `a`/`_fit_scale` step, and `_run_total`'s
pilot-share stop -- with a single loop (`run_joint`) that:

  * starts from one leaf (the root, all N points); the root needs NO
    evaluation (U_root == 0 exactly: influences sum to zero over the
    data) -- its own "update" is the CENTRING shift psi_hat <- psi_hat
    - mean(psi_hat), applied once, unconditionally (spec 4.0);
  * every later split measures its SMALL child by a CENTRAL pair at the
    child's own step (`differences.central_step`/`step_parameter`,
    exactly the stencils' own form), the large child by conservation;
  * ranks open leaves by measured error against the update's own noise
    term (A20's amendment, `calibrate.rank_scores`), falls back to the
    leaf's own within contribution W;
  * sizes a round by the EXPECTED gain of each candidate's own cut
    (f_l*W_l*kappa == G_l*kappa algebraically, spec 4.2) against the
    stop's excess on the binding output (`calibrate.round_prefix`);
  * closes a leaf whose cut is infeasible, or whose expected gain is
    below the split's own noise floor n_Delta + b_Delta on every
    output (spec 4.1, the measured-check spec's terms, "as today");
  * stops when `calibrate.stop_test` says so (kappa-calibrated within,
    its own standard error, both margined by `z`, against `eps` of the
    total, with at least `n_min` splits made); caps at `L_max` leaves.

THE ARCHITECTURE RULE (spec/QIJ_mods_waves.md A20, governs this module):
ONE per-point vector psi_hat exists. It starts as the pilot (`psi0_all`,
survey units, centred by `offset` then by its own mean -- the root's
update, spec 4.0); every measurement -- a split's two children -- then
rewrites it in place (`_apply_update`, VERBATIM from `archive/seeded-
tree:core/seed.py`, A20). Every later reader -- the ranking, a leaf's
own W, a cut, V_win, the noise floor, the end-of-run estimate -- reads
that one vector, never a fixed copy of anything else. `bin_m` is the
leaf's own `ebar`-moment reference value m_pre (psi_hat's mean over the
leaf's points at measurement time, before its own update), the same
quantity `ebar` is built from.

A18's survey-to-full-data scale is NOT a separate step any more (the
old `_fit_scale`/`a` is gone, spec 4.0): it enters bin by bin, through
the ordinary update rule, the moment the first splits' children are
measured and `_apply_update` rewrites them onto the full-data scale.

Units (spec section 3): W_lc = p_l*Var_l(psi_hat)/N; D_sc (realized)
and G_sc (predicted) = (n_a*(U_a-U_k)**2 + n_b*(U_b-U_k)**2)/N**2, G
from psi_hat's own pre-update child means, D from the measured ones;
V_btw = 0 at the root, += D_s per split (today's V_btw, exactly, with
every leaf measured). `core/calibrate.py` (Agent B's module, imported
below) carries every estimator/stop/ranking/round-sizing formula that
consumes these units; this module supplies them.

`JointResult.state_psi_hat` carries the final vector out; `qij.py`
takes it directly as the method's own `psi_hat` (no field+rho
reconstruction). `B_hat`/`a_bca` (the ABC interval's bias/acceleration
ingredients, spec A10) are ruled by the spec's section 4.5 (the
reviewer's audit, 2 October 2026), not left to this module's own
judgement: `a_bca` is formed over the FINAL LEAVES -- a true
partition, every leaf carrying a U (measured or by conservation,
`bin_mass`/`bin_U` below) -- with exactly `ivq.bias_and_acceleration`'s
own `a` formula; `B_hat` is NaN on every output always (its summand,
a central pair's second difference, exists only for a measured small
child and is not conservable to the large child, so no partition of
the final leaves carries it -- the ABC bias stays parked by the
author's standing ruling, normal intervals only). Both are now at
MEASURED width (q, matching `bin_U`/`bin_mass`), not `theta_hat`'s
full q_full: the final leaves' own U is already measured-width (A11
restricts `psi0_all`/`model` before this module ever sees them), so
carrying `a_bca` at that same width is the simpler of the spec's two
options and `qij.py` no longer slices `jr.a_bca`/`jr.B_hat` by
`measured` -- it takes them as given (a judgement call, reported with
the build). The old per-split small-child accumulation over every
depth (double-counting nested regions, Sigma p != 1) is removed.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple

import numpy as np

from ..parallel import call_T
from .calibrate import kappa_hat, rank_scores, round_prefix, stop_test, vwin_hat
from .differences import central_step, perturbed_weights, step_parameter

__all__ = ["JointResult", "run_joint", "two_means_split"]


@dataclass
class JointResult:
    """The unified loop's complete result (spec/QIJ_unified_loop_spec.md
    section 4.5; interface spec/QIJ_mods_waves.md A20/A21). Every array
    is per measured output unless noted.

    V_btw, V_win_hat, se_V_win, V_tot_hat (= V_btw + V_win_hat), kappa,
    se_kappa, margin (X_c/(eps*V_tot_hat) at termination): the stop's
    own report (spec 4.4/4.5). B_hat, a_bca: the ABC interval's
    ingredients, AT MEASURED WIDTH (q) -- see module docstring. B_hat
    is NaN on every output always; a_bca is formed over the final
    leaves (bin_mass, bin_U below), NaN where the leaf-level
    denominator is 0.

    L (final leaf count), n_splits, n_rounds, n_evals (== 2*the number
    of split attempts, successful or not), capped (L_max bound),
    stop_met (the loop ended because `calibrate.stop_test` said so, as
    opposed to the cap or nothing left to split), failed (the shared
    constant-path pre-check failed; every variance quantity here is
    void, as `qij.py` itself NaNs them out).

    Bin constituents, one row per FINAL leaf: bin_mass (n/N), bin_n,
    bin_U (measured mean), bin_m (= m_pre, psi_hat's mean over the
    leaf's points just before its own update -- THE ARCHITECTURE RULE,
    module docstring), bin_ebar (= bin_U - bin_m, the ranking's own
    diagnostic), bin_W (p*Var(psi_hat)/N, the leaf's own within
    contribution); bin_label (N,) the point->leaf map.

    Split constituents, one row per split (successful only): the
    parent id, both children's ids, G (predicted) and D (realized),
    and the round it was made in.

    Round constituents, one row per round attempted: kappa/se_kappa/
    V_win/margin as computed at that round's own start (what sized and
    gated it), and the number of splits that round actually completed.

    n_update_scale/shift/negative: `_apply_update`'s own running counts
    over every measured bin (both children of every split; the root's
    own centring shift is NOT one of `_apply_update`'s calls and is not
    counted here -- see `run_joint`). state_psi_hat: the final vector,
    `qij.py`'s own psi_hat. rank_rule: 'measured_error' always (the
    method, no switch). z, n_min, L_max: the levers this run used
    (L_max resolved to M_X_used when the caller passed None).
    busy_delta: as the other pool stages report it.
    """

    V_btw: np.ndarray
    V_win_hat: np.ndarray
    se_V_win: np.ndarray
    V_tot_hat: np.ndarray
    kappa: np.ndarray
    se_kappa: np.ndarray
    margin: np.ndarray
    B_hat: np.ndarray
    a_bca: np.ndarray
    L: int
    n_splits: int
    n_rounds: int
    n_evals: int
    capped: bool
    stop_met: bool
    failed: bool
    bin_mass: np.ndarray
    bin_U: np.ndarray
    bin_m: np.ndarray
    bin_ebar: np.ndarray
    bin_W: np.ndarray
    bin_n: np.ndarray
    bin_label: np.ndarray
    split_parent: np.ndarray
    split_child_a: np.ndarray
    split_child_b: np.ndarray
    split_G: np.ndarray
    split_D: np.ndarray
    split_round: np.ndarray
    split_W_parent: np.ndarray   # (S,q) the parent's W at the moment of its split (diagnostic)
    split_n_parent: np.ndarray   # (S,) the parent's point count
    split_floor: np.ndarray      # (S,q) the split's noise floor n_Delta + b_Delta (diagnostic)
    round_kappa: np.ndarray
    round_se_kappa: np.ndarray
    round_V_win: np.ndarray
    round_margin: np.ndarray
    round_n_splits: np.ndarray
    n_update_scale: np.ndarray
    n_update_shift: np.ndarray
    n_update_negative: np.ndarray
    state_psi_hat: np.ndarray
    rank_rule: str
    z: float
    n_min: int
    L_max: int
    busy_delta: float


def two_means_split(rows: np.ndarray) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Deterministic two-means on `rows` (n, q), the split rule the loop
    uses at every round (spec/QIJ_unified_loop_spec.md 4.2): initial
    centroids are the means of the two halves of `rows` split at the
    median of the projection onto the first principal axis (an eigh of
    the rows' own q x q covariance, q the output count, never a
    prototype-sized array), then Lloyd iterations (nearest centroid,
    ties to the lower one) until assignments stop changing or 100
    iterations. None if `rows` holds fewer than two distinct rows, or a
    Lloyd update would empty one side. Returns local (idx_a, idx_b)
    into `rows`' own 0..n-1."""
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


# A20 (spec/QIJ_mods_waves.md), amended by A21's architecture rule:
# `_apply_update` was copied VERBATIM from `archive/seeded-tree:
# src/qij_joint/core/seed.py` for A20's build, including its `sd_hat`
# parameter -- the GP posterior sd, carried alongside psi_hat and fed
# no decision. A21's architecture rule drops that second vector from
# the state entirely, so `sd_hat` is removed from this signature and
# body; the GP posterior sd survives only as `QIJResult.sigma`. `
# run_joint` is this function's only caller here; kept module-level and
# verbatim across the unified-loop rewrite (interface spec, item A).
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


def _central_task(T, case, X: np.ndarray, task):
    """One round's one-shot central-pair evaluation, on the pool (the
    same failure boundary as `ivq._bin_task`'s pool path -- module
    docstring, "the central-stencil machinery and its pool batching"):
    `task` is (leaf id, signed step t, member mask, start, eta);
    `call_T` applies the estimator's prepared state, the start-
    continuation rule (A9) and the per-call eta override (A15). Two
    calls per leaf (+t, -t) go into the SAME pool batch as every other
    chosen leaf's pair, in task order, so pairing by position
    (`run_joint`'s own `2*i`/`2*i+1`) is bit-identical across worker
    counts. Returns (leaf id, signed t, evaluation, failure flag, this
    call's own wall time)."""
    leaf_id, t, mask, start, eta = task
    omega = perturbed_weights(np.ones(len(X)), mask, t)
    t0 = time.perf_counter()
    try:
        result = call_T(T, X, omega, start, eta)
        result = np.asarray(result, dtype=float)
        failed = bool(np.any(np.isnan(result)))
    except Exception:
        result = np.full(len(T.outputs), np.nan)
        failed = True
    return leaf_id, t, result, failed, time.perf_counter() - t0


def _failed_result(N: int, q: int, z: float, n_min: int, L_max: int,
                    busy_delta: float = 0.0) -> JointResult:
    """A failed draw: the shared constant-path pre-check failed before
    any evaluation. Every variance/leaf/split/round quantity is void;
    `qij.py` NaNs the per-output fields it reads directly, so the exact
    fill here matters only for shape. `B_hat`/`a_bca` are now at
    MEASURED width (q), same as every other per-output field here --
    module docstring."""
    nan_q = np.full(q, np.nan)
    return JointResult(
        V_btw=nan_q, V_win_hat=nan_q, se_V_win=nan_q, V_tot_hat=nan_q,
        kappa=nan_q, se_kappa=nan_q, margin=nan_q, B_hat=nan_q, a_bca=nan_q,
        L=0, n_splits=0, n_rounds=0, n_evals=0, capped=False, stop_met=False, failed=True,
        bin_mass=np.zeros(0), bin_U=np.zeros((0, q)), bin_m=np.zeros((0, q)),
        bin_ebar=np.zeros((0, q)), bin_W=np.zeros((0, q)), bin_n=np.zeros(0, dtype=int),
        bin_label=np.full(N, -1, dtype=int),
        split_parent=np.zeros(0, dtype=int), split_child_a=np.zeros(0, dtype=int),
        split_child_b=np.zeros(0, dtype=int), split_G=np.zeros((0, q)),
        split_D=np.zeros((0, q)), split_round=np.zeros(0, dtype=int),
        split_W_parent=np.zeros((0, q)), split_n_parent=np.zeros(0, dtype=int),
        split_floor=np.zeros((0, q)),
        round_kappa=np.zeros((0, q)), round_se_kappa=np.zeros((0, q)),
        round_V_win=np.zeros((0, q)), round_margin=np.zeros((0, q)),
        round_n_splits=np.zeros(0, dtype=int),
        n_update_scale=np.zeros(q, dtype=int), n_update_shift=np.zeros(q, dtype=int),
        n_update_negative=np.zeros(q, dtype=int), state_psi_hat=np.full((N, q), np.nan),
        rank_rule='measured_error', z=z, n_min=n_min, L_max=L_max, busy_delta=busy_delta,
    )


def run_joint(
    X: np.ndarray, counter, theta_hat: np.ndarray, psi0_all: np.ndarray,
    model, xvq, eta: float, eps: float, offset: np.ndarray, pool=None,
    start: np.ndarray = None, measured: Optional[Sequence[int]] = None, *,
    z: float = 2.0, n_min: int = 30, L_max: Optional[int] = None,
) -> JointResult:
    """
    The unified measured loop (module docstring; spec/QIJ_unified_loop_
    spec.md sections 0, 4). `psi0_all` (N, q) is the GP pilot's point
    predictions, survey units, already restricted to `measured`'s
    columns by the caller (as `model` is); `theta_hat` stays FULL width
    (q_full) as passed, for interface compatibility with `qij.py`'s
    existing convention, but this module reads only `theta_hat[measured]`
    (`theta_abs`, below) -- no evaluation here is measured against the
    full base fit any more (`B_hat`'s old full-width central-pair second
    difference is gone, module docstring). `offset` (q,) is `psi0_all`'s
    own centering constant
    (`model.offset`): subtracted here, then immediately superseded by
    the root's own centring shift below (spec 4.0) -- any constant
    subtracted before an unconditional mean-subtraction cancels in the
    result, so `offset`'s value does not change `psi_hat`; it is kept
    as a parameter for interface compatibility with `qij.py`'s existing
    convention, not because it still does anything (a judgement call,
    reported with the build).

    `measured` (spec/QIJ_mods_waves.md A11) is the absolute indices,
    into T's full output, that `psi0_all`/`model` already carry;
    `measured=None` defaults to identity (0..q-1). `z`/`n_min`/`L_max`
    are the three user-exposed levers (spec section 5); `L_max=None`
    resolves to `xvq.M_used` (the survey's own prototype count, spec's
    ruling for the cap's default).

    `start` (A9) passes through unchanged to every evaluation this
    function makes; `start=None` reproduces today's evaluations bit for
    bit. All of one round's evaluations run in a single `pool.map` call
    (spec 4.2/4.3), bit-identical across worker counts (module
    docstring, `_central_task`).
    """
    N, q = psi0_all.shape
    if measured is None:
        measured = list(range(q))
    measured = np.asarray(measured, dtype=int)
    theta_hat = np.asarray(theta_hat, dtype=float)
    M_X_used = xvq.M_used
    if L_max is None:
        L_max = M_X_used

    if np.any(model.constant_path):
        return _failed_result(N, q, z, n_min, L_max)

    # std0 (spec 4.2): the per-output std of the POPULATED vector over
    # the whole cloud, a unit fixed once, at population -- every cut's
    # own standardization throughout the loop, replacing the old
    # growth-stage `growth.std*a` (A18's separate scale step is gone).
    std0 = np.std(psi0_all, axis=0)
    theta_abs = np.maximum(np.abs(theta_hat[measured]), np.finfo(float).eps)
    delta = central_step(eta)

    # THE STATE VECTOR (module docstring). The root's own update (spec
    # 4.0): `offset` is subtracted, then the CENTRING shift -- an
    # unconditional mean subtraction, NOT one of `_apply_update`'s
    # scale/shift branches (there is no step t_root: the root is never
    # evaluated, spec 4.0), so it is not counted in n_update_scale/
    # shift/negative.
    psi_hat = psi0_all - offset[None, :]
    m_pre_root = psi_hat.mean(axis=0)
    psi_hat = psi_hat - m_pre_root[None, :]

    next_id = 1
    leaves: Dict[int, dict] = {
        0: dict(indices=np.arange(N), n=N, U=np.zeros(q), m=m_pre_root,
                ebar=-m_pre_root, delta_U=np.zeros(q), var_hat=psi_hat.var(axis=0)),
    }
    closed: set = set()

    n_update_scale = np.zeros(q, dtype=int)
    n_update_shift = np.zeros(q, dtype=int)
    n_update_negative = np.zeros(q, dtype=int)

    V_btw = np.zeros(q)
    split_D: list = []
    split_G: list = []
    split_parent: list = []
    split_child_a: list = []
    split_child_b: list = []
    split_round: list = []
    split_W_parent: list = []
    split_n_parent: list = []
    split_floor: list = []
    round_kappa: list = []
    round_se_kappa: list = []
    round_V_win: list = []
    round_margin: list = []
    round_n_splits: list = []

    n_evals = 0
    capped = False
    stop_met = False
    busy_delta = 0.0
    round_idx = 0

    # Loop invariants read fresh every iteration (never cached across
    # rounds): `ids`/`L` the current leaf set, `kappa`/`V_win`/`X`/
    # `margin`/`c_star` the stop's own report AFTER every round made so
    # far (spec 4.4; kappa === 1, se === 0 before any split,
    # `calibrate.kappa_hat`'s own S == 0 branch).
    kappa = se_kappa = V_win = se_V_win = margin = None
    V_tot = None
    while True:
        ids = sorted(leaves.keys())
        L = len(ids)
        n_arr_all = np.array([leaves[i]['n'] for i in ids], dtype=float)
        var_all = np.array([leaves[i]['var_hat'] for i in ids])
        W_all = (n_arr_all[:, None] / N) * var_all / N

        D_arr = np.array(split_D) if split_D else np.zeros((0, q))
        G_arr = np.array(split_G) if split_G else np.zeros((0, q))
        kappa, se_kappa = kappa_hat(D_arr, G_arr)
        V_win, se_V_win = vwin_hat(W_all, n_arr_all, kappa, se_kappa)
        n_splits_so_far = len(split_D)
        stop, X_vec, margin, c_star = stop_test(V_btw, V_win, se_V_win, eps, z,
                                                 n_splits_so_far, n_min)
        V_tot = V_btw + V_win
        if stop:
            stop_met = True
            break
        if L >= L_max:
            capped = True
            break

        cand_ids = [i for i in ids if i not in closed and leaves[i]['n'] > 1]
        if not cand_ids:
            break  # nothing splittable remains

        p_cand = np.array([leaves[i]['n'] / N for i in cand_ids])
        n_cand = np.array([leaves[i]['n'] for i in cand_ids], dtype=float)
        ebar_cand = np.array([leaves[i]['ebar'] for i in cand_ids])
        delta_U_cand = np.array([leaves[i]['delta_U'] for i in cand_ids])
        var_cand = np.array([leaves[i]['var_hat'] for i in cand_ids])
        W_cand = (n_cand[:, None] / N) * var_cand / N
        r = rank_scores(p_cand, ebar_cand, delta_U_cand, W_cand, N)
        order = sorted(range(len(cand_ids)), key=lambda j: (-r[j, c_star], cand_ids[j]))

        # Walk the ranked list once (spec 4.1/4.2): form every
        # candidate's own cut, apply the closing test (infeasible cut,
        # or expected gain G_l*kappa below the noise floor n_Delta +
        # b_Delta on every output -- f_l*W_l*kappa == G_l*kappa
        # algebraically, since f_l = G_l/W_l by definition, so neither
        # f_l nor W_l needs computing separately); a survivor's
        # expected gain on the binding output c* becomes its entry in
        # `calibrate.round_prefix`'s own input array. A judgement call
        # (spec silent on the exact walk order): every open, non-closed
        # leaf is evaluated this way every round, not just a prefix cut
        # short once the running sum reaches the excess -- recomputing
        # an unused cut next round is cheap and this keeps the pass
        # simple and unambiguous; `round_prefix` still sizes the round
        # to the smallest prefix of SURVIVORS.
        survivors: list = []
        for j in order:
            k = cand_ids[j]
            leaf = leaves[k]
            idx = leaf['indices']
            cut = two_means_split(psi_hat[idx] / std0)
            if cut is None:
                closed.add(k)
                continue
            idx_a, idx_b = idx[cut[0]], idx[cut[1]]
            if idx_a.size <= idx_b.size:
                idx_small, idx_large = idx_a, idx_b
            else:
                idx_small, idx_large = idx_b, idx_a
            n_small, n_large = int(idx_small.size), int(idx_large.size)
            p_small, p_large = n_small / N, n_large / N
            U_small_pred = psi_hat[idx_small].mean(axis=0)
            U_large_pred = psi_hat[idx_large].mean(axis=0)
            U_k = leaf['U']
            G_l = (n_small * (U_small_pred - U_k) ** 2
                   + n_large * (U_large_pred - U_k) ** 2) / N ** 2
            t_small = step_parameter(delta, p_small)
            t_large = step_parameter(delta, p_large)
            delta_U_small = eta * theta_abs / t_small
            # n_Delta/b_Delta (spec 4.1, "the noise floor for closing";
            # the measured-check spec's terms, spec/
            # QIJ_joint_check_measured_spec.md section 4 -- "as today",
            # spec section 5): evaluated here PRE-measurement, so
            # U_small/U_large are the predicted (psi_hat) values, the
            # same substitution G_l already makes; delta_U at the small
            # child's own step (another judgement call, spec silent on
            # which step feeds a priori floor terms).
            n_delta = ((2.0 * p_small / N) * (np.abs(U_small_pred) + np.abs(U_large_pred))
                       * delta_U_small)
            b_delta = (p_small / N) * (1.0 + p_small / p_large) * delta_U_small ** 2
            floor = n_delta + b_delta
            expected_gain = G_l * kappa
            if np.all(expected_gain < floor):
                closed.add(k)
                continue
            survivors.append((k, float(expected_gain[c_star]), dict(
                idx_small=idx_small, idx_large=idx_large, n_small=n_small, n_large=n_large,
                p_small=p_small, p_large=p_large, t_small=t_small, t_large=t_large,
                U_small_pred=U_small_pred, U_large_pred=U_large_pred, G_l=G_l, floor=floor,
            )))

        if not survivors:
            break  # every candidate closed: nothing splittable remains

        gains = np.array([s[1] for s in survivors])
        k_prefix = round_prefix(gains, float(X_vec[c_star]))
        chosen = survivors[:k_prefix]

        # The leaf cap (spec 4.4, the author's ruling): L <= L_max.
        remaining_budget = L_max - L
        if len(chosen) > remaining_budget:
            chosen = chosen[:remaining_budget]
            capped = True

        round_idx += 1
        round_kappa.append(kappa)
        round_se_kappa.append(se_kappa)
        round_V_win.append(V_win)
        round_margin.append(margin)

        pool_tasks = []
        for leaf_id, _, info in chosen:
            mask = np.zeros(N, dtype=bool)
            mask[info['idx_small']] = True
            t_small = info['t_small']
            pool_tasks.append((leaf_id, t_small, mask, start, eta))
            pool_tasks.append((leaf_id, -t_small, mask, start, eta))
        n_evals += len(pool_tasks)

        if pool is None:
            raw_results = []
            for leaf_id, t_signed, mask, s, e in pool_tasks:
                omega = perturbed_weights(np.ones(N), mask, t_signed)
                val = np.asarray(counter(X, omega, start=s, eta=e), dtype=float)
                failed = bool(np.any(np.isnan(val)))
                raw_results.append((leaf_id, t_signed, val, failed, 0.0))
        else:
            t_map0 = time.perf_counter()
            raw_results = pool.map(_central_task, pool_tasks)
            busy_delta += sum(r[4] for r in raw_results) - (time.perf_counter() - t_map0)
            for _, _, _, failed, _ in raw_results:
                counter.add(1, N, int(failed))

        this_round_splits = 0
        for i, (leaf_id, _, info) in enumerate(chosen):
            _, _, val_plus, failed_p, _ = raw_results[2 * i]
            _, _, val_minus, failed_m, _ = raw_results[2 * i + 1]
            if failed_p or failed_m:
                closed.add(leaf_id)  # cancelled: the parent stays as a final bin, closed
                continue
            t_small = info['t_small']
            t_large = info['t_large']
            idx_small = info['idx_small']
            idx_large = info['idx_large']
            p_small = info['p_small']
            p_large = info['p_large']
            n_small = info['n_small']
            n_large = info['n_large']

            U_small_full = (val_plus - val_minus) / (2.0 * t_small)
            U_small = U_small_full[measured]

            leaf = leaves.pop(leaf_id)
            U_k = leaf['U']
            p_parent = leaf['n'] / N
            U_large = (p_parent * U_k - p_small * U_small) / p_large
            D_s = (n_small * (U_small - U_k) ** 2 + n_large * (U_large - U_k) ** 2) / N ** 2
            G_s = info['G_l']
            V_btw = V_btw + D_s

            split_D.append(D_s)
            split_G.append(G_s)
            split_parent.append(leaf_id)
            split_child_a.append(next_id)
            split_child_b.append(next_id + 1)
            split_round.append(round_idx)
            # diagnostic only: the parent's W in the vector state at this moment, the
            # same formula as W_all (before the children's update; nothing reads it)
            split_W_parent.append((leaf['n'] / N) * leaf['var_hat'] / N)
            split_n_parent.append(int(leaf['n']))
            split_floor.append(info['floor'])

            m_pre_small = info['U_small_pred']
            m_pre_large = info['U_large_pred']
            ebar_small = U_small - m_pre_small
            ebar_large = U_large - m_pre_large
            delta_U_small_vec = eta * theta_abs / t_small
            delta_U_large_vec = eta * theta_abs / t_large

            _apply_update(psi_hat, idx_small, U_small, t_small, eta, theta_abs,
                          n_update_scale, n_update_shift, n_update_negative)
            _apply_update(psi_hat, idx_large, U_large, t_large, eta, theta_abs,
                          n_update_scale, n_update_shift, n_update_negative)

            leaves[next_id] = dict(indices=idx_small, n=n_small, U=U_small, m=m_pre_small,
                                    ebar=ebar_small, delta_U=delta_U_small_vec,
                                    var_hat=psi_hat[idx_small].var(axis=0))
            leaves[next_id + 1] = dict(indices=idx_large, n=n_large, U=U_large, m=m_pre_large,
                                        ebar=ebar_large, delta_U=delta_U_large_vec,
                                        var_hat=psi_hat[idx_large].var(axis=0))
            next_id += 2
            this_round_splits += 1

        round_n_splits.append(this_round_splits)

    ordered_ids = sorted(leaves.keys())
    L_final = len(ordered_ids)
    bin_n = np.array([leaves[i]['n'] for i in ordered_ids], dtype=int)
    bin_mass = bin_n.astype(float) / N
    bin_U = np.array([leaves[i]['U'] for i in ordered_ids])
    bin_m = np.array([leaves[i]['m'] for i in ordered_ids])
    bin_ebar = np.array([leaves[i]['ebar'] for i in ordered_ids])
    var_final = np.array([leaves[i]['var_hat'] for i in ordered_ids])
    bin_W = (bin_n[:, None].astype(float) / N) * var_final / N
    bin_label = np.empty(N, dtype=int)
    for new_id, old_id in enumerate(ordered_ids):
        bin_label[leaves[old_id]['indices']] = new_id

    n_splits_total = len(split_D)

    # ABC ingredients (spec/QIJ_unified_loop_spec.md 4.5, the reviewer's
    # ruling, 2 October 2026; module docstring): B_hat is NaN on every
    # output, always -- its summand (a central pair's second difference)
    # is not conservable to a large child, so no partition of the final
    # leaves carries it, and the old per-split small-child accumulation
    # over every depth (double-counting nested regions, Sigma p != 1) is
    # removed entirely. a_bca is formed over the FINAL LEAVES
    # (bin_mass, bin_U above) -- a true partition, every leaf measured
    # or by conservation -- with exactly `ivq.bias_and_acceleration`'s
    # own `a` formula, AT MEASURED WIDTH (q): NaN where the
    # leaf-level denominator is 0 (the root-only, zero-split case included).
    B_hat = np.full(q, np.nan)
    abc_num = np.sum(bin_mass[:, None] * bin_U ** 3, axis=0)
    abc_den = np.sum(bin_mass[:, None] * bin_U ** 2, axis=0)
    a_bca_out = np.full(q, np.nan)
    valid = abc_den > 0.0
    a_bca_out[valid] = abc_num[valid] / (6.0 * np.sqrt(N) * abc_den[valid] ** 1.5)

    return JointResult(
        V_btw=V_btw, V_win_hat=V_win, se_V_win=se_V_win, V_tot_hat=V_tot,
        kappa=kappa, se_kappa=se_kappa, margin=margin, B_hat=B_hat, a_bca=a_bca_out,
        L=L_final, n_splits=n_splits_total, n_rounds=round_idx, n_evals=n_evals,
        capped=capped, stop_met=stop_met, failed=False,
        bin_mass=bin_mass, bin_U=bin_U, bin_m=bin_m, bin_ebar=bin_ebar, bin_W=bin_W,
        bin_n=bin_n, bin_label=bin_label,
        split_parent=np.array(split_parent, dtype=int),
        split_child_a=np.array(split_child_a, dtype=int),
        split_child_b=np.array(split_child_b, dtype=int),
        split_G=np.array(split_G) if split_G else np.zeros((0, q)),
        split_D=np.array(split_D) if split_D else np.zeros((0, q)),
        split_round=np.array(split_round, dtype=int),
        split_W_parent=np.array(split_W_parent) if split_W_parent else np.zeros((0, q)),
        split_n_parent=np.array(split_n_parent, dtype=int),
        split_floor=np.array(split_floor) if split_floor else np.zeros((0, q)),
        round_kappa=np.array(round_kappa) if round_kappa else np.zeros((0, q)),
        round_se_kappa=np.array(round_se_kappa) if round_se_kappa else np.zeros((0, q)),
        round_V_win=np.array(round_V_win) if round_V_win else np.zeros((0, q)),
        round_margin=np.array(round_margin) if round_margin else np.zeros((0, q)),
        round_n_splits=np.array(round_n_splits, dtype=int),
        n_update_scale=n_update_scale, n_update_shift=n_update_shift,
        n_update_negative=n_update_negative, state_psi_hat=psi_hat,
        rank_rule='measured_error', z=z, n_min=n_min, L_max=L_max, busy_delta=busy_delta,
    )
