"""Pilot feedback: trigger and cell attribution (section 0 step 1, section 6
REVISION 2 of spec/QIJ_pilot_feedback_interface.md). Pure numpy, no
estimator calls.

TRIGGER (section 0 step 1): after a round's splits are measured, output c
fires when the round's realized-minus-expected excess exceeds eps * Vtot_c,
where excess_c = sum_{s in round} (D_sc - floor_sc) - SW_c, SW_c the raw
(kappa-free) pilot-within total standing at the START of the round, and
Vtot_c = V_btw_c + V_win_c also at the START of the round. SW and Vtot are
supplied by the caller (joint.py), built from the stored D/V_win/kappa, per
the interface; this module only forms the sum, ratio and fire flag.

WHERE (section 6, rule 2', INTRA-CELL, author-approved 4 Oct -- supersedes
the original section 0 step 2's child-level rule): consider the splits made
SINCE THE LAST PASS (all splits so far, for the first pass -- a pass
consumes its evidence). For split s with parent point set P_s (= idx_small
union idx_large) under the CURRENT bmu: if every point of P_s shares one
prototype j, the split is INTRA-CELL and, for every output c that FIRED this
round, e_sc = max(0, D_sc - floor_sc - W_parent_sc) (the trigger's own
per-split term) is added to A_jc. A split whose parent spans >= 2 cells is
attributed to no cell. Cell j is SELECTED iff sum_c A_jc > 0; for each
selected j, n_min_child_j is the smallest child (min(n_small, n_large)) over
j's own intra-cell splits that had e > 0 on some fired output. (Why
intra-cell, not child-level: child-level shares are not identifiable --
conservation forces R_s/P_s to be the same ratio on both children, so the
"realized vs predicted" shares collapse to plain point-count shares -- and
any attribution spread over a parent larger than one cell would select every
cell at round 1, since the only split then is the root. A split only carries
evidence about the SURVEY, i.e. about one particular cell's prototype, when
its whole parent already lies inside that one cell.)
"""
import numpy as np


def round_trigger(D_round, floor_round, SW, Vtot, eps):
    """D_round, floor_round: (S_r, q); SW, Vtot: (q,). Returns (fire (q,) bool, ratio (q,), excess (q,))."""
    D_round = np.asarray(D_round, dtype=float)
    floor_round = np.asarray(floor_round, dtype=float)
    SW = np.asarray(SW, dtype=float)
    Vtot = np.asarray(Vtot, dtype=float)
    if D_round.size:
        round_sum = (D_round - floor_round).sum(axis=0)
    else:
        round_sum = np.zeros_like(SW)
    excess = round_sum - SW
    ratio = excess / (eps * Vtot)
    fire = ratio > 1.0
    return fire, ratio, excess


def attribute_intracell(splits, bmu, fire):
    """splits: list of dict(idx_small, idx_large, D (q,), floor (q,), W_parent (q,)) for the splits since the last
    pass; bmu (N,); fire (q,) bool. Returns plan: list of (j, n_min_child_j) sorted by j, and scores dict j -> (q,) A_j.

    Rule 2' (module docstring): a split is INTRA-CELL iff every point of its
    parent (idx_small union idx_large) shares one `bmu` value j; a split
    spanning >= 2 cells contributes to no cell. For an intra-cell split's
    own cell j, e_sc = max(0, D_sc - floor_sc - W_parent_sc) is added to
    A_jc for every FIRED output c (unfired columns of A_j stay exactly 0 --
    this pass's evidence says nothing about them). j is selected iff
    A_j.sum() > 0; n_min_child_j is the smallest child size
    (min(n_small, n_large)) over j's own splits that had e_sc > 0 on some
    fired output (ties among several such splits: the smallest wins, since
    the refiner needs the tightest child to size k_j, rule 3')."""
    bmu = np.asarray(bmu)
    fire = np.asarray(fire, dtype=bool)
    fired = np.flatnonzero(fire)
    q = fire.size

    scores: dict = {}
    n_min_child: dict = {}
    for s in splits:
        idx_small = np.asarray(s['idx_small'])
        idx_large = np.asarray(s['idx_large'])
        idx_parent = np.concatenate([idx_small, idx_large])
        if idx_parent.size == 0:
            continue
        cells_in_parent = np.unique(bmu[idx_parent])
        if cells_in_parent.size != 1:
            continue  # spans >= 2 cells: no attribution (rule 2')
        j = int(cells_in_parent[0])
        D = np.asarray(s['D'], dtype=float)
        floor = np.asarray(s['floor'], dtype=float)
        W_parent = np.asarray(s['W_parent'], dtype=float)
        e = np.maximum(0.0, D - floor - W_parent)  # (q,)

        A_j = scores.setdefault(j, np.zeros(q))
        contributed = False
        for c in fired:
            if e[c] > 0.0:
                A_j[c] += e[c]
                contributed = True
        if contributed:
            m = min(int(idx_small.size), int(idx_large.size))
            if j not in n_min_child or m < n_min_child[j]:
                n_min_child[j] = m

    plan = [(j, n_min_child[j]) for j in sorted(scores)
            if j in n_min_child and scores[j].sum() > 0.0]
    return plan, scores
