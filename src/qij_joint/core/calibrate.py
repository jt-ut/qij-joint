"""Calibration estimators for the unified measured loop.

Pure numpy; no package imports. Implements the five fixed quantities of
the unified-loop spec, section 4.4 (the per-round estimator and stop
test) and section 4.1 (the ranking rule), plus the round-sizing prefix
used by section 4.2. See:

    spec/QIJ_unified_loop_spec.md  -- SS3 (state and units), SS4.1
    (ranking), SS4.2 (a round / round sizing), SS4.4 (the stop).
    spec/QIJ_mods_waves.md A20 -- the measured-error ranking (the
    r_kc formula and its fallback to W).

Units (SS3): for leaf l and output c, W_lc = p_l * Var_l(psi_hat) / N
(a variance-over-N quantity); for split s, D_sc and G_sc are
(n_a*(U_a - U_k)**2 + n_b*(U_b - U_k)**2) / N**2, with G_sc computed
from psi_hat's pre-update child means and D_sc from the measured child
means. kappa_hat and vwin_hat consume sums of these; stop_test and
round_prefix consume the resulting V_win/V_btw/excess quantities, all
in the same variance-over-N units. rank_scores produces r in those same
units (A20).

All functions are vectorized over the output axis q; arrays are plain
numpy, dtype float unless noted.
"""

from __future__ import annotations

import numpy as np


def kappa_hat(D, G):
    """Ratio-of-sums calibration factor kappa and its sandwich SE.

    Spec SS4.4: kappa_c = sum_s D_sc / sum_s G_sc (ratio of sums over
    ALL splits made so far); kappa_c = 1 and se = 0 where sum_s G_sc ==
    0 (including S == 0, no splits yet -- both D and G are then
    shape (0, q), whose column sums are identically zero, so that edge
    falls out of the sum == 0 branch without a separate check).

        se(kappa_c)**2 = sum_s G_sc**2 * (D_sc/G_sc - kappa_c)**2
                         / (sum_s G_sc)**2

    Computed here as sum_s (D_sc - kappa_c*G_sc)**2 / (sum_s G_sc)**2,
    algebraically identical (G_sc**2*(D_sc/G_sc - k)**2 == (D_sc -
    k*G_sc)**2) but avoiding division by the possibly-zero per-split
    G_sc.

    Parameters
    ----------
    D, G : (S, q) arrays
        Per-split realized gain D_sc and predicted gain G_sc (SS3
        units: variance-over-N**2, i.e. the per-split contribution to
        V_btw).

    Returns
    -------
    kappa : (q,) array
    se_kappa : (q,) array
    """
    D = np.asarray(D, dtype=float)
    G = np.asarray(G, dtype=float)

    # G.sum(axis=0) on a (0, q) array (S == 0, no splits yet) already
    # returns zeros(q), so the S == 0 edge needs no separate branch.
    sumG = G.sum(axis=0)
    sumD = D.sum(axis=0)

    zero_mask = sumG == 0
    safe_sumG = np.where(zero_mask, 1.0, sumG)
    kappa = np.where(zero_mask, 1.0, sumD / safe_sumG)

    resid = D - kappa[np.newaxis, :] * G
    num = (resid ** 2).sum(axis=0)
    se_sq = np.where(zero_mask, 0.0, num / safe_sumG ** 2)
    se_kappa = np.sqrt(se_sq)

    return kappa, se_kappa


def vwin_hat(W, n, kappa, se_kappa):
    """Calibrated within variance V_win and its SE.

    Spec SS4.4:
        V_win,c = kappa_c * sum_l W_lc
        se(V_win,c)**2 = (sum_l W_lc)**2 * se(kappa_c)**2
                         + kappa_c**2 * sum_{l: n_l >= 2} W_lc**2 * 2/(n_l - 1)

    The chi-squared term is restricted to leaves with n_l >= 2 (SS4.4);
    n_l == 1 leaves contribute zero there (a one-point leaf has no
    within-leaf scatter to read a variance from).

    Parameters
    ----------
    W : (L, q) array
        Per-leaf within term, W_lc = p_l * Var_l(psi_hat) / N.
    n : (L,) array
        Per-leaf point counts.
    kappa, se_kappa : (q,) arrays
        From kappa_hat.

    Returns
    -------
    V_win : (q,) array
    se_V_win : (q,) array
    """
    W = np.asarray(W, dtype=float)
    n = np.asarray(n)
    kappa = np.asarray(kappa, dtype=float)
    se_kappa = np.asarray(se_kappa, dtype=float)

    sumW = W.sum(axis=0)
    V_win = kappa * sumW

    mask = n >= 2
    denom = np.maximum(n - 1, 1)  # avoid /0 for n<2; those rows are masked to 0 anyway
    factor = np.where(mask, 2.0 / denom, 0.0)
    chi2_term = (W ** 2 * factor[:, np.newaxis]).sum(axis=0)

    se_V_win = np.sqrt(sumW ** 2 * se_kappa ** 2 + kappa ** 2 * chi2_term)
    return V_win, se_V_win


def stop_test(V_btw, V_win, se_V_win, eps, z, n_splits, n_min):
    """The per-round stop test (spec SS4.4).

        V_tot,c = V_btw,c + V_win,c
        X_c = V_win,c + z*se(V_win,c) - eps*V_tot,c
        margin_c = X_c / (eps * V_tot,c)
        c* = argmax_c margin_c

    STOP iff n_splits >= n_min and X_c <= 0 on every output c.

    Parameters
    ----------
    V_btw, V_win, se_V_win : (q,) arrays
    eps, z : float
    n_splits, n_min : int

    Returns
    -------
    stop : bool
    X : (q,) array
    margin : (q,) array
    c_star : int
        argmax of margin (ties: numpy's argmax takes the first/lowest
        index, matching the spec's "ties to the lowest id" convention
        used elsewhere in the method).
    """
    V_btw = np.asarray(V_btw, dtype=float)
    V_win = np.asarray(V_win, dtype=float)
    se_V_win = np.asarray(se_V_win, dtype=float)

    V_tot = V_btw + V_win
    X = V_win + z * se_V_win - eps * V_tot
    with np.errstate(divide="ignore", invalid="ignore"):
        margin = X / (eps * V_tot)

    c_star = int(np.argmax(margin))
    stop = bool(n_splits >= n_min) and bool(np.all(X <= 0))

    return stop, X, margin, c_star


def rank_scores(p, ebar, delta_U, W, N):
    """Leaf ranking scores (spec SS4.1; A20's measured-error rule).

        r_lc = p_l * ebar_lc**2 / N   if |ebar_lc| > delta_U_lc
        r_lc = W_lc                   otherwise (fallback, same units)

    Both branches are in the units of W (variance-over-N); the
    fallback keeps the ranking well-defined for leaves whose measured
    error is at or below the measurement noise floor delta_U.

    Parameters
    ----------
    p : (L,) array
        Leaf mass fractions n_l / N.
    ebar, delta_U, W : (L, q) arrays
        Per-leaf, per-output measured error, noise floor, and within
        term.
    N : int or float
        Total point count.

    Returns
    -------
    r : (L, q) array
    """
    p = np.asarray(p, dtype=float)
    ebar = np.asarray(ebar, dtype=float)
    delta_U = np.asarray(delta_U, dtype=float)
    W = np.asarray(W, dtype=float)

    measured = (p[:, np.newaxis] * ebar ** 2) / N
    use_measured = np.abs(ebar) > delta_U
    return np.where(use_measured, measured, W)


def round_prefix(gains_in_rank_order, X_cstar):
    """Size a round: smallest rank prefix whose summed expected gain
    reaches the stop's excess on the binding output (spec SS4.2).

    k is the smallest k >= 1 such that
        cumsum(gains_in_rank_order)[:k].sum() >= X_cstar
    (i.e. cumsum[k-1] >= X_cstar); K (the whole list) if the cumulative
    sum never reaches X_cstar. At least one leaf is always included
    (k >= 1) whenever there is at least one candidate (K >= 1); for an
    empty candidate list (K == 0) there is nothing to size, so 0 is
    returned.

    Parameters
    ----------
    gains_in_rank_order : (K,) array
        Per-leaf expected gain f_l * W_lc* * kappa_c*, in rank order.
    X_cstar : float
        The stop's excess X_c* on the binding output c*.

    Returns
    -------
    k : int
    """
    gains = np.asarray(gains_in_rank_order, dtype=float)
    K = gains.shape[0]
    if K == 0:
        return 0

    cumsum = np.cumsum(gains)
    reached = cumsum >= X_cstar
    if reached.any():
        return int(np.argmax(reached)) + 1
    return K
