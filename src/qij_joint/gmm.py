"""A 2-D Gaussian-mixture estimator (K components, free full
covariances) with an analytic influence function via Louis's (1982)
identity.

`GMM2D(K, n_starts=20, seed=0, reference=None)`
fits by multi-start weighted EM (SQUAREM-accelerated), then finishes the
winning start with a trust-region Newton step
(`scipy.optimize.minimize(method='trust-exact')`, no gate, no halving
loop) in the unconstrained parametrization of `gmm_param.py`, and
reports the score and observed information at the fit for `influence`.
Follows `estimators.py`'s conventions: `T(X, w) -> ndarray(p,)` never
raises (a failed fit is NaN); the one exception to "no state between
calls" is `last_fit_info` (spec/QIJ_estimator_fit_spec.md 2.4), a
read-only diagnostic `__call__`/`influence`/`fit_and_influence` each
overwrite with their own fit's status, final scaled score, EM and
Newton iteration counts and labelling assignment, never consulted by
`T`'s own returned value. Without `start` (see below), `reference`
labels every evaluation's components by the assignment to it minimizing
total Bhattacharyya distance (method_notes section 5); without either,
components are ordered by ascending first-mean coordinate.

`GMM2D.takes_start = True`: `T`, `influence` and `fit_and_influence` all
accept an optional `start` (theta in `T`'s own output layout and
labelling). With `start` given, the multi-start search is skipped and a
single accelerated-EM run continues from it at the given weights, then
the same Newton gate, polish and acceptance rule run as always; without
`start` the fit is unchanged (bit-identical). This makes a finite
difference of `T` well-defined: the perturbed fit is the continuation of
the base fit, not a fresh multi-start winner that can differ from it by
more than the perturbation itself (method_notes section 5). With
`start`, the continued fit's components are labelled by the same
minimum-Bhattacharyya assignment, but against `start`'s OWN components,
never `reference` and never the canonical sort: a co-located pair (e.g.
a point source and the structure it sits on) can otherwise swap labels
between the start and the continuation under a sort or a fixed external
reference, which breaks the continuation's whole point -- the same
component keeping the same label across a perturbed or resampled
evaluation.

The observed information `A` and the score `psi_bar` are both normalized
by `sum(w)`, so `T` is invariant to a common rescaling of the weights
regardless of what a caller's own convention for `sum(w)` is (method_notes
section 5); at `sum(w) == N`, the row count, this is the same number as
normalizing by `N`.

The estimand is the maximizer of the penalized log-likelihood ell_p =
ell - a * sum_k[tr(S Sigma_k^-1) + log det Sigma_k] (Chen and Tan 2009,
arXiv:0805.3906, eq. 2), a = 1/sum(w), S the w-weighted covariance of
the rows passed to T -- always on, for every call, since the free-
covariance mixture likelihood is otherwise unbounded. EM's M-step, the
trust-region finish's score and observed information, and the analytic
influence are all those of ell_p; method_notes section 5 gives every
closed form.

Parameter layout, the two-phase multi-start/EM design, the penalized
M-step, SQUAREM acceleration, the Newton gate and polish, and the
score/Louis's-identity construction of `(psi, A)` are documented in
`spec/method_notes.md`, section "GMM2D"; each is implemented here
exactly as described there.

`prepare(X)` holds the unweighted centering shift and the `(N, 6)`
feature buffer built from the centered X, both independent of `w` and
otherwise rebuilt on every evaluation of the same X.

The multi-start's start set is `n_starts` k-means starts plus greedy-EM
starts from `seeding.py` (spec/QIJ_mods_waves.md A12 round 5, A15): a
component grown to K by residual-driven insertion after Verbeek,
Vlassis and Kroese (2003), one at 50*K over-segmentation cells and a
second at 25*K cells, the second contributing its own ×1/2 and ×2
covariance variants alongside it. Every kind of start goes through the
same two-phase screen.
"""

from collections import namedtuple
import math
import warnings

import numpy as np
from scipy.cluster.vq import kmeans2
from scipy.optimize import linear_sum_assignment, minimize

from .gmm_param import (chain, coordinate_scales, from_unconstrained, jacobian,
                         scaled_gradient_norm, to_unconstrained)
from .seeding import greedy_em_start, scaled_variants

_LOG2PI = float(np.log(2.0 * np.pi))
# The constructor's defaults for the two precision settings. `eta`, the
# fit's convergence tolerance on the scaled gradient, defaults to the
# square root of machine precision (the standard gradient tolerance, and
# the step a forward difference balances against it); a caller's
# measured eta overrides it per call. `cond_max` is the largest condition
# number of the observed information the influence accepts.
_ETA_DEFAULT = float(np.sqrt(np.finfo(float).eps))
_COND_MAX_DEFAULT = 1e12

# Phase 1's fixed screening budget and phase 2's promotion count, set by
# measurement (module report): fastest choice that never moved the
# winning start or theta_hat relative to running every start to full
# budget.
_SHORT_ITERS = 25
_PROMOTE_N = 3

# The second greedy-EM start's over-segmentation resolution
# (spec/QIJ_mods_waves.md A15), beside `seeding.py`'s own 50*K default,
# and the covariance factors of its two scaled variants (A12 item 6).
_GREEDY_CELL_FACTOR_2 = 25
_GREEDY_VARIANT_FACTORS = (0.5, 2.0)

# scipy.cluster.vq.kmeans2 runs exactly this many Lloyd iterations (its
# `thresh` argument is not implemented as a convergence test), so this is
# set well past the point a k-means run on data of this scale has any
# label left to reassign; further iterations after that point are exact
# no-ops, not a source of nondeterminism or extra failure.
_KMEANS_ITER = 300

_D = {
    'S11': np.array([[1.0, 0.0], [0.0, 0.0]]),
    'S12': np.array([[0.0, 1.0], [1.0, 0.0]]),
    'S22': np.array([[0.0, 0.0], [0.0, 1.0]]),
}
_STYPES = ('S11', 'S12', 'S22')

_Cfg = namedtuple('_Cfg', ['K', 'n_starts', 'seed', 'p'])


def _make_outputs(K: int) -> tuple:
    names = [f'pi{j + 1}' for j in range(K - 1)]
    names += [f'mu{k + 1}{ax}' for k in range(K) for ax in ('x', 'y')]
    names += [f'S{k + 1}_{s}' for k in range(K) for s in ('11', '12', '22')]
    return tuple(names)


def _idx_mu(K: int, k: int):
    base = (K - 1) + 2 * k
    return base, base + 1


def _idx_S(K: int, k: int):
    base = (K - 1) + 2 * K + 3 * k
    return base, base + 1, base + 2


def _pack(K: int, pis: np.ndarray, mus: np.ndarray, Ss: np.ndarray) -> np.ndarray:
    p = (K - 1) + 5 * K
    theta = np.empty(p)
    theta[:K - 1] = pis[:K - 1]
    for k in range(K):
        mx, my = _idx_mu(K, k)
        theta[mx], theta[my] = mus[k, 0], mus[k, 1]
        s11, s12, s22 = _idx_S(K, k)
        theta[s11], theta[s12], theta[s22] = Ss[k, 0], Ss[k, 1], Ss[k, 2]
    return theta


def _unpack(K: int, theta: np.ndarray):
    pis = np.empty(K)
    pis[:K - 1] = theta[:K - 1]
    pis[K - 1] = 1.0 - pis[:K - 1].sum()
    mus = np.empty((K, 2))
    Ss = np.empty((K, 3))
    for k in range(K):
        mx, my = _idx_mu(K, k)
        mus[k, 0], mus[k, 1] = theta[mx], theta[my]
        s11, s12, s22 = _idx_S(K, k)
        Ss[k, 0], Ss[k, 1], Ss[k, 2] = theta[s11], theta[s12], theta[s22]
    return pis, mus, Ss


# ======================================================================
# Weighted EM -- one start at a time.
# ======================================================================

def _starts(X: np.ndarray, K: int, n_starts: int, seed: int):
    """Deterministic in (X, seed) only -- never sees w (the estimator
    must stay continuous in w, which is why the starts cannot depend on
    it). Each of the `n_starts` starts is a standard k-means clustering
    of X into K clusters (`scipy.cluster.vq.kmeans2`, `minit='++'`,
    already a package dependency so no new one is added): a start's
    means are the cluster centroids, its mixing weights the cluster
    fractions, and each component's covariance is that cluster's own
    population covariance. Successive starts draw their k-means seeding
    from the same generator, so they differ from each other and the
    whole sequence is reproducible in (X, seed) alone."""
    rng = np.random.default_rng(seed)
    N = X.shape[0]
    data_cov = np.cov(X.T, bias=True)
    S0 = np.array([data_cov[0, 0], data_cov[0, 1], data_cov[1, 1]]) / K
    out = []
    for _ in range(n_starts):
        with warnings.catch_warnings():
            # A cluster left empty by ++ seeding is not a failure here:
            # its covariance below is not finite/PD and falls back to
            # S0, or (if that leaves the start unusable) phase 1 drops
            # the whole start, exactly as any other degenerate start.
            warnings.simplefilter('ignore')
            centroids, labels = kmeans2(X, K, iter=_KMEANS_ITER, minit='++', seed=rng)
        pis0 = np.bincount(labels, minlength=K).astype(float) / N
        mus0 = centroids.astype(float).copy()
        Ss0 = np.empty((K, 3))
        for k in range(K):
            pts = X[labels == k]
            if pts.shape[0] == 0:
                Ss0[k] = S0
                continue
            c = np.cov(pts.T, bias=True)
            a, b, cc = c[0, 0], c[0, 1], c[1, 1]
            det = a * cc - b * b
            # A covariance from fewer than 3 distinct points, or from
            # collinear points, is singular (det <= 0): fall back to S0
            # so no start is degenerate at birth.
            Ss0[k] = (a, b, cc) if (np.isfinite(det) and det > 0.0) else S0
        out.append((pis0, mus0, Ss0))
    return out


def _build_features(X: np.ndarray):
    """One (N,6) buffer QX = [x, y, x^2, xy, y^2, 1]; XP = QX[:, :5] is a
    contiguous view (the M-step's moment matrix). Depends on X alone."""
    x = X[:, 0]
    y = X[:, 1]
    N = X.shape[0]
    QX = np.empty((N, 6))
    QX[:, 0] = x
    QX[:, 1] = y
    QX[:, 2] = x * x
    QX[:, 3] = x * y
    QX[:, 4] = y * y
    QX[:, 5] = 1.0
    XP = QX[:, :5]
    return QX, XP


def _sigma_terms(Ss: np.ndarray):
    """Ss (K,3) = [S11,S12,S22] -> a,b,c,det each (K,). Raises
    np.linalg.LinAlgError on a non-PD/non-finite covariance, the signal
    the EM loop and polish treat as a fit failure."""
    a = Ss[:, 0]
    b = Ss[:, 1]
    c = Ss[:, 2]
    det = a * c - b * b
    if not (np.all(np.isfinite(det)) and np.all(det > 0.0)):
        raise np.linalg.LinAlgError("non-positive-definite covariance")
    return a, b, c, det


def _penalty_sum(SA, SB, SC, det, Scov: np.ndarray):
    """sum_k[tr(Scov Sigma_k^-1) + log det Sigma_k] from Sigma_k's own
    monomials (`_sigma_terms`'s `a,b,c,det`, renamed to avoid clashing
    with the penalty's `a_pen`). Scalar; the penalty's magnitude."""
    s11, s12, s22 = Scov[0, 0], Scov[0, 1], Scov[1, 1]
    trace_k = (SA * s22 + SC * s11 - 2.0 * SB * s12) / det
    return float(np.sum(trace_k + np.log(det)))


def _penalty_sum_batched(SA, SB, SC, det, Scov: np.ndarray, K: int, S: int):
    """Same as `_penalty_sum` but per start: (K*S,) monomials -> (S,)
    sums, one per start's own K components."""
    s11, s12, s22 = Scov[0, 0], Scov[0, 1], Scov[1, 1]
    trace_k = (SA * s22 + SC * s11 - 2.0 * SB * s12) / det
    return (trace_k + np.log(det)).reshape(S, K).sum(axis=1)


def _weighted_cov(Xc: np.ndarray, w: np.ndarray):
    """W = sum(w), a_pen = 1/W, the w-weighted covariance Scov of the
    (already unweighted-mean-centered) rows Xc -- the penalty's S -- and
    the rows' residuals `d` from their own w-weighted mean (what the
    influence needs beyond the raw score, method_notes section 5).
    Recomputed once per fit,
    since S depends on w."""
    W = float(w.sum())
    xbar = (w @ Xc) / W
    d = Xc - xbar
    Scov = (d * w[:, None]).T @ d / W
    return W, 1.0 / W, Scov, d


def _penalty_terms(Ss: np.ndarray, Scov: np.ndarray, a_pen: float):
    """The penalty's contribution to the theta-gradient and Hessian of
    ell_p, both zero outside the Sigma_k blocks (pi, mu are untouched):
    penalty_sum (scalar), g (p,) = d(-a_pen*penalty_sum)/dtheta
    (`-a(Sigma_k^-1 - Sigma_k^-1 S Sigma_k^-1)`), H (p,p, block-
    diagonal across k) = d^2(-a_pen*penalty_sum)/dtheta^2, closed form
    (method_notes section 5). Raises np.linalg.LinAlgError if any
    Sigma_k is not PD."""
    K = Ss.shape[0]
    p = (K - 1) + 5 * K
    SA, SB, SC, det = _sigma_terms(Ss)
    penalty_sum = _penalty_sum(SA, SB, SC, det, Scov)
    g = np.zeros(p)
    H = np.zeros((p, p))
    for k in range(K):
        Pk = np.array([[SC[k], -SB[k]], [-SB[k], SA[k]]]) / det[k]
        PS = Pk @ Scov
        Gk = Pk - PS @ Pk  # Sigma_k^-1 - Sigma_k^-1 Scov Sigma_k^-1
        idxs = _idx_S(K, k)
        g[idxs[0]] = -a_pen * Gk[0, 0]
        g[idxs[1]] = -a_pen * (Gk[0, 1] + Gk[1, 0])
        g[idxs[2]] = -a_pen * Gk[1, 1]

        for bi, btype in enumerate(_STYPES):
            Db = _D[btype]
            PDPb = Pk @ Db @ Pk
            # d(Gk)/dDb: product rule through Sigma_k^-1's two factors.
            Mb = -PDPb + PDPb @ Scov @ Pk + Pk @ Scov @ PDPb
            for ai, atype in enumerate(_STYPES):
                if atype == 'S11':
                    val = Mb[0, 0]
                elif atype == 'S22':
                    val = Mb[1, 1]
                else:
                    val = Mb[0, 1] + Mb[1, 0]
                H[idxs[ai], idxs[bi]] = -a_pen * val
    return penalty_sum, g, H


def _penalty_influence_extra(Ss: np.ndarray, Scov: np.ndarray, a_pen: float,
                              W: float, d: np.ndarray) -> np.ndarray:
    """(N, p) correction the penalized M-estimator's influence needs
    beyond the raw score: d(penalized score)/domega_i through S(omega)
    and a_pen(omega) at fixed theta (method_notes section 5). `d` =
    the rows' residuals from their own omega-weighted mean, the same
    mean Scov is built from. Zero outside the Sigma_k blocks."""
    K = Ss.shape[0]
    N = d.shape[0]
    p = (K - 1) + 5 * K
    SA, SB, SC, det = _sigma_terms(Ss)
    extra = np.zeros((N, p))
    for k in range(K):
        Pk = np.array([[SC[k], -SB[k]], [-SB[k], SA[k]]]) / det[k]
        PS = Pk @ Scov
        Gk = Pk - PS @ Pk
        PkScovPk = PS @ Pk
        u = d @ Pk  # (N,2): Pk @ d_i for every i (Pk symmetric)

        idxs = _idx_S(K, k)
        M00 = (a_pen ** 2) * Gk[0, 0] + (a_pen / W) * (u[:, 0] ** 2 - PkScovPk[0, 0])
        M11 = (a_pen ** 2) * Gk[1, 1] + (a_pen / W) * (u[:, 1] ** 2 - PkScovPk[1, 1])
        M01 = (a_pen ** 2) * Gk[0, 1] + (a_pen / W) * (u[:, 0] * u[:, 1] - PkScovPk[0, 1])
        extra[:, idxs[0]] = M00
        extra[:, idxs[1]] = 2.0 * M01
        extra[:, idxs[2]] = M11
    return extra


def _log_density_coeffs(pis: np.ndarray, mus: np.ndarray,
                         a: np.ndarray, b: np.ndarray, c: np.ndarray, det: np.ndarray):
    """(6, K) coefficients C such that QX @ C is log(pi_k) + log phi_k(x)
    for every (observation, component)."""
    K = pis.shape[0]
    mux = mus[:, 0]
    muy = mus[:, 1]
    C = np.empty((6, K))
    C[0] = (c * mux - b * muy) / det
    C[1] = (-b * mux + a * muy) / det
    C[2] = -0.5 * c / det
    C[3] = b / det
    C[4] = -0.5 * a / det
    C[5] = (-0.5 * (c * mux ** 2 - 2.0 * b * mux * muy + a * muy ** 2) / det
            + np.log(pis) - _LOG2PI - 0.5 * np.log(det))
    return C


def _e_step_fast(Q: np.ndarray, pis: np.ndarray, mus: np.ndarray,
                  a: np.ndarray, b: np.ndarray, c: np.ndarray, det: np.ndarray):
    """One (N,6) @ (6,K) matmul. Returns R (N,K) responsibilities and
    log_norm (N,)."""
    C = _log_density_coeffs(pis, mus, a, b, c, det)
    L = Q @ C
    Lmax = L.max(axis=1)
    E = np.exp(L - Lmax[:, None])
    ssum = E.sum(axis=1)
    R = E / ssum[:, None]
    log_norm = Lmax + np.log(ssum)
    return R, log_norm


def _m_step(w: np.ndarray, R: np.ndarray, XP: np.ndarray,
            Scov: np.ndarray, a_pen: float):
    """R (N,K) -> pis (K,), mus (K,2), Ss (K,3). One (K,N) @ (N,5) matmul
    for the weighted moments of every component at once. pi, mu are the
    ordinary weighted-mean M-step (unchanged by the penalty); Sigma_k is
    the closed-form inverse-Wishart MAP blend of the raw weighted covariance with
    `Scov`, `Sigma_k = (n_k*Sigma_k_raw + 2*a_pen*Scov)/(n_k+2*a_pen)`."""
    N, K = R.shape
    rw = w[:, None] * R
    n_k = rw.sum(axis=0)
    M = rw.T @ XP
    denom = n_k.sum()
    pis = n_k / denom
    with np.errstate(invalid='ignore', divide='ignore'):
        mux = M[:, 0] / n_k
        muy = M[:, 1] / n_k
        Exx = M[:, 2] / n_k
        Exy = M[:, 3] / n_k
        Eyy = M[:, 4] / n_k
    S11_raw = Exx - mux * mux
    S12_raw = Exy - mux * muy
    S22_raw = Eyy - muy * muy
    denom_pen = n_k + 2.0 * a_pen
    S11 = (n_k * S11_raw + 2.0 * a_pen * Scov[0, 0]) / denom_pen
    S12 = (n_k * S12_raw + 2.0 * a_pen * Scov[0, 1]) / denom_pen
    S22 = (n_k * S22_raw + 2.0 * a_pen * Scov[1, 1]) / denom_pen
    mus = np.stack([mux, muy], axis=-1)
    Ss = np.stack([S11, S12, S22], axis=-1)
    return pis, mus, Ss


def _penalized_ll(Q, w, W, pis, mus, Ss, Scov, a_pen):
    """ell_p per unit weight at (pis, mus, Ss): the E-step's log-
    normalizer plus the penalty, no M-step. Raises np.linalg.LinAlgError
    if Ss is not PD."""
    a, b, c, det = _sigma_terms(Ss)
    _, log_norm = _e_step_fast(Q, pis, mus, a, b, c, det)
    return float(np.dot(w, log_norm)) / W - a_pen * _penalty_sum(a, b, c, det, Scov) / W


def _em_step(Q, XP, w, W, pis, mus, Ss, Scov, a_pen):
    """One penalized-EM update -> (pis_new, mus_new, Ss_new, ll), `ll`
    at the INPUT (the E-step's by-product, so a caller needing both
    pays no extra pass). Raises np.linalg.LinAlgError if the input
    covariance is not PD."""
    a, b, c, det = _sigma_terms(Ss)
    r, log_norm = _e_step_fast(Q, pis, mus, a, b, c, det)
    ll = float(np.dot(w, log_norm)) / W - a_pen * _penalty_sum(a, b, c, det, Scov) / W
    pis_new, mus_new, Ss_new = _m_step(w, r, XP, Scov, a_pen)
    return pis_new, mus_new, Ss_new, ll


def _feasible(pis: np.ndarray, Ss: np.ndarray) -> bool:
    """True iff every weight is finite and positive and every
    covariance is PD -- SQUAREM's safeguard test (method_notes section
    5)."""
    if not (np.all(np.isfinite(pis)) and np.all(pis > 0.0)):
        return False
    try:
        _sigma_terms(Ss)
    except np.linalg.LinAlgError:
        return False
    return True


def _squarem_round(Q, XP, w, W, pis, mus, Ss, Scov, a_pen, K, budget, m):
    """One SQUAREM round, spending at most `budget` (>=1) EM steps
    (method_notes section 5: theta1, theta2 from two EM steps; r, v,
    alpha = -norm(r)/norm(v), held at most -1 so the step is never
    shorter than two EM steps, and at least -m, m multiplied by 4 each
    time that limit binds; the extrapolated point followed by one EM
    step is the candidate, discarded for theta2 if infeasible or if its
    ll is below theta2's -- so the returned ll is always >= theta2's,
    hence monotone as plain EM is). `budget` < 3 skips the candidate
    trial and returns theta1 (`budget`==1) or theta2 (`budget`==2)
    plain. Returns (pis, mus, Ss, ll, n_em, m) with ll at the RETURNED
    point, n_em <= budget the EM steps used and m the step limit for
    the next round. Raises np.linalg.LinAlgError if (pis, mus, Ss) is
    already infeasible."""
    theta0 = _pack(K, pis, mus, Ss)
    pis1, mus1, Ss1, _ = _em_step(Q, XP, w, W, pis, mus, Ss, Scov, a_pen)
    if budget == 1:
        return pis1, mus1, Ss1, _penalized_ll(Q, w, W, pis1, mus1, Ss1, Scov, a_pen), 1, m

    theta1 = _pack(K, pis1, mus1, Ss1)
    pis2, mus2, Ss2, _ = _em_step(Q, XP, w, W, pis1, mus1, Ss1, Scov, a_pen)
    ll2 = _penalized_ll(Q, w, W, pis2, mus2, Ss2, Scov, a_pen)
    if budget == 2:
        return pis2, mus2, Ss2, ll2, 2, m

    theta2 = _pack(K, pis2, mus2, Ss2)
    r = theta1 - theta0
    v = (theta2 - theta1) - r
    vnorm = np.linalg.norm(v)
    # v == 0 means r == 0 too at a genuine fixed point; alpha = -1 then
    # returns theta2 with no 0/0 division.
    alpha = min(-np.linalg.norm(r) / vnorm, -1.0) if vnorm > 0.0 else -1.0
    if alpha < -m:
        alpha = -m
        m *= 4.0
    pis_sq, mus_sq, Ss_sq = _unpack(K, theta0 - 2.0 * alpha * r + alpha ** 2 * v)

    if _feasible(pis_sq, Ss_sq):
        try:
            pis3, mus3, Ss3, _ = _em_step(Q, XP, w, W, pis_sq, mus_sq, Ss_sq,
                                           Scov, a_pen)
            ll3 = _penalized_ll(Q, w, W, pis3, mus3, Ss3, Scov, a_pen)
        except np.linalg.LinAlgError:
            ll3 = None
        if ll3 is not None and np.isfinite(ll3) and ll3 >= ll2:
            return pis3, mus3, Ss3, ll3, 3, m

    return pis2, mus2, Ss2, ll2, 2, m


def _em_accelerated(X, Q, XP, w, W, pis, mus, Ss, ll, eta, Scov, a_pen, K, budget=None):
    """SQUAREM rounds (`_squarem_round`) from (pis, mus, Ss) with its own
    `ll`, until BOTH halves of the stop test pass or the 20*p iteration
    cap fires (spec/QIJ_estimator_fit_spec.md section 5 items 1-2):
    the scaled gradient norm <= sqrt(eta), and the Aitken-projected
    remaining gain |ell_inf - ell_k| <= eta * max(|ell_k|, 1), with
    ell_k the penalized ll/W at iteration k, c_k = (ell_k - ell_{k-1}) /
    (ell_{k-1} - ell_{k-2}) and ell_inf = ell_{k-1} + (ell_k -
    ell_{k-1}) / (1 - c_k) (McLachlan and Krishnan section 4.9). The
    Aitken test costs nothing beyond the ll history this loop already
    keeps; the gradient test additionally forms the penalized observed
    information (`_score_info_penalized` gives psi_bar and A together,
    so there is no way to get one without paying for the other), so it
    is evaluated only once the Aitken test has already passed --
    cheap-before-expensive, and never both in the same failing
    iteration for nothing. Reaching the cap without the gradient test
    ever passing still reports a real `score_scaled` (formed once more
    at the final iterate) rather than leaving it unset.

    Returns (pis, mus, Ss, ll, n_used, status, score_scaled,
    last_step_u): `n_used` the EM-step count (each round costs 1-3,
    method_notes section 5), `status` 'converged' or 'em_cap',
    `score_scaled` the last scaled gradient norm evaluated, and
    `last_step_u` the unconstrained-coordinate difference between the
    last two iterates (the finish's initial trust radius). `budget`,
    when given, replaces the 20*p cap: the greedy seeding's fixed-step
    insertions. Raises
    np.linalg.LinAlgError on a degenerate covariance anywhere along the
    trajectory."""
    p = (K - 1) + 5 * K
    iter_cap = 20 * p if budget is None else int(budget)
    sqrt_eta = np.sqrt(eta)
    eps = np.finfo(float).eps
    s = coordinate_scales(K, Scov)

    n_used = 0
    m = 4.0
    ll_hist = [ll]
    u_prev = to_unconstrained(K, _pack(K, pis, mus, Ss))
    last_step_u = np.zeros(p)
    score_scaled = None
    status = 'em_cap'

    while n_used < iter_cap:
        budget = iter_cap - n_used
        pis, mus, Ss, ll_new, n_em, m = _squarem_round(
            Q, XP, w, W, pis, mus, Ss, Scov, a_pen, K, budget, m)
        if not np.isfinite(ll_new):
            raise np.linalg.LinAlgError('non-finite penalized log-likelihood')
        n_used += n_em
        ll_hist.append(ll_new)
        if len(ll_hist) > 3:
            ll_hist.pop(0)

        u_new = to_unconstrained(K, _pack(K, pis, mus, Ss))
        last_step_u = u_new - u_prev
        u_prev = u_new
        ll = ll_new

        if len(ll_hist) == 3:
            ell_km2, ell_km1, ell_k = ll_hist
            denom = ell_km1 - ell_km2
            aitken_ok = False
            # Guard: a denominator too small to divide safely means
            # either a genuine 0/0 fixed point (ell has stopped moving,
            # so the remaining gain is trivially the last increment, not
            # tested here since the projection itself is skipped) or
            # noise; c_k within eps of 1 means the extrapolated limit is
            # numerically unstable (a small error in c_k is amplified
            # without bound). Either way the projection is not trusted
            # this round and EM keeps iterating -- the cap is the
            # backstop, never a false "converged".
            tol_ell = eta * max(abs(ell_k), 1.0)
            if abs(denom) > eps * max(abs(ell_km1), abs(ell_km2), 1.0):
                c_k = (ell_k - ell_km1) / denom
                one_minus_c = 1.0 - c_k
                if abs(one_minus_c) > eps * max(abs(c_k), 1.0):
                    ell_inf = ell_km1 + (ell_k - ell_km1) / one_minus_c
                    aitken_ok = abs(ell_inf - ell_k) <= tol_ell
            else:
                # ell has stopped moving: the remaining gain is the last
                # increment itself; the gradient test below still guards
                # against a plateau beside a saddle.
                aitken_ok = abs(ell_k - ell_km1) <= tol_ell
            if aitken_ok:
                psi_bar = _score_info_penalized(X, Q, w, K, pis, mus, Ss,
                                                 Scov, a_pen, W)[3]
                g_u = jacobian(K, u_new).T @ psi_bar
                score_scaled = scaled_gradient_norm(g_u, s, ell_k)
                if score_scaled <= sqrt_eta:
                    status = 'converged'
                    break

    if score_scaled is None:
        # The cap fired before the Aitken test ever passed once, so the
        # gradient norm was never formed; form it here so the caller
        # always gets a real number rather than a placeholder.
        psi_bar = _score_info_penalized(X, Q, w, K, pis, mus, Ss,
                                         Scov, a_pen, W)[3]
        g_u = jacobian(K, u_prev).T @ psi_bar
        score_scaled = scaled_gradient_norm(g_u, s, ll)

    return pis, mus, Ss, ll, n_used, status, score_scaled, last_step_u


def _run_em(X, Q, XP, w, pis0, mus0, Ss0, eta, Scov, a_pen):
    """Weighted EM for a single start, SQUAREM-accelerated
    (`_em_accelerated`, spec/QIJ_estimator_fit_spec.md section 5). None
    the moment the start degenerates (non-PD covariance or non-finite
    penalized ll) anywhere along the trajectory; otherwise runs to the
    joint Aitken/scaled-gradient stop test or the 20*p iteration cap
    (`_em_accelerated`; cap exhaustion is not a failure -- the start is
    finalized with `status='em_cap'`). Returns dict(pis, mus, Ss, ll,
    converged, n_iter, status, score_scaled, last_step_u): `ll` =
    ell_p/W, `n_iter` the EM-step count used, `converged` = (status ==
    'converged')."""
    W = float(w.sum())
    K = pis0.shape[0]
    try:
        ll0 = _penalized_ll(Q, w, W, pis0, mus0, Ss0, Scov, a_pen)
        if not np.isfinite(ll0):
            return None
        pis, mus, Ss, ll, n_used, status, score_scaled, last_step_u = _em_accelerated(
            X, Q, XP, w, W, pis0, mus0, Ss0, ll0, eta, Scov, a_pen, K)
    except np.linalg.LinAlgError:
        return None
    return dict(pis=pis, mus=mus, Ss=Ss, ll=ll, converged=(status == 'converged'),
                n_iter=n_used, status=status, score_scaled=score_scaled,
                last_step_u=last_step_u)


def _sigma_terms_batched(Ss: np.ndarray):
    """Same monomials as `_sigma_terms` but never raises: a batched
    start's non-PD covariance turns to NaN/Inf confined to its own
    columns, checked once after the loop (`_phase1_batch`)."""
    a = Ss[:, 0]
    b = Ss[:, 1]
    c = Ss[:, 2]
    det = a * c - b * b
    return a, b, c, det


def _e_step_batched(Q: np.ndarray, pis: np.ndarray, mus: np.ndarray,
                     a: np.ndarray, b: np.ndarray, c: np.ndarray, det: np.ndarray,
                     K: int, S: int):
    """One (N,6) @ (6, K*S) matmul; softmax normalized per start (block
    of K columns), so one start's numbers never leak into another's
    responsibilities. Returns R (N, K*S), log_norm (N, S)."""
    N = Q.shape[0]
    C = _log_density_coeffs(pis, mus, a, b, c, det)
    L = Q @ C
    L3 = L.reshape(N, S, K)
    Lmax = L3.max(axis=2, keepdims=True)
    E = np.exp(L3 - Lmax)
    ssum = E.sum(axis=2, keepdims=True)
    R3 = E / ssum
    log_norm = Lmax[:, :, 0] + np.log(ssum[:, :, 0])
    return R3.reshape(N, K * S), log_norm


def _m_step_batched(w: np.ndarray, R: np.ndarray, XP: np.ndarray, K: int, S: int,
                     Scov: np.ndarray, a_pen: float):
    """Same moments as `_m_step`, and the same penalized Sigma_k
    blend (`Scov`, `a_pen` are the same for every start -- they depend
    on (X, w) alone), for all K*S components at once; only the mixing-
    weight normalization is taken within each start's own K components,
    not across all K*S."""
    rw = w[:, None] * R
    n_k = rw.sum(axis=0)
    M = rw.T @ XP
    n_k3 = n_k.reshape(S, K)
    denom = n_k3.sum(axis=1)
    pis = (n_k3 / denom[:, None]).reshape(K * S)
    mux = M[:, 0] / n_k
    muy = M[:, 1] / n_k
    Exx = M[:, 2] / n_k
    Exy = M[:, 3] / n_k
    Eyy = M[:, 4] / n_k
    S11_raw = Exx - mux * mux
    S12_raw = Exy - mux * muy
    S22_raw = Eyy - muy * muy
    denom_pen = n_k + 2.0 * a_pen
    S11 = (n_k * S11_raw + 2.0 * a_pen * Scov[0, 0]) / denom_pen
    S12 = (n_k * S12_raw + 2.0 * a_pen * Scov[0, 1]) / denom_pen
    S22 = (n_k * S22_raw + 2.0 * a_pen * Scov[1, 1]) / denom_pen
    mus = np.stack([mux, muy], axis=-1)
    Ss = np.stack([S11, S12, S22], axis=-1)
    return pis, mus, Ss


def _phase1_batch(Q: np.ndarray, XP: np.ndarray, w: np.ndarray, starts: list,
                   K: int, iters: int, Scov: np.ndarray, a_pen: float):
    """Run every start in `starts` for exactly `iters` EM iterations,
    batched in lockstep (one E-step matmul, one M-step matmul per
    iteration, S = len(starts)), M-step the penalized one; no
    convergence test, no per-iteration bookkeeping (so no per-iteration
    penalized `ll` either -- only the trailing one, used for ranking).
    Degeneracy is checked once, after the loop, start by start -- the
    batched matmuls are column-blocked per start, so a NaN in one
    start's columns cannot reach another's. Returns a list of dicts
    (`_run_em`'s shape, `converged=False`, `n_iter=iters` always), one
    per surviving start."""
    S = len(starts)
    W = float(w.sum())
    pis = np.concatenate([s[0] for s in starts])
    mus = np.concatenate([s[1] for s in starts], axis=0)
    Ss = np.concatenate([s[2] for s in starts], axis=0)

    with np.errstate(all='ignore'):
        for _ in range(iters):
            a, b, c, det = _sigma_terms_batched(Ss)
            R, _ = _e_step_batched(Q, pis, mus, a, b, c, det, K, S)
            pis, mus, Ss = _m_step_batched(w, R, XP, K, S, Scov, a_pen)

        # One trailing E-step pairs the reported ll with the last M-step.
        a, b, c, det = _sigma_terms_batched(Ss)
        _, log_norm = _e_step_batched(Q, pis, mus, a, b, c, det, K, S)
        ll_raw = (w @ log_norm) / W  # (S,)
        ll = ll_raw - a_pen * _penalty_sum_batched(a, b, c, det, Scov, K, S) / W

    pis_s = pis.reshape(S, K)
    mus_s = mus.reshape(S, K, 2)
    Ss_s = Ss.reshape(S, K, 3)

    out = []
    for s in range(S):
        ll_i = float(ll[s])
        pis_i, mus_i, Ss_i = pis_s[s], mus_s[s], Ss_s[s]
        if not (np.isfinite(ll_i) and np.all(np.isfinite(pis_i))
                and np.all(np.isfinite(mus_i))):
            continue
        try:
            _sigma_terms(Ss_i)
        except np.linalg.LinAlgError:
            continue
        out.append(dict(pis=pis_i.copy(), mus=mus_i.copy(), Ss=Ss_i.copy(),
                         ll=ll_i, converged=False, n_iter=iters))
    return out


def _fit_em_multistart(X, Q, XP, w, K, n_starts, seed, eta, Scov, a_pen):
    """Pool = every surviving start, ranked by penalized weighted log-
    likelihood (ell_p/W), via the two-phase screen: phase 1 runs
    every start's short budget in lockstep (`_phase1_batch`); the best
    few are carried, one at a time, to `_run_em`'s own stop test
    (phase 2, sequential `_run_em`). Starts are the `n_starts` k-means starts (`_starts`),
    the greedy-EM insertion start of A12 at `seeding.py`'s own 50*K
    cells, a second at `_GREEDY_CELL_FACTOR_2`*K cells (A15), and that
    second start's own `_GREEDY_VARIANT_FACTORS` covariance-scaled
    variants (A12 item 6) -- appended after the k-means starts and the
    50*K one, so their presence never reorders or drops any start the
    ported pool already had. `seeding.greedy_em_start` takes this module's own
    `_em_accelerated`/`_penalized_ll` as arguments rather than importing
    this module (seeding.py must not import gmm.py, which already
    imports it)."""
    starts = _starts(X, K, n_starts, seed)
    starts += greedy_em_start(X, w, K, seed, Q, XP, Scov, a_pen, eta,
                               em_accelerated=_em_accelerated, penalized_ll=_penalized_ll)
    greedy2 = greedy_em_start(X, w, K, seed, Q, XP, Scov, a_pen, eta,
                               em_accelerated=_em_accelerated, penalized_ll=_penalized_ll,
                               cell_factor=_GREEDY_CELL_FACTOR_2)
    if greedy2:
        starts += greedy2 + scaled_variants(greedy2[0], _GREEDY_VARIANT_FACTORS)
    phase1_budget = _SHORT_ITERS

    screened = _phase1_batch(Q, XP, w, starts, K, phase1_budget, Scov, a_pen)
    if not screened:
        return None

    screened.sort(key=lambda r: -r['ll'])
    top = screened[:min(_PROMOTE_N, len(screened))]
    finished = []
    for r in top:
        res = _run_em(X, Q, XP, w, r['pis'], r['mus'], r['Ss'], eta, Scov, a_pen)
        if res is not None:
            finished.append(res)
    if not finished:
        return None
    return max(finished, key=lambda r: r['ll'])


def _canonical_sort(pis, mus, Ss):
    order = np.lexsort((mus[:, 1], mus[:, 0]))
    return pis[order].copy(), mus[order].copy(), Ss[order].copy(), order


def _covs_from_Ss(Ss: np.ndarray) -> np.ndarray:
    """(K,3) [S11,S12,S22] -> (K,2,2) full symmetric covariances."""
    K = Ss.shape[0]
    covs = np.empty((K, 2, 2))
    covs[:, 0, 0], covs[:, 0, 1], covs[:, 1, 1] = Ss[:, 0], Ss[:, 1], Ss[:, 2]
    covs[:, 1, 0] = covs[:, 0, 1]
    return covs


def _bhattacharyya(mus: np.ndarray, Ss: np.ndarray,
                    ref_means: np.ndarray, ref_covs: np.ndarray) -> np.ndarray:
    """(K,K) Bhattacharyya distance between K fitted Gaussians (mus, Ss)
    and K reference Gaussians (ref_means, ref_covs) (method_notes
    section 5)."""
    covs = _covs_from_Ss(Ss)
    det_a = covs[:, 0, 0] * covs[:, 1, 1] - covs[:, 0, 1] * covs[:, 1, 0]
    det_b = ref_covs[:, 0, 0] * ref_covs[:, 1, 1] - ref_covs[:, 0, 1] * ref_covs[:, 1, 0]
    Sigma = 0.5 * (covs[:, None, :, :] + ref_covs[None, :, :, :])
    det_bar = Sigma[..., 0, 0] * Sigma[..., 1, 1] - Sigma[..., 0, 1] * Sigma[..., 1, 0]
    diff = mus[:, None, :] - ref_means[None, :, :]
    inv00, inv01, inv11 = Sigma[..., 1, 1] / det_bar, -Sigma[..., 0, 1] / det_bar, Sigma[..., 0, 0] / det_bar
    quad = diff[..., 0] ** 2 * inv00 + 2.0 * diff[..., 0] * diff[..., 1] * inv01 + diff[..., 1] ** 2 * inv11
    return 0.125 * quad + 0.5 * np.log(det_bar / np.sqrt(det_a[:, None] * det_b[None, :]))


def _reference_sort(pis, mus, Ss, reference):
    """Relabel K fitted components by the assignment to the reference
    components minimizing total Bhattacharyya distance (method_notes
    section 5)."""
    ref_means, ref_covs = reference
    _, col = linear_sum_assignment(_bhattacharyya(mus, Ss, ref_means, ref_covs))
    order = np.argsort(col)
    return pis[order].copy(), mus[order].copy(), Ss[order].copy(), order


def reference_from_theta(theta: np.ndarray, K: int) -> tuple:
    """A fitted theta (p,) as (means (K,2), covs (K,2,2)), the format
    `GMM2D`'s `reference` argument takes (method_notes section 5)."""
    _, mus, Ss = _unpack(K, theta)
    return mus.copy(), _covs_from_Ss(Ss)


# ======================================================================
# Score and Louis's-identity information.
# ======================================================================

def _score_info(X: np.ndarray, Q: np.ndarray, w: np.ndarray, K: int,
                 pis: np.ndarray, mus: np.ndarray, Ss: np.ndarray):
    """(psi (N, p), A (p, p), ll) at (pis, mus, Ss); A = Louis's-identity
    information / sum(w) (the same normalization `ll` and `psi_bar` use
    downstream, so a common rescaling of `w` leaves `T` unchanged).
    Raises np.linalg.LinAlgError if any component's covariance is not
    PD."""
    N = X.shape[0]
    W = float(w.sum())
    p = (K - 1) + 5 * K

    a, b, c, det = _sigma_terms(Ss)
    Pinv = []
    PDP = []  # PDP[k][stype] = Pinv_k @ D_stype @ Pinv_k (constant per k)
    for k in range(K):
        Pk = np.array([[c[k], -b[k]], [-b[k], a[k]]]) / det[k]
        Pinv.append(Pk)
        pdpk = {}
        for stype in _STYPES:
            pdpk[stype] = Pk @ _D[stype] @ Pk
        PDP.append(pdpk)

    r, log_norm = _e_step_fast(Q, pis, mus, a, b, c, det)
    ll = float(np.dot(w, log_norm) / W)

    psi = np.zeros((N, p))
    for j in range(K - 1):
        psi[:, j] = r[:, j] / pis[j] - r[:, K - 1] / pis[K - 1]

    term1 = np.zeros((p, p))
    term2 = np.zeros((p, p))
    n_k_arr = np.empty(K)

    for k in range(K):
        Pk = Pinv[k]
        pdpk = PDP[k]
        res = X - mus[k]
        u = res @ Pk

        wr_k = w * r[:, k]
        nk = wr_k.sum()
        n_k_arr[k] = nk

        G00 = 0.5 * (u[:, 0] ** 2 - Pk[0, 0])
        G11 = 0.5 * (u[:, 1] ** 2 - Pk[1, 1])
        G01 = 0.5 * (u[:, 0] * u[:, 1] - Pk[0, 1])

        mx, my = _idx_mu(K, k)
        s11, s12, s22 = _idx_S(K, k)
        psi[:, mx] = r[:, k] * u[:, 0]
        psi[:, my] = r[:, k] * u[:, 1]
        psi[:, s11] = r[:, k] * G00
        psi[:, s12] = r[:, k] * 2.0 * G01
        psi[:, s22] = r[:, k] * G11

        # term1: sum_i w_i r_ik B_ik, reduced to weighted moments -- no
        # (N, p, p) array at any K.
        if nk > 0.0:
            rbar = (wr_k @ res) / nk
            R2 = (res * wr_k[:, None]).T @ res / nk
        else:
            rbar = np.zeros(2)
            R2 = np.zeros((2, 2))

        wbar = {st: rbar @ pdpk[st] for st in _STYPES}
        Z = {st: pdpk[st] @ R2 @ Pk for st in _STYPES}

        five_bar = np.empty((5, 5))
        five_bar[0, 0], five_bar[0, 1] = Pk[0, 0], Pk[0, 1]
        five_bar[1, 0], five_bar[1, 1] = Pk[0, 1], Pk[1, 1]
        for bi, btype in enumerate(_STYPES):
            col = 2 + bi
            wb = wbar[btype]
            five_bar[0, col] = wb[0]
            five_bar[col, 0] = wb[0]
            five_bar[1, col] = wb[1]
            five_bar[col, 1] = wb[1]
        for ai, atype in enumerate(_STYPES):
            for bi, btype in enumerate(_STYPES):
                Zb = Z[btype]
                PDPb = pdpk[btype]
                if atype == 'S11':
                    val = Zb[0, 0] - 0.5 * PDPb[0, 0]
                elif atype == 'S22':
                    val = Zb[1, 1] - 0.5 * PDPb[1, 1]
                else:
                    val = Zb[0, 1] + Zb[1, 0] - PDPb[0, 1]
                five_bar[2 + ai, 2 + bi] = val

        bidx = [mx, my, s11, s12, s22]
        term1[np.ix_(bidx, bidx)] += nk * five_bar

        # term2: sum_i w_i r_ik s_ik s_ik^T, one small matmul per k -- not
        # reducible to moments (outer product of the full per-obs score).
        Fk = np.empty((N, K - 1 + 5))
        for j in range(K - 1):
            val = (1.0 / pis[j]) if j == k else 0.0
            if k == K - 1:
                val -= 1.0 / pis[K - 1]
            Fk[:, j] = val
        Fk[:, K - 1 + 0] = u[:, 0]
        Fk[:, K - 1 + 1] = u[:, 1]
        Fk[:, K - 1 + 2] = G00
        Fk[:, K - 1 + 3] = 2.0 * G01
        Fk[:, K - 1 + 4] = G11
        block = (Fk * wr_k[:, None]).T @ Fk
        idx_full = list(range(K - 1)) + bidx
        term2[np.ix_(idx_full, idx_full)] += block

    if K - 1 > 0:
        term1[:K - 1, :K - 1] += np.diag(n_k_arr[:K - 1] / pis[:K - 1] ** 2)
        term1[:K - 1, :K - 1] += n_k_arr[K - 1] / pis[K - 1] ** 2

    term3 = (w[:, None] * psi).T @ psi

    A = (term1 - term2 + term3) / W

    return psi, A, ll


# ======================================================================
# Full fit: multi-start EM -> canonical order -> Newton polish -> resid.
# ======================================================================

def _score_info_penalized(X, Q, w, K, pis, mus, Ss, Scov, a_pen, W):
    """(psi_raw (N,p), A (p,p), ll, psi_bar (p,)) at (pis,mus,Ss): the
    raw per-observation Louis's-identity score `psi_raw` (unpenalized,
    `_score_info`), and the penalized observed information `A`,
    penalized log-likelihood `ll` (= ell_p/W) and penalized weighted-
    mean score `psi_bar` (= d(ell_p/W)/dtheta) -- the raw `A_raw` (itself
    /W) and `psi_bar_raw/W` each get the closed-form penalty
    Hessian/gradient added, both scaled by the same `W = sum(w)`
    (method_notes section 5). Raises np.linalg.LinAlgError if any
    Sigma_k is not PD."""
    psi_raw, A_raw, ll_raw = _score_info(X, Q, w, K, pis, mus, Ss)
    penalty_sum, g, H = _penalty_terms(Ss, Scov, a_pen)
    ll = ll_raw - a_pen * penalty_sum / W
    psi_bar = np.dot(w, psi_raw) / W + g / W
    A = A_raw - H / W
    return psi_raw, A, ll, psi_bar


def _cholesky_ok(A: np.ndarray) -> bool:
    """True iff -H (A) is positive definite (a Cholesky succeeds). No
    longer a pre-Newton gate (method_notes section 5): the trust-region
    subproblem handles an indefinite `A` itself. It survives only as the
    check a converged fit's `A` must still pass for `influence` to make
    sense of it as an observed information -- a tiny score at a saddle
    is not a maximum, whatever the trust region declares."""
    try:
        np.linalg.cholesky(A)
        return True
    except np.linalg.LinAlgError:
        return False


def _score_at(X, Q, w, cfg, pis, mus, Ss, Scov, a_pen, W, s):
    """(psi_raw, A, ll, u, g_u, H_u, score) at (pis, mus, Ss): the
    penalized score/information, its unconstrained-parametrization
    mapping (`gmm_param.chain`, `g_theta = psi_bar`, `H_theta = -A`,
    module docstring's convention), and the scaled gradient norm of 2.3
    -- one place both the EM-endpoint shortcut and the trust-region
    finish's own evaluations get all of it from `(pis, mus, Ss)` alone.
    Raises np.linalg.LinAlgError if any Sigma_k is not PD (never true of
    a `from_unconstrained` point, only of EM's own raw output)."""
    psi_raw, A, ll, psi_bar = _score_info_penalized(X, Q, w, cfg.K, pis, mus, Ss,
                                                      Scov, a_pen, W)
    theta = _pack(cfg.K, pis, mus, Ss)
    u = to_unconstrained(cfg.K, theta)
    g_u, H_u = chain(cfg.K, u, psi_bar, -A)
    score = scaled_gradient_norm(g_u, s, ll)
    return psi_raw, A, ll, u, g_u, H_u, score


def _fit(X: np.ndarray, w: np.ndarray, cfg: _Cfg, Q: np.ndarray, XP: np.ndarray,
         eta: float, reference=None, start: np.ndarray = None):
    """(dict_or_None, status), status in {'converged', 'em_cap',
    'newton_cap', 'newton_stalled', 'infeasible', 'linalg'}
    (spec/QIJ_estimator_fit_spec.md 2.4). The dict is None only for
    'infeasible' (no start survived EM) and 'linalg' (the score/
    information could not be formed, or formed but is not PD at a point
    the finish calls converged -- a saddle wearing a small gradient);
    every other status returns a dict(theta, pis, mus, Ss, A, ll, resid,
    status, n_iter_em, n_iter_newton, label_order), and only 'converged'
    additionally carries `psi` (the raw mixture score plus the
    per-point sensitivity of the penalty to that point's own weight --
    the influence's own numerator, not worth forming for a fit `GMM2D`
    is about to NaN anyway). `ll` is ell_p/W; `A` is the penalized
    observed information; `resid` is the final scaled gradient norm of
    2.3 regardless of status. Operates in the caller's own
    (already-centered) coordinates with the caller's own (X-only) `Q`,
    `XP`. Components are labelled against `start`'s own components
    (min-Bhattacharyya assignment) when `start` is given; otherwise by
    `reference` when given, else ordered by ascending first-mean
    coordinate.

    With `start` (theta (p,) in this function's own centered, packed
    layout) given, a single accelerated-EM run (`_run_em`) from `start`
    at the given `w` replaces the multi-start search below; everything
    after (labelling against `start`, the trust-region finish) is
    unchanged, so the perturbed fit is the continuation of `start`,
    never a fresh multi-start winner, and never relabelled away from
    it. Without `start`, the multi-start search, each start run by the
    same EM stop test and finished the same way.

    The trust-region finish (2.2, 5) runs only when EM's OWN status is
    'converged' (`_run_em`'s stopping rule, not this function's, fired):
    an EM iterate that only hit its iteration cap is not the branch the
    finish's initial trust radius (5.3) assumes it starts on, so `_fit`
    reports `em_cap` and stops there. Otherwise the finish runs
    unconditionally -- no PD gate, no halving loop, indefiniteness
    handled inside `scipy.optimize.minimize`'s own trust-region
    subproblem -- to `2 * cfg.p` iterations (5.4, `newton_cap`) or a
    stall it detects itself once the accepted step is smaller than the
    fit's own reproducibility (5.5, `newton_stalled`); either ending
    returns EM's endpoint, never the finish's own last (possibly worse)
    iterate."""
    W, a_pen, Scov, d = _weighted_cov(X, w)

    if start is None:
        best = _fit_em_multistart(X, Q, XP, w, cfg.K, cfg.n_starts, cfg.seed,
                                   eta, Scov, a_pen)
    else:
        pis0, mus0, Ss0 = _unpack(cfg.K, start)
        best = _run_em(X, Q, XP, w, pis0, mus0, Ss0, eta, Scov, a_pen)
    if best is None:
        return None, 'infeasible'
    pis, mus, Ss = best['pis'], best['mus'], best['Ss']

    # A continuation is labelled against START's own components (min
    # Bhattacharyya assignment, A6's machinery), never the canonical
    # sort: two co-located components can swap under ascending-mu_x
    # sorting from a tiny perturbation, which breaks the continuation's
    # whole point (the same component keeping the same label across a
    # perturbed or resampled evaluation). This takes priority over
    # `reference` too, since `start` (when given) is the more specific,
    # already-correctly-labelled anchor for this particular evaluation.
    start_reference = reference_from_theta(start, cfg.K) if start is not None else None

    def _label(pis, mus, Ss):
        if start_reference is not None:
            return _reference_sort(pis, mus, Ss, start_reference)
        if reference is None:
            return _canonical_sort(pis, mus, Ss)
        return _reference_sort(pis, mus, Ss, reference)

    pis, mus, Ss, label_order = _label(pis, mus, Ss)
    s = coordinate_scales(cfg.K, Scov)
    try:
        psi, A, ll, u, g_u, _, score = _score_at(X, Q, w, cfg, pis, mus, Ss,
                                                   Scov, a_pen, W, s)
    except np.linalg.LinAlgError:
        return None, 'linalg'

    n_iter_em = best['n_iter']

    def _stopped(status, pis, mus, Ss, A, ll, resid, n_iter_newton):
        theta = _pack(cfg.K, pis, mus, Ss)
        return dict(theta=theta, pis=pis, mus=mus, Ss=Ss, A=A, ll=ll, resid=resid,
                    status=status, n_iter_em=n_iter_em, n_iter_newton=n_iter_newton,
                    label_order=label_order), status

    if best['status'] != 'converged':
        # EM never reached its own stopping rule (2.1): the finish's
        # initial trust radius (5.3) and its "stay on EM's branch"
        # rationale both assume an iterate EM itself calls converged.
        return _stopped('em_cap', pis, mus, Ss, A, ll, score, 0)

    # The finish always takes at least one step, even when EM's endpoint
    # already meets 2.3: a Newton step from there converges quadratically,
    # so the returned fit is reproducible well below eta, which the
    # finite differences built on it need.

    # Trust-region finish (2.2, 5.3-5.6), in v = u / s so the algorithm's
    # own trust radius, and every step it takes, is already in the
    # scaled coordinates 2.3's and 5's criteria are stated in; `s` is
    # constant through the finish (fixed by Scov at EM's endpoint), so
    # this is a linear change of variables, not a reparametrization.
    p = cfg.p
    cache: dict = {}

    def _eval(v: np.ndarray) -> dict:
        key = v.tobytes()
        if cache.get('key') != key:
            u_v = v * s
            theta_v = from_unconstrained(cfg.K, u_v)
            pis_v, mus_v, Ss_v = _unpack(cfg.K, theta_v)
            psi_v, A_v, ll_v, psi_bar_v = _score_info_penalized(
                X, Q, w, cfg.K, pis_v, mus_v, Ss_v, Scov, a_pen, W)
            g_uv, H_uv = chain(cfg.K, u_v, psi_bar_v, -A_v)
            cache['key'] = key
            cache['val'] = dict(theta=theta_v, pis=pis_v, mus=mus_v, Ss=Ss_v,
                                 psi=psi_v, A=A_v, ll=ll_v, g_u=g_uv, H_u=H_uv)
        return cache['val']

    def fun(v):
        return -_eval(v)['ll']

    def jac(v):
        return -_eval(v)['g_u'] * s

    def hess(v):
        H_uv = _eval(v)['H_u']
        return -(H_uv * s[:, None]) * s[None, :]

    def _score_v(v):
        e = _eval(v)
        return scaled_gradient_norm(e['g_u'], s, e['ll'])

    v0 = u / s
    last_step_u = np.asarray(best.get('last_step_u', np.zeros(p)), dtype=float)
    radius0 = max(float(np.linalg.norm(last_step_u / s)), math.sqrt(eta))
    n_cap = 2 * p

    stall = {'v': v0.copy(), 'n': 0, 'converged': False, 'stalled': False}

    def _callback(intermediate_result):
        # scipy halts a trust-region minimize only on a raised
        # StopIteration from the callback it was given (its return value
        # is not consulted), so the two exact stopping tests of 2.3/5.5
        # both raise rather than return.
        v = np.asarray(intermediate_result.x, dtype=float)
        if _score_v(v) <= eta:
            stall['converged'] = True
            raise StopIteration
        step = float(np.linalg.norm(v - stall['v']))
        if step == 0.0:
            # A rejected step: the point did not move and the trust
            # region shrinks and tries again. Not evidence of a stall.
            return
        stall['v'] = v
        stall['n'] = stall['n'] + 1 if step < eta else 0
        # A step below the fit's own reproducibility, twice in a row (so
        # one small-but-real step near convergence is never mistaken for
        # a stall), cannot improve it further (5.5): give up here rather
        # than let the cap (5.4) spend the rest of its budget on nothing.
        if stall['n'] >= 2:
            stall['stalled'] = True
            raise StopIteration

    result = minimize(fun, v0, jac=jac, hess=hess, method='trust-exact',
                       callback=_callback,
                       options=dict(initial_trust_radius=radius0,
                                    max_trust_radius=max(1e3, 10.0 * radius0),
                                    gtol=0.0, maxiter=n_cap))
    n_iter_newton = int(result.nit)
    final = _eval(np.asarray(result.x, dtype=float))
    resid = scaled_gradient_norm(final['g_u'], s, final['ll'])

    # A finish that stalls has reached the precision floor: its steps are
    # below the fit's own reproducibility and cannot lower the gradient
    # further. It counts as converged when the gradient there is within
    # sqrt(eta), the level EM's own stop test accepts; a stall farther
    # from a maximum than that is still a failure.
    # scipy's own status 2 (its model can no longer predict an
    # improvement: the trust region has collapsed) is the same floor.
    stalled = stall['stalled'] or result.status == 2
    at_floor = stalled and resid <= math.sqrt(eta)
    if stall['converged'] or resid <= eta or at_floor:
        if not _cholesky_ok(final['A']):
            return None, 'linalg'
        extra = _penalty_influence_extra(final['Ss'], Scov, a_pen, W, d)
        fit, _ = _stopped('converged', final['pis'], final['mus'], final['Ss'],
                           final['A'], final['ll'], resid, n_iter_newton)
        fit['psi'] = final['psi'] + extra
        return fit, 'converged'

    status = 'newton_cap' if (result.status == 1 and not stall['stalled']) else 'newton_stalled'
    return _stopped(status, pis, mus, Ss, A, ll, resid, n_iter_newton)


def _center_start(start, K: int, xmean: np.ndarray):
    """A caller's `start` (theta (p,) in `T`'s own OUTPUT layout, means
    in data coordinates) shifted into `_fit`'s centered frame -- the
    inverse of the `+= xmean` a public method applies to its own result
    -- so a continuation from a previous `T(...)` return lands where
    `_fit` would place it unshifted. None if `start` is None."""
    if start is None:
        return None
    start_c = np.array(start, dtype=float, copy=True)
    for k in range(K):
        mx, my = _idx_mu(K, k)
        start_c[mx] -= xmean[0]
        start_c[my] -= xmean[1]
    return start_c


FitInfo = namedtuple('FitInfo', ['status', 'score', 'n_iter_em', 'n_iter_newton',
                                  'label_order'])


def _fit_info(fit, status) -> FitInfo:
    """`GMM2D.last_fit_info` from `_fit`'s own return
    (spec/QIJ_estimator_fit_spec.md 2.4): `score` is the final scaled
    gradient norm of 2.3 whatever `status` is; `n_iter_em`/`n_iter_newton`
    /`label_order` are 0/0/None on 'infeasible' or 'linalg', since those
    two statuses have no dict (`_fit`'s docstring)."""
    if fit is None:
        return FitInfo(status=status, score=float('nan'), n_iter_em=0,
                        n_iter_newton=0, label_order=None)
    return FitInfo(status=status, score=fit['resid'], n_iter_em=fit['n_iter_em'],
                   n_iter_newton=fit['n_iter_newton'], label_order=fit['label_order'])


class GMM2D:
    """See module docstring. `T(X, w) -> ndarray(p,)`, `T.influence(X,
    w) -> ndarray(N, p)`, `T.fit_and_influence(X, w) -> (ndarray(p,),
    ndarray(N, p))`; all three accept `K`, `n_starts`, `seed` as per-call
    keyword overrides, an optional `prep` from
    `T.prepare(X)`, an optional `start` (theta (p,) in `T`'s own
    output layout: a continuation of a previous fit, see module
    docstring), and an optional `eta`: when given, it replaces
    `self.eta` in the trust-region finish's convergence test for that
    call alone (spec/QIJ_mods_waves.md A15's measured eta_full), leaving
    `self.eta` itself untouched for every other call. With `start`
    given, all three label components against `start`'s own components
    (never `self.reference`, and never the canonical sort); without it,
    against `self.reference` (method_notes section 5) when it is not
    None, else the canonical sort. `self.name`, `self.outputs`,
    `self.eta`, `self.p` are fixed at construction from the
    constructor's own `K`.

    `self.last_fit_info` (a `FitInfo`) is the one exception to "no state
    between calls": `__call__`, `influence` and `fit_and_influence` each
    overwrite it with their own fit's outcome (`_fit`'s status, final
    scaled score, EM and Newton iteration counts, and the labelling
    assignment against `start`/`reference`/the canonical order) before
    returning, whatever that outcome is -- `T`'s own returned value NaNs
    on anything but 'converged' exactly as before; `last_fit_info` is
    where the status behind that NaN (or a converged fit's own numbers)
    is read from. None before any call."""

    name = 'gmm2d'
    takes_start = True

    def __init__(self, K: int, n_starts: int = 20, seed: int = 0, reference=None,
                 eta: float = _ETA_DEFAULT, cond_max: float = _COND_MAX_DEFAULT):
        self.K = int(K)
        self.n_starts = int(n_starts)
        self.seed = int(seed)
        self.reference = reference

        self.p = (self.K - 1) + 5 * self.K
        self.outputs = _make_outputs(self.K)
        self.eta = float(eta)
        self.cond_max = float(cond_max)
        self.last_fit_info = None

    def _resolve(self, kwargs: dict) -> _Cfg:
        K = int(kwargs.get('K', self.K))
        return _Cfg(
            K=K,
            n_starts=int(kwargs.get('n_starts', self.n_starts)),
            seed=int(kwargs.get('seed', self.seed)),
            p=(K - 1) + 5 * K,
        )

    def prepare(self, X: np.ndarray) -> tuple:
        """The unweighted centering shift and the feature buffer built
        from the centered X -- both independent of w and of any per-call
        kwarg override (none changes how X is centered or featurized)."""
        X = np.asarray(X, dtype=float)
        xmean = X.mean(axis=0)
        Xc = X - xmean
        Q, XP = _build_features(Xc)
        return xmean, Xc, Q, XP

    def __call__(self, X: np.ndarray, w: np.ndarray, prep: tuple = None,
                 start: np.ndarray = None, eta: float = None, **kwargs) -> np.ndarray:
        cfg = self._resolve(kwargs)
        try:
            w = np.asarray(w, dtype=float)
            xmean, Xc, Q, XP = prep if prep is not None else self.prepare(X)
            start_c = _center_start(start, cfg.K, xmean)
            eta_use = self.eta if eta is None else eta
            fit, status = _fit(Xc, w, cfg, Q, XP, eta_use, self.reference, start_c)
            self.last_fit_info = _fit_info(fit, status)
            if status != 'converged':
                return np.full(cfg.p, np.nan)
            theta = fit['theta']
            for k in range(cfg.K):
                mx, my = _idx_mu(cfg.K, k)
                theta[mx] += xmean[0]
                theta[my] += xmean[1]
            return theta
        except Exception:
            # A failure this module's own statuses do not name (an
            # unexpected exception, not one of `_fit`'s own LinAlgError
            # sites): 'linalg' is the closest of the six, not a claim
            # about the exception's actual type.
            self.last_fit_info = _fit_info(None, 'linalg')
            return np.full(cfg.p, np.nan)

    def influence(self, X: np.ndarray, w: np.ndarray, prep: tuple = None,
                  start: np.ndarray = None, eta: float = None, **kwargs) -> np.ndarray:
        cfg = self._resolve(kwargs)
        N = len(X)
        try:
            w = np.asarray(w, dtype=float)
            xmean, Xc, Q, XP = prep if prep is not None else self.prepare(X)
            start_c = _center_start(start, cfg.K, xmean)
            eta_use = self.eta if eta is None else eta
            fit, status = _fit(Xc, w, cfg, Q, XP, eta_use, self.reference, start_c)
            self.last_fit_info = _fit_info(fit, status)
            if status != 'converged':
                return np.full((N, cfg.p), np.nan)
            A = fit['A']
            if np.linalg.cond(A) > self.cond_max:
                return np.full((N, cfg.p), np.nan)
            IF = np.linalg.solve(A, fit['psi'].T).T
            if not np.all(np.isfinite(IF)):
                return np.full((N, cfg.p), np.nan)
            return IF
        except Exception:
            self.last_fit_info = _fit_info(None, 'linalg')
            return np.full((N, cfg.p), np.nan)

    def fit_and_influence(self, X: np.ndarray, w: np.ndarray, prep: tuple = None,
                           start: np.ndarray = None, eta: float = None, **kwargs):
        """(theta (p,), psi (N,p)) from a single fit -- for a caller that
        wants both (`ij`, always), `T(X,w)` then `T.influence(X,w)` pays
        for the multi-start search twice."""
        cfg = self._resolve(kwargs)
        N = len(X)
        nan_theta = np.full(cfg.p, np.nan)
        nan_psi = np.full((N, cfg.p), np.nan)
        try:
            w = np.asarray(w, dtype=float)
            xmean, Xc, Q, XP = prep if prep is not None else self.prepare(X)
            start_c = _center_start(start, cfg.K, xmean)
            eta_use = self.eta if eta is None else eta
            fit, status = _fit(Xc, w, cfg, Q, XP, eta_use, self.reference, start_c)
            self.last_fit_info = _fit_info(fit, status)
            if status != 'converged':
                return nan_theta, nan_psi
            theta = fit['theta'].copy()
            for k in range(cfg.K):
                mx, my = _idx_mu(cfg.K, k)
                theta[mx] += xmean[0]
                theta[my] += xmean[1]
            A = fit['A']
            if np.linalg.cond(A) > self.cond_max:
                return theta, nan_psi
            IF = np.linalg.solve(A, fit['psi'].T).T
            if not np.all(np.isfinite(IF)):
                return theta, nan_psi
            return theta, IF
        except Exception:
            self.last_fit_info = _fit_info(None, 'linalg')
            return nan_theta, nan_psi
