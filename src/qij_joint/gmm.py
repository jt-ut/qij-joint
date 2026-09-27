"""A 2-D Gaussian-mixture estimator (K components, free full
covariances) with an analytic influence function via Louis's (1982)
identity.

`GMM2D(K, n_starts=20, seed=0, tol=1e-8, max_iter=500, reference=None)`
fits by multi-start weighted EM (SQUAREM-accelerated), polishes the
winning start with a damped Newton step gated on the observed
information's positive-definiteness, and reports the score and observed
information at the fit for `influence`. Follows `estimators.py`'s
conventions: `T(X, w) -> ndarray(p,)` never raises (a failed fit is
NaN); no state between calls. With `reference`, every evaluation labels
its components by the assignment to it minimizing total Bhattacharyya
distance (method_notes section 5); without one, components are ordered
by ascending first-mean coordinate.

The estimand is the maximizer of the penalized log-likelihood ell_p =
ell - a * sum_k[tr(S Sigma_k^-1) + log det Sigma_k] (Chen and Tan 2009,
arXiv:0805.3906, eq. 2), a = 1/sum(w), S the w-weighted covariance of
the rows passed to T -- always on, for every call, since the free-
covariance mixture likelihood is otherwise unbounded. EM's M-step,
Newton's score and observed information, and the analytic influence are
all those of ell_p; method_notes section 5 gives every closed form.

Parameter layout, the two-phase multi-start/EM design, the penalized
M-step, SQUAREM acceleration, the Newton gate and polish, and the
score/Louis's-identity construction of `(psi, A)` are documented in
`spec/method_notes.md`, section "GMM2D"; each is implemented here
exactly as described there.

`prepare(X)` holds the unweighted centering shift and the `(N, 6)`
feature buffer built from the centered X, both independent of `w` and
otherwise rebuilt on every evaluation of the same X.
"""

from collections import namedtuple
import warnings

import numpy as np
from scipy.cluster.vq import kmeans2
from scipy.optimize import linear_sum_assignment

_LOG2PI = float(np.log(2.0 * np.pi))
_ACCEPT_TOL = 1e-9
_COND_MAX = 1e12
_MAX_NEWTON = 20
_MAX_HALVINGS = 30
# Declared from the score norm the polish reliably reaches on the demo
# mixture's multi-start (module report), rounded up to a power of ten.
_ETA = 1e-12

# Phase 1's fixed screening budget and phase 2's promotion count, set by
# measurement (module report): fastest choice that never moved the
# winning start or theta_hat relative to running every start to full
# budget.
_SHORT_ITERS = 25
_PROMOTE_N = 3

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

_Cfg = namedtuple('_Cfg', ['K', 'n_starts', 'seed', 'tol', 'max_iter', 'p'])


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


def _em_accelerated(Q, XP, w, W, pis, mus, Ss, ll, tol, max_iter, Scov, a_pen, K):
    """SQUAREM rounds (`_squarem_round`) from (pis, mus, Ss) with its own
    `ll`, to `tol` or `max_iter` EM steps (each round costs 1-3, method_notes
    section 5). Returns (pis, mus, Ss, ll, n_used, converged). Raises
    np.linalg.LinAlgError on a degenerate covariance anywhere along the
    trajectory."""
    n_used = 0
    converged = False
    m = 4.0
    while n_used < max_iter:
        budget = max_iter - n_used
        pis, mus, Ss, ll_new, n_em, m = _squarem_round(
            Q, XP, w, W, pis, mus, Ss, Scov, a_pen, K, budget, m)
        if not np.isfinite(ll_new):
            raise np.linalg.LinAlgError('non-finite penalized log-likelihood')
        n_used += n_em
        converged = abs(ll_new - ll) / max(abs(ll), 1e-300) < tol
        ll = ll_new
        if converged:
            break
    return pis, mus, Ss, ll, n_used, converged


def _run_em(Q, XP, w, pis0, mus0, Ss0, tol, max_iter, Scov, a_pen):
    """Weighted EM for a single start, SQUAREM-accelerated
    (`_em_accelerated`, method_notes section 5). None the moment the
    start degenerates (non-PD covariance or non-finite penalized ll)
    anywhere along the trajectory; otherwise runs to `tol` or
    `max_iter` EM steps (budget exhaustion is not a failure -- the
    start is finalized `converged=False`). Returns dict(pis, mus, Ss,
    ll, converged, n_iter), `ll` = ell_p/W, `n_iter` the EM-step count
    used."""
    W = float(w.sum())
    K = pis0.shape[0]
    try:
        ll0 = _penalized_ll(Q, w, W, pis0, mus0, Ss0, Scov, a_pen)
        if not np.isfinite(ll0):
            return None
        pis, mus, Ss, ll, n_used, converged = _em_accelerated(
            Q, XP, w, W, pis0, mus0, Ss0, ll0, tol, max_iter, Scov, a_pen, K)
    except np.linalg.LinAlgError:
        return None
    return dict(pis=pis, mus=mus, Ss=Ss, ll=ll, converged=converged, n_iter=n_used)


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


def _fit_em_multistart(X, Q, XP, w, K, n_starts, seed, tol, max_iter, Scov, a_pen):
    """Pool = every surviving start, ranked by penalized weighted log-
    likelihood (ell_p/W), via the two-phase screen: phase 1 runs
    every start's short budget in lockstep (`_phase1_batch`); the best
    few are carried, one at a time, to `max_iter` (phase 2, sequential
    `_run_em`)."""
    starts = _starts(X, K, n_starts, seed)
    phase1_budget = min(_SHORT_ITERS, max_iter)

    screened = _phase1_batch(Q, XP, w, starts, K, phase1_budget, Scov, a_pen)
    if not screened:
        return None

    screened.sort(key=lambda r: -r['ll'])
    top = screened[:min(_PROMOTE_N, len(screened))]
    finished = []
    for r in top:
        res = _run_em(Q, XP, w, r['pis'], r['mus'], r['Ss'], tol, max_iter,
                       Scov, a_pen)
        if res is not None:
            finished.append(res)
    if not finished:
        return None
    return max(finished, key=lambda r: r['ll'])


def _canonical_sort(pis, mus, Ss):
    order = np.lexsort((mus[:, 1], mus[:, 0]))
    return pis[order].copy(), mus[order].copy(), Ss[order].copy()


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
    return pis[order].copy(), mus[order].copy(), Ss[order].copy()


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
    """(psi (N, p), A (p, p), ll) at (pis, mus, Ss). Raises
    np.linalg.LinAlgError if any component's covariance is not PD."""
    N = X.shape[0]
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
    ll = float(np.dot(w, log_norm) / w.sum())

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

    A = (term1 - term2 + term3) / N

    return psi, A, ll


# ======================================================================
# Full fit: multi-start EM -> canonical order -> Newton polish -> resid.
# ======================================================================

def _score_info_penalized(X, Q, w, K, pis, mus, Ss, Scov, a_pen, W):
    """(psi_raw (N,p), A (p,p), ll, psi_bar (p,)) at (pis,mus,Ss): the
    raw per-observation Louis's-identity score `psi_raw` (unpenalized,
    `_score_info`), and the penalized observed information `A`,
    penalized log-likelihood `ll` (= ell_p/W) and penalized weighted-
    mean score `psi_bar` (= d(ell_p/W)/dtheta) -- the raw `A/N` and
    `psi_bar_raw/W` each get the closed-form penalty Hessian/gradient
    added (method_notes section 5). Raises np.linalg.LinAlgError if any
    Sigma_k is not PD."""
    N = X.shape[0]
    psi_raw, A_raw, ll_raw = _score_info(X, Q, w, K, pis, mus, Ss)
    penalty_sum, g, H = _penalty_terms(Ss, Scov, a_pen)
    ll = ll_raw - a_pen * penalty_sum / W
    psi_bar = np.dot(w, psi_raw) / W + g / W
    A = A_raw - H / N
    return psi_raw, A, ll, psi_bar


def _cholesky_ok(A: np.ndarray) -> bool:
    """True iff -H (A) is positive definite (a Cholesky succeeds) --
    the Newton gate of method_notes section 5."""
    try:
        np.linalg.cholesky(A)
        return True
    except np.linalg.LinAlgError:
        return False


def _fit(X: np.ndarray, w: np.ndarray, cfg: _Cfg, Q: np.ndarray, XP: np.ndarray,
         eta: float, reference=None):
    """None on failure; else dict(theta, pis, mus, Ss, psi, A, ll,
    score_history, polished, resid) -- the penalized-likelihood
    estimand (method_notes section 5). `ll` is ell_p/W; `A` is the
    penalized observed information; `psi` is the raw mixture score plus
    the per-point sensitivity of the penalty to that point's own
    weight. Operates in the caller's own (already-centered) coordinates
    with the caller's own (X-only) `Q`, `XP`. Components are labelled
    by `reference` when given, else ordered by ascending first-mean
    coordinate.

    Newton gate: the polish only runs once -H (`A`) is PD (a Cholesky
    succeeds); otherwise one more block of accelerated EM (the same
    `max_iter` cap) and a single re-test, no further retry. NaN when
    -H is still not PD after that, or the polish does not bring the
    score norm under `eta`."""
    W, a_pen, Scov, d = _weighted_cov(X, w)

    best = _fit_em_multistart(X, Q, XP, w, cfg.K, cfg.n_starts, cfg.seed,
                               cfg.tol, cfg.max_iter, Scov, a_pen)
    if best is None:
        return None
    pis, mus, Ss = best['pis'], best['mus'], best['Ss']

    def _label(pis, mus, Ss):
        if reference is None:
            return _canonical_sort(pis, mus, Ss)
        return _reference_sort(pis, mus, Ss, reference)

    pis, mus, Ss = _label(pis, mus, Ss)
    try:
        psi, A, ll, psi_bar = _score_info_penalized(X, Q, w, cfg.K, pis, mus, Ss,
                                                      Scov, a_pen, W)
    except np.linalg.LinAlgError:
        return None

    if not _cholesky_ok(A):
        # Not near a maximum yet: one more accelerated-EM block, then
        # re-label and re-test once, no further retry.
        try:
            pis, mus, Ss, ll, _, _ = _em_accelerated(
                Q, XP, w, W, pis, mus, Ss, ll, cfg.tol, cfg.max_iter,
                Scov, a_pen, cfg.K)
        except np.linalg.LinAlgError:
            return None
        pis, mus, Ss = _label(pis, mus, Ss)
        try:
            psi, A, ll, psi_bar = _score_info_penalized(
                X, Q, w, cfg.K, pis, mus, Ss, Scov, a_pen, W)
        except np.linalg.LinAlgError:
            return None
        if not _cholesky_ok(A):
            return None

    theta = _pack(cfg.K, pis, mus, Ss)
    score_history = [float(np.linalg.norm(psi_bar))]

    # Damped Newton polish: A is -H of ell_p and PD here (the gate
    # above), so the full step is the local quadratic model; halving
    # only guards a step that leaves the feasible set or drops ell_p.
    polished = False
    for _ in range(_MAX_NEWTON):
        norm = float(np.linalg.norm(psi_bar))
        if norm <= eta:
            break
        try:
            delta = np.linalg.solve(A, psi_bar)
        except np.linalg.LinAlgError:
            break

        accepted = False
        for n_halvings in range(_MAX_HALVINGS + 1):
            theta_new = theta + delta * (0.5 ** n_halvings)
            pis_new, mus_new, Ss_new = _unpack(cfg.K, theta_new)
            if np.any(pis_new <= 0.0):
                continue
            try:
                psi_new, A_new, ll_new, psi_bar_new = _score_info_penalized(
                    X, Q, w, cfg.K, pis_new, mus_new, Ss_new, Scov, a_pen, W)
            except np.linalg.LinAlgError:
                continue
            if not np.isfinite(ll_new) or ll_new < ll - _ACCEPT_TOL:
                continue
            accepted = True
            break
        if not accepted:
            break

        theta, pis, mus, Ss = theta_new, pis_new, mus_new, Ss_new
        psi, A, ll, psi_bar = psi_new, A_new, ll_new, psi_bar_new
        polished = True
        score_history.append(float(np.linalg.norm(psi_bar)))

    resid = float(np.linalg.norm(psi_bar))
    if resid > eta:
        return None

    # The influence's per-point term: the raw score plus the per-point
    # sensitivity of the penalty (through S(w) and a_pen(w)) to that
    # point's own weight (method_notes section 5).
    extra = _penalty_influence_extra(Ss, Scov, a_pen, W, d)
    psi_full = psi + extra

    return dict(theta=theta, pis=pis, mus=mus, Ss=Ss, psi=psi_full, A=A, ll=ll,
                score_history=score_history, polished=polished, resid=resid)


class GMM2D:
    """See module docstring. `T(X, w) -> ndarray(p,)`, `T.influence(X,
    w) -> ndarray(N, p)`, `T.fit_and_influence(X, w) -> (ndarray(p,),
    ndarray(N, p))`; all three accept `K`, `n_starts`, `seed`, `tol`,
    `max_iter` as per-call keyword overrides, and an optional `prep`
    from `T.prepare(X)`; all three label components against
    `self.reference` (method_notes section 5) when it is not None.
    `self.name`, `self.outputs`, `self.eta`, `self.p` are fixed at
    construction from the constructor's own `K`."""

    name = 'gmm2d'

    def __init__(self, K: int, n_starts: int = 20, seed: int = 0,
                 tol: float = 1e-8, max_iter: int = 500, reference=None):
        self.K = int(K)
        self.n_starts = int(n_starts)
        self.seed = int(seed)
        self.tol = float(tol)
        self.max_iter = int(max_iter)
        self.reference = reference

        self.p = (self.K - 1) + 5 * self.K
        self.outputs = _make_outputs(self.K)
        self.eta = _ETA

    def _resolve(self, kwargs: dict) -> _Cfg:
        K = int(kwargs.get('K', self.K))
        return _Cfg(
            K=K,
            n_starts=int(kwargs.get('n_starts', self.n_starts)),
            seed=int(kwargs.get('seed', self.seed)),
            tol=float(kwargs.get('tol', self.tol)),
            max_iter=int(kwargs.get('max_iter', self.max_iter)),
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
                 **kwargs) -> np.ndarray:
        cfg = self._resolve(kwargs)
        try:
            w = np.asarray(w, dtype=float)
            xmean, Xc, Q, XP = prep if prep is not None else self.prepare(X)
            fit = _fit(Xc, w, cfg, Q, XP, self.eta, self.reference)
            if fit is None:
                return np.full(cfg.p, np.nan)
            theta = fit['theta']
            for k in range(cfg.K):
                mx, my = _idx_mu(cfg.K, k)
                theta[mx] += xmean[0]
                theta[my] += xmean[1]
            return theta
        except Exception:
            return np.full(cfg.p, np.nan)

    def influence(self, X: np.ndarray, w: np.ndarray, prep: tuple = None,
                  **kwargs) -> np.ndarray:
        cfg = self._resolve(kwargs)
        N = len(X)
        try:
            w = np.asarray(w, dtype=float)
            xmean, Xc, Q, XP = prep if prep is not None else self.prepare(X)
            fit = _fit(Xc, w, cfg, Q, XP, self.eta, self.reference)
            if fit is None:
                return np.full((N, cfg.p), np.nan)
            A = fit['A']
            if np.linalg.cond(A) > _COND_MAX:
                return np.full((N, cfg.p), np.nan)
            IF = np.linalg.solve(A, fit['psi'].T).T
            if not np.all(np.isfinite(IF)):
                return np.full((N, cfg.p), np.nan)
            return IF
        except Exception:
            return np.full((N, cfg.p), np.nan)

    def fit_and_influence(self, X: np.ndarray, w: np.ndarray, prep: tuple = None,
                           **kwargs):
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
            fit = _fit(Xc, w, cfg, Q, XP, self.eta, self.reference)
            if fit is None:
                return nan_theta, nan_psi
            theta = fit['theta'].copy()
            for k in range(cfg.K):
                mx, my = _idx_mu(cfg.K, k)
                theta[mx] += xmean[0]
                theta[my] += xmean[1]
            A = fit['A']
            if np.linalg.cond(A) > _COND_MAX:
                return theta, nan_psi
            IF = np.linalg.solve(A, fit['psi'].T).T
            if not np.all(np.isfinite(IF)):
                return theta, nan_psi
            return theta, IF
        except Exception:
            return nan_theta, nan_psi
