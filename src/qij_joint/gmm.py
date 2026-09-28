"""A 2-D Gaussian-mixture estimator (K components, free full
covariances) with an analytic influence function via Louis's (1982)
identity.

`GMM2D(K, tol=1e-8, max_iter=500, reference=None)` fits a cold call by
deterministic annealing EM (SQUAREM-accelerated; spec/QIJ_mods_waves.md
A16), polishes the beta=1 result with a damped Newton step gated on the
observed information's positive-definiteness, and reports the score and
observed information at the fit for `influence`. Follows
`estimators.py`'s conventions: `T(X, w) -> ndarray(p,)` never raises (a
failed fit is NaN); no state between calls (the `last_fit_info`
diagnostic side channel, spec A16 items 6-7, is never read by T itself,
so this holds). Without `start` (see below), `reference` labels every
evaluation's components by the assignment to it minimizing total
Bhattacharyya distance (method_notes section 5); without either,
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

The cold search (`start` not given) is deterministic annealing EM (Ueda
and Nakano 1998; spec/QIJ_mods_waves.md A16), the estimator's ONLY cold
search: all K components start at the sample (weighted) mean and
covariance with equal weights and a deterministic symmetry break, then
track the penalized objective's maximizer as the inverse temperature
beta rises on a fixed geometric schedule from 0.02 to 1, tempered
E-step, penalized weighted M-step at every step. A continuation
(`start` given) never anneals -- it runs EM from its own start, since
annealing would erase the branch a derivative depends on (A16 item 5).
"""

from collections import namedtuple
from typing import Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment

_LOG2PI = float(np.log(2.0 * np.pi))
_ACCEPT_TOL = 1e-9
_COND_MAX = 1e12
_MAX_NEWTON = 40
_MAX_HALVINGS = 30
# Declared from the score norm the polish reliably reaches on the demo
# mixture's annealed fit (module report), rounded up to a power of ten.
_ETA = 1e-12

# A16's fixed annealing schedule: inverse temperature beta from
# `_BETA_MIN` to 1 in `_N_BETA_STEPS` geometric steps (`_BETA_FACTOR`
# each), the "standard schedule"; the only permitted change, on a
# failed acceptance draw, is a finer factor (A16's own words).
_BETA_MIN = 0.02
_N_BETA_STEPS = 25
_BETA_FACTOR = _BETA_MIN ** (-1.0 / _N_BETA_STEPS)  # ~1.169 ("~1.17")

# A16 item 6's "effective number of components": weight above this
# floor over N (not sum(w) -- A16's own words, "weights above 5/N").
_EFF_WEIGHT_FLOOR = 5.0

_D = {
    'S11': np.array([[1.0, 0.0], [0.0, 0.0]]),
    'S12': np.array([[0.0, 1.0], [1.0, 0.0]]),
    'S22': np.array([[0.0, 0.0], [0.0, 1.0]]),
}
_STYPES = ('S11', 'S12', 'S22')

_Cfg = namedtuple('_Cfg', ['K', 'tol', 'max_iter', 'p'])


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

def _beta_schedule() -> np.ndarray:
    """A16 item 1's geometric schedule, `_BETA_MIN` to 1 in
    `_N_BETA_STEPS` steps of `_BETA_FACTOR`; the last entry forced to
    exactly 1.0 (the point where the penalized likelihood itself is the
    objective)."""
    betas = _BETA_MIN * _BETA_FACTOR ** np.arange(_N_BETA_STEPS + 1)
    betas[-1] = 1.0
    return betas


def _anneal_start(K: int, Scov: np.ndarray, xbar: np.ndarray):
    """A16 item 2's cold start: all K components at the sample (weighted)
    mean `xbar` and covariance `Scov`, equal weights, with the
    deterministic symmetry break -- component k (1-indexed) displaced by
    (k - (K+1)/2) * 1e-3 * sigma_1 along `Scov`'s leading eigenvector,
    sigma_1 its standard deviation -- since symmetric EM would otherwise
    keep every component identical forever."""
    eigvals, eigvecs = np.linalg.eigh(Scov)
    sigma1 = float(np.sqrt(max(eigvals[-1], 0.0)))
    v1 = eigvecs[:, -1]
    offsets = (np.arange(1, K + 1) - 0.5 * (K + 1)) * 1e-3 * sigma1
    mus = xbar[None, :] + offsets[:, None] * v1[None, :]
    Ss = np.tile(np.array([Scov[0, 0], Scov[0, 1], Scov[1, 1]]), (K, 1))
    pis = np.full(K, 1.0 / K)
    return pis, mus, Ss


def _last_change_beta(trace: list) -> float:
    """A16 item 6: the beta at which the effective component count
    (`trace`'s own n_eff, `_EFF_WEIGHT_FLOOR`) LAST changed over the
    annealing run; the first beta if it never changes; NaN for an empty
    trace."""
    if not trace:
        return float('nan')
    last_beta, prev_n = trace[0][0], trace[0][1]
    for beta, n_eff, _ in trace[1:]:
        if n_eff != prev_n:
            last_beta, prev_n = beta, n_eff
    return last_beta


def _anneal_fit(X: np.ndarray, Q: np.ndarray, XP: np.ndarray, w: np.ndarray, K: int,
                 tol: float, max_iter: int, Scov: np.ndarray, a_pen: float):
    """Deterministic annealing EM (Ueda and Nakano 1998,
    spec/QIJ_mods_waves.md A16): the mixture estimator's ONLY cold
    search, replacing the multi-start. Tracks the penalized objective's
    maximizer from `_anneal_start` as beta rises over `_beta_schedule`
    (tempered E-step, the SAME penalized weighted M-step as any other
    beta -- SQUAREM-accelerated, `_em_accelerated`, warm from the
    previous beta, A16 item 3), then the final beta = 1 fit IS the
    penalized-likelihood fit any other caller sees. None if the
    trajectory degenerates (a non-PD covariance or non-finite free
    energy at any beta). Returns `_run_em`'s own dict shape plus
    `beta_star`/`beta_trace` (A16 item 6)."""
    N = X.shape[0]
    W = float(w.sum())
    xbar = (w @ X) / W
    pis, mus, Ss = _anneal_start(K, Scov, xbar)
    trace = []
    ll = None
    try:
        for beta in _beta_schedule():
            ll0 = _penalized_ll(Q, w, W, pis, mus, Ss, Scov, a_pen, beta)
            if not np.isfinite(ll0):
                return None
            pis, mus, Ss, ll, _, _ = _em_accelerated(
                Q, XP, w, W, pis, mus, Ss, ll0, tol, max_iter, Scov, a_pen, K, beta)
            n_eff = int(np.sum(pis > _EFF_WEIGHT_FLOOR / N))
            trace.append((float(beta), n_eff, float(ll)))
    except np.linalg.LinAlgError:
        return None
    return dict(pis=pis, mus=mus, Ss=Ss, ll=ll, converged=True, n_iter=0,
                beta_star=_last_change_beta(trace), beta_trace=trace)


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
                  a: np.ndarray, b: np.ndarray, c: np.ndarray, det: np.ndarray,
                  beta: float = 1.0):
    """One (N,6) @ (6,K) matmul. Returns R (N,K) tempered responsibilities,
    r_ik(beta) propto [pi_k phi_k(x_i)]^beta normalized over k
    (spec/QIJ_mods_waves.md A16 item 3), and log_norm (N,) the matching
    free-energy term log(sum_k [pi_k phi_k]^beta)/beta. `beta=1.0`
    (every caller but `_anneal_fit`) reproduces the untempered E-step
    bit for bit: multiplying/dividing by 1.0 changes no bit."""
    C = _log_density_coeffs(pis, mus, a, b, c, det)
    L = beta * (Q @ C)
    Lmax = L.max(axis=1)
    E = np.exp(L - Lmax[:, None])
    ssum = E.sum(axis=1)
    R = E / ssum[:, None]
    log_norm = (Lmax + np.log(ssum)) / beta
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


def _penalized_ll(Q, w, W, pis, mus, Ss, Scov, a_pen, beta=1.0):
    """ell_p per unit weight at (pis, mus, Ss) (or, at `beta` != 1, the
    matching tempered free energy, A16 item 3): the E-step's log-
    normalizer plus the penalty, no M-step. Raises np.linalg.LinAlgError
    if Ss is not PD."""
    a, b, c, det = _sigma_terms(Ss)
    _, log_norm = _e_step_fast(Q, pis, mus, a, b, c, det, beta)
    return float(np.dot(w, log_norm)) / W - a_pen * _penalty_sum(a, b, c, det, Scov) / W


def _em_step(Q, XP, w, W, pis, mus, Ss, Scov, a_pen, beta=1.0):
    """One penalized-EM update -> (pis_new, mus_new, Ss_new, ll), `ll`
    at the INPUT (the E-step's by-product, so a caller needing both
    pays no extra pass); tempered at `beta` (A16 item 3) -- the M-step
    itself is the same penalized weighted one at every beta, only the
    responsibilities `r` feeding it are tempered. Raises
    np.linalg.LinAlgError if the input covariance is not PD."""
    a, b, c, det = _sigma_terms(Ss)
    r, log_norm = _e_step_fast(Q, pis, mus, a, b, c, det, beta)
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


def _squarem_round(Q, XP, w, W, pis, mus, Ss, Scov, a_pen, K, budget, m, beta=1.0):
    """One SQUAREM round, spending at most `budget` (>=1) EM steps
    (method_notes section 5: theta1, theta2 from two EM steps; r, v,
    alpha = -norm(r)/norm(v), held at most -1 so the step is never
    shorter than two EM steps, and at least -m, m multiplied by 4 each
    time that limit binds; the extrapolated point followed by one EM
    step is the candidate, discarded for theta2 if infeasible or if its
    ll is below theta2's -- so the returned ll is always >= theta2's,
    hence monotone as plain EM is). `budget` < 3 skips the candidate
    trial and returns theta1 (`budget`==1) or theta2 (`budget`==2)
    plain. `beta` (spec/QIJ_mods_waves.md A16 item 3) tempers every EM
    step's E-step alike; `beta=1.0` is bit-identical to the untempered
    round. Returns (pis, mus, Ss, ll, n_em, m) with ll at the RETURNED
    point, n_em <= budget the EM steps used and m the step limit for
    the next round. Raises np.linalg.LinAlgError if (pis, mus, Ss) is
    already infeasible."""
    theta0 = _pack(K, pis, mus, Ss)
    pis1, mus1, Ss1, _ = _em_step(Q, XP, w, W, pis, mus, Ss, Scov, a_pen, beta)
    if budget == 1:
        return pis1, mus1, Ss1, _penalized_ll(Q, w, W, pis1, mus1, Ss1, Scov, a_pen, beta), 1, m

    theta1 = _pack(K, pis1, mus1, Ss1)
    pis2, mus2, Ss2, _ = _em_step(Q, XP, w, W, pis1, mus1, Ss1, Scov, a_pen, beta)
    ll2 = _penalized_ll(Q, w, W, pis2, mus2, Ss2, Scov, a_pen, beta)
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
                                           Scov, a_pen, beta)
            ll3 = _penalized_ll(Q, w, W, pis3, mus3, Ss3, Scov, a_pen, beta)
        except np.linalg.LinAlgError:
            ll3 = None
        if ll3 is not None and np.isfinite(ll3) and ll3 >= ll2:
            return pis3, mus3, Ss3, ll3, 3, m

    return pis2, mus2, Ss2, ll2, 2, m


def _em_accelerated(Q, XP, w, W, pis, mus, Ss, ll, tol, max_iter, Scov, a_pen, K, beta=1.0):
    """SQUAREM rounds (`_squarem_round`) from (pis, mus, Ss) with its own
    `ll`, to `tol` or `max_iter` EM steps (each round costs 1-3, method_notes
    section 5), tempered at `beta` (spec/QIJ_mods_waves.md A16 item 3;
    `beta=1.0`, every caller but `_anneal_fit`, is bit-identical to the
    untempered loop). Returns (pis, mus, Ss, ll, n_used, converged).
    Raises np.linalg.LinAlgError on a degenerate covariance anywhere
    along the trajectory."""
    n_used = 0
    converged = False
    m = 4.0
    while n_used < max_iter:
        budget = max_iter - n_used
        pis, mus, Ss, ll_new, n_em, m = _squarem_round(
            Q, XP, w, W, pis, mus, Ss, Scov, a_pen, K, budget, m, beta)
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
    """True iff -H (A) is positive definite (a Cholesky succeeds) --
    the Newton gate of method_notes section 5."""
    try:
        np.linalg.cholesky(A)
        return True
    except np.linalg.LinAlgError:
        return False


def _fit(X: np.ndarray, w: np.ndarray, cfg: _Cfg, Q: np.ndarray, XP: np.ndarray,
         eta: float, reference=None, start: np.ndarray = None):
    """None on failure; else dict(theta, pis, mus, Ss, psi, A, ll,
    score_history, polished, resid, beta_star, beta_trace) -- the
    penalized-likelihood estimand (method_notes section 5). `ll` is
    ell_p/W; `A` is the penalized observed information; `psi` is the raw
    mixture score plus the per-point sensitivity of the penalty to that
    point's own weight. `beta_star`/`beta_trace` are A16 items 6-7's
    annealing diagnostics (NaN/empty for a continuation, which never
    anneals). Operates in the caller's own (already-centered)
    coordinates with the caller's own (X-only) `Q`, `XP`. Components are
    labelled against `start`'s own components (min-Bhattacharyya
    assignment) when `start` is given; otherwise by `reference` when
    given, else ordered by ascending first-mean coordinate.

    With `start` (theta (p,) in this function's own centered, packed
    layout) given, a single accelerated-EM run (`_run_em`) from `start`
    at the given `w` replaces the cold annealed search below (A16 item
    5: a continuation never anneals, since annealing would erase the
    branch a derivative depends on); everything after (labelling
    against `start`, Newton gate, polish, acceptance) is unchanged, so
    the perturbed fit is the continuation of `start`, never a fresh cold
    fit, and never relabelled away from it. Without `start`, the cold
    fit is deterministic annealing EM (`_anneal_fit`, spec/
    QIJ_mods_waves.md A16), the estimator's only cold search.

    Newton gate: the polish only runs once -H (`A`) is PD (a Cholesky
    succeeds); otherwise one more block of accelerated EM (the same
    `max_iter` cap) and a single re-test, no further retry. NaN when
    -H is still not PD after that, or the polish does not bring the
    score norm under `eta`."""
    W, a_pen, Scov, d = _weighted_cov(X, w)

    if start is None:
        best = _anneal_fit(X, Q, XP, w, cfg.K, cfg.tol, cfg.max_iter, Scov, a_pen)
    else:
        pis0, mus0, Ss0 = _unpack(cfg.K, start)
        best = _run_em(Q, XP, w, pis0, mus0, Ss0, cfg.tol, cfg.max_iter,
                        Scov, a_pen)
    if best is None:
        return None
    pis, mus, Ss = best['pis'], best['mus'], best['Ss']
    beta_star = best.get('beta_star', float('nan'))
    beta_trace = best.get('beta_trace', [])

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
                score_history=score_history, polished=polished, resid=resid,
                beta_star=beta_star, beta_trace=beta_trace)


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


def _uncenter_theta(theta: np.ndarray, K: int, xmean: np.ndarray) -> np.ndarray:
    """A copy of `_fit`'s own centered `theta` shifted into T's own
    (uncentered) output layout -- the inverse of `_center_start`, and
    the one place every public method builds its returned/cached theta,
    so `self.last_fit_info['theta']` is always in the same layout a
    caller could pass back in as `start`."""
    theta = theta.copy()
    for k in range(K):
        mx, my = _idx_mu(K, k)
        theta[mx] += xmean[0]
        theta[my] += xmean[1]
    return theta


class GMM2D:
    """See module docstring. `T(X, w) -> ndarray(p,)`, `T.influence(X,
    w) -> ndarray(N, p)`, `T.fit_and_influence(X, w) -> (ndarray(p,),
    ndarray(N, p))`; all three accept `K`, `tol`, `max_iter` as per-call
    keyword overrides, an optional `prep` from `T.prepare(X)`, an
    optional `start` (theta (p,) in `T`'s own output layout: a
    continuation of a previous fit, see module docstring), and an
    optional `eta`: when given, it replaces `self.eta` in the polish's
    acceptance rule for that call alone (spec/QIJ_mods_waves.md A15's
    measured eta_full), leaving `self.eta` itself untouched for every
    other call. With `start` given, all three label components against
    `start`'s own components (never `self.reference`, and never the
    canonical sort); without it, against `self.reference` (method_notes
    section 5) when it is not None, else the canonical sort. `self.name`,
    `self.outputs`, `self.eta`, `self.p` are fixed at construction from
    the constructor's own `K`. The cold search (A16) is deterministic
    annealing, which needs no seed: `self.last_fit_info` is a diagnostic
    side channel (never read by T itself) exposing the most recent
    call's `beta_star`/`beta_trace`/`ll` (A16 items 6-7), None before any
    call or after a failed one."""

    name = 'gmm2d'
    takes_start = True

    def __init__(self, K: int, tol: float = 1e-8, max_iter: int = 500, reference=None):
        self.K = int(K)
        self.tol = float(tol)
        self.max_iter = int(max_iter)
        self.reference = reference

        self.p = (self.K - 1) + 5 * self.K
        self.outputs = _make_outputs(self.K)
        self.eta = _ETA
        self._last_fit = None

    @property
    def last_fit_info(self):
        """The most recent call's raw fit dict (`_fit`'s own shape:
        `beta_star`, `beta_trace`, `ll`, ... -- A16 items 6-7), or None
        before any call or after a failed one. A diagnostic side channel
        only: never read by T itself, so the returned theta/psi never
        depend on it (purity holds)."""
        return self._last_fit

    def _resolve(self, kwargs: dict) -> _Cfg:
        K = int(kwargs.get('K', self.K))
        return _Cfg(
            K=K,
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
                 start: np.ndarray = None, eta: float = None, **kwargs) -> np.ndarray:
        cfg = self._resolve(kwargs)
        try:
            w = np.asarray(w, dtype=float)
            xmean, Xc, Q, XP = prep if prep is not None else self.prepare(X)
            start_c = _center_start(start, cfg.K, xmean)
            eta_use = self.eta if eta is None else eta
            fit = _fit(Xc, w, cfg, Q, XP, eta_use, self.reference, start_c)
            if fit is None:
                self._last_fit = None
                return np.full(cfg.p, np.nan)
            theta = _uncenter_theta(fit['theta'], cfg.K, xmean)
            self._last_fit = dict(fit, theta=theta)
            return theta
        except Exception:
            self._last_fit = None
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
            fit = _fit(Xc, w, cfg, Q, XP, eta_use, self.reference, start_c)
            if fit is None:
                self._last_fit = None
                return np.full((N, cfg.p), np.nan)
            self._last_fit = dict(fit, theta=_uncenter_theta(fit['theta'], cfg.K, xmean))
            A = fit['A']
            if np.linalg.cond(A) > _COND_MAX:
                return np.full((N, cfg.p), np.nan)
            IF = np.linalg.solve(A, fit['psi'].T).T
            if not np.all(np.isfinite(IF)):
                return np.full((N, cfg.p), np.nan)
            return IF
        except Exception:
            self._last_fit = None
            return np.full((N, cfg.p), np.nan)

    def fit_and_influence(self, X: np.ndarray, w: np.ndarray, prep: tuple = None,
                           start: np.ndarray = None, eta: float = None, **kwargs):
        """(theta (p,), psi (N,p)) from a single fit -- for a caller that
        wants both (`ij`, always), `T(X,w)` then `T.influence(X,w)` pays
        for the cold search twice."""
        cfg = self._resolve(kwargs)
        N = len(X)
        nan_theta = np.full(cfg.p, np.nan)
        nan_psi = np.full((N, cfg.p), np.nan)
        try:
            w = np.asarray(w, dtype=float)
            xmean, Xc, Q, XP = prep if prep is not None else self.prepare(X)
            start_c = _center_start(start, cfg.K, xmean)
            eta_use = self.eta if eta is None else eta
            fit = _fit(Xc, w, cfg, Q, XP, eta_use, self.reference, start_c)
            if fit is None:
                self._last_fit = None
                return nan_theta, nan_psi
            theta = _uncenter_theta(fit['theta'], cfg.K, xmean)
            self._last_fit = dict(fit, theta=theta)
            A = fit['A']
            if np.linalg.cond(A) > _COND_MAX:
                return theta, nan_psi
            IF = np.linalg.solve(A, fit['psi'].T).T
            if not np.all(np.isfinite(IF)):
                return theta, nan_psi
            return theta, IF
        except Exception:
            self._last_fit = None
            return nan_theta, nan_psi

    def score_at(self, X: np.ndarray, w: np.ndarray, theta: np.ndarray,
                 prep: tuple = None) -> Tuple[float, float]:
        """(ll, resid): the penalized log-likelihood and Newton score
        norm of a given `theta` (T's own output layout) at (X, w) -- no
        fitting, the same objective `_fit`'s own acceptance and polish
        read, for a caller scoring an externally supplied point (the
        search audit's cold fit and truth continuation,
        spec/QIJ_mods_waves.md A16.7) against one another. (nan, nan) if
        `theta`'s covariances are not PD or the computation fails."""
        try:
            w = np.asarray(w, dtype=float)
            xmean, Xc, Q, XP = prep if prep is not None else self.prepare(X)
            theta_c = _center_start(np.asarray(theta, dtype=float), self.K, xmean)
            pis, mus, Ss = _unpack(self.K, theta_c)
            W, a_pen, Scov, _d = _weighted_cov(Xc, w)
            _, _, ll, psi_bar = _score_info_penalized(Xc, Q, w, self.K, pis, mus, Ss,
                                                        Scov, a_pen, W)
            return ll, float(np.linalg.norm(psi_bar))
        except Exception:
            return float('nan'), float('nan')
