"""A 2-D Gaussian-mixture estimator (K components, free full
covariances) with an analytic influence function via Louis's (1982)
identity.

`GMM2D(K, n_starts=20, seed=0, tol=1e-8, max_iter=500, reference=None)`
fits by multi-start weighted EM, polishes the winning start with an
exact Newton step on the mixture log-likelihood, and reports the score
and observed information at the fit for `influence`. Follows
`estimators.py`'s conventions: `T(X, w) -> ndarray(p,)` never raises (a
failed fit is NaN); no state between calls. With `reference`, every
evaluation labels its components by the assignment to it minimizing
total Bhattacharyya distance (method_notes section 5); without one,
components are ordered by ascending first-mean coordinate.

Parameter layout, the two-phase multi-start/EM design, the Newton
polish and residual test, and the score/Louis's-identity construction
of `(psi, A)` are documented in `spec/method_notes.md`, section
"GMM2D"; each is implemented here exactly as described there.

`prepare(X)` holds the unweighted centering shift and the `(N, 6)`
feature buffer built from the centered X, both independent of `w` and
otherwise rebuilt on every evaluation of the same X.
"""

from collections import namedtuple

import numpy as np
from scipy.optimize import linear_sum_assignment

_LOG2PI = float(np.log(2.0 * np.pi))
_ACCEPT_TOL = 1e-9
_COND_MAX = 1e12
_MAX_NEWTON = 20
_NEWTON_SCORE_TOL = 1e-14
_NEWTON_PROGRESS_FLOOR = 1e-11
_NEWTON_PROGRESS_FACTOR = 0.1
_ETA_DEFAULT = 1e-12

# Phase 1's fixed screening budget and phase 2's promotion count, set by
# measurement (module report): fastest choice that never moved the
# winning start or theta_hat relative to running every start to full
# budget.
_SHORT_ITERS = 25
_PROMOTE_N = 3

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
    """Deterministic in (X, seed) only -- never sees w."""
    rng = np.random.default_rng(seed)
    N = X.shape[0]
    data_cov = np.cov(X.T, bias=True)
    cov0 = data_cov / K
    S0 = np.array([cov0[0, 0], cov0[0, 1], cov0[1, 1]])
    out = []
    for _ in range(n_starts):
        idx = rng.choice(N, size=K, replace=False)
        mus0 = X[idx].astype(float).copy()
        Ss0 = np.tile(S0, (K, 1))
        pis0 = np.full(K, 1.0 / K)
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


def _m_step(w: np.ndarray, R: np.ndarray, XP: np.ndarray):
    """R (N,K) -> pis (K,), mus (K,2), Ss (K,3). One (K,N) @ (N,5) matmul
    for the weighted moments of every component at once."""
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
    S11 = Exx - mux * mux
    S12 = Exy - mux * muy
    S22 = Eyy - muy * muy
    mus = np.stack([mux, muy], axis=-1)
    Ss = np.stack([S11, S12, S22], axis=-1)
    return pis, mus, Ss


def _run_em(X, Q, XP, w, pis0, mus0, Ss0, tol, max_iter):
    """Weighted EM for a single start. Returns None the moment the start
    degenerates (non-PD covariance or non-finite log-likelihood).
    Otherwise runs to `max_iter`; hitting that budget is not a failure
    -- the start is finalized at its current point (`converged=False`)
    and still returned. Returns dict(pis, mus, Ss, ll, converged, n_iter).
    """
    W = float(w.sum())
    pis, mus, Ss = pis0.copy(), mus0.copy(), Ss0.copy()
    prev_ll = None

    for it in range(max_iter):
        try:
            a, b, c, det = _sigma_terms(Ss)
        except np.linalg.LinAlgError:
            return None
        r, log_norm = _e_step_fast(Q, pis, mus, a, b, c, det)
        ll = float(np.dot(w, log_norm) / W)
        if not np.isfinite(ll):
            return None

        if prev_ll is not None:
            denom = max(abs(prev_ll), 1e-300)
            if abs(ll - prev_ll) / denom < tol:
                return dict(pis=pis, mus=mus, Ss=Ss, ll=ll,
                            converged=True, n_iter=it + 1)

        prev_ll = ll
        pis, mus, Ss = _m_step(w, r, XP)

    # Budget exhausted: finalize at the current point with one more
    # E-step so the reported ll pairs with the last M-step's output.
    try:
        a, b, c, det = _sigma_terms(Ss)
    except np.linalg.LinAlgError:
        return None
    r, log_norm = _e_step_fast(Q, pis, mus, a, b, c, det)
    ll = float(np.dot(w, log_norm) / W)
    if not np.isfinite(ll):
        return None
    return dict(pis=pis, mus=mus, Ss=Ss, ll=ll, converged=False, n_iter=max_iter)


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


def _m_step_batched(w: np.ndarray, R: np.ndarray, XP: np.ndarray, K: int, S: int):
    """Same moments as `_m_step` for all K*S components at once; only the
    mixing-weight normalization is taken within each start's own K
    components, not across all K*S."""
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
    S11 = Exx - mux * mux
    S12 = Exy - mux * muy
    S22 = Eyy - muy * muy
    mus = np.stack([mux, muy], axis=-1)
    Ss = np.stack([S11, S12, S22], axis=-1)
    return pis, mus, Ss


def _phase1_batch(Q: np.ndarray, XP: np.ndarray, w: np.ndarray, starts: list,
                   K: int, iters: int):
    """Run every start in `starts` for exactly `iters` EM iterations,
    batched in lockstep (one E-step matmul, one M-step matmul per
    iteration, S = len(starts)); no convergence test, no per-iteration
    bookkeeping. Degeneracy is checked once, after the loop, start by
    start -- the batched matmuls are column-blocked per start, so a NaN
    in one start's columns cannot reach another's. Returns a list of
    dicts (`_run_em`'s shape, `converged=False`, `n_iter=iters` always),
    one per surviving start."""
    S = len(starts)
    W = float(w.sum())
    pis = np.concatenate([s[0] for s in starts])
    mus = np.concatenate([s[1] for s in starts], axis=0)
    Ss = np.concatenate([s[2] for s in starts], axis=0)

    with np.errstate(all='ignore'):
        for _ in range(iters):
            a, b, c, det = _sigma_terms_batched(Ss)
            R, _ = _e_step_batched(Q, pis, mus, a, b, c, det, K, S)
            pis, mus, Ss = _m_step_batched(w, R, XP, K, S)

        # One trailing E-step pairs the reported ll with the last M-step.
        a, b, c, det = _sigma_terms_batched(Ss)
        _, log_norm = _e_step_batched(Q, pis, mus, a, b, c, det, K, S)
        ll = (w @ log_norm) / W  # (S,)

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


def _fit_em_multistart(X, Q, XP, w, K, n_starts, seed, tol, max_iter):
    """Pool = every surviving start, ranked by weighted log-likelihood,
    via the two-phase screen: phase 1 runs every start's short budget in
    lockstep (`_phase1_batch`); the best few are carried, one at a time,
    to `max_iter` (phase 2, sequential `_run_em`)."""
    starts = _starts(X, K, n_starts, seed)
    phase1_budget = min(_SHORT_ITERS, max_iter)

    screened = _phase1_batch(Q, XP, w, starts, K, phase1_budget)
    if not screened:
        return None

    screened.sort(key=lambda r: -r['ll'])
    top = screened[:min(_PROMOTE_N, len(screened))]
    finished = []
    for r in top:
        res = _run_em(X, Q, XP, w, r['pis'], r['mus'], r['Ss'], tol, max_iter)
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

def _fit(X: np.ndarray, w: np.ndarray, cfg: _Cfg, Q: np.ndarray, XP: np.ndarray,
         reference=None):
    """None on failure; else dict(theta, pis, mus, Ss, psi, A, ll,
    score_history, polished, resid). Operates in the caller's own
    (already-centered) coordinates with the caller's own (X-only) `Q`,
    `XP`. Components are labelled by `reference` when given
    (method_notes section 5), else ordered by ascending first-mean
    coordinate."""
    best = _fit_em_multistart(X, Q, XP, w, cfg.K, cfg.n_starts, cfg.seed,
                               cfg.tol, cfg.max_iter)
    if best is None:
        return None
    if reference is None:
        pis, mus, Ss = _canonical_sort(best['pis'], best['mus'], best['Ss'])
    else:
        pis, mus, Ss = _reference_sort(best['pis'], best['mus'], best['Ss'], reference)

    try:
        psi, A, ll = _score_info(X, Q, w, cfg.K, pis, mus, Ss)
    except np.linalg.LinAlgError:
        return None

    theta = _pack(cfg.K, pis, mus, Ss)
    score_history = [float(np.linalg.norm(np.dot(w, psi) / w.sum()))]

    polished = False
    resid = float('nan')
    prev_norm = None
    for _ in range(_MAX_NEWTON):
        psi_bar = np.dot(w, psi) / w.sum()
        norm = float(np.linalg.norm(psi_bar))
        if norm < _NEWTON_SCORE_TOL:
            break
        if (prev_norm is not None and prev_norm < _NEWTON_PROGRESS_FLOOR
                and norm > _NEWTON_PROGRESS_FACTOR * prev_norm):
            # The residual has already reached the noise floor and
            # stopped improving by a clear factor: further steps just
            # re-assemble A for no change in theta.
            break
        prev_norm = norm
        try:
            delta = np.linalg.solve(A, psi_bar)
        except np.linalg.LinAlgError:
            break
        theta_new = theta + delta
        pis_new, mus_new, Ss_new = _unpack(cfg.K, theta_new)
        if np.any(pis_new <= 0.0):
            break
        try:
            psi_new, A_new, ll_new = _score_info(X, Q, w, cfg.K, pis_new, mus_new, Ss_new)
        except np.linalg.LinAlgError:
            break
        if not np.isfinite(ll_new) or ll_new < ll - _ACCEPT_TOL:
            break
        theta, pis, mus, Ss = theta_new, pis_new, mus_new, Ss_new
        psi, A, ll = psi_new, A_new, ll_new
        polished = True
        score_history.append(float(np.linalg.norm(np.dot(w, psi) / w.sum())))

    psi_bar = np.dot(w, psi) / w.sum()
    try:
        resid = float(np.linalg.norm(np.linalg.solve(A, psi_bar))
                       / (1.0 + np.linalg.norm(theta)))
    except np.linalg.LinAlgError:
        resid = float('inf')
    if resid > _ETA_DEFAULT:
        return None

    return dict(theta=theta, pis=pis, mus=mus, Ss=Ss, psi=psi, A=A, ll=ll,
                score_history=score_history, polished=polished, resid=resid)


class GMM2D:
    """See module docstring. `T(X, w) -> ndarray(p,)`, `T.influence(X,
    w) -> ndarray(N, p)`, `T.fit_and_influence(X, w) -> (ndarray(p,),
    ndarray(N, p))`; all three accept `K`, `n_starts`, `seed`, `tol`,
    `max_iter` as per-call keyword overrides, and an optional `prep`
    from `T.prepare(X)`; all three label components against
    `self.reference` (method_notes section 5) when it is not None.
    `self.name`,
    `self.outputs`, `self.eta`, `self.p` are fixed at construction from
    the constructor's own `K`.
    """

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
        self.eta = _ETA_DEFAULT

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
            fit = _fit(Xc, w, cfg, Q, XP, self.reference)
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
            fit = _fit(Xc, w, cfg, Q, XP, self.reference)
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
            fit = _fit(Xc, w, cfg, Q, XP, self.reference)
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
