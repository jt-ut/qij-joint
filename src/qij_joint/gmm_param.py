"""The unconstrained parametrization of `GMM2D`'s theta
(spec/QIJ_estimator_fit_spec.md section 2.2), for the trust-region Newton
finish: pure functions, no state, no finite differences -- every
Jacobian and second derivative below is a closed form.

theta layout, unchanged from `gmm.py`'s `_pack`/`_unpack`/`_idx_mu`/
`_idx_S` (p = (K - 1) + 5*K): pis[0..K-2] (pi_K = 1 - sum, not stored),
then every component's (mu_x, mu_y) contiguously, then every
component's (S11, S12, S22) contiguously. This module never imports
`gmm.py`; its own `_idx_mu`/`_idx_S` below reproduce those index
formulas exactly, since theta's layout is the shared contract the
whole estimator (and this parametrization) is built on.

The unconstrained vector u has the same length and the same block
layout as theta -- weight block, mean block, covariance block, each
the same size and at the same offsets -- with a different map inside
each block:

- weights: u's weight block is the logits a_1..a_{K-1} of a softmax
  with a_K pinned at 0 as the reference component, pi = softmax(a, 0).
- means: unchanged (the identity map); theta's mean block IS u's mean
  block.
- covariances: per component, u's three entries (u1, u2, u3) are the
  Cholesky factor L = [[l11, 0], [l21, l22]] with l11 = exp(u1),
  l21 = u2, l22 = exp(u3); Sigma = L L^T gives S11 = l11**2,
  S12 = l11*l21, S22 = l21**2 + l22**2. u1, u3 sit at theta's S11, S22
  slots and u2 at theta's S12 slot, block index for block index, since
  the two blocks are the same length and offset.

Chain-rule convention (`chain`, below): the caller is maximizing an
objective (the penalized log-likelihood ell_p) of theta; `g_theta` and
`H_theta` are that objective's OWN gradient and Hessian with respect
to theta -- `g_theta = psi_bar` and `H_theta = -A`, `A` the penalized
observed information `_score_info_penalized` already forms (A is the
negative Hessian, so the Hessian itself is -A). `chain` returns the
same objective's gradient and Hessian with respect to u.
"""

import numpy as np


def _idx_mu(K: int, k: int):
    """Same formula as `gmm.py`'s `_idx_mu`: component k's (mu_x, mu_y)
    indices, both in theta and in u (the two layouts share this block
    verbatim)."""
    base = (K - 1) + 2 * k
    return base, base + 1


def _idx_S(K: int, k: int):
    """Same formula as `gmm.py`'s `_idx_S`: component k's (S11, S12,
    S22) indices in theta, and (u1, u2, u3)'s in u -- the two blocks
    are the same length and offset, so one triple of indices serves
    both."""
    base = (K - 1) + 2 * K + 3 * k
    return base, base + 1, base + 2


def _weight_pi(K: int, u: np.ndarray) -> np.ndarray:
    """pi_1..pi_{K-1} (length K-1; pi_K = 1 - sum is never stored) from
    u's weight-block logits a_1..a_{K-1}, softmax against the pinned
    reference a_K = 0: pi_j = exp(a_j) / (1 + sum_i exp(a_i))."""
    a = u[:K - 1]
    ea = np.exp(a)
    D = 1.0 + float(ea.sum())
    return ea / D


def to_unconstrained(K: int, theta: np.ndarray) -> np.ndarray:
    """theta (p,) -> u (p,), the inverse of `from_unconstrained`. Weight
    block: a_j = log(pi_j) - log(pi_K) (the softmax's reference-ratio
    inverse). Mean block: copied. Covariance block: theta's (S11, S12,
    S22) inverted through the Cholesky factor -- l11 = sqrt(S11),
    l21 = S12/l11, l22 = sqrt(S22 - l21**2) (S22 - l21**2 =
    det(Sigma)/S11 > 0 whenever Sigma is PD) -- to u = (log l11, l21,
    log l22)."""
    K = int(K)
    p = (K - 1) + 5 * K
    theta = np.asarray(theta, dtype=float)
    u = np.empty(p)

    pis = theta[:K - 1]
    piK = 1.0 - float(pis.sum())
    u[:K - 1] = np.log(pis) - np.log(piK)

    mstart, mend = K - 1, (K - 1) + 2 * K
    u[mstart:mend] = theta[mstart:mend]

    for k in range(K):
        s11, s12, s22 = _idx_S(K, k)
        S11, S12, S22 = theta[s11], theta[s12], theta[s22]
        l11 = np.sqrt(S11)
        l21 = S12 / l11
        l22 = np.sqrt(S22 - l21 * l21)
        u[s11] = np.log(l11)
        u[s12] = l21
        u[s22] = np.log(l22)
    return u


def from_unconstrained(K: int, u: np.ndarray) -> np.ndarray:
    """u (p,) -> theta (p,) (module docstring). Weight block: softmax
    against the pinned a_K = 0. Mean block: copied. Covariance block:
    the Cholesky map S11 = l11**2, S12 = l11*l21, S22 = l21**2 + l22**2
    with l11 = exp(u1), l21 = u2, l22 = exp(u3)."""
    K = int(K)
    p = (K - 1) + 5 * K
    u = np.asarray(u, dtype=float)
    theta = np.empty(p)

    theta[:K - 1] = _weight_pi(K, u)

    mstart, mend = K - 1, (K - 1) + 2 * K
    theta[mstart:mend] = u[mstart:mend]

    for k in range(K):
        s11, s12, s22 = _idx_S(K, k)
        l11 = np.exp(u[s11])
        l21 = u[s12]
        l22 = np.exp(u[s22])
        theta[s11] = l11 * l11
        theta[s12] = l11 * l21
        theta[s22] = l21 * l21 + l22 * l22
    return theta


def jacobian(K: int, u: np.ndarray) -> np.ndarray:
    """J (p, p), J[i, j] = d theta_i / d u_j, block-diagonal in the
    weight/mean/covariance blocks (no block mixes with another, since
    each is its own closed map of its own u entries).

    Weight block: the reduced-softmax Jacobian d pi_j/d a_i =
    pi_j*(delta_ij - pi_i) (standard softmax derivative, restricted to
    the K-1 free logits; a_K = 0 is a constant, not a variable, so it
    contributes no column).

    Mean block: the identity (the map is the identity).

    Covariance block, per component: d(S11, S12, S22)/d(u1, u2, u3) =
    [[2*S11, 0, 0], [S12, l11, 0], [0, 2*l21, 2*l22**2]] from
    S11 = l11**2, S12 = l11*l21, S22 = l21**2 + l22**2 and
    l11 = exp(u1), l21 = u2, l22 = exp(u3)."""
    K = int(K)
    p = (K - 1) + 5 * K
    u = np.asarray(u, dtype=float)
    J = np.zeros((p, p))

    pi = _weight_pi(K, u)
    J[:K - 1, :K - 1] = -np.outer(pi, pi)
    diag_idx = np.arange(K - 1)
    J[diag_idx, diag_idx] += pi

    mstart, mend = K - 1, (K - 1) + 2 * K
    J[mstart:mend, mstart:mend] = np.eye(mend - mstart)

    for k in range(K):
        s11, s12, s22 = _idx_S(K, k)
        l11 = np.exp(u[s11])
        l21 = u[s12]
        l22 = np.exp(u[s22])
        S11 = l11 * l11
        S12 = l11 * l21
        J[s11, s11] = 2.0 * S11
        J[s12, s11] = S12
        J[s12, s12] = l11
        J[s22, s12] = 2.0 * l21
        J[s22, s22] = 2.0 * l22 * l22
    return J


def chain(K: int, u: np.ndarray, g_theta: np.ndarray, H_theta: np.ndarray):
    """(g_u, H_u), the objective's gradient and Hessian with respect to
    u, from its gradient `g_theta` and Hessian `H_theta` with respect
    to theta (module docstring's convention: the caller passes
    `g_theta = psi_bar`, `H_theta = -A`). g_u = J^T g_theta;
    H_u = J^T H_theta J + sum_i g_theta[i] * d^2 theta_i/du du^T, the
    second term the curvature of the map itself (zero wherever the map
    is affine, i.e. everywhere but the weight and covariance blocks).

    The correction term is block-local, closed form:

    Weight block (from the softmax's second derivative,
    d^2 pi_j/(da_a da_b) = delta_aj*delta_jb*pi_j - delta_aj*pi_j*pi_b
    - delta_jb*pi_a*pi_j - delta_ab*pi_a*pi_j + 2*pi_a*pi_j*pi_b,
    contracted with g_theta's own weight-block entries gw):
    diag(gw*pi) - outer(gw*pi, pi) - outer(pi, gw*pi)
    - diag(pi)*dot(gw, pi) + 2*dot(gw, pi)*outer(pi, pi).

    Covariance block, per component (from S11 = l11**2 having only
    d^2/du1^2 = 4*S11 nonzero; S12 = l11*l21 having d^2/du1^2 = S12 and
    d^2/(du1 du2) = l11; S22 = l21**2 + l22**2 having d^2/du2^2 = 2 and
    d^2/du3^2 = 4*l22**2; all other second partials in the block zero):
    the 3x3 correction gS11*[[4*S11,0,0],[0,0,0],[0,0,0]]
    + gS12*[[S12,l11,0],[l11,0,0],[0,0,0]]
    + gS22*[[0,0,0],[0,2,0],[0,0,4*l22**2]].

    Mean block: no correction (the map is the identity)."""
    K = int(K)
    u = np.asarray(u, dtype=float)
    g_theta = np.asarray(g_theta, dtype=float)
    H_theta = np.asarray(H_theta, dtype=float)

    J = jacobian(K, u)
    g_u = J.T @ g_theta
    H_u = J.T @ H_theta @ J

    if K - 1 > 0:
        pi = _weight_pi(K, u)
        gw = g_theta[:K - 1]
        gwpi = gw * pi
        s = float(np.dot(gw, pi))
        corr = (np.diag(gwpi) - np.outer(gwpi, pi) - np.outer(pi, gwpi)
                - np.diag(pi) * s + 2.0 * s * np.outer(pi, pi))
        H_u[:K - 1, :K - 1] += corr

    for k in range(K):
        s11, s12, s22 = _idx_S(K, k)
        l11 = np.exp(u[s11])
        l21 = u[s12]
        l22 = np.exp(u[s22])
        S11 = l11 * l11
        S12 = l11 * l21
        gS11, gS12, gS22 = g_theta[s11], g_theta[s12], g_theta[s22]

        corr = np.zeros((3, 3))
        corr[0, 0] += gS11 * 4.0 * S11
        corr[0, 0] += gS12 * S12
        corr[0, 1] += gS12 * l11
        corr[1, 0] += gS12 * l11
        corr[1, 1] += gS22 * 2.0
        corr[2, 2] += gS22 * 4.0 * l22 * l22

        idxs = (s11, s12, s22)
        for ii, ai in enumerate(idxs):
            for jj, bj in enumerate(idxs):
                H_u[ai, bj] += corr[ii, jj]

    return g_u, H_u


def coordinate_scales(K: int, Scov: np.ndarray) -> np.ndarray:
    """s (p,), the unconstrained coordinates' own scale (spec section
    5): 1 for the weight logits and the covariance log-diagonals (u1,
    u3); the data's column standard deviation, taken from `Scov`'s
    diagonal (`_weighted_cov`'s w-weighted covariance), for the means
    (x or y matching the coordinate) and for the off-diagonal Cholesky
    entry u2. Choice made here (not settled by the spec text): u2's
    scale is sqrt(Scov[0, 0]) always, not sqrt(Scov[1, 1]) -- u2 = l21
    is an x-times-a-dimensionless-slope term (l11, an x-scale, times
    the correlation-bearing ratio), so the x column standard deviation
    is its natural unit."""
    K = int(K)
    p = (K - 1) + 5 * K
    Scov = np.asarray(Scov, dtype=float)
    s = np.ones(p)
    sx = np.sqrt(Scov[0, 0])
    sy = np.sqrt(Scov[1, 1])

    for k in range(K):
        mx, my = _idx_mu(K, k)
        s[mx] = sx
        s[my] = sy

    for k in range(K):
        _, s12, _ = _idx_S(K, k)
        s[s12] = sx
    return s


def scaled_gradient_norm(g_u: np.ndarray, s: np.ndarray, ell: float) -> float:
    """Dennis and Schnabel's relative gradient (spec sections 2.3, 5):
    max_i |g_u[i]| * s[i] / max(|ell|, 1)."""
    g_u = np.asarray(g_u, dtype=float)
    s = np.asarray(s, dtype=float)
    return float(np.max(np.abs(g_u) * s) / max(abs(ell), 1.0))
