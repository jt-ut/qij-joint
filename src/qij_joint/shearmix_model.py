"""Pure math for `ShearMix2D` (spec/QIJ_shearmix_interface.md section 3):
`Kg` free full-covariance Gaussians (cloud and cores) plus ONE sheared-
Gaussian filament component,

    f(x) = sum_{k=1..Kg} pi_k N(x | mu_k, Sigma_k)  +  pi_f f_fil(x)
    f_fil(x, y) = N(x | m, s2x) * N(y - h(x) | 0, s2p)
    h(x) = b0 + b1 sin(omega x) + b2 cos(omega x)

`omega` is FIXED (constructor argument of `ShearMix2D`, default
`OMEGA = pi/2.64`), never estimated. The basis uses RAW data x --
no centering anywhere (unlike `gmm.py`, which shifts X by its own
unweighted mean before fitting): `Sigma`'s double role there is to
protect `E[xx^T] - mu mu^T` from catastrophic cancellation far from the
origin, but this estimator never forms that difference -- the Gaussian
block's quadratic form and the filament's two 1-D residuals are both
built directly from `(x - mu)` / `(x - m)` / `(y - h(x))`, so there is
nothing for a shift to protect.

Parameter layout (theta, data coordinates), p = 6*Kg + 6:

    pi1..pi{Kg}  (pi_f = 1 - sum, implied, NOT stored)
    mu1x, mu1y, ..., mu{Kg}x, mu{Kg}y
    S1_11, S1_12, S1_22, ..., S{Kg}_11, S{Kg}_12, S{Kg}_22
    fil_m, fil_s2x, fil_b0, fil_b1, fil_b2, fil_s2p

The estimand is the maximizer of the penalized log-likelihood per unit
weight

    ell_p(theta) = (1/W) sum_i w_i log f(x_i) - a * P(theta),
    W = sum w_i,  a = 1/W,
    P = sum_k[tr(S Sigma_k^-1) + log det Sigma_k]
        + [S11/s2x + log s2x + S22/s2p + log s2p],

`S` the w-weighted covariance of the rows (Chen and Tan 2009), the
Gaussian part exactly `gmm.py`'s own penalty and the filament term the
same Chen-Tan form applied to its two variances with the diagonal of
`S`. This module matches `gmm.py`'s OWN normalization convention
exactly, not a literal reading of the formula above: `gmm._penalized_ll`
computes `dot(w, log_norm)/W - a_pen * penalty_sum / W` with
`a_pen = 1/W`, i.e. the penalty is divided by `W` TWICE over (once via
`a_pen`, once via the per-unit-weight normalization) -- `penalized_ll`
below reproduces that exactly (`ll = ell_raw/W - a_pen*P(theta)/W`),
confirmed against `gmm.py`'s own code and method_notes.md section 5
("internally, `ll` always means `ell_p` ... per unit weight
(`ell_p / sum(w)`)", with `ell_p` itself at the Chen-Tan TOTAL
(unnormalized) scale, `ell_p = ell - a*P`, `a = 1/sum(w)`).

`info` is NOT Louis's identity here: per the build interface, `A` is
the negative Jacobian of this module's own analytic `grad`, by central
finite differences (no analytic Hessian is implemented), symmetrized.

A note on `score_rows` and the "weighted mean == grad" contract some
reports else where expect of an M-estimator's per-row score: this
module's `score_rows` mirrors GMM2D's own penalized per-row psi
EXACTLY (`gmm._score_info` / `_score_info_penalized` /
`_penalty_influence_extra`), i.e. the raw per-row score PLUS the
penalty's own sensitivity to a row's weight through `S(w)` and `a(w)`
at fixed theta -- this is what influence (`A^-1 @ score_rows.T`) needs
to be correct. Mirroring that formula means the weighted mean of
`score_rows` is NOT exactly `grad(theta)` at a finite `W`: the two
differ by `2*g_pen/W` where `g_pen` is the penalty's own contribution to
`grad` (order `a_pen = 1/W`, so the gap shrinks as `1/N`). See this
module's own self-check report for the measured size and the algebraic
reason (verified both by hand and empirically against `gmm.py` itself).
"""

from collections import namedtuple

import numpy as np

OMEGA = float(np.pi / 2.64)
_LOG2PI = float(np.log(2.0 * np.pi))

Prep = namedtuple('Prep', ['X', 'Bx'])
Pen = namedtuple('Pen', ['W', 'a', 'Scov'])


# ======================================================================
# Layout.
# ======================================================================

def n_params(Kg: int) -> int:
    return 6 * int(Kg) + 6


def make_outputs(Kg: int) -> tuple:
    Kg = int(Kg)
    names = [f'pi{j + 1}' for j in range(Kg)]
    names += [f'mu{k + 1}{ax}' for k in range(Kg) for ax in ('x', 'y')]
    names += [f'S{k + 1}_{s}' for k in range(Kg) for s in ('11', '12', '22')]
    names += ['fil_m', 'fil_s2x', 'fil_b0', 'fil_b1', 'fil_b2', 'fil_s2p']
    return tuple(names)


def _idx_mu(Kg: int, k: int):
    base = Kg + 2 * k
    return base, base + 1


def _idx_S(Kg: int, k: int):
    base = 3 * Kg + 3 * k
    return base, base + 1, base + 2


def _idx_fil(Kg: int):
    base = 6 * Kg
    return base, base + 1, base + 2, base + 3, base + 4, base + 5


def pack(Kg: int, pis_g: np.ndarray, mus: np.ndarray, covs: np.ndarray,
          fil) -> np.ndarray:
    Kg = int(Kg)
    theta = np.empty(n_params(Kg))
    theta[:Kg] = pis_g
    for k in range(Kg):
        mx, my = _idx_mu(Kg, k)
        theta[mx], theta[my] = mus[k, 0], mus[k, 1]
        s11, s12, s22 = _idx_S(Kg, k)
        theta[s11], theta[s12], theta[s22] = covs[k, 0, 0], covs[k, 0, 1], covs[k, 1, 1]
    fm, fs2x, fb0, fb1, fb2, fs2p = _idx_fil(Kg)
    theta[fm], theta[fs2x], theta[fb0], theta[fb1], theta[fb2], theta[fs2p] = fil
    return theta


def unpack(Kg: int, theta: np.ndarray):
    Kg = int(Kg)
    theta = np.asarray(theta, dtype=float)
    pis_g = theta[:Kg].copy()
    pi_f = 1.0 - float(pis_g.sum())
    mus = np.empty((Kg, 2))
    covs = np.empty((Kg, 2, 2))
    for k in range(Kg):
        mx, my = _idx_mu(Kg, k)
        mus[k, 0], mus[k, 1] = theta[mx], theta[my]
        s11, s12, s22 = _idx_S(Kg, k)
        covs[k, 0, 0] = theta[s11]
        covs[k, 0, 1] = covs[k, 1, 0] = theta[s12]
        covs[k, 1, 1] = theta[s22]
    fm, fs2x, fb0, fb1, fb2, fs2p = _idx_fil(Kg)
    fil = np.array([theta[fm], theta[fs2x], theta[fb0], theta[fb1], theta[fb2], theta[fs2p]])
    return pis_g, pi_f, mus, covs, fil


def feasible(Kg: int, theta: np.ndarray) -> bool:
    Kg = int(Kg)
    theta = np.asarray(theta, dtype=float)
    if not np.all(np.isfinite(theta)):
        return False
    pis_g = theta[:Kg]
    if not np.all(pis_g > 0.0):
        return False
    pi_f = 1.0 - float(pis_g.sum())
    if not (pi_f > 0.0):
        return False
    for k in range(Kg):
        s11, s12, s22 = _idx_S(Kg, k)
        S11, S12, S22 = theta[s11], theta[s12], theta[s22]
        det = S11 * S22 - S12 * S12
        if not (S11 > 0.0 and det > 0.0):
            return False
    fm, fs2x, fb0, fb1, fb2, fs2p = _idx_fil(Kg)
    if not (theta[fs2x] > 0.0 and theta[fs2p] > 0.0):
        return False
    return True


# ======================================================================
# Prep / penalty setup.
# ======================================================================

def prepare(X: np.ndarray, omega: float = OMEGA) -> Prep:
    X = np.asarray(X, dtype=float)
    x = X[:, 0]
    Bx = np.empty((X.shape[0], 3))
    Bx[:, 0] = 1.0
    Bx[:, 1] = np.sin(omega * x)
    Bx[:, 2] = np.cos(omega * x)
    return Prep(X.copy(), Bx)


def penalty_setup(prep: Prep, w: np.ndarray) -> Pen:
    """W = sum(w), a = 1/W (`gmm._weighted_cov`'s own `a_pen`, GMM2D's
    convention), Scov the w-weighted covariance of `prep.X` about ITS
    OWN w-weighted mean. No unweighted shift is applied anywhere in this
    module (module docstring), but `Scov` is shift-invariant (it is
    built about its own weighted mean) so this is identical to applying
    `gmm._weighted_cov` to any constant shift of X, including none."""
    w = np.asarray(w, dtype=float)
    X = prep.X
    W = float(w.sum())
    xbar = (w @ X) / W
    d = X - xbar
    Scov = (d * w[:, None]).T @ d / W
    return Pen(W, 1.0 / W, Scov)


# ======================================================================
# Densities / E-step.
# ======================================================================

def log_joint(prep: Prep, Kg: int, theta: np.ndarray) -> np.ndarray:
    """(N, Kg+1): column k < Kg is log(pi_k) + log N(x | mu_k, Sigma_k);
    column Kg is log(pi_f) + log f_fil(x, y). Raises
    np.linalg.LinAlgError if any Sigma_k is not PD."""
    Kg = int(Kg)
    X, Bx = prep.X, prep.Bx
    N = X.shape[0]
    pis_g, pi_f, mus, covs, fil = unpack(Kg, theta)
    out = np.empty((N, Kg + 1))
    for k in range(Kg):
        a, b, c = covs[k, 0, 0], covs[k, 0, 1], covs[k, 1, 1]
        det = a * c - b * b
        if not (np.isfinite(det) and det > 0.0):
            raise np.linalg.LinAlgError("non-positive-definite covariance")
        rx = X[:, 0] - mus[k, 0]
        ry = X[:, 1] - mus[k, 1]
        quad = (c * rx * rx - 2.0 * b * rx * ry + a * ry * ry) / det
        out[:, k] = np.log(pis_g[k]) - _LOG2PI - 0.5 * np.log(det) - 0.5 * quad
    m, s2x, b0, b1, b2, s2p = fil
    if not (s2x > 0.0 and s2p > 0.0):
        raise np.linalg.LinAlgError("non-positive filament variance")
    hx = Bx @ np.array([b0, b1, b2])
    ex = X[:, 0] - m
    ey = X[:, 1] - hx
    logfil = (-_LOG2PI - 0.5 * np.log(s2x) - 0.5 * np.log(s2p)
              - 0.5 * ex * ex / s2x - 0.5 * ey * ey / s2p)
    out[:, Kg] = np.log(pi_f) + logfil
    return out


def e_step(prep: Prep, Kg: int, theta: np.ndarray):
    """(R (N, Kg+1), logf (N,)): responsibilities and log f(x_i)."""
    L = log_joint(prep, Kg, theta)
    Lmax = L.max(axis=1)
    E = np.exp(L - Lmax[:, None])
    ssum = E.sum(axis=1)
    R = E / ssum[:, None]
    logf = Lmax + np.log(ssum)
    return R, logf


# ======================================================================
# Penalized M-step.
# ======================================================================

def m_step(prep: Prep, w: np.ndarray, R: np.ndarray, pen: Pen, Kg: int) -> np.ndarray:
    """Closed-form penalized M-step. Gaussians: Chen-Tan MAP blend of the
    responsibility-weighted covariance with `pen.Scov`, exactly
    `gmm._m_step`'s own formula. Filament: `m`, `s2x` the weighted
    mean/variance of x (variance Chen-Tan-blended with `Scov[0,0]`);
    `b` the weighted least-squares fit of y on `Bx` (untouched by the
    penalty, like `mu_k`); `s2p` the weighted residual variance of that
    fit (blended with `Scov[1,1]`)."""
    Kg = int(Kg)
    X, Bx = prep.X, prep.Bx
    W, a_pen, Scov = pen
    wr = w[:, None] * R  # (N, Kg+1)
    n = wr.sum(axis=0)  # (Kg+1,)

    pis_g = n[:Kg] / W
    mus = np.empty((Kg, 2))
    covs = np.empty((Kg, 2, 2))
    for k in range(Kg):
        nk = n[k]
        wrk = wr[:, k]
        mu_k = (wrk @ X) / nk
        res = X - mu_k
        Craw = (res * wrk[:, None]).T @ res / nk
        Sigma = (nk * Craw + 2.0 * a_pen * Scov) / (nk + 2.0 * a_pen)
        mus[k] = mu_k
        covs[k] = Sigma

    nf = n[Kg]
    wrf = wr[:, Kg]
    m = (wrf @ X[:, 0]) / nf
    s2x_raw = (wrf @ (X[:, 0] - m) ** 2) / nf
    s2x = (nf * s2x_raw + 2.0 * a_pen * Scov[0, 0]) / (nf + 2.0 * a_pen)

    Wmat = Bx * wrf[:, None]
    AtA = Bx.T @ Wmat
    rhs = Wmat.T @ X[:, 1]
    b = np.linalg.solve(AtA, rhs)
    resid = X[:, 1] - Bx @ b
    s2p_raw = (wrf @ (resid ** 2)) / nf
    s2p = (nf * s2p_raw + 2.0 * a_pen * Scov[1, 1]) / (nf + 2.0 * a_pen)

    fil = np.array([m, s2x, b[0], b[1], b[2], s2p])
    return pack(Kg, pis_g, mus, covs, fil)


# ======================================================================
# Penalized log-likelihood, gradient, per-row score.
# ======================================================================

def _penalty_value(Kg: int, theta: np.ndarray, Scov: np.ndarray) -> float:
    pis_g, pi_f, mus, covs, fil = unpack(Kg, theta)
    total = 0.0
    for k in range(Kg):
        a, b, c = covs[k, 0, 0], covs[k, 0, 1], covs[k, 1, 1]
        det = a * c - b * b
        Pk = np.array([[c, -b], [-b, a]]) / det
        total += float(np.trace(Scov @ Pk)) + float(np.log(det))
    m, s2x, b0, b1, b2, s2p = fil
    total += Scov[0, 0] / s2x + np.log(s2x) + Scov[1, 1] / s2p + np.log(s2p)
    return total


def penalized_ll(prep: Prep, w: np.ndarray, Kg: int, theta: np.ndarray, pen: Pen) -> float:
    """ell_p per unit weight (GMM2D's own convention, module docstring).
    NaN if infeasible."""
    Kg = int(Kg)
    theta = np.asarray(theta, dtype=float)
    if not feasible(Kg, theta):
        return float('nan')
    W, a_pen, Scov = pen
    try:
        _, logf = e_step(prep, Kg, theta)
    except np.linalg.LinAlgError:
        return float('nan')
    ll_raw = float(np.dot(w, logf)) / W
    Pval = _penalty_value(Kg, theta, Scov)
    return ll_raw - a_pen * Pval / W


def _raw_score_rows(prep: Prep, Kg: int, theta: np.ndarray, R: np.ndarray) -> np.ndarray:
    """(N, p) unpenalized per-row score d log f(x_i) / d theta."""
    Kg = int(Kg)
    X, Bx = prep.X, prep.Bx
    N = X.shape[0]
    pis_g, pi_f, mus, covs, fil = unpack(Kg, theta)
    psi = np.zeros((N, n_params(Kg)))

    for j in range(Kg):
        psi[:, j] = R[:, j] / pis_g[j] - R[:, Kg] / pi_f

    for k in range(Kg):
        a, b, c = covs[k, 0, 0], covs[k, 0, 1], covs[k, 1, 1]
        det = a * c - b * b
        Pk = np.array([[c, -b], [-b, a]]) / det
        res = X - mus[k]
        u = res @ Pk
        rk = R[:, k]
        mx, my = _idx_mu(Kg, k)
        s11, s12, s22 = _idx_S(Kg, k)
        psi[:, mx] = rk * u[:, 0]
        psi[:, my] = rk * u[:, 1]
        G00 = 0.5 * (u[:, 0] ** 2 - Pk[0, 0])
        G11 = 0.5 * (u[:, 1] ** 2 - Pk[1, 1])
        G01 = 0.5 * (u[:, 0] * u[:, 1] - Pk[0, 1])
        psi[:, s11] = rk * G00
        psi[:, s12] = rk * 2.0 * G01
        psi[:, s22] = rk * G11

    m, s2x, b0, b1, b2, s2p = fil
    rf = R[:, Kg]
    ex = X[:, 0] - m
    hx = Bx @ np.array([b0, b1, b2])
    ey = X[:, 1] - hx
    fm, fs2x, fb0, fb1, fb2, fs2p = _idx_fil(Kg)
    psi[:, fm] = rf * ex / s2x
    psi[:, fs2x] = rf * (-0.5 / s2x + 0.5 * ex * ex / (s2x * s2x))
    psi[:, fb0] = rf * ey / s2p * Bx[:, 0]
    psi[:, fb1] = rf * ey / s2p * Bx[:, 1]
    psi[:, fb2] = rf * ey / s2p * Bx[:, 2]
    psi[:, fs2p] = rf * (-0.5 / s2p + 0.5 * ey * ey / (s2p * s2p))
    return psi


def _penalty_grad(Kg: int, theta: np.ndarray, Scov: np.ndarray, a_pen: float) -> np.ndarray:
    """g (p,) = d(-a_pen * P(theta)) / d theta, zero outside the Sigma_k
    and (s2x, s2p) blocks -- `gmm._penalty_terms`'s own formula, plus
    the scalar (1-D) analog for the filament's two variances."""
    Kg = int(Kg)
    pis_g, pi_f, mus, covs, fil = unpack(Kg, theta)
    g = np.zeros(n_params(Kg))
    for k in range(Kg):
        a, b, c = covs[k, 0, 0], covs[k, 0, 1], covs[k, 1, 1]
        det = a * c - b * b
        Pk = np.array([[c, -b], [-b, a]]) / det
        Gk = Pk - Pk @ Scov @ Pk
        s11, s12, s22 = _idx_S(Kg, k)
        g[s11] = -a_pen * Gk[0, 0]
        g[s12] = -a_pen * (Gk[0, 1] + Gk[1, 0])
        g[s22] = -a_pen * Gk[1, 1]
    m, s2x, b0, b1, b2, s2p = fil
    Gs2x = 1.0 / s2x - Scov[0, 0] / (s2x * s2x)
    Gs2p = 1.0 / s2p - Scov[1, 1] / (s2p * s2p)
    fm, fs2x, fb0, fb1, fb2, fs2p = _idx_fil(Kg)
    g[fs2x] = -a_pen * Gs2x
    g[fs2p] = -a_pen * Gs2p
    return g


def grad(prep: Prep, w: np.ndarray, Kg: int, theta: np.ndarray, pen: Pen) -> np.ndarray:
    """Analytic d ell_p / d theta."""
    Kg = int(Kg)
    W, a_pen, Scov = pen
    R, _ = e_step(prep, Kg, theta)
    psi_raw = _raw_score_rows(prep, Kg, theta, R)
    psi_bar_raw = (w @ psi_raw) / W
    g = _penalty_grad(Kg, theta, Scov, a_pen)
    return psi_bar_raw + g / W


def score_rows(prep: Prep, w: np.ndarray, Kg: int, theta: np.ndarray, pen: Pen) -> np.ndarray:
    """(N, p) per-row psi for `influence = solve(A, psi.T).T`, EXACTLY
    GMM2D's definition (`gmm._score_info` for the raw part,
    `gmm._penalty_influence_extra` for the penalty's own data dependence
    through S(w) at fixed theta, extended with the filament's scalar
    (s2x, s2p) analog of the Gaussian Sigma_k correction). See the
    module docstring: the w-weighted mean of this is NOT exactly
    `grad(theta)` at finite W (gap `2*g_pen/W`, `O(1/N)`), which is a
    property of GMM2D's OWN construction, not a bug introduced here --
    verified both algebraically and empirically against `gmm.py` itself
    (this module's self-check report)."""
    Kg = int(Kg)
    W, a_pen, Scov = pen
    R, _ = e_step(prep, Kg, theta)
    psi_raw = _raw_score_rows(prep, Kg, theta, R)

    X = prep.X
    xbar = (w @ X) / W
    dvec = X - xbar
    pis_g, pi_f, mus, covs, fil = unpack(Kg, theta)
    extra = np.zeros_like(psi_raw)

    for k in range(Kg):
        a, b, c = covs[k, 0, 0], covs[k, 0, 1], covs[k, 1, 1]
        det = a * c - b * b
        Pk = np.array([[c, -b], [-b, a]]) / det
        PS = Pk @ Scov
        Gk = Pk - PS @ Pk
        PkScovPk = PS @ Pk
        u = dvec @ Pk
        s11, s12, s22 = _idx_S(Kg, k)
        M00 = (a_pen ** 2) * Gk[0, 0] + (a_pen / W) * (u[:, 0] ** 2 - PkScovPk[0, 0])
        M11 = (a_pen ** 2) * Gk[1, 1] + (a_pen / W) * (u[:, 1] ** 2 - PkScovPk[1, 1])
        M01 = (a_pen ** 2) * Gk[0, 1] + (a_pen / W) * (u[:, 0] * u[:, 1] - PkScovPk[0, 1])
        extra[:, s11] = M00
        extra[:, s12] = 2.0 * M01
        extra[:, s22] = M11

    m, s2x, b0, b1, b2, s2p = fil
    fm, fs2x, fb0, fb1, fb2, fs2p = _idx_fil(Kg)
    Gs2x = 1.0 / s2x - Scov[0, 0] / (s2x * s2x)
    Gs2p = 1.0 / s2p - Scov[1, 1] / (s2p * s2p)
    extra[:, fs2x] = (a_pen ** 2) * Gs2x + (a_pen / W) * (dvec[:, 0] ** 2 - Scov[0, 0]) / (s2x * s2x)
    extra[:, fs2p] = (a_pen ** 2) * Gs2p + (a_pen / W) * (dvec[:, 1] ** 2 - Scov[1, 1]) / (s2p * s2p)

    return psi_raw + extra


def info(prep: Prep, w: np.ndarray, Kg: int, theta: np.ndarray, pen: Pen,
         rel_step: float = 1e-5) -> np.ndarray:
    """A = -(d grad / d theta), central finite differences of the
    analytic `grad`, step h_j = rel_step * max(1, |theta_j|),
    symmetrized."""
    Kg = int(Kg)
    theta = np.asarray(theta, dtype=float)
    p = n_params(Kg)
    A = np.empty((p, p))
    for j in range(p):
        h = rel_step * max(1.0, abs(theta[j]))
        tp = theta.copy()
        tp[j] += h
        tm = theta.copy()
        tm[j] -= h
        gp = grad(prep, w, Kg, tp, pen)
        gm = grad(prep, w, Kg, tm, pen)
        A[:, j] = -(gp - gm) / (2.0 * h)
    return 0.5 * (A + A.T)


# ======================================================================
# Unconstrained parametrization (mirrors gmm_param.py).
# ======================================================================

def _weight_pi(Kg: int, u: np.ndarray) -> np.ndarray:
    a = u[:Kg]
    ea = np.exp(a)
    D = 1.0 + float(ea.sum())
    return ea / D


def to_u(Kg: int, theta: np.ndarray) -> np.ndarray:
    """Weights: log-ratio to pi_f (a_j = log(pi_j) - log(pi_f), the
    softmax's reference-ratio inverse, pi_f the reference category).
    Gaussian covs: Cholesky log/identity, as `gmm_param.to_unconstrained`.
    Filament: m, b0, b1, b2 identity; s2x, s2p -> log."""
    Kg = int(Kg)
    theta = np.asarray(theta, dtype=float)
    p = n_params(Kg)
    u = np.empty(p)

    pis_g = theta[:Kg]
    pi_f = 1.0 - float(pis_g.sum())
    u[:Kg] = np.log(pis_g) - np.log(pi_f)

    mstart, mend = Kg, 3 * Kg
    u[mstart:mend] = theta[mstart:mend]

    for k in range(Kg):
        s11, s12, s22 = _idx_S(Kg, k)
        S11, S12, S22 = theta[s11], theta[s12], theta[s22]
        l11 = np.sqrt(S11)
        l21 = S12 / l11
        l22 = np.sqrt(S22 - l21 * l21)
        u[s11] = np.log(l11)
        u[s12] = l21
        u[s22] = np.log(l22)

    fm, fs2x, fb0, fb1, fb2, fs2p = _idx_fil(Kg)
    u[fm] = theta[fm]
    u[fs2x] = np.log(theta[fs2x])
    u[fb0], u[fb1], u[fb2] = theta[fb0], theta[fb1], theta[fb2]
    u[fs2p] = np.log(theta[fs2p])
    return u


def from_u(Kg: int, u: np.ndarray) -> np.ndarray:
    Kg = int(Kg)
    u = np.asarray(u, dtype=float)
    p = n_params(Kg)
    theta = np.empty(p)

    theta[:Kg] = _weight_pi(Kg, u)

    mstart, mend = Kg, 3 * Kg
    theta[mstart:mend] = u[mstart:mend]

    for k in range(Kg):
        s11, s12, s22 = _idx_S(Kg, k)
        l11 = np.exp(u[s11])
        l21 = u[s12]
        l22 = np.exp(u[s22])
        theta[s11] = l11 * l11
        theta[s12] = l11 * l21
        theta[s22] = l21 * l21 + l22 * l22

    fm, fs2x, fb0, fb1, fb2, fs2p = _idx_fil(Kg)
    theta[fm] = u[fm]
    theta[fs2x] = np.exp(u[fs2x])
    theta[fb0], theta[fb1], theta[fb2] = u[fb0], u[fb1], u[fb2]
    theta[fs2p] = np.exp(u[fs2p])
    return theta


def jac_u(Kg: int, u: np.ndarray) -> np.ndarray:
    """J[i, j] = d theta_i / d u_j, block-diagonal in the weight/mean/
    covariance/filament blocks."""
    Kg = int(Kg)
    u = np.asarray(u, dtype=float)
    p = n_params(Kg)
    J = np.zeros((p, p))

    pi_g = _weight_pi(Kg, u)
    J[:Kg, :Kg] = -np.outer(pi_g, pi_g)
    diag_idx = np.arange(Kg)
    J[diag_idx, diag_idx] += pi_g

    mstart, mend = Kg, 3 * Kg
    J[mstart:mend, mstart:mend] = np.eye(mend - mstart)

    for k in range(Kg):
        s11, s12, s22 = _idx_S(Kg, k)
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

    fm, fs2x, fb0, fb1, fb2, fs2p = _idx_fil(Kg)
    J[fm, fm] = 1.0
    J[fs2x, fs2x] = np.exp(u[fs2x])
    J[fb0, fb0] = 1.0
    J[fb1, fb1] = 1.0
    J[fb2, fb2] = 1.0
    J[fs2p, fs2p] = np.exp(u[fs2p])
    return J


def coordinate_scales(Kg: int, Scov: np.ndarray) -> np.ndarray:
    """s (p,): 1 for the weight logits and the covariance/filament
    log-mapped entries (u1, u3 per Gaussian; fil_s2x, fil_s2p); the
    data's column standard deviation (`Scov`'s diagonal) for the means
    and for the off-diagonal Cholesky entry u2 (as `gmm_param`), and for
    the filament's own location-type entries (`fil_m` an x-location like
    mu_x; `fil_b0/b1/b2` y-scale entries, since `h(x)` has y's units,
    like mu_y)."""
    Kg = int(Kg)
    Scov = np.asarray(Scov, dtype=float)
    p = n_params(Kg)
    s = np.ones(p)
    sx = np.sqrt(Scov[0, 0])
    sy = np.sqrt(Scov[1, 1])

    for k in range(Kg):
        mx, my = _idx_mu(Kg, k)
        s[mx] = sx
        s[my] = sy
    for k in range(Kg):
        _, s12, _ = _idx_S(Kg, k)
        s[s12] = sx

    fm, fs2x, fb0, fb1, fb2, fs2p = _idx_fil(Kg)
    s[fm] = sx
    s[fb0] = sy
    s[fb1] = sy
    s[fb2] = sy
    return s
