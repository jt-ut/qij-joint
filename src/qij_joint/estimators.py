"""The paper's estimators: pareto shape/tail, mvt nu/tail, fp, Chabrier.

Every estimator is a callable `T(X, w) -> ndarray(q)` carrying `name`,
`outputs`, `eta`, and an analytic `T.influence(X, w) -> ndarray(N, q)`.
`w` is non-negative and sums to `len(X)`. Three conventions hold for
every estimator: it never raises (a failed fit returns NaN of shape
`(q,)`); it holds no state between calls; it does not know which method
is calling it, so the weight vector alone distinguishes a bootstrap
replicate from a QIJ perturbation.

An estimator whose calls recompute a quantity that depends on X alone
also exposes `T.prepare(X) -> prep`, so a caller evaluating the same X
at many weight vectors computes that quantity once and passes it back
as `T(X, w, prep=prep)`; a plain `T(X, w)` still recomputes it inline
and returns bit-identical results.

Closed forms (`pareto_shape`, `pareto_tail`, `mvt_tail`) are exact
algebraic evaluations, so `eta` is machine precision. `mvt_nu` is a
solved scalar and `fp` a vector closed form; each declares the
precision of its own solve. `Chabrier` is a lognormal-below/power-law-
above fit, C0-joined at a fixed break mass, by a bounded trust-constr
search supplied its own exact gradient and Hessian.
"""

from math import exp, log, pi, sqrt

import numpy as np
from scipy import optimize
from scipy.special import digamma, ndtr
from scipy.stats import f as f_dist

from . import datasets as _datasets

EPS = float(np.finfo(np.float64).eps)


def estimator(outputs, eta, name=None):
    """Attach `name`, `outputs`, `eta` to a closed-form estimator function."""
    def decorate(fn):
        fn.name = name if name is not None else fn.__name__
        fn.outputs = tuple(outputs)
        fn.eta = float(eta)
        return fn
    return decorate


# A fit that runs to a bound is not a stationary point of its objective, so
# neither the fit nor its influence means anything there: the evaluation is
# a counted failure (NaN), never a silent number. Tolerance is a fraction
# of the box's own WIDTH, not of eta, since eta measures how precisely the
# objective is evaluated, not how close to a bound a search parks.
_BOX_TOL = 1e-4


def _at_bound(value: float, lo: float, hi: float, tol: float = _BOX_TOL) -> bool:
    """True when `value` lies within `tol` of the width of [lo, hi] of
    either end, in the coordinate the fit's own search walks in."""
    width = hi - lo
    tol_abs = tol * width
    return abs(value - lo) <= tol_abs or abs(value - hi) <= tol_abs


# ======================================================================
# Pareto shape (Hill estimator) and Pareto tail probability
# ======================================================================

PARETO_X_MIN = 1.0
PARETO_ALPHA_TRUE = 2.0
# The 99th percentile of the true Pareto(alpha=2.0, x_min=1.0) law, frozen
# here and never recomputed from a draw.
PARETO_TAIL_C = PARETO_X_MIN * 0.01 ** (-1.0 / PARETO_ALPHA_TRUE)


def _pareto_shape_prepare(X: np.ndarray) -> dict:
    x = np.asarray(X, dtype=float).reshape(-1)
    return dict(log_ratio=np.log(x / PARETO_X_MIN))


@estimator(outputs=('alpha',), eta=EPS, name='shape')
def pareto_shape(X: np.ndarray, w: np.ndarray, prep: dict = None) -> np.ndarray:
    """Hill estimator: alpha = sum(w) / sum(w log(X / x_min))."""
    try:
        log_ratio = prep['log_ratio'] if prep is not None else _pareto_shape_prepare(X)['log_ratio']
        alpha = float(w.sum() / np.dot(w, log_ratio))
        if not np.isfinite(alpha):
            return np.full(1, np.nan)
        return np.array([alpha])
    except Exception:
        return np.full(1, np.nan)


def _pareto_shape_influence(X: np.ndarray, w: np.ndarray) -> np.ndarray:
    """psi_i = 1/alpha - log(x_i/x_min); A = 1/alpha^2; IF = psi/A."""
    theta = pareto_shape(X, w)
    if not np.all(np.isfinite(theta)):
        return np.full((len(X), 1), np.nan)
    alpha = theta[0]
    x = np.asarray(X, dtype=float).reshape(-1)
    psi = 1.0 / alpha - np.log(x / PARETO_X_MIN)
    return (psi * alpha ** 2)[:, None]


pareto_shape.prepare = _pareto_shape_prepare
pareto_shape.influence = _pareto_shape_influence


def _pareto_tail_prepare(X: np.ndarray) -> dict:
    x = np.asarray(X, dtype=float).reshape(-1)
    return dict(indicator=(x > PARETO_TAIL_C).astype(float))


@estimator(outputs=('P_tail',), eta=EPS, name='tail')
def pareto_tail(X: np.ndarray, w: np.ndarray, prep: dict = None) -> np.ndarray:
    """T = sum(w 1{X > c}) / sum(w), c fixed at the true 99th percentile."""
    try:
        indicator = prep['indicator'] if prep is not None else _pareto_tail_prepare(X)['indicator']
        theta = float(np.dot(w, indicator) / w.sum())
        return np.array([theta])
    except Exception:
        return np.full(1, np.nan)


def _pareto_tail_influence(X: np.ndarray, w: np.ndarray) -> np.ndarray:
    """psi_i = 1{x_i > c} - theta; A = 1."""
    theta = pareto_tail(X, w)
    if not np.all(np.isfinite(theta)):
        return np.full((len(X), 1), np.nan)
    x = np.asarray(X, dtype=float).reshape(-1)
    indicator = (x > PARETO_TAIL_C).astype(float)
    return (indicator - theta[0])[:, None]


pareto_tail.prepare = _pareto_tail_prepare
pareto_tail.influence = _pareto_tail_influence


# ======================================================================
# MVT nu (solved scalar) and MVT tail probability
# ======================================================================

MVT_D = 10
MVT_NU_TRUE = 5.0
# The 99th percentile of ||x|| under the true d=10, nu=5.0 law.
MVT_TAIL_C = float(np.sqrt(MVT_D * f_dist.ppf(0.99, MVT_D, MVT_NU_TRUE)))

_MVT_NU_LO, _MVT_NU_HI = 0.5, 200.0
_MVT_NU_MIN, _MVT_NU_MAX = 0.1, 1.0e6
_MVT_NU_EXT_FACTOR = 4.0


def _mvt_score_terms(r_sq: np.ndarray, nu: float, d: int):
    """(g, h) with psi_i = g + h_i, the per-observation profile score for nu."""
    half_nu = nu / 2.0
    half_nu_d = (nu + d) / 2.0
    g = (digamma(half_nu) - np.log(half_nu)
         - digamma(half_nu_d) + np.log(half_nu_d))
    h = np.log((nu + r_sq) / (nu + d)) - (nu + d) / (nu + r_sq) + 1.0
    return float(g), h


def _mvt_score_weighted(r_sq: np.ndarray, nu: float, w: np.ndarray, d: int) -> float:
    g, h = _mvt_score_terms(r_sq, nu, d)
    return g + float(np.dot(w, h)) / float(w.sum())


def _mvt_nu_prepare(X: np.ndarray) -> dict:
    return dict(r_sq=np.sum(np.asarray(X, dtype=float) ** 2, axis=1))


@estimator(outputs=('nu',), eta=1e-12, name='nu')
def mvt_nu(X: np.ndarray, w: np.ndarray, prep: dict = None) -> np.ndarray:
    """Root-find of the weighted profile score in log(nu).

    Brackets [_MVT_NU_LO, _MVT_NU_HI] in log(nu), extending geometrically
    to [_MVT_NU_MIN, _MVT_NU_MAX] when the nominal bracket contains no
    root; a one-signed score over the whole admissible range returns the
    matching cap, unless that cap rests on the box (`_at_bound`), in
    which case the evaluation is a counted failure like any other rather
    than a silent 1e6.
    """
    try:
        d = X.shape[1]
        r_sq = prep['r_sq'] if prep is not None else _mvt_nu_prepare(X)['r_sq']
        eta = mvt_nu.eta

        def score_log(log_nu):
            return _mvt_score_weighted(r_sq, float(np.exp(log_nu)), w, d)

        log_lo, log_hi = np.log(_MVT_NU_LO), np.log(_MVT_NU_HI)
        log_nu_min, log_nu_max = np.log(_MVT_NU_MIN), np.log(_MVT_NU_MAX)
        log_ext = np.log(_MVT_NU_EXT_FACTOR)

        s_lo, s_hi = score_log(log_lo), score_log(log_hi)
        while s_lo * s_hi > 0.0 and log_hi < log_nu_max:
            log_hi = min(log_hi + log_ext, log_nu_max)
            s_hi = score_log(log_hi)
        while s_lo * s_hi > 0.0 and log_lo > log_nu_min:
            log_lo = max(log_lo - log_ext, log_nu_min)
            s_lo = score_log(log_lo)

        if s_lo * s_hi > 0.0:
            log_nu_hat = log_nu_max if s_hi < 0.0 else log_nu_min
        else:
            log_nu_hat = optimize.brentq(score_log, log_lo, log_hi, xtol=eta)
        nu_hat = float(np.exp(log_nu_hat))

        if not np.isfinite(nu_hat):
            return np.full(1, np.nan)
        if _at_bound(log_nu_hat, log_nu_min, log_nu_max):
            return np.full(1, np.nan)
        return np.array([nu_hat])
    except Exception:
        return np.full(1, np.nan)


def _mvt_nu_influence(X: np.ndarray, w: np.ndarray) -> np.ndarray:
    """psi_i = g + h_i at nu_hat; A = -d/dnu E_w[psi] by central difference."""
    theta = mvt_nu(X, w)
    if not np.all(np.isfinite(theta)):
        return np.full((len(X), 1), np.nan)
    nu_hat = theta[0]
    d = X.shape[1]
    r_sq = np.sum(np.asarray(X, dtype=float) ** 2, axis=1)
    g, h = _mvt_score_terms(r_sq, nu_hat, d)
    psi = g + h
    delta = 1e-4 * nu_hat
    s_hi = _mvt_score_weighted(r_sq, nu_hat + delta, w, d)
    s_lo = _mvt_score_weighted(r_sq, nu_hat - delta, w, d)
    A = -(s_hi - s_lo) / (2.0 * delta)
    if not np.isfinite(A) or abs(A) < 1e-300:
        return np.full((len(X), 1), np.nan)
    return (psi / A)[:, None]


mvt_nu.prepare = _mvt_nu_prepare
mvt_nu.influence = _mvt_nu_influence


def _mvt_tail_prepare(X: np.ndarray) -> dict:
    return dict(r=np.sqrt(np.sum(np.asarray(X, dtype=float) ** 2, axis=1)))


@estimator(outputs=('P_tail',), eta=EPS, name='tail')
def mvt_tail(X: np.ndarray, w: np.ndarray, prep: dict = None) -> np.ndarray:
    """T = sum(w 1{||x|| > c}) / sum(w), c fixed at the true 99th percentile."""
    try:
        r = prep['r'] if prep is not None else _mvt_tail_prepare(X)['r']
        indicator = (r > MVT_TAIL_C).astype(float)
        theta = float(np.dot(w, indicator) / w.sum())
        return np.array([theta])
    except Exception:
        return np.full(1, np.nan)


def _mvt_tail_influence(X: np.ndarray, w: np.ndarray) -> np.ndarray:
    theta = mvt_tail(X, w)
    if not np.all(np.isfinite(theta)):
        return np.full((len(X), 1), np.nan)
    r = np.sqrt(np.sum(np.asarray(X, dtype=float) ** 2, axis=1))
    indicator = (r > MVT_TAIL_C).astype(float)
    return (indicator - theta[0])[:, None]


mvt_tail.prepare = _mvt_tail_prepare
mvt_tail.influence = _mvt_tail_influence


# ======================================================================
# Fundamental Plane: weighted total least squares
# ======================================================================

# Population standardization, derived once from the cached FP pool
# (`datasets._fp_pool`), never a second read of `fp_sdss.npz`.
_fp_pool_data = _datasets._fp_pool()
FP_POP_MEAN = _fp_pool_data.mean(axis=0)
FP_POP_STD = _fp_pool_data.std(axis=0, ddof=0)
del _fp_pool_data


def _fp_standardize(X: np.ndarray) -> np.ndarray:
    return (np.asarray(X, dtype=float) - FP_POP_MEAN) / FP_POP_STD


def _fp_tls_fit(X_std: np.ndarray, w: np.ndarray) -> dict:
    """Weighted TLS plane via one eigendecomposition of the weighted
    covariance. w normalized to sum 1."""
    mu = X_std.T @ w
    centered = X_std - mu
    S = (centered * w[:, None]).T @ centered
    eigvals_asc, eigvecs_asc = np.linalg.eigh(S)
    idx = np.argsort(eigvals_asc)[::-1]
    eigvals = eigvals_asc[idx]
    eigvecs = eigvecs_asc[:, idx]
    if eigvecs[2, 2] < 0:
        eigvecs[:, 2] *= -1
    v3 = eigvecs[:, 2]
    n_sig, n_I, n_R = float(v3[0]), float(v3[1]), float(v3[2])
    d = float(v3 @ mu)
    a = -n_sig / n_R
    b = -n_I / n_R
    c = d / n_R
    scatter = float(np.sqrt(max(eigvals[2], 0.0)))
    return dict(a=a, b=b, c=c, scatter=scatter, mu=mu, eigvals=eigvals,
                eigvecs=eigvecs, v3=v3, d=d, lambda3=float(eigvals[2]))


def _fp_prepare(X: np.ndarray) -> dict:
    return dict(X_std=_fp_standardize(X))


@estimator(outputs=('a', 'b', 'c', 'scatter'), eta=1e-12, name='fp')
def fp(X: np.ndarray, w: np.ndarray, prep: dict = None) -> np.ndarray:
    try:
        X_std = prep['X_std'] if prep is not None else _fp_prepare(X)['X_std']
        wn = w / w.sum()
        fit = _fp_tls_fit(X_std, wn)
        return np.array([fit['a'], fit['b'], fit['c'], fit['scatter']])
    except Exception:
        return np.full(4, np.nan)


def _fp_compute_if(X_std: np.ndarray, fit: dict) -> np.ndarray:
    """Closed-form implicit-function influence (N, 4)."""
    n = len(X_std)
    mu, eigvals, eigvecs, v3 = fit['mu'], fit['eigvals'], fit['eigvecs'], fit['v3']
    d, lambda3, scatter = fit['d'], fit['lambda3'], fit['scatter']
    n_sig, n_I, n_R = float(v3[0]), float(v3[1]), float(v3[2])

    centered = X_std - mu
    r = centered @ eigvecs

    IF_v3 = np.zeros((n, 3))
    for m in range(2):
        denom = eigvals[2] - eigvals[m]
        coeff = r[:, m] * r[:, 2] / denom
        IF_v3 += coeff[:, None] * eigvecs[:, m]

    IF_d = r[:, 2] + IF_v3 @ mu
    IF_lam3 = r[:, 2] ** 2 - lambda3

    IF_a = -(1.0 / n_R) * IF_v3[:, 0] + (n_sig / n_R ** 2) * IF_v3[:, 2]
    IF_b = -(1.0 / n_R) * IF_v3[:, 1] + (n_I / n_R ** 2) * IF_v3[:, 2]
    IF_c = (1.0 / n_R) * IF_d - (d / n_R ** 2) * IF_v3[:, 2]
    # scatter's influence divides by scatter itself; guard the degenerate
    # zero-scatter fit (not reached by any real draw) rather than raise.
    IF_s = IF_lam3 / (2.0 * scatter) if scatter > 1e-12 else np.zeros(n)

    return np.column_stack([IF_a, IF_b, IF_c, IF_s])


def _fp_influence(X: np.ndarray, w: np.ndarray) -> np.ndarray:
    try:
        X_std = _fp_standardize(X)
        wn = w / w.sum()
        fit = _fp_tls_fit(X_std, wn)
        return _fp_compute_if(X_std, fit)
    except Exception:
        return np.full((len(X), 4), np.nan)


fp.prepare = _fp_prepare
fp.influence = _fp_influence


# ======================================================================
# Chabrier IMF: lognormal below a fixed break m_b, C0-joined to a power
# law above it (Chabrier 2003).
# ======================================================================

_NLL_PENALTY = 1e10
_SQRT2PI = sqrt(2.0 * pi)


def _chabrier_norm(a: float, sigma: float, slope: float, m_b: float,
                    m_min: float) -> tuple:
    """Z(theta) and its first/second partials in (a=log10(m_c), sigma,
    slope), for the Chabrier density normalized on [m_min, m_b] union
    (m_b, inf): a lognormal piece below m_b (an erf, via u=log10(m)) and
    an elementary power-law tail above it, C0-joined at m_b. Returns
    (Z, Za, Zs, Zx, Zaa, Zas, Zax, Zss, Zsx, Zxx).
    """
    ln10 = log(10.0)
    u_b = log(m_b) / ln10
    u_min = log(m_min) / ln10
    btilde = a - u_b
    am = a - u_min

    z_b = -btilde / sigma
    z_m = -am / sigma
    phi_b = exp(-0.5 * z_b * z_b) / _SQRT2PI
    phi_m = exp(-0.5 * z_m * z_m) / _SQRT2PI
    Phi_b = float(ndtr(z_b))
    Phi_m = float(ndtr(z_m))
    D = Phi_b - Phi_m
    E = btilde * phi_b - am * phi_m

    I_lo = ln10 * _SQRT2PI * sigma * D
    Ia = ln10 * _SQRT2PI * (phi_m - phi_b)
    Is = ln10 * _SQRT2PI * (D + E / sigma)
    Iaa = ln10 * _SQRT2PI * (z_m * phi_m - z_b * phi_b) / sigma
    Ias = ln10 * _SQRT2PI * (z_b * phi_b * btilde - z_m * phi_m * am) / sigma ** 2
    Iss = ln10 * _SQRT2PI * (-btilde ** 2 * z_b * phi_b + am ** 2 * z_m * phi_m) / sigma ** 3

    rho = 0.5 * btilde * btilde / (sigma * sigma)
    c2 = exp(-rho)
    T2 = c2 / slope
    T2a = -c2 * btilde / (sigma ** 2 * slope)
    T2s = c2 * btilde ** 2 / (sigma ** 3 * slope)
    T2x = -c2 / slope ** 2
    T2aa = c2 * (btilde ** 2 - sigma ** 2) / (sigma ** 4 * slope)
    T2as = c2 * btilde * (2.0 * sigma ** 2 - btilde ** 2) / (sigma ** 5 * slope)
    T2ss = c2 * btilde ** 2 * (btilde ** 2 - 3.0 * sigma ** 2) / (sigma ** 6 * slope)
    T2ax = c2 * btilde / (sigma ** 2 * slope ** 2)
    T2sx = -c2 * btilde ** 2 / (sigma ** 3 * slope ** 2)
    T2xx = 2.0 * c2 / slope ** 3

    Z = I_lo + T2
    Za, Zs, Zx = Ia + T2a, Is + T2s, T2x
    Zaa, Zas, Zax = Iaa + T2aa, Ias + T2as, T2ax
    Zss, Zsx, Zxx = Iss + T2ss, T2sx, T2xx
    return Z, Za, Zs, Zx, Zaa, Zas, Zax, Zss, Zsx, Zxx


def _chabrier_score_hessian(masses: np.ndarray, w: np.ndarray, m_b: float,
                             m_min: float, m_c: float, sigma: float,
                             slope: float) -> tuple:
    """(psi (N, 3), A (3, 3), Z) for Chabrier at its fitted (m_c, sigma,
    slope), reported basis. Internally works in a = log10(m_c) (the
    density's natural variable); `psi`/`A` are mapped back to (m_c,
    sigma, slope) by the chain rule for a = a(m_c), the only
    reparameterized coordinate. `A` is the negative Hessian of the
    weighted mean log density.
    """
    masses = np.asarray(masses, dtype=float).reshape(-1)
    w = np.asarray(w, dtype=float)
    ln10 = log(10.0)
    a = log(m_c) / ln10
    u_b = log(m_b) / ln10
    btilde = a - u_b
    W = float(w.sum())

    Z, Za, Zs, Zx, Zaa, Zas, Zax, Zss, Zsx, Zxx = _chabrier_norm(
        a, sigma, slope, m_b, m_min)

    low = masses <= m_b
    u = np.log(masses) / ln10

    psi = np.empty((masses.size, 3))
    psi[low, 0] = (u[low] - a) / sigma ** 2 - Za / Z
    psi[low, 1] = (u[low] - a) ** 2 / sigma ** 3 - Zs / Z
    psi[low, 2] = -Zx / Z
    psi[~low, 0] = -btilde / sigma ** 2 - Za / Z
    psi[~low, 1] = btilde ** 2 / sigma ** 3 - Zs / Z
    psi[~low, 2] = (log(m_b) - np.log(masses[~low])) - Zx / Z

    w_lo, w_hi = w[low], w[~low]
    Shi0 = float(w_hi.sum())
    d_lo = u[low] - a
    Slo1 = float(np.dot(w_lo, d_lo))
    Slo2 = float(np.dot(w_lo, d_lo ** 2))

    Haa_lnZ = Zaa / Z - Za * Za / Z ** 2
    Has_lnZ = Zas / Z - Za * Zs / Z ** 2
    Hax_lnZ = Zax / Z - Za * Zx / Z ** 2
    Hss_lnZ = Zss / Z - Zs * Zs / Z ** 2
    Hsx_lnZ = Zsx / Z - Zs * Zx / Z ** 2
    Hxx_lnZ = Zxx / Z - Zx * Zx / Z ** 2

    A_aa = 1.0 / sigma ** 2 + Haa_lnZ
    A_as = 2.0 * (Slo1 - Shi0 * btilde) / (W * sigma ** 3) + Has_lnZ
    A_ax = Hax_lnZ
    A_ss = 3.0 * (Slo2 + Shi0 * btilde ** 2) / (W * sigma ** 4) + Hss_lnZ
    A_sx = Hsx_lnZ
    A_xx = Hxx_lnZ

    J1 = 1.0 / (m_c * ln10)
    f_a = float(np.dot(w, psi[:, 0])) / W

    A = np.array([
        [J1 ** 2 * A_aa + f_a * J1 / m_c, J1 * A_as, J1 * A_ax],
        [J1 * A_as, A_ss, A_sx],
        [J1 * A_ax, A_sx, A_xx],
    ])
    psi_final = np.column_stack([psi[:, 0] * J1, psi[:, 1], psi[:, 2]])
    return psi_final, A, Z


class Chabrier:
    """Chabrier (2003)-form IMF estimator: lognormal below a fixed break
    m_b, C0-joined to a power law above it. Free parameters (m_c, sigma,
    x): the lognormal's peak mass and log10-width below m_b, and the
    power-law slope above it (density m^-(x+1)). `m_b` and `m_min`
    (the population's own minimum mass) are fixed at construction, never
    a draw's own values.
    """

    name = 'chabrier'
    outputs = ('m_c', 'sigma', 'x')
    eta = 1e-6   # sqrt of the objective (NLL) tolerance

    def __init__(self, m_min: float, m_b: float, bounds):
        self.m_b = float(m_b)
        self.m_min = float(m_min)
        self.bounds = tuple((float(lo), float(hi)) for lo, hi in bounds)

    def _at_box(self, params: np.ndarray) -> bool:
        return any(_at_bound(v, lo, hi) for v, (lo, hi) in zip(params, self.bounds))

    def prepare(self, X: np.ndarray) -> dict:
        masses = np.asarray(X, dtype=float).reshape(-1)
        low_mask = masses <= self.m_b
        ln10 = log(10.0)
        u = np.log(masses) / ln10
        return dict(masses=masses, low_mask=low_mask,
                    n_lo=int(low_mask.sum()), n_hi=int((~low_mask).sum()),
                    u=u, m_lo=masses[low_mask], u_lo=u[low_mask],
                    m_hi=masses[~low_mask])

    def _moment_start(self, u_lo: np.ndarray, w_lo: np.ndarray,
                       m_hi: np.ndarray, w_hi: np.ndarray) -> np.ndarray:
        """Method-of-moments start: (m_c0, sigma0) from the weighted
        mean/std of log10(mass) below m_b; slope0 from the weighted Hill
        estimator above m_b (an exact, uncut power law there, so no
        inversion is needed)."""
        Wlo = float(w_lo.sum())
        a0 = float(np.dot(w_lo, u_lo) / Wlo)
        var0 = float(np.dot(w_lo, (u_lo - a0) ** 2) / Wlo)
        sigma0 = sqrt(max(var0, 1e-4))
        m_c0 = 10.0 ** a0

        Whi = float(w_hi.sum())
        log_ratio = np.log(m_hi / self.m_b)
        denom = float(np.dot(w_hi, log_ratio))
        slope0 = Whi / denom if denom > 0 else 1.35

        return np.array([m_c0, sigma0, slope0])

    def __call__(self, X: np.ndarray, w: np.ndarray, prep: dict = None) -> np.ndarray:
        try:
            w = np.asarray(w, dtype=float)
            if prep is None:
                prep = self.prepare(X)
            n_lo, n_hi = prep['n_lo'], prep['n_hi']
            if n_lo < 3 or n_hi < 3:
                return np.full(3, np.nan)
            low_mask = prep['low_mask']
            m_lo, u_lo, m_hi = prep['m_lo'], prep['u_lo'], prep['m_hi']
            m_b, m_min = self.m_b, self.m_min
            w_lo, w_hi = w[low_mask], w[~low_mask]

            x0 = self._moment_start(u_lo, w_lo, m_hi, w_hi)
            bounds_lo = np.array([b[0] for b in self.bounds])
            bounds_hi = np.array([b[1] for b in self.bounds])
            x0 = np.clip(x0, bounds_lo, bounds_hi)
            box_centre = 0.5 * (bounds_lo + bounds_hi)

            def nll(params):
                m_c, sigma, slope = params
                if m_c <= 0 or sigma <= 0 or slope <= 0:
                    return _NLL_PENALTY, np.asarray(params, dtype=float) - box_centre
                try:
                    psi, A, Z = _chabrier_score_hessian(
                        prep['masses'], w, m_b, m_min, m_c, sigma, slope)
                except Exception:
                    return _NLL_PENALTY, np.asarray(params, dtype=float) - box_centre
                if not (np.isfinite(Z) and Z > 0):
                    return _NLL_PENALTY, np.asarray(params, dtype=float) - box_centre
                a = log(m_c) / log(10.0)
                btilde = a - log(m_b) / log(10.0)
                logZ = log(Z)
                ell = np.empty(prep['masses'].size)
                ell[low_mask] = -np.log(m_lo) - (u_lo - a) ** 2 / (2.0 * sigma ** 2)
                ell[~low_mask] = (slope * log(m_b) - btilde ** 2 / (2.0 * sigma ** 2)
                                   - (slope + 1.0) * np.log(m_hi))
                value = -float(np.dot(w, ell - logZ))
                grad = -(w[:, None] * psi).sum(axis=0)
                if not np.isfinite(value) or not np.all(np.isfinite(grad)):
                    return _NLL_PENALTY, np.asarray(params, dtype=float) - box_centre
                return value, grad

            def hess(params):
                m_c, sigma, slope = params
                if m_c <= 0 or sigma <= 0 or slope <= 0:
                    return np.eye(3)
                try:
                    _, A, Z = _chabrier_score_hessian(
                        prep['masses'], w, m_b, m_min, m_c, sigma, slope)
                except Exception:
                    return np.eye(3)
                if not (np.isfinite(Z) and Z > 0):
                    return np.eye(3)
                H = float(np.sum(w)) * A
                return H if np.all(np.isfinite(H)) else np.eye(3)

            res = optimize.minimize(
                nll, x0, method='trust-constr', jac=True, hess=hess,
                bounds=optimize.Bounds(bounds_lo, bounds_hi),
                options={'maxiter': 400, 'gtol': 1e-8, 'xtol': self.eta ** 2},
            )
            if not res.success:
                return np.full(3, np.nan)
            if self._at_box(res.x):
                return np.full(3, np.nan)
            return np.array([float(v) for v in res.x])
        except Exception:
            return np.full(3, np.nan)

    def influence(self, X: np.ndarray, w: np.ndarray) -> np.ndarray:
        """`IF = A^-1 psi` at the fitted parameters, from
        `_chabrier_score_hessian`. NaN whenever the fit fails, the fit
        rests on its box (a non-stationary evaluation, meaningless
        influence), or the solve against A fails.
        """
        try:
            theta = self(X, w)
            if not np.all(np.isfinite(theta)) or self._at_box(theta):
                return np.full((len(X), 3), np.nan)
            masses = np.asarray(X, dtype=float).reshape(-1)
            m_c, sigma, slope = (float(v) for v in theta)
            psi, A, _ = _chabrier_score_hessian(
                masses, np.asarray(w, dtype=float), self.m_b, self.m_min,
                m_c, sigma, slope)
            IF = np.linalg.solve(A, psi.T).T
            if not np.all(np.isfinite(IF)):
                return np.full((len(X), 3), np.nan)
            return IF
        except Exception:
            return np.full((len(X), 3), np.nan)
