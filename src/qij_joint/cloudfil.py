"""`P2Mixture`: the `cloudfil_G_B6_P3_v1` demo's estimand (spec/QIJ_mods_waves.md
A11). A thin wrapper around `GMM2D(K=10)` whose `outputs` are the 59 raw
mixture parameters followed by seven derived outputs of the on-filament
protostar P2 -- position (x, y), log effective radius, log axis ratio,
position angle, logit weight, log peak contrast (A11's table) -- with
their influence by the chain rule J_g @ psi_theta through GMM2D's own
analytic influence, J_g the 7x59 Jacobian of the closed forms below by
central finite differences.

P2 shares its mean exactly with a filament bead (`data/cloudfil_G_B6_P3_v1.txt`),
so no positional or weight rule can tell the two apart; at every
evaluation, P2's fitted component is identified as the one nearest the
truth's P2 by the Bhattacharyya distance of A6, using P2's own (mean,
covariance) read from the packaged `.npz` by its `role` key.
"""
import pathlib

import numpy as np

from .gmm import GMM2D

_DATA_DIR = pathlib.Path(__file__).parent / 'data'
_DATASET_FILE = _DATA_DIR / 'cloudfil_G_B6_P3_v1.npz'
_P2_ROLE = 'P2 on-filament'

_K = 10
_DERIVED_NAMES = ('p2_x', 'p2_y', 'p2_log_reff', 'p2_log_axis_ratio',
                  'p2_pa', 'p2_logit_w', 'p2_log_contrast')

# Central-difference step for J_g: relative to |theta_c| (absolute floor
# 1.0), so sqrt(eps)-scale truncation error stays far below the closed
# forms' own round-off at every parameter's scale (mixing weights ~1e-3,
# means and covariances ~1).
_FD_STEP = 1e-6


def _true_p2() -> tuple:
    """(mu (2,), Sigma (2,2)) of the P2 component, located by its `role`
    key in the packaged `.npz` -- never a hardcoded row index, since P2
    and a filament bead share the same mean."""
    with np.load(_DATASET_FILE, allow_pickle=True) as data:
        roles = [str(r) for r in data['role']]
        k = roles.index(_P2_ROLE)
        return data['means'][k].astype(float), data['covs'][k].astype(float)


_P2_MU_TRUE, _P2_SIGMA_TRUE = _true_p2()


def _unpack_all(theta: np.ndarray, K: int) -> tuple:
    """theta (p,) -> pis (K,), mus (K,2), Sigmas (K,2,2), GMM2D's own
    packed layout: (K-1) mixing weights, then per-component (mu_x, mu_y),
    then per-component (S11, S12, S22)."""
    pis = np.empty(K)
    pis[:K - 1] = theta[:K - 1]
    pis[K - 1] = 1.0 - pis[:K - 1].sum()
    base_mu = K - 1
    mus = theta[base_mu:base_mu + 2 * K].reshape(K, 2)
    base_S = base_mu + 2 * K
    S = theta[base_S:base_S + 3 * K].reshape(K, 3)
    Sigmas = np.empty((K, 2, 2))
    Sigmas[:, 0, 0], Sigmas[:, 0, 1], Sigmas[:, 1, 1] = S[:, 0], S[:, 1], S[:, 2]
    Sigmas[:, 1, 0] = Sigmas[:, 0, 1]
    return pis, mus, Sigmas


def _dets2x2(Sigmas: np.ndarray) -> np.ndarray:
    return Sigmas[..., 0, 0] * Sigmas[..., 1, 1] - Sigmas[..., 0, 1] ** 2


def _select_p2(theta: np.ndarray, K: int) -> int:
    """The fitted component nearest the truth's P2 by the Bhattacharyya
    distance of A6 (spec/QIJ_mods_waves.md A6): the metric that separates
    a spike from the blob sharing its mean through the covariance term."""
    _, mus, Sigmas = _unpack_all(theta, K)
    Sbar = 0.5 * (Sigmas + _P2_SIGMA_TRUE)
    det_a = _dets2x2(Sigmas)
    det_b = float(np.linalg.det(_P2_SIGMA_TRUE))
    det_bar = _dets2x2(Sbar)
    diff = mus - _P2_MU_TRUE
    inv00, inv01, inv11 = Sbar[:, 1, 1] / det_bar, -Sbar[:, 0, 1] / det_bar, Sbar[:, 0, 0] / det_bar
    quad = diff[:, 0] ** 2 * inv00 + 2.0 * diff[:, 0] * diff[:, 1] * inv01 + diff[:, 1] ** 2 * inv11
    D = 0.125 * quad + 0.5 * np.log(det_bar / np.sqrt(det_a * det_b))
    return int(np.argmin(D))


def _derived_outputs(theta: np.ndarray, K: int, s: int) -> np.ndarray:
    """The seven scalars of A11's table at fixed component `s`: x, y,
    log effective radius, log axis ratio, position angle, logit weight,
    log peak contrast. The peak contrast's density sum uses the
    unnormalized Gaussian shape exp(-quad/2)/sqrt(det); the (2 pi)^-1
    every phi carries is a common factor of the whole sum, so it cancels
    against the same factor in the numerator (A11's table, "+const")."""
    pis, mus, Sigmas = _unpack_all(theta, K)
    mu_s, Sigma_s = mus[s], Sigmas[s]

    lam2, lam1 = np.linalg.eigvalsh(Sigma_s)  # ascending: lam1 >= lam2
    det_s = lam1 * lam2
    log_reff = 0.25 * np.log(det_s)
    log_axis_ratio = np.log(lam2 / lam1)
    pa = (0.5 * np.arctan2(2.0 * Sigma_s[0, 1], Sigma_s[0, 0] - Sigma_s[1, 1])) % np.pi
    logit_w = np.log(pis[s] / (1.0 - pis[s]))

    det_all = _dets2x2(Sigmas)
    diff = mu_s - mus
    inv00, inv01, inv11 = Sigmas[:, 1, 1] / det_all, -Sigmas[:, 0, 1] / det_all, Sigmas[:, 0, 0] / det_all
    quad = diff[:, 0] ** 2 * inv00 + 2.0 * diff[:, 0] * diff[:, 1] * inv01 + diff[:, 1] ** 2 * inv11
    log_shape = -0.5 * quad - 0.5 * np.log(det_all)
    other = np.arange(K) != s
    density_sum = np.sum(pis[other] * np.exp(log_shape[other]))
    log_contrast = np.log(pis[s]) - 0.5 * np.log(det_s) - np.log(density_sum)

    return np.array([mu_s[0], mu_s[1], log_reff, log_axis_ratio, pa, logit_w, log_contrast])


def _jacobian(theta: np.ndarray, K: int, s: int) -> np.ndarray:
    """(7, p) Jacobian of `_derived_outputs` at `theta`, `s` held fixed
    -- central finite differences of the deterministic closed forms
    (spec/QIJ_mods_waves.md A11)."""
    p = theta.shape[0]
    J = np.empty((len(_DERIVED_NAMES), p))
    for c in range(p):
        h = _FD_STEP * max(1.0, abs(theta[c]))
        theta_p, theta_m = theta.copy(), theta.copy()
        theta_p[c] += h
        theta_m[c] -= h
        J[:, c] = (_derived_outputs(theta_p, K, s) - _derived_outputs(theta_m, K, s)) / (2.0 * h)
    return J


class P2Mixture:
    """See module docstring. `T(X, w, start=None, eta=None) ->
    ndarray(66,)`, `T.influence(X, w, start=None, eta=None) -> ndarray(N,
    66)`; both accept an optional `prep` from `T.prepare(X)` and forward
    every other keyword to the wrapped `GMM2D`. `start` is in `T`'s own
    66-long output layout; only its first 59 entries (GMM2D's own
    layout) are used. `eta`, when given, replaces the wrapped `GMM2D`'s
    own declared eta in the polish's acceptance for that call alone
    (spec/QIJ_mods_waves.md A15). `measured` defaults to the seven
    derived outputs' indices."""

    name = 'p2mixture'
    takes_start = True

    def __init__(self, reference=None, measured=None):
        self._gmm = GMM2D(K=_K, reference=reference)
        self.eta = self._gmm.eta
        self.outputs = self._gmm.outputs + _DERIVED_NAMES
        n_raw = self._gmm.p
        self.measured = (list(measured) if measured is not None
                          else list(range(n_raw, n_raw + len(_DERIVED_NAMES))))

    @property
    def last_fit_info(self):
        """The wrapped `GMM2D`'s own `last_fit_info`
        (spec/QIJ_estimator_fit_spec.md 2.4): `_select_p2`/the derived
        outputs' Jacobian add no fit of their own, so P2Mixture's status
        is exactly the raw mixture's."""
        return self._gmm.last_fit_info

    def prepare(self, X: np.ndarray) -> tuple:
        return self._gmm.prepare(X)

    def __call__(self, X: np.ndarray, w: np.ndarray, prep: tuple = None,
                 start: np.ndarray = None, eta: float = None, **kwargs) -> np.ndarray:
        raw_start = None if start is None else np.asarray(start, dtype=float)[:self._gmm.p]
        theta = self._gmm(X, w, prep=prep, start=raw_start, eta=eta, **kwargs)
        n_derived = len(_DERIVED_NAMES)
        if not np.all(np.isfinite(theta)):
            return np.concatenate([theta, np.full(n_derived, np.nan)])
        s = _select_p2(theta, self._gmm.K)
        return np.concatenate([theta, _derived_outputs(theta, self._gmm.K, s)])

    def influence(self, X: np.ndarray, w: np.ndarray, prep: tuple = None,
                  start: np.ndarray = None, eta: float = None, **kwargs) -> np.ndarray:
        raw_start = None if start is None else np.asarray(start, dtype=float)[:self._gmm.p]
        theta, IF_raw = self._gmm.fit_and_influence(X, w, prep=prep, start=raw_start,
                                                     eta=eta, **kwargs)
        n_derived = len(_DERIVED_NAMES)
        if not np.all(np.isfinite(theta)) or not np.all(np.isfinite(IF_raw)):
            return np.full((len(X), self._gmm.p + n_derived), np.nan)
        K = self._gmm.K
        s = _select_p2(theta, K)
        J = _jacobian(theta, K, s)
        IF_derived = IF_raw @ J.T
        return np.concatenate([IF_raw, IF_derived], axis=1)
