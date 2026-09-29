"""`P2Mixture`/`P1Mixture`: the `cloudfil_G_B6_P3_v1` demo's estimands
(spec/QIJ_mods_waves.md A11). Both are a thin wrapper around
`GMM2D(K=10)` whose `outputs` are the 59 raw mixture parameters followed
by seven derived outputs of one subject core -- position (x, y), log
effective radius, log axis ratio, position angle, logit weight, log peak
contrast (A11's table) -- with their influence by the chain rule
J_g @ psi_theta through GMM2D's own analytic influence, J_g the 7x59
Jacobian of the closed forms below by central finite differences.

Each subject shares its mean exactly with another packaged component
(`data/cloudfil_G_B6_P3_v1.npz`) -- P2 with a filament bead, P1 with the
broad cloud -- so no positional or weight rule can tell them apart; at
every evaluation, the subject's fitted component is identified as the
one nearest its truth by the Bhattacharyya distance of A6, using the
subject's own (mean, covariance) read from the packaged `.npz` by its
`role` key. Position angle wraps to a half-open branch of width pi whose
lower edge `pa_lo` keeps the subject's true angle away from the
0/pi wrap: p2 = 0 (near-axis true angle), p1 = -pi/2 (true angle near
mid-branch at 20 deg).
"""
import pathlib

import numpy as np

from .gmm import GMM2D

_DATA_DIR = pathlib.Path(__file__).parent / 'data'
_DATASET_FILE = _DATA_DIR / 'cloudfil_G_B6_P3_v1.npz'

_K = 10
_DERIVED_SUFFIXES = ('x', 'y', 'log_reff', 'log_axis_ratio',
                     'pa', 'logit_w', 'log_contrast')

# Central-difference step for J_g: relative to |theta_c| (absolute floor
# 1.0), so sqrt(eps)-scale truncation error stays far below the closed
# forms' own round-off at every parameter's scale (mixing weights ~1e-3,
# means and covariances ~1).
_FD_STEP = 1e-6


def _load_roles_means_covs() -> tuple:
    """(roles (10,) list of str, means (10,2), covs (10,2,2)) of the
    packaged `.npz`'s own components, read once."""
    with np.load(_DATASET_FILE, allow_pickle=True) as data:
        roles = [str(r) for r in data['role']]
        return roles, data['means'].astype(float), data['covs'].astype(float)


_ROLES, _MEANS_TRUE, _COVS_TRUE = _load_roles_means_covs()


def _true_subject(role: str) -> tuple:
    """(mu (2,), Sigma (2,2)) of the component with this `role` key --
    never a hardcoded row index, since a subject and another component
    can share the same mean."""
    k = _ROLES.index(role)
    return _MEANS_TRUE[k], _COVS_TRUE[k]


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


def _select_subject(theta: np.ndarray, K: int, mu_true: np.ndarray,
                     Sigma_true: np.ndarray) -> int:
    """The fitted component nearest the subject's truth (`mu_true`,
    `Sigma_true`) by the Bhattacharyya distance of A6
    (spec/QIJ_mods_waves.md A6): the metric that separates a spike from
    the blob sharing its mean through the covariance term."""
    _, mus, Sigmas = _unpack_all(theta, K)
    Sbar = 0.5 * (Sigmas + Sigma_true)
    det_a = _dets2x2(Sigmas)
    det_b = float(np.linalg.det(Sigma_true))
    det_bar = _dets2x2(Sbar)
    diff = mus - mu_true
    inv00, inv01, inv11 = Sbar[:, 1, 1] / det_bar, -Sbar[:, 0, 1] / det_bar, Sbar[:, 0, 0] / det_bar
    quad = diff[:, 0] ** 2 * inv00 + 2.0 * diff[:, 0] * diff[:, 1] * inv01 + diff[:, 1] ** 2 * inv11
    D = 0.125 * quad + 0.5 * np.log(det_bar / np.sqrt(det_a * det_b))
    return int(np.argmin(D))


def _derived_outputs(theta: np.ndarray, K: int, s: int, pa_lo: float) -> np.ndarray:
    """The seven scalars of A11's table at fixed component `s`: x, y,
    log effective radius, log axis ratio, position angle, logit weight,
    log peak contrast. `pa` wraps to the half-open branch [pa_lo, pa_lo +
    pi). The peak contrast's density sum uses the unnormalized Gaussian
    shape exp(-quad/2)/sqrt(det); the (2 pi)^-1 every phi carries is a
    common factor of the whole sum, so it cancels against the same
    factor in the numerator (A11's table, "+const")."""
    pis, mus, Sigmas = _unpack_all(theta, K)
    mu_s, Sigma_s = mus[s], Sigmas[s]

    lam2, lam1 = np.linalg.eigvalsh(Sigma_s)  # ascending: lam1 >= lam2
    det_s = lam1 * lam2
    log_reff = 0.25 * np.log(det_s)
    log_axis_ratio = np.log(lam2 / lam1)
    angle = 0.5 * np.arctan2(2.0 * Sigma_s[0, 1], Sigma_s[0, 0] - Sigma_s[1, 1])
    pa = (angle - pa_lo) % np.pi + pa_lo
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


def _jacobian(theta: np.ndarray, K: int, s: int, fd_step: float, pa_lo: float) -> np.ndarray:
    """(7, p) Jacobian of `_derived_outputs` at `theta`, `s` and `pa_lo`
    held fixed -- central finite differences of the deterministic closed
    forms (spec/QIJ_mods_waves.md A11)."""
    p = theta.shape[0]
    J = np.empty((len(_DERIVED_SUFFIXES), p))
    for c in range(p):
        h = fd_step * max(1.0, abs(theta[c]))
        theta_p, theta_m = theta.copy(), theta.copy()
        theta_p[c] += h
        theta_m[c] -= h
        J[:, c] = (_derived_outputs(theta_p, K, s, pa_lo) - _derived_outputs(theta_m, K, s, pa_lo)) / (2.0 * h)
    return J


class _SubjectMixture:
    """See module docstring. `T(X, w, start=None, eta=None) ->
    ndarray(66,)`, `T.influence(X, w, start=None, eta=None) -> ndarray(N,
    66)`; both accept an optional `prep` from `T.prepare(X)` and forward
    every other keyword to the wrapped `GMM2D`. `start` is in `T`'s own
    66-long output layout; only its first 59 entries (GMM2D's own
    layout) are used. `eta`, when given, replaces the wrapped `GMM2D`'s
    own declared eta in the polish's acceptance for that call alone
    (spec/QIJ_mods_waves.md A15). `measured` defaults to the seven
    derived outputs' indices.

    A subclass sets the class attributes `_role` (the subject's `role`
    key in the packaged `.npz`), `_prefix` (its output-name prefix) and
    `_pa_lo` (its position-angle branch's lower edge)."""

    takes_start = True
    _role: str = None
    _prefix: str = None
    _pa_lo: float = None

    def __init__(self, reference=None, measured=None, eta: float = None,
                 cond_max: float = None, fd_step: float = _FD_STEP,
                 smem_breadth: int = None, split_offset: float = None):
        kw = {k: v for k, v in (('eta', eta), ('cond_max', cond_max),
                                 ('smem_breadth', smem_breadth), ('split_offset', split_offset))
              if v is not None}
        self._gmm = GMM2D(K=_K, reference=reference, **kw)
        self.fd_step = float(fd_step)
        self.eta = self._gmm.eta
        self._mu_true, self._sigma_true = _true_subject(self._role)
        derived_names = tuple(f'{self._prefix}_{suffix}' for suffix in _DERIVED_SUFFIXES)
        self.outputs = self._gmm.outputs + derived_names
        n_raw = self._gmm.p
        self.measured = (list(measured) if measured is not None
                          else list(range(n_raw, n_raw + len(derived_names))))

    @property
    def last_fit_info(self):
        """The wrapped `GMM2D`'s own `last_fit_info`
        (spec/QIJ_estimator_fit_spec.md 2.4): component selection and the
        derived outputs' Jacobian add no fit of their own, so this
        status is exactly the raw mixture's."""
        return self._gmm.last_fit_info

    def prepare(self, X: np.ndarray) -> tuple:
        return self._gmm.prepare(X)

    def loglik(self, X: np.ndarray, w: np.ndarray, theta: np.ndarray,
               prep: tuple = None) -> float:
        """`GMM2D.loglik` on the raw mixture parameters (the derived
        outputs have no likelihood term)."""
        return self._gmm.loglik(X, w, np.asarray(theta, dtype=float)[:self._gmm.p], prep=prep)

    def __call__(self, X: np.ndarray, w: np.ndarray, prep: tuple = None,
                 start: np.ndarray = None, eta: float = None, **kwargs) -> np.ndarray:
        raw_start = None if start is None else np.asarray(start, dtype=float)[:self._gmm.p]
        theta = self._gmm(X, w, prep=prep, start=raw_start, eta=eta, **kwargs)
        n_derived = len(_DERIVED_SUFFIXES)
        if not np.all(np.isfinite(theta)):
            return np.concatenate([theta, np.full(n_derived, np.nan)])
        s = _select_subject(theta, self._gmm.K, self._mu_true, self._sigma_true)
        return np.concatenate([theta, _derived_outputs(theta, self._gmm.K, s, self._pa_lo)])

    def influence(self, X: np.ndarray, w: np.ndarray, prep: tuple = None,
                  start: np.ndarray = None, eta: float = None, **kwargs) -> np.ndarray:
        raw_start = None if start is None else np.asarray(start, dtype=float)[:self._gmm.p]
        theta, IF_raw = self._gmm.fit_and_influence(X, w, prep=prep, start=raw_start,
                                                     eta=eta, **kwargs)
        n_derived = len(_DERIVED_SUFFIXES)
        if not np.all(np.isfinite(theta)) or not np.all(np.isfinite(IF_raw)):
            return np.full((len(X), self._gmm.p + n_derived), np.nan)
        K = self._gmm.K
        s = _select_subject(theta, K, self._mu_true, self._sigma_true)
        J = _jacobian(theta, K, s, self.fd_step, self._pa_lo)
        IF_derived = IF_raw @ J.T
        return np.concatenate([IF_raw, IF_derived], axis=1)


class P2Mixture(_SubjectMixture):
    """The subject core is P2 (role 'P2 on-filament'), sharing its mean
    with a filament bead; `pa` wraps to [0, pi)."""

    name = 'p2mixture'
    _role = 'P2 on-filament'
    _prefix = 'p2'
    _pa_lo = 0.0


class P1Mixture(_SubjectMixture):
    """The subject core is P1 (role 'P1 embedded'), sharing its mean
    with the broad cloud; `pa` wraps to [-pi/2, pi/2), keeping P1's true
    angle (20 deg) mid-branch instead of beside the 0/pi wrap."""

    name = 'p1mixture'
    _role = 'P1 embedded'
    _prefix = 'p1'
    _pa_lo = -np.pi / 2
