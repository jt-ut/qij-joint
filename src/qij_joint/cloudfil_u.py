"""`P2ShearMixture`/`P1ShearMixture`: the `cloudfil_G_U_P3_v1` demo's
estimands (spec/QIJ_shearmix_interface.md section 5) -- `cloudfil.py`'s
`P2Mixture`/`P1Mixture` rebuilt on `shearmix.ShearMix2D` in place of
`gmm.GMM2D`, the author's well-specified replacement for the six-bead
filament demo. Both are a thin wrapper around `ShearMix2D(Kg=4)` whose
`outputs` are the 30 raw mixture parameters (section 2's layout: four
free Gaussians, cloud and three cores, plus the filament block) followed
by seven derived outputs of one subject Gaussian -- position (x, y), log
effective radius, log axis ratio, position angle, logit weight, log peak
contrast -- with their influence by the chain rule J_g @ psi_theta
through `ShearMix2D`'s own analytic influence, J_g the 7x30 Jacobian of
the closed forms below by central finite differences.

Each subject shares its mean exactly with another packaged component
(`data/cloudfil_G_U_P3_v1.npz`): P2 with the filament (it sits ON the
curve, unlike the six-bead demo where P2 shared a mean with one bead),
P1 with the broad cloud -- so no positional or weight rule can tell them
apart; at every evaluation, the subject's fitted Gaussian is identified
as the one (among the Kg=4 free Gaussians only; the filament is never a
subject candidate) nearest its truth by the Bhattacharyya distance of
A6, using the subject's own (mean, covariance) read from the packaged
`.npz` by its `role` key. Position angle wraps to a half-open branch of
width pi whose lower edge `pa_lo` keeps the subject's true angle away
from the 0/pi wrap: p2 = 0 (near-axis true angle), p1 = -pi/2 (true
angle near mid-branch at 20 deg) -- same branch choices as `cloudfil.py`.

The log peak contrast differs from `cloudfil.py`'s own closed form: its
denominator sums the OTHER Gaussians' densities AND the filament's own
density at the subject's mean, so (unlike the six-bead demo, where every
component is a plain Gaussian and the shared (2 pi)^-1 normalizing
factor cancels unused) this module uses fully normalized densities on
both sides of the ratio throughout --

    ln C = ln(pi_s N(mu_s | mu_s, Sigma_s)) - ln(sum_{k != s} f_k(mu_s))

-- `f_k(mu_s)` the other Gaussians' full densities at mu_s plus the
filament's full density `pi_f * f_fil(mu_s)` (shearmix_model.log_joint's
own normalization, `-log(2 pi) - 0.5 log det - 0.5 quad` per Gaussian
block and the matching two-factor form for the filament). The common
(2 pi)^-1 factor is in fact still common to every term (the filament's
density, written as the product of two normalized 1-D normals, is
itself a properly normalized 2-D density under the shear x -> (x, y -
h(x)), whose Jacobian is 1), so the result is numerically identical to
dropping it as `cloudfil.py` does -- using the fully normalized forms
throughout is simply the least error-prone way to add a non-Gaussian
term to the sum.
"""
import pathlib

import numpy as np

from . import shearmix_model
from .shearmix import ShearMix2D

_DATA_DIR = pathlib.Path(__file__).parent / 'data'
_DATASET_FILE = _DATA_DIR / 'cloudfil_G_U_P3_v1.npz'

_KG = 4
_DERIVED_SUFFIXES = ('x', 'y', 'log_reff', 'log_axis_ratio',
                     'pa', 'logit_w', 'log_contrast')
_LOG2PI = float(np.log(2.0 * np.pi))

# Central-difference step for J_g: relative to |theta_c| (absolute floor
# 1.0), exactly `cloudfil.py`'s own choice -- sqrt(eps)-scale truncation
# error stays far below the closed forms' own round-off at every
# parameter's scale.
_FD_STEP = 1e-6


def _load_roles_means_covs() -> tuple:
    """(roles (5,) list of str, means (4,2), covs (4,2,2)) of the
    packaged `.npz`'s own Gaussian components, read once. `roles[-1]`
    ('filament') has no corresponding (mean, cov) row -- only the first
    4 roles name a Gaussian."""
    with np.load(_DATASET_FILE, allow_pickle=True) as data:
        roles = [str(r) for r in data['role']]
        return roles, data['means'].astype(float), data['covs'].astype(float)


_ROLES, _MEANS_TRUE, _COVS_TRUE = _load_roles_means_covs()


def _true_subject(role: str) -> tuple:
    """(mu (2,), Sigma (2,2)) of the Gaussian with this `role` key --
    never a hardcoded row index, since a subject and another component
    can share the same mean."""
    k = _ROLES.index(role)
    return _MEANS_TRUE[k], _COVS_TRUE[k]


def _dets2x2(Sigmas: np.ndarray) -> np.ndarray:
    return Sigmas[..., 0, 0] * Sigmas[..., 1, 1] - Sigmas[..., 0, 1] ** 2


def _select_subject(theta: np.ndarray, Kg: int, mu_true: np.ndarray,
                     Sigma_true: np.ndarray) -> int:
    """The fitted Gaussian (among the Kg free Gaussians only; the
    filament is never a subject candidate) nearest the subject's truth
    (`mu_true`, `Sigma_true`) by the Bhattacharyya distance of A6
    (spec/QIJ_mods_waves.md A6): the metric that separates a spike from
    the blob/filament sharing its mean through the covariance term."""
    _, _, mus, covs, _ = shearmix_model.unpack(Kg, theta)
    Sbar = 0.5 * (covs + Sigma_true)
    det_a = _dets2x2(covs)
    det_b = float(np.linalg.det(Sigma_true))
    det_bar = _dets2x2(Sbar)
    diff = mus - mu_true
    inv00, inv01, inv11 = Sbar[:, 1, 1] / det_bar, -Sbar[:, 0, 1] / det_bar, Sbar[:, 0, 0] / det_bar
    quad = diff[:, 0] ** 2 * inv00 + 2.0 * diff[:, 0] * diff[:, 1] * inv01 + diff[:, 1] ** 2 * inv11
    D = 0.125 * quad + 0.5 * np.log(det_bar / np.sqrt(det_a * det_b))
    return int(np.argmin(D))


def _filament_log_density(fil: np.ndarray, omega: float, x: float, y: float) -> float:
    """log f_fil(x, y) (no pi_f factor), shearmix_model's own
    normalization: log N(x|m,s2x) + log N(y - h(x)|0,s2p)."""
    m, s2x, b0, b1, b2, s2p = fil
    hx = b0 + b1 * np.sin(omega * x) + b2 * np.cos(omega * x)
    ex = x - m
    ey = y - hx
    return float(-_LOG2PI - 0.5 * np.log(s2x) - 0.5 * np.log(s2p)
                 - 0.5 * ex * ex / s2x - 0.5 * ey * ey / s2p)


def _derived_outputs(theta: np.ndarray, Kg: int, s: int, pa_lo: float,
                      omega: float) -> np.ndarray:
    """The seven scalars at fixed free-Gaussian index `s`: x, y, log
    effective radius, log axis ratio, position angle, logit weight, log
    peak contrast. `pa` wraps to the half-open branch [pa_lo, pa_lo +
    pi). The log peak contrast uses fully normalized densities on both
    sides (module docstring): the denominator sums the OTHER Kg-1
    Gaussians' full densities at mu_s plus the filament's full density
    pi_f * f_fil(mu_s), combined by logsumexp for stability."""
    pis_g, pi_f, mus, covs, fil = shearmix_model.unpack(Kg, theta)
    mu_s, Sigma_s = mus[s], covs[s]

    lam2, lam1 = np.linalg.eigvalsh(Sigma_s)  # ascending: lam1 >= lam2
    det_s = lam1 * lam2
    log_reff = 0.25 * np.log(det_s)
    log_axis_ratio = np.log(lam2 / lam1)
    angle = 0.5 * np.arctan2(2.0 * Sigma_s[0, 1], Sigma_s[0, 0] - Sigma_s[1, 1])
    pa = (angle - pa_lo) % np.pi + pa_lo
    logit_w = np.log(pis_g[s] / (1.0 - pis_g[s]))

    log_num = np.log(pis_g[s]) - _LOG2PI - 0.5 * np.log(det_s)

    det_all = _dets2x2(covs)
    diff = mu_s - mus
    inv00, inv01, inv11 = covs[:, 1, 1] / det_all, -covs[:, 0, 1] / det_all, covs[:, 0, 0] / det_all
    quad = diff[:, 0] ** 2 * inv00 + 2.0 * diff[:, 0] * diff[:, 1] * inv01 + diff[:, 1] ** 2 * inv11
    log_gauss_k = np.log(pis_g) - _LOG2PI - 0.5 * np.log(det_all) - 0.5 * quad
    other = np.arange(Kg) != s
    log_fil = np.log(pi_f) + _filament_log_density(fil, omega, mu_s[0], mu_s[1])
    logs_other = np.concatenate([log_gauss_k[other], [log_fil]])
    peak = np.max(logs_other)
    log_density_sum = peak + np.log(np.sum(np.exp(logs_other - peak)))
    log_contrast = log_num - log_density_sum

    return np.array([mu_s[0], mu_s[1], log_reff, log_axis_ratio, pa, logit_w, log_contrast])


def _jacobian(theta: np.ndarray, Kg: int, s: int, fd_step: float, pa_lo: float,
              omega: float) -> np.ndarray:
    """(7, p) Jacobian of `_derived_outputs` at `theta`, `Kg`, `s`,
    `pa_lo` and `omega` held fixed -- central finite differences of the
    deterministic closed forms."""
    p = theta.shape[0]
    J = np.empty((len(_DERIVED_SUFFIXES), p))
    for c in range(p):
        h = fd_step * max(1.0, abs(theta[c]))
        theta_p, theta_m = theta.copy(), theta.copy()
        theta_p[c] += h
        theta_m[c] -= h
        J[:, c] = (_derived_outputs(theta_p, Kg, s, pa_lo, omega)
                   - _derived_outputs(theta_m, Kg, s, pa_lo, omega)) / (2.0 * h)
    return J


class _SubjectShearMixture:
    """See module docstring. `T(X, w, start=None, eta=None) ->
    ndarray(37,)`, `T.influence(X, w, start=None, eta=None) ->
    ndarray(N, 37)`; both accept an optional `prep` from `T.prepare(X)`
    and forward every other keyword to the wrapped `ShearMix2D`. `start`
    is in `T`'s own 37-long output layout; only its first 30 entries
    (`ShearMix2D`'s own layout) are used. `eta`, when given, replaces the
    wrapped `ShearMix2D`'s own declared eta in the polish's acceptance
    for that call alone. `measured` defaults to the seven derived
    outputs' indices.

    A subclass sets the class attributes `_role` (the subject's `role`
    key in the packaged `.npz`), `_prefix` (its output-name prefix) and
    `_pa_lo` (its position-angle branch's lower edge), plus `periodic`
    (its periodic outputs' names and periods, `{'<prefix>_pa': pi}`
    here, read by the sigma-points stage, `core/sigma_points.py`)."""

    takes_start = True
    _role: str = None
    _prefix: str = None
    _pa_lo: float = None

    def __init__(self, reference=None, measured=None, eta: float = None,
                 cond_max: float = None, fd_step: float = _FD_STEP,
                 omega: float = None, n_starts: int = None, seed: int = None):
        kw = {k: v for k, v in (('eta', eta), ('cond_max', cond_max),
                                 ('omega', omega), ('n_starts', n_starts), ('seed', seed))
              if v is not None}
        self._mix = ShearMix2D(Kg=_KG, reference=reference, **kw)
        self.fd_step = float(fd_step)
        self.eta = self._mix.eta
        self._mu_true, self._sigma_true = _true_subject(self._role)
        derived_names = tuple(f'{self._prefix}_{suffix}' for suffix in _DERIVED_SUFFIXES)
        self.outputs = self._mix.outputs + derived_names
        n_raw = self._mix.p
        self.measured = (list(measured) if measured is not None
                          else list(range(n_raw, n_raw + len(derived_names))))

    @property
    def last_fit_info(self):
        """The wrapped `ShearMix2D`'s own `last_fit_info`
        (spec/QIJ_estimator_fit_spec.md 2.4): component selection and the
        derived outputs' Jacobian add no fit of their own, so this
        status is exactly the raw mixture's."""
        return self._mix.last_fit_info

    def prepare(self, X: np.ndarray) -> tuple:
        return self._mix.prepare(X)

    def loglik(self, X: np.ndarray, w: np.ndarray, theta: np.ndarray,
               prep: tuple = None) -> float:
        """`ShearMix2D.loglik` on the raw mixture parameters (the
        derived outputs have no likelihood term)."""
        return self._mix.loglik(X, w, np.asarray(theta, dtype=float)[:self._mix.p], prep=prep)

    def __call__(self, X: np.ndarray, w: np.ndarray, prep: tuple = None,
                 start: np.ndarray = None, eta: float = None, **kwargs) -> np.ndarray:
        raw_start = None if start is None else np.asarray(start, dtype=float)[:self._mix.p]
        theta = self._mix(X, w, prep=prep, start=raw_start, eta=eta, **kwargs)
        n_derived = len(_DERIVED_SUFFIXES)
        if not np.all(np.isfinite(theta)):
            return np.concatenate([theta, np.full(n_derived, np.nan)])
        s = _select_subject(theta, self._mix.Kg, self._mu_true, self._sigma_true)
        return np.concatenate([theta, _derived_outputs(theta, self._mix.Kg, s,
                                                         self._pa_lo, self._mix.omega)])

    def influence(self, X: np.ndarray, w: np.ndarray, prep: tuple = None,
                  start: np.ndarray = None, eta: float = None, **kwargs) -> np.ndarray:
        raw_start = None if start is None else np.asarray(start, dtype=float)[:self._mix.p]
        theta, IF_raw = self._mix.fit_and_influence(X, w, prep=prep, start=raw_start,
                                                      eta=eta, **kwargs)
        n_derived = len(_DERIVED_SUFFIXES)
        if not np.all(np.isfinite(theta)) or not np.all(np.isfinite(IF_raw)):
            return np.full((len(X), self._mix.p + n_derived), np.nan)
        Kg = self._mix.Kg
        s = _select_subject(theta, Kg, self._mu_true, self._sigma_true)
        J = _jacobian(theta, Kg, s, self.fd_step, self._pa_lo, self._mix.omega)
        IF_derived = IF_raw @ J.T
        return np.concatenate([IF_raw, IF_derived], axis=1)


class P2ShearMixture(_SubjectShearMixture):
    """The subject is the free Gaussian nearest P2 (role 'P2
    on-filament'), which shares its mean with the filament; `pa` wraps
    to [0, pi)."""

    name = 'p2shearmix'
    _role = 'P2 on-filament'
    _prefix = 'p2'
    _pa_lo = 0.0
    periodic = {'p2_pa': np.pi}


class P1ShearMixture(_SubjectShearMixture):
    """The subject is the free Gaussian nearest P1 (role 'P1 embedded'),
    which shares its mean with the broad cloud; `pa` wraps to [-pi/2,
    pi/2), keeping P1's true angle (20 deg) mid-branch instead of beside
    the 0/pi wrap."""

    name = 'p1shearmix'
    _role = 'P1 embedded'
    _prefix = 'p1'
    _pa_lo = -np.pi / 2
    periodic = {'p1_pa': np.pi}
