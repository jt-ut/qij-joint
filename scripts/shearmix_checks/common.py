"""Shared helpers for the shearmix_model self-checks (Agent B). Builds
the ONE N=10,000, seed=1 test draw described in the build interface
(spec/QIJ_shearmix_interface.md section 3): the cloud and three cores
from `cloudfil_G_B6_P3_v1.npz` by role, plus a sheared-Gaussian filament
drawn by hand (the dataset function itself is Agent A's module, not yet
built), and the matching truth theta. NOT part of the package; a
one-off script, not imported by anything else."""

import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.join(_HERE, '..', '..', 'src')
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from qij_joint import shearmix_model as sm  # noqa: E402

_NPZ = os.path.join(_SRC, 'qij_joint', 'data', 'cloudfil_G_B6_P3_v1.npz')

KG = 4
FIL_TRUE = np.array([0.0, 1.25 ** 2, 0.0, 0.55, 0.0, 0.08 ** 2])
FIL_WEIGHT = 0.38


def _pop_by_role():
    with np.load(_NPZ, allow_pickle=True) as d:
        weights, means, covs, role = d['weights'], d['means'], d['covs'], d['role']
    order = ['cloud', 'P1 embedded', 'P2 on-filament', 'P3 off-filament']
    idx = [int(np.where(role == r)[0][0]) for r in order]
    return (weights[idx].astype(float), means[idx].astype(float), covs[idx].astype(float))


def truth_theta():
    """The packed true theta (Gaussians in file order: cloud, P1, P2,
    P3; filament last)."""
    w_g, means, covs = _pop_by_role()
    return sm.pack(KG, w_g, means, covs, FIL_TRUE), (w_g, means, covs)


def make_draw(N=10000, seed=1):
    """N draws: cloud, P1, P2, P3 (multivariate normal, by role, exact
    file order) then the filament (x ~ N(m, s2x), y = h(x) + sqrt(s2p)*eps),
    one rng, mirroring `datasets.cloudfil_G_B6_P3_v1`'s own multinomial-
    then-per-component draw pattern. Returns (X (N,2), w (N,) all ones)."""
    w_g, means, covs = _pop_by_role()
    weights = np.concatenate([w_g, [FIL_WEIGHT]])
    assert abs(weights.sum() - 1.0) < 1e-12
    rng = np.random.default_rng(seed)
    counts = rng.multinomial(N, weights)
    X = np.empty((N, 2))
    start = 0
    for k in range(KG):
        n_k = counts[k]
        if n_k:
            X[start:start + n_k] = rng.multivariate_normal(means[k], covs[k], size=n_k)
        start += n_k
    n_f = counts[KG]
    if n_f:
        m, s2x, b0, b1, b2, s2p = FIL_TRUE
        x = rng.normal(m, np.sqrt(s2x), size=n_f)
        h = b0 + b1 * np.sin(sm.OMEGA * x) + b2 * np.cos(sm.OMEGA * x)
        y = h + rng.normal(0.0, np.sqrt(s2p), size=n_f)
        X[start:start + n_f, 0] = x
        X[start:start + n_f, 1] = y
    w = np.ones(N)
    return X, w


def random_feasible_theta(rng, base_theta, scale=0.15):
    """A random feasible perturbation of `base_theta`: multiplicative
    (always-positive) jitter on the weights and the two filament
    variances, a Cholesky-factor jitter on every Sigma_k (guarantees
    PD), and small additive jitter elsewhere; resampled until feasible
    (the retry loop is a safety net -- the Cholesky construction almost
    always succeeds on its own)."""
    Kg = KG
    pis_g, pi_f, mus, covs, fil = sm.unpack(Kg, base_theta)
    for _ in range(200):
        pis = pis_g * np.exp(scale * rng.normal(size=Kg))
        if pis.sum() >= 0.9:
            pis *= 0.9 / pis.sum()
        mus2 = np.empty_like(mus)
        covs2 = np.empty_like(covs)
        for k in range(Kg):
            sx_k = np.sqrt(covs[k, 0, 0])
            sy_k = np.sqrt(covs[k, 1, 1])
            mus2[k, 0] = mus[k, 0] + scale * sx_k * rng.normal()
            mus2[k, 1] = mus[k, 1] + scale * sy_k * rng.normal()
            L = np.linalg.cholesky(covs[k])
            dL = np.tril(scale * 0.3 * rng.normal(size=(2, 2)))
            Lp = L + dL
            scale_k = float(np.exp(scale * rng.normal()))
            covs2[k] = scale_k * (Lp @ Lp.T)

        fil2 = fil.copy()
        sx0 = np.sqrt(covs[0, 0, 0])
        sy0 = np.sqrt(covs[0, 1, 1])
        fil2[0] = fil[0] + scale * sx0 * rng.normal()          # m
        fil2[1] = fil[1] * np.exp(scale * rng.normal())        # s2x
        fil2[2] = fil[2] + scale * sy0 * rng.normal()           # b0
        fil2[3] = fil[3] + scale * sy0 * rng.normal()           # b1
        fil2[4] = fil[4] + scale * sy0 * rng.normal()           # b2
        fil2[5] = fil[5] * np.exp(scale * rng.normal())        # s2p

        theta = sm.pack(Kg, pis, mus2, covs2, fil2)
        if sm.feasible(Kg, theta):
            return theta
    raise RuntimeError("could not draw a feasible perturbation")
