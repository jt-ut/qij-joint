"""
The X-VQ: quantize the data, then measure the influence at each
prototype by forward differences (spec/method_notes.md section 2).
`fit_xvq` fits the codebook (no estimator evaluations); `prototype_
influences` evaluates T once on the prototypes for theta_Q, then one
forward difference per prototype; `run_xvq` is stage 1 in full.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Tuple

import numpy as np
from vqlp import VQFitter

from .differences import forward_step, perturbed_weights, step_parameter

_KAPPA_REF = 2.7
# Row batch for the second-BMU repair: bounds the (rows, M_used)
# distance block at tens of megabytes rather than gigabytes at large N.
_BMU2_BATCH_CAP = 4096


@dataclass
class XVQ:
    centers: np.ndarray      # (M_used, d_z) whitened prototype positions
    labels: np.ndarray       # (N,) receptive-field index per point
    p: np.ndarray            # (M_used,) receptive-field masses, summing to 1
    bmu: np.ndarray          # (N,) best-matching prototype
    bmu2: np.ndarray         # (N,) second best-matching prototype, resolved
    conn: object             # (M_used, M_used) CADJ adjacency, sparse
    M_requested: int
    M_used: int


def cost_rule_M(N: int, q: int, eps: float) -> int:
    """M_X = ceil(sqrt((1 + 2*q*M_ref)*N/2)), M_ref = ceil(sqrt(2.7/eps)),
    floored at 20 and capped at N // 2 (spec/method_notes.md section 2)."""
    M_ref = math.ceil(math.sqrt(_KAPPA_REF / eps))
    M_X = math.ceil(math.sqrt((1.0 + 2.0 * q * M_ref) * N / 2.0))
    return min(max(M_X, 20), N // 2)


def fit_xvq(Z: np.ndarray, M: int, seed: int) -> XVQ:
    """k-means on Z via `vqlp.VQFitter`, empty receptive fields dropped
    (M_used <= M), first/second best-matching units and the CADJ
    adjacency reindexed into the live prototype space. No estimator
    evaluation."""
    Z = np.asarray(Z, dtype=float)
    N = Z.shape[0]

    fitter = VQFitter(M=M, p=2, max_bmu=2, random_state=seed, verbose=False)
    fitter.fit(Z)
    fitter.recall(Z)
    rec = fitter.recaller

    RFSize = rec.RFSize
    live_idx = np.where(RFSize > 0)[0]
    M_used = live_idx.size
    old2new = np.full(RFSize.shape[0], -1, dtype=np.intp)
    old2new[live_idx] = np.arange(M_used)

    BMU = rec.BMU
    bmu = old2new[BMU[:, 0]]
    bmu2 = old2new[BMU[:, 1]]

    p = RFSize[live_idx] / N
    centers = fitter.W[live_idx]
    conn = rec.CADJ.tocsr()[np.ix_(live_idx, live_idx)].tocsr()

    bmu2 = _resolve_bmu2(Z, centers, bmu, bmu2)

    return XVQ(
        centers=centers, labels=bmu, p=p, bmu=bmu, bmu2=bmu2,
        conn=conn, M_requested=M, M_used=M_used,
    )


def _resolve_bmu2(Z: np.ndarray, centers: np.ndarray, bmu: np.ndarray, bmu2: np.ndarray) -> np.ndarray:
    """`bmu2` is -1 wherever the second-BMU pointed at a prototype
    dropped as empty; repair to the nearest LIVE prototype other than
    `bmu`, batched over rows so the (rows, M_used) distance block stays
    bounded even when most points need repair."""
    bmu2 = np.array(bmu2, dtype=int, copy=True)
    missing = bmu2 < 0
    if not missing.any():
        return bmu2

    Zm = Z[missing]
    bmu_m = bmu[missing]
    centers_sq = np.sum(centers ** 2, axis=1)[None, :]
    nearest = np.empty(Zm.shape[0], dtype=int)
    for start in range(0, Zm.shape[0], _BMU2_BATCH_CAP):
        sl = slice(start, start + _BMU2_BATCH_CAP)
        Zb = Zm[sl]
        D2 = np.sum(Zb ** 2, axis=1)[:, None] + centers_sq - 2.0 * Zb @ centers.T
        D2[np.arange(D2.shape[0]), bmu_m[sl]] = np.inf
        nearest[sl] = np.argmin(D2, axis=1)
    bmu2[missing] = nearest
    return bmu2


def prototype_influences(W_X: np.ndarray, counter, p: np.ndarray, eta: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    theta_Q = T(W_X, M_used*p) once; then for each prototype j, one
    forward-differenced evaluation at t_j = step_parameter(delta_f,
    p_j), delta_f = forward_step(eta), mass-centred over the finite
    prototypes (spec/method_notes.md section 2). 1 + M_used evaluations
    of `counter` on M_used rows.
    """
    M_used = len(p)
    omega0 = M_used * p
    delta_f = forward_step(eta)

    theta_Q = np.asarray(counter(W_X, omega0), dtype=float)
    q = theta_Q.shape[0]

    I_proto = np.empty((M_used, q), dtype=float)
    for j in range(M_used):
        member = np.zeros(M_used, dtype=bool)
        member[j] = True
        t_j = step_parameter(delta_f, float(p[j]))
        omega = perturbed_weights(omega0, member, t_j)
        I_proto[j] = (counter(W_X, omega) - theta_Q) / t_j

    # A failed evaluation at a prototype is a missing response: centre
    # over the finite prototypes only, mass-weighted, per coordinate (a
    # vector estimator can fail one output and not another at the same
    # prototype). A NaN prototype stays NaN, not filled or dropped.
    finite = np.isfinite(I_proto)
    mass = np.sum(np.where(finite, p[:, None], 0.0), axis=0)
    psi_bar = np.sum(np.where(finite, p[:, None] * I_proto, 0.0), axis=0) / mass
    I_proto -= psi_bar[None, :]
    return theta_Q, I_proto


def run_xvq(
    Z: np.ndarray, inverse: Callable[[np.ndarray], np.ndarray], counter, eta: float, M: int, seed: int,
) -> Tuple[XVQ, np.ndarray, np.ndarray]:
    """Stage 1 in full: fit the codebook on Z, map its prototypes to
    T's native coordinates with `inverse`, and measure the prototype
    influences. The only function in `core/` that calls `inverse`."""
    xvq = fit_xvq(Z, M, seed)
    W_X = inverse(xvq.centers)
    theta_Q, I_proto = prototype_influences(W_X, counter, xvq.p, eta)
    return xvq, theta_Q, I_proto
