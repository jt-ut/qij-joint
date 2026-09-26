"""
The X-VQ: quantize the data, then measure the influence at each
prototype by forward differences (spec/method_notes.md section 2),
`prototype_influences`' per-prototype step run through `pool` when
given (method_notes section 2). `run_xvq` is stage 1 in full.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Callable, Tuple

import numpy as np
from vqlp import VQFitter

from .differences import forward_step, perturbed_weights, step_parameter

_KAPPA_REF = 2.7
# Row batch for the second-BMU repair: bounds the (rows, M_used)
# distance block at tens of megabytes rather than gigabytes at large N.
_BMU2_BATCH_CAP = 4096

# Per-process, per-draw `prepare(W_X)` cache, keyed on identity --
# `bootstrap._boot_task`'s pattern, since every survey task shares W_X.
_prep_W_X = None
_prep_value = None


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
    """k-means on Z via `vqlp.VQFitter`; empty receptive fields dropped
    (M_used <= M), BMUs and CADJ reindexed to the live prototypes. No
    estimator evaluation."""
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
    """Repair `bmu2 < 0` (its live prototype was dropped) to the nearest
    LIVE prototype other than `bmu`, batched over rows so the
    (rows, M_used) distance block stays bounded."""
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


def _survey_task(T, case, W_X: np.ndarray, task):
    """One task: evaluate T at prototype j's perturbed weights against
    the shared W_X, `T.prepare` cached per process (method_notes
    section 2). A failing evaluation is caught here, this task's own
    declared failure boundary. Returns (j, raw evaluation, failure
    flag, this call's own wall time)."""
    global _prep_W_X, _prep_value
    j, omega = task
    prep = None
    if hasattr(T, 'prepare'):
        if W_X is not _prep_W_X:
            _prep_value = T.prepare(W_X)
            _prep_W_X = W_X
        prep = _prep_value
    t0 = time.perf_counter()
    try:
        result = T(W_X, omega, prep=prep) if prep is not None else T(W_X, omega)
        result = np.asarray(result, dtype=float)
        failed = bool(np.any(np.isnan(result)))
    except Exception:
        result = np.full(len(T.outputs), np.nan)
        failed = True
    return j, result, failed, time.perf_counter() - t0


def _step_prototype(omega0: np.ndarray, p: np.ndarray, delta_f: float, j: int) -> Tuple[float, np.ndarray]:
    """t_j = step_parameter(delta_f, p_j) and omega0 perturbed at prototype j alone."""
    member = np.zeros(len(p), dtype=bool)
    member[j] = True
    t_j = step_parameter(delta_f, float(p[j]))
    omega = perturbed_weights(omega0, member, t_j)
    return t_j, omega


def prototype_influences(
    W_X: np.ndarray, counter, p: np.ndarray, eta: float, pool=None,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """theta_Q = T(W_X, M_used*p) once, then one forward difference per
    prototype at t_j = step_parameter(delta_f, p_j), on `pool` when
    given, mass-centred over the finite prototypes (spec/method_
    notes.md section 2). `busy_delta` is 0 with `pool=None`, else the
    tasks' summed wall time less `pool.map`'s own wall clock."""
    M_used = len(p)
    omega0 = M_used * p
    delta_f = forward_step(eta)
    theta_Q = np.asarray(counter(W_X, omega0), dtype=float)
    q = theta_Q.shape[0]

    I_proto = np.empty((M_used, q), dtype=float)
    busy_delta = 0.0
    if pool is None:
        for j in range(M_used):
            t_j, omega = _step_prototype(omega0, p, delta_f, j)
            I_proto[j] = (counter(W_X, omega) - theta_Q) / t_j
    else:
        pool.share(W_X)
        t_j = np.empty(M_used, dtype=float)
        tasks = []
        for j in range(M_used):
            t_j[j], omega = _step_prototype(omega0, p, delta_f, j)
            tasks.append((j, omega))
        t_map0 = time.perf_counter()
        results = pool.map(_survey_task, tasks)
        busy_delta = -(time.perf_counter() - t_map0)
        for j, result, failed, wall in results:
            I_proto[j] = (result - theta_Q) / t_j[j]
            counter.add(1, M_used, int(failed))
            busy_delta += wall

    # A failed evaluation at a prototype is a missing response: centre
    # over the finite prototypes only, mass-weighted, per coordinate (a
    # vector estimator can fail one output and not another at the same
    # prototype). A NaN prototype stays NaN, not filled or dropped.
    finite = np.isfinite(I_proto)
    mass = np.sum(np.where(finite, p[:, None], 0.0), axis=0)
    psi_bar = np.sum(np.where(finite, p[:, None] * I_proto, 0.0), axis=0) / mass
    I_proto -= psi_bar[None, :]
    return theta_Q, I_proto, busy_delta


def run_xvq(
    Z: np.ndarray, inverse: Callable[[np.ndarray], np.ndarray], counter, eta: float,
    M: int, seed: int, pool=None,
) -> Tuple[XVQ, np.ndarray, np.ndarray, float]:
    """Stage 1 in full: fit the codebook on Z, map its prototypes to
    T's native coordinates with `inverse`, and survey them, on `pool`
    when given (method_notes section 2). The only function in `core/`
    that calls `inverse`."""
    xvq = fit_xvq(Z, M, seed)
    W_X = inverse(xvq.centers)
    theta_Q, I_proto, busy_delta = prototype_influences(W_X, counter, xvq.p, eta, pool)
    return xvq, theta_Q, I_proto, busy_delta
