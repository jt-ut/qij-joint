"""
The X-VQ: quantize the data (`fit_xvq`, at `workers` FAISS threads),
then measure the influence at each prototype by forward differences
(spec/method_notes.md section 2), `prototype_influences`' per-prototype
step run through `pool` when given. `run_xvq` is stage 1 in full.
`survey='moments'` (spec/method_notes.md section 2) replaces a receptive
field's one row by a representation matching its native mean and
covariance exactly, for the fields too small or too rare for a single
quantized row to stand in for the data.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Callable, Optional, Tuple

import faiss
import numpy as np
from vqlp import VQFitter

from ..parallel import prepared
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


def fit_xvq(Z: np.ndarray, M: int, seed: int, workers: int = 1) -> XVQ:
    """k-means on Z via `vqlp.VQFitter`; empty receptive fields dropped
    (M_used <= M), BMUs and CADJ reindexed to the live prototypes. No
    estimator evaluation. FAISS's OpenMP thread count is set to
    `workers` for the fit and recall, then restored -- identical
    centers/bmu/bmu2 at 1, 4 and 8 threads (method_notes section 2)."""
    Z = np.asarray(Z, dtype=float)
    N = Z.shape[0]

    prev_threads = faiss.omp_get_max_threads()
    faiss.omp_set_num_threads(workers)
    try:
        fitter = VQFitter(M=M, p=2, max_bmu=2, random_state=seed, verbose=False)
        fitter.fit(Z)
        fitter.recall(Z)
    finally:
        faiss.omp_set_num_threads(prev_threads)
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


def _field_moments(X: np.ndarray, bmu: np.ndarray, M_used: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per live field j: point count n_j, native mean and native
    population covariance, by grouped sums over the field index
    `bmu` already carries (0..M_used-1, contiguous, so no argsort is
    needed before `bincount`)."""
    d_x = X.shape[1]
    n = np.bincount(bmu, minlength=M_used).astype(float)
    mean = np.empty((M_used, d_x))
    for a in range(d_x):
        mean[:, a] = np.bincount(bmu, weights=X[:, a], minlength=M_used) / n
    cov = np.empty((M_used, d_x, d_x))
    for a in range(d_x):
        for b in range(a, d_x):
            s2 = np.bincount(bmu, weights=X[:, a] * X[:, b], minlength=M_used)
            cov[:, a, b] = cov[:, b, a] = s2 / n - mean[:, a] * mean[:, b]
    return n, mean, cov


def _unit_simplex(d_x: int) -> np.ndarray:
    """The d_x+1 vertices (columns) of a regular simplex in R^d_x,
    centred at 0, with unweighted second moment exactly I_{d_x}: the
    Helmert contrast matrix (orthonormal rows, orthogonal to the
    all-ones vector) scaled by sqrt(d_x+1) (spec/method_notes.md section 2).
    """
    helmert = np.zeros((d_x, d_x + 1))
    for k in range(d_x):
        helmert[k, :k + 1] = 1.0 / math.sqrt((k + 1) * (k + 2))
        helmert[k, k + 1] = -(k + 1) / math.sqrt((k + 1) * (k + 2))
    return math.sqrt(d_x + 1) * helmert


def _align_first_vertex(V: np.ndarray) -> np.ndarray:
    """Reflect the canonical simplex `V` (Householder about the bisector
    of its first vertex and the leading axis) so vertex 0 sits on
    +e_1, before it is mapped through a field's own covariance factor;
    a reflection is an orthogonal map, so it leaves the second moment
    at I_{d_x} unchanged (the deterministic orientation convention of
    spec/method_notes.md section 2)."""
    d_x = V.shape[0]
    a = V[:, 0] / np.linalg.norm(V[:, 0])
    e1 = np.zeros(d_x)
    e1[0] = 1.0
    u = a - e1
    norm_u = np.linalg.norm(u)
    if norm_u < 1e-12:
        return V
    u = u / norm_u
    H = np.eye(d_x) - 2.0 * np.outer(u, u)
    return H @ V


def _moments_survey_rows(
    X: np.ndarray, bmu: np.ndarray, p: np.ndarray, M_used: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The `survey='moments'` rows (spec/method_notes.md section 2): field j
    keeps its own n_j native rows when n_j <= d_x+1; otherwise it is
    the d_x+1 vertices of a regular simplex reproducing the field's
    native mean and covariance exactly, mapped through the
    eigendecomposition-based factor V*sqrt(Lambda) of its covariance
    (so a rank-deficient field, collinear points, needs no
    regularization: a zero eigenvalue maps every vertex's component
    along it to zero) with the first vertex along the field's leading
    eigenvector. Every field's rows carry R*p_j in total, split evenly
    over its own rows, R the stacked row count, so the weights sum to the
    number of rows as they do for every other call of T (an estimator
    may normalize its score by the weight total and its information by
    the row count, which agree only then). Vectorized over fields: the field
    moments, the eigendecomposition and the simplex map are one batched
    call each, over every large field at once. Returns (rows, row of
    each row's field, that row's base weight)."""
    d_x = X.shape[1]
    n, mean, cov = _field_moments(X, bmu, M_used)
    small = n <= (d_x + 1)

    small_ids = np.where(small)[0]
    point_mask = np.isin(bmu, small_ids)
    large_ids = np.where(~small)[0]
    R = int(point_mask.sum()) + large_ids.size * (d_x + 1)

    rows_small = X[point_mask]
    field_small = bmu[point_mask]
    weight_small = R * p[field_small] / n[field_small]

    if large_ids.size:
        vals, vecs = np.linalg.eigh(cov[large_ids])  # ascending, per field
        order = np.argsort(-vals, axis=1)
        vals = np.take_along_axis(vals, order, axis=1)
        vecs = np.take_along_axis(vecs, order[:, None, :], axis=2)
        factor = vecs * np.sqrt(np.clip(vals, 0.0, None))[:, None, :]
        canon = _align_first_vertex(_unit_simplex(d_x))       # (d_x, d_x+1)
        offsets = np.matmul(factor, canon)                    # (M_large, d_x, d_x+1)
        verts = mean[large_ids][:, :, None] + offsets
        rows_large = np.transpose(verts, (0, 2, 1)).reshape(-1, d_x)
        field_large = np.repeat(large_ids, d_x + 1)
        weight_large = np.repeat(R * p[large_ids] / (d_x + 1), d_x + 1)
    else:
        rows_large = np.empty((0, d_x))
        field_large = np.empty((0,), dtype=int)
        weight_large = np.empty((0,))

    rows = np.concatenate([rows_small, rows_large], axis=0)
    row_field = np.concatenate([field_small, field_large])
    weight0 = np.concatenate([weight_small, weight_large])
    return rows, row_field, weight0


def _field_step_weights(
    omega0: np.ndarray, row_field: np.ndarray, j: int, p_j: float, t: float,
) -> np.ndarray:
    """The points path's own weight constructor (method_notes section
    1), generalized to a stacked-rows `omega0` whose sum need not equal
    its length: every row of field j is scaled by (1-t) + t/p_j, every
    other row by (1-t), with p_j the field's OWN mass fraction (not
    recomputed from `omega0`, since the stacked base weights do not sum
    to the row count) -- so raising the field's mass scales all its
    rows alike, and I_j keeps its definition (method_notes section 2)."""
    member = (row_field == j).astype(float)
    return (1.0 - t) * omega0 + t * omega0 * member / p_j


def _survey_task(T, case, rows: np.ndarray, task):
    """One task: evaluate T at prototype j's perturbed weights against
    the shared survey rows (the prototypes under 'points', the stacked
    field rows under 'moments'), `T.prepare` cached per process
    (method_notes section 2). A failing evaluation is caught here, this
    task's own declared failure boundary. Returns (j, raw evaluation,
    failure flag, this call's own wall time)."""
    j, omega = task
    prep = prepared(T, rows)
    t0 = time.perf_counter()
    try:
        result = T(rows, omega, prep=prep) if prep is not None else T(rows, omega)
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
    survey: str = 'points', X: Optional[np.ndarray] = None,
    bmu: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """theta_Q = T(rows, base weights) once, then one forward difference
    per prototype j at t_j = step_parameter(delta_f, p_j), on `pool`
    when given, mass-centred over the finite prototypes (spec/method_
    notes.md section 2). Under `survey='points'`
    `rows` is one row per prototype, weight M_used*p_j.
    Under `survey='moments'` `rows` is the
    stacked field representation of `_moments_survey_rows`, `X` (native
    coordinates) and `bmu` are required, and a forward step scales a
    whole field's rows together. `busy_delta` is 0 with `pool=None`,
    else the tasks' summed wall time less `pool.map`'s own wall clock."""
    M_used = len(p)
    delta_f = forward_step(eta)

    if survey == 'points':
        rows = W_X
        omega0 = M_used * p
    else:
        rows, row_field, omega0 = _moments_survey_rows(X, bmu, p, M_used)

    theta_Q = np.asarray(counter(rows, omega0), dtype=float)
    q = theta_Q.shape[0]

    I_proto = np.empty((M_used, q), dtype=float)
    busy_delta = 0.0
    if pool is None:
        for j in range(M_used):
            if survey == 'points':
                t_j, omega = _step_prototype(omega0, p, delta_f, j)
            else:
                t_j = step_parameter(delta_f, float(p[j]))
                omega = _field_step_weights(omega0, row_field, j, float(p[j]), t_j)
            I_proto[j] = (counter(rows, omega) - theta_Q) / t_j
    else:
        pool.share(rows)
        t_j = np.empty(M_used, dtype=float)
        tasks = []
        for j in range(M_used):
            if survey == 'points':
                t_j[j], omega = _step_prototype(omega0, p, delta_f, j)
            else:
                t_j[j] = step_parameter(delta_f, float(p[j]))
                omega = _field_step_weights(omega0, row_field, j, float(p[j]), t_j[j])
            tasks.append((j, omega))
        t_map0 = time.perf_counter()
        results = pool.map(_survey_task, tasks)
        busy_delta = -(time.perf_counter() - t_map0)
        row_count = rows.shape[0]
        for j, result, failed, wall in results:
            I_proto[j] = (result - theta_Q) / t_j[j]
            counter.add(1, row_count, int(failed))
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
    M: int, seed: int, pool=None, workers: int = 1,
    survey: str = 'points', X: Optional[np.ndarray] = None,
) -> Tuple[XVQ, np.ndarray, np.ndarray, float]:
    """Stage 1 in full: fit the codebook on Z (at `workers` FAISS
    threads), map its prototypes to T's native coordinates with
    `inverse`, and survey them, on `pool` when given (method_notes
    section 2). `survey='moments'` needs
    `X`, the draw in T's native coordinates (the rows T is called on);
    it is unused under `survey='points'`. The only function in `core/`
    that calls `inverse`."""
    xvq = fit_xvq(Z, M, seed, workers)
    W_X = inverse(xvq.centers)
    theta_Q, I_proto, busy_delta = prototype_influences(
        W_X, counter, xvq.p, eta, pool, survey=survey, X=X, bmu=xvq.bmu,
    )
    return xvq, theta_Q, I_proto, busy_delta
