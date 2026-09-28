"""
The initial influence estimate (spec/method_notes.md section 3): a
per-coordinate Gaussian process through the pairs (w_j, I_j) in
whitened coordinates,

    I_proto(w_j) = h(w_j)^T beta + f(w_j) + e_j,   f ~ GP(0, s^2 k)
    e_j ~ N(0, s^2 lam)                             (unweighted noise)

h(x) is the affine basis (1, x), or, under `gptrend='quadratic'`, the
quadratic basis (1, x, {x_a x_b}_{a<=b}); constant-only when the design
is too small for the requested basis. The kernel k is the Matern-3/2
`_matern32` at one length ell shared by the whole group under
`gpwidth='global'`, or the non-stationary `_matern32_nonstationary` at
ell_j = c*h_j (h_j prototype j's local CONN spacing) under
`gpwidth='local'`, one shared factor c per group. A failed prototype
evaluation is a missing response, per coordinate: each coordinate is
fitted only on the prototypes where its own I_proto column is finite,
so non-constant coordinates are grouped by their shared finite design
and the width (or c) search runs once per group.

The width/c search and the noise floor lam_c are detailed in
spec/method_notes.md section 3. `psi0`, `uncertainty` and
`bin_posterior_variance` query the same fitted posterior at the same N
rows of Z, served from one cached pass per draw (`_point_terms`),
through the group's shared eigendecomposition of the kernel rather
than a per-coordinate Cholesky solve at query time -- an equivalent
factorisation, agreeing with the Cholesky form to rounding, not bit
for bit; `psi0` itself is untouched (still `h(x)^T beta + k(x)^T
alpha` from the same Cholesky solve `fit_influence_model` performs).
"""

from __future__ import annotations

import math
import os
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.linalg import LinAlgError, cho_factor, cho_solve, solve_triangular
from scipy.sparse import csr_matrix
from threadpoolctl import threadpool_limits
from scipy.optimize import brentq, minimize_scalar
from scipy.spatial.distance import cdist

from .differences import forward_step

_SQRT3 = math.sqrt(3.0)
_SQRT2 = math.sqrt(2.0)
_LOG_LAM_LO = math.log(1e-10)  # the absolute floor
_LOG_LAM_HI = math.log(1e2)
_N_WIDTH_GRID = 5
_UNCERTAINTY_BATCH_CAP = 4096
_BPV_CHUNK = 2048
# The widest range of c*(x - o) a block of `_matern32_self_sum_1d` may
# span: e^{+-200} is far inside double range, and the reordering it
# costs is bounded by about 100 roundings on a sum of positive terms.
_SELF_SUM_LOG_RANGE = 200.0


@dataclass
class _PointTerms:
    """
    One cached pass of the posterior over the N rows of Z, held on the
    model as `_points` and keyed on the IDENTITY of the `Z` object it
    was computed for (`qij.py` threads one `Z` through `psi0`,
    `uncertainty` and every `run_refinement` call).

    Z       the array this pass was for (cache key, held by reference
            so its id cannot be reused while cached).
    Zw      (N, d_z) Z in the model's whitened coordinates.
    psi0    (N, q) the initial influence estimate at every point.
    sigma   (N, q) the posterior standard deviation at every point.
    R       per coordinate, (N, m_c) the affine-mean residual r_i =
            h(x_i) - Hb^T A_c^-1 k_i (`bin_posterior_variance`'s R is a
            row sum of it); None on the constant path. Kernel rows
            k(x_i, w_j) themselves are never cached: every
            consumer re-forms its own rows in bounded chunks.
    """

    Z: np.ndarray
    Zw: np.ndarray
    psi0: np.ndarray
    sigma: np.ndarray
    R: List[Optional[np.ndarray]]


@dataclass
class InfluenceModel:
    """
    The fitted initial influence estimate: one Gaussian process per
    estimand coordinate, sharing a whitening and, for the non-constant
    coordinates, a common kernel width across whichever other
    coordinates happen to share its design (spec/method_notes.md
    section 3).

    A failed prototype evaluation is a missing response: coordinate c
    is fitted only on the prototypes where I_proto[:, c] is finite, so
    every field below indexed "per coordinate" is a length-q list or
    array holding that coordinate's own (possibly smaller) design.
    """

    gptrend: str                   # 'affine' or 'quadratic' (method_notes section 3)
    gpwidth: str                   # 'global' or 'local' (method_notes section 3)
    alpha: List[np.ndarray]        # per coordinate, (M_c,) GP weights on the kernel term; length 0 on the constant path
    beta: List[np.ndarray]         # per coordinate, (m_c,) mean-basis coefficients; length 0 on the constant path
    centers: List[np.ndarray]      # per coordinate, (M_c, d_z) whitened positions of that coordinate's finite prototypes
    h_design: List[Optional[np.ndarray]]  # per coordinate, (M_c,) that group's own CONN spacing h[idx_g] (gpwidth='local' only); None under 'global' and on the constant path
    whitening: Tuple[np.ndarray, np.ndarray]  # (mean, transform): raw Z -> whitened coordinates
    width: np.ndarray              # (q,) Matern-3/2 length scale ell_c under gpwidth='global'; NaN under 'local' and on the constant path
    c: np.ndarray                  # (q,) fitted width factor under gpwidth='local' (ell_j = c*h_j, shared per group); NaN under 'global' and on the constant path
    h: np.ndarray                  # (M_used,) per-prototype CONN spacing h_j, over every live prototype; NaN under gpwidth='global'
    bmu: Optional[np.ndarray]      # (N,) best-matching prototype per data point, from xvq.bmu at fit time -- gives a query point its length scale under gpwidth='local'; None under 'global'
    lam: np.ndarray                # (q,) profiled noise-to-signal ratio lam_c; NaN on the constant path
    lam_floor: np.ndarray          # (q,) the declared-noise floor lam_floor,c at the final chosen width; NaN on the constant path
    s2: np.ndarray                 # (q,) profiled GP signal variance s^2_c
    at_bound: np.ndarray           # (q, 2) bool: (width parameter, lam_c) within 1% in log of its search bound; column 0 flags ell_c under 'global' or c under 'local'
    jitter: np.ndarray             # (q,) diagonal jitter added to A at the final solve; NaN on the constant path
    constant_path: np.ndarray      # (q,) bool: coordinate has no usable spread in I_proto over its own finite design
    const_value: np.ndarray        # (q,) the constant psi0 value returned on the constant path
    m: np.ndarray                  # (q,) int, basis size per coordinate: 1 (constant only), d_z+1 (affine) or 1+d_z+d_z(d_z+1)/2 (quadratic); 0 on the constant path
    n_width_evals: np.ndarray      # (q,) number of shared outer-objective evaluations used to pick the width, within that coordinate's group
    offset: np.ndarray             # (q,) mean of psi0 over Z (diagnostic; not subtracted from psi0)
    ml_wall_time: np.ndarray       # (q,) wall time of the marginal-likelihood fit, per coordinate
    median_sigma: np.ndarray       # (q,) median posterior sd (sigma) over Z
    p95_sigma: np.ndarray          # (q,) 95th percentile posterior sd (sigma) over Z
    chol_A: List[Optional[Tuple]]         # per coordinate, cho_factor of A = K + lam_c*I (None on the constant path)
    g_chol: List[Optional[Tuple]]         # per coordinate, cho_factor of G = Hb^T A^-1 Hb (None on the constant path)
    ainv_hb: List[Optional[np.ndarray]]   # per coordinate, A^-1 Hb, (M_c, m_c) (None on the constant path)
    # Shared eigendecomposition K = V Lambda V^T at the chosen width,
    # one `eigh` per coordinate group (same three objects across a
    # group): k_eigval is Lambda clipped at 0, k_eigvec is V, hb_eig is
    # V^T Hb. None on the constant path.
    k_eigval: List[Optional[np.ndarray]]  # per coordinate, (M_c,)
    k_eigvec: List[Optional[np.ndarray]]  # per coordinate, (M_c, M_c)
    hb_eig: List[Optional[np.ndarray]]    # per coordinate, (M_c, m_c)
    # The cached posterior pass over Z (`_PointTerms`), set by `_point_terms`.
    _points: Optional[_PointTerms] = field(default=None, repr=False, compare=False)


def _whitening_from(Z: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    PCA whitening from the full X-VQ coordinates Z, used to map any
    new point into the same whitened space as `xvq.centers`: mean =
    Z.mean(0); covariance with ddof=0; eigh; transform =
    diag(1/sqrt(eigval)) @ eigvec.T. At d_z = 1 this reduces to plain
    standardization.
    """
    Z = np.asarray(Z, dtype=float)
    if Z.ndim == 1:
        Z = Z.reshape(-1, 1)
    mean = Z.mean(axis=0)
    Zc = Z - mean
    cov = (Zc.T @ Zc) / Z.shape[0]
    eigval, eigvec = np.linalg.eigh(cov)
    transform = np.diag(1.0 / np.sqrt(eigval)) @ eigvec.T
    return mean, transform


def _matern32(r: np.ndarray, ell: float) -> np.ndarray:
    """k_ell(r) = (1 + sqrt(3) r / ell) exp(-sqrt(3) r / ell)."""
    s = (_SQRT3 / ell) * r
    return (1.0 + s) * np.exp(-s)


def _matern32_nonstationary(r: np.ndarray, ell_row: np.ndarray, ell_col: np.ndarray, d_z: int) -> np.ndarray:
    """
    Paciorek-Schervish non-stationary Matern-3/2 (method_notes
    section 3): for whitened distance r between two locations of
    lengths ell, ell',

        k(r; ell, ell') = (2*ell*ell'/(ell^2+ell'^2))^(d_z/2)
                          * kappa(r*sqrt(2)/sqrt(ell^2+ell'^2)),
        kappa(t) = (1 + sqrt(3)*t) * exp(-sqrt(3)*t).

    `r` is (n, m); `ell_row` (n,) and `ell_col` (m,) are the two
    sides' length scales. Equals `_matern32(r, ell)` mathematically
    when ell_row == ell_col == ell everywhere, not bit for bit --
    gpwidth='global' evaluates `_matern32` itself for that reason.
    """
    ell_row = np.asarray(ell_row, dtype=float).reshape(-1, 1)
    ell_col = np.asarray(ell_col, dtype=float).reshape(1, -1)
    sum2 = ell_row ** 2 + ell_col ** 2
    pref = (2.0 * ell_row * ell_col / sum2) ** (d_z / 2.0)
    s = _SQRT3 * (r * _SQRT2 / np.sqrt(sum2))
    return pref * (1.0 + s) * np.exp(-s)


def _conn_spacing(conn, D_full: np.ndarray) -> np.ndarray:
    """
    h_j = median over live prototype j's CONN neighbours of the
    whitened distance ||z_j - z_k|| (method_notes section 3):
    every live prototype has at least one CONN neighbour, so no floor,
    no minimum degree, no fallback. `conn` is the (M, M) symmetrized
    CADJ graph on the same M prototypes as `D_full`.
    """
    mask = np.asarray(conn.toarray(), dtype=bool)
    np.fill_diagonal(mask, False)
    return np.nanmedian(np.where(mask, D_full, np.nan), axis=1)


def _matern32_self_sum_1d(x: np.ndarray, ell: float) -> float:
    """
    sum_{i,j} k_ell(|x_i - x_j|) over one bin's OWN points, for a
    one-dimensional design, in O(n log n) via running sums over the
    sorted points (spec/method_notes.md section 3 derives the
    recursion) rather than the O(n^2) pairwise sum -- a shortcut valid
    only for this STATIONARY kernel (gpwidth='global'). The origin is
    re-based every `_SELF_SUM_LOG_RANGE`/c points so no exponential
    overflows; the carried running sums are exact across a rebase, so
    no pair is dropped or approximated -- only the summation order
    differs from the pairwise form.
    """
    xs = np.sort(np.asarray(x, dtype=float).ravel())
    n = xs.size
    if n <= 1:
        return float(n)

    c = _SQRT3 / ell
    block_span = _SELF_SUM_LOG_RANGE / c

    acc = 0.0
    s_carry = 0.0   # sum over earlier points of e^{c (x_j - o)}
    t_carry = 0.0   # sum over earlier points of (x_j - o) e^{c (x_j - o)}
    start = 0
    while start < n:
        end = int(np.searchsorted(xs, xs[start] + block_span, side='right'))
        u = xs[start:end] - xs[start]
        a = np.exp(-c * u)
        b = np.exp(c * u)
        ub = u * b
        # Exclusive cumulative sums, built by shifting an inclusive one
        # rather than subtracting the term back off: b increases
        # across a block, so `cumsum(b) - b` would cancel to nothing
        # wherever a gap makes b_i dominate everything before it.
        cb = np.empty(u.size, dtype=float)
        cb[0] = 0.0
        np.cumsum(b[:-1], out=cb[1:])
        cub = np.empty(u.size, dtype=float)
        cub[0] = 0.0
        np.cumsum(ub[:-1], out=cub[1:])

        s_tot = s_carry + cb
        S = a * s_tot
        T = u * S - a * (t_carry + cub)
        acc += float(np.add.reduce(S) + c * np.add.reduce(T))

        if end < n:
            s_all = s_carry + float(np.add.reduce(b))
            t_all = t_carry + float(np.add.reduce(ub))
            d = float(xs[end] - xs[start])
            phi = math.exp(-c * d)
            s_carry = phi * s_all
            t_carry = phi * (t_all - d * s_all)
        start = end

    return float(n) + 2.0 * acc


def _basis(Zw: np.ndarray, m: int) -> np.ndarray:
    """
    h(x) in whitened coordinates (method_notes section 3): the
    constant basis (m = 1, also the small-design fallback for either
    trend), the affine basis (1, x) at m = d_z + 1, or the quadratic
    basis (1, x, {x_a x_b}_{a<=b}) at m = 1 + d_z + d_z(d_z+1)/2.
    """
    N, d_z = Zw.shape
    if m == 1:
        return np.ones((N, 1), dtype=float)
    if m == d_z + 1:
        return np.hstack([np.ones((N, 1), dtype=float), Zw])
    quad = np.stack([Zw[:, a] * Zw[:, b] for a in range(d_z) for b in range(a, d_z)], axis=1)
    return np.hstack([np.ones((N, 1), dtype=float), Zw, quad])


def _length_scale_bounds(conn, D_full: np.ndarray) -> Tuple[float, float]:
    """
    ell_min = the median whitened distance between CONN-connected
    prototypes; fallback = the smallest positive inter-prototype
    distance if no CONN edges of positive length exist, or 1.0 if
    every prototype coincides. ell_max = 10x the largest
    inter-prototype distance, forced to 10x ell_min if that is not
    larger than ell_min. `conn`/`D_full` are taken over whichever
    design is in play (a coordinate group's own finite subset); under
    gpwidth='local' these same two bounds are converted to the bounds
    on c (method_notes section 3) rather than searched directly.
    """
    M = D_full.shape[0]
    iu = np.triu_indices(M, k=1)
    all_d = D_full[iu]
    pos_all = all_d[all_d > 0.0]

    conn_coo = conn.tocoo()
    mask = conn_coo.row < conn_coo.col
    ii, jj = conn_coo.row[mask], conn_coo.col[mask]
    conn_d = D_full[ii, jj]
    conn_d = conn_d[conn_d > 0.0]

    if conn_d.size > 0:
        ell_min = float(np.median(conn_d))
    elif pos_all.size > 0:
        ell_min = float(pos_all.min())
    else:
        ell_min = 1.0  # degenerate: every prototype coincides

    ell_max = float(10.0 * all_d.max()) if all_d.size > 0 else 10.0 * ell_min
    if not (ell_max > ell_min):
        ell_max = ell_min * 10.0

    return ell_min, ell_max


def _cholesky_with_jitter(A: np.ndarray, K: np.ndarray, M_used: int):
    """
    One Cholesky of A, jitter only here: 0 -> 1e-10*trace(K)/M_used
    -> x10 -> x10 (at most three escalations). Raises RuntimeError if
    it still fails; `QIJ.fit` records that as a failed draw.
    Returns ((c, lower), jitter_used).
    """
    base = 1e-10 * float(np.trace(K)) / M_used
    eye = np.eye(M_used)
    for j in (0.0, base, base * 10.0, base * 100.0):
        try:
            chol = cho_factor(A + j * eye, lower=True)
            return chol, float(j)
        except LinAlgError:
            continue
    raise RuntimeError(
        "influence_model.fit_influence_model: Cholesky of A failed after jitter escalation"
    )


def _within_1pct_log(x: float, lo: float, hi: float) -> bool:
    """Within 1% in log of either bound."""
    if not (np.isfinite(x) and x > 0.0):
        return False
    near_lo = lo > 0.0 and abs(math.log(x) - math.log(lo)) < 0.01
    near_hi = hi > 0.0 and abs(math.log(x) - math.log(hi)) < 0.01
    return bool(near_lo or near_hi)


def _floor_equation(log_lam: float, z: np.ndarray, Lambda: np.ndarray, M_minus_m: int) -> float:
    """lam * s_c^2(lam) (spec section 3, declared noise floor), as a
    function of lam alone: sum(z^2 * lam / (Lambda + lam)) / (M - m).
    Increasing in lam."""
    lam = math.exp(log_lam)
    return float(np.sum(z ** 2 * lam / (Lambda + lam)) / M_minus_m)


def _lambda_floor(z: np.ndarray, Lambda: np.ndarray, M_minus_m: int, n_c2: float) -> float:
    """
    The declared-noise floor lam_floor,c for one coordinate at one
    candidate width (spec section 3): the unique root of
    lam*s_c^2(lam) = n_c2 in log lam, found by one `brentq` call, since
    `_floor_equation` is increasing in lam. Pinned to the search
    domain's own edges, [1e-10, 1e2], when the root falls outside it:
    the lower edge when even the smallest lam already exceeds n_c2, the
    upper edge when even the largest lam cannot reach it.
    """
    lo, hi = _LOG_LAM_LO, _LOG_LAM_HI
    g_lo = _floor_equation(lo, z, Lambda, M_minus_m) - n_c2
    if g_lo >= 0.0:
        return math.exp(lo)
    g_hi = _floor_equation(hi, z, Lambda, M_minus_m) - n_c2
    if g_hi <= 0.0:
        return math.exp(hi)
    root = brentq(lambda x: _floor_equation(x, z, Lambda, M_minus_m) - n_c2, lo, hi)
    return math.exp(root)


def _width_grid_candidate(
    log_param: float, gpwidth: str, D_full_g: np.ndarray, h_design_g: Optional[np.ndarray],
    d_z: int, W_g: np.ndarray, tau_g: float, M_minus_m: int,
    cols_g: Sequence[int], psi_proj: Dict[int, np.ndarray], n_c2_g: Dict[int, float],
) -> Tuple[dict, float]:
    """
    One width/c-grid candidate's outer objective (method_notes section
    3): kernel, mean basis projected out, one shared `eigh`, then each
    coordinate's profiled REML search over lam. Called directly or
    through `_grid_task` (a pool task): same operations, same order
    either way, so the grid's numbers do not depend on the pool.
    Returns the trace entry (log_param, param, K, per_c, nll) and this
    call's own wall time.
    """
    t0 = time.perf_counter()
    param = math.exp(log_param)
    if gpwidth == 'global':
        K = _matern32(D_full_g, param)
    else:
        ell_vec = param * h_design_g
        K = _matern32_nonstationary(D_full_g, ell_vec, ell_vec, d_z)

    C = W_g.T @ K                      # (m_g, M_g)
    E = C @ W_g                        # (m_g, m_g)
    E[np.diag_indices_from(E)] += tau_g
    WC = W_g @ C                       # (M_g, M_g)
    A = K - WC - WC.T + W_g @ (E @ W_g.T)
    Lambda, V = np.linalg.eigh(A)
    Lambda = np.maximum(Lambda[:M_minus_m], 0.0)
    V = V[:, :M_minus_m]

    per_c: Dict[int, dict] = {}
    total_nll = 0.0
    for c in cols_g:
        z = V.T @ psi_proj[c]

        def inner(log_lam: float, _z=z, _Lambda=Lambda) -> float:
            denom = _Lambda + math.exp(log_lam)
            s2_val = np.sum(_z ** 2 / denom) / M_minus_m
            s2_val = max(s2_val, 1e-300)
            return (M_minus_m / 2.0) * math.log(s2_val) + 0.5 * np.sum(np.log(denom))

        # lam_c is bounded below by lam_floor,c at THIS candidate width.
        lam_floor_c = _lambda_floor(z, Lambda, M_minus_m, n_c2_g[c])
        lo_bound = max(math.log(lam_floor_c), _LOG_LAM_LO)
        if lo_bound >= _LOG_LAM_HI:
            # The floor already pins lam_c at the ceiling: nothing to search.
            log_lam_star = _LOG_LAM_HI
            nll_c = inner(log_lam_star)
        else:
            ir = minimize_scalar(inner, bounds=(lo_bound, _LOG_LAM_HI), method='bounded')
            log_lam_star = float(ir.x)
            nll_c = float(ir.fun)
        per_c[c] = dict(log_lam=log_lam_star, nll=nll_c, lam_floor=lam_floor_c)
        total_nll += nll_c

    entry = dict(log_param=log_param, param=param, K=K, per_c=per_c, nll=total_nll)
    return entry, time.perf_counter() - t0


def _grid_task(T, case, X, task):
    """Pool wrapper matching `Pool`'s task signature; the width grid
    needs none of T, case or X -- `task`'s order matches `_width_grid_
    candidate`'s own positional arguments, at most one M x M array."""
    return _width_grid_candidate(*task)


def fit_influence_model(
    Z: np.ndarray, xvq, I_proto: np.ndarray, theta_Q: np.ndarray, eta: float,
    gptrend: str = 'affine', gpwidth: str = 'global', pool=None,
) -> Tuple[InfluenceModel, float]:
    """
    The initial influence estimate for every estimand coordinate
    (spec/method_notes.md section 3): Gaussian-process regression with
    a mean basis set by `gptrend` ('affine' or 'quadratic') and a
    Matern-3/2 kernel whose width is either one shared ell per group
    (`gpwidth='global'`) or a per-prototype length
    ell_j = c*h_j at one shared factor c per group (`gpwidth='local'`).
    `xvq.p` and `theta_Q` are used for the constant-response threshold
    and, with `eta`, for the declared noise floor on lam_c; the
    kernel-regression noise itself is unweighted.

    A prototype whose evaluation failed for coordinate c is a missing
    response: coordinate c's design is exactly the prototypes where
    `I_proto[:, c]` is finite, grouped by shared finite design so the
    width search runs once per group; with no failures every
    coordinate is one group.

    `xvq.centers` are prototype positions in Z's native (unwhitened)
    coordinates; whitened here with the SAME transform derived from Z,
    so a coordinate's design lands in the space `psi0`/`uncertainty`
    query in. Cost: O(M_X_used^3), independent of N.

    With a `pool`, every group's five grid candidates run as pool tasks
    submitted together (the bounded refinement stays serial: method_
    notes section 3). Returns (model, busy_delta); busy_delta is 0.0
    with `pool=None`, else the grid tasks' wall time less `pool.map`'s.
    """
    p = np.asarray(xvq.p, dtype=float)
    I_proto = np.asarray(I_proto, dtype=float)
    theta_Q = np.asarray(theta_Q, dtype=float)
    M_X_used, q = I_proto.shape
    raw_centers = np.asarray(xvq.centers, dtype=float)
    d_z = raw_centers.shape[1]

    # Declared noise floor (spec section 3): delta_f is the SAME
    # forward step `xvq.prototype_influences` used to measure I_proto,
    # so t_j below reproduces that function's own t_j exactly.
    delta_f = forward_step(eta)

    mean, transform = _whitening_from(Z)
    centers_full = (raw_centers - mean) @ transform.T

    if gpwidth == 'local':
        D_proto_full = cdist(centers_full, centers_full)
        h_full = _conn_spacing(xvq.conn, D_proto_full)
        bmu_full = np.asarray(xvq.bmu, dtype=np.intp).copy()
    else:
        h_full = np.full(M_X_used, np.nan, dtype=float)
        bmu_full = None

    finite = np.isfinite(I_proto)  # (M_X_used, q)

    constant_path = np.zeros(q, dtype=bool)
    const_value = np.zeros(q, dtype=float)
    offset = np.zeros(q, dtype=float)
    m_arr = np.zeros(q, dtype=int)

    centers: List[np.ndarray] = [np.empty((0, d_z), dtype=float)] * q
    h_design: List[Optional[np.ndarray]] = [None] * q
    alpha: List[np.ndarray] = [np.empty(0, dtype=float)] * q
    beta: List[np.ndarray] = [np.empty(0, dtype=float)] * q
    width = np.full(q, np.nan, dtype=float)
    c_arr = np.full(q, np.nan, dtype=float)
    lam = np.full(q, np.nan, dtype=float)
    lam_floor = np.full(q, np.nan, dtype=float)
    s2 = np.zeros(q, dtype=float)
    at_bound = np.zeros((q, 2), dtype=bool)
    jitter = np.full(q, np.nan, dtype=float)
    n_width_evals = np.zeros(q, dtype=int)
    ml_wall_time = np.zeros(q, dtype=float)

    chol_A: List[Optional[tuple]] = [None] * q
    g_chol: List[Optional[tuple]] = [None] * q
    ainv_hb: List[Optional[np.ndarray]] = [None] * q
    k_eigval: List[Optional[np.ndarray]] = [None] * q
    k_eigvec: List[Optional[np.ndarray]] = [None] * q
    hb_eig: List[Optional[np.ndarray]] = [None] * q

    groups: Dict[tuple, List[int]] = {}
    for c in range(q):
        idx_c = np.where(finite[:, c])[0]
        M_c = idx_c.size

        if M_c == 0:
            # No finite prototype for this output: nothing to centre a
            # mean on, so the constant-path value is 0 (confirmed
            # reachable on a stage-1 collapse where every prototype
            # evaluation failed for this coordinate).
            psi_bar = 0.0
        else:
            p_c = p[idx_c]
            mass_c = float(p_c.sum())
            psi_bar = float(np.sum(p_c * I_proto[idx_c, c]) / mass_c) if mass_c > 0.0 else 0.0

        coord_constant = M_c < 3
        if not coord_constant:
            p_c = p[idx_c]
            mass_c = float(p_c.sum())
            p_c_norm = p_c / mass_c
            psi_col = I_proto[idx_c, c]
            spread = float(np.sqrt(np.sum(p_c_norm * (psi_col - psi_bar) ** 2)))
            coord_constant = spread <= 1e-10 * max(abs(float(theta_Q[c])), 1e-300)

        if coord_constant:
            constant_path[c] = True
            const_value[c] = psi_bar
            offset[c] = psi_bar
        else:
            groups.setdefault(tuple(idx_c.tolist()), []).append(c)

    # The requested basis size for the trend in use; the small-design
    # fallback (m_g = 1) applies whenever a group's own finite design
    # cannot support it (method_notes section 3).
    m_full = (d_z + 1) if gptrend == 'affine' else (1 + d_z + d_z * (d_z + 1) // 2)

    # Each group's static context, built first so every group's five
    # grid candidates can go to the pool together in one batch below.
    group_ctx = []
    tasks = []
    task_group = []
    for idx_key, cols_g in groups.items():
        idx_g = np.asarray(idx_key, dtype=np.intp)
        M_g = idx_g.size
        centers_g = centers_full[idx_g]
        m_g = 1 if M_g <= m_full + 1 else m_full
        Hb_g = _basis(centers_g, m_g)

        D_full_g = cdist(centers_g, centers_g)
        conn_g = xvq.conn[idx_g, :][:, idx_g]
        ell_min, ell_max = _length_scale_bounds(conn_g, D_full_g)

        # The outer search runs over ell under 'global', or over c under
        # 'local' (bounds converted through the group's CONN spacing, so
        # c_min is about 1; method_notes section 3).
        if gpwidth == 'global':
            param_min, param_max = ell_min, ell_max
            h_design_g = None
        else:
            h_design_g = h_full[idx_g]
            param_min = ell_min / float(np.median(h_design_g))
            param_max = ell_max / float(np.min(h_design_g))

        grid_log_param = np.linspace(math.log(param_min), math.log(param_max), _N_WIDTH_GRID)

        Qfull, _R = np.linalg.qr(Hb_g, mode='complete')
        Q = Qfull[:, m_g:]
        M_minus_m = M_g - m_g

        # A_param = (I-P) K_param (I-P) + tau*P, P = W W^T, gives the
        # same spectrum as Q^T K_param Q in O(m_g M_g^2), not O(M_g^3)
        # (derivation: spec/method_notes.md section 3).
        W_g = Qfull[:, :m_g]
        tau_g = 2.0 * float(M_g)
        psi_proj = {c: Q @ (Q.T @ I_proto[idx_g, c]) for c in cols_g}

        # n_c^2 = 2*eta^2*theta_Q,c^2*median_j(1/t_j^2), independent of
        # width so computed once per group (method_notes section 3).
        p_g = p[idx_g]
        t_g = delta_f * p_g / (1.0 - p_g)
        inv_t2_median_g = float(np.median(1.0 / t_g ** 2))
        n_c2_g = {c: 2.0 * eta ** 2 * float(theta_Q[c]) ** 2 * inv_t2_median_g for c in cols_g}

        gi = len(group_ctx)
        group_ctx.append((idx_g, M_g, centers_g, m_g, Hb_g, D_full_g, h_design_g,
                           param_min, param_max, grid_log_param, W_g, tau_g, M_minus_m,
                           cols_g, psi_proj, n_c2_g))
        for lp in grid_log_param:
            tasks.append((float(lp), gpwidth, D_full_g, h_design_g, d_z, W_g, tau_g,
                          M_minus_m, cols_g, psi_proj, n_c2_g))
            task_group.append(gi)

    # Every group's five grid candidates -- independent of each other
    # and of every other group's (method_notes section 3) -- go to the
    # pool as one batch, or run serially with no pool.
    model_busy = 0.0
    if not tasks:
        grid_results = []  # every coordinate is on the constant path
    elif pool is None:
        grid_results = [_width_grid_candidate(*task) for task in tasks]
    else:
        t_map0 = time.perf_counter()
        grid_results = pool.map(_grid_task, tasks)
        model_busy = -(time.perf_counter() - t_map0) + sum(wall for _entry, wall in grid_results)

    traces: List[List[dict]] = [[] for _ in group_ctx]
    for gi, (entry, _wall) in zip(task_group, grid_results):
        traces[gi].append(entry)

    # Per group, the bounded refinement between the grid's neighbours
    # (always serial: method_notes section 3) and the final
    # per-coordinate solve. One kernel and one eigendecomposition of
    # Q^T K Q per candidate, shared across the group; each coordinate
    # profiles its own lam_c from that shared eigendecomposition, and
    # the outer objective is the SUM of the group's per-coordinate
    # profiled restricted NLLs.
    for gi, (idx_g, M_g, centers_g, m_g, Hb_g, D_full_g, h_design_g, param_min, param_max,
             grid_log_param, W_g, tau_g, M_minus_m, cols_g, psi_proj, n_c2_g) in enumerate(group_ctx):
        t_shared0 = time.perf_counter()
        trace = traces[gi]

        def outer_obj(log_param: float, _trace=trace, _cols_g=cols_g,
                       _psi_proj=psi_proj, _W=W_g, _tau=tau_g, _M_minus_m=M_minus_m,
                       _n_c2=n_c2_g, _gpwidth=gpwidth, _D=D_full_g, _h=h_design_g,
                       _dz=d_z) -> float:
            entry, _wall = _width_grid_candidate(
                log_param, _gpwidth, _D, _h, _dz, _W, _tau, _M_minus_m, _cols_g, _psi_proj, _n_c2)
            _trace.append(entry)
            return entry['nll']

        grid_nlls = [t['nll'] for t in trace[:_N_WIDTH_GRID]]
        best_idx = int(np.argmin(grid_nlls))
        # Skip the bounded refinement when the best grid point is the
        # upper endpoint: past about the upper bound the kernel family
        # collapses and the width is not identified there (spec/
        # method_notes.md section 3). The lower endpoint keeps the
        # refinement.
        if best_idx == _N_WIDTH_GRID - 1:
            lo = hi = None
        elif best_idx == 0:
            lo, hi = grid_log_param[0], grid_log_param[1]
        else:
            lo, hi = grid_log_param[best_idx - 1], grid_log_param[best_idx + 1]

        if lo is not None and hi > lo:
            minimize_scalar(outer_obj, bounds=(float(lo), float(hi)), method='bounded')

        best = min(trace, key=lambda t: t['nll'])
        n_shared_evals = len(trace)
        t_shared_total = time.perf_counter() - t_shared0
        t_shared_share = t_shared_total / len(cols_g)

        param_val = best['param']
        K_c = best['K']

        # The group's shared eigendecomposition of K: one `eigh` of the
        # chosen width's kernel, shared by every coordinate in the
        # group and by every later query of the posterior. Lambda is
        # clipped at 0 exactly as the outer search clips its own
        # projected spectrum -- lam_c is at least 1e-10, so
        # Lambda + lam_c + jit_c is strictly positive.
        Lambda_K, V_K = np.linalg.eigh(K_c)
        Lambda_K = np.maximum(Lambda_K, 0.0)
        HbV_g = V_K.T @ Hb_g

        # Per-coordinate final solve (A depends on lam_c).
        for c in cols_g:
            t0 = time.perf_counter()
            psi_c = I_proto[idx_g, c]
            lam_c = math.exp(best['per_c'][c]['log_lam'])
            lam_floor_final = best['per_c'][c]['lam_floor']

            A = K_c + lam_c * np.eye(M_g)
            chol, jit = _cholesky_with_jitter(A, K_c, M_g)

            AinvHb_c = cho_solve(chol, Hb_g)
            G = Hb_g.T @ AinvHb_c
            G_chol_c = cho_factor(G, lower=True)

            Ainv_psi = cho_solve(chol, psi_c)
            u_vec = Hb_g.T @ Ainv_psi
            beta_c = cho_solve(G_chol_c, u_vec)
            alpha_c = Ainv_psi - AinvHb_c @ beta_c

            # denom_m = M_g - m_g is always > 0 here: every coordinate
            # in this group left the constant path, which requires
            # M_g >= 3, and m_g is 1 (M_g <= m_full+1) or m_full
            # (M_g > m_full+1) -- either way M_g - m_g >= 2.
            denom_m = M_g - m_g
            resid_quad = float(psi_c @ Ainv_psi - u_vec @ beta_c)
            s2_c = max(resid_quad, 0.0) / denom_m

            centers[c] = centers_g
            h_design[c] = h_design_g
            m_arr[c] = m_g
            if gpwidth == 'global':
                width[c] = param_val
            else:
                c_arr[c] = param_val
            lam[c] = lam_c
            lam_floor[c] = lam_floor_final
            s2[c] = s2_c
            jitter[c] = jit
            alpha[c] = alpha_c
            beta[c] = beta_c
            n_width_evals[c] = n_shared_evals
            at_bound[c, 0] = _within_1pct_log(param_val, param_min, param_max)
            # lam_c's own search bound is [max(lam_floor_final, 1e-10),
            # 1e2], not the fixed [1e-10, 1e2]: the declared floor,
            # when active, moves the lower edge.
            at_bound[c, 1] = _within_1pct_log(lam_c, max(lam_floor_final, 1e-10), 1e2)

            chol_A[c] = chol
            g_chol[c] = G_chol_c
            ainv_hb[c] = AinvHb_c
            k_eigval[c] = Lambda_K
            k_eigvec[c] = V_K
            hb_eig[c] = HbV_g

            ml_wall_time[c] = t_shared_share + (time.perf_counter() - t0)

    model = InfluenceModel(
        gptrend=gptrend,
        gpwidth=gpwidth,
        alpha=alpha,
        beta=beta,
        centers=centers,
        h_design=h_design,
        whitening=(mean, transform),
        width=width,
        c=c_arr,
        h=h_full,
        bmu=bmu_full,
        lam=lam,
        lam_floor=lam_floor,
        s2=s2,
        at_bound=at_bound,
        jitter=jitter,
        constant_path=constant_path,
        const_value=const_value,
        m=m_arr,
        n_width_evals=n_width_evals,
        offset=offset,
        ml_wall_time=ml_wall_time,
        median_sigma=np.zeros(q, dtype=float),
        p95_sigma=np.zeros(q, dtype=float),
        chol_A=chol_A,
        g_chol=g_chol,
        ainv_hb=ainv_hb,
        k_eigval=k_eigval,
        k_eigvec=k_eigvec,
        hb_eig=hb_eig,
    )

    preds = psi0(model, Z)
    for c in range(q):
        if not constant_path[c]:
            model.offset[c] = float(preds[:, c].mean())

    sigma_full = uncertainty(model, Z)
    for c in range(q):
        if not constant_path[c]:
            model.median_sigma[c] = float(np.median(sigma_full[:, c]))
            model.p95_sigma[c] = float(np.percentile(sigma_full[:, c], 95))

    return model, model_busy


def _coordinate_groups(model: InfluenceModel) -> List[List[int]]:
    """The non-constant coordinates grouped by shared design, in
    coordinate order: `fit_influence_model` assigns every coordinate in
    a group the SAME `centers` array object, so object identity is the
    group key. With no failed prototype evaluation this is one group
    holding every coordinate."""
    groups: Dict[int, List[int]] = {}
    for c in range(len(model.centers)):
        if model.constant_path[c]:
            continue
        groups.setdefault(id(model.centers[c]), []).append(c)
    return list(groups.values())


def _point_terms(model: InfluenceModel, Z: np.ndarray) -> _PointTerms:
    """
    The one pass of the posterior over the N rows of Z, cached on the
    model and keyed on the identity of `Z`. Returns psi0, sigma and the
    per-point affine-mean residuals r_i, which `psi0`, `uncertainty` and
    `bin_posterior_variance` read instead of recomputing. psi0 and
    sigma are produced together because they share the kernel rows
    k(x_i, w_j), formed in row chunks of at most `_UNCERTAINTY_BATCH_
    CAP` and never held beyond the chunk that used them. Under
    `gpwidth='local'` those rows come from `_matern32_nonstationary` at
    each query row's own length scale c*h[bmu_i] (`model.bmu`, set at
    fit time) against the design's c*h_design; under 'global' from
    `_matern32` at the group's shared ell.
    """
    cached = model._points
    if cached is not None and cached.Z is Z:
        return cached

    Za = np.asarray(Z, dtype=float)
    if Za.ndim == 1:
        Za = Za.reshape(-1, 1)
    mean, transform = model.whitening
    Zw = (Za - mean) @ transform.T
    N = Zw.shape[0]
    d_z = Zw.shape[1]
    q = len(model.centers)

    psi0_out = np.empty((N, q), dtype=float)
    sigma_out = np.zeros((N, q), dtype=float)
    R_out: List[Optional[np.ndarray]] = [None] * q

    for c in range(q):
        if model.constant_path[c]:
            psi0_out[:, c] = model.const_value[c]

    groups = _coordinate_groups(model)
    batch = _UNCERTAINTY_BATCH_CAP
    for cols in groups:
        c0 = cols[0]
        centers_g = model.centers[c0]
        m_g = int(model.m[c0])
        V_K = model.k_eigvec[c0]
        Lambda_K = model.k_eigval[c0]
        HbV_g = model.hb_eig[c0]
        if model.gpwidth == 'global':
            ell_g = float(model.width[c0])
        else:
            c_val_g = float(model.c[c0])
            h_col_g = c_val_g * model.h_design[c0]

        # Per-coordinate constants of the eigen form: w_c = 1 /
        # (Lambda + lam_c + jit_c) is A_c^-1's spectrum, and B_c =
        # (V^T Hb) scaled by it turns the affine-mean correction into
        # one (nb, M) x (M, m) product against the shared P.
        w_by_c = {}
        B_by_c = {}
        for c in cols:
            w_c = 1.0 / (Lambda_K + float(model.lam[c]) + float(model.jitter[c]))
            w_by_c[c] = w_c
            B_by_c[c] = HbV_g * w_c[:, None]
            R_out[c] = np.empty((N, m_g), dtype=float)

        for start in range(0, N, batch):
            sl = slice(start, start + batch)
            Zc = Zw[sl]
            Hc = _basis(Zc, m_g)
            if model.gpwidth == 'global':
                Kc = _matern32(cdist(Zc, centers_g), ell_g)  # (nb, M_g)
            else:
                ell_row = c_val_g * model.h[model.bmu[sl]]
                Kc = _matern32_nonstationary(cdist(Zc, centers_g), ell_row, h_col_g, d_z)

            for c in cols:
                psi0_out[sl, c] = Hc @ model.beta[c] + Kc @ model.alpha[c]

            Pc = Kc @ V_K  # (nb, M_g); ONE product for every output
            for c in cols:
                Rc = Hc - Pc @ B_by_c[c]  # (nb, m_g)
                R_out[c][sl] = Rc

            # Pc^2 goes into Kc's buffer: Kc has done its work above
            # and the two have identical shape, so the pass holds two
            # (nb, M_g) arrays at a time.
            np.multiply(Pc, Pc, out=Kc)
            for c in cols:
                term1 = Kc @ w_by_c[c]
                Rc = R_out[c][sl]
                GinvRcT = cho_solve(model.g_chol[c], Rc.T)  # (m_c, nb)
                term2 = np.einsum('ij,ji->i', Rc, GinvRcT)
                sigma2 = model.s2[c] * (1.0 - term1 + term2)
                sigma2 = np.maximum(sigma2, 0.0)
                sigma_out[sl, c] = np.sqrt(sigma2)

    terms = _PointTerms(Z=Z, Zw=Zw, psi0=psi0_out, sigma=sigma_out, R=R_out)
    model._points = terms
    return terms


def psi0(model: InfluenceModel, Z: np.ndarray) -> np.ndarray:
    """
    The initial influence estimate psi0(x) at Z (N, q): psi0(x) =
    h(x)^T beta + sum_j alpha_j k(x, w_j), over coordinate c's OWN
    design. Constant-path coordinates return `const_value[c]` at every
    row. Cost O(N * M_X_used) per coordinate.

    Computed by `_point_terms`, in the same pass as `uncertainty` and
    cached with it; a second call with the same `Z` costs nothing. THE
    RETURNED ARRAY IS THE CACHE, not a copy: callers must not write
    into it.
    """
    return _point_terms(model, Z).psi0


def uncertainty(model: InfluenceModel, Z: np.ndarray) -> np.ndarray:
    """
    Posterior standard deviation sigma at Z (N, q) (R&W eq. 2.42):
    sigma_i^2 = s^2 * [1 - k_i^T A^-1 k_i + r_i^T G^-1 r_i], r_i =
    h(x_i) - Hb^T A^-1 k_i, all coordinate c's own design, A, G.
    Clipped at 0 before the square root. Constant-path coordinates
    return 0.

    Both A^-1 terms come from the group's shared eigendecomposition
    K = V Lambda V^T rather than a per-output Cholesky of A_c: P =
    K_n V is formed ONCE for every output in the group. Cost
    O(N * M_X_used^2) once per coordinate group, plus
    O(N * M_X_used * m) per coordinate.

    Computed and cached by `_point_terms`, in the same pass as `psi0`;
    the returned array is the cache, read-only by convention.
    """
    return _point_terms(model, Z).sigma


def bin_posterior_variance(
    model: InfluenceModel,
    Z: np.ndarray,
    coordinate: int,
    groups: Sequence[np.ndarray],
    sigma_c: np.ndarray,
    with_mean: bool = False,
):
    """
    The within-bin posterior variance v_k for coordinate `coordinate`
    at each group in `groups` (spec/method_notes.md section 3):

        v_k = mean(diag Sigma_k) - mean(Sigma_k)
        Sigma_ij = s2_c * [k(x_i, x_j) - k_i^T A^-1 k_j + r_i^T G^-1 r_j]

    `mean(diag Sigma_k)` is the mean of sigma_i^2 over the group, taken
    directly from `sigma_c` (N,), the per-point posterior sd already
    computed by `uncertainty(model, Z)` -- NOT re-derived, since that
    would repeat the O(N * M_X_used^2) product `uncertainty` already
    paid for.

    `mean(Sigma_k)` is computed from bin-summed vectors, never by
    holding an n_k x n_k matrix: with s = sum_{i in k} k_i and
    R = sum_{i in k} r_i,

        mean(Sigma_k) = s2_c / n_k^2 * [SS_k - s^T A^-1 s + R^T G^-1 R]
        SS_k = sum_{i,j in k} k(x_i, x_j)

    `R` is a row sum of the per-point residuals `_point_terms` already
    cached; `s` is accumulated over the group's rows in chunks of at
    most 2048, from kernel rows formed fresh in each chunk (E4: no
    N x M array, cached or not). `SS_k` is the one term chunking alone
    does not shrink -- a sum of raw kernel values between the bin's OWN
    points. Under `gpwidth='global'` in one dimension,
    `_matern32_self_sum_1d` gets it in O(n_k log n_k) (a shortcut valid
    only for that stationary kernel); otherwise, and always under
    `gpwidth='local'` (the kernel is non-stationary, so no running-sum
    shortcut applies), the pairwise double sum stands, chunked on both
    sides so no block larger than 2048 x 2048 is ever materialized.

    `s^T A^-1 s` is taken through the group's shared eigendecomposition
    (never a Cholesky solve of A_c): v_k is a difference of two nearly
    equal quantities, and using the same factorisation of A_c^-1 that
    `sigma_c` was built from on both sides of the subtraction keeps the
    cancellation clean.

    Returns 0.0 for a group of size <= 1, and 0.0 for every group when
    `model.constant_path[coordinate]` is true. Clipped at 0 from below.

    With `with_mean=True` (the joint path's B4 measured check,
    spec/method_notes.md section 6), returns `(v_k, u_k)` instead, u_k =
    mean(Sigma_k) itself, read from the same `mean_Sigma` this function
    already computes for the v_k subtraction -- not a second pass. u_k
    is 0.0 wherever v_k is (group size <= 1, or the constant path). The
    default path's arithmetic and return are unchanged.
    """
    c = coordinate
    n_groups = len(groups)
    out = np.zeros(n_groups, dtype=float)
    out_mean = np.zeros(n_groups, dtype=float) if with_mean else None
    if model.constant_path[c]:
        return (out, out_mean) if with_mean else out

    terms = _point_terms(model, Z)
    Zw_full = terms.Zw
    d_z = Zw_full.shape[1]
    R_full = terms.R[c]
    sigma_c = np.asarray(sigma_c, dtype=float)

    s2_c = float(model.s2[c])
    g_chol_c = model.g_chol[c]
    centers_c = model.centers[c]
    M = centers_c.shape[0]
    V_K = model.k_eigvec[c]
    w_c = 1.0 / (model.k_eigval[c] + float(model.lam[c]) + float(model.jitter[c]))

    is_local = model.gpwidth == 'local'
    if is_local:
        c_val = float(model.c[c])
        h_col = c_val * model.h_design[c]
        bmu = model.bmu
        ell_c = None
    else:
        ell_c = float(model.width[c])
        c_val = h_col = bmu = None

    # A one-dimensional design takes SS_k from the sorted running sums
    # instead of the pairwise double sum, but only for the stationary
    # kernel: under 'local' the pairwise path always stands.
    one_dim = (d_z == 1) and not is_local

    for gi, idx in enumerate(groups):
        idx = np.asarray(idx)
        n_k = idx.size
        if n_k <= 1:
            continue

        Zw_k = Zw_full[idx]

        mean_diag = float(np.mean(sigma_c[idx] ** 2))

        s_vec = np.zeros(M, dtype=float)
        SS_k = _matern32_self_sum_1d(Zw_k[:, 0], ell_c) if one_dim else 0.0
        ell_row_bin = c_val * model.h[bmu[idx]] if is_local else None

        for start in range(0, n_k, _BPV_CHUNK):
            sl = slice(start, start + _BPV_CHUNK)
            Zc = Zw_k[sl]

            if is_local:
                Kc = _matern32_nonstationary(cdist(Zc, centers_c), ell_row_bin[sl], h_col, d_z)
            else:
                Kc = _matern32(cdist(Zc, centers_c), ell_c)  # (nb, M)

            s_vec += Kc.sum(axis=0)

            if one_dim:
                continue
            for start2 in range(0, n_k, _BPV_CHUNK):
                sl2 = slice(start2, start2 + _BPV_CHUNK)
                Dcc = cdist(Zc, Zw_k[sl2])
                if is_local:
                    Kcc = _matern32_nonstationary(Dcc, ell_row_bin[sl], ell_row_bin[sl2], d_z)
                else:
                    Kcc = _matern32(Dcc, ell_c)
                SS_k += float(Kcc.sum())

        R_vec = R_full[idx].sum(axis=0)
        Vt_s = s_vec @ V_K
        quad_A = float(np.sum(Vt_s ** 2 * w_c))
        Ginv_R = cho_solve(g_chol_c, R_vec)
        mean_Sigma = (s2_c / (n_k ** 2)) * (SS_k - quad_A + R_vec @ Ginv_R)

        out[gi] = max(mean_diag - mean_Sigma, 0.0)
        if with_mean:
            out_mean[gi] = mean_Sigma

    if with_mean:
        return out, out_mean

    return out


# --- A17: the stencils update the influence model, and stage 1 itself is a
# set-observation model under 'stencils' (spec/QIJ_A17_stencil_update.md
# section 10, final; supersedes sections 3, 8 and 9 where they differ).
# Every row of the design -- a prototype's own receptive field, or an
# output's own current leaf -- is a SET (a point-index array): a stencil, or
# the prototype survey itself, is an observation of the mean of psi_c over
# that set's own points, never a value at a point. `fit_rows_model` is stage
# 1's own fit under 'stencils' (section 10.1); `core.stencils` owns the round
# schedule, the "one full refit" after a_c, and every output's own
# noise/response bookkeeping. Two exact linear dependences are removed
# structurally, before any solve, rather than cut by rank afterward (section
# 10.2): a split's parent row is replaced by its children's rows
# (`core.stencils`'s own concern); and, since every partition of the N
# points implies the same global-mean functional, the row of the LARGEST
# set is dropped from each partition (the receptive-field partition, here;
# each output's own leaf partition, `core.stencils`'s). The posterior solve
# is Cholesky with the ridge ladder `fit_influence_model` already uses,
# never an eigendecomposition-based rank cut.

_SET_CHUNK = 2048  # row chunk for one set's own kernel-to-N-points column (E4)


def _set_kernel_column(
    Zw: np.ndarray, idx_S: np.ndarray, gpwidth: str, ell_or_c: float,
    h_full: Optional[np.ndarray], bmu: Optional[np.ndarray], d_z: int,
) -> np.ndarray:
    """k(x_i, S) for every one of the N rows of `Zw` and one set S (its
    member indices `idx_S`), chunked over the N rows -- a K.A product,
    A a single (n_S, 1) column of 1/n_S (spec section 8 item 2), never
    an N x N array (E4)."""
    N = Zw.shape[0]
    Zs = Zw[idx_S]
    out = np.empty(N, dtype=float)
    if gpwidth == 'local':
        ell_col = ell_or_c * h_full[bmu[idx_S]]
    for start in range(0, N, _SET_CHUNK):
        sl = slice(start, start + _SET_CHUNK)
        if gpwidth == 'global':
            Kc = _matern32(cdist(Zw[sl], Zs), ell_or_c)
        else:
            ell_row = ell_or_c * h_full[bmu[sl]]
            Kc = _matern32_nonstationary(cdist(Zw[sl], Zs), ell_row, ell_col, d_z)
        out[sl] = Kc.mean(axis=1)
    return out


def _set_basis_row(Zw: np.ndarray, idx_S: np.ndarray, m_g: int) -> np.ndarray:
    """h(S) = (1/n_S) sum_{i in S} h(x_i) (spec section 3): the group's
    own basis, averaged over the set's members, never the basis of the
    set's mean position (h is quadratic under `gptrend='quadratic'`)."""
    return _basis(Zw[idx_S], m_g).mean(axis=0)


def assemble_K_ss(Kx: np.ndarray, all_idx: Sequence[np.ndarray]) -> np.ndarray:
    """
    K_ss (n, n) = k(S, S') for the design's own rows `all_idx` (one
    point-index array per row), read back from the already-formed `Kx`
    (N, n): row a's own entry against every column is the mean of
    `Kx`'s rows at row a's own members (spec section 8 item 2's K.A
    construction, applied a second time to `Kx` itself rather than a
    fresh pairwise pass), symmetrized once since the two ways of
    deriving one off-diagonal entry agree only to rounding. No new
    kernel evaluation: pure numpy indexing over `Kx`, so it is always
    cheap to rebuild every round even when `Kx`'s own columns are
    cached (spec section 9 item 2's caching ruling is about the K.A
    kernel evaluations that built `Kx`, not this assembly).
    """
    K_ss = np.stack([Kx[idx].mean(axis=0) for idx in all_idx], axis=0)
    return 0.5 * (K_ss + K_ss.T)


def _cholesky_ridge_step(A: np.ndarray, K: np.ndarray, M_used: int):
    """`_cholesky_with_jitter`, plus which ladder position it used (0-3),
    for the products spec section 10.2/10.3 ask for: `_cholesky_with_
    jitter` itself is untouched (still exactly what `fit_influence_
    model`'s own final solve calls), so the ladder step is read back
    from its own returned jitter value against the same base it would
    have computed, rather than changed to return one itself."""
    chol, jit = _cholesky_with_jitter(A, K, M_used)
    base = 1e-10 * float(np.trace(K)) / M_used
    if jit == 0.0:
        step = 0
    elif jit == base:
        step = 1
    elif jit == base * 10.0:
        step = 2
    else:
        step = 3
    return chol, step


def _rows_kernel_matrix(
    Zw: np.ndarray, all_idx: Sequence[np.ndarray], gpwidth: str, param: float,
    h_full: Optional[np.ndarray], bmu: Optional[np.ndarray], d_z: int,
) -> np.ndarray:
    """K(S, S') for every row in `all_idx` at kernel parameter `param`
    (spec section 10.1): the K.A construction (`_set_kernel_column`
    against every one of the N draw points, `assemble_K_ss` to read the
    row-against-row values back), formed FRESH -- the hyperparameter
    search calls this once per candidate, since the covariance itself
    depends on the width/c being searched over."""
    return assemble_K_ss(rows_kernel_columns(Zw, all_idx, gpwidth, param, h_full, bmu, d_z), all_idx)


_KA_CHUNK = 1024  # point rows per chunk of the one N x N kernel pass in `rows_kernel_columns`


def _averaging_matrix(N: int, all_idx: Sequence[np.ndarray]):
    """A (N, n), sparse: column S holds 1/n_S on S's members."""
    rows = np.concatenate([np.asarray(idx) for idx in all_idx])
    cols = np.concatenate([np.full(len(idx), s) for s, idx in enumerate(all_idx)])
    vals = np.concatenate([np.full(len(idx), 1.0 / len(idx)) for idx in all_idx])
    return csr_matrix((vals, (rows, cols)), shape=(N, len(all_idx)))


def rows_kernel_columns(
    Zw: np.ndarray, all_idx: Sequence[np.ndarray], gpwidth: str, param: float,
    h_full: Optional[np.ndarray], bmu: Optional[np.ndarray], d_z: int,
) -> np.ndarray:
    """Kx = K . A (N, n) in one chunked pass over the N x N point kernel:
    each chunk of rows is evaluated against all N points once and
    averaged onto every set by the sparse A (E4: never an N x N array),
    instead of one kernel call per set. Equals stacking
    `_set_kernel_column` over `all_idx`, to rounding."""
    N = Zw.shape[0]
    At = _averaging_matrix(N, all_idx).T.tocsr()
    Kx = np.empty((N, len(all_idx)), dtype=float)
    if gpwidth == 'local':
        ell_all = param * h_full[bmu]
    for start in range(0, N, _KA_CHUNK):
        sl = slice(start, start + _KA_CHUNK)
        D = cdist(Zw[sl], Zw)
        if gpwidth == 'global':
            Kc = _matern32(D, param)
        else:
            Kc = _matern32_nonstationary(D, ell_all[sl], ell_all, d_z)
        Kx[sl] = (At @ Kc.T).T
    return Kx


def _blas_threads():
    """All cores for the row model's dense algebra, which runs in the
    parent process while the evaluation pool is idle (the parent is
    pinned to one BLAS thread for the pool's sake)."""
    return threadpool_limits(limits=os.cpu_count() or 1, user_api='blas')


def _chol_quad_diag(chol, B: np.ndarray) -> np.ndarray:
    """diag(B M^{-1} B^T) for M = LL^T given as `cho_factor`'s (c, lower):
    one triangular solve and a column sum of squares, half the work of
    `cho_solve` followed by an einsum."""
    c, lower = chol
    W = solve_triangular(c, B.T, lower=lower, trans='N' if lower else 'T', check_finite=False)
    return np.einsum('ij,ij->j', W, W)


def _with_blas_threads(fn):
    """Run `fn` under `_blas_threads()`."""
    import functools

    @functools.wraps(fn)
    def wrapped(*args, **kwargs):
        with _blas_threads():
            return fn(*args, **kwargs)
    return wrapped


def _rows_grid_candidate(
    log_param: float, gpwidth: str, Zw: np.ndarray, all_idx: Sequence[np.ndarray],
    h_full: Optional[np.ndarray], bmu: Optional[np.ndarray], d_z: int,
    W_g: np.ndarray, tau_g: float, M_minus_m: int, cols_g: Sequence[int],
    psi_proj: Dict[int, np.ndarray], n_c2_g: Dict[int, float],
) -> Tuple[dict, float]:
    """One width/c-grid candidate's outer objective over the ROW
    covariance (spec section 10.1): mirrors `_width_grid_candidate`
    exactly from the kernel onward (same QR-projected eigh, same
    per-coordinate profiled-lam REML, same `_lambda_floor`), differing
    only in how K is formed (`_rows_kernel_matrix` in place of a direct
    point kernel). Run serially here (`fit_rows_model` does not batch
    this on a pool) -- 10.6: not answered by the document; the ported
    SEARCH SHAPE stands, its pool batching does not, and is noted here
    rather than built."""
    t0 = time.perf_counter()
    param = math.exp(log_param)
    K = _rows_kernel_matrix(Zw, all_idx, gpwidth, param, h_full, bmu, d_z)

    C = W_g.T @ K
    E = C @ W_g
    E[np.diag_indices_from(E)] += tau_g
    WC = W_g @ C
    A = K - WC - WC.T + W_g @ (E @ W_g.T)
    Lambda, V = np.linalg.eigh(A)
    Lambda = np.maximum(Lambda[:M_minus_m], 0.0)
    V = V[:, :M_minus_m]

    per_c: Dict[int, dict] = {}
    total_nll = 0.0
    for c in cols_g:
        z = V.T @ psi_proj[c]

        def inner(log_lam: float, _z=z, _Lambda=Lambda) -> float:
            denom = _Lambda + math.exp(log_lam)
            s2_val = np.sum(_z ** 2 / denom) / M_minus_m
            s2_val = max(s2_val, 1e-300)
            return (M_minus_m / 2.0) * math.log(s2_val) + 0.5 * np.sum(np.log(denom))

        lam_floor_c = _lambda_floor(z, Lambda, M_minus_m, n_c2_g[c])
        lo_bound = max(math.log(lam_floor_c), _LOG_LAM_LO)
        if lo_bound >= _LOG_LAM_HI:
            log_lam_star = _LOG_LAM_HI
            nll_c = inner(log_lam_star)
        else:
            ir = minimize_scalar(inner, bounds=(lo_bound, _LOG_LAM_HI), method='bounded')
            log_lam_star = float(ir.x)
            nll_c = float(ir.fun)
        per_c[c] = dict(log_lam=log_lam_star, nll=nll_c, lam_floor=lam_floor_c)
        total_nll += nll_c

    entry = dict(log_param=log_param, param=param, K=K, per_c=per_c, nll=total_nll)
    return entry, time.perf_counter() - t0


def _rows_group_fit(
    Zw: np.ndarray, gpwidth: str, h_full: Optional[np.ndarray], bmu: Optional[np.ndarray],
    d_z: int, all_idx: Sequence[np.ndarray], Hb_g: np.ndarray, m_g: int, cols_g: Sequence[int],
    response: Dict[int, np.ndarray], param_min: float, param_max: float,
    n_c2_g: Dict[int, float],
) -> Tuple[dict, dict]:
    """
    One group's full hyperparameter search and final per-coordinate
    solve over its own ROW design (spec section 10.1): the 5-point
    log-spaced grid plus one bounded refinement `fit_influence_model`'s
    own width search uses (`_rows_grid_candidate`), then, at the chosen
    width/c, each coordinate's own lam_c (from the search) and a
    Cholesky-with-ridge-ladder final solve (section 10.2 -- no eigh,
    no rank cut; `noise[c]` on each row is strictly positive by
    construction, the caller's job). Returns (per_coordinate dict,
    shared dict with `param`, `K` and the group's own `formation_wall_
    time` -- the search's own total wall time, section 10.3).
    """
    n_g = len(all_idx)
    M_minus_m = n_g - m_g

    Qfull, _R = np.linalg.qr(Hb_g, mode='complete')
    Q = Qfull[:, m_g:]
    W_g = Qfull[:, :m_g]
    tau_g = 2.0 * float(n_g)
    psi_proj = {c: Q @ (Q.T @ response[c]) for c in cols_g}

    grid_log_param = np.linspace(math.log(param_min), math.log(param_max), _N_WIDTH_GRID)
    t_formation = 0.0
    trace = []
    for lp in grid_log_param:
        entry, wall = _rows_grid_candidate(
            float(lp), gpwidth, Zw, all_idx, h_full, bmu, d_z, W_g, tau_g, M_minus_m,
            cols_g, psi_proj, n_c2_g)
        trace.append(entry)
        t_formation += wall

    grid_nlls = [t['nll'] for t in trace]
    best_idx = int(np.argmin(grid_nlls))
    if best_idx == _N_WIDTH_GRID - 1:
        lo = hi = None
    elif best_idx == 0:
        lo, hi = grid_log_param[0], grid_log_param[1]
    else:
        lo, hi = grid_log_param[best_idx - 1], grid_log_param[best_idx + 1]
    if lo is not None and hi > lo:
        def outer_obj(log_param: float) -> float:
            entry, wall = _rows_grid_candidate(
                log_param, gpwidth, Zw, all_idx, h_full, bmu, d_z, W_g, tau_g, M_minus_m,
                cols_g, psi_proj, n_c2_g)
            trace.append(entry)
            nonlocal_wall[0] += wall
            return entry['nll']
        nonlocal_wall = [0.0]
        minimize_scalar(outer_obj, bounds=(float(lo), float(hi)), method='bounded')
        t_formation += nonlocal_wall[0]

    best = min(trace, key=lambda t: t['nll'])
    param_val = best['param']
    K = best['K']

    per_coord = {}
    for c in cols_g:
        lam_c = math.exp(best['per_c'][c]['log_lam'])
        noise = np.full(n_g, lam_c, dtype=float)
        A = K + np.diag(noise)
        chol, ridge_step = _cholesky_ridge_step(A, K, n_g)
        AinvHb = cho_solve(chol, Hb_g)
        G = Hb_g.T @ AinvHb
        G_chol = cho_factor(G, lower=True)
        resp_c = response[c]
        Ainv_resp = cho_solve(chol, resp_c)
        u_vec = Hb_g.T @ Ainv_resp
        beta_c = cho_solve(G_chol, u_vec)
        alpha_c = Ainv_resp - AinvHb @ beta_c
        resid_quad = float(resp_c @ Ainv_resp - u_vec @ beta_c)
        s2_c = max(resid_quad, 0.0) / M_minus_m
        per_coord[c] = dict(alpha=alpha_c, beta=beta_c, chol=chol, g_chol=G_chol,
                             ainv_hb=AinvHb, s2=s2_c, lam=lam_c, ridge_step=ridge_step)

    shared = dict(param=param_val, K=K, formation_wall_time=t_formation, all_idx=list(all_idx),
                  Hb=Hb_g, m=m_g)
    return per_coord, shared


@_with_blas_threads
def fit_rows_model(
    Z: np.ndarray, xvq, I_proto: np.ndarray, theta_Q: np.ndarray, eta: float,
    gptrend: str = 'affine', gpwidth: str = 'global',
    extra_idx: Optional[Sequence[np.ndarray]] = None,
    extra_response: Optional[np.ndarray] = None,
    extra_noise_ratio: Optional[np.ndarray] = None,
) -> Tuple[dict, dict, dict, dict]:
    """
    The set-observation influence model, from its first fit (spec
    section 10.1): every row is a receptive field (its own member
    points, never the prototype's position), or -- when `extra_idx` is
    given (the 'one full refit' after a_c, called from `core.stencils`)
    -- also an output's own current leaf, its declared noise-to-signal
    ratio `extra_noise_ratio`'s own column (already floored, already
    positive: section 10.2's own rule is the caller's to keep). Section
    10.2b's structural redundancy removal (the largest receptive field
    row dropped from each group) happens here, once, before any solve;
    a caller drops each output's own largest leaf row from `extra_idx`
    itself before calling.

    Grouping mirrors `fit_influence_model`'s own (coordinates sharing
    an identical finite `I_proto` pattern -- spec/QIJ_A17_stencil_
    update.md section 9 item 1's per-output missing-row rule, not
    superseded by section 10); a coordinate whose own spread over its
    finite receptive fields is degenerate is `constant_path`, exactly
    the stage-1 rule (method_notes section 3), ported since section 10
    does not address it (10.6).

    Returns (per_coordinate, shared_by_group, constant, aux):
    per_coordinate[c] is `_rows_group_fit`'s own per-coordinate dict;
    shared_by_group[gi] is its own shared dict, plus `cols` (the
    group's own coordinates) and `ids_reduced` (the group's own live
    receptive-field ids, post section 10.2b's drop); constant[c] is
    (is_constant, const_value); aux carries `whitening`, `h_full`,
    `bmu_full`, `d_z` and `proto_idx` (id -> point-index array, every
    live receptive field, pre section 10.2b) for a caller building
    `Zw`/further rows (`core.stencils`'s own 'one full refit' and every
    later round) without re-deriving them.
    """
    p = np.asarray(xvq.p, dtype=float)
    I_proto = np.asarray(I_proto, dtype=float)
    theta_Q = np.asarray(theta_Q, dtype=float)
    M_X_used, q = I_proto.shape
    raw_centers = np.asarray(xvq.centers, dtype=float)
    d_z = raw_centers.shape[1]

    mean, transform = _whitening_from(Z)
    centers_full = (raw_centers - mean) @ transform.T
    Za = np.asarray(Z, dtype=float)
    if Za.ndim == 1:
        Za = Za.reshape(-1, 1)
    Zw = (Za - mean) @ transform.T

    if gpwidth == 'local':
        D_proto_full = cdist(centers_full, centers_full)
        h_full = _conn_spacing(xvq.conn, D_proto_full)
        bmu_full = np.asarray(xvq.bmu, dtype=np.intp)
    else:
        h_full = np.full(M_X_used, np.nan, dtype=float)
        bmu_full = None

    proto_idx: Dict[int, np.ndarray] = {j: np.where(np.asarray(xvq.bmu) == j)[0] for j in range(M_X_used)}

    finite = np.isfinite(I_proto)
    constant: Dict[int, Tuple[bool, float]] = {}
    groups: Dict[tuple, List[int]] = {}
    for c in range(q):
        idx_c = np.where(finite[:, c])[0]
        M_c = idx_c.size
        if M_c == 0:
            constant[c] = (True, 0.0)
            continue
        p_c = p[idx_c]
        mass_c = float(p_c.sum())
        psi_bar = float(np.sum(p_c * I_proto[idx_c, c]) / mass_c) if mass_c > 0.0 else 0.0
        coord_constant = M_c < 3
        if not coord_constant:
            p_c_norm = p_c / mass_c
            spread = float(np.sqrt(np.sum(p_c_norm * (I_proto[idx_c, c] - psi_bar) ** 2)))
            coord_constant = spread <= 1e-10 * max(abs(float(theta_Q[c])), 1e-300)
        if coord_constant:
            constant[c] = (True, psi_bar)
        else:
            constant[c] = (False, 0.0)
            groups.setdefault(tuple(idx_c.tolist()), []).append(c)

    m_full = (d_z + 1) if gptrend == 'affine' else (1 + d_z + d_z * (d_z + 1) // 2)
    delta_f = forward_step(eta)

    per_coordinate: Dict[int, dict] = {}
    shared_by_group: Dict[int, dict] = {}
    for gi, (idx_key, cols_g) in enumerate(groups.items()):
        idx_g = list(idx_key)
        # Section 10.2b: the group's own largest receptive field row is
        # structurally redundant (its partition's mass-weighted mean is
        # the same global-mean functional every partition implies) and
        # is dropped before any solve; an empty one (no data point has
        # that BMU) was never a row to begin with (ruling 9.1).
        idx_g = [j for j in idx_g if proto_idx[j].size > 0]
        if idx_g:
            drop_j = max(idx_g, key=lambda j: p[j])
            idx_g = [j for j in idx_g if j != drop_j]
        all_idx = [proto_idx[j] for j in idx_g] + list(extra_idx or [])
        n_g = len(all_idx)
        m_g = 1 if n_g <= m_full + 1 else m_full
        Hb_g = np.stack([_set_basis_row(Zw, idx, m_g) for idx in all_idx], axis=0)

        response = {}
        for c in cols_g:
            proto_resp = np.array([I_proto[j, c] for j in idx_g])
            if extra_idx:
                extra_resp_c = np.asarray(extra_response)[:, c]
                response[c] = np.concatenate([proto_resp, extra_resp_c])
            else:
                response[c] = proto_resp

        centers_g = centers_full[idx_g]
        conn_g = xvq.conn[idx_g, :][:, idx_g]
        D_full_g = cdist(centers_g, centers_g)
        ell_min, ell_max = _length_scale_bounds(conn_g, D_full_g)
        if gpwidth == 'global':
            param_min, param_max = ell_min, ell_max
        else:
            h_design_g = h_full[idx_g]
            param_min = ell_min / float(np.median(h_design_g))
            param_max = ell_max / float(np.min(h_design_g))

        p_g = p[idx_g]
        t_g = delta_f * p_g / (1.0 - p_g)
        inv_t2_median_g = float(np.median(1.0 / t_g ** 2))
        n_c2_g = {c: 2.0 * eta ** 2 * float(theta_Q[c]) ** 2 * inv_t2_median_g for c in cols_g}

        pc, shared = _rows_group_fit(
            Zw, gpwidth, h_full, bmu_full, d_z, all_idx, Hb_g, m_g, cols_g, response,
            param_min, param_max, n_c2_g,
        )
        shared['cols'] = list(cols_g)
        shared['ids_reduced'] = idx_g
        for c in cols_g:
            per_coordinate[c] = pc[c]
        shared_by_group[gi] = shared

    aux = dict(whitening=(mean, transform), h_full=h_full, bmu_full=bmu_full, d_z=d_z,
               proto_idx=proto_idx)
    return per_coordinate, shared_by_group, constant, aux


@_with_blas_threads
def rows_solve(
    Kx: np.ndarray, all_idx: Sequence[np.ndarray], Hb: np.ndarray, n_proto: int, m_g: int,
    cols_g: Sequence[int], response: Dict[int, np.ndarray], lam_by_c: Dict[int, float],
    leaf_noise_ratio_by_c: Dict[int, np.ndarray],
) -> dict:
    """
    Re-solve every coordinate in `cols_g` at a FIXED width/c (already
    baked into the given `Kx`, spec section 10.1: the hyperparameter
    search runs once, after a_c) and FIXED lam_c (`lam_by_c`), over the
    CURRENT `all_idx`/`Kx`/`Hb` -- a caller (`core.stencils`) forms
    these from its own cached per-row columns (section 10.3: a
    receptive field's column is a per-draw invariant, formed once and
    reused; a leaf's is formed at its creation and dropped at its
    split), so no kernel evaluation happens here, only the assembly
    (`assemble_K_ss`) and the final solve. `all_idx`'s first `n_proto`
    rows are the group's own live receptive fields (lam_c on their
    diagonal); the rest are leaves (`leaf_noise_ratio_by_c[c]`'s own
    declared ratio, already floored positive by the caller, section
    10.2). No eigh, no rank cut -- Cholesky with the ridge ladder
    (`_cholesky_ridge_step`), exactly `_rows_group_fit`'s own final
    solve.
    """
    K = assemble_K_ss(Kx, all_idx)
    n = len(all_idx)
    M_minus_m = n - m_g

    per_coord = {}
    for c in cols_g:
        noise = np.concatenate([np.full(n_proto, lam_by_c[c]), leaf_noise_ratio_by_c[c]])
        A = K + np.diag(noise)
        chol, ridge_step = _cholesky_ridge_step(A, K, n)
        AinvHb = cho_solve(chol, Hb)
        G = Hb.T @ AinvHb
        G_chol = cho_factor(G, lower=True)
        resp_c = response[c]
        Ainv_resp = cho_solve(chol, resp_c)
        u_vec = Hb.T @ Ainv_resp
        beta_c = cho_solve(G_chol, u_vec)
        alpha_c = Ainv_resp - AinvHb @ beta_c
        resid_quad = float(resp_c @ Ainv_resp - u_vec @ beta_c)
        s2_c = max(resid_quad, 0.0) / M_minus_m if M_minus_m > 0 else 0.0
        per_coord[c] = dict(alpha=alpha_c, beta=beta_c, chol=chol, g_chol=G_chol,
                             ainv_hb=AinvHb, s2=s2_c, lam=lam_by_c[c], ridge_step=ridge_step)

    return per_coord


@_with_blas_threads
def rows_point_terms(
    per_coordinate: Dict[int, dict], cols_g: Sequence[int], Kx: np.ndarray, Zw: np.ndarray,
    m_g: int,
) -> Tuple[Dict[int, np.ndarray], Dict[int, np.ndarray], Dict[int, np.ndarray]]:
    """
    psi0 and sigma at every one of the N draw points `Zw`, for every
    coordinate in `cols_g`, from each coordinate's own solve
    (`per_coordinate[c]`) and the ALREADY-FORMED `Kx` (N, n): a caller
    (`core.stencils`) forms `Kx` from its own cached per-row columns
    (spec section 10.3 -- a receptive field's column is a per-draw
    invariant, a leaf's is formed at its creation and dropped at its
    split), so no kernel evaluation happens here. Standard posterior
    mean/variance from a Cholesky solve, never eigh (section 10.2).
    Returns (psi0, sigma, R) dicts keyed by coordinate, R the (N, m_g)
    affine-mean residual `rows_bin_posterior_variance` needs.
    """
    N = Zw.shape[0]

    psi0_out = {c: np.empty(N, dtype=float) for c in cols_g}
    sigma_out = {c: np.zeros(N, dtype=float) for c in cols_g}
    R_out = {c: np.empty((N, m_g), dtype=float) for c in cols_g}

    batch = _UNCERTAINTY_BATCH_CAP
    for start in range(0, N, batch):
        sl = slice(start, start + batch)
        Hc = _basis(Zw[sl], m_g)
        Kc = Kx[sl]
        for c in cols_g:
            st = per_coordinate[c]
            psi0_out[c][sl] = Hc @ st['beta'] + Kc @ st['alpha']
            term1 = _chol_quad_diag(st['chol'], Kc)
            Rc = Hc - Kc @ st['ainv_hb']
            R_out[c][sl] = Rc
            term2 = _chol_quad_diag(st['g_chol'], Rc)
            sigma2 = st['s2'] * (1.0 - term1 + term2)
            sigma2 = np.maximum(sigma2, 0.0)
            sigma_out[c][sl] = np.sqrt(sigma2)

    return psi0_out, sigma_out, R_out


@_with_blas_threads
def rows_bin_posterior_variance(
    per_coordinate: Dict[int, dict], coordinate: int, groups: Sequence[np.ndarray],
    sigma_c: np.ndarray, Zw: np.ndarray, R_full: np.ndarray, Kx: np.ndarray,
    gpwidth: str, param: float, h_full: Optional[np.ndarray], bmu: Optional[np.ndarray],
) -> np.ndarray:
    """
    v_k for `coordinate`'s row-observation posterior, at each group in
    `groups` (point-index arrays into the N draw points): the same
    formula `bin_posterior_variance` implements. SS_k (the bin's own
    raw point-pair kernel sum) is a point-level quantity, unaffected by
    the row representation, and computed exactly as `bin_posterior_
    variance`'s own chunked pass; `s_vec` is a row-sum of `Kx` (already
    formed by `rows_point_terms`, no new kernel evaluation); `R_vec` is
    a row-sum of `R_full`. Both quadratic forms go through this
    coordinate's own Cholesky solve (`st['chol']`/`st['g_chol']`) --
    never a shared eigendecomposition or a rank cut (section 10.2).
    """
    st = per_coordinate[coordinate]
    n_groups = len(groups)
    out = np.zeros(n_groups, dtype=float)
    d_z = Zw.shape[1]
    is_local = gpwidth == 'local'
    one_dim = (d_z == 1) and not is_local
    s2_c = st['s2']

    for gi, idx in enumerate(groups):
        idx = np.asarray(idx)
        n_k = idx.size
        if n_k <= 1:
            continue
        Zw_k = Zw[idx]
        mean_diag = float(np.mean(sigma_c[idx] ** 2))

        SS_k = _matern32_self_sum_1d(Zw_k[:, 0], param) if one_dim else 0.0
        if not one_dim:
            for start in range(0, n_k, _BPV_CHUNK):
                sl = slice(start, start + _BPV_CHUNK)
                Zc = Zw_k[sl]
                for start2 in range(0, n_k, _BPV_CHUNK):
                    sl2 = slice(start2, start2 + _BPV_CHUNK)
                    Dcc = cdist(Zc, Zw_k[sl2])
                    if is_local:
                        ell_row_bin = param * h_full[bmu[idx[sl]]]
                        ell_row_bin2 = param * h_full[bmu[idx[sl2]]]
                        Kcc = _matern32_nonstationary(Dcc, ell_row_bin, ell_row_bin2, d_z)
                    else:
                        Kcc = _matern32(Dcc, param)
                    SS_k += float(Kcc.sum())

        s_vec = Kx[idx].sum(axis=0)
        R_vec = R_full[idx].sum(axis=0)

        Ainv_s = cho_solve(st['chol'], s_vec)
        quad_A = float(s_vec @ Ainv_s)
        Ginv_R = cho_solve(st['g_chol'], R_vec)
        mean_Sigma = (s2_c / (n_k ** 2)) * (SS_k - quad_A + R_vec @ Ginv_R)
        out[gi] = max(mean_diag - mean_Sigma, 0.0)

    return out
