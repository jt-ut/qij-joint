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
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.linalg import LinAlgError, cho_factor, cho_solve
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

# The per-draw kernel-row cache's memory budget: the (N, M_g) kernel
# matrices of every coordinate group together may occupy at most this
# many bytes, or none of them is cached and every consumer re-forms its
# rows in chunks instead -- bit-identical either way (spec section 3).
_KERNEL_CACHE_BYTES = 256 * 1024 ** 2


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
            row sum of it); None on the constant path.
    K       per coordinate, (N, M_c) kernel rows k(x_i, w_j) against
            that coordinate's own design -- the SAME array object for
            every coordinate sharing a design; None when the total
            would exceed `_KERNEL_CACHE_BYTES` (re-formed in chunks
            instead, to bit-identical values) and on the constant path.
    """

    Z: np.ndarray
    Zw: np.ndarray
    psi0: np.ndarray
    sigma: np.ndarray
    R: List[Optional[np.ndarray]]
    K: List[Optional[np.ndarray]]


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


def fit_influence_model(
    Z: np.ndarray, xvq, I_proto: np.ndarray, theta_Q: np.ndarray, eta: float,
    gptrend: str = 'affine', gpwidth: str = 'global',
) -> InfluenceModel:
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

    for idx_key, cols_g in groups.items():
        idx_g = np.asarray(idx_key, dtype=np.intp)
        M_g = idx_g.size
        centers_g = centers_full[idx_g]
        m_g = 1 if M_g <= m_full + 1 else m_full
        Hb_g = _basis(centers_g, m_g)

        D_full_g = cdist(centers_g, centers_g)
        conn_g = xvq.conn[idx_g, :][:, idx_g]
        ell_min, ell_max = _length_scale_bounds(conn_g, D_full_g)

        # The outer search runs over ell directly under 'global'
        # (`kernel_at` an exact alias of `_matern32`), or over c under
        # 'local': its bounds carried from ell_min,
        # ell_max via this group's own prototypes' CONN spacing, so
        # c_min is about 1 (spec section 3).
        if gpwidth == 'global':
            param_min, param_max = ell_min, ell_max
            h_design_g = None

            def kernel_at(param: float, _D=D_full_g) -> np.ndarray:
                return _matern32(_D, param)
        else:
            h_design_g = h_full[idx_g]
            param_min = ell_min / float(np.median(h_design_g))
            param_max = ell_max / float(np.min(h_design_g))

            def kernel_at(param: float, _D=D_full_g, _h=h_design_g, _dz=d_z) -> np.ndarray:
                ell_vec = param * _h
                return _matern32_nonstationary(_D, ell_vec, ell_vec, _dz)

        grid_log_param = np.linspace(math.log(param_min), math.log(param_max), _N_WIDTH_GRID)

        Qfull, _R = np.linalg.qr(Hb_g, mode='complete')
        Q = Qfull[:, m_g:]
        M_minus_m = M_g - m_g

        Qt_psi = {c: Q.T @ I_proto[idx_g, c] for c in cols_g}

        # A_param = (I-P) K_param (I-P) + tau*P, P = W W^T, gives the
        # same spectrum as Q^T K_param Q in O(m_g M_g^2) rather than
        # two M_g x M_g matmuls, with the m_g structural eigenvalues
        # shifted above the kernel's own (derivation: spec/method_notes.md
        # section 3).
        W_g = Qfull[:, :m_g]
        tau_g = 2.0 * float(M_g)
        psi_proj = {c: Q @ Qt_psi[c] for c in cols_g}

        # Declared noise floor, per coordinate in this group: n_c^2 =
        # 2*eta^2*theta_Q,c^2 * median_j(1/t_j^2) over the group's own
        # finite design, independent of the width so computed once per group.
        p_g = p[idx_g]
        t_g = delta_f * p_g / (1.0 - p_g)
        inv_t2_median_g = float(np.median(1.0 / t_g ** 2))
        n_c2_g = {c: 2.0 * eta ** 2 * float(theta_Q[c]) ** 2 * inv_t2_median_g for c in cols_g}

        # Joint outer search over log(width), within this group: one
        # kernel and one eigendecomposition of Q^T K Q per candidate,
        # shared across the group; each coordinate profiles its own
        # lam_c from that shared eigendecomposition, and the outer
        # objective is the SUM of the group's per-coordinate profiled
        # restricted NLLs.
        t_shared0 = time.perf_counter()
        trace: List[dict] = []

        def outer_obj(log_param: float, _trace=trace, _cols_g=cols_g,
                       _psi_proj=psi_proj, _W=W_g, _tau=tau_g,
                       _M_minus_m=M_minus_m, _n_c2=n_c2_g, _kernel_at=kernel_at) -> float:
            param = math.exp(log_param)
            K = _kernel_at(param)
            C = _W.T @ K                      # (m_g, M_g)
            E = C @ _W                        # (m_g, m_g)
            E[np.diag_indices_from(E)] += _tau
            WC = _W @ C                       # (M_g, M_g)
            A = K - WC - WC.T + _W @ (E @ _W.T)
            Lambda, V = np.linalg.eigh(A)
            Lambda = np.maximum(Lambda[:_M_minus_m], 0.0)
            V = V[:, :_M_minus_m]

            per_c: Dict[int, dict] = {}
            total_nll = 0.0
            for c in _cols_g:
                z = V.T @ _psi_proj[c]

                def inner(log_lam: float, _z=z, _Lambda=Lambda) -> float:
                    denom = _Lambda + math.exp(log_lam)
                    s2_val = np.sum(_z ** 2 / denom) / _M_minus_m
                    s2_val = max(s2_val, 1e-300)
                    return (_M_minus_m / 2.0) * math.log(s2_val) + 0.5 * np.sum(np.log(denom))

                # The search for lam_c is bounded below by lam_floor,c
                # at THIS candidate width (Lambda, z both depend on it).
                lam_floor_c = _lambda_floor(z, Lambda, _M_minus_m, _n_c2[c])
                lo_bound = max(math.log(lam_floor_c), _LOG_LAM_LO)
                if lo_bound >= _LOG_LAM_HI:
                    # The declared floor already pins lam_c at the
                    # ceiling: no interval left to search.
                    log_lam_star = _LOG_LAM_HI
                    nll_c = inner(log_lam_star)
                else:
                    ir = minimize_scalar(inner, bounds=(lo_bound, _LOG_LAM_HI), method='bounded')
                    log_lam_star = float(ir.x)
                    nll_c = float(ir.fun)
                per_c[c] = dict(log_lam=log_lam_star, nll=nll_c, lam_floor=lam_floor_c)
                total_nll += nll_c

            _trace.append(dict(log_param=log_param, param=param, K=K, per_c=per_c, nll=total_nll))
            return total_nll

        for lp in grid_log_param:
            outer_obj(float(lp))

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

    return model


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
    model and keyed on the identity of `Z`. Returns psi0, sigma, the
    per-point affine-mean residuals r_i and -- within the memory
    budget -- the kernel rows themselves, all of which `psi0`,
    `uncertainty` and `bin_posterior_variance` read instead of
    recomputing. psi0 and sigma are produced together because they
    share the kernel rows k(x_i, w_j), which every caller asks for on
    every draw. Under `gpwidth='local'` those rows come from
    `_matern32_nonstationary` at each query row's own length scale
    c*h[bmu_i] (`model.bmu`, set at fit time) against the design's
    c*h_design; under 'global' from `_matern32` at the group's shared
    ell.
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
    K_out: List[Optional[np.ndarray]] = [None] * q

    for c in range(q):
        if model.constant_path[c]:
            psi0_out[:, c] = model.const_value[c]

    groups = _coordinate_groups(model)

    # The kernel-row cache is all-or-nothing across the groups: one
    # budget, so a draw either keeps every group's rows or re-forms
    # every group's rows, and no group's numbers depend on how another
    # group's memory came out.
    cache_bytes = sum(N * model.centers[g[0]].shape[0] for g in groups) * 8
    cache_rows = cache_bytes <= _KERNEL_CACHE_BYTES

    batch = _UNCERTAINTY_BATCH_CAP
    for cols in groups:
        c0 = cols[0]
        centers_g = model.centers[c0]
        M_g = centers_g.shape[0]
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

        K_g = np.empty((N, M_g), dtype=float) if cache_rows else None
        for c in cols:
            K_out[c] = K_g

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

            if K_g is not None:
                K_g[sl] = Kc

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

    terms = _PointTerms(Z=Z, Zw=Zw, psi0=psi0_out, sigma=sigma_out,
                        R=R_out, K=K_out)
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
) -> np.ndarray:
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
    most 2048, from the cached kernel rows when available. `SS_k` is
    the one term no cache removes -- a sum of raw kernel values between
    the bin's OWN points. Under `gpwidth='global'` in one dimension,
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
    """
    c = coordinate
    n_groups = len(groups)
    out = np.zeros(n_groups, dtype=float)
    if model.constant_path[c]:
        return out

    terms = _point_terms(model, Z)
    Zw_full = terms.Zw
    d_z = Zw_full.shape[1]
    R_full = terms.R[c]
    K_full = terms.K[c]
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

            if K_full is not None:
                Kc = K_full[idx[sl]]  # (nb, M), the rows already formed
            elif is_local:
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

    return out
