"""A residual-driven insertion start for `GMM2D` (spec/QIJ_mods_waves.md
A12, round 5, the author's ruling replacing rounds 1-4's two-level
peak seeding; round 6, weight-aware; A15, a second start at a finer
resolution). `greedy_em_start(...)` and `scaled_variants(...)` are the
only public entry points, called from `gmm.py`'s multi-start
construction beside its own 20 k-means starts. This module does not
import `gmm`: the accelerated-EM and penalized-log-likelihood callables
it needs are passed in by the caller (`em_accelerated`, `penalized_ll`)
-- `gmm.py` already imports this module at load time, so the reverse
import would be circular; dependency injection avoids it without
duplicating gmm.py's EM machinery here.

One over-segmentation, `cell_factor`*K k-means cells on X (never the
weights); `gmm.py` calls this at both 50*K (the round-5 default) and
25*K (A15, a second, finer start beside it). Cells of fewer than 5
members carry no insertion candidate (a raw member count, unweighted,
per the ruling). A single component (the weighted mean and covariance
of the whole draw) is grown to K by inserting, one at a time, a new
component at the cell whose weight sum W_j most exceeds what the
CURRENT mixture already predicts there (W_j - (sum w)*A_j*f(c_j), A_j =
pi*r_j^2 the cell's own area -- geometric, from point positions alone,
so unweighted -- f the current mixture's density at the cell's centre
c_j): the clearest sign of structure the model has not yet captured.
The new component's mean and covariance come from that cell's own
members only, weighted by their own w_i (round 6: `_cell_stats`'s mean
and covariance are the cell's OWN weighted statistics, not just its
members' positions, since a weighted fit should treat every downstream
quantity by each point's own weight -- the only two terms that stay
deliberately unweighted are the >= 5 eligibility count and the
geometric area). The new component's weight is a floor-protected share
of the excess (of `sum(w)`, not `N`), and every existing weight is
scaled down to make room. Twenty accelerated-EM steps follow each
insertion before the residual is recomputed, so the model has settled
before the next cell is judged against it.

At unit weights every weighted quantity here reduces to its round-5
unweighted counterpart bit for bit: `w * X == X` and `np.cov(...,
aweights=ones)` reproduces `np.cov(...)` exactly (both verified), and
`sum(w) == N` exactly for integer-valued unit weights.

`scaled_variants` (A12 item 6) turns one start into further starts at
the same means and weights with the covariances scaled by each of the
given factors, so a compact core seeded correctly in position and
weight can also be tried seeded compactly (or loosely) in scale, through
the same two-phase screen as any other start.
"""
import warnings

import numpy as np
from scipy.cluster.vq import kmeans2

# scipy.cluster.vq.kmeans2 runs exactly this many Lloyd iterations (its
# `thresh` argument is not implemented as a convergence test); matches
# `gmm.py`'s own constant for the 20 k-means starts.
_KMEANS_ITER = 300
_DEFAULT_CELL_FACTOR = 50
_MIN_CELL_COUNT = 5
_INSERT_EM_STEPS = 20
_MIN_INSERT_SHARE = 5.0


def _cell_stats(X: np.ndarray, w: np.ndarray, centres: np.ndarray, bmu: np.ndarray, M: int):
    """Per over-segmentation cell: member count (unweighted, for the
    >= 5 eligibility rule only), weight sum, area (pi*r^2, r the median
    member distance to the centre -- geometric, unweighted), and the
    cell's OWN w-weighted member mean and covariance. Count/weight-
    sum/mean are grouped sums (`np.bincount`); area and covariance need
    a per-cell median/second-moment, so the cell loop (M small, the
    over-segmentation's own count) matches every other cell statistic
    in this package. A cell of fewer than 3 members gets area 0 and a
    zero covariance -- never read, since a candidate needs >= 5
    members."""
    count = np.bincount(bmu, minlength=M).astype(float)
    wsum = np.bincount(bmu, weights=w, minlength=M)
    sums = np.stack([np.bincount(bmu, weights=w * X[:, 0], minlength=M),
                      np.bincount(bmu, weights=w * X[:, 1], minlength=M)], axis=1)
    with np.errstate(invalid='ignore', divide='ignore'):
        mean = sums / wsum[:, None]

    order = np.argsort(bmu, kind='stable')
    bounds = np.searchsorted(bmu[order], np.arange(M + 1))
    area = np.zeros(M)
    cov = np.zeros((M, 3))
    for j in range(M):
        lo, hi = bounds[j], bounds[j + 1]
        if hi - lo < 2:
            continue
        pts = X[order[lo:hi]]
        r = float(np.median(np.linalg.norm(pts - centres[j], axis=1)))
        area[j] = np.pi * r * r
        if hi - lo >= 3:
            c = np.cov(pts.T, bias=True, aweights=w[order[lo:hi]])
            cov[j] = (c[0, 0], c[0, 1], c[1, 1])
    return count, wsum, area, mean, cov


def _mixture_density(Z: np.ndarray, pis: np.ndarray, mus: np.ndarray, Ss: np.ndarray) -> np.ndarray:
    """f(Z) = sum_k pi_k * phi_k(Z), the current mixture's density at
    query points Z (M, 2) -- the insertion rule's f(c_j). Raises
    np.linalg.LinAlgError if any Sigma_k is not PD."""
    a, b, c = Ss[:, 0], Ss[:, 1], Ss[:, 2]
    det = a * c - b * b
    if not (np.all(np.isfinite(det)) and np.all(det > 0.0)):
        raise np.linalg.LinAlgError("non-positive-definite covariance")
    inv00, inv01, inv11 = c / det, -b / det, a / det
    dx = Z[:, 0][:, None] - mus[:, 0][None, :]
    dy = Z[:, 1][:, None] - mus[:, 1][None, :]
    quad = dx * dx * inv00 + 2.0 * dx * dy * inv01 + dy * dy * inv11
    phi = np.exp(-0.5 * quad) / (2.0 * np.pi * np.sqrt(det))
    return phi @ pis


def greedy_em_start(X: np.ndarray, w: np.ndarray, K: int, seed: int,
                     Q: np.ndarray, XP: np.ndarray, Scov: np.ndarray, a_pen: float,
                     tol: float, max_iter: int, em_accelerated, penalized_ll,
                     trace: list = None, cell_factor: int = _DEFAULT_CELL_FACTOR) -> list:
    """One greedy-EM start of A12 (round 6, weight-aware), over-
    segmented at `cell_factor`*K k-means cells (A15: `gmm.py` calls this
    twice, at 50*K and at 25*K, for two independent starts). `em_
    accelerated` and `penalized_ll` are `gmm.py`'s own
    `_em_accelerated`/`_penalized_ll` (passed in, not imported, to keep
    this module free of a `gmm` dependency). `trace`, if given a list,
    gets one dict per insertion (`cell`, `mean`, `n`, `w`, `e`) for
    diagnostics -- no effect on the returned start. Returns
    `[(pis, mus, Ss)]` (K components) or `[]` if the seeding degenerates
    (a non-PD covariance anywhere along the way, or too few rows for a
    single `cell_factor`*K-cell over-segmentation)."""
    N = X.shape[0]
    W = float(w.sum())
    M = min(cell_factor * K, max(K, N - 1))
    if M < 1:
        return []
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        centres, bmu = kmeans2(X, M, iter=_KMEANS_ITER, minit='++',
                                seed=np.random.default_rng(seed))
    count, wsum, area, mean, cov = _cell_stats(X, w, centres, bmu, M)
    eligible = count >= _MIN_CELL_COUNT
    if not eligible.any():
        return []

    xbar = (w @ X) / W
    d = X - xbar
    c0 = (d * w[:, None]).T @ d / W
    pis = np.array([1.0])
    mus = xbar[None, :].copy()
    Ss = np.array([[c0[0, 0], c0[0, 1], c0[1, 1]]])

    try:
        ll0 = penalized_ll(Q, w, W, pis, mus, Ss, Scov, a_pen)
        if not np.isfinite(ll0):
            return []
        pis, mus, Ss, ll, _, _ = em_accelerated(
            Q, XP, w, W, pis, mus, Ss, ll0, tol, max_iter, Scov, a_pen, 1)

        for m in range(1, K):
            f = _mixture_density(centres, pis, mus, Ss)
            e = np.where(eligible, wsum - W * area * f, -np.inf)
            j = int(np.argmax(e))
            if not np.isfinite(e[j]):
                return []
            if trace is not None:
                trace.append(dict(cell=j, mean=mean[j].copy(), n=float(count[j]),
                                   w=float(wsum[j]), e=float(e[j])))

            new_w = max(e[j] / W, _MIN_INSERT_SHARE / W)
            pis = np.concatenate([pis * (1.0 - new_w), [new_w]])
            mus = np.concatenate([mus, mean[j][None, :]], axis=0)
            Ss = np.concatenate([Ss, cov[j][None, :]], axis=0)

            ll0 = penalized_ll(Q, w, W, pis, mus, Ss, Scov, a_pen)
            if not np.isfinite(ll0):
                return []
            pis, mus, Ss, ll, _, _ = em_accelerated(
                Q, XP, w, W, pis, mus, Ss, ll0, tol, _INSERT_EM_STEPS, Scov, a_pen, m + 1)
    except np.linalg.LinAlgError:
        return []

    return [(pis, mus, Ss)]


def scaled_variants(start: tuple, factors) -> list:
    """`start` (pis, mus, Ss) turned into one further start per factor in
    `factors`, same means and weights, covariances scaled by the factor
    (A12 item 6): a compact core seeded compactly (factor < 1) or loosely
    (factor > 1), through the same two-phase screen as `start` itself."""
    pis, mus, Ss = start
    return [(pis.copy(), mus.copy(), Ss * f) for f in factors]
