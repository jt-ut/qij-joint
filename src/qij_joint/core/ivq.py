"""
The I-VQ: bins of the initial influence estimate psi0_c, one quantizer
per estimand coordinate, and each bin's central-stencil measurement on
the full data (spec/method_notes.md sections 1 and 4).

`build_bins` runs 1-D k-means on the raw psi0_c values at the initial
count M_init = min(ceil(sqrt(2.7/eps)), n_distinct). `bin_differences`
measures the resulting bins on the full data (`core.differences.
difference`), giving each bin's finite-differenced influence U_k.
`between_terms` reduces those to V_btw. `kmeans_1d` lives here, not in
`refine.py`, because `build_bins` and `refine.py`'s level split both
use the same one-dimensional quantizer.

A failed evaluation of an initial bin (spec section 5) makes the whole
draw's QIJ result for this coordinate a write-off: `bin_differences`
stops at the bin that failed and returns a `BinSet` with `failed=True`
and `U` all NaN; `refine.run_refinement` turns that into a NaN
`CoordinateResult`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

import numpy as np

from .differences import central_step, difference, perturbed_weights

__all__ = ["BinSet", "kmeans_1d", "within_share", "build_bins", "bin_differences", "between_terms"]


@dataclass
class BinSet:
    """
    One estimand coordinate's I-VQ: the initial bins over psi0_c and
    their full-data measurement.

    labels    (N,) int, bin index per point, contiguous 0..M_used-1.
    n         (M_used,) int, bin sizes, all > 0.
    U         (M_used, q) centered bin influences, (M_used, 0) before
              `bin_differences` has run.
    centering_residual  (q,) the bin-mass-weighted residual sum_k p_k
              U_k subtracted out of U; `refine.run_refinement` subtracts
              this same residual from every split it measures, so a
              split's U stays on the initial bins' centered scale.
              zeros(0) before `bin_differences` has run.
    M_init    the initial bin count from the count rule.
    M_used    bins actually used (<= M_init; empty bins are dropped).
    within_share  achieved share of Var(psi0_c) left inside the bins.
    failed    True when a bin's full-data evaluation returned any NaN
              (spec section 5.2): U/centering_residual are then all
              NaN and MUST NOT be read by `between_terms`.
    """

    labels: np.ndarray
    n: np.ndarray
    U: np.ndarray
    centering_residual: np.ndarray
    M_init: int
    M_used: int
    within_share: float
    failed: bool = False

    @property
    def p(self) -> np.ndarray:
        """Bin masses n / N."""
        return self.n / self.labels.size


def kmeans_1d(values: np.ndarray, M: int):
    """
    Lloyd's iteration on the sorted 1-D array `values`, M target
    prototypes (spec/method_notes.md section 4; also `refine.py`'s
    level split, at M=2).

    Init: the M quantiles at levels (k+1/2)/M of the distinct sorted
    values. Assignment: `searchsorted` against the midpoints between
    consecutive prototypes (a boundary value goes to the lower
    prototype; with one prototype the midpoint array is empty and
    every point gets label 0). Update: each prototype becomes the mean
    of its bin, via two `bincount`s (counts, sums); a prototype whose
    bin is empty after an assignment is dropped and the rest
    relabelled contiguously. Converges when the assignment is unchanged
    from the previous iteration, or after 100 iterations.

    Returns (labels, prototypes): labels (N,) int, contiguous
    0..M_used-1, ordered by increasing prototype value; prototypes
    (M_used,), strictly increasing.
    """
    v = np.asarray(values, dtype=float).ravel()
    u = np.unique(v)
    M = int(M)

    prototypes = np.quantile(u, (np.arange(M) + 0.5) / M)

    prev_labels = None
    labels = None
    for _ in range(100):
        mids = (prototypes[:-1] + prototypes[1:]) / 2.0
        labels = np.searchsorted(mids, v, side="left")

        counts = np.bincount(labels, minlength=prototypes.size)
        if not counts.all():
            present = np.flatnonzero(counts)
            remap = np.full(prototypes.size, -1, dtype=int)
            remap[present] = np.arange(present.size)
            labels = remap[labels]
            prototypes = prototypes[present]
            counts = counts[present]

        converged = (
            prev_labels is not None
            and prev_labels.shape == labels.shape
            and np.array_equal(prev_labels, labels)
        )
        if converged:
            break
        prev_labels = labels

        # Every bin is non-empty here, so `counts` is strictly
        # positive and no mean is undefined.
        sums = np.bincount(labels, weights=v, minlength=prototypes.size)
        prototypes = sums / counts

    labels = labels.astype(int)
    return labels, prototypes


def within_share(values: np.ndarray, labels: np.ndarray) -> float:
    """Share of the plain variance of `values` left inside the bins
    defined by `labels`: sum_k sum_{i in k}(v_i-mean_k)^2 / sum_i(v_i-mean)^2.
    0.0 for one bin or zero total variance. Two `bincount` passes
    (counts, sums), not a per-group Python loop."""
    v = np.asarray(values, dtype=float).ravel()
    labels = np.asarray(labels)

    total_var = float(np.sum((v - v.mean()) ** 2))
    if np.unique(labels).size <= 1 or total_var == 0.0:
        return 0.0

    counts = np.bincount(labels)
    sums = np.bincount(labels, weights=v)
    within = float(np.sum(v ** 2) - np.sum(sums ** 2 / counts))
    return within / total_var


def build_bins(values: np.ndarray, eps: float) -> BinSet:
    """
    Build the I-VQ for one coordinate from the raw psi0_c(x_i) at all N
    points, at the initial count M_init = min(ceil(sqrt(2.7/eps)),
    n_distinct) (spec/method_notes.md section 4). No minimum bin size,
    no merging: empty bins are dropped during `kmeans_1d`'s iteration
    and M_used <= M_init is what the returned `BinSet` reports (the
    INITIAL bins only; refinement may grow the count further). U comes
    back (M_used, 0): `bin_differences` fills it in.
    """
    v = np.asarray(values, dtype=float).ravel()
    N = v.size
    D = int(np.unique(v).size)

    m_ref = math.ceil(math.sqrt(2.7 / eps))
    M_init = int(min(m_ref, D))

    if M_init <= 1:
        labels = np.zeros(N, dtype=int)
        n = np.array([N], dtype=int)
    else:
        labels, _prototypes = kmeans_1d(v, M_init)
        n = np.bincount(labels)

    share = within_share(v, labels)
    L = int(n.size)

    return BinSet(
        labels=labels, n=n, U=np.zeros((L, 0)), centering_residual=np.zeros(0),
        M_init=M_init, M_used=L, within_share=share, failed=False,
    )


def bin_differences(X: np.ndarray, counter, theta_hat: np.ndarray, binset: BinSet, eta: float) -> BinSet:
    """
    Measure one coordinate's I-VQ on the full data (spec section 4):
    for each bin k, in order, the central stencil against the bin's
    own mass (`differences.difference`, step delta = central_step(eta)).
    A NaN in any U_k stops the loop at that bin and returns
    `failed=True` with U/centering_residual all NaN. Otherwise U is
    centered by the bin-mass-weighted residual centering_residual =
    sum_k p_k U_k, kept on the returned `BinSet` for `refine.
    run_refinement` to reuse. At most M_used evaluations of `counter`,
    never a Python loop over the N data points.
    """
    N = binset.labels.size
    theta_hat = np.asarray(theta_hat, dtype=float)
    q = theta_hat.size
    M_used = binset.M_used

    delta = central_step(eta)
    p = binset.p
    U = np.empty((M_used, q))

    for k in range(M_used):
        mask = binset.labels == k

        def evaluate(t: float, _mask=mask) -> np.ndarray:
            omega = perturbed_weights(np.ones(N), _mask, t)
            return counter(X, omega)

        U_k = difference(float(p[k]), delta, evaluate)
        if np.any(np.isnan(U_k)):
            return replace(
                binset, U=np.full((M_used, q), np.nan),
                centering_residual=np.full(q, np.nan), failed=True,
            )
        U[k] = U_k

    centering_residual = p @ U
    U = U - centering_residual

    return replace(binset, U=U, centering_residual=centering_residual, failed=False)


def between_terms(binset: BinSet, coordinate: int) -> float:
    """V_btw = (1/N) sum_k p_k U[k, coordinate]^2 (spec section 4).
    Callers must not call this on a `BinSet` with `failed=True`."""
    N = binset.labels.size
    p = binset.p
    U_c = binset.U[:, coordinate]
    return float(np.sum(p * U_c ** 2)) / N
