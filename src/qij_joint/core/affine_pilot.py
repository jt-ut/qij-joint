"""
The affine pilot for QIJ (spec/QIJ_affine_pilot_spec.md section 2): a
mean-preserving affine field per X-VQ cell, in place of the Gaussian
process pilot, with no posterior variance of any kind.

Per cell j, mu_j is the mean of the cell's own points in Z; g_j is the
weighted least-squares gradient fit over the cell's CADJ neighbours'
survey means (section 2.1),

    I_k - I_j ~= g_j^T (mu_k - mu_j),   k in N(j),   weight p_k,

and the pilot at point i in cell j is psi0(z_i) = I_j + g_j^T
(z_i - mu_j) -- exactly I_j on average over the cell's own points. The
bridge score Delta_jk, one per CADJ pair (j, k) with a non-empty
second-order cell (points whose best- and second-best-matching
prototypes are j and k), is the disagreement between the two cells' own
fields on that shared cell (section 2.2), with mass m_jk = |(jk)|/N.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp

__all__ = ["AffinePilot", "fit"]


@dataclass
class AffinePilot:
    """The affine pilot's products for one draw, every output (spec
    section 2.3).

    psi0        (N, q_full) the pilot at every point.
    cell_mu     (M_used, d_z) each cell's own mean, in Z.
    cell_g      (M_used, d_z, q_full) each cell's fitted gradient; 0
                where the cell has fewer than d_z live CADJ neighbours,
                its normal matrix is singular, or the cell itself is a
                failed prototype (section 2.1, 6).
    pair_j, pair_k  (P,) int, the CADJ pairs with a non-empty
                second-order cell.
    pair_n      (P,) int, |(jk)|, the pair's own point count.
    pair_mass   (P,) m_jk = pair_n / N.
    pair_delta  (P, q_full) Delta_jk (section 2.2).
    pair_id     (N,) int, index into `pair_j`/`pair_k`/... for a point
                whose own (bmu, bmu2) is one of them; P (one past the
                end) for every other point.
    """

    psi0: np.ndarray
    cell_mu: np.ndarray
    cell_g: np.ndarray
    pair_j: np.ndarray
    pair_k: np.ndarray
    pair_n: np.ndarray
    pair_mass: np.ndarray
    pair_delta: np.ndarray
    pair_id: np.ndarray


def _cell_means(Z: np.ndarray, labels: np.ndarray, M_used: int) -> np.ndarray:
    """mu_j (M_used, d_z): the mean of cell j's own points in Z, by
    grouped sums over the field index `labels` already carries
    (0..M_used-1, contiguous -- no argsort needed before `bincount`)."""
    d_z = Z.shape[1]
    n = np.bincount(labels, minlength=M_used).astype(float)
    mu = np.empty((M_used, d_z))
    for a in range(d_z):
        mu[:, a] = np.bincount(labels, weights=Z[:, a], minlength=M_used) / n
    return mu


def _fit_gradients(conn, mu: np.ndarray, I_proto: np.ndarray, p: np.ndarray,
                    M_used: int) -> np.ndarray:
    """g_j for every cell (spec section 2.1): the weighted normal
    equations summed over `conn`'s CADJ edges by one sparse matmul per
    normal-equation entry (E1/E2, never a per-cell Python loop), solved
    by `numpy.linalg.solve` on the whole (M_used, d_z, d_z) stack at
    once. g_j = 0 where the cell has fewer than d_z live neighbours, its
    normal matrix is singular at relative tolerance 1e-12 of its largest
    eigenvalue, or the edge touches a failed prototype (I_proto's row
    not finite -- the finite/mass_c convention of `xvq.prototype_
    influences`, spec section 6)."""
    d_z = mu.shape[1]
    q_full = I_proto.shape[1]
    owner, neighbour = conn.tocsr().nonzero()
    keep = owner != neighbour
    proto_ok = np.all(np.isfinite(I_proto), axis=1)
    keep &= proto_ok[owner] & proto_ok[neighbour]
    owner, neighbour = owner[keep], neighbour[keep]

    delta_mu = mu[neighbour] - mu[owner]              # (E, d_z)
    delta_I = I_proto[neighbour] - I_proto[owner]      # (E, q_full)
    w = p[neighbour]

    n_edges = owner.size
    S = sp.csr_matrix((np.ones(n_edges), (np.arange(n_edges), owner)),
                       shape=(n_edges, M_used))

    A = np.zeros((M_used, d_z, d_z))
    for a in range(d_z):
        for b in range(a, d_z):
            col = S.T @ (w * delta_mu[:, a] * delta_mu[:, b])
            A[:, a, b] = A[:, b, a] = col

    B = np.zeros((M_used, d_z, q_full))
    for a in range(d_z):
        B[:, a, :] = S.T @ (w[:, None] * delta_mu[:, a:a + 1] * delta_I)

    deg = np.bincount(owner, minlength=M_used)
    eigval = np.linalg.eigvalsh(A)  # (M_used, d_z), ascending
    valid = (deg >= d_z) & (eigval[:, 0] > 1e-12 * np.maximum(eigval[:, -1], 0.0))

    g = np.zeros((M_used, d_z, q_full))
    if valid.any():
        g[valid] = np.linalg.solve(A[valid], B[valid])
    return g


def _bridge_pairs(bmu: np.ndarray, bmu2: np.ndarray, conn, psi0: np.ndarray,
                   psi0_bmu2: np.ndarray, M_used: int, N: int):
    """The bridge table (spec section 2.2): for every CADJ pair (j, k)
    with a non-empty second-order cell, Delta_jk and m_jk, grouped by
    one sort of the (bmu, bmu2) key (E1/E2 -- never a Python loop over
    pairs or points). `psi0` is cell j's own field at its points (bmu ==
    j already, so `psi0` itself IS psi0^{(j)}` there); `psi0_bmu2` is
    cell bmu2's field evaluated at the same points, `psi0^{(k)}`."""
    owner, neighbour = conn.tocsr().nonzero()
    edge_key = owner[owner != neighbour].astype(np.int64) * M_used + \
        neighbour[owner != neighbour].astype(np.int64)

    key = bmu.astype(np.int64) * M_used + bmu2.astype(np.int64)
    uniq, inverse, counts = np.unique(key, return_inverse=True, return_counts=True)
    is_cadj = np.isin(uniq, edge_key)

    q_full = psi0.shape[1]
    mean_own = np.empty((uniq.size, q_full))
    mean_cross = np.empty((uniq.size, q_full))
    for c in range(q_full):
        mean_own[:, c] = np.bincount(inverse, weights=psi0[:, c]) / counts
        mean_cross[:, c] = np.bincount(inverse, weights=psi0_bmu2[:, c]) / counts

    valid = np.where(is_cadj)[0]
    remap = np.full(uniq.size, valid.size, dtype=np.int64)
    remap[valid] = np.arange(valid.size)

    pair_j = (uniq[valid] // M_used).astype(np.int64)
    pair_k = (uniq[valid] % M_used).astype(np.int64)
    pair_n = counts[valid].astype(np.int64)
    pair_mass = pair_n.astype(float) / N
    pair_delta = np.abs(mean_own[valid] - mean_cross[valid])
    pair_id = remap[np.asarray(inverse).reshape(-1)]
    return pair_j, pair_k, pair_n, pair_mass, pair_delta, pair_id


def fit(Z: np.ndarray, xvq, I_proto: np.ndarray) -> AffinePilot:
    """The affine pilot for every output (spec section 2): the
    mean-preserving per-cell field `psi0`, its per-cell gradients, and
    the bridge score on every CADJ pair's second-order cell. `Z` is the
    quantizer's own coordinates, exactly what `xvq` was fitted on (spec
    section 6); `I_proto` (M_used, q_full) is the mass-centred survey
    influence, as `xvq.run_xvq` returns it."""
    Z = np.asarray(Z, dtype=float)
    I_proto = np.asarray(I_proto, dtype=float)
    labels = np.asarray(xvq.bmu, dtype=np.intp)
    bmu2 = np.asarray(xvq.bmu2, dtype=np.intp)
    p = np.asarray(xvq.p, dtype=float)
    M_used = xvq.M_used
    N = Z.shape[0]

    mu = _cell_means(Z, labels, M_used)
    g = _fit_gradients(xvq.conn, mu, I_proto, p, M_used)

    delta_own = Z - mu[labels]
    psi0 = I_proto[labels] + np.einsum('id,idc->ic', delta_own, g[labels])

    delta_cross = Z - mu[bmu2]
    psi0_bmu2 = I_proto[bmu2] + np.einsum('id,idc->ic', delta_cross, g[bmu2])

    pair_j, pair_k, pair_n, pair_mass, pair_delta, pair_id = _bridge_pairs(
        labels, bmu2, xvq.conn, psi0, psi0_bmu2, M_used, N)

    return AffinePilot(
        psi0=psi0, cell_mu=mu, cell_g=g,
        pair_j=pair_j, pair_k=pair_k, pair_n=pair_n, pair_mass=pair_mass,
        pair_delta=pair_delta, pair_id=pair_id,
    )
