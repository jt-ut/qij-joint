"""
The affine pilot for QIJ (spec/QIJ_pilot_floor_options_interface.md
Option 1): a mean-preserving affine field per X-VQ cell, in place of
the Gaussian-process pilot `core.influence_model`, with no posterior
variance of any kind.

Per cell j (an X-VQ prototype), mu_j is the mean of the cell's own
points in Z (bmu membership); g_j is the weighted least-squares
gradient fit over the cell's CADJ neighbours' survey means,

    I_k - I_j ~= g_j^T (mu_k - mu_j),   k in N(j),   weight p_k,

and the pilot at point i in cell j is psi0(z_i) = I_j + g_j^T
(z_i - mu_j) -- exactly I_j on average over the cell's own points.
Ported from commit 6c8fb74's `core/affine_pilot.py`, WITHOUT its bridge
table/bridge pricing (the unified loop has no adjacency splits).

Per coordinate, `constant_path`/`offset` mirror `influence_model.
InfluenceModel`'s own definitions exactly -- the same psi_bar/spread
threshold against `theta_Q`, over that coordinate's own finite design
-- so the loop and `qij.fit` read them identically whichever pilot is
in use. A coordinate on the constant path never gets an affine field:
`psi0` is pinned at its mass-weighted mean `const_value[c]` at every
point, and `offset[c]` is that same value; a non-constant coordinate's
`offset[c]` is the mean of its own (affine) `psi0` over Z, as
`fit_influence_model` computes it (diagnostic only, never subtracted).

Cost O(N d_z + M d_z^2): one grouped sum over Z for `cell_mu`, one
sparse matmul per normal-equation entry over `xvq.conn`'s edges (never
an M x M array), and one `numpy.linalg.solve` over the whole
(M_used, d_z, d_z) stack at once.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np
import scipy.sparse as sp

__all__ = ["AffinePilot", "fit_affine_pilot", "affine_psi0"]


@dataclass
class AffinePilot:
    """The affine pilot's products for one draw, every (measured)
    output (spec/QIJ_pilot_floor_options_interface.md Option 1).

    psi0            (N, q) the pilot at every point, at the Z
                    `fit_affine_pilot` was given.
    cell_mu         (M_used, d_z) each cell's own mean, in Z.
    cell_I          (M_used, q) the survey influence I_proto at each
                    cell, as given to `fit_affine_pilot` (mass-centred,
                    as stored) -- `psi0`'s I_j term.
    gradients       (M_used, q, d_z) each cell's fitted gradient g_j;
                    0 where the cell has fewer than d_z live CADJ
                    neighbours, its normal matrix is singular (relative
                    tolerance 1e-12), or an incident edge touches a
                    failed prototype (I_proto's row not finite).
    n_flat_cells    int, the number of cells with g_j = 0 (the same
                    cells for every output: the normal matrix A_j does
                    not depend on the output column).
    constant_path   (q,) bool, mirroring `influence_model.
                    InfluenceModel.constant_path` exactly: coordinate
                    has no usable spread in I_proto over its own finite
                    design, relative to theta_Q.
    const_value     (q,) the constant psi0 value on the constant path
                    (mass-weighted mean of I_proto over the finite
                    design); 0 elsewhere.
    offset          (q,) mean of psi0 over Z (diagnostic; not
                    subtracted from psi0) -- const_value[c] itself on
                    the constant path, since psi0 is const_value[c]
                    everywhere there.
    """

    psi0: np.ndarray
    cell_mu: np.ndarray
    cell_I: np.ndarray
    gradients: np.ndarray
    n_flat_cells: int
    constant_path: np.ndarray
    const_value: np.ndarray
    offset: np.ndarray

    # Cached fit-time identity, so `affine_psi0` can serve the common
    # call (the SAME Z `fit_affine_pilot` was given -- qij.py's own
    # `_psi0(model, Z)` call site) at no extra cost, exactly reproducing
    # the cell membership the fit used. Not part of the public product.
    _fit_Z: Optional[np.ndarray] = field(default=None, repr=False, compare=False)
    _fit_labels: Optional[np.ndarray] = field(default=None, repr=False, compare=False)


def _cell_means(Z: np.ndarray, labels: np.ndarray, M_used: int) -> np.ndarray:
    """mu_j (M_used, d_z): the mean of cell j's own points in Z, by
    grouped sums over the field index `labels` (0..M_used-1,
    contiguous -- no argsort needed before `bincount`)."""
    d_z = Z.shape[1]
    n = np.bincount(labels, minlength=M_used).astype(float)
    mu = np.empty((M_used, d_z))
    for a in range(d_z):
        mu[:, a] = np.bincount(labels, weights=Z[:, a], minlength=M_used) / n
    return mu


def _fit_gradients(conn, mu: np.ndarray, I_proto: np.ndarray, p: np.ndarray,
                    M_used: int, d_z: int) -> Tuple[np.ndarray, np.ndarray]:
    """g_j (M_used, d_z, q): the weighted normal equations summed over
    `conn`'s CADJ edges (symmetrized -- "either direction", Option 1)
    by one sparse matmul per normal-equation entry, solved by
    `numpy.linalg.solve` on the whole (M_used, d_z, d_z) stack at once
    -- never a per-cell Python loop, never an M x M array. Returns
    (g, valid): `valid[j]` is False, and g[j] left at 0, where cell j
    has fewer than d_z live CADJ neighbours, its normal matrix is
    singular at relative tolerance 1e-12 of its largest eigenvalue, or
    an incident edge touches a failed prototype (I_proto's row not
    finite, the whole-row convention `xvq.prototype_influences`
    returns). `valid` does not depend on the output column: the normal
    matrix A_j is built from `mu` alone, shared by every column's RHS."""
    q = I_proto.shape[1]
    conn_csr = conn.tocsr()
    conn_sym = conn_csr.maximum(conn_csr.T)
    owner, neighbour = conn_sym.nonzero()
    keep = owner != neighbour
    proto_ok = np.all(np.isfinite(I_proto), axis=1)
    keep &= proto_ok[owner] & proto_ok[neighbour]
    owner, neighbour = owner[keep], neighbour[keep]

    delta_mu = mu[neighbour] - mu[owner]              # (E, d_z)
    delta_I = I_proto[neighbour] - I_proto[owner]      # (E, q)
    w = p[neighbour]

    n_edges = owner.size
    S = sp.csr_matrix((np.ones(n_edges), (np.arange(n_edges), owner)),
                       shape=(n_edges, M_used))

    A = np.zeros((M_used, d_z, d_z))
    for a in range(d_z):
        for b in range(a, d_z):
            col = S.T @ (w * delta_mu[:, a] * delta_mu[:, b])
            A[:, a, b] = A[:, b, a] = col

    B = np.zeros((M_used, d_z, q))
    for a in range(d_z):
        B[:, a, :] = S.T @ (w[:, None] * delta_mu[:, a:a + 1] * delta_I)

    deg = np.bincount(owner, minlength=M_used)
    eigval = np.linalg.eigvalsh(A)  # (M_used, d_z), ascending
    valid = (deg >= d_z) & (eigval[:, 0] > 1e-12 * np.maximum(eigval[:, -1], 0.0))

    g = np.zeros((M_used, d_z, q))
    if valid.any():
        g[valid] = np.linalg.solve(A[valid], B[valid])
    return g, valid


def fit_affine_pilot(Z: np.ndarray, xvq, I_proto: np.ndarray, theta_Q: np.ndarray) -> AffinePilot:
    """The affine pilot for every (measured) output
    (spec/QIJ_pilot_floor_options_interface.md Option 1): the
    mean-preserving per-cell field `psi0` and its per-cell gradients,
    plus `constant_path`/`offset` mirroring `influence_model.
    fit_influence_model`'s own definitions. `Z` is the quantizer's own
    coordinates, exactly what `xvq` was fitted on (qij.py's
    `vq_transform` output, the same Z `fit_influence_model` would be
    given); `I_proto` (M_used, q) is the mass-centred survey influence,
    already restricted to the measured outputs the same way qij.py
    restricts it before `fit_influence_model`; `theta_Q` (q,) is used
    only for the constant-path threshold, exactly as
    `fit_influence_model` uses it."""
    Z = np.asarray(Z, dtype=float)
    if Z.ndim == 1:
        Z = Z.reshape(-1, 1)
    I_proto = np.asarray(I_proto, dtype=float)
    theta_Q = np.asarray(theta_Q, dtype=float)
    labels = np.asarray(xvq.bmu, dtype=np.intp)
    p = np.asarray(xvq.p, dtype=float)
    M_used = xvq.M_used
    d_z = Z.shape[1]
    q = I_proto.shape[1]

    mu = _cell_means(Z, labels, M_used)
    g, valid = _fit_gradients(xvq.conn, mu, I_proto, p, M_used, d_z)
    n_flat_cells = int(np.sum(~valid))

    # constant_path / const_value, mirroring influence_model.
    # fit_influence_model's own definitions exactly (same psi_bar/spread
    # threshold against theta_Q, over coordinate c's own finite design).
    finite = np.isfinite(I_proto)
    constant_path = np.zeros(q, dtype=bool)
    const_value = np.zeros(q, dtype=float)
    for c in range(q):
        idx_c = np.where(finite[:, c])[0]
        M_c = idx_c.size
        if M_c == 0:
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

    delta_own = Z - mu[labels]
    psi0_out = I_proto[labels] + np.einsum('id,idc->ic', delta_own, g[labels])
    if np.any(constant_path):
        psi0_out[:, constant_path] = const_value[None, constant_path]

    offset = np.empty(q, dtype=float)
    offset[constant_path] = const_value[constant_path]
    if np.any(~constant_path):
        offset[~constant_path] = psi0_out[:, ~constant_path].mean(axis=0)

    gradients = np.transpose(g, (0, 2, 1))  # (M_used, q, d_z)

    return AffinePilot(
        psi0=psi0_out, cell_mu=mu, cell_I=I_proto, gradients=gradients,
        n_flat_cells=n_flat_cells, constant_path=constant_path,
        const_value=const_value, offset=offset,
        _fit_Z=Z, _fit_labels=labels,
    )


def affine_psi0(pilot: AffinePilot, Z: np.ndarray) -> np.ndarray:
    """psi0(x) at Z (N, q) from an already-fitted `AffinePilot`:
    I_j + g_j^T (z_i - mu_j) at point i's own cell j, with the
    constant-path columns pinned at `const_value`. Cost O(N d_z).

    When `Z` is the SAME array `fit_affine_pilot` was given (qij.py's
    own call site: `fit_affine_pilot` then `affine_psi0` on the one Z
    the draw uses throughout), the fit's own cell membership is reused
    exactly, by identity -- never recomputed. For any other `Z`, cell
    membership is resolved by nearest cell mean (Euclidean, in Z);
    never exercised by the unified loop itself, which queries only its
    one Z, but kept correct for any other caller."""
    Z = np.asarray(Z, dtype=float)
    if Z.ndim == 1:
        Z = Z.reshape(-1, 1)

    if pilot._fit_Z is not None and Z is pilot._fit_Z:
        labels = pilot._fit_labels
    else:
        d2 = ((Z[:, None, :] - pilot.cell_mu[None, :, :]) ** 2).sum(axis=-1)
        labels = np.argmin(d2, axis=1)

    delta = Z - pilot.cell_mu[labels]
    out = pilot.cell_I[labels] + np.einsum('id,icd->ic', delta, pilot.gradients[labels])
    if np.any(pilot.constant_path):
        out[:, pilot.constant_path] = pilot.const_value[None, pilot.constant_path]
    return out
