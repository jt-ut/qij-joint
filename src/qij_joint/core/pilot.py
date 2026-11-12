"""The pilot (the method's step 1, `method.py`): the cells, their kernel scales, and the weight vectors of the pilot evaluations.

Cells. M cells by sklearn's KMeans(M, n_init=1, random_state=seed): greedy k-means++ seeding (2 + ln M candidates per step), then
Lloyd's iteration to convergence. M is the budget rule (`pilot_budget`; spec/METHOD_pilot_budget.md, ruled 2026-11-12) unless
given: M(eps) = 3 K_r(eps), K_r = (kappa_r / eps)^(r/2), r = min(d, q), d the dimension of the clustering coordinates, q the
number of outputs the method works on; M does not depend on N. The clustering coordinates are the case's own
(`vq_transform(X)[0]` when the case has one) or else the standardized data. Empty cells are dropped and the labels renumbered
0 .. M_used - 1. Ported from the test bed's `bed/data.py` (quantize, _cells).

Scales. The kernel coordinates are the standardized data (always, whatever the clustering coordinates). The scale of a cell is
the CONN-weighted mean distance from its mean (in kernel coordinates) to the means of the cells it shares a Voronoi boundary
with: n_jk counts the points whose two nearest cell means are j and k, and sigma_j = sum_k n_jk D_jk / sum_k n_jk; a cell with no
neighbour takes the distance to its nearest cell mean. Every point takes its cell's scale. Ported from `bed/data.py` (_conn,
geometry) with the distances computed only for connected pairs (the bed's dense M x M x d array is 4 GB at N = 10^5).

The pilot evaluation of cell c is m_c, the mean influence of the cell: the evaluation at the weight vector u = 1{cell c}/n_c
(`cell_weights`), made like every other evaluation of the method, one refit on the full data (method.py)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
import scipy.sparse as sp


def zador_kappa(r: int) -> float:
    """kappa_r = 2 Gamma(r/2 + 1)^(2/r) ((r + 2)/r)^((r + 2)/2) / (r (r + 2)): Zador's law for the standard normal in r dimensions,
    per coordinate, with the sphere bound for the cell-shape constant (spec/METHOD_pilot_budget.md). 2.72, 1.00, 0.578, 0.398 at
    r = 1 .. 4."""
    from scipy.special import gamma
    r = int(r)
    return 2.0 * gamma(r / 2.0 + 1.0) ** (2.0 / r) * ((r + 2.0) / r) ** ((r + 2.0) / 2.0) / (r * (r + 2.0))


def pilot_budget(eps: float, d: int, q: int) -> int:
    """M(eps) = 3 K_r(eps), K_r = (kappa_r / eps)^(r/2), r = min(d, q): the pilot gets the cost the tree will spend, with the tree's
    leaf count taken as Zador's count for a normal influence of dimension r at within variance eps (spec/METHOD_pilot_budget.md).
    300 for r = 2 at eps = 0.01; 1319 for r = 3."""
    r = int(min(d, q)); assert r >= 1, (d, q)
    K = (zador_kappa(r) / float(eps)) ** (r / 2.0)
    return int(round(3.0 * K))


def standardize(X: np.ndarray) -> np.ndarray:
    """(X - mean) / std per column (population std); a 1-D sample becomes (N, 1)."""
    X = np.asarray(X, float)
    if X.ndim == 1:
        X = X[:, None]
    return (X - X.mean(0)) / X.std(0)


def quantize(Xc: np.ndarray, M: int, seed: int = 0) -> np.ndarray:
    """Cell labels: sklearn KMeans(M, n_init=1, random_state=seed) on the clustering coordinates Xc."""
    from sklearn.cluster import KMeans
    Xc = np.asarray(Xc, float)
    if Xc.ndim == 1:
        Xc = Xc[:, None]
    return KMeans(int(M), n_init=1, random_state=seed).fit(Xc).labels_


@dataclass
class Pilot:
    inv: np.ndarray        # (N,) cell of each point, 0 .. C-1
    C: int                 # cells used (empty cells dropped)
    ncell: np.ndarray      # (C,) points per cell, float
    proto: np.ndarray      # (C, d) cell means in kernel coordinates
    Xs: np.ndarray         # (N, d) kernel coordinates (standardized data)
    sigma: np.ndarray      # (C,) cell scales
    sigma_pt: np.ndarray   # (N,) = sigma[inv]
    M_requested: int
    Xc: np.ndarray = None  # (N, dc) clustering coordinates (the cells' and the refinement's)


def _cell_means(Y: np.ndarray, inv: np.ndarray, C: int, n: np.ndarray) -> np.ndarray:
    return np.stack([np.bincount(inv, Y[:, a], C) for a in range(Y.shape[1])], 1) / n[:, None]


def cell_scales(Xs: np.ndarray, inv: np.ndarray, C: int, proto: np.ndarray) -> np.ndarray:
    """sigma (C,): CONN-weighted mean distance to the neighbouring cell means, or the nearest cell mean without a neighbour."""
    N = len(Xs); sq = (proto ** 2).sum(1); two = np.empty((N, 2), int)
    for a in range(0, N, 2000):
        Xa = Xs[a:a + 2000]
        D = (Xa ** 2).sum(1)[:, None] - 2 * Xa @ proto.T + sq[None]
        two[a:a + 2000] = np.argsort(D, 1)[:, :2]
    j, k = two[two[:, 0] != two[:, 1]].T
    Mc = sp.coo_matrix((np.ones(2 * len(j)), (np.r_[j, k], np.r_[k, j])), shape=(C, C)).tocsr(); Mc.sum_duplicates(); Mc = Mc.tocoo()
    r, c, cnt = Mc.row, Mc.col, Mc.data
    Dc = np.sqrt(np.maximum(((proto[r] - proto[c]) ** 2).sum(1), 0))
    den = np.bincount(r, cnt, C); num = np.bincount(r, cnt * Dc, C)
    sigma = num / np.maximum(den, 1)
    lone = np.flatnonzero(den <= 0)
    if len(lone):
        from scipy.spatial import cKDTree
        dist, _ = cKDTree(proto).query(proto[lone], k=2)
        sigma[lone] = dist[:, 1]
    return sigma


def build_cells(X: np.ndarray, M: Optional[int] = None, vq_transform: Optional[Callable] = None,
                cells: Optional[np.ndarray] = None, seed: int = 0, eps: Optional[float] = None, q: Optional[int] = None) -> Pilot:
    """The cells and scales of the pilot. `cells` (N,) given: those labels are used as they are (renumbered, empties dropped);
    otherwise `quantize` on the clustering coordinates, M cells: M as given, else `pilot_budget(eps, d, q)` with d the dimension of
    the clustering coordinates (eps and q are then required)."""
    Xs = standardize(X); N = len(Xs)
    Xc = Xs if vq_transform is None else np.asarray(vq_transform(X)[0], float)
    if Xc.ndim == 1: Xc = Xc[:, None]
    if M is None and cells is None:
        assert eps is not None and q is not None, 'M, or eps and q for the budget rule, or the cells themselves'
        M = pilot_budget(eps, Xc.shape[1], q)
    M_req = None if M is None else int(M)
    if cells is None:
        cells = quantize(Xc, M_req, seed)
    _, inv = np.unique(np.asarray(cells), return_inverse=True); inv = inv.astype(int); C = int(inv.max()) + 1
    n = np.bincount(inv, minlength=C).astype(float)
    proto = _cell_means(Xs, inv, C, n)
    sigma = cell_scales(Xs, inv, C, proto)
    return Pilot(inv=inv, C=C, ncell=n, proto=proto, Xs=Xs, sigma=sigma, sigma_pt=sigma[inv], M_requested=M_req, Xc=Xc)


def cell_weights(pilot: Pilot) -> list:
    """The pilot's weight vectors, one per cell, as (sorted point indices, weights 1/n_c): m_c = sum_i u_i psi_i."""
    order = np.argsort(pilot.inv, kind='stable'); bounds = np.r_[0, np.cumsum(pilot.ncell.astype(int))]
    return [(order[bounds[c]:bounds[c + 1]], np.full(int(pilot.ncell[c]), 1.0 / pilot.ncell[c])) for c in range(pilot.C)]
