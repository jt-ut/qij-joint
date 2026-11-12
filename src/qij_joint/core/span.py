"""The span of the active constraint rows, for the redundancy rule of `method.py` (identical copy: the test bed's bed/span.py).

The rule. A new evaluation row a (weights on the points) whose weight pattern lies in the span of the active rows does not
enter psi-hat: the exact interpolant can satisfy dependent rows only when their values obey the same linear relation, which
exact influences do and any evaluation with an error does not. The row's value is kept: the method compares it with the value
the active rows imply (a . psi-hat, after the next fit) and reports the discrepancy as the measured evaluation error.

The test is algebraic, as the determined-leaf test (method._leverage): the squared norm of the projection residual of a onto
the span of the active rows, relative to |a|^2, is at most REDUNDANT_TOL.

How it is computed. The pilot rows (cell means) are always active and mutually orthogonal; their span is the cell indicators.
Every other active row is kept as its residual after the cell projection (sparse: spread over the cells the row meets), and the
Euclidean Gram matrix of those residuals is kept as a Cholesky factor L in the order the rows entered. A test is one sparse
product and one triangular solve; a row that enters appends one row to L; a row that leaves (a split leaf's mean row, when its
children's rows enter) is deleted from L by a rank-one update of the trailing block."""
import numpy as np
import scipy.linalg as sl
import scipy.sparse as sp

REDUNDANT_TOL = 1e-8


class Span:
    def __init__(self, inv, ncell):
        self.inv = np.asarray(inv, int); self.ncell = np.asarray(ncell, float); self.C = len(self.ncell); self.N = len(self.inv)
        order = np.argsort(self.inv, kind='stable'); b = np.r_[0, np.cumsum(self.ncell.astype(int))]
        self.cell_pts = [order[b[c]:b[c + 1]] for c in range(self.C)]
        self.ids = []; self.res = {}; self.L = np.zeros((0, 0)); self._R = None

    def _residual(self, idx, w):
        """The row's residual after the projection on the cell indicators, as (sorted indices, values)."""
        idx = np.asarray(idx, np.int64); w = np.asarray(w, float)
        coef = np.bincount(self.inv[idx], w, self.C) / self.ncell
        cells = np.flatnonzero(coef != 0.0)
        pts = np.unique(np.concatenate([idx] + [self.cell_pts[c] for c in cells])) if len(cells) else idx
        r = np.zeros(len(pts)); r[np.searchsorted(pts, idx)] = w
        r -= coef[self.inv[pts]]
        return pts, r

    def _matrix(self):
        if self._R is None:
            if not self.ids: self._R = sp.csr_matrix((0, self.N)); return self._R
            ptr = np.r_[0, np.cumsum([len(self.res[i][0]) for i in self.ids])]
            self._R = sp.csr_matrix((np.concatenate([self.res[i][1] for i in self.ids]), np.concatenate([self.res[i][0] for i in self.ids]), ptr),
                                    shape=(len(self.ids), self.N))
        return self._R

    def test(self, idx, w):
        """(redundant, relative squared residual, the data `add` needs)."""
        pts, r = self._residual(idx, w)
        na2 = float(np.dot(w, w)); nr2 = float(np.dot(r, r))
        if self.ids:
            g = self._matrix()[:, pts] @ r
            x = sl.solve_triangular(self.L, g, lower=True, check_finite=False)
            piv = nr2 - float(np.dot(x, x))
        else:
            x = np.zeros(0); piv = nr2
        rel = max(piv, 0.0) / na2
        return rel <= REDUNDANT_TOL, rel, (pts, r, x, piv)

    def add(self, rid, data):
        """The row rid enters (data from `test` on the current span)."""
        pts, r, x, piv = data; t = len(self.ids)
        L = np.zeros((t + 1, t + 1)); L[:t, :t] = self.L; L[t, :t] = x; L[t, t] = np.sqrt(piv)
        self.L = L; self.ids.append(rid); self.res[rid] = (pts, r); self._R = None

    def remove(self, rid):
        """The row rid leaves: delete its row and column of L; the trailing block takes the rank-one update L33 L33' + l l'."""
        k = self.ids.index(rid); t = len(self.ids)
        l = self.L[k + 1:, k].copy(); L33 = self.L[k + 1:, k + 1:].copy()
        for j in range(len(l)):                                # Cholesky rank-one update of L33 by l
            r = np.hypot(L33[j, j], l[j]); c = r / L33[j, j]; s = l[j] / L33[j, j]; L33[j, j] = r
            if j + 1 < len(l):
                L33[j + 1:, j] = (L33[j + 1:, j] + s * l[j + 1:]) / c
                l[j + 1:] = c * l[j + 1:] - s * L33[j + 1:, j]
        keep = [i for i in range(t) if i != k]
        L = self.L[np.ix_(keep, keep)].copy(); L[k:, k:] = L33
        self.L = L; del self.ids[k]; del self.res[rid]; self._R = None
