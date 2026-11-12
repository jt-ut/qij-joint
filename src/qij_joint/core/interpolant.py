"""psi-hat: the exact kernel interpolant of every evaluation made so far (the method's step 2, `method.py`).

Kernel: exponential (Matern nu = 1/2) with per-point scales sigma_i (the cell scales of `core.pilot.cell_scales`),
    k(x_i, x_j) = (2 sigma_i sigma_j / S)^(d/2) exp(-sqrt(2 D2 / S)),   S = sigma_i^2 + sigma_j^2,   D2 = |x_i - x_j|^2.
An evaluation is one constraint row a (weights on the points): sum_i a_i psi-hat_i = (the returned value). The row's column is
c = K a' (length N). For the active rows A (m x N): Gram matrix G = A K A', alpha solves (G + jit I) alpha = y and
psi-hat = K A' alpha. The reproduction |A psi-hat - y| <= 1e-8 max(1, |y|) is asserted (one step of iterative refinement first
if it fails): it is what makes psi-hat interpolate. `method.py` hands this module evaluations in its own scaled units (each
output divided by a fixed scale), so the tolerance's floor of 1 is on the scale of the data.

The jitter. jit = 1e-10 x (the largest diagonal entry of the pilot block G11), fixed at the first fit for the whole run. Its job
is the exact linear dependencies among the rows (the cell means and the leaf means always share the total-mean direction, so G
is singular): it is small against the kernel's scale and does not change during the run, so the factor is kept from one fit to
the next.

The factor, by blocks. The first fit's rows (the M pilot rows) keep positions 0 .. M-1 for the whole run. L11 = chol(G11 + jit I)
is computed once. Every other position r (the tree and test rows, "the tail") keeps its row of R = G21 L11^-T, computed once
when the row enters; the tail's Schur complement S = G22 + jit I - R R' is updated at new positions only, and each fit refactors
only S. Solves with G + jit I go through the blocks (`solve`); no m x m Gram matrix is kept.

The column store. One chunk per fit: the pilot's columns are chunk 0; a fit that appends rows allocates one chunk holding exactly
those rows' columns; a row that takes an existing position (a split's smaller child takes its parent's) is written into that
position's chunk. A column is computed once, in one of two ways, both exact identities:
  from the points: the kernel between all N points and the union of the new rows' supports, BLOCK entries per temporary;
  as a combination: a row that is a linear combination of other rows (a split's larger child, n_b row_b = n_p row_p - n_a row_a)
  has the same combination of their columns.

Ported from the test bed's `bed/estimators.py` (class Matern12) without change to the arithmetic."""
import numpy as np
import scipy.linalg as sl
import scipy.sparse as sp

BLOCK = 1 << 17          # kernel entries per temporary block when columns are computed from the points (1 MB per temporary)


class Interpolant:
    """The exact interpolant psi-hat on the points Xs (N, d) with per-point kernel scales sigma_pt (N,)."""

    def __init__(self, Xs, sigma_pt):
        self.X = np.ascontiguousarray(np.asarray(Xs, float)); self.N, self.d = self.X.shape; self.sq = (self.X ** 2).sum(1)
        self.sig = np.asarray(sigma_pt, float); self.h2 = 0.5 * self.sig ** 2
        self.chunks = []; self.start = []; self.ids = []; self.alpha = None; self.jit = None
        self.P0 = None; self.L11 = None; self.R = np.zeros((0, 0)); self.S = np.zeros((0, 0)); self.L22 = np.zeros((0, 0)); self._buf = {}
        self.kentries = 0; self.n_refine = 0

    # ---- the kernel and the columns ----
    def kernel(self, I, lo, hi):
        """k(x_i, x_j) for i in I (sorted point indices) and j = lo .. hi-1: an (len(I), hi - lo) array."""
        X, sq, sig, h2 = self.X, self.sq, self.sig, self.h2
        K = X[I] @ X[lo:hi].T; K *= -2.0; K += sq[I][:, None] + sq[lo:hi][None]; np.maximum(K, 0.0, out=K)
        a, b = np.searchsorted(I, [lo, hi]); K[np.arange(a, b), I[a:b] - lo] = 0.0      # the self-distance is exactly 0
        Sh = np.add.outer(h2[I], h2[lo:hi])                                                # S / 2
        K /= Sh; np.sqrt(K, out=K); np.negative(K, out=K); np.exp(K, out=K)
        B = np.multiply.outer(sig[I], sig[lo:hi]); B /= Sh                                 # 2 sigma_i sigma_j / S
        P = np.sqrt(B) if self.d % 2 else np.ones_like(B)
        for _ in range(self.d // 2): P *= B                                                # B^(d/2) without a general power
        K *= P
        return K

    def columns(self, A, out):
        """out[r] = K a' for every row a of the CSR matrix A: block by block over the points, the kernel between the block and the
        union U of the rows' supports, times the weights."""
        U = np.unique(A.indices); N = self.N
        W = sp.csr_matrix((A.data, np.searchsorted(U, A.indices), A.indptr), shape=(A.shape[0], len(U)))
        b = max(1, BLOCK // len(U))
        for lo in range(0, N, b):
            hi = min(N, lo + b); out[:, lo:hi] = W @ self.kernel(U, lo, hi)
        self.kentries += N * len(U)

    def _where(self, r):
        k = int(np.searchsorted(self.start, r, side='right')) - 1
        return k, r - self.start[k]

    def col(self, r):
        k, o = self._where(r); return self.chunks[k][o]

    def dot_T(self, alpha):
        """K A' alpha = sum over chunks of chunk' alpha[chunk's positions]: (N, J)."""
        out = np.zeros((self.N, alpha.shape[1]))
        for s, c in zip(self.start, self.chunks): out += c.T @ alpha[s:s + len(c)]
        return out

    def leaf_dot(self, x, idx):
        """x' (K A')[:, idx]: (len(idx),)."""
        out = np.zeros(len(idx))
        for s, c in zip(self.start, self.chunks): out += x[s:s + len(c)] @ c[:, idx]
        return out

    # ---- the redundancy rule in the kernel's metric ----
    # A row whose kernel column lies (nearly) in the span of the active rows' kernel columns adds no constraint the interpolant
    # can hold: the fit would carry it on the jitter, alpha would grow as 1 / (its Schur complement), and with values known only
    # to the step term the reproduction fails (fp, 2026-11-12: 2-3-point leaves, |alpha| 2e5, residual = jit x alpha). Such a
    # row is kept as a measurement, not a constraint, exactly as core/span.py keeps a row dependent in the weights. The test:
    # s = G_rr - g' H g, the row's Schur complement against the rows of the last fit (less the rows that are leaving) and the
    # rows accepted earlier in the same batch; redundant when s <= KERNEL_TOL G_rr (the span rule's relative form) or
    # s <= 100 jit (within two orders of the jitter: the jitter, not the data, would be holding the row).
    KERNEL_TOL = 1e-8

    def begin_batch(self):
        """Start a batch of candidate rows (called by the method whenever its rows may change between two fits)."""
        self._cache = {}; self._leave = []; self._bids = []; self._bU = np.zeros((len(self.ids), 0)); self._bL = np.zeros((0, 0))
        self._H_leave = None

    def leaving(self, rid):
        """Register a row of the last fit that leaves before the next fit (a split parent, a repaired leaf's old row). Every
        leaving row of a batch is registered before the batch's first candidate: the batch's Schur-complement factor is built
        against the inverse without the leaving rows, and a row leaving later would make it stale."""
        if not hasattr(self, '_cache'): self.begin_batch()
        if self.P0 is None or rid not in self.ids: return
        assert not self._bids, 'a row leaves after candidates were accepted in this batch: register the leaving rows first'
        pos = self.ids.index(rid)
        if pos not in self._leave: self._leave.append(pos); self._H_leave = None

    def kernel_test(self, rid, idx, w, A_fit, combo=None):
        """Test the candidate row (idx, w) in the kernel's metric and cache its column for the next fit. combo = (ids, coefs)
        when the row is that combination of stored or batch-cached rows (its column is formed from theirs); else the column is
        computed from the points. A_fit: the CSR of the rows of the last fit, in their order. Returns (redundant, rel) with
        rel = s / G_rr."""
        if not hasattr(self, '_cache'): self.begin_batch()
        idx = np.asarray(idx, np.int64); w = np.asarray(w, float)
        if combo is not None:
            c = np.zeros(self.N)
            for j, coef in zip(*combo):
                c += coef * (self._cache[j] if j in self._cache else self.col(self.ids.index(j)))
        else:
            Arow = sp.csr_matrix((w, idx, np.array([0, len(idx)])), shape=(1, self.N)); c = np.empty((1, self.N)); self.columns(Arow, c); c = c[0]
        G_rr = float(w @ c[idx])
        if self.P0 is None:                                  # before the first fit (the pilot rows): nothing to test against
            self._cache[rid] = c; return False, 1.0
        g = np.asarray(A_fit @ c, float)
        if self._leave:                                      # the rows that leave: their border entries are dropped, and the
            L = np.asarray(self._leave, int); g[L] = 0.0     # inverse without them is H' = H - H_L (H_LL)^-1 H_L'
        x = self.solve(g); s = G_rr - float(g @ x)
        if self._leave:
            if self._H_leave is None:
                HL = self.inverse_columns(L); self._H_leave = (HL, HL[L])
            HL, HLL = self._H_leave
            v = x[L]; s += float(v @ np.linalg.solve(HLL, v))
            xb = HL @ np.linalg.solve(HLL, v)                # H g - H' g, needed for the batch cross terms
        else: xb = np.zeros_like(x)
        if self._bids:                                       # the rows accepted earlier in this batch: Schur complement of the batch block
            C = np.array([float(w @ self._cache[j][idx]) for j in self._bids])
            u = C - self._bU.T @ (x - xb)
            z = sl.solve_triangular(self._bL, u, lower=True, check_finite=False); s -= float(z @ z)
        else: u = np.zeros(0); z = np.zeros(0)
        rel = s / G_rr if G_rr > 0 else 0.0
        if s <= self.KERNEL_TOL * G_rr or s <= 100.0 * self.jit:
            return True, rel
        self._cache[rid] = c                                 # accepted: extend the batch's Schur-complement factor
        b = len(self._bids); Lb = np.zeros((b + 1, b + 1)); Lb[:b, :b] = self._bL; Lb[b, :b] = z; Lb[b, b] = np.sqrt(max(s, 1e-300))
        self._bL = Lb; self._bU = np.column_stack([self._bU, g]) if b else g[:, None]; self._bids.append(rid)
        return False, rel

    # ---- the fit ----
    def fit(self, ids, A, y, combos):
        """Fit psi-hat through the rows ids (the method's row order; A: their CSR, y: (m, J)). Against the previous fit, a position
        whose id changed or that is new gets a new column; combos {id: (ids_j, coefs_j)} gives the rows that are the combination
        sum_j coefs_j row_j of other rows. A column cached by `kernel_test` is used as it is. Returns psi-hat (N, J)."""
        m = len(ids); m0 = len(self.ids); assert m >= m0
        cache = getattr(self, '_cache', {})
        new = [r for r in range(m) if r >= m0 or ids[r] != self.ids[r]]
        stored = {i: r for r, i in enumerate(self.ids)}; at = {ids[r]: r for r in new}
        cached = [r for r in new if ids[r] in cache]
        comb = [r for r in new if ids[r] in combos and ids[r] not in cache]; part = np.zeros((len(comb), self.N))
        for c, r in enumerate(comb):                         # the stored part of every combination, before its positions are overwritten
            for j, w in zip(*combos[ids[r]]):
                if j not in at: part[c] += w * self.col(stored[j])
        if m > m0: self.chunks.append(np.empty((m - m0, self.N))); self.start.append(m0)
        for r in cached: self.col(r)[:] = cache[ids[r]]
        direct = [r for r in new if ids[r] not in combos and ids[r] not in cache]
        app = [r for r in direct if r >= m0]; inpl = [r for r in direct if r < m0]
        if app:
            if app == list(range(m0, m)): self.columns(A[app], self.chunks[-1])
            else:
                tmp = np.empty((len(app), self.N)); self.columns(A[app], tmp); self.chunks[-1][np.asarray(app) - m0] = tmp
        if inpl:
            tmp = np.empty((len(inpl), self.N)); self.columns(A[inpl], tmp)
            for c, r in enumerate(inpl): self.col(r)[:] = tmp[c]
        for c, r in enumerate(comb):
            for j, w in zip(*combos[ids[r]]):
                if j in at: assert ids[at[j]] not in combos or ids[at[j]] in cache; part[c] += w * self.col(at[j])
            self.col(r)[:] = part[c]
        Bd = np.empty((m, len(new)))                         # the border of the new positions: G[:, new] = A c_new
        for c, r in enumerate(new): Bd[:, c] = A @ self.col(r)
        self._factor(m, new, Bd)
        self.ids = list(ids)
        self.begin_batch()
        alpha = self.solve(y); psi = self.dot_T(alpha); tol = 1e-8 * max(1.0, float(np.abs(y).max()))
        res = y - A @ psi
        if np.abs(res).max() > tol:                          # rounding of the jittered solve: one refinement step
            alpha = alpha + self.solve(res); psi = self.dot_T(alpha); res = y - A @ psi; self.n_refine += 1
        assert np.abs(res).max() <= tol, ('the kernel interpolant does not reproduce its constraints', float(np.abs(res).max()), tol)
        self.alpha = alpha
        return psi

    def _factor(self, m, new, Bd):
        """Update the block factor of G + jit I at the positions `new` (border columns Bd = G[:, new])."""
        from scipy.linalg.lapack import dpotrf
        if self.P0 is None:                                  # the first fit: the pilot block
            assert new == list(range(m))
            G11 = 0.5 * (Bd + Bd.T); self.P0 = m; self.jit = 1e-10 * max(float(G11.diagonal().max()), 1e-300)
            G11.flat[::m + 1] += self.jit; self.L11 = np.linalg.cholesky(G11); self.R = np.zeros((0, m)); return
        P0 = self.P0; t = m - P0; assert all(r >= P0 for r in new), 'a pilot position changed'
        nt = [r - P0 for r in new]; t0 = len(self.S)
        R = self._view('R', (t, P0), keep=t0 * P0)
        R[nt] = sl.solve_triangular(self.L11, Bd[:P0], lower=True, check_finite=False).T
        S = self._view('S1' if self._is('S0', self.S) else 'S0', (t, t)); S[:t0, :t0] = self.S
        col = Bd[P0:] - R @ R[nt].T                          # Schur complement at the new positions
        S[:, nt] = col; S[nt, :] = col.T; nn = np.ix_(nt, nt); S[nn] = 0.5 * (col[nt] + col[nt].T); S[nt, nt] += self.jit
        self.R, self.S = R, S
        Lb = self._view('L', (t, t)); Lb[:] = S; L22, info = dpotrf(Lb.T, lower=1, overwrite_a=1, clean=0)
        assert info == 0 and np.shares_memory(L22, Lb), ('Cholesky of the tail block', info)
        self.L22 = L22

    def _is(self, name, a):
        b = self._buf.get(name); return b is not None and a.size > 0 and np.shares_memory(a, b)

    def _view(self, name, shape, keep=0):
        """A C-ordered view of a reusable flat buffer whose capacity doubles when exceeded (the tail blocks R, S and its factor)."""
        n = int(np.prod(shape)); b = self._buf.get(name)
        if b is None or b.size < n:
            nb = np.empty(max(n, 2 * (0 if b is None else b.size)))
            if keep: nb[:keep] = b[:keep]
            b = self._buf[name] = nb
        return b[:n].reshape(shape)

    def solve(self, Y):
        """(G + jit I)^-1 Y through the blocks of the factor; Y (m,) or (m, p)."""
        Y = np.asarray(Y, float); one = Y.ndim == 1; Y = Y.reshape(len(Y), -1); P0 = self.P0
        st = lambda L, b, tr: sl.solve_triangular(L, b, lower=True, trans=tr, check_finite=False)
        z1 = st(self.L11, Y[:P0], 0)
        if len(Y) > P0:
            z2 = st(self.L22, Y[P0:] - self.R @ z1, 0); x2 = st(self.L22, z2, 1); x1 = st(self.L11, z1 - self.R.T @ x2, 1)
            X = np.vstack([x1, x2])
        else: X = st(self.L11, z1, 1)
        return X[:, 0] if one else X

    def inverse_columns(self, pos):
        """(G + jit I)^-1 e_pos (m x len(pos)), from the factor of the last fit."""
        E = np.zeros((len(self.ids), len(pos))); E[pos, np.arange(len(pos))] = 1.0
        return self.solve(E)
