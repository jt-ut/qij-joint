"""GATE check (run BEFORE touching shearmix.py): does scoring candidate
small Gaussians against the CONVERGED background (one Gaussian + the
filament, no cores) rank the three true cores' own over-segmentation
cells above every other cell, on draw 2 of cloudfil_G_U_P3_v1 (a FAILING
draw under the current code: search gap -0.0040, P2 missed)?

Mirrors the existing `shearmix._greedy_insert` machinery exactly (same
cell geometry, same `_insertion_gains`/`_trial_insertion_theta`), with
ONE change: the background (Kg=1 Gaussian + filament) is run to FULL EM
convergence before scoring, instead of the current code's 20-step
budget -- that is the whole candidate fix, tested here before it is
built into the module.

No changes to shearmix.py, shearmix_model.py, or any other file; this
script only imports and calls their existing private functions.
"""
import sys
import time

import numpy as np

sys.path.insert(0, 'src')

from qij_joint import datasets, gmm, shearmix as sx, shearmix_model as sm
from qij_joint.seeding import _cell_stats

SEED = 2
N = 10000
KG_TARGET = 4
ETA = gmm._ETA_DEFAULT


def cell_of(X, bmu, target, tol=0.05):
    """Majority k-means cell among rows within `tol` of `target` (x,y)."""
    d = np.sqrt(((X - np.asarray(target)) ** 2).sum(axis=1))
    idx = np.nonzero(d <= tol)[0]
    if idx.size == 0:
        return None, 0
    cells, counts = np.unique(bmu[idx], return_counts=True)
    j = cells[np.argmax(counts)]
    return int(j), int(idx.size)


def rank_of(gains, j):
    if j is None:
        return None
    order = np.argsort(-gains)
    return int(np.nonzero(order == j)[0][0]) + 1  # 1-based


def main():
    t0 = time.time()
    X = datasets.cloudfil_G_U_P3_v1(N, SEED)
    w = np.ones(N)
    prep = sm.prepare(X)
    pen = sm.penalty_setup(prep, w)
    W = pen.W

    fil0 = sx._filament_seed(prep, w)
    pi_f = sx._PI_F_SEED
    xbar = (w @ X) / W
    theta_bg0 = sm.pack(1, np.array([1.0 - pi_f]), xbar[None, :],
                        pen.Scov[None, :, :], fil0)
    ll0 = sm.penalized_ll(prep, w, 1, theta_bg0, pen)
    print(f'background seed ll0={ll0:.6f}')

    result = sx._em_accelerated(prep, w, pen, 1, theta_bg0, ll0, ETA, budget=None)
    assert result is not None, 'background EM degenerated'
    theta_bg, ll_bg, n_used, status, score_scaled, last_step_u = result
    print(f'background EM: status={status} n_iter={n_used} ll={ll_bg:.6f} '
          f'score_scaled={score_scaled:.3e} wall={time.time() - t0:.1f}s')

    _, _, mus_bg, covs_bg, fil_bg = sm.unpack(1, theta_bg)
    print(f'background Gaussian: mu={mus_bg[0]}, pi={theta_bg[0]:.4f}')
    print(f'background filament: m={fil_bg[0]:.4f} s2x={fil_bg[1]:.4f} '
          f'b={fil_bg[2:5]} s2p={fil_bg[5]:.6f}')

    Z = sx._unsheared_coords(X, prep.Bx, fil_bg)
    M = min(sx._GREEDY_CELL_FACTOR * KG_TARGET, max(KG_TARGET, N - 1))
    import warnings
    from scipy.cluster.vq import kmeans2
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        centres_z, bmu = kmeans2(Z, M, iter=sx._KMEANS_ITER, minit='++',
                                  seed=np.random.default_rng(0))
    count, wsum, _, mean, cov3 = _cell_stats(X, w, centres_z, bmu, M)
    eligible = count >= sx._MIN_CELL_COUNT
    print(f'cells: M={M} eligible={eligible.sum()}')

    gains1 = sx._insertion_gains(prep, w, pen, 1, theta_bg, ll_bg, eligible,
                                  wsum, mean, cov3, W)

    j_p2, n_p2 = cell_of(X, bmu, (1.32, 0.55))
    j_p1, n_p1 = cell_of(X, bmu, (0.0, 0.0))
    j_p3, n_p3 = cell_of(X, bmu, (-1.2, -1.1))
    r_p2 = rank_of(gains1, j_p2)
    r_p1 = rank_of(gains1, j_p1)
    r_p3 = rank_of(gains1, j_p3)
    core_cells = {c for c in (j_p2, j_p1, j_p3) if c is not None}
    best_noncore = -np.inf
    order1 = np.argsort(-gains1)
    for j in order1:
        if j not in core_cells and np.isfinite(gains1[j]):
            best_noncore = gains1[j]
            break

    print('\n=== ROUND 1: cores vs converged background ===')
    print(f'P2 cell={j_p2} (n_match={n_p2}) rank={r_p2} gain={gains1[j_p2] if j_p2 is not None else float("nan"):.6f}')
    print(f'P1 cell={j_p1} (n_match={n_p1}) rank={r_p1} gain={gains1[j_p1] if j_p1 is not None else float("nan"):.6f}')
    print(f'P3 cell={j_p3} (n_match={n_p3}) rank={r_p3} gain={gains1[j_p3] if j_p3 is not None else float("nan"):.6f}')
    print(f'best non-core cell gain = {best_noncore:.6f}')
    top3 = order1[:3]
    print(f'top-3 cell indices by gain: {list(top3)}; are these exactly the 3 core cells? '
          f'{set(top3.tolist()) == core_cells}')

    gate_pass = set(top3.tolist()) == core_cells
    if not gate_pass:
        print('\nGATE FAILS: the three cores are NOT the top-3 candidates. STOP.')
        return

    print('\nGATE ROUND 1 PASSES. Adding the top-ranked core and re-converging...')
    j_top = int(order1[0])
    new_w = max(wsum[j_top] / W, sx._MIN_INSERT_SHARE / W)
    cov_top = np.array([[cov3[j_top, 0], cov3[j_top, 1]], [cov3[j_top, 1], cov3[j_top, 2]]])
    theta2_0 = sx._trial_insertion_theta(1, theta_bg, new_w, mean[j_top], cov_top)
    ll2_0 = sm.penalized_ll(prep, w, 2, theta2_0, pen)
    result2 = sx._em_accelerated(prep, w, pen, 2, theta2_0, ll2_0, ETA, budget=None)
    assert result2 is not None, 'round-2 EM degenerated'
    theta2, ll2, n_used2, status2, score2, _ = result2
    print(f'round-2 EM (Kg=2): status={status2} n_iter={n_used2} ll={ll2:.6f}')

    eligible2 = eligible.copy()
    gains2 = sx._insertion_gains(prep, w, pen, 2, theta2, ll2, eligible2,
                                  wsum, mean, cov3, W)
    remaining = {c for c in (j_p2, j_p1, j_p3) if c is not None} - {j_top}
    order2 = np.argsort(-gains2)
    top2 = order2[:2]
    print('\n=== ROUND 2: remaining 2 cores vs converged (cloud+1 core+fil) ===')
    which = {j_p2: 'P2', j_p1: 'P1', j_p3: 'P3'}
    print(f'cell added in round 1: {j_top} ({which.get(j_top, "?")})')
    for c in remaining:
        r = rank_of(gains2, c)
        print(f'{which.get(c)} cell={c} rank={r} gain={gains2[c]:.6f}')
    print(f'top-2 cell indices by gain: {list(top2)}; equal to remaining 2 cores? '
          f'{set(top2.tolist()) == remaining}')
    print(f'\nwall time total: {time.time() - t0:.1f}s')


if __name__ == '__main__':
    main()
