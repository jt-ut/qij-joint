"""Validate the two changes made to ShearMix2D's cold search (shearmix.py):

  CHANGE 1: `_MIN_CELL_COUNT` reverted 30 -> 3 (the mathematical minimum
  to estimate a 2-D covariance; see that constant's own comment).
  CHANGE 2: partial-EM refinement of every candidate cell before gain-
  scoring (`_refine_candidates`, Verbeek, Vlassis and Kroese 2003).

On exactly the two draws the task calls for (draw 2, draw 1), reusing
`_converge_background`/`_core_cells`/`_insertion_gains`/`_insert_best`/
`_relocate_cores`/`_converge_and_finish` directly (the same pieces
`_cold_search` itself calls) so the growth loop can be logged insertion
by insertion: chosen cell's row count/refined weight/gain, and P2's own
cell's refined gain/rank at each insertion.

No other draws, no oracle/bootstrap/ij/qij pipeline.
"""
import json
import sys
import time
import warnings

import numpy as np
from scipy.cluster.vq import kmeans2

sys.path.insert(0, 'src')

from qij_joint import datasets, gmm
from qij_joint import shearmix as sx
from qij_joint import shearmix_model as sm
from qij_joint.seeding import _cell_stats

N = 10000
KG = 4
ETA = gmm._ETA_DEFAULT
P2_TARGET = (1.32, 0.55)

with open('src/qij_joint/data/truth/cloudfil_u.json') as f:
    truth_json = json.load(f)
theta_truth = np.array(truth_json['p2']['values'][:30])
_, _, mus_truth, covs_truth, _ = sm.unpack(KG, theta_truth)


def cell_of(X, bmu, target, tol=0.05):
    d = np.sqrt(((X - np.asarray(target)) ** 2).sum(axis=1))
    idx = np.nonzero(d <= tol)[0]
    if idx.size == 0:
        return None
    cells, counts = np.unique(bmu[idx], return_counts=True)
    return int(cells[np.argmax(counts)])


def run_draw(seed):
    print(f'\n========== draw seed={seed} ==========', flush=True)
    X = datasets.cloudfil_G_U_P3_v1(N, seed)
    w = np.ones(N)
    prep = sm.prepare(X)
    pen = sm.penalty_setup(prep, w)
    W = pen.W

    t0 = time.time()
    bg = sx._converge_background(prep, w, pen, ETA)
    assert bg is not None, 'background degenerated'
    theta_bg, ll_bg, fil_bg = bg
    print(f'background converged: ll={ll_bg:.6f} (t={time.time()-t0:.1f}s)', flush=True)

    # Over-segmentation, replicated (not just _core_cells) so we also get
    # each cell's own row COUNT for the insertion log.
    Z = sx._unsheared_coords(X, prep.Bx, fil_bg)
    M = min(sx._GREEDY_CELL_FACTOR * KG, max(KG, N - 1))
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        centres_z, bmu = kmeans2(Z, M, iter=sx._KMEANS_ITER, minit='++',
                                  seed=np.random.default_rng(seed))
    count, wsum, _, mean, cov3 = _cell_stats(X, w, centres_z, bmu, M)
    eligible = count >= sx._MIN_CELL_COUNT
    print(f'cells: M={M} eligible={eligible.sum()} (_MIN_CELL_COUNT={sx._MIN_CELL_COUNT})',
          flush=True)

    j_p2 = cell_of(X, bmu, P2_TARGET)
    print(f'P2 truth-matched cell = {j_p2} (rows={count[j_p2] if j_p2 is not None else None})',
          flush=True)

    theta, ll = theta_bg, ll_bg
    t_grow0 = time.time()
    for step in range(1, KG):
        t_step0 = time.time()
        gains, new_w, new_mu, new_cov = sx._insertion_gains(
            prep, w, pen, step, theta, ll, eligible, wsum, mean, cov3, W, ETA)
        order = np.argsort(-gains)
        j_best = int(order[0])
        rank_p2 = int(np.nonzero(order == j_p2)[0][0]) + 1 if j_p2 is not None else None
        gain_p2 = float(gains[j_p2]) if j_p2 is not None else float('nan')
        det_best = float(np.linalg.det(new_cov[j_best]))
        print(f'  insertion {step} (Kg_cur={step}): chosen cell={j_best} '
              f'rows={int(count[j_best])} refined_w={new_w[j_best]:.5f} '
              f'det(cov)={det_best:.4e} gain={gains[j_best]:.6f} '
              f'(refine t={time.time()-t_step0:.1f}s)', flush=True)
        print(f'    P2 cell={j_p2} rows={int(count[j_p2]) if j_p2 is not None else -1} '
              f'rank={rank_p2} refined_gain={gain_p2:.6f}', flush=True)

        out = sx._insert_best(prep, w, pen, step, theta, ll, wsum, mean, cov3, eligible, W, ETA)
        assert out is not None, f'insertion {step} degenerated'
        theta, ll, j_chosen = out
        assert j_chosen == j_best, (j_chosen, j_best)
        print(f'    re-converged: ll={ll:.6f}', flush=True)
    print(f'grow_cores total wall={time.time()-t_grow0:.1f}s', flush=True)

    t_reloc0 = time.time()
    theta, ll_reloc = sx._relocate_cores(prep, w, pen, KG, theta, ll, wsum, mean, cov3,
                                          eligible, W, ETA)
    print(f'relocation: ll={ll_reloc:.6f} (t={time.time()-t_reloc0:.1f}s)', flush=True)

    fit, status = sx._converge_and_finish(prep, w, pen, KG, theta, ETA)
    wall_cold = time.time() - t0
    ll_cold = fit['ll'] if fit is not None else float('nan')
    print(f'cold fit: status={status} wall={wall_cold:.1f}s ll={ll_cold:.6f}', flush=True)

    t0c = time.time()
    fit_cont, status_cont = sx._converge_and_finish(prep, w, pen, KG, theta_truth, ETA)
    wall_cont = time.time() - t0c
    ll_cont = fit_cont['ll'] if fit_cont is not None else float('nan')
    print(f'continuation-from-truth: status={status_cont} wall={wall_cont:.1f}s ll={ll_cont:.6f}',
          flush=True)

    gap = (ll_cold - ll_cont) if (np.isfinite(ll_cold) and np.isfinite(ll_cont)) else float('nan')
    passed = np.isfinite(gap) and gap >= -1e-8
    print(f'search gap (cold - continuation) = {gap:.3e} per unit weight '
          f'({"PASS" if passed else "FAIL"}, threshold -1e-8)', flush=True)

    if fit is not None:
        pis_g, pi_f, mus, covs, fil = sm.unpack(KG, fit['theta'])
        pis_l, mus_l, covs_l, order_l = sx._label_gaussians(
            pis_g, mus, covs, 'reference', (mus_truth, covs_truth))
        print('recovered (labelled to truth order cloud, P1, P2, P3):', flush=True)
        for lbl, pi, mu in zip(('cloud', 'P1', 'P2', 'P3'), pis_l, mus_l):
            print(f'  {lbl}: pi={pi:.5f} mu={mu}', flush=True)
        print(f'  filament: m={fil[0]:.4f} s2x={fil[1]:.4f} b={fil[2:5]} s2p={fil[5]:.6f}',
              flush=True)

    return dict(seed=seed, status=status, wall_cold=wall_cold, ll_cold=ll_cold,
                ll_cont=ll_cont, gap=gap, passed=passed)


seeds = [int(a) for a in sys.argv[1:]] or [2, 1]
results = {}
for seed in seeds:
    results[seed] = run_draw(seed)

print('\n========== summary ==========', flush=True)
for seed, r in results.items():
    print(seed, r, flush=True)
