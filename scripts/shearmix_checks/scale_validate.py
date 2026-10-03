"""Validate the scale-separation redesign of ShearMix2D's cold search
(shearmix.py: `_converge_background`/`_core_cells`/`_grow_cores`/
`_relocate_cores`/`_cold_search`), on exactly the two draws the task
calls for:

  - draw 2 (a FAILING draw under the OLD code: search gap -0.0040, P2
    missed): cold fit status, wall time, search gap vs the continuation
    from the truth (gap must be >= -1e-8/unit weight), recovered
    P1/P2/P3.
  - draw 1 (seed 1, a PASSING draw under the OLD code): no-regression
    check, same numbers.

No other draws, no oracle/bootstrap/ij/qij pipeline.
"""
import json
import sys
import time

import numpy as np

sys.path.insert(0, 'src')

from qij_joint import datasets
from qij_joint.shearmix import ShearMix2D
from qij_joint import shearmix_model as sm

N = 10000
KG = 4

with open('src/qij_joint/data/truth/cloudfil_u.json') as f:
    truth_json = json.load(f)
theta_truth = np.array(truth_json['p2']['values'][:30])
_, _, mus_truth, covs_truth, _ = sm.unpack(KG, theta_truth)
print('truth mus (cloud, P1, P2, P3):')
for lbl, mu in zip(('cloud', 'P1', 'P2', 'P3'), mus_truth):
    print(f'  {lbl}: {mu}')


def run_draw(seed):
    print(f'\n========== draw seed={seed} ==========')
    X = datasets.cloudfil_G_U_P3_v1(N, seed)
    w = np.ones(N)

    est = ShearMix2D(Kg=KG, reference=(mus_truth, covs_truth))

    t0 = time.time()
    theta_cold = est(X, w)
    wall_cold = time.time() - t0
    status_cold = est.last_fit_info.status
    ll_cold = est.loglik(X, w, theta_cold) if status_cold == 'converged' else float('nan')
    search_info = est.last_fit_info.search
    print(f'cold fit: status={status_cold} wall={wall_cold:.1f}s ll={ll_cold:.6f}')
    print(f'  search diagnostics: {search_info}')

    t0 = time.time()
    theta_cont = est(X, w, start=theta_truth)
    wall_cont = time.time() - t0
    status_cont = est.last_fit_info.status
    ll_cont = est.loglik(X, w, theta_cont) if status_cont == 'converged' else float('nan')
    print(f'continuation-from-truth: status={status_cont} wall={wall_cont:.1f}s ll={ll_cont:.6f}')

    gap = ll_cold - ll_cont if np.isfinite(ll_cold) and np.isfinite(ll_cont) else float('nan')
    print(f'search gap (cold - continuation) = {gap:.3e} per unit weight '
          f'({"PASS" if np.isfinite(gap) and gap >= -1e-8 else "FAIL"}, threshold -1e-8)')

    if status_cold == 'converged':
        pis_g, pi_f, mus, covs, fil = sm.unpack(KG, theta_cold)
        print('recovered (labelled to truth order cloud, P1, P2, P3):')
        for lbl, pi, mu in zip(('cloud', 'P1', 'P2', 'P3'), pis_g, mus):
            print(f'  {lbl}: pi={pi:.5f} mu={mu}')
        print(f'  filament: m={fil[0]:.4f} s2x={fil[1]:.4f} b={fil[2:5]} s2p={fil[5]:.6f}')
    return dict(seed=seed, status_cold=status_cold, wall_cold=wall_cold, ll_cold=ll_cold,
                ll_cont=ll_cont, gap=gap)


results = {}
for seed in (2, 1):
    results[seed] = run_draw(seed)

print('\n========== summary ==========')
for seed, r in results.items():
    print(seed, r)
