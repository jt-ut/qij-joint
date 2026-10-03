"""REVIEWER diagnostic: is review_influence.py's handful of outlier
relative errors (p2_2 mu-block, p1 S-block, p2_2's p2_pa derived output)
explained by shearmix_model.info()'s finite-difference Hessian step
(rel_step, default 1e-5) rather than a wrong psi formula? Recomputes A
at several rel_step values for the SAME converged theta_hat/psi and
checks whether solve(A, psi.T).T for those two specific rows converges
as rel_step shrinks (before FD-in-theta noise takes over), and reports
cond(A) at each step.

Run: PYTHONPATH=src python3.9 scripts/shearmix_checks/review_hessian_step.py
"""
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.join(_HERE, '..', '..', 'src')
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

import json  # noqa: E402

from qij_joint import datasets  # noqa: E402
from qij_joint.shearmix import ShearMix2D  # noqa: E402
from qij_joint import shearmix_model as sm  # noqa: E402

_TRUTH = os.path.join(_SRC, 'qij_joint', 'data', 'truth', 'cloudfil_u.json')


def main():
    N = 10000
    seed = 1
    X = datasets.cloudfil_G_U_P3_v1(N, seed)
    w = np.ones(N)
    with open(_TRUTH) as f:
        truth = json.load(f)
    truth_raw = np.array(truth['p2']['values'][:30], dtype=float)
    Kg = 4

    est = ShearMix2D(Kg=Kg, seed=0)
    prep = est.prepare(X)
    theta_hat, IF_default = est.fit_and_influence(X, w, prep=prep, start=truth_raw,
                                                    eta=1e-12)
    assert np.all(np.isfinite(theta_hat))
    pen = sm.penalty_setup(prep, w)
    psi = sm.score_rows(prep, w, Kg, theta_hat, pen)

    rows = dict(p2_2=6059, p1=5995)
    # mu-block index for component 2 (the P2-ish fitted Gaussian), S-block
    # for whichever component p1's row loads onto -- just report the full
    # vector's max relative CHANGE in the mu/S blocks between successive
    # rel_steps, plus explicit values at the specific coordinates flagged
    # by review_influence.py: component index 2 mu (for p2_2) and
    # component index 1 S (for p1), using the theta_hat Gaussian order
    # (unpacked, pre any further relabel -- theta_hat is already labelled
    # against truth_raw by construction of the continuation fit).

    for rel_step in (1e-4, 1e-5, 1e-6, 1e-7, 1e-8):
        A = sm.info(prep, w, Kg, theta_hat, pen, rel_step=rel_step)
        cond = np.linalg.cond(A)
        IF = np.linalg.solve(A, psi.T).T
        print('rel_step=%.0e  cond(A)=%.4e' % (rel_step, cond))
        for name, i in rows.items():
            print('  row %-6s IF[mu-block]=%s' %
                  (name, np.round(IF[i, Kg:3 * Kg], 6)))
            print('  row %-6s IF[S-block] =%s' %
                  (name, np.round(IF[i, 3 * Kg:6 * Kg], 6)))


if __name__ == '__main__':
    main()
