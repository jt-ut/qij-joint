"""REVIEWER check (task 3): confirm the FD-vs-IF/N scaling convention on
GMM2D itself (the OLD dataset, cloudfil_G_B6_P3_v1 / P2Mixture), as a
control for scripts/shearmix_checks/review_influence.py's methodology.
ONE draw (seed=1, N=10000), 2 rows only.

Run: PYTHONPATH=src python3.9 scripts/shearmix_checks/review_influence_gmm2d.py [eta]
"""
import os
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.join(_HERE, '..', '..', 'src')
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from qij_joint import datasets, registry  # noqa: E402
from qij_joint.cloudfil import P2Mixture  # noqa: E402


def main():
    eta_override = float(sys.argv[1]) if len(sys.argv) > 1 else None
    N = 10000
    seed = 1
    X = datasets.cloudfil_G_B6_P3_v1(N, seed)
    w = np.ones(N)

    truth_vals = registry.truth('cloudfil', 'p2')
    truth_raw = np.array(truth_vals[:59], dtype=float)

    est = P2Mixture()
    prep = est.prepare(X)
    print('eta_override =', eta_override)

    t0 = time.time()
    theta_hat = est(X, w, prep=prep, start=truth_raw, eta=eta_override)
    IF = est.influence(X, w, prep=prep, start=truth_raw, eta=eta_override)
    t1 = time.time()
    print('wall time: %.2f s' % (t1 - t0))
    print('last_fit_info status:', est.last_fit_info.status)
    assert np.all(np.isfinite(theta_hat)) and np.all(np.isfinite(IF))

    # Two rows: one near a core, one far (cloud-like), by simple distance.
    d0 = np.hypot(X[:, 0], X[:, 1])
    row_far = int(np.argmax(d0))
    # a row near the P2 bead's mean (1.32, 0.55)
    d_p2 = np.hypot(X[:, 0] - 1.32, X[:, 1] - 0.55)
    row_p2 = int(np.argmin(d_p2))

    rows = dict(p2=row_p2, far=row_far)
    print('rows:', {k: (v, X[v]) for k, v in rows.items()})

    for name, i in rows.items():
        print('\nRow %s (idx %d):' % (name, i))
        for h in (1e-3, 1e-4):
            w2 = w.copy()
            w2[i] += h
            theta_pert = est(X, w2, prep=prep, start=theta_hat, eta=eta_override)
            if not np.all(np.isfinite(theta_pert)):
                print('  h=%.0e: perturbed fit FAILED' % h)
                continue
            FD = (theta_pert - theta_hat) / h
            IF_pred = IF[i] / N
            num = np.abs(FD - IF_pred)
            den = np.maximum(np.abs(FD), np.abs(IF_pred))
            den = np.where(den < 1e-12, 1.0, den)
            rel = num / den
            print('  h=%.0e: max rel err = %.3e (at output %s)' %
                  (h, rel.max(), est.outputs[int(np.argmax(rel))]))


if __name__ == '__main__':
    main()
