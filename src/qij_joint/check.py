"""`python -m qij_joint.check`: the one pass/fail check in the package.

The weighted-mean scale identity: S_btw + S_win == (1/N) sum_i psi_i^2,
where psi_i = x_i - weighted_mean(x) is the weighted mean's exact
analytic influence, evaluated on the pipeline's own final I-VQ bins
(`bin_label`): S_btw = sum_k p_k*psi_bar_k^2, S_win = sum_k p_k*Var_k(psi).
An exact ANOVA decomposition of the same psi values, so it holds to
machine precision for any partition, or the partition arithmetic is
wrong.
"""
from __future__ import annotations

import numpy as np

from .qij import QIJ


def _weighted_mean(X, w):
    return np.array([np.average(X[:, 0], weights=w)])


def main() -> None:
    rng = np.random.default_rng(0)
    X = rng.normal(size=(1000, 1))
    N = X.shape[0]

    res = QIJ().fit(X, _weighted_mean)
    psi = X[:, 0] - res.theta_hat[0]

    labels = res.bin_label[:, 0]
    L = int(res.L[0])
    counts = np.bincount(labels, minlength=L).astype(float)
    p = counts / N
    psi_bar = np.bincount(labels, weights=psi, minlength=L) / counts
    psi_sq_bar = np.bincount(labels, weights=psi ** 2, minlength=L) / counts
    psi_var = psi_sq_bar - psi_bar ** 2

    S_btw = float(np.sum(p * psi_bar ** 2))
    S_win = float(np.sum(p * psi_var))
    S_tot = float(np.mean(psi ** 2))

    rel_err = abs((S_btw + S_win) - S_tot) / abs(S_tot)
    status = 'PASS' if rel_err <= 1e-12 else 'FAIL'
    print(f'S_btw + S_win == (1/N) sum psi^2: rel_err={rel_err:.3e}  {status}')


if __name__ == '__main__':
    main()
