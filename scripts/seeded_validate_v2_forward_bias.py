#!/usr/bin/env python3.9
"""V2's forward-bias line (spec/QIJ_seeded_measured_tree_spec.md section
7, the '(3.3)' paragraph and section 7's V2): on one draw, at one eps,
every seed bin (section 3.1-3.2's own Zador-sized minimax k-means seed)
is measured BOTH by the production forward stencil AND, here only, by
a central stencil (K' extra evaluation pairs), and the relative
difference (U_fwd - U_ctr)/U_ctr per bin and output, and its effect on
V_btw per output, are reported. Not part of `joint_mode='seeded'`'s own
code path (an environment-variable or CLI switch inside `core/seed.py`
would be a branch with no production use, R2) -- a separate script,
reusing `core.seed`'s own seed-count/partition functions and
`core.differences`'s own central-stencil machinery, run by the author
once, offline, never by the package itself.

    PYTHONPATH=src python3.9 scripts/seeded_validate_v2_forward_bias.py \\
        cloudfil p1 --N 10000 --draw 1 --eps 0.01

NOT RUN as part of this build (no estimator draw, per the build's own
instructions); the author runs this once section 7's validation starts.
"""
import os

for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
           'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(_v, '1')

import argparse

import numpy as np

from qij_joint import registry
from qij_joint.core.counter import Counter
from qij_joint.core.differences import (
    central_step, difference, forward_step, perturbed_weights, step_parameter,
)
from qij_joint.core.eta import measure_eta_full
from qij_joint.core.influence_model import fit_influence_model
from qij_joint.core.influence_model import psi0 as _psi0
from qij_joint.core.joint import grow
from qij_joint.core.seed import _seed_count, _seed_partition
from qij_joint.core.xvq import cost_rule_M, run_xvq
from qij_joint.products import QIJ_CANONICAL


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('dataset')
    p.add_argument('estimator')
    p.add_argument('--N', type=int, required=True)
    p.add_argument('--draw', type=int, required=True)
    p.add_argument('--eps', type=float, required=True)
    p.add_argument('--seed-base', type=int, default=0)
    p.add_argument('--gpwidth', choices=['global', 'local'], default='local')
    p.add_argument('--fit-weights', dest='fit_weights', choices=['none', 'mass'], default='mass')
    args = p.parse_args()

    case = registry.case(args.dataset, args.estimator)
    T = case.make_T()
    dseed = args.seed_base + args.draw
    X = case.draw(args.N, dseed)
    N = len(X)
    counter = Counter(T, N)
    eta = T.eta

    theta_hat = np.asarray(counter(X, np.ones(N)), dtype=float)
    eta_full, _n, _ = measure_eta_full(counter, X, theta_hat)
    q_full = len(T.outputs)
    # The production path's own A11 slicing (qij.py): the prototype count,
    # the pilot fit and the seed read only T's MEASURED outputs.
    measured = list(getattr(T, 'measured', range(q_full)))
    q = len(measured)
    M_requested = cost_rule_M(N, q, args.eps)

    Z, inverse = (case.vq_transform(X) if case.vq_transform is not None
                  else (X, lambda A: A))
    Z = np.asarray(Z, dtype=float)
    if Z.ndim == 1:
        Z = Z.reshape(-1, 1)

    xvq, theta_Q, I_proto, _busy, sv = run_xvq(
        Z, inverse, counter, eta, M_requested, dseed, None, 1,
        survey=QIJ_CANONICAL['survey'], X=X,
        quantized_start=QIJ_CANONICAL['quantized_start'], theta_hat=theta_hat,
        eta_rows=eta_full if getattr(counter, 'takes_start', False) else None)
    model, _mbusy = fit_influence_model(
        Z, xvq, I_proto[:, measured], theta_Q[measured], eta, gptrend=QIJ_CANONICAL['gptrend'],
        gpwidth=args.gpwidth, fit_weights=args.fit_weights)
    psi0_all = _psi0(model, Z)

    growth = grow(psi0_all, args.eps, xvq.M_used)
    K_Z, K, lam, v1 = _seed_count(psi0_all, growth.std, X.shape[1], args.eps,
                                   growth.L0, xvq.M_used)
    labels, L, passes, S_pred = _seed_partition(psi0_all, growth.std, growth.std ** 2, v1, K)

    start = theta_hat if getattr(T, 'takes_start', False) else None
    delta_f = forward_step(eta_full)
    delta_c = central_step(eta_full)
    p_mass = np.bincount(labels, minlength=L).astype(float) / N

    U_fwd = np.empty((L, q))
    U_ctr = np.empty((L, q))
    for k in range(L):
        mask = labels == k
        pk = float(p_mass[k])
        t_f = step_parameter(delta_f, pk)
        val = np.asarray(counter(X, _weights(N, mask, t_f), start=start, eta=eta_full),
                          dtype=float)
        U_fwd[k] = (val[measured] - theta_hat[measured]) / t_f

        def evaluate(t, _mask=mask):
            return counter(X, _weights(N, _mask, t), start=start, eta=eta_full)

        U_ctr_full, _D2 = difference(pk, delta_c, evaluate, theta_hat)
        U_ctr[k] = np.asarray(U_ctr_full)[measured]

    rel_diff = (U_fwd - U_ctr) / U_ctr
    V_btw_fwd = np.sum(p_mass[:, None] * U_fwd ** 2, axis=0) / N
    V_btw_ctr = np.sum(p_mass[:, None] * U_ctr ** 2, axis=0) / N
    effect = np.abs(V_btw_fwd - V_btw_ctr) / np.where(V_btw_ctr > 0, V_btw_ctr, np.nan)

    print(f'{args.dataset}/{args.estimator} N={N} draw={args.draw} eps={args.eps} K={L}')
    for j, o in enumerate(T.outputs[i] for i in measured):
        print(f'  {o}: effect on V_btw = {effect[j]:.4%}; '
              f'median|rel diff| = {np.median(np.abs(rel_diff[:, j])):.4%}; '
              f'max|rel diff| = {np.max(np.abs(rel_diff[:, j])):.4%}')
    worst = float(np.nanmax(effect))
    verdict = 'forward' if worst < 0.01 else 'CENTRAL (ruling: exceeds 1%)'
    print(f'seed stays: {verdict} (worst effect {worst:.4%})')


def _weights(N: int, mask: np.ndarray, t: float) -> np.ndarray:
    return perturbed_weights(np.ones(N), mask, t)


if __name__ == '__main__':
    main()
