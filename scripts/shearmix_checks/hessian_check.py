"""Check script for the analytic-Hessian task (ShearMix2D, time-boxed):
verifies `shearmix_model.hessian`/`info` against the old finite-
difference `_info_fd`, verifies the u-space Hessian used by the Newton
finish (`shearmix._hess_u_analytic`, via `shearmix_model.chain`) against
the old `shearmix._hess_u_fd`, verifies influence is unchanged, and
times bootstrap-style replicates before/after against the GMM2D
yardstick.

Run: PYTHONPATH=src OMP_NUM_THREADS=1 python3.9 scripts/shearmix_checks/hessian_check.py
"""
import json
import os
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.join(_HERE, '..', '..', 'src')
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from qij_joint import datasets  # noqa: E402
from qij_joint import gmm  # noqa: E402
from qij_joint import shearmix  # noqa: E402
from qij_joint import shearmix_model as sm  # noqa: E402
from qij_joint.cloudfil import P2Mixture  # noqa: E402
from qij_joint.cloudfil_u import P2ShearMixture  # noqa: E402

_TRUTH_SM = os.path.join(_SRC, 'qij_joint', 'data', 'truth', 'cloudfil_u.json')
_TRUTH_GMM = os.path.join(_SRC, 'qij_joint', 'data', 'truth', 'cloudfil.json')

KG = 4
BLOCK_NAMES_SM = (['pi'] + [f'mu{k}' for k in range(KG)] + [f'S{k}' for k in range(KG)]
                  + ['fil'])


def _blocks_sm(Kg):
    """index lists per named block, matching BLOCK_NAMES_SM."""
    out = {'pi': list(range(Kg))}
    for k in range(Kg):
        out[f'mu{k}'] = list(sm._idx_mu(Kg, k))
        out[f'S{k}'] = list(sm._idx_S(Kg, k))
    out['fil'] = list(sm._idx_fil(Kg))
    return out


def max_rel_block(A_new, A_old, blocks):
    out = {}
    for name, idx in blocks.items():
        sub_new = A_new[np.ix_(idx, idx)]
        sub_old = A_old[np.ix_(idx, idx)]
        scale = np.maximum(np.abs(sub_old), 1e-12)
        out[name] = float(np.max(np.abs(sub_new - sub_old) / scale))
    return out


def main():
    t0 = time.time()
    N = 10000
    seed = 1
    X = datasets.cloudfil_G_U_P3_v1(N, seed)
    w = np.ones(N)
    prep = sm.prepare(X)
    pen = sm.penalty_setup(prep, w)

    with open(_TRUTH_SM) as f:
        truth_sm = json.load(f)
    truth_raw = np.array(truth_sm['p2']['values'][:30], dtype=float)

    # ------------------------------------------------------------------
    # A fitted theta_hat: continuation from truth (fast; NOT the full
    # cold search), the shape of fit every bootstrap/QIJ replicate
    # actually runs (task's own WHY).
    # ------------------------------------------------------------------
    est = P2ShearMixture()
    theta_hat_full = est(X, w, prep=prep, start=truth_raw, eta=1e-12)
    assert np.all(np.isfinite(theta_hat_full)), "continuation from truth failed to converge"
    theta_hat = theta_hat_full[:30]
    print(f"[setup] continuation-from-truth theta_hat converged ({time.time()-t0:.1f}s so far)")

    # ------------------------------------------------------------------
    # (a) hessian vs _info_fd at theta_hat; u-space Hessian vs old
    # shearmix._hess_u_fd.
    # ------------------------------------------------------------------
    A_new = sm.info(prep, w, KG, theta_hat, pen)
    A_old = sm._info_fd(prep, w, KG, theta_hat, pen)
    blocks = _blocks_sm(KG)
    rel_theta = max_rel_block(A_new, A_old, blocks)
    print("\n(a) theta-space A: max relative error per block (analytic vs FD):")
    for name, val in rel_theta.items():
        print(f"    {name:6s}  {val:.3e}")

    u = sm.to_u(KG, theta_hat)
    g_theta = sm.grad(prep, w, KG, theta_hat, pen)
    H_theta_new = sm.hessian(prep, w, KG, theta_hat, pen)
    _, H_u_new = sm.chain(KG, u, g_theta, H_theta_new)
    H_u_old = shearmix._hess_u_fd(prep, w, pen, KG, u)
    scale_u = np.maximum(np.abs(H_u_old), 1e-12)
    rel_u = float(np.max(np.abs(H_u_new - H_u_old) / scale_u))
    print(f"\n(a) u-space H_u: max relative error (analytic-via-chain vs old _hess_u_fd) = {rel_u:.3e}")

    # ------------------------------------------------------------------
    # (c) influence unchanged.
    # ------------------------------------------------------------------
    psi = sm.score_rows(prep, w, KG, theta_hat, pen)
    IF_new = np.linalg.solve(A_new, psi.T).T
    IF_old = np.linalg.solve(A_old, psi.T).T
    rel_if = float(np.max(np.abs(IF_new - IF_old)) / np.max(np.abs(IF_old)))
    print(f"\n(c) influence: max|IF_new-IF_old| / max|IF_old| = {rel_if:.3e}")

    # ------------------------------------------------------------------
    # (b) Fits unchanged: a continuation (start = truth, SAME EM
    # trajectory since EM is untouched by this change) run through the
    # Newton finish with the NEW analytic Hessian vs the OLD FD Hessian,
    # monkeypatching the finish's own Hessian call back to the old path.
    # (The FULL cold search is skipped here -- its own docstring reports
    # 530-580s wall time per draw, which does not fit this task's time
    # box twice over; the cold search's own EM phases -- growth,
    # relocation -- never call hessian()/info() at all, only the final
    # `_converge_and_finish`'s Newton finish does, and that finish is
    # exactly what this comparison exercises.)
    # ------------------------------------------------------------------
    prep_c = est._mix.prepare(X)
    pen_c = sm.penalty_setup(prep_c, w)
    run = shearmix._run_em(prep_c, w, pen_c, KG, truth_raw, est._mix.eta)
    assert run is not None and run['status'] == 'converged'

    fit_new, status_new = shearmix._finish(prep_c, w, pen_c, KG, run['theta'],
                                             run['last_step_u'], est._mix.eta)

    _orig_hess_analytic = shearmix._hess_u_analytic

    def _hess_u_fd_adapter(prep_, w_, pen_, Kg_, u_, g_theta_, theta_):
        return shearmix._hess_u_fd(prep_, w_, pen_, Kg_, u_)

    shearmix._hess_u_analytic = _hess_u_fd_adapter
    try:
        fit_old, status_old = shearmix._finish(prep_c, w, pen_c, KG, run['theta'],
                                                  run['last_step_u'], est._mix.eta)
    finally:
        shearmix._hess_u_analytic = _orig_hess_analytic

    theta_diff = float(np.max(np.abs(fit_new['theta'] - fit_old['theta'])))
    print(f"\n(b) Newton finish (continuation from truth): status_new={status_new} "
          f"n_iter_new={fit_new['n_iter_newton']}  status_old={status_old} "
          f"n_iter_old={fit_old['n_iter_newton']}  max|theta_new-theta_old|={theta_diff:.3e}")
    print("    (full cold-search before/after comparison SKIPPED -- see report)")

    # ------------------------------------------------------------------
    # (d) Cost: 20 bootstrap-style replicates, ShearMix2D (new) vs the
    # GMM2D yardstick. ShearMix2D's "cold theta_hat" is approximated by
    # the continuation-from-truth theta_hat_full above (the real cold
    # search costs 530-580s per draw by its own docstring -- doubling
    # that for a before/after timing comparison does not fit this time
    # box; see report).
    # ------------------------------------------------------------------
    rng = np.random.default_rng(0)
    n_rep = 20

    def time_replicates(est_, X_, prep_, start_):
        times = []
        for _ in range(n_rep):
            wb = rng.multinomial(N, np.full(N, 1.0 / N)).astype(float)
            t1 = time.perf_counter()
            theta_b = est_(X_, wb, prep=prep_, start=start_, eta=est_.eta)
            t2 = time.perf_counter()
            times.append(t2 - t1)
            assert np.all(np.isfinite(theta_b)), "bootstrap replicate failed to converge"
        return times

    t_sm = time_replicates(est, X, prep, theta_hat_full)
    print(f"\n(d) ShearMix2D (new analytic Hessian): mean {np.mean(t_sm):.4f}s/replicate "
          f"(n={n_rep}, min {np.min(t_sm):.4f}, max {np.max(t_sm):.4f})")

    with open(_TRUTH_GMM) as f:
        truth_gmm = json.load(f)
    truth_gmm_raw = np.array(truth_gmm['p2']['values'][:59], dtype=float)

    X_gmm = datasets.cloudfil_G_B6_P3_v1(N, seed)
    w_gmm = np.ones(N)
    est_gmm = P2Mixture()
    prep_gmm = est_gmm.prepare(X_gmm)
    theta_hat_gmm = est_gmm(X_gmm, w_gmm, prep=prep_gmm, start=truth_gmm_raw, eta=1e-12)
    assert np.all(np.isfinite(theta_hat_gmm))

    t_gmm = time_replicates(est_gmm, X_gmm, prep_gmm, theta_hat_gmm)
    print(f"(d) GMM2D (yardstick):                 mean {np.mean(t_gmm):.4f}s/replicate "
          f"(n={n_rep}, min {np.min(t_gmm):.4f}, max {np.max(t_gmm):.4f})")

    print(f"\n[total script wall time: {time.time()-t0:.1f}s]")


if __name__ == '__main__':
    main()
