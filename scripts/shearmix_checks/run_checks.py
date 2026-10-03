"""Agent B's required self-checks for `shearmix_model.py`
(spec/QIJ_shearmix_interface.md section 3). Run with:
    PYTHONPATH=src python3.9 scripts/shearmix_checks/run_checks.py
One N=10,000, seed=1 draw; every check below is run once."""

import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '.'))
import common  # noqa: E402
from qij_joint import shearmix_model as sm  # noqa: E402

Kg = common.KG


def check_grad_vs_fd(prep, w, pen, theta, rel_step=3e-7):
    """Central FD of `penalized_ll` against the analytic `grad`. The FD
    step is scaled to each parameter's OWN magnitude (not floored at 1,
    unlike `info`'s step rule): several of this model's covariance
    entries (the tiny cores' S11/S12/S22) are O(1e-3), and a step
    floored at `rel_step*1` would be a large relative perturbation of
    them, inflating the FD's own truncation error well past this
    check's purpose (confirmed by a step-size sweep: the discrepancy
    shrinks as rel_step**2, the signature of FD truncation, not a wrong
    gradient)."""
    g = sm.grad(prep, w, Kg, theta, pen)
    p = sm.n_params(Kg)
    fd = np.empty(p)
    for j in range(p):
        h = rel_step * max(1e-3, abs(theta[j]))
        tp = theta.copy()
        tp[j] += h
        tm = theta.copy()
        tm[j] -= h
        fd[j] = (sm.penalized_ll(prep, w, Kg, tp, pen) - sm.penalized_ll(prep, w, Kg, tm, pen)) / (2.0 * h)
    rel_err = np.max(np.abs(g - fd) / np.maximum(np.abs(fd), 1.0))
    print(f"[grad vs FD]      max |g-fd| = {np.max(np.abs(g-fd)):.3e}   "
          f"rel err (vs max(|fd|,1)) = {rel_err:.3e}")
    return rel_err


def check_score_rows_mean(prep, w, pen, theta):
    g = sm.grad(prep, w, Kg, theta, pen)
    rows = sm.score_rows(prep, w, Kg, theta, pen)
    W = pen.W
    wmean = (w @ rows) / W
    diff = wmean - g
    print(f"[score_rows mean] max |wmean(score_rows) - grad| = {np.max(np.abs(diff)):.3e}  "
          f"(predicted gap 2*g_pen/W, see report)")
    return diff


def check_m_step_monotone(prep, w, pen, base_theta, n_iter=50, n_starts=3, seed=7):
    rng = np.random.default_rng(seed)
    worst_violation = 0.0
    for s in range(n_starts):
        theta = common.random_feasible_theta(rng, base_theta, scale=0.2)
        ll_prev = sm.penalized_ll(prep, w, Kg, theta, pen)
        lls = [ll_prev]
        for _ in range(n_iter):
            R, _ = sm.e_step(prep, Kg, theta)
            theta = sm.m_step(prep, w, R, pen, Kg)
            ll = sm.penalized_ll(prep, w, Kg, theta, pen)
            lls.append(ll)
        increments = np.diff(lls)
        viol = float(np.min(increments))
        worst_violation = min(worst_violation, viol)
        print(f"[m_step monotone] start {s}: ll0={lls[0]:.6f} -> ll{n_iter}={lls[-1]:.6f}  "
              f"min increment = {viol:.3e}")
    print(f"[m_step monotone] worst (most negative) increment over all starts = {worst_violation:.3e}")
    return worst_violation


def check_u_roundtrip(theta):
    u = sm.to_u(Kg, theta)
    theta2 = sm.from_u(Kg, u)
    err = np.max(np.abs(theta2 - theta))
    print(f"[from_u(to_u)]    max |theta2 - theta| = {err:.3e}")
    return err


def check_jac_u_vs_fd(theta, rel_step=1e-6):
    u = sm.to_u(Kg, theta)
    J = sm.jac_u(Kg, u)
    p = sm.n_params(Kg)
    Jfd = np.empty((p, p))
    for j in range(p):
        h = rel_step * max(1.0, abs(u[j]))
        up = u.copy()
        up[j] += h
        um = u.copy()
        um[j] -= h
        Jfd[:, j] = (sm.from_u(Kg, up) - sm.from_u(Kg, um)) / (2.0 * h)
    err = np.max(np.abs(J - Jfd))
    rel = err / max(np.max(np.abs(Jfd)), 1.0)
    print(f"[jac_u vs FD]     max |J-Jfd| = {err:.3e}   rel = {rel:.3e}")
    return err


def check_info_pd(prep, w, pen, theta):
    A = sm.info(prep, w, Kg, theta, pen)
    sym_err = np.max(np.abs(A - A.T))
    eigvals = np.linalg.eigvalsh(A)
    print(f"[info symmetry]   max |A-A.T| (pre-symmetrize check; A already symmetrized) = {sym_err:.3e}")
    print(f"[info PD]         min eigenvalue = {eigvals.min():.6e}   max = {eigvals.max():.6e}   "
          f"cond = {eigvals.max()/eigvals.min():.3e}")
    return eigvals


def main():
    t0 = time.time()
    X, w = common.make_draw(N=10000, seed=1)
    theta_true, _ = common.truth_theta()
    prep = sm.prepare(X)
    pen = sm.penalty_setup(prep, w)

    print(f"N={X.shape[0]}  p={sm.n_params(Kg)}  W={pen.W}  a_pen={pen.a:.3e}")
    print(f"Scov=\n{pen.Scov}")
    assert sm.feasible(Kg, theta_true), "truth theta must be feasible"

    rng = np.random.default_rng(42)
    theta_rand = common.random_feasible_theta(rng, theta_true, scale=0.15)

    print("\n-- grad vs central FD of penalized_ll, at a random feasible theta --")
    check_grad_vs_fd(prep, w, pen, theta_rand)

    print("\n-- w-weighted mean of score_rows vs grad, at the same theta --")
    check_score_rows_mean(prep, w, pen, theta_rand)

    print("\n-- w-weighted mean of score_rows vs grad, AT THE TRUTH --")
    check_score_rows_mean(prep, w, pen, theta_true)

    print("\n-- m_step EM monotonicity, 50 iterations, 3 random feasible starts --")
    check_m_step_monotone(prep, w, pen, theta_true, n_iter=50, n_starts=3)

    print("\n-- from_u(to_u(theta)) == theta, at the random theta --")
    check_u_roundtrip(theta_rand)

    print("\n-- jac_u vs FD of from_u, at the random theta --")
    check_jac_u_vs_fd(theta_rand)

    print("\n-- info symmetric / PD, AT THE TRUTH --")
    check_info_pd(prep, w, pen, theta_true)

    print(f"\nTotal wall time: {time.time()-t0:.1f} s")


if __name__ == '__main__':
    main()
