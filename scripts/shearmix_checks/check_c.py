"""Agent C's checks for `ShearMix2D` (spec/QIJ_shearmix_interface.md
section 4), ONE draw, N = 10,000, seed 1. Direct calls only; no
oracle/bootstrap/ij/qij pipeline. Run from the worktree root:

    PYTHONPATH=src python3.9 scripts/shearmix_checks/check_c.py
"""
import json
import time

import numpy as np

from qij_joint import datasets, shearmix, shearmix_model as sm

N = 10000
SEED = 1
Kg = 4

print("Generating draw: cloudfil_G_U_P3_v1, N=%d, seed=%d" % (N, SEED))
X = datasets.cloudfil_G_U_P3_v1(N, SEED)
w = np.ones(N)

truth = json.load(open('src/qij_joint/data/truth/cloudfil_u.json'))['p2']
theta_true = np.array(truth['values'][:30])
outputs = sm.make_outputs(Kg)
assert list(outputs) == truth['outputs'][:30]

blocks = {
    'pi': slice(0, 4),
    'mu': slice(4, 12),
    'S': slice(12, 24),
    'fil': slice(24, 30),
}

est = shearmix.ShearMix2D(Kg=Kg, seed=SEED)

# ----------------------------------------------------------------------
# 1. Continuation from the true theta.
# ----------------------------------------------------------------------
print("\n=== (1) Continuation from truth ===")
t0 = time.time()
theta_cont = est(X, w, start=theta_true)
t1 = time.time()
info_cont = est.last_fit_info
print("status:", info_cont.status)
print("n_iter_em:", info_cont.n_iter_em, "n_iter_newton:", info_cont.n_iter_newton)
print("score (scaled grad):", info_cont.score, "g0:", info_cont.g0)
print("wall time: %.3f s" % (t1 - t0))
print("theta_cont finite:", np.all(np.isfinite(theta_cont)))
if np.all(np.isfinite(theta_cont)):
    diff = np.abs(theta_cont - theta_true)
    for name, sl in blocks.items():
        print("  max |theta_hat - truth| [%s]: %.6g" % (name, diff[sl].max()))
    ll_cont = est.loglik(X, w, theta_cont)
    print("penalized ll at continuation:", ll_cont)
else:
    ll_cont = float('nan')

# ----------------------------------------------------------------------
# 2. Cold fit.
# ----------------------------------------------------------------------
print("\n=== (2) Cold fit ===")
t0 = time.time()
theta_cold = est(X, w)
t1 = time.time()
info_cold = est.last_fit_info
wall_cold = t1 - t0
print("status:", info_cold.status)
print("wall time: %.3f s" % wall_cold)
print("search:", info_cold.search)
print("n_iter_em:", info_cold.n_iter_em, "n_iter_newton:", info_cold.n_iter_newton)
print("score:", info_cold.score, "g0:", info_cold.g0)
if np.all(np.isfinite(theta_cold)):
    ll_cold = est.loglik(X, w, theta_cold)
    print("penalized ll at cold fit:", ll_cold)
    print("search gap (cold ll - continuation ll):", ll_cold - ll_cont)
    pis_g, pi_f, mus, covs, fil = sm.unpack(Kg, theta_cold)
    print("recovered pis_g:", pis_g, "pi_f:", pi_f)
    print("recovered mus:\n", mus)
    print("recovered fil:", fil)
    print("truth   mus:\n", sm.unpack(Kg, theta_true)[2])
    print("truth   pis_g:", sm.unpack(Kg, theta_true)[0])
else:
    print("cold fit did not converge; theta is NaN")

# ----------------------------------------------------------------------
# 3. Influence (at the continuation from truth, the well-specified case).
# ----------------------------------------------------------------------
print("\n=== (3) Influence (continuation from truth) ===")
t0 = time.time()
theta_if, IF = est.fit_and_influence(X, w, start=theta_true)
t1 = time.time()
info_if = est.last_fit_info
print("status:", info_if.status, "wall time: %.3f s" % (t1 - t0))
finite = np.all(np.isfinite(IF))
print("IF finite:", finite)
if finite:
    pen = sm.penalty_setup(est.prepare(X), w)
    A = sm.info(est.prepare(X), w, Kg, theta_if, pen)
    cond_A = np.linalg.cond(A)
    print("cond(A):", cond_A)
    se = np.sqrt(np.mean(IF ** 2, axis=0) / N)
    fil_names = outputs[24:30]
    print("IJ standard errors, filament block:")
    for name, val in zip(fil_names, se[24:30]):
        print("  %-10s %.6g" % (name, val))
else:
    print("IF is NaN; cond_max refused the influence or the fit failed")

print("\nDone.")
