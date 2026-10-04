"""Agent D self-check (2): replay_state WITHOUT a refiner reproduces a
normal run's own psi_hat/G/W_parent exactly (draw 45 of cloudfil_u p2,
eps 0.01, study settings). feedback=False throughout -- replay_records
are still stored (unconditional per the interface)."""
import os
for v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
          'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(v, '1')

import numpy as np
import qij_joint.qij as qij_mod
from qij_joint.core.joint import replay_state
from qij_joint import datasets, cloudfil_u

captured = {}
_real_run_joint = qij_mod.run_joint


def _spy_run_joint(X, counter, theta_hat, psi0_all, model, xvq, eta, eps, offset,
                    pool=None, start=None, measured=None, **kw):
    jr = _real_run_joint(X, counter, theta_hat, psi0_all, model, xvq, eta, eps, offset,
                         pool=pool, start=start, measured=measured, **kw)
    captured['theta_hat'] = np.asarray(theta_hat, dtype=float)
    captured['psi0_all'] = np.asarray(psi0_all, dtype=float)
    captured['offset'] = np.asarray(offset, dtype=float)
    captured['eta'] = eta
    captured['measured'] = np.asarray(measured if measured is not None else range(psi0_all.shape[1]), dtype=int)
    captured['jr'] = jr
    return jr


qij_mod.run_joint = _spy_run_joint

X = datasets.cloudfil_G_U_P3_v1(10000, 45)
T = cloudfil_u.P2ShearMixture()
method = qij_mod.QIJ(eps=0.01, seed=0, gptrend='quadratic', gpwidth='local',
                      survey='moments', quantized_start='full-data',
                      fit_weights='mass', feedback=False)
result = method.fit(X, T, pool=None)

jr = captured['jr']
records = jr._replay_records
print("n_splits (records):", len(records))
print("n_rounds:", jr.n_rounds, "L:", jr.L, "n_evals:", jr.n_evals)

q = captured['psi0_all'].shape[1]
N = captured['psi0_all'].shape[0]
theta_abs = np.maximum(np.abs(captured['theta_hat'][captured['measured']]), np.finfo(float).eps)

psi_hat_replay, split_replay, leaf_replay = replay_state(
    captured['psi0_all'], captured['offset'], records, N, q, captured['eta'], theta_abs)

diff_psi = np.max(np.abs(psi_hat_replay - jr.state_psi_hat))
print("max |psi_hat_replay - jr.state_psi_hat| =", diff_psi)

diff_G = np.max(np.abs(split_replay['G'] - jr.split_G)) if jr.split_G.size else 0.0
diff_Wp = np.max(np.abs(split_replay['W_parent'] - jr.split_W_parent)) if jr.split_W_parent.size else 0.0
print("max |G_replay - split_G| =", diff_G)
print("max |W_parent_replay - split_W_parent| =", diff_Wp)

ok = diff_psi <= 1e-12 and diff_G <= 1e-12 and diff_Wp <= 1e-12
print("PASS" if ok else "FAIL (check tolerances above)")
