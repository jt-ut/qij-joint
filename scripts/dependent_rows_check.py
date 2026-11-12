"""The redundancy rule under inexact evaluations. The bed's case (shearmix draw 0, exact influences Z with V = 1 per output,
the bed's cells, k = 16, scale 1): the exact run and the same run with independent noise of 1e-6 and 2.5e-3 (sqrt V units)
added to every evaluation. For each: K, E, Vhat, U, the rows the rule kept out of psi-hat and the eval_error summary, and the
true error R = mean (Z - psi-tilde)^2 of the reported model (V = 1, so R is in units of V). Plain script, no pytest. ~30 s.
Run: PYTHONPATH=src python3.9 scripts/dependent_rows_check.py"""
import os
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[_v] = '1'
import types
import numpy as np
from qij_joint import method

CASE = '/Users/jtaylor/Dropbox/Software/JT_Py_Pkgs/vqboot/fable/bed/cache/shearmix_s00000_M1086/case.npz'
z = np.load(CASE); Z = z['Z']; X = z['X']; inv = z['inv']
T = types.SimpleNamespace(outputs=tuple(str(o) for o in z['outs']), name='bed')


def f(v, fm='%.2e'):
    return '[' + ' '.join(fm % x for x in np.asarray(v, float)) + ']'


def exact(u):
    i = np.flatnonzero(u); return u[i] @ Z[i]


for lvl in (0.0, 1e-6, 2.5e-3):
    rng = np.random.RandomState(1)

    def ev(u, lvl=lvl):
        return exact(u) + (lvl * rng.standard_normal(7) if lvl else 0.0)
    rec = {}
    try:
        r = method.run(T, X, evaluator=ev, cells=inv, k=16, scale=np.ones(7), record=rec)
    except AssertionError as exc:
        print('noise %.1e: FAILED: %s' % (lvl, exc)); continue
    R = ((Z - rec['psi_tilde']) ** 2).mean(0); ee = r.eval_error
    print('noise %.1e per evaluation: K %d, E %d, rounds %d' % (lvl, r.K, r.E, r.rounds))
    print('  Vhat             ' + f(r.Vhat, '%.6f'))
    print('  U                ' + f(r.U))
    print('  delta            ' + f(r.delta))
    print('  true R (units V) ' + f(R) + '   R <= U on every output: %s;  Vhat <= 1 <= Vhat + U: %s'
          % (bool(np.all(R <= r.U)), bool(np.all((r.Vhat <= 1) & (1 <= r.Vhat + r.U)))))
    print('  redundant rows %d %s; relative squared residual: redundant max %.1e, entered min %.1e'
          % (ee['n'], ee['kinds'], ee['rel_residual_redundant_max'] or 0.0, ee['rel_residual_entered_min'] or 0.0))
    print('  eval_error max   ' + f(ee['max']) + '   rms ' + f(ee['rms']))
