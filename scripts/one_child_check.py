"""Acceptance A of the one-child rule (2026-11-07): the bed-equivalence check with the exact evaluator, set up as
scripts/port_check.py item 1 (the bed's cells for shearmix draw 0 M1086, k = 16, eps 0.01, level 0.84, scale = ones), against
the bed's ONE-child record: K 179, E 1370 (+ 7 report evaluations), the same leaves split per round in order, the same n_taken
and E per round, Vhat and U to 1e-9 relative (U <= 1e-6 counts as a pass: the record's U is known only to ~5e-7). With the exact
evaluator the bound and the offset are 0 and no repair happens. A plain script, no pytest.
Run: OMP_NUM_THREADS=1 PYTHONPATH=src python3.9 scripts/one_child_check.py"""
import os
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_v] = '1'
import json, time, types, warnings
warnings.filterwarnings('ignore')
import numpy as np

from qij_joint import method

BED = '/Users/jtaylor/Dropbox/Software/JT_Py_Pkgs/vqboot/fable/bed'
CASE = BED + '/cache/shearmix_s00000_M1086'
REC = BED + '/runs/shearmix_s00000_M1086/matern12__childmean+method+means+L84__K400+halt/record.json'

z = np.load(CASE + '/case.npz'); Z, X, inv = z['Z'], z['X'], z['inv'].astype(int); J = Z.shape[1]
T = types.SimpleNamespace(outputs=tuple(str(o) for o in z['outs']), name='bed')


def ev(u):
    i = np.flatnonzero(u); return u[i] @ Z[i]


t0 = time.perf_counter()
r = method.run(T, X, evaluator=ev, cells=inv, k=16, eps=0.01, level=0.84, scale=np.ones(J))
wall = time.perf_counter() - t0
d = json.load(open(REC)); mr = d['extra']['method']['rounds']; lk = d['extra']['looks'][-1]; ms = d['stops']['method_stop']

first_diff = None
print('%5s %6s | %6s %6s | %6s %6s | %s' % ('round', 'phase', 'bed n', 'pkg n', 'bed E', 'pkg E', 'same leaves, order, n_taken, phase, E'))
for b, p in zip(mr, r.rounds_table):
    same = b['leaves'] == p.get('split', []) and b['n_taken'] == p['n_taken'] and b['phase'] == p['phase'] and b['E_end'] == p['E']
    if not same and first_diff is None: first_diff = b['round']
    print('%5d %6s | %6d %6d | %6d %6d | %s' % (b['round'], b['phase'], b['n_taken'], p['n_taken'], b['E_end'], p['E'], same))
if len(mr) != len(r.rounds_table) and first_diff is None: first_diff = min(len(mr), len(r.rounds_table))
rel = lambda a, b: float(np.max(np.abs(np.asarray(a) - np.asarray(b)) / np.abs(np.asarray(b))))
rV, rU, rd = rel(r.Vhat, lk['Vhat']), rel(r.U, lk['U_R']), rel(r.delta, lk['l8']['delta'])
print('rounds bed %d pkg %d | K bed %d pkg %d | E bed %d pkg %d (= bed + %d report)' % (len(mr), r.rounds, ms['K'], r.K, ms['E'], r.E, r.E - ms['E']))
print('max relative difference: Vhat %.2e  U %.2e  delta %.2e' % (rV, rU, rd))
print('offset %s  n_repaired %d  n_inferred %d  bound_eval %s' % (r.offset, r.n_repaired, r.n_inferred, r.bound_eval))
print('first round that differs: %s' % first_diff)
dec = first_diff is None and ms['K'] == r.K and r.E == ms['E'] + J
ok = dec and rV <= 1e-9 and rd <= 1e-9 and rU <= 1e-6 and r.n_repaired == 0 and np.all(r.offset == 0)
print('ACCEPTANCE A: %s (wall %.1f s)' % ('PASS' if ok else 'FAIL', wall))
