"""One question: after a per-output scale correction measured on 16 cells, how good are the quantized cell means (the points
survey of scripts/pilot_diag.py: refits on the 1086 cell means of shearmix draw 0) as steering information?
For each output j: a_j by least squares q = a e on 16 random cells (RandomState(0)); the affine q = a e + b on the same 16;
a_j on all cells. Over all cells: sum_c p_c (corrected - e)^2 / V_j for the raw, scale-only and affine corrections; the same
on the 20 worst cells (by the corrected error) and their share; corr(q, e) (unchanged by an affine correction, so reported
once). Cap 90 s; every line written to the log and flushed. Plain script, no pytest.
Run: PYTHONPATH=src python3.9 scripts/scale_check.py [--cache NPZ] [--log PATH]
  --cache: an npz with I (M, 37), the quantized cell means (pilot_diag's points survey); recomputed (about 25 s) when absent."""
import os
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[_v] = '1'
import signal, sys, time, warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd

T0 = time.time(); CAP_S = 90
arg = lambda k, d: sys.argv[sys.argv.index(k) + 1] if k in sys.argv else d
LOG = arg('--log', os.path.join(os.path.dirname(os.path.abspath(__file__)), 'scale_check.log')); CACHE = arg('--cache', None)
CASE = '/Users/jtaylor/Dropbox/Software/JT_Py_Pkgs/vqboot/fable/bed/cache/shearmix_s00000_M1086/case.npz'
PSI = '/Users/jtaylor/Dropbox/Research/QIJ_joint/runs/shearmix4/cloudfil_u_p2_N10000/ij/s00000.psi.parquet'
OUTS = ['p2_x', 'p2_y', 'p2_log_reff', 'p2_log_axis_ratio', 'p2_pa', 'p2_logit_w', 'p2_log_contrast']
_fh = open(LOG, 'w')


def log(m=''):
    line = '[%5.1f s] %s' % (time.time() - T0, m); print(line, flush=True); _fh.write(line + '\n'); _fh.flush()


class Cap(BaseException):
    pass


def _alarm(s, f):
    raise Cap()


signal.signal(signal.SIGALRM, _alarm); signal.alarm(CAP_S)


def main():
    inv = np.load(CASE)['inv'].astype(int); C = int(inv.max()) + 1; N = len(inv)
    n = np.bincount(inv, minlength=C).astype(float); p = n / N
    P = pd.read_parquet(PSI)[['psi_' + o for o in OUTS]].to_numpy(float); V = (P ** 2).mean(0)
    e = np.stack([np.bincount(inv, P[:, j], C) for j in range(7)], 1) / n[:, None]
    if CACHE and os.path.exists(CACHE):
        q = np.load(CACHE)['I'][:, 30:37]; log('quantized cell means from %s' % CACHE)
    else:
        from qij_joint import registry
        from qij_joint.core.counter import Counter
        from qij_joint.core.xvq import prototype_influences
        c = registry.case('cloudfil_u', 'p2'); T = c.make_T(); X = np.asarray(c.draw(N, 0), float); cnt = Counter(T, N)
        th = cnt(X, np.ones(N)); rows = np.stack([np.bincount(inv, X[:, a], C) for a in range(2)], 1) / n[:, None]
        _, I, _, _ = prototype_influences(rows, cnt, p, T.eta, quantized_start='full-data', theta_hat=th)
        q = I[:, list(T.measured)]; log('quantized cell means recomputed (points survey)')
    cal = np.random.RandomState(0).choice(C, 16, replace=False)
    log('calibration cells (RandomState(0).choice(%d, 16)): %s' % (C, list(cal)))
    hdr = '%-18s %8s %8s %8s %8s | %8s %8s %8s | %8s %8s %8s | %8s %8s'
    log(hdr % ('output', 'a(16)', 'b(16)/sV', 'a_aff', 'a(all)', 'raw', 'scale', 'affine', 'w20 scl', 'shr scl', 'w20 aff', 'corr', 'corr_aff'))
    for j, o in enumerate(OUTS):
        ec, qc = e[cal, j], q[cal, j]
        a = float(ec @ qc / (ec @ ec))                                   # q = a e
        A = np.c_[ec, np.ones(16)]; aa, bb = np.linalg.lstsq(A, qc, rcond=None)[0]   # q = a e + b
        a_all = float((p * e[:, j]) @ q[:, j] / ((p * e[:, j]) @ e[:, j]))     # p-weighted, all cells
        raw = (p * (q[:, j] - e[:, j]) ** 2).sum() / V[j]
        rs = p * (q[:, j] / a - e[:, j]) ** 2 / V[j]; ra = p * ((q[:, j] - bb) / aa - e[:, j]) ** 2 / V[j]
        ws, wa = np.sort(rs)[::-1][:20].sum(), np.sort(ra)[::-1][:20].sum()
        cr = np.corrcoef(q[:, j], e[:, j])[0, 1]
        log(hdr.replace('%8s', '%8.3f').replace('%-18s', '%-18s') % (o, a, bb / np.sqrt(V[j]), aa, a_all, raw, rs.sum(), ra.sum(), ws, ws / rs.sum(), wa, cr, cr))
    log('columns: raw = sum_c p_c (q - e)^2 / V; scale = with q / a(16); affine = with (q - b)/a_aff; w20 = the 20 worst cells'
        ' of that residual, shr = their share; corr = corr(q, e), the same after any affine correction')


try:
    main()
except Cap:
    log('CAP: %d s reached' % CAP_S)
finally:
    signal.alarm(0); log('wall %.1f s' % (time.time() - T0)); _fh.close()
