"""One question: does splitting cells bring the statistic on the quantized data to the statistic on the data (theta-hat) at a
modest number of prototypes? Shearmix draw 0 and mvt draw 0, N = 10^4, the 1086 k-means++ cells. No evaluations: fits only.
Pass 0: prototypes = the cell means (in the statistic's coordinates) with their counts as weights; the statistic on the
quantized data, warm from theta-hat; per output (theta_quantized - theta-hat) / sd, sd = sqrt(V_j / N) from the exact
influence (the bed's parquet psi for shearmix; T.influence for mvt). Passes 1-5: split the cells whose points, replaced by
their prototype, change the statistic's per-point loss at theta-hat most (sum over the cell's points minus count x the loss at
the prototype, in absolute value), each by two-means on the clustering coordinates, as many cells as make the prototype count
grow by about 20 %; refit; report. Stop when the max distance is below 0.1 sd. Cap 120 s; every line flushed to the log.
Run: PYTHONPATH=src python3.9 scripts/fit_gap_check.py [--log PATH]   (default log: scripts/fit_gap_check.log)"""
import os
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[_v] = '1'
import signal, sys, time, warnings
warnings.filterwarnings('ignore')
import numpy as np

T0 = time.time(); CAP_S = 120
LOG = sys.argv[sys.argv.index('--log') + 1] if '--log' in sys.argv else os.path.join(os.path.dirname(os.path.abspath(__file__)), 'fit_gap_check.log')
CASE = '/Users/jtaylor/Dropbox/Software/JT_Py_Pkgs/vqboot/fable/bed/cache/shearmix_s00000_M1086/case.npz'
PSI = '/Users/jtaylor/Dropbox/Research/QIJ_joint/runs/shearmix4/cloudfil_u_p2_N10000/ij/s00000.psi.parquet'
_fh = open(LOG, 'w')


def log(m=''):
    line = '[%5.1f s] %s' % (time.time() - T0, m); print(line, flush=True); _fh.write(line + '\n'); _fh.flush()


class Cap(BaseException):
    pass


def _alarm(s, f):
    raise Cap()


signal.signal(signal.SIGALRM, _alarm); signal.alarm(CAP_S)


def one(name, T, X, Xc, inv, meas, psi):
    from qij_joint.core.xvq import _two_means_points
    N = len(X); sd = np.sqrt((psi ** 2).mean(0) / N); outs = [T.outputs[i] for i in meas]
    per = getattr(T, 'periodic', {}) or {}; period = np.array([per.get(o, 0.0) for o in outs])
    th = np.asarray(T(X, np.ones(N)), float)
    lp = np.asarray(T.per_point_loss(X, th), float)
    members = [np.flatnonzero(inv == c) for c in range(int(inv.max()) + 1)]

    def proto(pts):
        return X[pts].mean(0)

    def mismatch(pts):
        if len(pts) < 2: return 0.0
        return abs(float(lp[pts].sum() - len(pts) * T.per_point_loss(proto(pts)[None, :], th)[0]))

    mis = [mismatch(m) for m in members]
    log('%s: outputs %s; theta-hat fitted' % (name, ', '.join(outs)))
    log('%s  %4s %10s %8s %16s %-20s %s' % (name, 'pass', 'prototypes', 'N/prot', 'max |dist| (sd)', 'worst output', 'distance per output (sd)'))
    for p in range(0, 6):
        rows = np.stack([proto(m) for m in members]); w = np.array([len(m) for m in members], float)
        tq = np.asarray(T(rows, w, start=th) if getattr(T, 'takes_start', False) else T(rows, w), float)
        d = (tq - th)[meas]
        pp = period > 0
        if pp.any(): d[pp] = (d[pp] + 0.5 * period[pp]) % period[pp] - 0.5 * period[pp]
        dist = d / sd; jw = int(np.argmax(np.abs(dist)))
        log('%s  %4d %10d %8.2f %16.3f %-20s %s' % (name, p, len(members), N / len(members), float(np.abs(dist).max()), outs[jw],
                                                   '[' + ' '.join('%.3f' % x for x in dist) + ']'))
        if np.abs(dist).max() < 0.1:
            return len(members), p
        if p == 5: break
        target = int(np.ceil(0.2 * len(members))); order = np.argsort(-np.asarray(mis)); done = 0
        for u in order:
            if done >= target: break
            pts = members[u]
            side = _two_means_points(Xc[pts]) if len(pts) >= 2 else None
            if side is None: continue
            members[u] = pts[side]; members.append(pts[~side]); mis[u] = mismatch(members[u]); mis.append(mismatch(members[-1])); done += 1
    return None, None


def main():
    import pandas as pd
    from qij_joint import registry
    from qij_joint.core.pilot import build_cells
    out = {}
    c = registry.case('cloudfil_u', 'p2'); T = c.make_T(); X = np.asarray(c.draw(10000, 0), float)
    inv = np.load(CASE)['inv'].astype(int); pil = build_cells(X, cells=inv)
    psi = pd.read_parquet(PSI)[['psi_' + T.outputs[i] for i in T.measured]].to_numpy(float)
    out['shearmix'] = one('shearmix', T, X, pil.Xc, pil.inv, list(T.measured), psi)
    c = registry.case('mvt', 'all'); T = c.make_T(); X = np.asarray(c.draw(10000, 0), float)
    pil = build_cells(X, M=1086, vq_transform=c.vq_transform, seed=0)
    psi = np.asarray(T.influence(X, np.ones(len(X))), float)
    out['mvt'] = one('mvt', T, X, pil.Xc, pil.inv, list(range(len(T.outputs))), psi)
    for k, (m, p) in out.items():
        log('%s: %s' % (k, ('within 0.1 sd at %d prototypes (pass %d)' % (m, p)) if m else 'not within 0.1 sd within 5 passes'))


try:
    main()
except Cap:
    log('CAP: %d s reached; stopped here' % CAP_S)
finally:
    signal.alarm(0); log('wall %.1f s' % (time.time() - T0)); _fh.close()
