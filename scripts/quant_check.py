"""One question: on shearmix draw 0 (N = 10^4, ('cloudfil_u', 'p2')), are evaluations on the refined quantized representation
R_S accurate enough (delta_j <= eps/4 in the pilot's unit sqrt(B_j)) and cheaper per evaluation than full-data evaluations?

  1. the calibration loop as built in qij_joint.method (pilot on R, sqrt(B), the 16 worst-mismatch cells and the median cell
     paired with full-data evaluations, refine, repeat), at most 3 passes, fewer if the next would not fit under the cap;
  2. 20 paired evaluations beyond the pilot cells on the final R: 10 leaf-like supports (a k-means piece of 20 to 200 points
     of the union of a random cell and its 4 nearest cells, straddling at least 2 cells; weights 1/n) and 10 test-like
     supports (centred Gaussian weights / n on such pieces): quantized on R_S against the full data, |difference| / sqrt(B_j),
     and the wall time of each evaluation (R_S: two refits; full data: one refit, the base being theta-hat);
  3. the table and the verdict.
Hard cap: 300 s wall (SIGALRM); every line is written to the log and flushed as it happens. Plain script, no pytest.
Run: PYTHONPATH=src python3.9 scripts/quant_check.py [--log PATH]   (default log: scripts/quant_check.log)"""
import os
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[_v] = '1'
import signal, sys, time, warnings
warnings.filterwarnings('ignore')
import numpy as np

CAP_S = 300.0
T0 = time.time()
LOG = sys.argv[sys.argv.index('--log') + 1] if '--log' in sys.argv else os.path.join(os.path.dirname(os.path.abspath(__file__)), 'quant_check.log')
_fh = open(LOG, 'w')


def log(msg=''):
    line = '[%6.1f s] %s' % (time.time() - T0, msg)
    print(line, flush=True); _fh.write(line + '\n'); _fh.flush()


class Cap(BaseException):                                     # not an Exception: the evaluation's own failure handler must not catch it
    pass


def _alarm(sig, frm):
    raise Cap()


signal.signal(signal.SIGALRM, _alarm); signal.alarm(int(CAP_S))


def f(v, fm='%.4f'):
    return '[' + ' '.join(fm % x for x in np.asarray(v, float)) + ']'


def main():
    from qij_joint import registry
    from qij_joint.method import _Evaluations
    from qij_joint.core.pilot import build_cells, cell_weights
    from qij_joint.core.xvq import Representation
    eps = 0.01; k = 16
    c = registry.case('cloudfil_u', 'p2'); T = c.make_T(); X = np.asarray(c.draw(10000, 0), float); N = len(X)
    meas = list(T.measured); J = len(meas)
    log('shearmix draw 0, N %d, outputs %s; cap %.0f s' % (N, ', '.join(T.outputs[i] for i in meas), CAP_S))
    ev = _Evaluations(T, X, None, None, meas); ev.base()
    log('theta-hat and eta_full (4 full-data refits): eta_full %.1e, step delta %.2e' % (ev.eta_full, ev.delta))
    pil = build_cells(X); C = pil.C; pc = pil.ncell / N; pu = cell_weights(pil)
    R = Representation(X, pil.Xc, pil.inv, T.per_point_loss, ev.theta); ev.set_units(R.labels)
    rank = np.argsort(-np.abs(R.delta), kind='stable'); cal = [int(x) for x in rank[:max(8, k)]]
    med = int(rank[len(rank) // 2]); cal += [med] if med not in cal else []
    t = time.time(); full_cal = ev([pu[x] for x in cal], full=True)
    log('cells %d; calibration cells %d paired with full-data evaluations (%.1f s)' % (C, len(cal), time.time() - t))
    log('pass  |R|  N/|R|  seconds  delta_j (|quantized - full| / sqrt(B_j), max over the paired cells; eps/4 = %.4f)' % (eps / 4))
    passes = []; s = None; last_s = None
    for p in range(1, 4):
        if passes:
            est = last_s * R.size / passes[-1][1]
            if time.time() - T0 + est + 40.0 > CAP_S:
                log('pass %d skipped: about %.0f s more would pass the cap (%.0f s used)' % (p, est, time.time() - T0)); break
        t = time.time(); ev.set_units(R.labels); Yp = ev(pu); last_s = time.time() - t
        s = np.sqrt((pc[:, None] * Yp ** 2).sum(0)); qd = np.abs(Yp[cal] - full_cal).max(0) / s
        passes.append((p, R.size, qd, last_s))
        log('%4d %5d %6.2f %8.1f  %s' % (p, R.size, N / R.size, last_s, f(qd)))
        if np.all(qd <= eps / 4) or p == 3: break
        n_sp = R.refine(); log('      refined: %d splits, |R| now %d, total mismatch %.4g' % (n_sp, R.size, R.total_mismatch()))
    # 2. twenty paired evaluations beyond the pilot cells, on the R of the last pass
    ev.set_units(R.labels)
    from sklearn.cluster import KMeans
    from scipy.spatial import cKDTree
    rng = np.random.RandomState(0); tree = cKDTree(pil.proto); groups = []
    while len(groups) < 10:
        c0 = int(rng.randint(C)); _, nb = tree.query(pil.proto[c0], k=5)
        pts = np.flatnonzero(np.isin(pil.inv, nb))
        lab = KMeans(int(rng.randint(2, 5)), n_init=1, random_state=int(rng.randint(1 << 30))).fit(pil.Xc[pts]).labels_
        for g in np.unique(lab):
            q = np.sort(pts[lab == g])
            if 20 <= len(q) <= 200 and len(np.unique(pil.inv[q])) >= 2:
                groups.append(q); break
    sup = [(q, np.full(len(q), 1.0 / len(q))) for q in groups]
    for q in groups:
        v = rng.standard_normal(len(q)); v -= v.mean(); sup.append((q, v / len(q)))
    log('20 supports: sizes %s, cells met %s' % ([len(q) for q in groups], [len(np.unique(pil.inv[q])) for q in groups]))
    tq = []; tf = []; dq = []
    for i, u in enumerate(sup):
        t = time.time(); yq = ev([u])[0]; tq.append(time.time() - t)
        t = time.time(); yf = ev([u], full=True)[0]; tf.append(time.time() - t)
        dq.append(np.abs(yq - yf) / s)
        log('pair %2d (%s, n %3d): |q - full| / sqrt(B) %s   R_S %.3f s, full %.3f s' % (i + 1, 'leaf' if i < 10 else 'test', len(u[0]), f(dq[-1]), tq[-1], tf[-1]))
    D = np.array(dq); L, Tt = D[:10], D[10:]
    log('')
    log('TABLE')
    log('passes %d (cap 3); final |R| %d, compression N/|R| %.2f' % (len(passes), passes[-1][1], N / passes[-1][1]))
    log('delta_j at the last pass (pilot cells)    %s' % f(passes[-1][2]))
    log('leaf-like pairs, max    %s' % f(L.max(0))); log('leaf-like pairs, median %s' % f(np.median(L, 0)))
    log('test-like pairs, max    %s' % f(Tt.max(0))); log('test-like pairs, median %s' % f(np.median(Tt, 0)))
    log('time per evaluation: R_S %.3f s (two refits, %.0f rows on average) vs full data %.3f s (one refit)'
        % (np.mean(tq), ev.stats['rs_rows'] / max(ev.stats['rs_refits'], 1), np.mean(tf)))
    ok = bool(np.all(passes[-1][2] <= eps / 4) and np.all(D.max(0) <= eps / 4))
    log('VERDICT: %s at eps/4 = %.4f after %d passes; compression %.2f; an R_S evaluation costs %.2fx a full-data evaluation'
        % ('admissible' if ok else 'NOT admissible', eps / 4, len(passes), N / passes[-1][1], np.mean(tq) / np.mean(tf)))


try:
    main()
except Cap:
    log('CAP: %.0f s reached; stopped here (the lines above are what was measured)' % CAP_S)
finally:
    signal.alarm(0); log('wall %.1f s' % (time.time() - T0)); _fh.close()
