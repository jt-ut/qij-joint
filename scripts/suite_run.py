"""The method (qij_joint.method.run: full-data evaluations, one child per split, the offset from eta, repair) on shearmix draws
0-19 and mvt draws 0-4 at N = 10^4, M = 300 (the pilot budget rule at eps = 0.01), one process per draw, 16 at a time, a cap on
the whole suite, one log line per draw as it finishes. Per draw: Vhat / V - 1 per output against the exact V (the bed's parquet
psi for shearmix, T.influence for mvt), K, E and its decomposition, rounds, inferred and repaired leaves, offset / delta, U / delta,
wall; beside it the certified method's max |Vhat - 1| from the bed record (M = 1086, exact influences). Plain script, no pytest.
Run: PYTHONPATH=src python3.9 scripts/suite_run.py --log PATH [--M 300] [--cap 360]      worker: ... --worker <dataset> <draw> <M>"""
import json, os, subprocess, sys, time

EPS = 0.01; LEVEL = 0.84; PARALLEL = 16
BED = '/Users/jtaylor/Dropbox/Software/JT_Py_Pkgs/vqboot/fable/bed'
PSI = '/Users/jtaylor/Dropbox/Research/QIJ_joint/runs/shearmix4/cloudfil_u_p2_N10000/ij/s%05d.psi.parquet'
DRAWS = [('mvt', s) for s in range(5)] + [('shearmix', s) for s in range(20)]


def worker(ds, s, M):
    for v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'): os.environ[v] = '1'
    import warnings; warnings.filterwarnings('ignore')
    import numpy as np
    from qij_joint import registry, method
    N = 10000
    if ds == 'shearmix':
        import pandas as pd
        c = registry.case('cloudfil_u', 'p2'); T = c.make_T(); X = np.asarray(c.draw(N, s), float)
        psi = pd.read_parquet(PSI % s)[['psi_' + T.outputs[i] for i in T.measured]].to_numpy(float)
    else:
        c = registry.case('mvt', 'all'); T = c.make_T(); X = np.asarray(c.draw(N, s), float)
        psi = np.asarray(T.influence(X, np.ones(N)), float)
    V = (psi ** 2).mean(0)
    rec = {}
    t0 = time.time()
    r = method.run(T, X, eps=EPS, level=LEVEL, M=M, k=16, seed=0, pool=None, vq_transform=c.vq_transform, record=rec)
    wall = time.time() - t0
    rel = r.Vhat / V - 1.0
    bed = os.path.join(BED, 'runs', '%s_s%05d_M1086' % (ds, s), 'matern12__childmean+method+means+L84__K400+halt', 'record.json')
    bv = None
    if os.path.exists(bed):
        ms = json.load(open(bed))['stops']['method_stop']; bv = None if ms is None else [x - 1.0 for x in ms['Vhat']]
    out = dict(ds=ds, s=s, outputs=list(r.outputs), rel=list(map(float, rel)), within=bool(np.abs(rel).max() <= EPS),
               K=int(r.K), E=int(r.E), counts=rec.get('counts'), rounds=int(r.rounds), n_inferred=int(r.n_inferred),
               n_repaired=int(r.n_repaired), offset_over_delta=list(map(float, np.asarray(r.offset) / np.asarray(r.delta))),
               U_over_delta=list(map(float, np.asarray(r.U) / np.asarray(r.delta))), R_true_over_delta=list(map(float, (V - r.Vhat) / np.asarray(r.delta))),
               eta_full=float(r.eta_full), wall=wall, total_s=float(r.timings.get('total_s', wall)), bed_rel=bv)
    print('RESULT ' + json.dumps(out), flush=True)
    try:                                                      # where the time goes, per draw (the method's own timings)
        tp = '/private/tmp/claude-501/-Users-jtaylor-Dropbox-Software-JT-Py-Pkgs-vqboot-fable/7d34c8ea-d6d7-4327-9b2a-e62e7e8ab91d/scratchpad/suite_timings.jsonl'
        with open(tp, 'a') as f: f.write(json.dumps(dict(ds=ds, s=s, M=M, K=int(r.K), E=int(r.E), rounds=int(r.rounds), wall=wall, timings={k_: float(v) for k_, v in r.timings.items()})) + '\n')
    except Exception: pass


def main():
    import numpy as np
    a = sys.argv
    M = int(a[a.index('--M') + 1]) if '--M' in a else 300
    cap = float(a[a.index('--cap') + 1]) if '--cap' in a else 360.0
    log_path = a[a.index('--log') + 1] if '--log' in a else os.path.join(os.path.dirname(os.path.abspath(__file__)), 'suite_run.log')
    fh = open(log_path, 'w'); T0 = time.time()

    def log(m=''):
        line = '[%5.1f s] %s' % (time.time() - T0, m); print(line, flush=True); fh.write(line + '\n'); fh.flush()
    only = a[a.index('--only') + 1] if '--only' in a else None
    par = int(a[a.index('--parallel') + 1]) if '--parallel' in a else PARALLEL
    draws = [d for d in DRAWS if only is None or d[0] == only]
    log('launch: %d draws at M = %d, eps %.3f, %d at a time, cap %.0f s' % (len(draws), M, EPS, par, cap))
    queue = list(draws); run = {}; res = []
    while queue or run:
        while queue and len(run) < par:
            ds, s = queue.pop(0)
            run[(ds, s)] = (subprocess.Popen([sys.executable, os.path.abspath(__file__), '--worker', ds, str(s), str(M)], stdout=subprocess.PIPE,
                                             stderr=subprocess.STDOUT, text=True), time.time())
        for key, (pr, t0) in list(run.items()):
            if pr.poll() is None: continue
            out = pr.stdout.read(); del run[key]
            line = [l for l in out.splitlines() if l.startswith('RESULT ')]
            if not line:
                log('%s %d: FAILED (%s)' % (key[0], key[1], ' | '.join(l for l in out.splitlines() if 'cpu_info' not in l)[-400:])); continue
            r = json.loads(line[0][7:]); res.append(r)
            log('%-8s %2d  within eps %-5s  max|Vhat/V-1|/eps %.3f  per output/eps %s  K %3d  E %4d  rounds %2d  inferred %3d  repaired %2d  '
                'max offset/delta %.1e  max U/delta %.2f  wall %.1f s  bed certified max|Vhat-1|/eps %s'
                % (r['ds'], r['s'], r['within'], max(abs(x) for x in r['rel']) / EPS, '[' + ' '.join('%+.2f' % (x / EPS) for x in r['rel']) + ']',
                   r['K'], r['E'], r['rounds'], r['n_inferred'], r['n_repaired'], max(r['offset_over_delta']), max(r['U_over_delta']), r['wall'],
                   ('%.3f' % (max(abs(x) for x in r['bed_rel']) / EPS)) if r['bed_rel'] else '-'))
        if time.time() - T0 > cap:
            for key, (pr, t0) in run.items(): pr.kill(); log('%s %d: killed at the cap' % key)
            run = {}; log('CAP %.0f s reached; %d draws not launched' % (cap, len(queue))); break
        time.sleep(0.5)
    log('')
    for ds in ('shearmix', 'mvt'):
        rs = [r for r in res if r['ds'] == ds]
        if not rs: log('%s: no draw finished' % ds); continue
        R = np.abs(np.array([r['rel'] for r in rs])) / EPS; outs = rs[0]['outputs']
        log('SUMMARY %s: %d draws at M = %d; within eps on every output %d / %d (design level 84 %%)' % (ds, len(rs), M, sum(r['within'] for r in rs), len(rs)))
        log('  %-20s %10s %12s %12s %12s %14s' % ('output', 'within', 'med |.|/eps', 'max |.|/eps', 'bed med/eps', 'R>U (draws)'))
        B = [r['bed_rel'] for r in rs if r['bed_rel']]
        Bm = np.median(np.abs(np.array(B)), 0) / EPS if B else None
        RU = np.array([[rt > u for rt, u in zip(r['R_true_over_delta'], r['U_over_delta'])] for r in rs])
        for j, o in enumerate(outs):
            log('  %-20s %10s %12.3f %12.3f %12s %14s' % (o, '%d/%d' % ((R[:, j] <= 1).sum(), len(rs)), np.median(R[:, j]), R[:, j].max(),
                                                       ('%.3f' % Bm[j]) if Bm is not None else '-', '%d/%d' % (RU[:, j].sum(), len(rs))))
        log('  median E %d (min %d max %d); median K %d; median rounds %d; median inferred leaves %d; draws with a repair %d; '
            'max offset/delta over draws %.1e; median wall %.1f s'
            % (np.median([r['E'] for r in rs]), min(r['E'] for r in rs), max(r['E'] for r in rs), np.median([r['K'] for r in rs]),
               np.median([r['rounds'] for r in rs]), np.median([r['n_inferred'] for r in rs]), sum(r['n_repaired'] > 0 for r in rs),
               max(max(r['offset_over_delta']) for r in rs), np.median([r['wall'] for r in rs])))
        log('  per draw max_j |Vhat/V - 1|/eps and the bed\'s certified max_j |Vhat - 1|/eps: ' +
            ', '.join('%d: %.2f / %s' % (r['s'], max(abs(x) for x in r['rel']) / EPS, ('%.2f' % (max(abs(x) for x in r['bed_rel']) / EPS)) if r['bed_rel'] else '-')
                      for r in sorted(rs, key=lambda r: r['s'])))
    log('wall %.1f s' % (time.time() - T0)); fh.close()


if __name__ == '__main__':
    if '--worker' in sys.argv:
        i = sys.argv.index('--worker'); worker(sys.argv[i + 1], int(sys.argv[i + 2]), int(sys.argv[i + 3]))
    else:
        main()
