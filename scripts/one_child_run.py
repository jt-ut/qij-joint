"""The one real draw of the one-child rule (2026-11-07): shearmix draw 0, N = 10^4, registry ('cloudfil_u', 'p2'), every
evaluation on the full data, M = 300, eps 0.01, level 0.84, k = 16, seed 0, the default scale, vq_transform = c.vq_transform, one
process (pool None). The worker runs in a subprocess killed at a wall cap of 300 s. One log line per round as it runs (round,
phase, K, E, n_taken, n_repaired, offset / delta and U / delta per output), then the report, to scripts/one_child_run.log.
Vhat / V - 1 against the exact V of the bed's parquet psi; beside it the bed's certified result (the one-child record). Plain
script, no pytest.  Run: OMP_NUM_THREADS=1 PYTHONPATH=src python3.9 scripts/one_child_run.py [--log PATH]"""
import json, os, subprocess, sys, time

CAP_S = 300.0
BED = '/Users/jtaylor/Dropbox/Software/JT_Py_Pkgs/vqboot/fable/bed'
REC = BED + '/runs/shearmix_s00000_M1086/matern12__childmean+method+means+L84__K400+halt/record.json'
PSI = '/Users/jtaylor/Dropbox/Research/QIJ_joint/runs/shearmix4/cloudfil_u_p2_N10000/ij/s00000.psi.parquet'


def worker(log_path):
    for v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'): os.environ[v] = '1'
    import warnings; warnings.filterwarnings('ignore')
    import numpy as np
    import pandas as pd
    from qij_joint import registry, method
    fh = open(log_path, 'a'); T0 = time.time()

    def log(m=''):
        line = '[%6.1f s] %s' % (time.time() - T0, m); print(line, flush=True); fh.write(line + '\n'); fh.flush()
    f = lambda v, fm='%.3g': '[' + ' '.join(fm % x for x in np.asarray(v, float)) + ']'

    c = registry.case('cloudfil_u', 'p2'); T = c.make_T(); N = 10000
    X = np.asarray(c.draw(N, 0), float)
    psi = pd.read_parquet(PSI)[['psi_' + T.outputs[i] for i in T.measured]].to_numpy(float); V = (psi ** 2).mean(0)
    log('shearmix draw 0, N %d, outputs %s; M 300, eps 0.01, level 0.84, k 16, seed 0, pool None' % (N, [T.outputs[i] for i in T.measured]))

    def progress(row, s2):
        d = row['delta']
        tail = ('offset/delta %s  U/delta %s' % (f(row['offset'] / d), f(row['U'] / d))) if d is not None else \
               ('offset %s (delta not yet defined: pre phase)' % f(row['offset']))
        log('round %3d %-4s K %4d E %5d n_taken %2d n_repaired %2d  %s' % (row['round'], row['phase'], row['K'], row['E'], row['n_taken'],
                                                                           row['n_repaired'], tail))
    rec = {}; t0 = time.perf_counter()
    r = method.run(T, X, eps=0.01, level=0.84, M=300, k=16, pool=None, seed=0, vq_transform=c.vq_transform, record=rec, progress=progress)
    wall = time.perf_counter() - t0
    cn = rec['counts']
    log('')
    log('K %d  E %d = base %d + pilot %d + splits %d + tests %d + repairs %d + q %d   rounds %d   wall %.1f s   mean refit %.4f s'
        % (r.K, r.E, cn['base'], cn['pilot'], cn['splits'], cn['tests'], cn['repairs'], cn['q'], r.rounds, wall, r.timings['full_refit_mean_s']))
    log('eta_full %.3g   delta %s' % (r.eta_full, f(r.delta)))
    log('bound per output (statistic units, sqrt(eta_full) x s) %s' % f(r.bound_eval))
    log('offset at the stop %s   offset/delta %s   U/delta %s' % (f(r.offset), f(r.offset / r.delta), f(r.U / r.delta)))
    inf = [t['n'] for t in r.leaves_table if t['inferred']]
    log('n_inferred %d; inferred leaf sizes min %s median %s max %s' % (r.n_inferred, min(inf) if inf else '-', np.median(inf) if inf else '-',
                                                                     max(inf) if inf else '-'))
    log('n_repaired %d; repaired leaf sizes %s' % (r.n_repaired, [d['n'] for d in r.repairs]))
    if r.repairs:
        ratios = np.array([d['ratio'] for d in r.repairs])
        for d in r.repairs:
            log('  repair round %d leaf %d n %d entered %s  discrepancy (units of s) %s  ratio |disc|/bound %s'
                % (d['round'], d['leaf'], d['n'], d['entered'], f(d['discrepancy'], '%.2e'), f(d['ratio'], '%.3f')))
        log('  ratio max over repairs per output %s; median %s; max overall %.3f; median overall %.3f'
            % (f(ratios.max(0)), f(np.median(ratios, 0)), ratios.max(), np.median(ratios)))
    rel = r.Vhat / V - 1.0
    log('Vhat/V - 1 per output %s   max |.| %.4f' % (f(rel, '%+.4f'), np.abs(rel).max()))
    ee = r.eval_error
    log('redundant rows %d %s; eval_error max (units of s) %s' % (ee['n'], ee['kinds'], f(ee['max'], '%.2e')))
    d = json.load(open(REC)); ms = d['stops']['method_stop']
    log('bed certified result (one-child record, exact influences, M 1086): K %d  E %d  max|Vhat - 1| %.4f  per output %s'
        % (ms['K'], ms['E'], max(abs(x - 1.0) for x in ms['Vhat']), f(np.array(ms['Vhat']) - 1.0, '%+.4f')))
    print('WORKER DONE', flush=True); fh.close()


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    log_path = sys.argv[sys.argv.index('--log') + 1] if '--log' in sys.argv else os.path.join(here, 'one_child_run.log')
    open(log_path, 'w').close()
    t0 = time.time()
    pr = subprocess.Popen([sys.executable, os.path.abspath(__file__), '--worker', log_path], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    try:
        out, _ = pr.communicate(timeout=CAP_S)
    except subprocess.TimeoutExpired:
        pr.kill(); out, _ = pr.communicate()
        with open(log_path, 'a') as fh: fh.write('KILLED at the cap of %.0f s\n' % CAP_S)
    print('\n'.join(l for l in out.splitlines() if 'cpu_info' not in l))
    print('exit %s, wall %.1f s, log %s' % (pr.returncode, time.time() - t0, log_path))


if __name__ == '__main__':
    if '--worker' in sys.argv: worker(sys.argv[sys.argv.index('--worker') + 1])
    else: main()
