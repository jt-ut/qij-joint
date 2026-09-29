#!/usr/bin/env python3.9
"""V4 (spec/QIJ_qijdt_spec.md section 12): the existing all-methods
`cloudfil` smoke (runbook.sh's `cloudfil_test` block: `oracle`
draws 0:20, `ij`/`boot` (B=50)/`qij` draws 0:2, N=10000, workers 14,
seed 0, diag-draws 0:1, `qij` at the demo settings) run unchanged,
plus `qijdt` added at eps = 0.05 (draws 0:2, same N/seed/workers).

One flag in runbook.sh's `QIJ_DEMO` string is stale: `--refine-trigger
measured` -- `run.py` (checked directly) has no `--refine-trigger`
option any more; it is dropped here for the `qij` call (flagged in the
report, not silently fixed elsewhere).

    PYTHONPATH=src python3.9 scripts/qijdt_validate_v4.py \\
        --out /Users/jtaylor/Dropbox/Research/QIJ_joint/qijdt_validation/v4

Each `run.py` call is wrapped in a `perl alarm` hard kill, sized per
method: 20 min (`oracle`, 20 draws), 10 min (`ij`), 30 min (`boot`,
B=50), 90 min (`qij`, the demo settings), 30 min (`qijdt`, eps 0.05,
qijdt's own per-draw cost is small by calibration).
"""
import os

for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
           'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(_v, '1')

import argparse
import json
import subprocess
import sys
import time

from qij_joint import products

_RUN_PY = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'run.py')
_SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src')

DATASET, ESTIMATOR = 'cloudfil', 'p2'
N = 10000
DRAWS = (0, 1)
ORACLE_DRAWS = tuple(range(2))
SEED = 0
WORKERS = 14
B = 50
DIAG = (0, 1)

# runbook.sh's QIJ_DEMO, with the stale --refine-trigger flag dropped (flagged, not fixed
# upstream): "--survey moments --quantized-start full-data --gptrend quadratic --gpwidth
# local --ivqbins marginal --refine-schedule rounds --refine-trigger measured --eps 0.01"
QIJ_DEMO_ORIGINAL = ('--survey', 'moments', '--quantized-start', 'full-data', '--gptrend',
                      'quadratic', '--gpwidth', 'local', '--ivqbins', 'marginal',
                      '--refine-schedule', 'rounds', '--refine-trigger', 'measured',
                      '--eps', '0.01')
STALE_FLAGS = ('--refine-trigger',)


def _drop_stale(args: tuple) -> list:
    """Drops every flag in `STALE_FLAGS` and its value from `args`."""
    out, skip = [], False
    for a in args:
        if skip:
            skip = False
            continue
        if a in STALE_FLAGS:
            skip = True
            continue
        out.append(a)
    return out


QIJ_DEMO = _drop_stale(QIJ_DEMO_ORIGINAL)

METHODS = {
    'oracle': (ORACLE_DRAWS, [], 1200),
    'ij': (DRAWS, [], 600),
    'boot': (DRAWS, ['--B', str(B)], 1800),
    'qij': (DRAWS, ['--diag-draws', f'{DIAG[0]}:{DIAG[-1] + 1}'] + QIJ_DEMO, 5400),
    'qijdt': (DRAWS, ['--eps', '0.05'], 1800),
}


def _run_cmd(cmd: list, timeout_s: int) -> int:
    """Hard-kill wrapper (module docstring): `perl`'s `alarm` is pending
    across `exec`, so it reaches the real process. Returns the exit
    code (nonzero on failure or the alarm firing)."""
    env = dict(os.environ, PYTHONPATH=_SRC)
    guarded = ['perl', '-e', 'alarm(shift @ARGV); exec(@ARGV) or die $!', str(timeout_s)] + cmd
    return subprocess.run(guarded, env=env).returncode


def run_method(method: str, draws: tuple, extra_args: list, timeout_s: int,
               runs_root: str) -> dict:
    """Runs one method over `draws`, timed by this script's own clock;
    `success` requires both a zero exit code and every draw's scalar
    row present afterward."""
    md = products.method_dir(runs_root, DATASET, ESTIMATOR, N, method)
    cmd = [sys.executable, _RUN_PY, DATASET, ESTIMATOR, method,
           '--N', str(N), '--draws', f'{draws[0]}:{draws[-1] + 1}', '--seed', str(SEED),
           '--workers', str(WORKERS), '--out', runs_root] + extra_args
    t0 = time.perf_counter()
    code = _run_cmd(cmd, timeout_s)
    wall = time.perf_counter() - t0
    written = all(products.is_done(md, s) for s in draws)
    ratios = {}
    # V ratio "where applicable" (spec V4): qij's own variance column is `V_tot_hat_<o>`,
    # qijdt's is `V_tot_<o>` (spec 11.1); ij and boot have no single variance estimate of
    # their own to ratio against ij (boot stores its raw replicates only).
    v_col = {'qij': 'V_tot_hat', 'qijdt': 'V_tot'}.get(method)
    ij_md = products.method_dir(runs_root, DATASET, ESTIMATOR, N, 'ij')
    if written and v_col is not None and products.is_done(ij_md, draws[0]):
        ij_row = products.collect(runs_root, DATASET, ESTIMATOR, N, 'ij').set_index('s')
        m_row = products.collect(runs_root, DATASET, ESTIMATOR, N, method).set_index('s')
        lead = f'{v_col}_'
        outputs = [c[len(lead):] for c in m_row.columns if c.startswith(lead)
                   and f'V_ij_{c[len(lead):]}' in ij_row.columns]
        for s in draws:
            if s not in m_row.index or s not in ij_row.index:
                continue
            for o in outputs:
                ratios.setdefault(int(s), {})[o] = (
                    float(m_row.loc[s, f'{lead}{o}']) / float(ij_row.loc[s, f'V_ij_{o}']))
    return {'method': method, 'success': bool(code == 0 and written), 'exit_code': code,
            'wall_time': wall, 'V_ratio': ratios}


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', default='/Users/jtaylor/Dropbox/Research/QIJ_joint/qijdt_validation/v4')
    p.add_argument('--runs', default=None)
    args = p.parse_args(argv)
    out_dir = args.out
    runs_root = args.runs or os.path.join(out_dir, 'runs')
    os.makedirs(out_dir, exist_ok=True)

    results = [run_method(method, draws, extra_args, timeout_s, runs_root)
               for method, (draws, extra_args, timeout_s) in METHODS.items()]

    with open(os.path.join(out_dir, 'v4.json'), 'w') as f:
        json.dump({'stale_flags_dropped': list(STALE_FLAGS), 'results': results}, f, indent=2)

    lines = [f'# qijdt validation V4: all-methods cloudfil smoke '
             f'(N={N}, draws {DRAWS[0]}:{DRAWS[-1] + 1}, workers {WORKERS})', '',
             f'Stale flag dropped from runbook.sh QIJ_DEMO (no longer in run.py): '
             f'{", ".join(STALE_FLAGS)}', '',
             '| method | success | exit_code | wall_time (s) |', '|---|---|---|---|']
    for r in results:
        lines.append(f"| {r['method']} | {r['success']} | {r['exit_code']} | "
                      f"{r['wall_time']:.1f} |")
    lines.append('')
    lines.append('V ratio (method variance / ij V_ij), where applicable:')
    for r in results:
        if not r['V_ratio']:
            continue
        lines.append(f"### {r['method']}")
        for s, per_o in r['V_ratio'].items():
            row = ', '.join(f'{o}={v:.4g}' for o, v in per_o.items())
            lines.append(f'- draw {s}: {row}')
    with open(os.path.join(out_dir, 'v4.md'), 'w') as f:
        f.write('\n'.join(lines))


if __name__ == '__main__':
    main()
