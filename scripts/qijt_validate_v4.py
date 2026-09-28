#!/usr/bin/env python3.9
"""V4 (spec/QIJ_qijt_spec.md section 12): the all-methods `cloudfil`
smoke, unchanged (`oracle`, `ij`, `boot` at B = 50, `qij` at the demo
settings -- survey moments, quantized-start full-data, gptrend
quadratic, gpwidth global, refine-schedule rounds, M_X 1094 -- N =
10000, draws 0:2, workers 14), plus `qijt` at budget 50/10/5, M_X 1094.
Reports per method whether every draw's product was written and this
script's own wall time around the call.

    PYTHONPATH=src python3.9 scripts/qijt_validate_v4.py \\
        --out /Users/jtaylor/Dropbox/Research/QIJ_joint/qijt_validation/v4

Each `run.py` call is wrapped in a `perl alarm` hard kill, sized per
method for 2 draws at workers 14: 10 min (`oracle`, `ij`), 30 min
(`boot`, B = 50), 90 min (`qij`, the demo settings), 30 min (`qijt`,
budget 50/10/5, the smallest of this build's runs).
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
SEED = 0
WORKERS = 14
M_X = 1094

METHODS = {
    'oracle': ([], 600),
    'ij': ([], 600),
    'boot': (['--B', '50'], 1800),
    'qij': (['--M-X', str(M_X), '--survey', 'moments', '--quantized-start', 'full-data',
              '--gptrend', 'quadratic', '--gpwidth', 'global', '--refine-schedule', 'rounds'],
             5400),
    'qijt': (['--M-X', str(M_X), '--budget', '50', '--budget-win', '10', '--budget-quad', '5'],
              1800),
}


def _run_cmd(cmd: list, timeout_s: int) -> int:
    """Hard-kill wrapper (module docstring): `perl`'s `alarm` is pending
    across `exec`, so it reaches the real process. Returns the exit
    code (nonzero on failure or the alarm firing)."""
    env = dict(os.environ, PYTHONPATH=_SRC)
    guarded = ['perl', '-e', 'alarm(shift @ARGV); exec(@ARGV) or die $!', str(timeout_s)] + cmd
    return subprocess.run(guarded, env=env).returncode


def run_method(method: str, extra_args: list, timeout_s: int, runs_root: str) -> dict:
    """Runs one method over `DRAWS`, timed by this script's own clock;
    `success` requires both a zero exit code and every draw's scalar
    row present afterward."""
    md = products.method_dir(runs_root, DATASET, ESTIMATOR, N, method)
    cmd = [sys.executable, _RUN_PY, DATASET, ESTIMATOR, method,
           '--N', str(N), '--draws', f'{DRAWS[0]}:{DRAWS[-1] + 1}', '--seed', str(SEED),
           '--workers', str(WORKERS), '--out', runs_root] + extra_args
    t0 = time.perf_counter()
    code = _run_cmd(cmd, timeout_s)
    wall = time.perf_counter() - t0
    written = all(products.is_done(md, s) for s in DRAWS)
    return {'method': method, 'success': bool(code == 0 and written), 'exit_code': code,
            'wall_time': wall}


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', default='/Users/jtaylor/Dropbox/Research/QIJ_joint/qijt_validation/v4')
    p.add_argument('--runs', default=None)
    args = p.parse_args(argv)
    out_dir = args.out
    runs_root = args.runs or os.path.join(out_dir, 'runs')
    os.makedirs(out_dir, exist_ok=True)

    results = [run_method(method, extra_args, timeout_s, runs_root)
               for method, (extra_args, timeout_s) in METHODS.items()]

    with open(os.path.join(out_dir, 'v4.json'), 'w') as f:
        json.dump(results, f, indent=2)

    lines = [f'# qijt validation V4: all-methods cloudfil smoke '
             f'(N={N}, draws {DRAWS[0]}:{DRAWS[-1] + 1}, workers {WORKERS})', '',
             '| method | success | exit_code | wall_time (s) |', '|---|---|---|---|']
    for r in results:
        lines.append(f"| {r['method']} | {r['success']} | {r['exit_code']} | "
                      f"{r['wall_time']:.1f} |")
    with open(os.path.join(out_dir, 'v4.md'), 'w') as f:
        f.write('\n'.join(lines))


if __name__ == '__main__':
    main()
