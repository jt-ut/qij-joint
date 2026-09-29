#!/usr/bin/env python3.9
"""V3 (spec/QIJ_qijdt_spec.md section 12): bit identity, the build's one
pass/fail item. (pareto, tail) draw 0 (N = 2000, eps = 0.01) and
`cloudfil` draw 0 (N = 10000, eps = 0.01) each run at workers 1 and
workers 8; every product file for that draw must be identical after
dropping the timing columns (any column whose name contains 'wall' or
'time', case-insensitively, plus `workers` itself). "Identical" is
checked as data frames (parquet decoded, timing columns dropped,
columns sorted, rows in their stored order), not raw bytes, since the
timing columns must be removed first; `DataFrame.equals` treats two
NaNs at the same cell as equal.

    PYTHONPATH=src python3.9 scripts/qijdt_validate_v3.py \\
        --out /Users/jtaylor/Dropbox/Research/QIJ_joint/qijdt_validation/v3

Each `run.py` call (one draw) is wrapped in a `perl alarm` hard kill:
30 min for pareto/tail at either worker count, 2 h for cloudfil at
workers 1 (serial), 1 h for cloudfil at workers 8.
"""
import os

for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
           'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(_v, '1')

import argparse
import json
import subprocess
import sys

import pandas as pd

from qij_joint import products

_RUN_PY = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'run.py')
_SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src')

SEED = 0
DRAW = 0
EPS = 0.01

CASES = {
    'pareto_tail': dict(dataset='pareto', estimator='tail', N=2000, timeout=1800),
    'cloudfil': dict(dataset='cloudfil', estimator='p2', N=10000, timeout=(7200, 3600)),
}


def _run_cmd(cmd: list, timeout_s: int) -> None:
    """Hard-kill wrapper (module docstring): `perl`'s `alarm` is pending
    across `exec`, so it reaches the real process."""
    env = dict(os.environ, PYTHONPATH=_SRC)
    guarded = ['perl', '-e', 'alarm(shift @ARGV); exec(@ARGV) or die $!', str(timeout_s)] + cmd
    result = subprocess.run(guarded, env=env)
    if result.returncode != 0:
        raise RuntimeError(f'command failed (code {result.returncode}, timeout {timeout_s}s): '
                            f'{" ".join(cmd)}')


def _run_qijdt(dataset: str, estimator: str, N: int, workers: int, runs_root: str,
               timeout_s: int) -> None:
    md = products.method_dir(runs_root, dataset, estimator, N, 'qijdt')
    if products.is_done(md, DRAW):
        return
    cmd = [sys.executable, _RUN_PY, dataset, estimator, 'qijdt',
           '--N', str(N), '--draws', f'{DRAW}:{DRAW + 1}', '--seed', str(SEED),
           '--workers', str(workers), '--out', runs_root, '--eps', str(EPS),
           '--diag-draws', f'{DRAW}:{DRAW + 1}']
    _run_cmd(cmd, timeout_s)


def _array_kinds(md: str, s: int) -> list:
    """Every array kind written for draw `s` (the file names' own
    `.<kind>.` segment), from a plain directory listing rather than a
    hardcoded product list."""
    prefix, kinds = f's{s:05d}.', []
    for name in sorted(os.listdir(md)):
        if name.startswith(prefix) and name.endswith('.parquet') and name.count('.') == 2:
            kinds.append(name[len(prefix):-len('.parquet')])
    return kinds


def _drop_timing(df: pd.DataFrame) -> pd.DataFrame:
    """Every column whose name contains 'wall' or 'time' (case-
    insensitive), dropped -- covers `busy_time` and every `wall_<stage>`
    without naming each one; `workers` itself also dropped."""
    keep = [c for c in df.columns if 'wall' not in c.lower() and 'time' not in c.lower()
            and c != 'workers']
    return df[keep]


def _load(md: str, s: int, kind: str = None) -> pd.DataFrame:
    name = f's{s:05d}.parquet' if kind is None else f's{s:05d}.{kind}.parquet'
    df = _drop_timing(pd.read_parquet(os.path.join(md, name)))
    return df[sorted(df.columns)].reset_index(drop=True)


def compare_draw(md_a: str, md_b: str, s: int) -> dict:
    """{'row': bool, '<kind>': bool, ...}: whether every product file
    for draw `s` under `md_a`/`md_b` is identical once the timing
    columns are dropped."""
    result = {'row': _load(md_a, s).equals(_load(md_b, s))}
    kinds_a, kinds_b = set(_array_kinds(md_a, s)), set(_array_kinds(md_b, s))
    for kind in sorted(kinds_a | kinds_b):
        if kind not in kinds_a or kind not in kinds_b:
            result[kind] = False
            continue
        result[kind] = _load(md_a, s, kind).equals(_load(md_b, s, kind))
    return result


def run_case(name: str, spec: dict, out_dir: str) -> dict:
    dataset, estimator, N = spec['dataset'], spec['estimator'], spec['N']
    timeout = spec['timeout']
    t1, t8 = timeout if isinstance(timeout, tuple) else (timeout, timeout)

    root1 = os.path.join(out_dir, f'{name}_w1')
    root8 = os.path.join(out_dir, f'{name}_w8')
    _run_qijdt(dataset, estimator, N, 1, root1, t1)
    _run_qijdt(dataset, estimator, N, 8, root8, t8)

    md1 = products.method_dir(root1, dataset, estimator, N, 'qijdt')
    md8 = products.method_dir(root8, dataset, estimator, N, 'qijdt')
    per_file = compare_draw(md1, md8, DRAW)
    return {'case': name, 'per_file': per_file, 'pass': all(per_file.values())}


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', default='/Users/jtaylor/Dropbox/Research/QIJ_joint/qijdt_validation/v3')
    args = p.parse_args(argv)
    os.makedirs(args.out, exist_ok=True)

    results = [run_case(name, spec, args.out) for name, spec in CASES.items()]
    overall = all(r['pass'] for r in results)

    with open(os.path.join(args.out, 'v3.json'), 'w') as f:
        json.dump({'overall_pass': overall, 'cases': results}, f, indent=2)

    lines = [f'# qijdt validation V3: bit identity (workers 1 vs 8, draw {DRAW})', '',
             f'**Overall: {"PASS" if overall else "FAIL"}**', '']
    for r in results:
        lines.append(f"## {r['case']}: {'PASS' if r['pass'] else 'FAIL'}")
        for kind, ok in r['per_file'].items():
            lines.append(f"- {kind or 'scalar row'}: {'identical' if ok else 'DIFFERS'}")
        lines.append('')
    with open(os.path.join(args.out, 'v3.md'), 'w') as f:
        f.write('\n'.join(lines))

    print('PASS' if overall else 'FAIL')


if __name__ == '__main__':
    main()
