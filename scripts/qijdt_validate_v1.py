#!/usr/bin/env python3.9
"""V1 (spec/QIJ_qijdt_spec.md section 12): the six paper cases at
N = 2000, draws 0-9, eps = 0.01, workers 8: from `curve`, the
evals_total at which V_btw,o first reaches 0.99 of the stored `ij`
V_ij,o ('never' if it does not), V_tot,o / V_ij,o and evals_total at the
end, tree_status, the three intervals' coverage of the registry truth
over the 10 draws, b_hat/accel/c_q per output. Beside it `qij`'s own
cost on the same draws, in evaluations and in rows/N.

    PYTHONPATH=src python3.9 scripts/qijdt_validate_v1.py \\
        --out /Users/jtaylor/Dropbox/Research/QIJ_joint/qijdt_validation/v1

`ij` and `qij` products for these exact draws (N=2000, seed 0) already
exist under
/Users/jtaylor/Dropbox/Research/QIJ_joint/qijt_validation/v1/runs
(qijt's own V1 run); this script copies them into its own `runs_root`
if present there and covering draws 0-9, else runs them fresh via
`run.py`. `qijdt` is always run fresh (no cached product reused for
it). Each `run.py` call is wrapped in a `perl alarm` hard kill (a plain
subprocess timeout leaves the pool's worker processes running): 20 min
for `ij`, 90 min for `qij`, 30 min for `qijdt` per case's whole 10-draw
call (qijdt's own per-node cost is a handful of evaluations per draw
even for the slow-per-evaluation `imf` case, confirmed by a calibration
run: ~90 evals/draw)."""
import os
import shutil

for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
           'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(_v, '1')

import argparse
import json
import subprocess
import sys

import pandas as pd

from qij_joint import products, registry

_RUN_PY = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'run.py')
_SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src')

CASES = (('pareto', 'shape'), ('pareto', 'tail'), ('mvt', 'nu'), ('mvt', 'tail'),
          ('fp', 'all'), ('imf', 'all'))
N = 2000
DRAWS = range(0, 10)
SEED = 0
WORKERS = 8
EPS = 0.01
TIMEOUT_IJ, TIMEOUT_QIJ, TIMEOUT_QIJDT = 1200, 5400, 1800

SOURCE_RUNS = '/Users/jtaylor/Dropbox/Research/QIJ_joint/qijt_validation/v1/runs'

QIJDT_STAGES = ('full_fit', 'eta_full', 'tree', 'bias', 'within', 'quadratic', 'curvature')
QIJ_STAGES = ('prototype', 'full_data', 'refinement', 'curvature', 'eta_full')
LEVELS = (('lo', 'hi'), ('lo_btw', 'hi_btw'), ('lo_abc', 'hi_abc'))


def _run_cmd(cmd: list, timeout_s: int) -> None:
    """Run `cmd` with a hard `timeout_s`-second kill via `perl`'s
    `alarm` (POSIX preserves a pending alarm across `exec`, so the
    signal reaches the real process rather than a wrapper subprocess
    timeout would leave hanging pool workers). Raises on a nonzero
    exit, including one killed by the alarm."""
    env = dict(os.environ, PYTHONPATH=_SRC)
    guarded = ['perl', '-e', 'alarm(shift @ARGV); exec(@ARGV) or die $!', str(timeout_s)] + cmd
    result = subprocess.run(guarded, env=env)
    if result.returncode != 0:
        raise RuntimeError(f'command failed (code {result.returncode}, timeout {timeout_s}s): '
                            f'{" ".join(cmd)}')


def _copy_or_run(dataset: str, estimator: str, method: str, runs_root: str,
                  extra_args: list, timeout_s: int) -> None:
    """If `runs_root` already has every draw 0-9 for this (dataset,
    estimator, method), do nothing; else if `SOURCE_RUNS` has every
    draw 0-9 (seed 0), copy that method's whole directory in; else run
    it fresh via `run.py`."""
    md = products.method_dir(runs_root, dataset, estimator, N, method)
    draws = list(DRAWS)
    if all(products.is_done(md, s) for s in draws):
        return
    src_md = products.method_dir(SOURCE_RUNS, dataset, estimator, N, method)
    if os.path.isdir(src_md) and all(products.is_done(src_md, s) for s in draws):
        os.makedirs(os.path.dirname(md), exist_ok=True)
        shutil.copytree(src_md, md, dirs_exist_ok=True)
        return
    cmd = [sys.executable, _RUN_PY, dataset, estimator, method,
           '--N', str(N), '--draws', f'{draws[0]}:{draws[-1] + 1}',
           '--seed', str(SEED), '--workers', str(WORKERS), '--out', runs_root] + extra_args
    _run_cmd(cmd, timeout_s)


def crossing_evals(curve: pd.DataFrame, outputs, V_ij: dict) -> dict:
    """Per output: the smallest `evals_total` (curve, ascending round)
    at which V_btw_o reaches 0.99*V_ij_o; 'never' if the stored curve
    never does (spec section 12, V1)."""
    curve = curve.sort_values('round')
    out = {}
    for o in outputs:
        threshold = 0.99 * V_ij[o]
        hit = curve[curve[f'V_btw_{o}'] >= threshold]
        out[o] = int(hit['evals_total'].iloc[0]) if len(hit) else 'never'
    return out


def coverage(rows: pd.DataFrame, outputs, truth: dict, lo: str, hi: str) -> dict:
    """Per output: the fraction of `rows` (one per draw) whose stored
    [lo, hi] interval contains `truth_o` (spec section 12, V1)."""
    out = {}
    for o in outputs:
        contains = (truth[o] >= rows[f'{lo}_{o}']) & (truth[o] <= rows[f'{hi}_{o}'])
        out[o] = float(contains.mean())
    return out


def qijdt_cost(row: pd.Series, outputs) -> dict:
    return {'evals': int(row['evals_total']), 'rows_per_N': float(row['evals_total'])}


def qij_stage_cost(row: pd.Series, stages) -> dict:
    evals = int(sum(row[f'evals_{s}'] for s in stages))
    rows_n = float(sum(row[f'rows_{s}'] for s in stages)) / row['N']
    return {'evals': evals, 'rows_per_N': rows_n}


def run_case(dataset: str, estimator: str, runs_root: str) -> dict:
    T = registry.case(dataset, estimator).make_T()

    _copy_or_run(dataset, estimator, 'ij', runs_root, [], TIMEOUT_IJ)
    _copy_or_run(dataset, estimator, 'qij', runs_root, [], TIMEOUT_QIJ)
    md_qijdt = products.method_dir(runs_root, dataset, estimator, N, 'qijdt')
    if not all(products.is_done(md_qijdt, s) for s in DRAWS):
        cmd = [sys.executable, _RUN_PY, dataset, estimator, 'qijdt',
               '--N', str(N), '--draws', f'{DRAWS[0]}:{DRAWS[-1] + 1}',
               '--seed', str(SEED), '--workers', str(WORKERS), '--out', runs_root,
               '--eps', str(EPS)]
        _run_cmd(cmd, TIMEOUT_QIJDT)

    ij_df = products.collect(runs_root, dataset, estimator, N, 'ij').set_index('s')
    qij_df = products.collect(runs_root, dataset, estimator, N, 'qij').set_index('s')
    qijdt_df = products.collect(runs_root, dataset, estimator, N, 'qijdt').set_index('s')
    curve_df = products.collect_array(runs_root, dataset, estimator, N, 'qijdt',
                                       'curve', draws=list(DRAWS))

    outputs = [c[len('V_tot_'):] for c in qijdt_df.columns if c.startswith('V_tot_')]
    truth_arr = registry.truth(dataset, estimator)
    T_outputs = list(T.outputs)
    truth = {o: float(truth_arr[T_outputs.index(o)]) for o in outputs}

    per_draw = []
    for s in DRAWS:
        row_ij, row_qijdt, row_qij = ij_df.loc[s], qijdt_df.loc[s], qij_df.loc[s]
        V_ij = {o: float(row_ij[f'V_ij_{o}']) for o in outputs}
        crossing = crossing_evals(curve_df[curve_df['s'] == s], outputs, V_ij)
        ratio = {o: float(row_qijdt[f'V_tot_{o}']) / V_ij[o] for o in outputs}
        per_draw.append({
            's': s, 'crossing_evals': crossing, 'V_tot_over_V_ij': ratio,
            'tree_status': row_qijdt['tree_status'],
            'evals_total_qijdt': int(row_qijdt['evals_total']),
            'abc_terms': {o: {k: float(row_qijdt[f'{k}_{o}']) for k in ('b_hat', 'accel', 'c_q')}
                          for o in outputs},
            'qijdt_cost': {'evals': int(row_qijdt['evals_total'])},
            'qij_cost': qij_stage_cost(row_qij, QIJ_STAGES),
        })

    cov = {f'{lo}_{hi}': coverage(qijdt_df.loc[list(DRAWS)], outputs, truth, lo, hi)
           for lo, hi in LEVELS}

    return {'dataset': dataset, 'estimator': estimator, 'outputs': outputs,
            'truth': truth, 'per_draw': per_draw, 'coverage': cov}


def _md_table(headers: list, rows: list) -> str:
    lines = ['| ' + ' | '.join(headers) + ' |',
             '|' + '|'.join('---' for _ in headers) + '|']
    for r in rows:
        lines.append('| ' + ' | '.join(str(v) for v in r) + ' |')
    return '\n'.join(lines)


def write_report(cases: list, out_dir: str) -> None:
    """`v1.json` and `v1.md` under `out_dir`."""
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, 'v1.json'), 'w') as f:
        json.dump(cases, f, indent=2)

    lines = [f'# qijdt validation V1 (N={N}, draws 0-9, eps {EPS}, workers {WORKERS})', '']
    for case in cases:
        lines.append(f"## {case['dataset']} {case['estimator']}")
        lines.append('')
        rows = []
        for d in case['per_draw']:
            for o in case['outputs']:
                t = d['abc_terms'][o]
                rows.append([d['s'], o, d['crossing_evals'][o],
                             round(d['V_tot_over_V_ij'][o], 4), d['tree_status'],
                             d['evals_total_qijdt'],
                             f"{t['b_hat']:.3g}", f"{t['accel']:.3g}", f"{t['c_q']:.3g}"])
        lines.append(_md_table(['draw', 'output', 'crossing_evals', 'V_tot/V_ij',
                                'tree_status', 'evals_total', 'b_hat', 'accel', 'c_q'], rows))
        lines.append('')
        cov_rows = [[o] + [round(case['coverage'][f'{lo}_{hi}'][o], 2) for lo, hi in LEVELS]
                    for o in case['outputs']]
        lines.append('Coverage of registry truth over the 10 draws (normal/btw/abc):')
        lines.append(_md_table(['output', 'normal', 'btw', 'abc'], cov_rows))
        lines.append('')
        cost_rows = []
        for d in case['per_draw']:
            cost_rows.append([d['s'], d['qijdt_cost']['evals'],
                               d['qij_cost']['evals'], round(d['qij_cost']['rows_per_N'], 3)])
        lines.append('Cost by draw: qijdt evals_total vs qij evals/rows-per-N:')
        lines.append(_md_table(['draw', 'qijdt evals', 'qij evals', 'qij rows/N'], cost_rows))
        lines.append('')
    with open(os.path.join(out_dir, 'v1.md'), 'w') as f:
        f.write('\n'.join(lines))


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', default='/Users/jtaylor/Dropbox/Research/QIJ_joint/qijdt_validation/v1')
    p.add_argument('--runs', default=None,
                    help='product root for ij/qij/qijdt runs (default: <out>/runs)')
    args = p.parse_args(argv)
    runs_root = args.runs or os.path.join(args.out, 'runs')

    cases = [run_case(dataset, estimator, runs_root) for dataset, estimator in CASES]
    write_report(cases, args.out)


if __name__ == '__main__':
    main()
