#!/usr/bin/env python3.9
"""V1 (spec/QIJ_qijt_spec.md section 12): the six paper cases at N =
2000, draws 0-9, M_X = `cost_rule_M(N, q, 0.01)`, budget 400/100/50,
workers 8. A report, no pass/fail: the crossing evaluation count at
which V_btw,o first reaches 0.99 of the stored `ij` V_ij,o, the final
V_tot,o / V_ij,o, the three intervals' truth coverage over the 10
draws, and the evaluation/row cost by stage in size-N units, alongside
`qij`'s own cost at its defaults on the same draws.

    PYTHONPATH=src python3.9 scripts/qijt_validate_v1.py \\
        --out /Users/jtaylor/Dropbox/Research/QIJ_joint/qijt_validation/v1

`ij` and `qij` are run first if their products are absent under
`--runs`; `qijt` likewise. Each `run.py` call is wrapped in a `perl
alarm` hard kill (a plain subprocess timeout leaves the pool's worker
processes running): 20 min for `ij` (10 analytic draws at N = 2000),
90 min for `qij` (multistart plus refinement), 120 min for `qijt`
(tree growth to budget 400 plus anchors/within/quadratic/curvature),
each for the whole 10-draw call.
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

from qij_joint import products, registry
from qij_joint.core.xvq import cost_rule_M

_RUN_PY = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'run.py')
_SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src')

CASES = (('pareto', 'shape'), ('pareto', 'tail'), ('mvt', 'nu'), ('mvt', 'tail'),
          ('fp', 'all'), ('imf', 'all'))
N = 2000
DRAWS = range(0, 10)
SEED = 0
WORKERS = 8
BUDGET, BUDGET_WIN, BUDGET_QUAD = 400, 100, 50
COST_EPS = 0.01
TIMEOUT_IJ, TIMEOUT_QIJ, TIMEOUT_QIJT = 1200, 5400, 7200

QIJT_STAGES = ('full_fit', 'eta_full', 'xvq', 'base_Q', 'eta_Q', 'tree',
               'anchors', 'bias', 'within', 'quadratic', 'drift', 'curvature')
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


def _ensure(dataset: str, estimator: str, method: str, runs_root: str,
            extra_args: list, timeout_s: int) -> None:
    """Runs `scripts/run.py <dataset> <estimator> <method>` over `DRAWS`
    if any draw's product under `runs_root` is missing."""
    md = products.method_dir(runs_root, dataset, estimator, N, method)
    draws = list(DRAWS)
    if all(products.is_done(md, s) for s in draws):
        return
    cmd = [sys.executable, _RUN_PY, dataset, estimator, method,
           '--N', str(N), '--draws', f'{draws[0]}:{draws[-1] + 1}',
           '--seed', str(SEED), '--workers', str(WORKERS), '--out', runs_root] + extra_args
    _run_cmd(cmd, timeout_s)


def crossing_evals(curve: pd.DataFrame, outputs, V_ij: dict) -> dict:
    """Per output: the smallest `evals_tree` (curve, ascending round) at
    which V_btw_o reaches 0.99*V_ij_o; 'never' if the stored curve
    never does (spec section 12, V1)."""
    curve = curve.sort_values('round')
    out = {}
    for o in outputs:
        threshold = 0.99 * V_ij[o]
        hit = curve[curve[f'V_btw_{o}'] >= threshold]
        out[o] = int(hit['evals_tree'].iloc[0]) if len(hit) else 'never'
    return out


def coverage(rows: pd.DataFrame, outputs, truth: dict, lo: str, hi: str) -> dict:
    """Per output: the fraction of `rows` (one per draw) whose stored
    [lo, hi] interval contains `truth_o` (spec section 12, V1)."""
    out = {}
    for o in outputs:
        contains = (truth[o] >= rows[f'{lo}_{o}']) & (truth[o] <= rows[f'{hi}_{o}'])
        out[o] = float(contains.mean())
    return out


def stage_cost(row: pd.Series, stages) -> dict:
    """(evals, rows_per_N) summed over `stages` for one scalar row."""
    evals = int(sum(row[f'evals_{s}'] for s in stages))
    rows_n = float(sum(row[f'rows_{s}'] for s in stages)) / row['N']
    return {'evals': evals, 'rows_per_N': rows_n}


def run_case(dataset: str, estimator: str, runs_root: str) -> dict:
    """Everything reported for one case: ensures the three methods'
    products exist, then computes the crossing, ratio, coverage and
    stage-cost quantities per draw and the coverage average over
    draws."""
    T = registry.case(dataset, estimator).make_T()
    outputs = T.outputs
    M_X = cost_rule_M(N, len(outputs), COST_EPS)

    _ensure(dataset, estimator, 'ij', runs_root, [], TIMEOUT_IJ)
    _ensure(dataset, estimator, 'qij', runs_root, [], TIMEOUT_QIJ)
    _ensure(dataset, estimator, 'qijt', runs_root,
            ['--M-X', str(M_X), '--budget', str(BUDGET),
             '--budget-win', str(BUDGET_WIN), '--budget-quad', str(BUDGET_QUAD)],
            TIMEOUT_QIJT)

    ij_df = products.collect(runs_root, dataset, estimator, N, 'ij').set_index('s')
    qij_df = products.collect(runs_root, dataset, estimator, N, 'qij').set_index('s')
    qijt_df = products.collect(runs_root, dataset, estimator, N, 'qijt').set_index('s')
    curve_df = products.collect_array(runs_root, dataset, estimator, N, 'qijt',
                                       'curve', draws=list(DRAWS))

    truth_arr = registry.truth(dataset, estimator)
    truth = {o: float(truth_arr[j]) for j, o in enumerate(outputs)}

    per_draw = []
    for s in DRAWS:
        row_ij, row_qijt, row_qij = ij_df.loc[s], qijt_df.loc[s], qij_df.loc[s]
        V_ij = {o: float(row_ij[f'V_ij_{o}']) for o in outputs}
        crossing = crossing_evals(curve_df[curve_df['s'] == s], outputs, V_ij)
        ratio = {o: float(row_qijt[f'V_tot_{o}']) / V_ij[o] for o in outputs}
        per_draw.append({
            's': s, 'crossing_evals': crossing, 'V_tot_over_V_ij': ratio,
            'qijt_cost': stage_cost(row_qijt, QIJT_STAGES),
            'qij_cost': stage_cost(row_qij, QIJ_STAGES),
        })

    cov = {f'{lo}_{hi}': coverage(qijt_df.loc[list(DRAWS)], outputs, truth, lo, hi)
           for lo, hi in LEVELS}

    return {'dataset': dataset, 'estimator': estimator, 'outputs': list(outputs),
            'M_X': M_X, 'truth': truth, 'per_draw': per_draw, 'coverage': cov}


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

    lines = [f'# qijt validation V1 (N={N}, draws 0-9, budget {BUDGET}/'
              f'{BUDGET_WIN}/{BUDGET_QUAD}, workers {WORKERS})', '']
    for case in cases:
        lines.append(f"## {case['dataset']} {case['estimator']} (M_X={case['M_X']})")
        lines.append('')
        rows = []
        for d in case['per_draw']:
            for o in case['outputs']:
                rows.append([d['s'], o, d['crossing_evals'][o],
                             round(d['V_tot_over_V_ij'][o], 4)])
        lines.append(_md_table(['draw', 'output', 'crossing_evals', 'V_tot/V_ij'], rows))
        lines.append('')
        cov_rows = [[o] + [round(case['coverage'][f'{lo}_{hi}'][o], 2) for lo, hi in LEVELS]
                    for o in case['outputs']]
        lines.append('Coverage of registry truth over the 10 draws:')
        lines.append(_md_table(['output', 'normal', 'btw', 'abc'], cov_rows))
        lines.append('')
        cost_rows = []
        for d in case['per_draw']:
            cost_rows.append([d['s'], d['qijt_cost']['evals'],
                               round(d['qijt_cost']['rows_per_N'], 3),
                               d['qij_cost']['evals'], round(d['qij_cost']['rows_per_N'], 3)])
        lines.append('Cost by stage, summed, in size-N units:')
        lines.append(_md_table(['draw', 'qijt evals', 'qijt rows/N', 'qij evals', 'qij rows/N'],
                                cost_rows))
        lines.append('')
    with open(os.path.join(out_dir, 'v1.md'), 'w') as f:
        f.write('\n'.join(lines))


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', default='/Users/jtaylor/Dropbox/Research/QIJ_joint/qijt_validation/v1')
    p.add_argument('--runs', default=None,
                    help='product root for ij/qij/qijt runs (default: <out>/runs)')
    args = p.parse_args(argv)
    runs_root = args.runs or os.path.join(args.out, 'runs')

    cases = [run_case(dataset, estimator, runs_root) for dataset, estimator in CASES]
    write_report(cases, args.out)


if __name__ == '__main__':
    main()
