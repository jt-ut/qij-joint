#!/usr/bin/env python3.9
"""Direct-estimator validation of the mixture cold search
(spec/QIJ_estimator_search_spec.md section 3, items 1-3): every check
calls `GMM2D`/`P2Mixture` (via the `cloudfil`/`p2` registry case)
directly -- no `qij`, `boot`, `oracle`, `ij` or `ijfd` pipeline
invocation, at any size. Runs unchanged against the pre-search-change
tree and the rewritten one: `--src` selects which package tree is
imported.

    PYTHONPATH=src python3.9 scripts/validate_gmm_search.py \\
        --src src --out /tmp/gmm_search_validate --workers 8

Items 1-2 (the eight audited draws' cold fit + search audit, and their
cost) run together, one task per draw, over a `qij_joint.parallel.Pool`
built with the registry key (contract rule 6): `Pool(workers,
case=('cloudfil', 'p2'))`. Item 3 (the five diagnostic continuations)
runs in-process, five fits, reusing `scripts/validate_gmm_fit.py`'s own
case builders (`_build_case_defs`, `_status_of`) by import rather than
copy, so the two scripts' notion of the five cases never drifts apart.

Every field new to the search rewrite (`n_starts_screened`,
`screen_rounds`, `n_survivors`, `screen_status`, `n_smem_tried`,
`n_smem_accepted`, `smem_wall_time`) is read off `T.last_fit_info` with
`getattr(..., default=nan)`, so this script also runs, and reports NaN
for those columns, against the current (pre-rewrite) code, which has
none of them.
"""
import os

for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
           'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(_v, '1')

import argparse
import json
import sys
import time
from datetime import datetime

import numpy as np

DATASET = 'cloudfil'
ESTIMATOR = 'p2'
N_CLOUDFIL = 10000
DEFAULT_DRAWS = '45,3,39,0,1,2,10,20'

# The `last_fit_info` fields new to the search rewrite (planner's
# rewrite of spec/QIJ_estimator_search_spec.md, 29 September 2026,
# section 2.4's product list): read with a NaN default so this script
# runs unchanged against code that does not yet carry them.
_SEARCH_INFO_FIELDS = ('n_starts_screened', 'screen_rounds', 'n_survivors',
                        'screen_status', 'n_smem_tried', 'n_smem_accepted',
                        'smem_wall_time')
_FIT_INFO_FIELDS = ('score', 'n_iter_em', 'n_iter_newton') + _SEARCH_INFO_FIELDS


# ======================================================================
# `last_fit_info` reader -- old and new code alike.
# ======================================================================

def _info_dict(info) -> dict:
    """`T.last_fit_info` -> a plain dict of `_FIT_INFO_FIELDS`, each read
    by `getattr` with a NaN default: present on the current (pre-search-
    change) code for the first three, absent (and so NaN here) for the
    seven search-rewrite fields until that code lands."""
    out = {}
    search = (getattr(info, 'search', None) or {}) if info is not None else {}
    for name in _FIT_INFO_FIELDS:
        v = getattr(info, name, search.get(name, float('nan'))) if info is not None else float('nan')
        out[name] = float('nan') if v is None else v
    return out


# ======================================================================
# Items 1-2 -- cold fit + search audit, one task per audited draw.
# ======================================================================

def _cold_audit_task(T, case, X, task):
    """One draw: the cold fit `T(X_s, ones)`, then
    `qij_joint.audit.search_audit` (spec item 1), called only after this
    fit's own `last_fit_info` has been read -- the audit's own
    continuation overwrites it. Imports are done here, at call time, so
    this function is correct whether the pool runs in-process, forks
    (inheriting the parent's `sys.path.insert`), or spawns fresh
    interpreters under `PYTHONPATH` (contract rule 6)."""
    from qij_joint.audit import search_audit
    from qij_joint.parallel import fit_status

    s, N = task
    t0 = time.perf_counter()
    Xd = case.draw(N, s)
    w = np.ones(N)
    try:
        theta_hat = np.asarray(T(Xd, w), dtype=float)
        status = fit_status(T, theta_hat)
    except Exception:
        theta_hat = np.full(len(T.outputs), np.nan)
        status = fit_status(T, theta_hat, raised=True)
    wall_cold = time.perf_counter() - t0

    info = _info_dict(getattr(T, 'last_fit_info', None))
    audit = search_audit(T, Xd, theta_hat, case.dataset, case.estimator)

    eta = float(getattr(T, 'eta', float('nan')))
    gap = audit['search_gap']
    passed = bool(np.isfinite(gap) and gap >= -eta)
    n_smem = info['n_smem_accepted']
    if not passed:
        mechanism = 'FAIL'
    elif isinstance(n_smem, float) and np.isnan(n_smem):
        mechanism = 'n/a (no smem data)'
    elif n_smem == 0:
        mechanism = 'screen alone'
    else:
        mechanism = 'split-merge needed'

    row = dict(draw=s, status=status, eta=eta, wall_time_cold=wall_cold,
                score=info['score'], n_iter_em=info['n_iter_em'],
                n_iter_newton=info['n_iter_newton'],
                n_starts_screened=info['n_starts_screened'],
                screen_rounds=info['screen_rounds'],
                n_survivors=info['n_survivors'],
                screen_status=info['screen_status'],
                n_smem_tried=info['n_smem_tried'],
                n_smem_accepted=info['n_smem_accepted'],
                smem_wall_time=info['smem_wall_time'],
                search_gap=gap, search_failed=audit['search_failed'],
                search_status=audit['search_status'],
                search_wall_time=audit['search_wall_time'],
                pass_=passed, mechanism=mechanism,
                theta=theta_hat.tolist())
    return s, row


def run_items_1_2(mod, args, draws) -> tuple:
    Pool = mod['Pool']
    pool = Pool(args.workers, case=(DATASET, ESTIMATOR))
    tasks = [(s, N_CLOUDFIL) for s in draws]
    t0 = time.perf_counter()
    results = pool.map(_cold_audit_task, tasks)
    wall = time.perf_counter() - t0
    pool.close()
    rows = [r for _, r in results]
    return rows, wall


# ======================================================================
# Item 3 -- the five diagnostic continuation cases, reused from
# `validate_gmm_fit.py` by import (never copied).
# ======================================================================

def run_item3(mod, vgf, args) -> list:
    registry = mod['registry']
    xvq_mod = mod['xvq']
    case_defs = vgf._build_case_defs(xvq_mod, registry)
    case = registry.case(DATASET, ESTIMATOR)
    rows = []
    for name, (Xc, wc, theta_hat, eta, label) in case_defs.items():
        T = case.make_T()
        t0 = time.perf_counter()
        theta = np.asarray(T(Xc, np.asarray(wc, dtype=float), start=theta_hat, eta=eta),
                            dtype=float)
        wall = time.perf_counter() - t0
        status, info = vgf._status_of(theta, T)
        rows.append(dict(
            case=name, label=label, n_rows=int(len(Xc)), eta=eta, status=status,
            em_iters=vgf._info_field(info, 'em_iters'),
            newton_iters=vgf._info_field(info, 'newton_iters'),
            score_scaled=vgf._info_field(info, 'score_scaled'),
            wall_time=wall, theta=theta.tolist(),
        ))
    return rows


def _apply_reference(rows: list, reference_path: str) -> float:
    """Diffs each item-3 row's `theta` against `reference_path` (a prior
    run of this same script's own JSON, or an `item3.rows` list of the
    same shape), by case name; adds `ref_max_rel_diff`/`ref_pass` (pass
    at 1e-7 relative) to each row in place. Returns the overall max
    relative difference across all five cases."""
    with open(reference_path) as f:
        data = json.load(f)
    ref_rows = data['item3']['rows'] if isinstance(data, dict) and 'item3' in data else data
    ref_by_case = {r['case']: np.asarray(r['theta'], dtype=float) for r in ref_rows}
    overall = 0.0
    for r in rows:
        ref = ref_by_case.get(r['case'])
        a = np.asarray(r['theta'], dtype=float)
        if ref is None or len(ref) != len(a):
            r['ref_max_rel_diff'] = float('nan')
            r['ref_pass'] = False
            continue
        scale = np.maximum(np.abs(a), np.abs(ref))
        scale = np.where(scale > 0, scale, 1.0)
        rel = np.abs(a - ref) / scale
        m = float(np.nanmax(rel))
        r['ref_max_rel_diff'] = m
        r['ref_pass'] = bool(m <= 1e-7)
        overall = max(overall, m)
    return overall


# ======================================================================
# Import bootstrap, CLI, and JSON/Markdown reporting.
# ======================================================================

def _load_package(src: str) -> dict:
    """Prepend `src` to `sys.path` and import the package under test --
    the same script then runs unchanged against the old and the new
    search code by pointing `--src` at each tree's `src/` directory."""
    sys.path.insert(0, os.path.abspath(src))
    from qij_joint import registry
    from qij_joint.core import xvq as xvq_mod
    from qij_joint.parallel import Pool
    return dict(registry=registry, xvq=xvq_mod, Pool=Pool)


def _load_vgf():
    """Import `scripts/validate_gmm_fit.py` (this script's own sibling)
    by module name, for its five-case builders (`_build_case_defs`,
    `_status_of`, `_info_field`) -- reused, not copied (item 3)."""
    scripts_dir = os.path.dirname(os.path.abspath(__file__))
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)
    import validate_gmm_fit as vgf
    return vgf


def _json_default(o):
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(str(type(o)))


_SKIP_COLS = {'theta'}


def _table(rows: list) -> list:
    if not rows:
        return ['_no rows_']
    cols = [c for c in rows[0].keys() if c not in _SKIP_COLS]
    lines = ['| ' + ' | '.join(cols) + ' |', '|' + '---|' * len(cols)]
    for r in rows:
        lines.append('| ' + ' | '.join(str(r.get(c, '')) for c in cols) + ' |')
    return lines


def _write_markdown(results: dict, path: str, src: str) -> None:
    lines = ['# GMM cold-search validation -- spec/QIJ_estimator_search_spec.md section 3, items 1-3',
              '', f'Package under test: `{os.path.abspath(src)}`', '']

    e12 = results['item1_2']
    lines.append('## Items 1-2 -- eight audited draws: cold fit, search audit, cost, mechanism')
    lines.append(f"(item wall time: {e12['item_wall_time']:.1f} s, run at {e12['timestamp']}, "
                  f"overall PASS: {e12['overall_pass']})")
    lines.append('')
    lines.extend(_table(e12['rows']))
    lines.append('')

    e3 = results['item3']
    ref_note = ''
    if e3.get('reference'):
        ref_note = (f", reference: `{e3['reference']}`, "
                    f"max rel diff: {e3['ref_max_rel_diff']:.3e}, "
                    f"pass (<=1e-7): {e3['ref_max_rel_diff'] <= 1e-7}")
    lines.append('## Item 3 -- five diagnostic continuations (unchanged to 1e-7 relative)')
    lines.append(f"(item wall time: {e3['item_wall_time']:.1f} s, run at {e3['timestamp']}{ref_note})")
    lines.append('')
    lines.extend(_table(e3['rows']))
    lines.append('')

    with open(path, 'w') as f:
        f.write('\n'.join(lines) + '\n')


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--src', required=True, help="package's src/ directory to import")
    ap.add_argument('--out', required=True, help='output directory')
    ap.add_argument('--workers', type=int, default=1, help='Pool size for items 1-2')
    ap.add_argument('--draws', default=DEFAULT_DRAWS,
                     help='comma list of draw seeds for items 1-2 (default: the eight audited draws)')
    ap.add_argument('--reference', default=None,
                     help="item 3's reference JSON (a prior run of this script) to diff theta "
                          "against by case name; pass at 1e-7 relative")
    args = ap.parse_args(argv)

    mod = _load_package(args.src)
    vgf = _load_vgf()
    draws = [int(x) for x in args.draws.split(',') if x.strip()]

    os.makedirs(args.out, exist_ok=True)

    rows_12, wall_12 = run_items_1_2(mod, args, draws)
    overall_pass = all(r['pass_'] for r in rows_12)
    print(f'items 1-2: done in {wall_12:.1f}s, overall PASS: {overall_pass}', file=sys.stderr)

    t0 = time.perf_counter()
    rows_3 = run_item3(mod, vgf, args)
    wall_3 = time.perf_counter() - t0
    ref_max_rel = float('nan')
    if args.reference:
        ref_max_rel = _apply_reference(rows_3, args.reference)
    print(f'item 3: done in {wall_3:.1f}s', file=sys.stderr)

    results = {
        'item1_2': dict(rows=rows_12, item_wall_time=wall_12, overall_pass=overall_pass,
                         draws=draws, src=os.path.abspath(args.src),
                         timestamp=datetime.now().isoformat(timespec='seconds')),
        'item3': dict(rows=rows_3, item_wall_time=wall_3,
                      reference=os.path.abspath(args.reference) if args.reference else None,
                      ref_max_rel_diff=ref_max_rel if args.reference else None,
                      src=os.path.abspath(args.src),
                      timestamp=datetime.now().isoformat(timespec='seconds')),
    }
    json_path = os.path.join(args.out, 'validate_gmm_search.json')
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2, default=_json_default)
    _write_markdown(results, os.path.join(args.out, 'validate_gmm_search.md'), args.src)
    print(f'done -> {json_path}', file=sys.stderr)


if __name__ == '__main__':
    main()
