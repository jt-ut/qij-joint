#!/usr/bin/env python3.9
"""Direct-estimator validation of the mixture fit (spec/QIJ_estimator_fit_spec.md
section 3, items 1-6): every check calls `GMM2D`/`P2Mixture` (via the
`cloudfil`/`p2` registry case) directly -- no `qij`, `boot`, `oracle`,
`ij` or `ijfd` pipeline invocation, at any size. Runs unchanged against
the old (`clean`) and the rewritten `_fit`: `--src` selects which
package tree is imported, so the same script and the same reference
artifacts (the stored `cloudfil_clean/cloudfil_p2_N10000` products) are
used both times.

    PYTHONPATH=src python3.9 scripts/validate_gmm_fit.py \\
        --src src --out /tmp/gmm_validate --items 1,2,3,4,5,6 --workers 14

Items 2-4 use `qij_joint.parallel.Pool`, built with either the
estimator object (`T=`) or a registry key (`case=`), exactly as the
package's own survey/bootstrap/oracle loops do; items 1, 5 and 6 run
in-process (five, ~20 and one fit respectively -- not worth a pool).

Reference artifacts (item 1's `theta_hat`/`eta_full`, item 2's stored
`ij` influence and `qij` `eta_full`) are read from the stored
`cloudfil_clean` run under `--run-root`, never regenerated: those
products are pre-existing measurements this script validates against,
not outputs of the code under test.
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
import pandas as pd

# The stored reference run this script measures item 1's continuations
# and item 2's r^2 check against (spec section 3): pre-existing
# products, read-only, never regenerated here.
RUN_ROOT = '/Users/jtaylor/Dropbox/Research/QIJ_joint/runs/cloudfil_clean/cloudfil_p2_N10000'
N_CLOUDFIL = 10000
M_SURVEY = 1094


# ======================================================================
# Stored-product readers (oracle theta_hat, qij eta_full, ij psi).
# ======================================================================

def _oracle_theta_hat(s: int, outputs) -> np.ndarray:
    df = pd.read_parquet(os.path.join(RUN_ROOT, 'oracle', f's{s:05d}.parquet'))
    return np.array([float(df[f'theta_hat_{o}'].iloc[0]) for o in outputs])


def _qij_eta_full(s: int) -> float:
    df = pd.read_parquet(os.path.join(RUN_ROOT, 'qij', f's{s:05d}.parquet'))
    return float(df['eta_full'].iloc[0])


def _ij_psi(s: int, outputs) -> np.ndarray:
    df = pd.read_parquet(os.path.join(RUN_ROOT, 'ij', f's{s:05d}.psi.parquet'))
    return np.column_stack([df[f'psi_{o}'].to_numpy() for o in outputs])


# ======================================================================
# Status: new code's `last_fit_info` when present, else 'ok'/'nan' from
# the returned theta's own finiteness (the old code has no such info).
# ======================================================================

def _status_of(theta: np.ndarray, T) -> tuple:
    info = getattr(T, 'last_fit_info', None)
    finite = bool(np.all(np.isfinite(np.asarray(theta, dtype=float))))
    if isinstance(info, dict) and 'status' in info:
        return str(info['status']), info
    return ('ok' if finite else 'nan'), info


def _info_field(info, key):
    if isinstance(info, dict) and info.get(key) is not None:
        try:
            return float(info[key])
        except (TypeError, ValueError):
            return info[key]
    return float('nan')


# ======================================================================
# Worker-side task functions for items 2-4's `Pool` (module level, so
# `loky`'s cloudpickle can ship them to the persistent pool by value).
# ======================================================================

def _resample_task(T, case, X, task):
    """One multinomial-weight continuation (item 3)."""
    b, counts, start, eta = task
    theta = np.asarray(T(X, counts, start=start, eta=eta), dtype=float)
    status, info = _status_of(theta, T)
    return b, theta, status, info


def _cold_task(T, case, X, task):
    """One cold fit on its own draw (item 4), mirroring
    `pipeline._oracle_task`'s pattern without calling `run_oracle`."""
    s, N, base_seed = task
    Xd = case.draw(N, base_seed + s)
    t0 = time.perf_counter()
    theta = np.asarray(T(Xd, np.ones(N)), dtype=float)
    status, info = _status_of(theta, T)
    return s, theta, status, info, time.perf_counter() - t0


# ======================================================================
# Item 1 -- the five diagnostic cases, continued from theta_hat.
# ======================================================================

def _build_case_defs(xvq_mod, registry):
    """The five point sets of `gmm_diag/REPORT_step1_6.md`, rebuilt from
    the package's own draw and quantizer (never read from that report's
    cache): A/B/C the survey's vor2/vor1 receptive-field centroids
    (weight = cell/pair count), D the moments-survey rows
    (`core.xvq._moments_survey_rows`), E the full data at unit weight.
    Each entry is (rows, weights, theta_hat, eta_full, label)."""
    fit_xvq = xvq_mod.fit_xvq
    moments_rows = xvq_mod._moments_survey_rows
    case = registry.case('cloudfil', 'p2')
    outputs = case.make_T().outputs

    def cell_centroids(labels, n_cells, X):
        n = np.bincount(labels, minlength=n_cells).astype(float)
        cent = np.stack([np.bincount(labels, weights=X[:, c], minlength=n_cells)
                          for c in range(X.shape[1])], axis=1) / n[:, None]
        return cent, n

    def draw_products(s):
        X = case.draw(N_CLOUDFIL, s)
        xvq = fit_xvq(X, M_SURVEY, s, 1)
        theta_hat = _oracle_theta_hat(s, outputs)
        eta_full = _qij_eta_full(s)
        key2 = xvq.bmu.astype(np.int64) * xvq.M_used + xvq.bmu2.astype(np.int64)
        _, lab2 = np.unique(key2, return_inverse=True)
        vor1_pts, vor1_w = cell_centroids(xvq.bmu, xvq.M_used, X)
        vor2_pts, vor2_w = cell_centroids(lab2, int(lab2.max()) + 1, X)
        mom_pts, _, mom_w = moments_rows(X, xvq.bmu, xvq.p, xvq.M_used)
        return dict(X=X, theta_hat=theta_hat, eta_full=eta_full,
                    vor1=(vor1_pts, vor1_w), vor2=(vor2_pts, vor2_w),
                    mom=(mom_pts, mom_w))

    d0 = draw_products(0)
    d1 = draw_products(1)
    return {
        'A_vor2_draw1': (*d1['vor2'], d1['theta_hat'], d1['eta_full'],
                         'draw 1, vor2 (BMU-pair) centroids'),
        'B_vor1_draw1': (*d1['vor1'], d1['theta_hat'], d1['eta_full'],
                         'draw 1, vor1 (BMU) centroids'),
        'C_vor2_draw0': (*d0['vor2'], d0['theta_hat'], d0['eta_full'],
                         'draw 0, vor2 centroids'),
        'D_moments_draw1': (*d1['mom'], d1['theta_hat'], d1['eta_full'],
                            'draw 1, moments-survey rows'),
        'E_full_draw1': (d1['X'], np.ones(N_CLOUDFIL), d1['theta_hat'], d1['eta_full'],
                         'draw 1, full data, unit weights'),
    }


def run_item1(mod, args):
    registry = mod['registry']
    case_defs = _build_case_defs(mod['xvq'], registry)
    case = registry.case('cloudfil', 'p2')
    rows = []
    for name, (Xc, wc, theta_hat, eta, label) in case_defs.items():
        T = case.make_T()
        t0 = time.perf_counter()
        theta = T(Xc, np.asarray(wc, dtype=float), start=theta_hat, eta=eta)
        wall = time.perf_counter() - t0
        status, info = _status_of(theta, T)
        rows.append(dict(
            case=name, label=label, n_rows=int(len(Xc)), eta=eta,
            status=status, em_iters=_info_field(info, 'em_iters'),
            newton_iters=_info_field(info, 'newton_iters'),
            score_scaled=_info_field(info, 'score_scaled'),
            wall_time=wall, theta=np.asarray(theta, dtype=float).tolist(),
        ))
    return rows


# ======================================================================
# Item 2 -- the survey's 1094 continuations on draws 0 and 1.
# ======================================================================

def _field_mean(values: np.ndarray, bmu: np.ndarray, M_used: int) -> np.ndarray:
    n = np.bincount(bmu, minlength=M_used).astype(float)
    out = np.empty((M_used, values.shape[1]))
    for c in range(values.shape[1]):
        out[:, c] = np.bincount(bmu, weights=values[:, c], minlength=M_used) / n
    return out


def _r_squared_per_output(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    q = a.shape[1]
    r2 = np.full(q, np.nan)
    for c in range(q):
        x, y = a[:, c], b[:, c]
        finite = np.isfinite(x) & np.isfinite(y)
        if finite.sum() < 2:
            continue
        xv, yv = x[finite], y[finite]
        if np.std(xv) == 0.0 or np.std(yv) == 0.0:
            continue
        r2[c] = float(np.corrcoef(xv, yv)[0, 1]) ** 2
    return r2


def run_item2(mod, args):
    datasets = mod['datasets']
    registry = mod['registry']
    xvq_mod = mod['xvq']
    Counter = mod['Counter']
    Pool = mod['Pool']
    case = registry.case('cloudfil', 'p2')
    rows = []
    for s in (0, 1):
        X = datasets.cloudfil_G_B6_P3_v1(N_CLOUDFIL, s)
        T = case.make_T()
        outputs = T.outputs
        theta_hat = _oracle_theta_hat(s, outputs)
        eta_full = _qij_eta_full(s)
        counter = Counter(T, N_CLOUDFIL)
        pool = Pool(args.workers, T=T)
        t0 = time.perf_counter()
        xvq = xvq_mod.fit_xvq(X, M_SURVEY, s, 14)
        theta_Q, I_proto, busy_delta, sv = xvq_mod.prototype_influences(
            xvq.centers, counter, xvq.p, T.eta, pool, survey='moments',
            X=X, bmu=xvq.bmu, quantized_start='full-data',
            theta_hat=theta_hat, eta_rows=eta_full)
        wall = time.perf_counter() - t0
        pool.close()

        n_failed = int(np.any(np.isnan(I_proto), axis=1).sum())
        psi_true = _ij_psi(s, outputs)
        field_mean = _field_mean(psi_true, xvq.bmu, xvq.M_used)
        r2 = _r_squared_per_output(I_proto, field_mean)
        rows.append(dict(
            draw=s, M_used=int(xvq.M_used), n_failed=n_failed,
            eta_Q=float(sv.eta_Q), r2_min=float(np.nanmin(r2)),
            r2_median=float(np.nanmedian(r2)), r2_max=float(np.nanmax(r2)),
            wall_time=wall, r2_by_output={o: float(v) for o, v in zip(outputs, r2)},
        ))
    return rows


# ======================================================================
# Item 3 -- 50 resampled-weight (multinomial) fits on draw 1.
# ======================================================================

def run_item3(mod, args):
    datasets = mod['datasets']
    registry = mod['registry']
    measure_eta_full = mod['measure_eta_full']
    Pool = mod['Pool']
    case = registry.case('cloudfil', 'p2')
    s = 1
    B = 50
    X = datasets.cloudfil_G_B6_P3_v1(N_CLOUDFIL, s)
    T = case.make_T()
    theta_hat = _oracle_theta_hat(s, T.outputs)
    eta_full, _ = measure_eta_full(T, X, theta_hat)

    rng = np.random.default_rng(0)  # fixed rng seed, per spec item 3
    probs = np.full(N_CLOUDFIL, 1.0 / N_CLOUDFIL)
    counts = np.empty((B, N_CLOUDFIL))
    for b in range(B):
        counts[b] = rng.multinomial(N_CLOUDFIL, probs)

    pool = Pool(args.workers, T=T)
    pool.share(X)
    tasks = [(b, counts[b], theta_hat, eta_full) for b in range(B)]
    t0 = time.perf_counter()
    results = pool.map(_resample_task, tasks)
    wall = time.perf_counter() - t0
    pool.close()

    statuses, n_degenerate = {}, 0
    for b, theta, status, info in results:
        statuses[status] = statuses.get(status, 0) + 1
        if not np.all(np.isfinite(theta)):
            n_degenerate += 1
    return [dict(draw=s, B=B, eta_full=eta_full, n_degenerate=n_degenerate,
                 status_breakdown=statuses, wall_time=wall)]


# ======================================================================
# Item 4 -- 20 cold fits, one per draw 0-19.
# ======================================================================

def run_item4(mod, args):
    Pool = mod['Pool']
    n_draws = 20
    pool = Pool(args.workers, case=('cloudfil', 'p2'))
    tasks = [(s, N_CLOUDFIL, 0) for s in range(n_draws)]
    t0 = time.perf_counter()
    results = pool.map(_cold_task, tasks)
    wall = time.perf_counter() - t0
    pool.close()

    statuses, n_failed = {}, 0
    for s, theta, status, info, draw_wall in results:
        statuses[status] = statuses.get(status, 0) + 1
        if not np.all(np.isfinite(theta)):
            n_failed += 1
    return [dict(n_draws=n_draws, n_failed=n_failed, status_breakdown=statuses,
                 wall_time=wall)]


# ======================================================================
# Item 5 -- finite-difference agreement of T.influence, draw 0 full data.
# ======================================================================

def run_item5(mod, args):
    datasets = mod['datasets']
    registry = mod['registry']
    case = registry.case('cloudfil', 'p2')
    s = 0
    X = datasets.cloudfil_G_B6_P3_v1(N_CLOUDFIL, s)
    T = case.make_T()
    theta_hat = _oracle_theta_hat(s, T.outputs)
    eta_full = _qij_eta_full(s)
    omega0 = np.ones(N_CLOUDFIL)
    W = float(omega0.sum())

    t0 = time.perf_counter()
    IF = T.influence(X, omega0, start=theta_hat, eta=eta_full)
    points = list(range(10))  # the first ten points; every point carries
    steps = (1.0, 0.1)        # the same mass under unit weights (A13's convention)
    records = []
    for i in points:
        pred = IF[i] / W
        for t in steps:
            w_plus = omega0.copy(); w_plus[i] += t
            w_minus = omega0.copy(); w_minus[i] -= t
            th_plus = T(X, w_plus, start=theta_hat, eta=eta_full)
            th_minus = T(X, w_minus, start=theta_hat, eta=eta_full)
            central = (th_plus - th_minus) / (2.0 * t)
            scale = np.maximum(np.abs(pred), 1e-12)
            rel = np.abs(central - pred) / scale
            records.append(dict(point=i, t=t, max_rel_err=float(np.nanmax(rel)),
                                 max_abs_err=float(np.nanmax(np.abs(central - pred)))))
    wall = time.perf_counter() - t0
    return [dict(draw=s, n_points=len(points), steps=list(steps),
                 max_rel_err=float(max(r['max_rel_err'] for r in records)),
                 records=records, wall_time=wall)]


# ======================================================================
# Item 6 -- duplicated-points weight test (gmm_diag/weights, rerun once).
# ======================================================================

def run_item6(mod, args):
    datasets = mod['datasets']
    registry = mod['registry']
    case = registry.case('cloudfil', 'p2')
    s = 1
    n_unique = 1500
    X = datasets.cloudfil_G_B6_P3_v1(N_CLOUDFIL, s)[:n_unique]
    T = case.make_T()
    theta_hat = _oracle_theta_hat(s, T.outputs)

    k = (1 + np.arange(n_unique) % 3).astype(float)  # k_i = 1 + (i mod 3)
    copy_of = np.repeat(np.arange(n_unique), k.astype(int))
    D1 = X[copy_of]                # 3000 rows, each point repeated k_i times
    w1 = np.ones(D1.shape[0])
    w2 = k

    def fit_with_fallback(rows, w):
        theta, eta_used = None, None
        for eta in (1e-6, 1e-5):
            theta = T(rows, w, start=theta_hat, eta=eta)
            eta_used = eta
            if np.all(np.isfinite(theta)):
                break
        return theta, eta_used

    t0 = time.perf_counter()
    theta1, eta1 = fit_with_fallback(D1, w1)
    theta2, eta2 = fit_with_fallback(X, w2)
    rel_theta = np.abs(theta1 - theta2) / np.maximum(np.abs(theta1), np.abs(theta2))
    max_rel_theta = float(np.nanmax(rel_theta))

    IF1 = T.influence(D1, w1, start=theta_hat, eta=eta1)
    IF2 = T.influence(X, w2, start=theta_hat, eta=eta2)

    max_within = 0.0
    first_copy = np.empty(n_unique, dtype=int)
    for i in range(n_unique):
        rows_i = np.where(copy_of == i)[0]
        first_copy[i] = rows_i[0]
        if rows_i.size > 1:
            max_within = max(max_within, float(np.max(np.abs(IF1[rows_i] - IF1[rows_i[0]]))))

    diff = IF2 - IF1[first_copy]
    max_abs_IF1 = float(np.max(np.abs(IF1))) or 1.0
    max_diff_over_max_abs = float(np.max(np.abs(diff)) / max_abs_IF1)
    scale = np.maximum(np.abs(IF1[first_copy]), np.abs(IF2))
    scale = np.where(scale > 0, scale, 1.0)
    max_rel_elemwise = float(np.max(np.abs(diff) / scale))
    wall = time.perf_counter() - t0

    return [dict(draw=s, n_unique=n_unique, n_dup=int(D1.shape[0]), eta1=eta1, eta2=eta2,
                 max_rel_theta=max_rel_theta, max_within_copy_diff=max_within,
                 max_diff_over_max_abs_IF=max_diff_over_max_abs,
                 max_rel_elemwise_IF=max_rel_elemwise, wall_time=wall)]


# ======================================================================
# Import bootstrap, CLI, and JSON/Markdown reporting.
# ======================================================================

def _load_package(src: str) -> dict:
    """Prepend `src` to `sys.path` and import the package under test --
    the same script then runs unchanged against the old and the new
    `_fit` by pointing `--src` at each tree's `src/` directory."""
    sys.path.insert(0, os.path.abspath(src))
    from qij_joint import datasets, registry
    from qij_joint.core import xvq as xvq_mod
    from qij_joint.core.counter import Counter
    from qij_joint.core.eta import measure_eta_full
    from qij_joint.parallel import Pool, call_T
    return dict(datasets=datasets, registry=registry, xvq=xvq_mod, Counter=Counter,
                measure_eta_full=measure_eta_full, Pool=Pool, call_T=call_T)


def _json_default(o):
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(str(type(o)))


_ITEM_TITLES = {
    'item1': '## Item 1 -- five diagnostic continuations from theta_hat',
    'item2': '## Item 2 -- 1094-prototype moments survey, draws 0 and 1',
    'item3': '## Item 3 -- 50 multinomial-weight fits, draw 1',
    'item4': '## Item 4 -- 20 cold fits, draws 0-19',
    'item5': '## Item 5 -- finite-difference influence check, draw 0',
    'item6': '## Item 6 -- duplicated-points weight test, draw 1',
}
_ITEM_SKIP_COLS = {'theta', 'records', 'r2_by_output', 'status_breakdown'}


def _write_markdown(results: dict, path: str, src: str) -> None:
    lines = ['# GMM fit validation -- spec section 3, items 1-6', '',
             f'Package under test: `{os.path.abspath(src)}`', '']
    for key in ('item1', 'item2', 'item3', 'item4', 'item5', 'item6'):
        if key not in results:
            continue
        entry = results[key]
        lines.append(_ITEM_TITLES[key])
        lines.append(f"(item wall time: {entry['item_wall_time']:.1f} s, "
                      f"run at {entry['timestamp']})")
        lines.append('')
        rows = entry['rows']
        if not rows:
            lines.append('_no rows_'); lines.append('')
            continue
        cols = [c for c in rows[0].keys() if c not in _ITEM_SKIP_COLS]
        lines.append('| ' + ' | '.join(cols) + ' |')
        lines.append('|' + '---|' * len(cols))
        for r in rows:
            lines.append('| ' + ' | '.join(str(r.get(c, '')) for c in cols) + ' |')
        for r in rows:
            if 'status_breakdown' in r:
                lines.append(f"- status breakdown: {r['status_breakdown']}")
        lines.append('')
    with open(path, 'w') as f:
        f.write('\n'.join(lines) + '\n')


_DISPATCH = {
    1: ('item1', run_item1), 2: ('item2', run_item2), 3: ('item3', run_item3),
    4: ('item4', run_item4), 5: ('item5', run_item5), 6: ('item6', run_item6),
}


def main(argv=None) -> None:
    global RUN_ROOT
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--src', required=True, help="package's src/ directory to import")
    ap.add_argument('--out', required=True, help='output directory')
    ap.add_argument('--items', default='1,2,3,4,5,6', help='comma list of item numbers')
    ap.add_argument('--workers', type=int, default=1, help='Pool size for items 2-4')
    ap.add_argument('--run-root', default=RUN_ROOT,
                     help='stored cloudfil_clean run to read theta_hat/eta_full/psi from')
    args = ap.parse_args(argv)
    RUN_ROOT = args.run_root

    mod = _load_package(args.src)
    items = [int(x) for x in args.items.split(',') if x.strip()]

    os.makedirs(args.out, exist_ok=True)
    json_path = os.path.join(args.out, 'validate_gmm_fit.json')
    results = {}
    if os.path.exists(json_path):
        with open(json_path) as f:
            results = json.load(f)

    for n in items:
        name, fn = _DISPATCH[n]
        t0 = time.perf_counter()
        rows = fn(mod, args)
        wall = time.perf_counter() - t0
        results[name] = dict(rows=rows, item_wall_time=wall,
                              src=os.path.abspath(args.src),
                              timestamp=datetime.now().isoformat(timespec='seconds'))
        with open(json_path, 'w') as f:
            json.dump(results, f, indent=2, default=_json_default)
        print(f'{name}: done in {wall:.1f}s -> {json_path}', file=sys.stderr)

    _write_markdown(results, os.path.join(args.out, 'validate_gmm_fit.md'), args.src)


if __name__ == '__main__':
    main()
