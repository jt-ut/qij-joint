"""The four methods' per-draw loops. `oracle` and `ij` run their draws
across a process pool; `boot` and `qij` loop over draws in sequence,
each parallel within a draw -- `boot` over replicate chunks, `qij` over
its evaluations (spec/METHOD_qij.md).
"""
from __future__ import annotations

import time
from typing import Iterable, Optional, Tuple

import numpy as np
import pandas as pd

from . import products, registry
from .bootstrap import Bootstrap
from .audit import search_audit
from .parallel import Pool, fit_status


def _oracle_task(T, case, X, task):
    """One draw's full-data estimate: `theta_hat = T(X, ones)`; `X` is
    drawn fresh in this worker since draws are shared only by (N, seed),
    not by value, when the pool runs across draws. `failed` is also set
    on a NaN return that raised no exception (a gate failing quietly),
    not only on an exception."""
    s, N, base_seed = task
    seed = base_seed + s
    t0 = time.perf_counter()
    Xd = case.draw(N, seed)
    try:
        theta_hat = np.asarray(T(Xd, np.ones(N)), dtype=float)
        failed = bool(np.any(np.isnan(theta_hat)))
        status = fit_status(T, theta_hat)
    except Exception:
        theta_hat = np.full(len(T.outputs), np.nan)
        failed = True
        status = fit_status(T, theta_hat, raised=True)
    wall = time.perf_counter() - t0
    info = getattr(T, 'last_fit_info', None)
    search = dict(getattr(info, 'search', None) or {})
    audit = search_audit(T, Xd, theta_hat, case.dataset, case.estimator)
    return s, seed, theta_hat, failed, status, wall, {**search, **audit}


def _ij_task(T, case, X, task):
    """One draw's analytic variance: `psi = T.influence(X, ones)`,
    `V_ij = mean(psi**2, axis=0) / N`, the infinitesimal-jackknife
    variance of theta_hat. `failed` is also set on a NaN return that
    raised no exception (a gate failing quietly), not only on an
    exception."""
    s, N, base_seed = task
    seed = base_seed + s
    t0 = time.perf_counter()
    Xd = case.draw(N, seed)
    try:
        psi = np.asarray(T.influence(Xd, np.ones(N)), dtype=float)
        failed = bool(np.any(np.isnan(psi)))
        status = fit_status(T, psi)
    except Exception:
        psi = np.full((N, len(T.outputs)), np.nan)
        failed = True
        status = fit_status(T, psi, raised=True)
    V_ij = np.mean(psi ** 2, axis=0) / N
    return s, seed, V_ij, psi, failed, status, time.perf_counter() - t0


def run_oracle(dataset: str, estimator: str, N: int, draws: Iterable[int], seed: int,
               out_dir: str, workers: int, force: bool, tag: str = '') -> Tuple[int, int]:
    """`oracle`: theta_true (the registry) and theta_hat = T(X, ones),
    parallel across draws."""
    draws = list(draws)
    md = products.method_dir(out_dir, dataset, estimator, N, 'oracle', tag)
    theta_true = registry.truth(dataset, estimator)
    outputs = registry.case(dataset, estimator).make_T().outputs
    pending = [s for s in draws if force or not products.is_done(md, s)]
    if pending:
        pool = Pool(workers, case=(dataset, estimator))
        tasks = [(s, N, seed) for s in pending]
        for s, dseed, theta_hat, failed, status, wall, audit in pool.map(_oracle_task, tasks):
            row = {'dataset': dataset, 'estimator': estimator, 'N': N,
                   's': s, 'seed': dseed, 'n_failed': int(failed), 'fit_status': status,
                   'wall_time': wall, 'busy_time': wall, 'workers': workers, **audit}
            for j, o in enumerate(outputs):
                row[f'theta_true_{o}'] = float(theta_true[j])
                row[f'theta_hat_{o}'] = float(theta_hat[j])
            products.write_draw(md, s, row)
        pool.close()
    return len(pending), len(draws) - len(pending)


def run_ij(dataset: str, estimator: str, N: int, draws: Iterable[int], seed: int,
           out_dir: str, workers: int, force: bool, tag: str = '') -> Tuple[int, int]:
    """`ij`: analytic influence and V_ij, parallel across draws; the
    `psi` array is stored for every draw written."""
    draws = list(draws)
    md = products.method_dir(out_dir, dataset, estimator, N, 'ij', tag)
    outputs = registry.case(dataset, estimator).make_T().outputs
    pending = [s for s in draws if force or not products.is_done(md, s)]
    if pending:
        pool = Pool(workers, case=(dataset, estimator))
        tasks = [(s, N, seed) for s in pending]
        for s, dseed, V_ij, psi, failed, status, wall in pool.map(_ij_task, tasks):
            row = {'dataset': dataset, 'estimator': estimator, 'N': N,
                   's': s, 'seed': dseed, 'n_failed': int(failed), 'fit_status': status,
                   'wall_time': wall, 'busy_time': wall, 'workers': workers}
            for j, o in enumerate(outputs):
                row[f'V_ij_{o}'] = float(V_ij[j])
            psi_df = pd.DataFrame({f'psi_{o}': psi[:, j] for j, o in enumerate(outputs)})
            products.write_draw(md, s, row, {'psi': psi_df})
        pool.close()
    return len(pending), len(draws) - len(pending)


def run_boot(dataset: str, estimator: str, N: int, draws: Iterable[int], seed: int,
             out_dir: str, workers: int, B: int, force: bool, tag: str = '') -> Tuple[int, int]:
    """`boot`: a sequential draw loop, each draw's B replicates spread
    across one persistent pool for the whole run."""
    draws = list(draws)
    md = products.method_dir(out_dir, dataset, estimator, N, 'boot', tag)
    case = registry.case(dataset, estimator)
    T = case.make_T()
    pool = Pool(workers, T=T)
    written = 0
    for s in draws:
        if not force and products.is_done(md, s):
            continue
        dseed = seed + s
        X = case.draw(N, dseed)
        res = Bootstrap(B=B, seed=dseed).fit(X, T, pool=pool)
        row = {'dataset': dataset, 'estimator': estimator, 'N': N,
               's': s, 'seed': dseed, 'n_degenerate': res.n_degenerate,
               'wall_time': res.wall_time, 'busy_time': res.busy_time,
               'workers': res.workers, 'eta_full': float(res.eta_full),
               'fit_status': res.theta_hat_status,
               **search_audit(T, X, res.theta_hat, dataset, estimator)}
        rep_df = pd.DataFrame({f'theta_{o}': res.replicates[:, j]
                                for j, o in enumerate(res.outputs)})
        rep_df['fit_status'] = res.replicate_status
        products.write_draw(md, s, row, {'replicates': rep_df})
        written += 1
    pool.close()
    return written, len(draws) - written


_TIMING_KEYS = ('pilot_cells_s', 'pilot_eval_s', 'fit_s', 'eval_s', 'total_s')


def _per_output(d: dict, outputs) -> dict:
    """One table row with every per-output value (an array of length J)
    spread into `<key>_<output>` columns; scalars pass through."""
    J = len(outputs)
    row = {}
    for key, value in d.items():
        a = np.asarray(value)
        if a.ndim == 1 and a.shape[0] == J and a.dtype != object:
            for j, o in enumerate(outputs):
                row[f'{key}_{o}'] = a[j].item()
        else:
            row[key] = value
    return row


def _qij_row(dataset: str, estimator: str, N: int, s: int, seed: int, eps: float,
             level: float, res, wall: float, workers: int) -> dict:
    """One `qij` scalar row: the run's settings, the method's counts
    (M, K, rounds, E, E_full), per output V-hat, U, delta, V_lb and, when
    the method returns them, the interval ends; the draw's wall time and
    the method's own timings."""
    row = {'dataset': dataset, 'estimator': estimator, 'N': N, 's': s, 'seed': seed,
           'eps': float(eps), 'level': float(level), 'M': int(res.M), 'K': int(res.K),
           'rounds': int(res.rounds), 'E': int(res.E), 'E_full': int(res.E_full),
           'wall_s': wall, 'workers': workers}
    for name in ('Vhat', 'U', 'delta', 'V_lb'):
        values = np.asarray(getattr(res, name), dtype=float)
        for j, o in enumerate(res.outputs):
            row[f'{name}_{o}'] = float(values[j])
    if res.intervals is not None:
        iv = np.asarray(res.intervals, dtype=float).reshape(len(res.outputs), 2)
        for j, o in enumerate(res.outputs):
            row[f'lo_{o}'] = float(iv[j, 0])
            row[f'hi_{o}'] = float(iv[j, 1])
    for key in _TIMING_KEYS:
        row[key] = float(res.timings.get(key, float('nan')))
    # the offset, the inferred and repaired leaves, and the measured evaluation error (qij_joint.method)
    row.update(n_inferred=int(res.n_inferred), n_repaired=int(res.n_repaired), eta_full=float(res.eta_full or 0.0))
    om = np.asarray(res.offset, dtype=float); be = np.asarray(res.bound_eval, dtype=float)
    ee = res.eval_error or {}
    for j, o in enumerate(res.outputs):
        row[f'offset_{o}'] = float(om[j]); row[f'bound_eval_{o}'] = float(be[j])
        if 'max' in ee: row[f'eval_error_max_{o}'] = float(np.asarray(ee['max'])[j])
    row['n_redundant'] = int(ee.get('n', 0))
    return row


def run_qij(dataset: str, estimator: str, N: int, draws: Iterable[int], seed: int,
            out_dir: str, eps: float = 0.01, level: float = 0.84, M: Optional[int] = None,
            workers: Optional[int] = None, tag: Optional[str] = None,
            force: bool = False) -> Tuple[int, int]:
    """`qij`: a sequential draw loop through the ported method
    (`qij_joint.method.run`, spec/METHOD_qij.md); one persistent pool for
    the run, built as `run_boot` builds its own (`workers=None` -> 1, in
    process), passed to every draw. Per draw: the scalar row, the
    `rounds` table and the `leaves` table. A draw whose row exists is
    skipped unless `force`."""
    # Imported here, not at module level, so the package imports before
    # (and independently of) the method module.
    from .method import run as run_method
    draws = list(draws)
    workers = 1 if workers is None else int(workers)
    md = products.method_dir(out_dir, dataset, estimator, N, 'qij', tag or '')
    case = registry.case(dataset, estimator)
    T = case.make_T()
    pool = Pool(workers, T=T)
    written = 0
    for s in draws:
        if not force and products.is_done(md, s):
            continue
        dseed = seed + s
        X = case.draw(N, dseed)
        t0 = time.perf_counter()
        res = run_method(T, X, eps=eps, level=level, M=M, pool=pool, seed=dseed, vq_transform=case.vq_transform)
        wall = time.perf_counter() - t0
        row = _qij_row(dataset, estimator, N, s, dseed, eps, level, res, wall, workers)
        arrays = {'rounds': pd.DataFrame([_per_output(r, res.outputs) for r in res.rounds_table]),
                  'leaves': pd.DataFrame([_per_output(r, res.outputs) for r in res.leaves_table])}
        products.write_draw(md, s, row, arrays)
        written += 1
    pool.close()
    return written, len(draws) - written
