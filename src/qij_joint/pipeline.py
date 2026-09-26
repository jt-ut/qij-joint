"""The four methods' per-draw loops (Sections 2 and 4). `oracle` and `ij`
run their draws across a process pool; `boot` and `qij` loop over draws
in sequence, each parallel within a draw -- `boot` over replicate
chunks, `qij` over its prototype survey (method_notes section 2).
"""
from __future__ import annotations

import time
from typing import Iterable, Optional, Tuple

import numpy as np
import pandas as pd

from . import products, registry
from .bootstrap import Bootstrap
from .parallel import Pool
from .qij import QIJ


def _oracle_task(T, case, X, task):
    """One draw's full-data estimate: `theta_hat = T(X, ones)`; `X` is
    drawn fresh in this worker since draws are shared only by (N, seed),
    not by value, when the pool runs across draws."""
    s, N, base_seed = task
    seed = base_seed + s
    t0 = time.perf_counter()
    Xd = case.draw(N, seed)
    try:
        theta_hat = np.asarray(T(Xd, np.ones(N)), dtype=float)
        failed = False
    except Exception:
        theta_hat = np.full(len(T.outputs), np.nan)
        failed = True
    return s, seed, theta_hat, failed, time.perf_counter() - t0


def _ij_task(T, case, X, task):
    """One draw's analytic variance: `psi = T.influence(X, ones)`,
    `V_ij = mean(psi**2, axis=0) / N`, the infinitesimal-jackknife
    variance of theta_hat."""
    s, N, base_seed = task
    seed = base_seed + s
    t0 = time.perf_counter()
    Xd = case.draw(N, seed)
    try:
        psi = np.asarray(T.influence(Xd, np.ones(N)), dtype=float)
        failed = False
    except Exception:
        psi = np.full((N, len(T.outputs)), np.nan)
        failed = True
    V_ij = np.mean(psi ** 2, axis=0) / N
    return s, seed, V_ij, psi, failed, time.perf_counter() - t0


def run_oracle(dataset: str, estimator: str, N: int, draws: Iterable[int], seed: int,
               out_dir: str, workers: int, force: bool) -> Tuple[int, int]:
    """`oracle`: theta_true (the registry) and theta_hat = T(X, ones),
    parallel across draws."""
    draws = list(draws)
    md = products.method_dir(out_dir, dataset, estimator, N, 'oracle')
    theta_true = registry.truth(dataset, estimator)
    outputs = registry.case(dataset, estimator).make_T().outputs
    pending = [s for s in draws if force or not products.is_done(md, s)]
    if pending:
        pool = Pool(workers, case=(dataset, estimator))
        tasks = [(s, N, seed) for s in pending]
        for s, dseed, theta_hat, failed, wall in pool.map(_oracle_task, tasks):
            row = {'dataset': dataset, 'estimator': estimator, 'N': N,
                   's': s, 'seed': dseed, 'n_failed': int(failed),
                   'wall_time': wall, 'busy_time': wall, 'workers': workers}
            for j, o in enumerate(outputs):
                row[f'theta_true_{o}'] = float(theta_true[j])
                row[f'theta_hat_{o}'] = float(theta_hat[j])
            products.write_draw(md, s, row)
        pool.close()
    return len(pending), len(draws) - len(pending)


def run_ij(dataset: str, estimator: str, N: int, draws: Iterable[int], seed: int,
           out_dir: str, workers: int, force: bool) -> Tuple[int, int]:
    """`ij`: analytic influence and V_ij, parallel across draws; the
    `psi` array is stored for every draw written."""
    draws = list(draws)
    md = products.method_dir(out_dir, dataset, estimator, N, 'ij')
    outputs = registry.case(dataset, estimator).make_T().outputs
    pending = [s for s in draws if force or not products.is_done(md, s)]
    if pending:
        pool = Pool(workers, case=(dataset, estimator))
        tasks = [(s, N, seed) for s in pending]
        for s, dseed, V_ij, psi, failed, wall in pool.map(_ij_task, tasks):
            row = {'dataset': dataset, 'estimator': estimator, 'N': N,
                   's': s, 'seed': dseed, 'n_failed': int(failed),
                   'wall_time': wall, 'busy_time': wall, 'workers': workers}
            for j, o in enumerate(outputs):
                row[f'V_ij_{o}'] = float(V_ij[j])
            psi_df = pd.DataFrame({f'psi_{o}': psi[:, j] for j, o in enumerate(outputs)})
            products.write_draw(md, s, row, {'psi': psi_df})
        pool.close()
    return len(pending), len(draws) - len(pending)


def run_boot(dataset: str, estimator: str, N: int, draws: Iterable[int], seed: int,
             out_dir: str, workers: int, B: int, force: bool) -> Tuple[int, int]:
    """`boot`: a sequential draw loop, each draw's B replicates spread
    across one persistent pool for the whole run."""
    draws = list(draws)
    md = products.method_dir(out_dir, dataset, estimator, N, 'boot')
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
               's': s, 'seed': dseed, 'n_failed': res.n_failed,
               'wall_time': res.wall_time, 'busy_time': res.busy_time,
               'workers': res.workers}
        rep_df = pd.DataFrame({f'theta_{o}': res.replicates[:, j]
                                for j, o in enumerate(res.outputs)})
        products.write_draw(md, s, row, {'replicates': rep_df})
        written += 1
    pool.close()
    return written, len(draws) - written


def _qij_row(dataset: str, estimator: str, N: int, s: int, seed: int, res) -> dict:
    """One `qij` scalar row: `QIJResult`'s scalars flattened to columns
    per stage and per output."""
    row = {'dataset': dataset, 'estimator': estimator, 'N': N,
           's': s, 'seed': seed, 'gptrend': res.gptrend, 'gpwidth': res.gpwidth,
           'M_X': int(res.M_X), 'M_X_source': res.M_X_source, 'n_failed': int(res.n_failed)}
    for stage in ('prototype', 'full_data', 'refinement', 'total'):
        row[f'evals_{stage}'] = int(res.evals_by_stage[stage])
        row[f'rows_{stage}'] = int(res.rows_by_stage[stage])
        row[f'wall_time_{stage}'] = float(res.wall_time_by_stage[stage])
    row['normalized_rows'] = float(res.rows_by_stage['total']) / res.N
    row['busy_time_total'] = float(res.busy_time_total)
    row['workers'] = int(res.workers)
    for j, o in enumerate(res.outputs):
        row[f'V_btw_{o}'] = float(res.V_btw[j])
        row[f'V_win_hat_{o}'] = float(res.V_win_hat[j])
        row[f'V_tot_hat_{o}'] = float(res.V_tot_hat[j])
        row[f'L_{o}'] = int(res.L[j])
        row[f'n_level_splits_{o}'] = int(res.n_level_splits[j])
        row[f'n_adjacency_splits_{o}'] = int(res.n_adjacency_splits[j])
        row[f'rho_{o}'] = float(res.rho[j])
        row[f'gain_ratio_{o}'] = float(res.gain_ratio[j])
        row[f'n_refine_evals_{o}'] = int(res.n_refine_evals[j])
        row[f'ell_{o}'] = float(res.ell[j])
        row[f'lam_{o}'] = float(res.lam[j])
        row[f'ell_bound_{o}'] = bool(res.ell_bound[j])
        row[f'lam_bound_{o}'] = bool(res.lam_bound[j])
        row[f'c_{o}'] = float(res.c[j])
        row[f'c_bound_{o}'] = bool(res.c_bound[j])
    return row


def _qij_points(res) -> pd.DataFrame:
    """`points` for `--diag-draws`: `i, bmu, psi0_<o>, sigma_<o>,
    psi_hat_<o>, bin_label_<o>` (Section 6.2)."""
    N = res.psi0.shape[0]
    data = {'i': np.arange(N), 'bmu': np.asarray(res.bmu, dtype=np.int32)}
    for j, o in enumerate(res.outputs):
        data[f'psi0_{o}'] = res.psi0[:, j]
        data[f'sigma_{o}'] = res.sigma[:, j]
        data[f'psi_hat_{o}'] = res.psi_hat[:, j]
        data[f'bin_label_{o}'] = np.asarray(res.bin_label[:, j], dtype=np.int32)
    return pd.DataFrame(data)


def _qij_prototypes(res) -> pd.DataFrame:
    """`prototypes` for `--diag-draws`: `j, p, w_<d>, I_<o>, h`
    (method_notes section 3)."""
    M = res.prototype_p.shape[0]
    W = np.atleast_2d(np.asarray(res.prototype_w, dtype=float).reshape(M, -1))
    data = {'j': np.arange(M), 'p': res.prototype_p}
    for d in range(W.shape[1]):
        data[f'w_{d}'] = W[:, d]
    for j, o in enumerate(res.outputs):
        data[f'I_{o}'] = res.prototype_I[:, j]
    data['h'] = res.prototype_h
    return pd.DataFrame(data)


def run_qij(dataset: str, estimator: str, N: int, draws: Iterable[int], seed: int,
            out_dir: str, eps: float, diag_draws: Optional[Iterable[int]], force: bool,
            workers: int = 1, gptrend: str = 'affine', gpwidth: str = 'global',
            M_X: Optional[int] = None) -> Tuple[int, int]:
    """`qij`: a sequential draw loop; with `workers > 1` one pool is
    created for the run and passed to every draw's fit, so only the
    prototype survey (method_notes section 2) runs in parallel -- the
    rest of a draw is serial regardless of `workers`."""
    draws = list(draws)
    md = products.method_dir(out_dir, dataset, estimator, N, 'qij')
    diag = set(diag_draws) if diag_draws is not None else set()
    case = registry.case(dataset, estimator)
    T = case.make_T()
    pool = Pool(workers, T=T) if workers > 1 else None
    written = 0
    for s in draws:
        if not force and products.is_done(md, s):
            continue
        dseed = seed + s
        X = case.draw(N, dseed)
        res = QIJ(eps=eps, seed=dseed, vq_transform=case.vq_transform,
                  gptrend=gptrend, gpwidth=gpwidth, M_X=M_X).fit(X, T, pool=pool)
        row = _qij_row(dataset, estimator, N, s, dseed, res)
        arrays = None
        if s in diag:
            arrays = {'points': _qij_points(res), 'prototypes': _qij_prototypes(res)}
        products.write_draw(md, s, row, arrays)
        written += 1
    if pool is not None:
        pool.close()
    return written, len(draws) - written
