#!/usr/bin/env python3.9
"""
A PRE-SCREEN FOR THE STUDY ONLY. NEVER PART OF THE METHOD.

Runs stage 1 for real (the package's own first-stage quantizer, prototype
survey with the real estimator on a pool, and influence-model fit -- exactly
`qij.py`'s own code path, called through the package's own functions), then
replaces every full-data evaluation of stage 2 (the marginal path's initial
bins and refinement, under either `refine_update`) by a linear counter
T(omega) = theta_hat + (omega - 1)^T Psi / N, Psi the stored analytic
influence and theta_hat the stored oracle fit. This is spec/QIJ_A17_stencil_
update.md section 7 step 2 / section 8 item 6's own pre-screen: a ONE-SIDED
check (a fail stops the rehearsal; a pass permits it and proves nothing
more, since the harness runs on the analytic influence with a linear T,
which is not the method and not available to a user).

    python3.9 scripts/stage2_simulator.py --runs <dir> --dataset cloudfil \
        --estimator p2 --N 10000 --draws 0:2 --workers 14 \
        --survey moments --quantized-start full-data --gptrend quadratic \
        --gpwidth global --refine-schedule rounds --refine-update none \
        --eps 0.01 --out <file.csv>

Required stored products, under `--runs`, for the SAME (dataset, estimator,
N) at the given draws: `oracle/s{draw:05d}.parquet` (the `theta_hat_<o>`
columns) and `ij/s{draw:05d}.psi.parquet` (the `psi_<o>` columns, N x 66)
-- both already stored by `scripts/run.py <dataset> <estimator> oracle`/`ij`.
The curvature stage (and anything after the refinement that would otherwise
spend evaluations) is SKIPPED: none of the reported columns need it, and
running it for real would mean a real, expensive full-data-scale T call on
the survey rows -- exactly the cost this harness exists to avoid; using the
linear counter for it instead would need extending that counter to survey
rows, out of scope for what was asked.
"""
import os

for _var in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
             'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(_var, '1')

import argparse
import csv
import sys
import time

import numpy as np
import pandas as pd

from qij_joint import products, registry
from qij_joint.core.counter import Counter
from qij_joint.core.influence_model import (
    rows_kernel_columns, fit_influence_model, fit_rows_model, psi0 as _psi0,
    rows_point_terms, uncertainty as _uncertainty,
)
from qij_joint.core.refine import run_refinement
from qij_joint.core.rounds import run_refinement_rounds
from qij_joint.core.stencils import run_refinement_stencils
from qij_joint.core.xvq import cost_rule_M, run_xvq
from qij_joint.parallel import Pool


class LinearCounter:
    """
    The stage-2 simulator's own T(omega) = theta_hat + (omega - 1)^T Psi
    / N: an exact linearization of the estimator about omega = 1 (spec
    section 7 step 2), replacing every full-data evaluation the initial
    bins and the refinement would otherwise make. Counts evaluations and
    rows exactly as `core.counter.Counter` does, so every audited count
    in the returned `QIJResult`-shaped products is meaningful; `X` is
    accepted (for interface compatibility with `core.refine`/`core.
    rounds`/`core.stencils`, which always call `counter(X, omega, ...)`)
    but never read -- the whole point of this counter is that it needs
    no data.
    """

    def __init__(self, theta_hat: np.ndarray, Psi: np.ndarray, outputs, name: str, eta: float):
        self.theta_hat = np.asarray(theta_hat, dtype=float)
        self.Psi = np.asarray(Psi, dtype=float)
        self.N = Psi.shape[0]
        self.outputs = outputs
        self.name = name
        self.eta = eta
        self.takes_start = False
        self.evaluations = 0
        self.rows = 0
        self.failed = 0

    def __call__(self, X, w, start=None, eta=None):
        w = np.asarray(w, dtype=float)
        result = self.theta_hat + (w - 1.0) @ self.Psi / self.N
        self.evaluations += 1
        self.rows += self.N
        if np.any(np.isnan(result)):
            self.failed += 1
        return result

    def snapshot(self):
        return (self.evaluations, self.rows)

    def add(self, evaluations: int, rows: int, failed: int) -> None:
        self.evaluations += evaluations
        self.rows += rows
        self.failed += failed


def _load_theta_hat_and_psi(runs: str, dataset: str, estimator: str, N: int, draw: int, outputs):
    """theta_hat (from `oracle`'s own `theta_hat_<o>` columns) and Psi
    (N, len(outputs)) from `ij`'s own `psi_<o>` array, both stored by
    `scripts/run.py <dataset> <estimator> {oracle,ij}` -- the exact
    products the spec names (`oracle/s{draw:05d}.parquet`, `ij/
    s{draw:05d}.psi.parquet`)."""
    oracle_md = products.method_dir(runs, dataset, estimator, N, 'oracle')
    oracle_row = pd.read_parquet(os.path.join(oracle_md, f's{draw:05d}.parquet'))
    theta_hat = np.array([float(oracle_row[f'theta_hat_{o}'].iloc[0]) for o in outputs])

    ij_md = products.method_dir(runs, dataset, estimator, N, 'ij')
    psi_df = pd.read_parquet(os.path.join(ij_md, f's{draw:05d}.psi.parquet'))
    Psi = np.stack([psi_df[f'psi_{o}'].to_numpy(dtype=float) for o in outputs], axis=1)
    return theta_hat, Psi


def _load_eta_full(runs: str, dataset: str, estimator: str, N: int, draw: int):
    """eta_full from a stored `qij` run's own scalar row, when one is
    present at this (dataset, estimator, N, draw) -- the value the
    validation-mode reference run itself used, read back rather than
    re-measured (`core.eta.measure_eta_full` is itself a real,
    full-data-evaluation-based procedure -- exactly the cost this
    harness exists to avoid). None if no `qij` product is stored here."""
    qij_md = products.method_dir(runs, dataset, estimator, N, 'qij')
    path = os.path.join(qij_md, f's{draw:05d}.parquet')
    if not os.path.exists(path):
        return None
    row = pd.read_parquet(path)
    if 'eta_full' not in row.columns:
        return None
    return float(row['eta_full'].iloc[0])


def run_one_draw(args, s: int, oracle_variance_pa=None):
    case = registry.case(args.dataset, args.estimator)
    T = case.make_T()
    if not hasattr(T, 'outputs'):
        T.outputs = ('theta',)
        T.eta = float(np.finfo(float).eps)
        T.name = getattr(T, '__name__', 'theta')
    outputs_full = T.outputs
    measured = list(getattr(T, 'measured', range(len(outputs_full))))
    outputs = tuple(outputs_full[i] for i in measured)
    q = len(measured)

    dseed = args.seed + s
    X = case.draw(args.N, dseed)
    N = len(X)

    theta_hat_full, Psi_full = _load_theta_hat_and_psi(
        args.runs, args.dataset, args.estimator, args.N, s, outputs_full)
    eta_full = _load_eta_full(args.runs, args.dataset, args.estimator, args.N, s)
    if eta_full is None:
        eta_full = T.eta

    real_counter = Counter(T, N)
    pool = Pool(args.workers, T=T) if args.workers > 1 else None
    if pool is not None:
        pool.share(X)

    Z, inverse = case.vq_transform(X) if case.vq_transform is not None else (X, lambda A: A)
    Z = np.asarray(Z, dtype=float)
    if Z.ndim == 1:
        Z = Z.reshape(-1, 1)

    M_requested = cost_rule_M(N, q, args.eps)

    t0 = time.perf_counter()
    xvq, theta_Q, I_proto, xvq_busy, sv = run_xvq(
        Z, inverse, real_counter, eta_full, M_requested, dseed, pool, args.workers,
        survey=args.survey, X=X, quantized_start=args.quantized_start,
        theta_hat=theta_hat_full,
    )
    rows_stage1 = args.refine_update == 'stencils' and args.ivqbins == 'marginal'
    if rows_stage1:
        stage1_per_coord, stage1_shared, stage1_constant, stage1_aux = fit_rows_model(
            Z, xvq, I_proto[:, measured], theta_Q[measured], eta_full,
            gptrend=args.gptrend, gpwidth=args.gpwidth)
        mean_w, transform_w = stage1_aux['whitening']
        Za = np.asarray(Z, dtype=float)
        Zw = (Za - mean_w) @ transform_w.T
        psi0_all = np.empty((N, q))
        sigma_all = np.empty((N, q))
        for gi, sh in stage1_shared.items():
            cols_g = sh['cols']
            Kx_g = rows_kernel_columns(Zw, sh['all_idx'], args.gpwidth, sh['param'],
                                       stage1_aux['h_full'], stage1_aux['bmu_full'], stage1_aux['d_z'])
            psi0_g, sigma_g, _R = rows_point_terms(stage1_per_coord, cols_g, Kx_g, Zw, sh['m'])
            for c in cols_g:
                psi0_all[:, c] = psi0_g[c]
                sigma_all[:, c] = sigma_g[c]
        constant_paths = [False] * q
        for c, (is_const, const_val) in stage1_constant.items():
            constant_paths[c] = is_const
            if is_const:
                psi0_all[:, c] = const_val
                sigma_all[:, c] = 0.0
        offset = list(psi0_all.mean(axis=0))
        model = None
    else:
        model, model_busy = fit_influence_model(
            Z, xvq, I_proto[:, measured], theta_Q[measured], eta_full,
            gptrend=args.gptrend, gpwidth=args.gpwidth, pool=pool)
        psi0_all = _psi0(model, Z)
        sigma_all = _uncertainty(model, Z)
        constant_paths = [bool(model.constant_path[j]) for j in range(q)]
        offset = [float(model.offset[j]) for j in range(q)]
    wall_stage1 = time.perf_counter() - t0

    lin_counter = LinearCounter(theta_hat_full, Psi_full, outputs_full, T.name, eta_full)

    t0 = time.perf_counter()
    if args.refine_update == 'stencils':
        coordinates, n_rounds, n_update_rounds, update_wall_time, stencil_products = run_refinement_stencils(
            X, lin_counter, theta_hat_full, list(measured), list(outputs),
            [psi0_all[:, j] for j in range(q)], [float(offset[j]) for j in range(q)],
            [sigma_all[:, j] for j in range(q)], [I_proto[:, c] for c in measured],
            xvq.bmu, xvq.bmu2, eta_full, args.eps, xvq.M_used, constant_paths, Z,
            list(range(q)), xvq, theta_Q[measured], args.gptrend, args.gpwidth,
            stage1_per_coord, stage1_aux, pool=None, start=None,
        )
        lambda_c_stage1 = {j: float(stage1_per_coord[j]['lam']) for j in range(q)}
        lambda_c_update = stencil_products['lambda_c_update']
        ridge_step_max = stencil_products['ridge_step_max']
    elif args.refine_schedule == 'rounds':
        coordinates, n_rounds = run_refinement_rounds(
            X, lin_counter, theta_hat_full, list(measured), list(outputs),
            [psi0_all[:, j] for j in range(q)], [float(offset[j]) for j in range(q)],
            [sigma_all[:, j] for j in range(q)], [I_proto[:, c] for c in measured],
            xvq.bmu, xvq.bmu2, eta_full, args.eps, xvq.M_used, constant_paths, Z,
            model, list(range(q)), pool=None, start=None,
        )
        n_update_rounds, update_wall_time = 0, 0.0
        lambda_c_stage1 = {j: float(model.lam[j]) for j in range(q)}
        lambda_c_update = {j: float('nan') for j in range(q)}
        ridge_step_max = 0
    else:
        coordinates = [
            run_refinement(
                X, lin_counter, theta_hat_full, c, name,
                psi0_all[:, j], float(offset[j]), sigma_all[:, j],
                I_proto[:, c], xvq.bmu, xvq.bmu2, eta_full, args.eps, xvq.M_used,
                constant_paths[j], Z, model, pool=None, start=None, model_index=j,
            )
            for j, (c, name) in enumerate(zip(measured, outputs))
        ]
        n_rounds = 0
        n_update_rounds, update_wall_time = 0, 0.0
        lambda_c_stage1 = {j: float(model.lam[j]) for j in range(q)}
        lambda_c_update = {j: float('nan') for j in range(q)}
        ridge_step_max = 0
    wall_stage2 = time.perf_counter() - t0

    if pool is not None:
        pool.close()

    Psi_centered = Psi_full[:, measured] - Psi_full[:, measured].mean(axis=0)
    V_ij = np.mean(Psi_centered ** 2, axis=0) / N

    rows_out = []
    guard = 1 + xvq.M_used
    for j, (c, name) in enumerate(zip(measured, outputs)):
        cr = coordinates[j]
        ratio = cr.V_btw / V_ij[j] if V_ij[j] > 0 else float('nan')
        at_guard = cr.n_refine_evals >= guard
        passed = (ratio >= 0.97) and not at_guard
        row = dict(
            draw=s, output=name, V_btw=cr.V_btw, V_ij=V_ij[j], ratio=ratio,
            L=cr.L, evals=cr.n_refine_evals, guard=guard, at_guard=at_guard,
            rounds=n_rounds, n_update_rounds=n_update_rounds, update_wall_time=update_wall_time,
            a_c=cr.a_c, lambda_c_stage1=lambda_c_stage1[j], lambda_c_update=lambda_c_update[j],
            ridge_step_max=ridge_step_max, PASS=passed,
            wall_stage1=wall_stage1, wall_stage2=wall_stage2,
        )
        if name == 'p2_pa' and oracle_variance_pa is not None:
            row['V_ij_over_oracle_var_pa'] = V_ij[j] / oracle_variance_pa
        rows_out.append(row)
    return rows_out


def _find_oracle_variance(runs, dataset, estimator, N, draws, output_name):
    """Look for a stored per-draw oracle SAMPLING variance for
    `output_name` (spec section 10.4's own ratio): checked in the
    `oracle` product's own scalar row and in `ij`'s (neither stores one
    as built -- `oracle` stores one point estimate per draw, `ij` its
    own analytic V_ij); returns None, reporting the absence, if not
    found rather than inventing one."""
    for method in ('oracle', 'ij'):
        md = products.method_dir(runs, dataset, estimator, N, method)
        for s in draws:
            path = os.path.join(md, f's{s:05d}.parquet')
            if not os.path.exists(path):
                continue
            row = pd.read_parquet(path)
            for col in row.columns:
                if 'oracle' in col.lower() and 'var' in col.lower() and output_name in col:
                    return float(row[col].iloc[0])
    return None


def main(argv=None):
    import logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s', stream=sys.stderr)
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--runs', required=True, help='directory holding the stored oracle/ij products')
    p.add_argument('--dataset', required=True)
    p.add_argument('--estimator', required=True)
    p.add_argument('--N', type=int, required=True)
    p.add_argument('--draws', required=True, help='a:b, e.g. 0:2')
    p.add_argument('--workers', type=int, default=1)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--eps', type=float, default=0.01)
    p.add_argument('--gptrend', choices=['affine', 'quadratic'], default='affine')
    p.add_argument('--gpwidth', choices=['global', 'local'], default='global')
    p.add_argument('--survey', choices=['points', 'moments'], default='points')
    p.add_argument('--quantized-start', dest='quantized_start',
                    choices=['multistart', 'full-data'], default='multistart')
    p.add_argument('--refine-schedule', dest='refine_schedule', choices=['queue', 'rounds'],
                    default='queue')
    p.add_argument('--refine-update', dest='refine_update', choices=['none', 'stencils'],
                    default='none')
    p.add_argument('--ivqbins', choices=['marginal', 'joint'], default='marginal')
    p.add_argument('--out', required=True, help='CSV path for the table')
    args = p.parse_args(argv)

    a, b = args.draws.split(':')
    draws = list(range(int(a), int(b)))

    oracle_var_pa = _find_oracle_variance(args.runs, args.dataset, args.estimator, args.N,
                                           draws, 'p2_pa')
    if oracle_var_pa is None:
        print("No stored oracle sampling variance for p2_pa found in oracle/ij products; "
              "skipping the V_ij/oracle-variance ratio for the angle (spec section 10.4).")

    all_rows = []
    for s in draws:
        t0 = time.perf_counter()
        rows = run_one_draw(args, s, oracle_variance_pa=oracle_var_pa)
        print(f'draw {s}: {time.perf_counter() - t0:.1f}s wall')
        all_rows.extend(rows)

    cols = ['draw', 'output', 'V_btw', 'V_ij', 'ratio', 'L', 'evals', 'guard', 'at_guard',
            'rounds', 'n_update_rounds', 'update_wall_time', 'a_c', 'lambda_c_stage1',
            'lambda_c_update', 'ridge_step_max', 'PASS', 'wall_stage1', 'wall_stage2']
    if oracle_var_pa is not None:
        cols.append('V_ij_over_oracle_var_pa')

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=cols)
        writer.writeheader()
        for row in all_rows:
            writer.writerow({k: row.get(k, '') for k in cols})

    header = ' | '.join(cols)
    print(header)
    print('-' * len(header))
    for row in all_rows:
        print(' | '.join(str(row.get(k, '')) for k in cols))


if __name__ == '__main__':
    main()
