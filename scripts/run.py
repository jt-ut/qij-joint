#!/usr/bin/env python3.9
"""CLI: one (dataset, estimator, method) over a range of draws (Section
6.2). `--out` is the study root; the method folder's name is derived
from the run's configuration (e.g. `qij_gp_eps0.02`) and recorded, then
checked on every later run, in that folder's `config.json`.

    python scripts/run.py pareto tail qij --N 2000 --draws 0:1000 --out runs/cloudfil_s1000
"""
import os

for _var in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
             'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(_var, '1')

import argparse
import sys

from qij_joint import pipeline, products


def _range(spec: str) -> range:
    a, b = spec.split(':')
    return range(int(a), int(b))


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('dataset')
    p.add_argument('estimator')
    p.add_argument('method', choices=['oracle', 'ij', 'boot', 'qij', 'qijt', 'qijdt', 'ijfd'])
    p.add_argument('--N', type=int, required=True)
    p.add_argument('--draws', type=_range, required=True)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--workers', type=int, default=1)
    p.add_argument('--out', required=True)
    p.add_argument('--force', action='store_true')
    p.add_argument('--B', type=int, default=2000, help='boot replicates')
    p.add_argument('--eps', type=float, default=0.01, help='qij cost tolerance')
    p.add_argument('--diag-draws', type=_range, default=None,
                    help='qij/qijt draws that also store points/prototypes')
    p.add_argument('--gptrend', choices=['affine', 'quadratic'], default='affine')
    p.add_argument('--gpwidth', choices=['global', 'local'], default='global')
    p.add_argument('--M-X', dest='M_X', type=int, default=None,
                   help='prototype count (required for qijt)')
    p.add_argument('--budget', type=int, default=None,
                   help='qijt tree contrast evaluations (required for qijt)')
    p.add_argument('--budget-win', dest='budget_win', type=int, default=None,
                   help='qijt within-term pairs (required for qijt)')
    p.add_argument('--budget-quad', dest='budget_quad', type=int, default=None,
                   help='qijt quadratic contrasts (required for qijt)')
    p.add_argument('--ivqbins', choices=['marginal', 'joint'], default='marginal')
    p.add_argument('--survey', choices=['points', 'moments'], default='points')
    p.add_argument('--quantized-start', dest='quantized_start',
                    choices=['multistart', 'full-data'], default='multistart')
    p.add_argument('--refine-schedule', dest='refine_schedule',
                    choices=['queue', 'rounds'], default='queue')
    p.add_argument('--pilot', choices=['affine', 'gp'], default='affine',
                    help='qij: the stage-1 initial influence estimate')
    p.add_argument('--sigma-points', dest='sigma_points', action='store_true',
                    help='qij: the optional sigma-points interval stage after refinement')
    p.add_argument('--check-rule', dest='check_rule', choices=['predicted', 'measured'],
                    default='predicted',
                    help='qij: the joint check continuation rule (ivqbins=joint)')
    p.add_argument('--fit-weights', dest='fit_weights', choices=['none', 'mass'],
                    default='none',
                    help='qij: the pilot fit\'s kernel-regression noise (pilot=gp)')
    p.add_argument('--tree-rule', dest='tree_rule', choices=['perbin', 'total'],
                    default='total',
                    help='qij: the joint tree\'s growth/share rule (ivqbins=joint, pilot=gp; default total)')
    p.add_argument('--point-curvature', dest='point_curvature', action='store_true',
                    help='ijfd: real per-point central-stencil b_hat/c_q/a (spec A13, '
                         'coordinator extension); off by default (planner ruling)')
    args = p.parse_args(argv)
    if args.method == 'qijt':
        missing = [name for name, val in (('--M-X', args.M_X), ('--budget', args.budget),
                                           ('--budget-win', args.budget_win),
                                           ('--budget-quad', args.budget_quad)) if val is None]
        if missing:
            p.error(f"qijt requires {', '.join(missing)}")

    config = dict(method=args.method, dataset=args.dataset, estimator=args.estimator,
                  N=args.N, seed=args.seed)
    if args.method == 'qij':
        config.update(eps=args.eps, pilot=args.pilot, gptrend=args.gptrend,
                      gpwidth=args.gpwidth, M_X=args.M_X, ivqbins=args.ivqbins,
                      survey=args.survey, quantized_start=args.quantized_start,
                      refine_schedule=args.refine_schedule, sigma_points=args.sigma_points,
                      check_rule=args.check_rule, fit_weights=args.fit_weights,
                      tree_rule=args.tree_rule)
    elif args.method == 'boot':
        config['B'] = args.B
    elif args.method == 'ijfd':
        config['point_curvature'] = args.point_curvature
    elif args.method == 'qijdt':
        config['eps'] = args.eps
    elif args.method == 'qijt':
        config.update(M_X=args.M_X, budget=args.budget,
                      budget_win=args.budget_win, budget_quad=args.budget_quad)

    tag = products.dir_tag(args.method, config)
    md = products.method_dir(args.out, args.dataset, args.estimator, args.N, args.method, tag)
    products.ensure_config(md, config)

    params = dict(N=args.N, draws=[args.draws.start, args.draws.stop],
                  seed=args.seed, workers=args.workers, force=args.force)

    if args.method == 'oracle':
        written, skipped = pipeline.run_oracle(
            args.dataset, args.estimator, args.N, args.draws, args.seed,
            args.out, args.workers, args.force, tag=tag)
    elif args.method == 'ij':
        written, skipped = pipeline.run_ij(
            args.dataset, args.estimator, args.N, args.draws, args.seed,
            args.out, args.workers, args.force, tag=tag)
    elif args.method == 'boot':
        params['B'] = args.B
        written, skipped = pipeline.run_boot(
            args.dataset, args.estimator, args.N, args.draws, args.seed,
            args.out, args.workers, args.B, args.force, tag=tag)
    elif args.method == 'qij':
        params['eps'] = args.eps
        params['diag_draws'] = [args.diag_draws.start, args.diag_draws.stop] \
            if args.diag_draws else []
        params.update(gptrend=args.gptrend, gpwidth=args.gpwidth, M_X=args.M_X,
                      ivqbins=args.ivqbins, survey=args.survey,
                      quantized_start=args.quantized_start,
                      refine_schedule=args.refine_schedule, pilot=args.pilot,
                      sigma_points=args.sigma_points, check_rule=args.check_rule,
                      fit_weights=args.fit_weights, tree_rule=args.tree_rule)
        written, skipped = pipeline.run_qij(
            args.dataset, args.estimator, args.N, args.draws, args.seed,
            args.out, args.eps, args.diag_draws, args.force,
            workers=args.workers, gptrend=args.gptrend, gpwidth=args.gpwidth, M_X=args.M_X,
            ivqbins=args.ivqbins, survey=args.survey, quantized_start=args.quantized_start,
            refine_schedule=args.refine_schedule, pilot=args.pilot,
            sigma_points=args.sigma_points, check_rule=args.check_rule,
            fit_weights=args.fit_weights, tree_rule=args.tree_rule, tag=tag)
    elif args.method == 'qijdt':
        params['eps'] = args.eps
        params['diag_draws'] = [args.diag_draws.start, args.diag_draws.stop] \
            if args.diag_draws else []
        written, skipped = pipeline.run_qijdt(
            args.dataset, args.estimator, args.N, args.draws, args.seed,
            args.out, args.eps, args.diag_draws, args.force, workers=args.workers, tag=tag)
    elif args.method == 'qijt':
        params['diag_draws'] = [args.diag_draws.start, args.diag_draws.stop] \
            if args.diag_draws else []
        params.update(M_X=args.M_X, budget=args.budget,
                      budget_win=args.budget_win, budget_quad=args.budget_quad)
        written, skipped = pipeline.run_qijt(
            args.dataset, args.estimator, args.N, args.draws, args.seed,
            args.out, args.M_X, args.budget, args.budget_win, args.budget_quad,
            args.diag_draws, args.force, workers=args.workers, tag=tag)
    else:
        params['point_curvature'] = args.point_curvature
        written, skipped = pipeline.run_ijfd(
            args.dataset, args.estimator, args.N, args.draws, args.seed,
            args.out, args.workers, args.point_curvature, args.force, tag=tag)

    products.append_log(md, sys.argv, params, written, skipped)


if __name__ == '__main__':
    main()
