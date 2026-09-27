#!/usr/bin/env python3.9
"""CLI: one (dataset, estimator, method) over a range of draws (Section
6.2).

    python scripts/run.py pareto tail qij --N 2000 --draws 0:1000 --out RUNS
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
    p.add_argument('method', choices=['oracle', 'ij', 'boot', 'qij'])
    p.add_argument('--N', type=int, required=True)
    p.add_argument('--draws', type=_range, required=True)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--workers', type=int, default=1)
    p.add_argument('--out', required=True)
    p.add_argument('--force', action='store_true')
    p.add_argument('--B', type=int, default=2000, help='boot replicates')
    p.add_argument('--eps', type=float, default=0.01, help='qij cost tolerance')
    p.add_argument('--diag-draws', type=_range, default=None,
                    help='qij draws that also store points/prototypes')
    p.add_argument('--gptrend', choices=['affine', 'quadratic'], default='affine')
    p.add_argument('--gpwidth', choices=['global', 'local'], default='global')
    p.add_argument('--M-X', dest='M_X', type=int, default=None)
    p.add_argument('--ivqbins', choices=['marginal', 'joint'], default='marginal')
    p.add_argument('--survey', choices=['points', 'moments'], default='points')
    args = p.parse_args(argv)

    md = products.method_dir(args.out, args.dataset, args.estimator, args.N, args.method)
    params = dict(N=args.N, draws=[args.draws.start, args.draws.stop],
                  seed=args.seed, workers=args.workers, force=args.force)

    if args.method == 'oracle':
        written, skipped = pipeline.run_oracle(
            args.dataset, args.estimator, args.N, args.draws, args.seed,
            args.out, args.workers, args.force)
    elif args.method == 'ij':
        written, skipped = pipeline.run_ij(
            args.dataset, args.estimator, args.N, args.draws, args.seed,
            args.out, args.workers, args.force)
    elif args.method == 'boot':
        params['B'] = args.B
        written, skipped = pipeline.run_boot(
            args.dataset, args.estimator, args.N, args.draws, args.seed,
            args.out, args.workers, args.B, args.force)
    else:
        params['eps'] = args.eps
        params['diag_draws'] = [args.diag_draws.start, args.diag_draws.stop] \
            if args.diag_draws else []
        params.update(gptrend=args.gptrend, gpwidth=args.gpwidth, M_X=args.M_X,
                      ivqbins=args.ivqbins, survey=args.survey)
        written, skipped = pipeline.run_qij(
            args.dataset, args.estimator, args.N, args.draws, args.seed,
            args.out, args.eps, args.diag_draws, args.force,
            workers=args.workers, gptrend=args.gptrend, gpwidth=args.gpwidth, M_X=args.M_X,
            ivqbins=args.ivqbins, survey=args.survey)

    products.append_log(md, sys.argv, params, written, skipped)


if __name__ == '__main__':
    main()
