#!/usr/bin/env python3.9
"""CLI: one (dataset, estimator, method) over a range of draws (Section
6.2). `--out` is the study root; the method folder's name is derived
from the run's configuration (e.g. `qij_eps0.02`) and recorded, then
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
    p.add_argument('method', choices=['oracle', 'ij', 'boot', 'qij'])
    p.add_argument('--N', type=int, required=True)
    p.add_argument('--draws', type=_range, required=True)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--workers', type=int, default=1)
    p.add_argument('--out', required=True)
    p.add_argument('--force', action='store_true')
    p.add_argument('--B', type=int, default=2000, help='boot replicates')
    p.add_argument('--eps', type=float, default=0.01, help='qij: the tolerance eps')
    p.add_argument('--level', type=float, default=0.84, help='qij: the test\'s level')
    p.add_argument('--M', type=int, default=None,
                   help='qij: pilot size (default: the budget rule M(eps) = 3 (kappa_r / eps)^(r/2), r = min(d, q))')
    args = p.parse_args(argv)

    config = dict(method=args.method, dataset=args.dataset, estimator=args.estimator,
                  N=args.N, seed=args.seed)
    if args.method == 'qij':
        config.update(eps=args.eps, level=args.level, M=args.M)
    elif args.method == 'boot':
        config['B'] = args.B

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
    else:
        params.update(eps=args.eps, level=args.level, M=args.M)
        written, skipped = pipeline.run_qij(
            args.dataset, args.estimator, args.N, args.draws, args.seed,
            args.out, eps=args.eps, level=args.level, M=args.M,
            workers=args.workers, tag=tag, force=args.force)

    products.append_log(md, sys.argv, params, written, skipped)


if __name__ == '__main__':
    main()
