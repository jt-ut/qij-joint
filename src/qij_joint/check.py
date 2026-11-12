"""`python -m qij_joint.check`: a smoke run of the package.

Imports the package and runs `pipeline.run_qij` for one draw of the
smallest registry case (pareto/shape, N=200) into a temporary directory,
then reads the three products back and prints their shapes. If
`qij_joint.method` is not present yet, a stub `run` returning a
MethodResult-shaped object stands in for it, so the runner and its
product writers are still exercised; the printout says which ran.
"""
from __future__ import annotations

import sys
import tempfile
import types
from importlib.util import find_spec

import numpy as np
import pandas as pd

import qij_joint
from qij_joint import pipeline, products


def _stub_run(T, X, *, eps=0.01, level=0.84, M=None, k=None, pool=None,
              evaluator=None, seed=0):
    """The method's contract with placeholder values (no estimate)."""
    J = len(T.outputs)
    z = np.zeros(J)
    rounds = [dict(round=0, phase='pilot', n_taken=4, E=4, K=1, R_hat=z, U=z, delta=z)]
    leaves = [dict(leaf=0, n=len(X), mean=z, term=z, determined=True)]
    timings = dict(pilot_cells_s=0.0, pilot_eval_s=0.0, fit_s=0.0, eval_s=0.0, total_s=0.0)
    return types.SimpleNamespace(Vhat=z, U=z, delta=z, V_lb=z, E=4, E_full=1, M=4, K=1,
                                 rounds=1, rounds_table=rounds, leaves_table=leaves,
                                 timings=timings, intervals=None, outputs=tuple(T.outputs))


def main() -> None:
    which = 'method'
    if find_spec('qij_joint.method') is None:
        which = 'stub'
        sys.modules['qij_joint.method'] = types.SimpleNamespace(run=_stub_run)
    dataset, estimator, N = 'pareto', 'shape', 200
    with tempfile.TemporaryDirectory() as out:
        written, skipped = pipeline.run_qij(dataset, estimator, N, [0], seed=0, out_dir=out)
        assert (written, skipped) == (1, 0), (written, skipped)
        row = products.collect(out, dataset, estimator, N, 'qij')
        md = products.method_dir(out, dataset, estimator, N, 'qij')
        rounds = pd.read_parquet(f'{md}/s00000.rounds.parquet')
        leaves = pd.read_parquet(f'{md}/s00000.leaves.parquet')
    print(f'qij_joint at {qij_joint.__file__}; run_qij through the {which}')
    print(f'row {row.shape}, rounds {rounds.shape}, leaves {leaves.shape}')
    print(row.T.to_string(header=False))


if __name__ == '__main__':
    main()
