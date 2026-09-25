"""Regenerates data/truth/{fp,imf}.json from the shipped populations.

Fits `estimators.fp` and `estimators.Chabrier` to their whole population
with unit weights -- `registry.truth`'s own definition of theta_true for
these two datasets -- and rewrites the two files with the result, at
full float precision. Rerun after any change to either estimator's
settings; not run as part of the build.
"""

import json
import pathlib

import numpy as np

from qij_joint import datasets, estimators

_TRUTH_DIR = (pathlib.Path(__file__).resolve().parent.parent
              / 'src' / 'qij_joint' / 'data' / 'truth')


def _write(dataset: str, estimator_key: str, outputs, values, source: str) -> None:
    path = _TRUTH_DIR / f'{dataset}.json'
    with open(path) as f:
        entry = json.load(f)
    entry[estimator_key] = dict(outputs=list(outputs),
                                 values=[float(v) for v in values], source=source)
    with open(path, 'w') as f:
        json.dump(entry, f, indent=2)
        f.write('\n')


def main() -> None:
    fp_pool = datasets._fp_pool()
    theta_fp = estimators.fp(fp_pool, np.ones(len(fp_pool)))
    _write('fp', 'all', estimators.fp.outputs, theta_fp,
           'estimators.fp fit with unit weights on the full '
           'data/fp_sdss.npz population')

    imf_pool = datasets._imf_pool()
    T = estimators.Chabrier(m_min=datasets.CHABRIER_M_MIN,
                             m_b=datasets.CHABRIER_M_B,
                             bounds=datasets.CHABRIER_BOUNDS)
    theta_imf = T(imf_pool, np.ones(len(imf_pool)))
    _write('imf', 'all', T.outputs, theta_imf,
           f'estimators.Chabrier(m_min={T.m_min!r}, m_b={T.m_b!r}, '
           f'bounds={T.bounds!r}) fit with unit weights on the full '
           'data/stars.h5 population')


if __name__ == '__main__':
    main()
