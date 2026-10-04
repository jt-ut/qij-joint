"""(dataset, estimator) -> Case, and truth() reading the stored theta_true.

`Case` bundles what a draw needs: the data-generating function, a
factory for a fresh estimator instance, and the VQ transform (if any)
QIJ should quantize in. Thirteen cases: the paper's six, matching
`datasets.py`'s four draws -- pareto (shape, tail), mvt (nu, tail), fp
(all), imf (all, Chabrier) -- plus their two joint (q=2) companions,
pareto (all) and mvt (all), each one pass over the same draw computing
both outputs at once -- plus five demo mixtures: mix11 (all, `GMM2D`,
spec/method_notes.md section 5), cloudfil's two subject cores, p2
(`P2Mixture`) and p1 (`P1Mixture`) (spec/QIJ_mods_waves.md A11), and
cloudfil_u's two subject cores -- the same demo rebuilt on a single
well-specified sheared-Gaussian filament -- p2 (`P2ShearMixture`) and p1
(`P1ShearMixture`) (spec/QIJ_shearmix_interface.md section 5).
"""

import json
import pathlib
from typing import Callable, NamedTuple, Optional

import numpy as np

from . import cloudfil, cloudfil_u, datasets, estimators, gmm

_TRUTH_DIR = pathlib.Path(__file__).parent / 'data' / 'truth'


class Case(NamedTuple):
    dataset: str
    estimator: str
    draw: Callable[[int, int], np.ndarray]
    make_T: Callable[[], object]
    vq_transform: Optional[Callable[[np.ndarray], tuple]]


_CASES = {
    ('pareto', 'shape'): Case('pareto', 'shape', datasets.pareto,
                               lambda: estimators.pareto_shape, None),
    ('pareto', 'tail'): Case('pareto', 'tail', datasets.pareto,
                              lambda: estimators.pareto_tail, None),
    ('pareto', 'all'): Case('pareto', 'all', datasets.pareto,
                             lambda: estimators.pareto_all, None),
    ('mvt', 'nu'): Case('mvt', 'nu', datasets.mvt,
                         lambda: estimators.mvt_nu, datasets.mvt_vq_transform),
    ('mvt', 'tail'): Case('mvt', 'tail', datasets.mvt,
                           lambda: estimators.mvt_tail, datasets.mvt_vq_transform),
    ('mvt', 'all'): Case('mvt', 'all', datasets.mvt,
                          lambda: estimators.mvt_all, datasets.mvt_vq_transform),
    ('fp', 'all'): Case('fp', 'all', datasets.fp,
                         lambda: estimators.fp, None),
    ('imf', 'all'): Case('imf', 'all', datasets.imf,
                          lambda: estimators.Chabrier(
                              m_min=datasets.CHABRIER_M_MIN,
                              m_b=datasets.CHABRIER_M_B,
                              bounds=datasets.CHABRIER_BOUNDS), None),
    ('mix11', 'all'): Case('mix11', 'all', datasets.mix11,
                            lambda: gmm.GMM2D(K=9), None),
    ('cloudfil', 'p2'): Case('cloudfil', 'p2', datasets.cloudfil_G_B6_P3_v1,
                              lambda: cloudfil.P2Mixture(), None),
    ('cloudfil', 'p1'): Case('cloudfil', 'p1', datasets.cloudfil_G_B6_P3_v1,
                              lambda: cloudfil.P1Mixture(), None),
    ('cloudfil_u', 'p2'): Case('cloudfil_u', 'p2', datasets.cloudfil_G_U_P3_v1,
                                lambda: cloudfil_u.P2ShearMixture(), None),
    ('cloudfil_u', 'p1'): Case('cloudfil_u', 'p1', datasets.cloudfil_G_U_P3_v1,
                               lambda: cloudfil_u.P1ShearMixture(), None),
}


def case(dataset: str, estimator: str) -> Case:
    """The Case for (dataset, estimator)."""
    return _CASES[(dataset, estimator)]


def truth(dataset: str, estimator: str) -> np.ndarray:
    """theta_true (q,) for (dataset, estimator), read from
    `data/truth/<dataset>.json`, in the stored (the estimator's own)
    output order."""
    with open(_TRUTH_DIR / f'{dataset}.json') as f:
        entry = json.load(f)[estimator]
    return np.array(entry['values'], dtype=float)
