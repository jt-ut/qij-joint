"""(dataset, estimator) -> Case, and truth() reading the stored theta_true.

`Case` bundles what a draw needs: the data-generating function, a
factory for a fresh estimator instance, and the VQ transform (if any)
QIJ should quantize in. Eight cases: the paper's six, matching
`datasets.py`'s four draws -- pareto (shape, tail), mvt (nu, tail), fp
(all), imf (all, Chabrier) -- plus two demo mixtures: mix11 (all,
`GMM2D`, spec/method_notes.md section 5) and cloudfil (p2, `P2Mixture`,
spec/QIJ_mods_waves.md A11).
"""

import json
import pathlib
from typing import Callable, NamedTuple, Optional

import numpy as np

from . import cloudfil, datasets, estimators, gmm
from .parallel import call_T

_TRUTH_DIR = pathlib.Path(__file__).parent / 'data' / 'truth'

# A16.7's search audit, when there is nothing to audit (no restart search,
# or `dataset`/`estimator` not given): NaN, no evaluation.
NO_COLD_DIAG = dict(beta_star=float('nan'), cold_ll=float('nan'),
                     search_gap=float('nan'), search_failed=False)


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
    ('mvt', 'nu'): Case('mvt', 'nu', datasets.mvt,
                         lambda: estimators.mvt_nu, datasets.mvt_vq_transform),
    ('mvt', 'tail'): Case('mvt', 'tail', datasets.mvt,
                           lambda: estimators.mvt_tail, datasets.mvt_vq_transform),
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


def search_audit(T, X: np.ndarray, theta_hat, dataset: str, estimator: str):
    """A16.7's search audit for one cold fit: one continuation from the
    TRUE parameters on the same draw (`start` = `truth(dataset,
    estimator)`, T's own output layout), then `search_gap` = the cold
    fit's penalized log-likelihood minus the truth continuation's (per
    point) and `search_failed` = that gap negative by more than the
    continuation's own polish residual. The truth never enters any
    method's own estimate, only judges the search (spec/
    QIJ_mods_waves.md A16.7). The one shared helper for every method's
    cold fit (oracle, ij, boot's theta_hat, qij's theta_hat, ijfd's
    theta_hat). (nan, False), no evaluation, for an estimator without a
    restart search to audit (no `takes_start`/`score_at`)."""
    if not (getattr(T, 'takes_start', False) and hasattr(T, 'score_at')
            and theta_hat is not None and np.all(np.isfinite(theta_hat))):
        return float('nan'), False
    N = len(X)
    w = np.ones(N)
    truth_theta = truth(dataset, estimator)
    theta_cont = np.asarray(call_T(T, X, w, start=truth_theta), dtype=float)
    if not np.all(np.isfinite(theta_cont)):
        return float('nan'), False
    ll_cold, _ = T.score_at(X, w, theta_hat)
    ll_cont, resid_cont = T.score_at(X, w, theta_cont)
    if not (np.isfinite(ll_cold) and np.isfinite(ll_cont) and np.isfinite(resid_cont)):
        return float('nan'), False
    gap = ll_cold - ll_cont
    return gap, bool(gap < -resid_cont)


def cold_fit_diagnostics(T, X: np.ndarray, theta_hat, dataset: str, estimator: str) -> dict:
    """Every product A16 and A16.7 name for one cold fit: `beta_star`,
    `cold_ll` (A16 item 6, from `T.last_fit_info`, read IMMEDIATELY
    after the cold call that produced `theta_hat` and before any other
    call on `T` -- a diagnostic side channel, never read across a
    process boundary except by this function's own return value), and
    `search_gap`/`search_failed` (A16.7, `search_audit`). `NO_COLD_DIAG`
    throughout when `dataset`/`estimator` are not given (no truth to
    audit against) or `T` has no annealed-search diagnostics."""
    if dataset is None or estimator is None:
        return dict(NO_COLD_DIAG)
    info = getattr(T, 'last_fit_info', None)
    beta_star = float(info['beta_star']) if info is not None else float('nan')
    cold_ll = float(info['ll']) if info is not None else float('nan')
    search_gap, search_failed = search_audit(T, X, theta_hat, dataset, estimator)
    return dict(beta_star=beta_star, cold_ll=cold_ll,
                search_gap=search_gap, search_failed=search_failed)
