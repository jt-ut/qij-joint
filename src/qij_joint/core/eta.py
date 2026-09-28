"""The full data's own measured reproducibility eta_full
(spec/QIJ_mods_waves.md A13 step 2, A9 item 3, A15): the polish-
acceptance tolerance any full-data evaluation of a restart estimator
must use in place of its declared `eta`, so a Newton polish is not
asked to reach a residual finer than the fit itself reproduces.
"""
from __future__ import annotations

from typing import Tuple

import numpy as np

from .xvq import _measure_eta_Q

# A13 step 2: two fits at the base weights plus one at a perturbed
# weight, always exactly three full-data evaluations when they run.
_ETA_FULL_EVALS = 3


def _round_up_pow10(x: float) -> float:
    """The smallest power of ten at or above `x` (A15), with `log10`'s
    own round-off nudged down first so an `x` that already IS a power of
    ten (to float64 precision) is not pushed up a decade."""
    if not np.isfinite(x) or x <= 0.0:
        return x
    return float(10.0 ** np.ceil(np.log10(x) - 1e-9))


def measure_eta_full(counter, X: np.ndarray,
                      theta_hat: np.ndarray) -> Tuple[float, int, bool]:
    """(eta_full, evals, failed). A13 step 2's reproducibility measure
    on the full data: two fits continued from `theta_hat` at unit
    weights and one continued at a relative 1e-6 perturbation of one
    point and back -- every point carries the same mass 1/N here, so
    `_measure_eta_Q`'s own argmax picks the first index, an arbitrary
    but fixed choice -- the largest relative difference among the
    three, floored at the polish residual, rounded UP to a power of ten
    (A15). `failed` is True the moment one of those three continuations
    itself fails (`_measure_eta_Q`, spec/QIJ_estimator_fit_spec.md 2.3);
    `eta_full` is then `counter.eta` rounded up, never NaN, the same
    guarantee `_measure_eta_Q` already gives its own caller, so a
    caller of THIS function that ignores `failed` still gets a finite,
    safe `eta_full`. `counter` is anything exposing `.eta`,
    `.takes_start` and `counter(X, w, start=...)`: a `core.counter.Counter`
    or an estimator itself. An estimator without `takes_start` has no
    continuation to measure, so this returns `(counter.eta, 0, False)`
    with no evaluation spent, keeping every audited non-restart path
    unchanged."""
    if not getattr(counter, 'takes_start', False):
        return float(counter.eta), 0, False
    N = X.shape[0]
    omega0 = np.ones(N)
    p = np.full(N, 1.0 / N)
    row_field = np.arange(N)
    eta_raw, failed = _measure_eta_Q(counter, X, omega0, row_field, p, theta_hat)
    return _round_up_pow10(eta_raw), _ETA_FULL_EVALS, failed
