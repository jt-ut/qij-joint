"""
`Counter`: wraps one estimator so the method sees only T(X, w) -> array,
counting evaluations, rows and NaN failures, and computing T.prepare(X)
once per distinct array object passed to it (spec/method_notes.md
section 1). Nothing else of T is reachable through a Counter.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np


class Counter:
    """
    Counting wrapper around one estimator T, over N rows of the full
    data. `outputs`/`name`/`eta` are copied from T at construction. If
    T has `prepare`, `T.prepare(A)` is computed once per distinct array
    object A this Counter is called on and reused for that object,
    held by reference (keyed on identity) so its id cannot be reused
    while cached. An estimator without `prepare` is called T(A, w).
    """

    def __init__(self, T, N: int) -> None:
        self._T = T
        self.N = N
        self.outputs = T.outputs
        self.name = T.name
        self.eta = T.eta
        self._has_prepare = hasattr(T, 'prepare')
        self._prep_cache = {}  # id(A) -> (A, prep); holds A alive

        self.evaluations = 0
        self.rows = 0
        self.failed = 0

    def __call__(self, X: np.ndarray, w: np.ndarray) -> np.ndarray:
        if self._has_prepare:
            key = id(X)
            entry = self._prep_cache.get(key)
            if entry is None or entry[0] is not X:
                entry = (X, self._T.prepare(X))
                self._prep_cache[key] = entry
            result = np.asarray(self._T(X, w, prep=entry[1]), dtype=float)
        else:
            result = np.asarray(self._T(X, w), dtype=float)
        self.evaluations += 1
        self.rows += len(X)
        if np.any(np.isnan(result)):
            self.failed += 1
        return result

    def snapshot(self) -> Tuple[int, int]:
        """Return (evaluations, rows) so far, for per-stage differencing."""
        return (self.evaluations, self.rows)

    def add(self, evaluations: int, rows: int, failed: int) -> None:
        """Account for evaluations a worker performed on `self`'s behalf
        (parallelism contract rule 4): the parent calls this once per
        task, with that task's own bookkeeping, in place of a direct
        `__call__` it never made in this process."""
        self.evaluations += evaluations
        self.rows += rows
        self.failed += failed
