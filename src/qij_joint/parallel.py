"""The one module that touches a process pool (plan Section 2's
parallelism contract): a persistent executor, an estimator resolved once
at pool start (by object or by registry key), and a draw's data shared
with workers once, as a memory-mapped file.
"""
from __future__ import annotations

import os
import shutil
import tempfile
from concurrent.futures import Future
from typing import Callable, Optional, Sequence, Tuple

import numpy as np
from joblib.externals.loky import get_reusable_executor

_BLAS_VARS = ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
              'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS')

# Process-local state, set once by the pool's initializer (contract rule
# 6) and read by every task this process runs; never re-sent per task,
# never read outside this module.
_worker_T = None
_worker_case = None
_loaded_X_path: Optional[str] = None
_loaded_X: Optional[np.ndarray] = None


def _init_worker(T, case_key: Optional[Tuple[str, str]]) -> None:
    """Pool initializer: pin the five BLAS thread variables, then build
    this process's estimator, either the object `Pool` was given directly
    or one resolved from a `(dataset, estimator)` registry key (contract
    rules 1 and 6); a user's `T` crosses the process boundary exactly
    once, at pool start, never per task."""
    for var in _BLAS_VARS:
        os.environ[var] = '1'
    global _worker_T, _worker_case
    if case_key is not None:
        from . import registry
        _worker_case = registry.case(*case_key)
        _worker_T = _worker_case.make_T()
    else:
        _worker_T = T


def _worker_call(fn: Callable, X_path: Optional[str], task):
    """Run one task in this process: load the draw's shared X once and
    keep the map until a later call names a different path (contract
    rule 3), then hand the worker's own `T`, `case` and `X` to `fn`."""
    global _loaded_X_path, _loaded_X
    if X_path is not None and X_path != _loaded_X_path:
        _loaded_X = np.load(X_path, mmap_mode='r')
        _loaded_X_path = X_path
    X = _loaded_X if X_path is not None else None
    return fn(_worker_T, _worker_case, X, task)


class Pool:
    """A persistent pool (contract rule 7) over `workers` processes, built
    once with either a user's estimator `T` or a registry key
    `case = (dataset, estimator)`. `workers == 1` creates no executor: it
    runs every task in this process, through the same `_worker_call`, so
    behaviour does not depend on worker count.
    """

    def __init__(self, workers: int, T=None,
                 case: Optional[Tuple[str, str]] = None) -> None:
        self.workers = workers
        self._tmpdir: Optional[str] = None
        self._share_count = 0
        self._X_path: Optional[str] = None
        if workers > 1:
            self._executor = get_reusable_executor(
                max_workers=workers, initializer=_init_worker,
                initargs=(T, case))
        else:
            self._executor = None
            _init_worker(T, case)

    def share(self, X: np.ndarray) -> str:
        """Write `X` once for the coming draw, as a `.npy` file in a
        private temporary directory; workers memory-map this same path
        until the next `share` call replaces it (contract rule 3)."""
        if self._tmpdir is None:
            self._tmpdir = tempfile.mkdtemp(prefix='qij_joint_')
        self._share_count += 1
        path = os.path.join(self._tmpdir, f'X_{self._share_count}.npy')
        np.save(path, np.asarray(X))
        self._X_path = path
        return path

    def map(self, fn: Callable, tasks: Sequence) -> list:
        """Run `fn(T, case, X, task)` once per task, in task order
        (never by completion order, so callers can reassemble results by
        position). `fn` returns its own bookkeeping -- evaluations, rows,
        a failure flag, its own wall time (contract rule 4); any
        exception `fn` raises while calling `T` is `fn`'s own concern
        (contract rule 5), not caught here."""
        if self._executor is None:
            return [_worker_call(fn, self._X_path, task) for task in tasks]
        futures = [self._executor.submit(_worker_call, fn, self._X_path, task)
                   for task in tasks]
        return [f.result() for f in futures]

    def submit(self, fn: Callable, task) -> Future:
        """Submit one task, returning a future a caller can poll or wait
        on alongside others, so a caller can keep several tasks in flight
        and replace each as it completes rather than waiting on a whole
        batch (contract rule 4 still applies to `fn`'s return value).
        `workers == 1` runs the task now and returns an already-resolved
        future, matching `map`'s in-process behaviour."""
        if self._executor is None:
            future: Future = Future()
            future.set_result(_worker_call(fn, self._X_path, task))
            return future
        return self._executor.submit(_worker_call, fn, self._X_path, task)

    def close(self) -> None:
        """Release this pool's temporary directory; the executor itself
        is process-wide and reusable, so it is not shut down here."""
        if self._tmpdir is not None:
            shutil.rmtree(self._tmpdir, ignore_errors=True)
            self._tmpdir = None
