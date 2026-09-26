"""`Bootstrap`: the comparator method (Section 6.2). `Bootstrap(B,
seed).fit(X, T, pool=None)` draws `B` multinomial resamples of `X` and
evaluates `T` on each. Identical at any worker count: the parent draws
every replicate's counts, in replicate order, from one RNG stream --
workers only evaluate `T` on rows the parent already drew.
"""
from __future__ import annotations

import time
from concurrent.futures import FIRST_COMPLETED, wait
from typing import Optional

import numpy as np

from .parallel import Pool, prepared
from .result import BootstrapResult


def _boot_task(T, case, X: np.ndarray, counts: np.ndarray):
    """Evaluate `T` at each row of `counts` against the shared `X`;
    returns the chunk's replicates and this call's own wall time. A
    failing evaluation is caught here, the one place an estimator's
    exception is allowed to turn into a NaN replicate rather than
    aborting the chunk."""
    prep = prepared(T, X)
    m, q = counts.shape[0], len(T.outputs)
    out = np.empty((m, q), dtype=float)
    t0 = time.perf_counter()
    for i in range(m):
        try:
            out[i] = T(X, counts[i], prep=prep) if prep is not None else T(X, counts[i])
        except Exception:
            out[i] = np.nan
    return out, time.perf_counter() - t0


class Bootstrap:
    """The percentile bootstrap (Section 6.2)."""

    def __init__(self, B: int = 2000, seed: int = 0) -> None:
        self.B = B
        self.seed = seed

    def fit(self, X: np.ndarray, T, pool: Optional[Pool] = None) -> BootstrapResult:
        """Draw `B` resamples of `X`, evaluate `T` on each, in chunks of
        about `B / (4 * workers)` replicates; up to `workers` chunks run
        on `pool` at once (an in-process `Pool(1)` when none is given),
        each replaced by a freshly drawn chunk as soon as it finishes, so
        no worker idles behind another chunk's evaluation."""
        X = np.asarray(X)
        N, q = len(X), len(T.outputs)
        own_pool = pool is None
        pool = Pool(1, T=T) if own_pool else pool
        workers = pool.workers
        chunk = max(1, self.B // (4 * workers))
        rng = np.random.default_rng(self.seed)
        probs = np.full(N, 1.0 / N)
        pool.share(X)

        replicates = np.empty((self.B, q), dtype=float)
        busy_time = 0.0
        t_start = time.perf_counter()

        def draw(m: int) -> np.ndarray:
            nonlocal busy_time
            td0 = time.perf_counter()
            counts = np.empty((m, N), dtype=float)
            for i in range(m):
                counts[i] = rng.multinomial(N, probs)
            busy_time += time.perf_counter() - td0
            return counts

        inflight = {}
        next_b = 0
        while next_b < self.B or inflight:
            while next_b < self.B and len(inflight) < workers:
                m = min(chunk, self.B - next_b)
                inflight[pool.submit(_boot_task, draw(m))] = (next_b, m)
                next_b += m
            done, _ = wait(list(inflight), return_when=FIRST_COMPLETED)
            for fut in done:
                offset, m = inflight.pop(fut)
                out, wall = fut.result()
                replicates[offset:offset + m] = out
                busy_time += wall
        wall_time = time.perf_counter() - t_start
        if own_pool:
            pool.close()

        n_failed = int(np.any(np.isnan(replicates), axis=1).sum())
        return BootstrapResult(
            outputs=T.outputs, replicates=replicates, n_failed=n_failed,
            wall_time=wall_time, busy_time=busy_time, workers=workers,
        )
