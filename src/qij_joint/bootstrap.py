"""`Bootstrap`: the comparator method (Section 6.2). `Bootstrap(B,
seed).fit(X, T, pool=None)` draws `B` multinomial resamples of `X` and
evaluates `T` on each. Identical at any worker count: the parent draws
every replicate's counts, in replicate order, from one RNG stream --
workers only evaluate `T` on rows the parent already drew. It also
evaluates `theta_hat = T(X, ones(N))` once, outside that RNG stream, so
`BootstrapResult.bc_interval` has the point estimate its z0 needs
(spec/QIJ_mods_waves.md A10) without touching the replicates themselves.
When `T.takes_start`, every replicate is warm-started from `theta_hat`
(spec/QIJ_mods_waves.md A11), so `theta_hat` is awaited before the first
replicate chunk is dispatched; an estimator without `takes_start` is
bit-identical to before, `theta_hat`'s own evaluation still overlapping
the first replicate chunks. When warm, `eta_full` is also measured from
`theta_hat` on the full data (spec/QIJ_mods_waves.md A15) and every
replicate's polish is held to it instead of `T`'s own declared `eta`,
which the Newton cap (`gmm.py`'s `_MAX_NEWTON`) can otherwise reach
before a fit at this scale has actually stalled.
"""
from __future__ import annotations

import time
from concurrent.futures import FIRST_COMPLETED, wait
from typing import Optional

import numpy as np

from . import registry
from .core.eta import measure_eta_full
from .parallel import Pool, call_T
from .result import BootstrapResult

_NO_COLD_DIAG = dict(beta_star=float('nan'), cold_ll=float('nan'),
                      search_gap=float('nan'), search_failed=False)


def _theta_hat_task(T, case, X: np.ndarray, task):
    """T(X, ones(N)) on the shared draw: the bootstrap's own full-data
    point estimate (spec/QIJ_mods_waves.md A10), submitted to the pool
    at the start of `fit` so it overlaps with the first replicate
    chunks rather than blocking them, and computed outside the resample
    loop's own RNG stream. A failing evaluation is caught here, this
    task's own declared failure boundary. `task` = (dataset, estimator)
    for `registry.cold_fit_diagnostics` (A16 items 6-7), read in THIS
    worker immediately after the cold fit, before any replicate's own
    `start=theta_hat` continuation could touch `T.last_fit_info` --
    `theta_hat`'s own fit runs in a separate process from `Bootstrap.fit`
    whenever `workers > 1`, so the diagnostics must be read here, not
    after `.result()`. Returns (evaluation, this call's own wall time,
    the diagnostics dict)."""
    dataset, estimator = task
    t0 = time.perf_counter()
    try:
        result = np.asarray(call_T(T, X, np.ones(len(X))), dtype=float)
    except Exception:
        result = np.full(len(T.outputs), np.nan)
    wall = time.perf_counter() - t0
    diag = (registry.cold_fit_diagnostics(T, X, result, dataset, estimator)
            if dataset is not None else dict(_NO_COLD_DIAG))
    return result, wall, diag


def _boot_task(T, case, X: np.ndarray, task):
    """Evaluate `T` at each row of `counts`, warm-started from `start`
    at polish tolerance `eta` (`call_T` applies the `T.takes_start` gate
    to both, so an estimator without it never sees either); returns the
    chunk's replicates and this call's own wall time. A failing
    evaluation is caught here, the one place an estimator's exception is
    allowed to turn into a NaN replicate rather than aborting the
    chunk."""
    counts, start, eta = task
    m, q = counts.shape[0], len(T.outputs)
    out = np.empty((m, q), dtype=float)
    t0 = time.perf_counter()
    for i in range(m):
        try:
            out[i] = call_T(T, X, counts[i], start=start, eta=eta)
        except Exception:
            out[i] = np.nan
    return out, time.perf_counter() - t0


class Bootstrap:
    """The percentile bootstrap (Section 6.2)."""

    def __init__(self, B: int = 2000, seed: int = 0) -> None:
        self.B = B
        self.seed = seed

    def fit(self, X: np.ndarray, T, pool: Optional[Pool] = None,
            dataset: str = None, estimator: str = None) -> BootstrapResult:
        """Draw `B` resamples of `X`, evaluate `T` on each, in chunks of
        about `B / (4 * workers)` replicates; up to `workers` chunks run
        on `pool` at once (an in-process `Pool(1)` when none is given),
        each replaced by a freshly drawn chunk as soon as it finishes, so
        no worker idles behind another chunk's evaluation. When
        `T.takes_start`, `theta_hat` is awaited up front, `eta_full`
        (spec/QIJ_mods_waves.md A15) is measured from it once, and every
        chunk carries both as the replicates' warm start and polish
        tolerance. `dataset`/`estimator`, when given, name this draw's
        case for `registry.cold_fit_diagnostics` (A16 items 6-7),
        computed in `_theta_hat_task`'s own worker process."""
        X = np.asarray(X)
        N, q = len(X), len(T.outputs)
        own_pool = pool is None
        pool = Pool(1, T=T) if own_pool else pool
        workers = pool.workers
        chunk = max(1, self.B // (4 * workers))
        rng = np.random.default_rng(self.seed)
        probs = np.full(N, 1.0 / N)
        pool.share(X)
        theta_future = pool.submit(_theta_hat_task, (dataset, estimator))

        warm = getattr(T, 'takes_start', False)
        busy_time = 0.0
        theta_hat = None
        eta_full = T.eta
        cold_diag = dict(_NO_COLD_DIAG)
        if warm:
            theta_hat, theta_wall, cold_diag = theta_future.result()
            busy_time += theta_wall
            t_eta0 = time.perf_counter()
            eta_full, _ = measure_eta_full(T, X, theta_hat)
            busy_time += time.perf_counter() - t_eta0

        replicates = np.empty((self.B, q), dtype=float)
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
                inflight[pool.submit(_boot_task, (draw(m), theta_hat, eta_full))] = (next_b, m)
                next_b += m
            done, _ = wait(list(inflight), return_when=FIRST_COMPLETED)
            for fut in done:
                offset, m = inflight.pop(fut)
                out, wall = fut.result()
                replicates[offset:offset + m] = out
                busy_time += wall

        if not warm:
            theta_hat, theta_wall, cold_diag = theta_future.result()
            busy_time += theta_wall
        wall_time = time.perf_counter() - t_start
        if own_pool:
            pool.close()

        n_degenerate = int(np.any(np.isnan(replicates), axis=1).sum())
        return BootstrapResult(
            outputs=T.outputs, theta_hat=theta_hat, replicates=replicates,
            n_degenerate=n_degenerate, wall_time=wall_time, busy_time=busy_time,
            workers=workers, eta_full=eta_full, **cold_diag,
        )
