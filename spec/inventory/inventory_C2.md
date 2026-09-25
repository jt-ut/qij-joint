# Inventory C2 — `qij` core scope

Scope: `src/qij/core/refine.py`, `src/qij/core/ivq.py`,
`src/qij/core/differences.py`, `src/qij/core/intervals.py`,
`src/qij/core/counter.py`, `src/qij/core/outputs.py`,
`src/qij/core/__init__.py`. Repo `/Users/jtaylor/Dropbox/Software/JT_Py_Pkgs/qij`
@ `025613e`, read-only. Trace source: shared scratch
`executed_lines.json` / `executed_by_task.json` / `trace_notes.md`, produced
by the tracer agent over 16 tasks (14 "normal" — s=0,1 for each of
pareto/shape, pareto/tail, mvt/nu, mvt/tail, fp/fp, imf/chabrier, plus
`check` and `gmm` — and 2 TACC-failed `mvt/nu` draws, s=342 and s=420, both
of failure kind "n_failed>0 & NaN V_btw"; no `theta_hat`-NaN failure kind
exists anywhere in the main study). The trace completed without exceptions
(`trace_notes.md`: "Exceptions: None").

**Status rule used below** (as given): MAIN = the item has executed lines
anywhere in the union of all 16 traced tasks, *including* the two
TACC-failed `mvt/nu` draws (those two draws are part of "the trace" as
specified for this inventory). DATA = zero executions across all 16 tasks,
but a real, documented, data-reachable path under main-path settings
(failure/degenerate handling). SWITCH = gated by an option value the main
path doesn't use. UNREACHED = dead under the current call graph/estimator
registry, not merely unlucky in this trace's data.

Where a function's status is mixed (some lines MAIN, a sub-branch DATA),
I split it into multiple rows so status is unambiguous per row.

---

## Table 1 — inventory

| item | file:lines | what it does | status | lines |
|---|---|---|---|---|
| module | core/__init__.py:1 | one-line package docstring naming the six core submodules; no executable logic | MAIN | 1 |
| module | core/counter.py:1-6 | module docstring: `Counter` wraps `T` so the method sees only `T(X,w)` | MAIN | 1 |
| class `Counter` | core/counter.py:15-46 | counting wrapper around one estimator `T`; exposes only `outputs`/`name`/`eta`, no `__getattr__` forwarding | MAIN | 15,16,25-46 |
| `Counter.__init__` | core/counter.py:25-34 | records `T`, `N`, copies `T.outputs/name/eta`, zeroes `evaluations/rows/failed` | MAIN | 25-30,32-34 |
| `Counter.__call__` (success path) | core/counter.py:36-39,42 | calls `T(X,w)`, counts the evaluation and its rows | MAIN | 36-39,42 |
| `Counter.__call__` NaN branch | core/counter.py:40-41 | `self.failed += 1` when the result carries any NaN | MAIN — confirmed **only** by the two `mvt/nu` s=342/420 failed draws; zero occurrences in the other 14 tasks (`trace_notes.md`, executed_by_task diff) | 40,41 |
| `Counter.snapshot` | core/counter.py:44-46 | returns `(evaluations, rows)` so far, for per-stage differencing | MAIN | 44,46 |
| module | core/outputs.py:1-4 | module docstring: BCa acceleration constant | MAIN | 1 |
| `acceleration` | core/outputs.py:11-26 | `a_bca = mean(field^3) / (6*sqrt(N)*mean(field^2)^1.5)`; full body executes, including on NaN `field` (from `_degenerate_result`/`_failed_result`, which still call it) | MAIN, full body confirmed | 11,22-26 |
| module | core/differences.py:1-12 | module docstring: the one weight constructor and one difference rule, shared by stage 1 and stage 2 | MAIN | 1 |
| `forward_step` | core/differences.py:21-28 | `delta_f = 2*sqrt(eta)`; used for every single-sided step (stage 1 and refinement's one-shot child measurement) | MAIN, full body | 21,28 |
| `central_step` | core/differences.py:31-37 | `delta = (3*eta)^(1/3)`; used by stage 2's bin differences | MAIN, full body | 31,37 |
| `step_parameter` | core/differences.py:40-53 | `t = delta*p/(1-p)`, the weight parameter for a relative step `delta` on a set of mass `p` | MAIN, full body | 40,53 |
| `perturbed_weights` | core/differences.py:56-79 | the one weight constructor: `omega_i(t) = (1-t)*omega0_i + t*omega0_i*1{i in K}/p` | MAIN, full body | 56,75-79 |
| `difference`, central branch | core/differences.py:82-125 | central three-point stencil `U=[T(+t)-T(-t)]/2t`, `d2T=[T(+t)-2T_base+T(-t)]/t^2`, taken when `delta<=1` | MAIN | 82,117-125 |
| `difference`, one-sided branch | core/differences.py:126-130 | one-sided second-order stencil, taken when `delta>1` (the downward step would go negative) | **UNREACHED** — 0 executions in all 16 tasks. `delta = central_step(eta) = cbrt(3*eta)`; every registered estimator's `eta` (EPS≈2.2e-16 for mean/pareto/mvt_tail, 1e-12 for mvt_nu/fp, 1e-6 for Chabrier **and** the (removed) two-regime IMF) gives `delta` ≤ 0.0144, so `delta<=1` always holds; would need an estimator to declare `eta > 1/3`. No estimator in `estimators.py` does. For: an estimator whose declared relative accuracy is so coarse that the downward central step would make a member's weight negative. | none |
| module | core/intervals.py:1-7 | module docstring: the two interval constructions, pure functions of their args, no interval ever stored | MAIN | 1 |
| `qij_interval` (def only) | core/intervals.py:15-16 | signature/def line executes at import; body (55-64: `z*sqrt(max(V_btw,0))`, `theta_hat ± half`) is **never called** in any of the 16 traced tasks | **UNREACHED in the traced pipeline** — see note below | 15 |
| `percentile_interval` (def only) | core/intervals.py:67 | signature/def line executes at import; body (80-84: `nanquantile` lo/hi) is **never called** in any of the 16 traced tasks | **UNREACHED in the traced pipeline** — see note below | 67 |
| module | core/ivq.py:1-33 | module docstring: the 𝓘-VQ, its bins, and full-data measurement | MAIN | 1 |
| `BinSet` dataclass + `.p` property | core/ivq.py:47-102 | the per-coordinate 𝓘-VQ container; `.p` = `n/N` | MAIN | 47-98,99-102 |
| `kmeans_1d`, main loop | core/ivq.py:105-197 | Lloyd's iteration on sorted 1-D `psi0_c`; assignment via `searchsorted` on midpoints, update via two `bincount`s (counts, sums), convergence or 100-iter cap | MAIN | 105,153-155,157,159-162,165,166,173,174,182-189,193,194,196,197 |
| `kmeans_1d`, `M==1` shortcut | core/ivq.py:162-163 | `labels = zeros(N)` when only one prototype remains | **UNREACHED** — 0 executions. Dead under current call sites: `build_bins` only calls `kmeans_1d` when `M_init>1` (the `M_init<=1` case is handled by `build_bins` itself without calling `kmeans_1d`), and `refine._try_level_split` always calls `kmeans_1d(psi_leaf, 2)`. So `kmeans_1d` is never invoked with `M==1` from anywhere in the package. | none |
| `kmeans_1d`, empty-bin drop | core/ivq.py:174-180 | drops a prototype whose bin is empty after assignment and relabels contiguously | **DATA** (not executed in this trace's 16 tasks; a genuinely possible outcome of quantile-based initialization on skewed/heavy-tailed data — e.g. Pareto/MVT tails — that this trace's sampled draws happened not to trigger) | none |
| `within_share`, main body | core/ivq.py:200-223 | share of `Var(psi0_c)` left inside the bins; O(M) Python loop over the present labels, each a vectorized reduction | MAIN | 200,211,212,214,215,216,219-223 |
| `within_share`, single-bin/zero-variance shortcut | core/ivq.py:216-217 | `return 0.0` when `unique_labels.size<=1` or total variance is 0 | MAIN — confirmed **only** by the two `mvt/nu` s=342/420 draws (stage-1 collapse: `build_bins` is called on a `labels` array of all-zero) | 216,217 |
| `build_bins`, main body | core/ivq.py:226-275 | 1-D k-means at `M_init = min(ceil(sqrt(2.7/eps)), n_distinct)`; returns the initial `BinSet` (U/d2T still (M_used,0) placeholders) | MAIN | 226,248-250,252,253,255,259,260,262,263,265-275 |
| `build_bins`, `M_init<=1` branch | core/ivq.py:255-257 | forces one bin holding every point, when the coordinate's psi0_c has at most 1 distinct value | MAIN — confirmed **only** by the two `mvt/nu` s=342/420 draws (the coordinate's initial-influence estimate collapsed to a single value on those draws) | 255-257 |
| `bin_differences`, `M_used==1` shortcut | core/ivq.py:327-329 | `U=d2T=0`, no evaluations, when the bin set has one bin | **UNREACHED** — 0 executions, and structurally dead: `refine.run_refinement` only calls `bin_differences` *after* checking `bins0.M_used<=1` and returning `_degenerate_result` early in that case (refine.py:508-511), so `bin_differences` is never called with `M_used==1` from the sole call site. | none |
| `bin_differences`, main per-bin loop | core/ivq.py:322-343,349-355 | for each of `M_used` bins (small count, ≤ ~1+M_X_used), two `evaluate()` calls via `differences.difference`, then mass-weighted centering of U | MAIN | 322-325,327,331-334,336,337,339-341,343,349,350,352,353,355 |
| `bin_differences`, NaN-failure branch | core/ivq.py:344-348 | stops at the failing bin, returns `BinSet(failed=True, U/d2T/centering_residual all NaN)` — the "ruled 18 September" stage-2 initial-bin failure | **DATA**, documented, real, and reachable, but **0 executions across all 16 tasks including the two TACC-failed draws** — those two failures were both the *stage-1 collapse* (`_degenerate_result`), not a stage-2 differencing failure. Not confirmed by this trace. | none |
| `between_terms` | core/ivq.py:358-396 | `V_btw = (1/N) sum p_k U_k^2`; `B_hat = sum p_k(1-p_k) d2T_k / 2N`; only called when `bins0.failed` is False | MAIN | 358,388-396 |
| module docstring | core/refine.py:1-168 | the refinement loop: proposal, splitting, stopping rule, V_win_hat formula; see Table 2 for its C1 density | MAIN (docstring only; see note) | 1 |
| `CoordinateResult` dataclass | core/refine.py:186-287 | one coordinate's complete refinement result (fields listed in file) | MAIN | 186-188,267-287 |
| `_variance` | core/refine.py:290-294 | population variance (ddof=0), 0.0 for empty | MAIN | 290,292,294 |
| `_split_gamma`, usable-info branch | core/refine.py:297-303 | `clip(delta_realized/g_expected, 0, 1)` when `g_expected>0` and both finite | MAIN, confirmed | 297,302,303 |
| `_split_gamma`, fallback branch | core/refine.py:304 | `return 1.0` — "no usable information" | **DATA** (0 executions; structurally near-unreachable in the common case since a split is only executed when `best['g']>=tau>=0`, but reachable when `V_btw==0` forces `tau==0`, or from a non-finite `Delta`/`g` from extreme estimator output) | none |
| `_try_level_split`, success path | core/refine.py:307-321 | 2-means on `psi0` leaf values; returns the two child index arrays | MAIN, confirmed (line 321 executes) | 307,311,312,314,315,317-319,321 |
| `_try_level_split`, `<2` distinct values | core/refine.py:311-313 | `return None` when the leaf has fewer than 2 distinct psi0 values | **DATA** (0 executions; possible with ties/near-degenerate psi0 within a bin) | none |
| `_try_level_split`, k-means collapse | core/refine.py:314-316 | `return None` when `kmeans_1d(psi_leaf,2)` collapses to <2 prototypes | **DATA** (0 executions) | none |
| `_try_level_split`, empty side | core/refine.py:317-320 | `return None` when one child index array is empty | **DATA** (0 executions) | none |
| `_try_adjacency_split`, fallback branch | core/refine.py:324-351 | `return None` when the CADJ mask is empty or all-true (caller falls back to a level split) | MAIN — **confirmed**, line 351 executes | 348-351 |
| `_try_adjacency_split`, success branch | core/refine.py:352 | `return idx[mask], idx[~mask]` | MAIN — confirmed | 352 |
| `_degenerate_result` | core/refine.py:355-408 | one synthetic bin holding every point; NaNs every variance quantity and sets `failed=True` — stage-1 collapse (constant path or `M_used<=1`) | MAIN — confirmed **only** by `mvt/nu` s=342 and s=420 (the documented "3 MVT nu draws" collapse case, 18/19 Sept ruling); 0 occurrences in the other 14 tasks | 355,394-407 |
| `_failed_result` | core/refine.py:411-447 | NaNs every variance quantity, keeps the diagnostics `bin_differences` established before its own failure — stage-2 initial-bin evaluation failure | **DATA**, documented ("ruled 18 September"), real, but **0 executions**: neither TACC-failed draw hit this path (both hit `_degenerate_result` instead) | none (def line 411 only, from import) |
| `run_refinement`, entry/setup | core/refine.py:450-546 | builds `bins0`, computes `V_btw0`/`B_hat0`, seeds the leaves dict from the initial bins | MAIN | 450,505,506,508,517,519-546 |
| `run_refinement`, `constant_path or M_used<=1` branch | core/refine.py:510-511 | routes to `_degenerate_result` | MAIN — confirmed by the same 2 `mvt/nu` draws | 510,511 |
| `run_refinement`, `bins0.failed` branch | core/refine.py:513-515 | routes to `_failed_result` | **DATA** (line 513/514 execute as the check itself is always made; line 515, the actual route, never fires — 0 executions) | 513,514 (515 not confirmed) |
| `leaf_ubar`/`sum_p_ubar2`/`compute_rho2` | core/refine.py:548-559 | per-leaf mean of centered psi0; mass-weighted sum of squared means; `rho^2 = V_btw / ((1/N) sum p ubar^2)` | MAIN, full bodies | 549,551-555,557-559 |
| `batch_v`, empty-qualifying branch | core/refine.py:575-577 | `return` with nothing set when every leaf in the batch has ≤1 point | MAIN — confirmed | 575-577 |
| `batch_v`, normal branch | core/refine.py:578-581 | one `bin_posterior_variance` call over all qualifying leaves; caches `leaf['v']` | MAIN — confirmed | 578-581 |
| `propose`, `n_k<=1` branch | core/refine.py:592-598 | closes a singleton leaf without proposing a split | MAIN — confirmed (occurs for a split's singleton child) | 592-598 |
| `propose`, level-split branch (`var_k>v_k`) | core/refine.py:600-622 | proposes a level split; gain `g = rho^2(p_a ubar_a^2+p_b ubar_b^2-p_k ubar_k^2)/N` | MAIN — confirmed | 600-609,614-622 |
| `propose`, level-split-fails-to-None sub-branch | core/refine.py:609-613 | closes the leaf if `_try_level_split` returns None inside the `var_k>v_k` arm | **DATA** (0 executions — consistent with `_try_level_split` never returning None anywhere in this trace) | none |
| `propose`, adjacency branch (`var_k<=v_k`) | core/refine.py:624-638 | tries CADJ adjacency split, falls back to level split on None; gain `g = rho^2 p_k v_k/N` | MAIN — confirmed, **including the fallback-to-level sub-case** (lines 627-628 execute) | 624-629,634-638 |
| `propose`, adjacency-and-fallback-both-fail sub-branch | core/refine.py:629-633 | closes the leaf if even the fallback level split returns None | **DATA** (0 executions) | none |
| `run_refinement` while-loop, empty-queue exit | core/refine.py:647-649 | `break` when no leaf is open | **DATA** (0 executions — every traced draw's loop exited via the tolerance or cap check instead) | none |
| `run_refinement` while-loop, tolerance exit | core/refine.py:650-653 | `break` when the best remaining gain is below `tau = eps*V_btw/len(leaves)` | MAIN — confirmed, this is the observed normal stopping condition | 647,648,650-653 |
| `run_refinement` while-loop, cost-cap exit | core/refine.py:654-655 | `break` when `n_refine_evals >= evals_cap = 1+M_X_used` | **DATA** (line 654, the check, executes every iteration; line 655, the actual break, never fires — the cost guard never bound in any of the 16 traced draws) | 654 (655 not confirmed) |
| `run_refinement` while-loop, split evaluation | core/refine.py:657-674 | forward-differences the smaller child at `t_small`, counts the evaluation | MAIN | 657-660,662,664-666,668-672,674 |
| `run_refinement` while-loop, split-eval NaN branch | core/refine.py:674-684 | cancels the split (closes the parent, no retry) when `T_small` carries a NaN — "ruled 18 September", case 2 | **DATA**, documented, real, **0 executions** in all 16 tasks (unlike the stage-1/stage-2 failures, no refinement-split evaluation ever returned NaN here, even in the two TACC-failed draws — their failure was the earlier stage-1 collapse, so this loop never ran for that coordinate at all) | none |
| `run_refinement` while-loop, split acceptance | core/refine.py:686-705,710,720-737,745 | computes `U_small`/`U_large`/`Delta`, updates `V_btw`, counts level/adjacency splits, computes `gamma_children`, creates the two child leaves, prices their `v` via `batch_v` | MAIN — **both** level (700-701) and adjacency (702-703) split counting confirmed executed | 686-705,710,720,721,723,725-728,730-733,735-737,745 |
| `run_refinement` while-loop, two-strike close | core/refine.py:747-749 | closes both children immediately on the second consecutive below-tolerance split in a lineage | MAIN — confirmed executed (both `close_children` outcomes occur in the trace) | 747-749 |
| `run_refinement` while-loop, re-propose | core/refine.py:751-753 | re-proposes both children when not two-strike-closed | MAIN — confirmed | 751-753 |
| `run_refinement`, final diagnostics — rho finite | core/refine.py:756-757,792-793 | `rho=sqrt(final_rho2)`; `field = U_arr[labels,c] + rho*(psi_centered-ubar)` when finite | MAIN — confirmed (the finite branch, 792-793, is the one that executes) | 756,757,792,793 |
| `run_refinement`, final diagnostics — rho non-finite | core/refine.py:794-795 | `field = NaN` array when `rho` is not finite (denominator ≤0) | **DATA** (0 executions — `rho` was finite in every accepted `CoordinateResult` across all 16 tasks) | none |
| `run_refinement`, V_win_hat gather | core/refine.py:759-768 | `V_win_hat = rho^2/N * sum p_k gamma_k (Var_k(psi0)+v_k)` over multi-point final bins; `V_tot_hat = V_btw+V_win_hat` | MAIN | 759-768 |
| `run_refinement`, gain_ratio | core/refine.py:770 | `sum(measured Delta)/sum(expected g)`, NaN if no split taken | MAIN (line executes; the finite/NaN sub-branch is not distinguishable at line granularity) | 770 |
| `run_refinement`, final bin-set gather + return | core/refine.py:772-781,788-790,797-804 | assembles `labels_final`, `U_arr`, `bin_mass`/`bin_influence`/`bin_d2T`, returns `CoordinateResult` | MAIN | 772-781,788-790,797-804 |

**Note on `intervals.py` (both functions).** Neither `qij_interval` nor
`percentile_interval` executes a single body line anywhere in the 16
traced tasks, because the tracer's entry point is `qij.study._run_draw`
(plus `check`/`gmm`), which builds and stores raw fields (`V_btw`, etc.)
via `study._qij_row` but never calls `.interval()`. Both functions are
real, non-paper-only production code: `result.py`'s `QIJResult.interval()`
and `BootstrapResult.interval()` call them directly and are the package's
documented "pinned user surface" (`result.py`'s own module docstring) —
but that call also never executes in this trace (`result.py`'s `.interval`
def line executes at 74/120, its body does not). The *only* other callers
in the whole repo are the paper-only `tables.py` (lines 56,178,179) and
`figures.py` (lines 117,424,436), which are out of scope per this
inventory's instructions. So: on the traced main path strictly, these are
UNREACHED; in the full package, they are reachable production code whose
exercise happens one call later than what was traced. Flagging this
distinction explicitly rather than letting either label stand alone.

**Stale/dead cross-reference found in this scope.** `study.py:207` and
`tables.py:21` both refer to `core.intervals.qij2_intervals_closed_form`
— a function that does not exist anywhere in the current repo.
`intervals.py`'s own docstring (lines 15-47) explains why: the
second-order QIJ interval was built, tested, and rejected (plan §36.1,
§36.2 ruling 6), and removed along with the function that computed it;
`qij_interval` and `percentile_interval` are the only two functions left
in the module. Separately, `estimators.py:67` still describes
`core.intervals.qij_interval`'s "`support` argument to clip the QIJ..." —
the current `qij_interval(theta_hat, V_btw, level)` signature has no
`support` parameter; the support-clip was removed along with the interval
now resting on `V_btw` alone (revision 10, per this task's own framing).
`CoordinateResult.bin_mass`/`bin_influence`/`bin_d2T` (refine.py:281-283
fields, populated at 788-790, stored per-output by `study._qij_row`) were
that removed second-order interval's inputs; they are still computed and
stored (MAIN, confirmed) but have no reader anywhere in `tables.py` or
`figures.py` any more (both files' own docstrings say so explicitly).

---

## Table 2 — standards findings (MAIN and DATA code only)

| file:lines | rule | what |
|---|---|---|
| core/ivq.py:219-223 (`within_share`, MAIN) | E2 | Per-group Python loop over `unique_labels`, each iteration doing `v[labels==k]` (an O(N) boolean-mask scan) and summing squared deviations — a grouped sum-of-squares that `kmeans_1d`, two functions above in the same file, already shows how to do without a per-group loop (`np.bincount(labels, weights=v)` for the sum, `np.bincount(labels, weights=v**2)` for the sum of squares, both over one pass). |
| core/refine.py:548-559 (`leaf_ubar`/`sum_p_ubar2`/`compute_rho2`, MAIN) | E5 | `leaf_ubar(leaf)` recomputes `psi_centered[leaf['indices']].mean()` from scratch on every call; it is a pure function of the leaf's fixed (never-mutated) point indices. `sum_p_ubar2()` loops over every *current* leaf and calls `leaf_ubar` for each, and `compute_rho2()` (which calls it) is invoked once at setup and once per accepted split — so the whole leaf set's `ubar` values are recomputed from scratch on every one of those calls, rather than caching each leaf's `ubar` once at creation (the file already does exactly this caching for `v_k` via `batch_v`, but not for `ubar`). The module's own docstring acknowledges the resulting O(L) per call / O(L^2) total cost as a known, accepted design point, but the *repeated recomputation of an unchanged per-leaf quantity* is the specific E5 fact. |
| core/refine.py:601,764 (`propose`, `run_refinement`'s V_win_hat gather; both MAIN) | E5 | `_variance(psi_leaf)` (the leaf's `Var(psi0)`) is computed once in `propose()` to decide the split test (line 601) and, for any leaf that survives to the final bin set unsplit, computed *again* from the same fixed indices in the final V_win_hat gather (line 764) — the leaf dict caches `g`/`gamma`/`v` but not `var_k`, so a final leaf's variance is evaluated twice from the same data. |
| core/ivq.py, core/refine.py, core/differences.py, core/counter.py, core/outputs.py | E1 | None found on MAIN/DATA code in this scope. All Python-level loops here are over bins/leaves/prototypes (small counts: `M_used`, `L`, refinement rounds, k-means's 100-iteration cap) with vectorized bodies, matching the stated exemption; none loops over the N data points or over prototype/replicate rows. |
| (all six files) | E3 | None found. No linear-algebra factorization or explicit inverse appears anywhere in this scope; `core/influence_model.py` (outside this classifier's files) is where that lives. |
| (all six files) | E6 | None found. `theta_hat` is evaluated once (in `qij.py`, outside this scope) and passed down as a parameter everywhere in this scope, never re-evaluated. `differences.difference`'s two `evaluate()` calls are always at distinct weight parameters (±t, or +t/+2t); `run_refinement`'s per-split `T_small` is a single evaluation per accepted split. |
| (all six files) | E7 | None found. No pandas usage anywhere in this scope. No `np.append`/`concatenate`/list-then-stack pattern; `refine.py`'s `leaves` bookkeeping is a plain `dict` keyed by leaf id (not an array grown in a loop), and `batch_v`'s `groups = [leaf['indices'] for leaf in qualifying]` is a single list comprehension, not incremental growth. |
| core/refine.py:1-168 (module docstring) | C1 | 17 lines matched the mechanical history-word screen (`revision`, `ruled`/`ruling`, `plan §`, `September`, `withdrawn`, `dropped from`): lines 18,19,21,23,24,27,46,53,64,89,90,110,115,119,121,124,152. The whole 168-line docstring is substantially organized as a revision narrative (revision 6→7→8, three ruled failure/degenerate cases with dates, a paragraph of features "dropped from the reference implementation"), not just these matched lines in isolation. |
| core/refine.py:188-265 (`CoordinateResult` docstring) | C1 | 4 lines: 222,245,256,261 (`plan §36.2(...)`, "ruled 18 September"). |
| core/refine.py:297-304 (`_split_gamma` docstring) | C1 | 1 line: 301 ("the settled ruling"). |
| core/refine.py:324-352 (`_try_adjacency_split` docstring) | C1 | 1 line: 336 ("the companion 18 September ruling"). |
| core/refine.py:355-408 (`_degenerate_result` docstring) | C1 | 4 lines: 366,374,390,393 (`plan §36.2(4)`, "19 September", "ruling"). |
| core/refine.py:411-447 (`_failed_result` docstring) | C1 | 2 lines: 412,424 ("ruled 18 September", `plan §4`). |
| core/refine.py:450-503 (`run_refinement` docstring) | C1 | 1 line: 486 (`plan §36.2(1)`). |
| core/refine.py:561-574 (`batch_v` docstring) | C1 | 1 line: 565 (`§36.2(1)`, "THE ONE HARD PART"). |
| core/refine.py:583-591 (`propose` docstring) | C1 | 2 lines: 585,588 ("revision 8", "revision 6"). |
| core/refine.py:675 (inline comment, split-eval-NaN branch) | C1 | 1 line: "Case 2 (ruled 18 September)". |
| core/refine.py:785 (inline comment, bin_d2T construction) | C1 | 1 line: `plan §36.2(6)`. |
| core/ivq.py:1-33 (module docstring) | C1 | 1 line: 21 ("ruled 18 September"). |
| core/ivq.py:47-98 (`BinSet` docstring) | C1 | 1 line: 79 ("ruled 18 September"). |
| core/ivq.py:106-152 (`kmeans_1d` docstring) | C1 | 2 lines: 133,143 ("the earlier form of this loop" — a before/after performance narrative for the bincount rewrite). |
| core/ivq.py:279-320 (`bin_differences` docstring) | C1 | 1 line: 301 ("ruled 18 September"). |
| core/ivq.py:359-386 (`between_terms` docstring) | C1 | 3 lines: 373,382,383 ("19 September", "commit correcting the omitted term, batch of four", "ruled 18 September"). |
| core/differences.py (whole file) | C1 | 0 lines matched. Clean of history language — notable given how history-laden the rest of this scope is. |
| core/intervals.py:1-7 (module docstring) | C1 | 1 line: 2 (`plan §3, §6` — a bare spec citation; C3 permits naming a spec section once, so this is borderline rather than a clear violation). |
| core/intervals.py:15-54 (`qij_interval` docstring) | C1 | 1 line: 40 ("the superseded two-regime IMF estimator") — plus the whole "WHAT CAME OUT, and the ablation..." paragraph (lines ~32-47) is a results-narrative from a specific ablation run, not a description of what the function computes now. |
| core/intervals.py:67-79 (`percentile_interval` docstring) | C1 | 2 lines: 73,74 ("plan section 4, the rare-support ruling"). |
| core/counter.py:1-6 (module docstring) | C1 | 1 line: 3 (`plan §6, interface sheet §4` — bare spec citation, borderline). |
| core/outputs.py:1-4 (module docstring) | C1 | 1 line: 2 (`plan §3, §6` — bare spec citation, borderline). |
| core/__init__.py:1 | C1 | 1 line: 1 (`plan §6` — bare spec citation, borderline). |

---

## Summary

### Lines by status per file (this trace)

| file | total lines (`wc -l`) | executed (MAIN, union of all 16 tasks) | DATA items found (0-exec, documented/plausible) | UNREACHED items found (structurally dead) |
|---|---|---|---|---|
| core/refine.py | 805 | 254 | 11 branches/functions (see Table 1) | 0 |
| core/ivq.py | 397 | 111 | 2 branches (`bin_differences` NaN-failure; `kmeans_1d` empty-bin drop) | 2 (`bin_differences` M==1 shortcut; `kmeans_1d` M==1 shortcut) |
| core/differences.py | 133 | 26 | 0 | 1 (`difference`'s one-sided branch) |
| core/intervals.py | 84 | 6 (def lines only) | 0 | 2 (`qij_interval`, `percentile_interval` bodies — see caveat above) |
| core/counter.py | 46 | 24 | 0 (the one branch, NaN-failure, is MAIN — confirmed by the two TACC-failed draws) | 0 |
| core/outputs.py | 26 | 9 (full body) | 0 | 0 |
| core/__init__.py | 1 | 1 | 0 | 0 |
| **total** | **1,492** | **431** | | |

(`executed` counts are the raw line-number union sizes from
`executed_lines.json`, which include docstring/blank lines that a
tracer records as part of a multi-line statement's first line; they are
not a "percent of logic covered" measure — see Table 1 for the
line-by-line functional determination, which is what the status column
above is actually based on.)

### Every option and its main-path / other values, in this scope

| option | where | main-path value | other supported value(s) |
|---|---|---|---|
| `eps` | `ivq.build_bins(values, eps)`, `refine.run_refinement(..., eps, ...)` | 0.01 (from `QIJ.__init__` default, outside this scope) | any positive float; continuous tolerance, not a discrete switch — sets `M_init = ceil(sqrt(2.7/eps))` and the refinement tolerance `tau = eps*V_btw/L` |
| `eta` | `differences.forward_step/central_step`, `ivq.bin_differences`, `refine.run_refinement` | the estimator's own `T.eta`: `EPS`≈2.2e-16 (mean, pareto shape/tail, mvt tail), `1e-12` (mvt nu, fp), `1e-6` (Chabrier) | any positive float (per-estimator, e.g. the removed two-regime IMF also used `1e-6`); the `difference()` central/one-sided selection is gated on `eta` but no current value reaches the threshold (see `difference`'s one-sided branch, UNREACHED) |
| `constant_path` | `refine.run_refinement(..., constant_path, ...)`, a `bool` set by `influence_model.py` (outside this scope) | `False` for 12 of the 14 normal draws' coordinates (the ones actually measured); `True` for the `mvt/nu` s=342/420 draws' one coordinate | `True` — routes to `_degenerate_result` |
| `M_X_used` | `refine.run_refinement(..., M_X_used, ...)` | the 𝒳-VQ's realized prototype count (an int from `xvq.py`, outside this scope); bounds `evals_cap = 1+M_X_used` | any positive int |
| `M` (kmeans_1d's target prototype count) | `ivq.build_bins` passes `M_init`; `refine._try_level_split` always passes `2` | `M_init` (≥2 when called; `eps=0.01` → `M_init=min(17, n_distinct)` typically 17 for continuous psi0) for `build_bins`; `2` for level splits | any `int` with `1<=M<=n_distinct`; `M==1` is never actually passed anywhere in this repo (see `kmeans_1d`'s UNREACHED `M==1` branch) |

### Comment/docstring lines vs code lines per file

Counting only the mechanical history-word screen's hits (a lower bound on
"carries history," since continuous narrative paragraphs without a
trigger word on every line aren't all counted individually):

| file | history-flagged comment/docstring lines (C1 screen) | total lines |
|---|---|---|
| core/refine.py | 35 | 805 |
| core/ivq.py | 8 | 397 |
| core/differences.py | 0 | 133 |
| core/intervals.py | 4 | 84 |
| core/counter.py | 1 (borderline spec citation) | 46 |
| core/outputs.py | 1 (borderline spec citation) | 26 |
| core/__init__.py | 1 (borderline spec citation) | 1 |

`refine.py`'s module docstring (168 lines) is the concentration point: it
reads as a running revision log (revision 6→7→8, three separately-dated
rulings with their own before/after narratives, and a "dropped from the
reference implementation" paragraph) rather than a description of the
code as it now stands, well beyond the 35 lines the keyword screen alone
catches. `differences.py` is the one file in this scope with zero
history language of any kind.
