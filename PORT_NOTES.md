# Port of the QIJ method from the test bed (2026-11-06)

Source: `vqboot/fable/bed` (METHOD_qij.md, method.py, estimators.py, data.py, run.py). Target: `src/qij_joint`.
Check: `scripts/port_check.py` at the time (removed 2026-11-12 with the quantized path it measured); the standing
equivalence check with the bed is `scripts/one_child_check.py`, and the method's suite is `scripts/suite_run.py`.

## What maps to what

| bed | package | change |
|---|---|---|
| `method.run(case, geom, est, Kmax, eps, halt_at_stop, batch)` | `method.run(T, X, *, eps, level, M, k, pool, evaluator, seed, ...)` | interface in the module docstring; no env switches (`BED_BATCH`, `BED_LEVEL`, `BED_PILOT`, `BED_PERF_CHECK` are gone) |
| `method._within, _two_means, _imhof_*, wchi2_*, _holds, _leverage, _Rows` | the same names in `method.py` | arithmetic unchanged; `_holds` returns only the boolean |
| the loop's truth diagnostics (`w_true`, `R`, `W`, `grid`, `gain`, `Rcf`, `t_vW`, the Rcf assert) | dropped | the package has no truth |
| `estimators.Matern12` | `core/interpolant.Interpolant(Xs, sigma_pt)` | same arithmetic; the `BED_PERF_CHECK` asserts and `gram`/`_check_factor` dropped |
| `data.pilot_M`, `quantize`, `_cells`, `_std` | `core/pilot.pilot_M`, `quantize`, `build_cells`, `standardize` | `quantize` takes `seed` (bed: 0) |
| `data._conn` + `geometry` (CONN scales) | `core/pilot.cell_scales` | distances only for connected pairs (the bed's dense M x M x d array is 4 GB at N = 10^5); same values to 3e-16 |
| the bed's pilot rows `A0 @ Z` (exact cell means) | `core/pilot.evaluate_cells_exact` (with `evaluator`) or `evaluate_cells_quantized` (with T) | see below |
| `case.Z` read at a split / a test / (implicitly) the report | `method._Evaluations`: `evaluator(u)` or a forward difference of T | the report is now J real evaluations (u = psi-tilde_j / N) |
| `RunRecord` | `MethodResult` (+ optional `record` dict) | |

Modules edited: `core/xvq.py` reduced to the pilot evaluation (`prototype_influences`, points survey only, plus
`_measure_eta_Q` which `core/eta.py` uses); FAISS/vqlp, `fit_xvq`, `XVQ`, the moments survey, `run_xvq`, `cost_rule_M` removed.
`core/differences.py`: one function added, `evaluation_weights(u, delta)`. Not edited: `eta.py`, `counter.py`, `abc.py`,
`parallel.py`.

## What the package does differently

1. **Evaluations.** evaluation(u) = (T(X, omega) - theta-hat) / t, omega = 1 + t (N u - sum u), t = delta / max|N u - sum u|,
   delta = forward_step(eta_full), every refit continued from theta-hat at tolerance eta_full (as the bootstrap). Weights keep
   their total N. For a cell or a leaf this is exactly `perturbed_weights` at `step_parameter(delta, p)`. Periodic outputs
   (`T.periodic`) have their difference wrapped to [-P/2, P/2). The method works on `T.measured` (the 7 derived outputs for
   shearmix), else on all outputs.
2. **Pilot evaluation.** With T, cell c's value is the forward difference of the refit on the M cell means (native
   coordinates, base weight M p_c), continued from theta-hat: `xvq.prototype_influences`. It costs (M + 14) refits on M rows,
   counted as (M + 14) M / N full refits (14 = theta_Q, eta_Q's three, the step-doubling's ten). With `evaluator`, the
   bed's exact cell means, counted as M^2 / N.
3. **Relative criteria.** Entry: pred <= eps q-tilde V_lb; the pre-test batch count uses the same limit; stop: U <= eps V_lb.
   V_lb = max(B, F / (1 + sqrt eps)^2), B = sum_l p_l m_l^2 over the leaves, F = |psi|^2 / N with psi = psi-hat before the
   test and psi-tilde at the test (the bed's formula at the test; before the test the bed used V = 1). On the bed's case
   this gives the bed's decisions exactly (item 1).
4. **Units of the ranking and the cut.** The bed ranks leaves by sum over outputs of p_l Var_l(psi-hat) and cuts by two-means
   in the J outputs, both in its units V_j = 1. The package needs a per-output unit: the default is s_j = sqrt(sum_c p_c m_c^2)
   from the pilot, fixed for the run (`scale=` overrides; the bed check passes ones). On the bed's case the default (s =
   0.50 .. 0.87) gives a different but valid run: K 182, E_full 419, R/delta 0.02 .. 0.07 (item 1, last lines).
5. **Pool.** Each round's test evaluations go to the pool in one `map`, then its child-mean evaluations in one `map` (the cuts
   are all made before any evaluation: they use psi-hat only), then the J report evaluations. k defaults to `pool.workers`.
   The pilot shares its rows on the pool and the method re-shares X afterwards.
6. **Counts.** E = M + splits + tests + J. E_full = pilot refits x M/N + splits + tests + J + full-data base evaluations
   (theta-hat and eta_full's three; none with `evaluator`).
7. **Intervals.** `intervals` is None: core.abc's ABC interval needs a, b, c, which this method does not measure, and
   theta-hat is not in MethodResult. See open issues.

## Acceptance

### 1. Bed equivalence (exact evaluator, the bed's cells, k = 16, eps 0.01, level 0.84, scale = ones): PASS

| quantity | bed record | package |
|---|---|---|
| rounds | 16 | 16 |
| leaves split per round, in order; n_taken; E per round | | identical, all 16 rounds |
| K at the stop | 179 | 179 |
| E (pilot + splits + tests) | 1370 | 1370 (1377 with the 7 report evaluations) |
| E_full | 402 + 7 report | 408.9 |
| leaf sizes at the stop | 179 leaves | identical |
| Vhat, max relative difference | | 1.1e-14 |
| U, max relative difference | | 1.0e-13 |
| delta, max relative difference | | 1.1e-15 |

The task brief said K 177; the bed's final record says 179 (the record is what is reproduced). The package's own k-means
(KMeans(1086, n_init=1, random_state=0) on the standardized X) reproduces the bed's cells exactly, and its CONN scales agree
to 3e-16.

### 2. Real evaluations, shearmix draw 0, N = 10^4, ('cloudfil_u', 'p2'): FAILED (cause below)

| item | value |
|---|---|
| X | identical to the bed's draw 0 |
| cells | identical to the bed's cells (1086) |
| full fit theta-hat (multistart, cold) | 8.3 to 12.9 s |
| warm refit at a perturbed weight | 0.07 to 0.15 s (bed: 0.13 s) |
| quantized pilot (1100 refits on 1086 rows, one process) | 20.7 s |
| the run | failed at the round-5 refit of psi-hat, 45.6 s in, after 31 splits |
| pool | the 16-worker pool cannot start in the agent sandbox (no POSIX semaphores); the run used Pool(1), the same code path and the same k = 16 |

The pilot gap (quantized pilot value vs the exact cell mean of the bed's parquet psi for this draw), per output
x, y, log_reff, log_axis_ratio, pa, logit_w, log_contrast:

| | x | y | log_reff | log_axis_ratio | pa | logit_w | log_contrast |
|---|---|---|---|---|---|---|---|
| 20 cells: median abs(diff) / abs(exact) | 1.79 | 6.83 | 3.53 | 12.2 | 3.06 | 0.26 | 3.53 |
| 20 cells: median abs(diff) / sqrt V | 0.011 | 0.021 | 0.045 | 0.064 | 0.048 | 0.020 | 0.048 |
| all cells: sum_c p_c (diff)^2 / V | 0.22 | 0.32 | 0.73 | 1.33 | 0.79 | 0.14 | 0.59 |
| all cells: sum_c p_c exact^2 / V (what the pilot should carry) | 0.48 | 0.66 | 0.51 | 0.35 | 0.25 | 0.77 | 0.67 |
| all cells: correlation | 0.91 | 0.83 | 0.57 | 0.69 | 0.67 | 0.90 | 0.73 |

The error is concentrated in a few cells (667, 459, 1028, 988, 924: up to 11.7 sqrt V), the cells around the subject
Gaussian, where the influence is largest. Diagnostic, not a run (same draw, pilot only): the old package's moments rows
(each cell represented by d + 1 rows matching its mean and covariance) give sum_c p_c diff^2 / V = 0.30, 0.16, 0.43, 0.25,
0.25, 0.25, 0.24, correlation 0.01 on pa. That is not a usable pilot either.

**Cause of the failure.** The pilot values are constraints of psi-hat but not means of psi. Where they are wrong, psi-hat
spikes; the ranking keeps choosing those leaves and the two-means cut takes off 1 to 22 points at a time inside cells 459, 988
and 195. When the full-data leaf rows inside a cell together span that cell's row, the cell's (quantized) value contradicts
the (full-data) leaf means, the constraint set has no solution, and the interpolant's reproduction assert fires (residual 4.9
against 1.6e-7). With exact cell means (the bed) the same dependency is consistent and harmless. This is the
step the METHOD document flagged as "to be measured on one draw before the package claims the bed's costs"; measured, the
quantized pilot is not a substitute for exact cell means on this statistic, and the method as specified cannot run on it.
No E, K, Vhat or U exist for item 2. I did not change the method to make it run.

### 3. Overhead with the exact evaluator (item 1's run, one process, one BLAS thread)

| | bed | package |
|---|---|---|
| total wall | 8.4 s | 7.5 to 8.8 s (two runs) |
| psi-hat fits | | 6.4 to 7.5 s |
| cells (scales only; the bed's cells passed) | | 0.5 to 0.6 s |
| ranking, cuts, test, quantiles | | 0.5 s |

## Open issues

1. **The pilot (blocking item 2).** The method needs pilot values that are exact (or at least consistent) linear functionals
   of psi. Options for the author, none implemented: (a) M full-data pilot evaluations (cost M, about 1086 refits at N =
   10^4, against 408 for the whole bed run); (b) a pilot row that is retired when full-data rows come to span it (the row
   space is known exactly: `_leverage` already computes it), with the quantized value used only as a starting guess for
   psi-hat; (c) a better quantized refit. This is a method decision, not a port detail.
2. **Units of the ranking and the cut** (difference 4): the bed's units come from its truth. The pilot-based default changes
   the run; whether it is the right unit is the author's call. With a wrong pilot (issue 1) it is wrong too.
3. **Intervals**: None. A normal interval theta-hat +- z sqrt(Vhat/N) needs theta-hat, which run computes but MethodResult
   does not carry (the interface was kept exactly). pipeline.py would need one extra fit, or MethodResult a `theta_hat` field.
4. **Pool not exercised with processes.** Item 2 ran on Pool(1) because the sandbox forbids semaphores. Running
   `PYTHONPATH=src python3.9 scripts/port_check.py` outside the sandbox uses the 16-worker pool; it will fail the same way
   (the failure does not depend on the pool).
5. **Scripts broken by the xvq reduction** (old research scripts, cleanup agent's area): `scripts/validate_gmm_fit.py`,
   `selective_cadj_oracle_metric_audit.py`, `oracle_metric_quantizer.py`, `oracle_metric_vq_test.py`,
   `selective_cadj_common_base_pilot.py` import `fit_xvq` or the moments survey.
6. No `Kmax`: the loop ends at the stop or when no leaf can be cut. A run that never passes is bounded only by N.

## Pilot diagnostic (scripts/pilot_diag.py, scripts/pilot_diag.log; 64 s)

Question: is the quantized pilot's error the quantizer (within-cell structure erased) or the quantized refit (basin, step,
convergence)? Shearmix draw 0, the bed's 1086 cells, the parquet psi; one draw each. No method change.

**1. Base fits.** theta_q (on the 1086 weighted cell means, continued from theta-hat) is in theta-hat's basin: the full-data
Newton finish from theta_q returns to theta-hat to 2.4e-10; a cold multistart on the rows finds the same theta_q; all fits
converged (score <= 3e-11; theta_q 27 EM + 2 Newton iterations, 0.05 s). The full-data objective is -2.83809 at theta-hat
and -2.84018 at theta_q. The subject moves by up to 1.9 sampling sd: log_reff -3.13 vs -3.36, log_axis_ratio -0.77 vs -1.08,
log_contrast 1.12 vs 1.40. Step: eta_Q 2.0e-6, delta 2.8e-3; the cell's weight goes up by a factor (1 + delta) and every
other row by (1 - t), t = delta p/(1 - p), so the total stays at M. The difference over t is the cell's MEAN influence (the
direction is e_c/p_c - 1), not n_c times it. Rows weighted to N instead of M (the penalty at the full data's weight total)
are worse, not better: the subject collapses (log_reff -4.89, log_contrast 4.40; 12 to 15 sampling sd).
Verdict: the fit is sound and in the right basin, but the quantized problem's subject Gaussian is a different
Gaussian (wider and rounder), and the weight total is not the fault.

**2. Error structure.**

| | x | y | lreff | laxr | pa | lgtw | lcon |
|---|---|---|---|---|---|---|---|
| sum_c p_c d_c^2 / V | 0.22 | 0.32 | 0.73 | 1.33 | 0.79 | 0.14 | 0.59 |
| corr(d_c^2, Var_c psi) | 0.61 | 0.62 | 0.53 | 0.67 | 0.58 | 0.61 | 0.50 |
| corr(abs d_c, 1/n_c) | -0.01 | -0.01 | 0.04 | 0.04 | 0.02 | 0.03 | 0.04 |
| share of sum p d^2 in the 20 worst cells | 0.998 | 0.991 | 0.989 | 0.989 | 0.995 | 0.987 | 0.988 |

All 6 cells within 3 r_eff of the subject's centre are among the 20 worst; the 10 worst, in units of sqrt V (x, lreff,
laxr, lcon):

| cell | n | dist / r_eff | exact | quantized | within-cell sd |
|---|---|---|---|---|---|
| 667 | 27 | 2.4 | 0.62 3.74 2.34 -2.16 | -2.08 9.55 14.00 -7.52 | 1.39 2.95 2.75 1.57 |
| 459 | 33 | 2.4 | -1.01 5.25 -0.25 -2.65 | -6.08 -3.55 -9.58 5.43 | 4.07 3.04 7.22 2.89 |
| 1028 | 40 | 2.3 | 5.32 6.27 6.21 -3.89 | 10.04 9.45 12.50 -6.22 | 4.67 4.94 5.59 2.50 |
| 988 | 71 | 0.8 | -4.47 -1.85 1.41 5.98 | -5.28 -6.69 -6.03 9.99 | 5.13 5.14 5.10 4.78 |
| 924 | 34 | 2.9 | 7.67 6.40 -6.67 -3.23 | 9.50 0.62 -8.00 2.52 | 3.78 2.63 4.64 2.15 |
| 195 | 48 | 0.7 | 1.86 -2.95 -1.17 7.50 | 0.37 -5.59 -4.54 9.63 | 5.25 4.74 3.84 4.15 |
| 12 | 22 | 3.8 | 0.26 0.11 0.11 -0.31 | 0.75 1.97 2.68 -1.92 | 0.28 0.66 0.71 0.53 |
| 660 | 42 | 3.4 | -0.46 1.05 0.56 -1.18 | -0.37 2.41 3.10 -2.56 | 0.60 1.80 0.99 1.13 |
| 188 | 1 | 186 | 0.01 0.03 0.04 0.04 | 0.20 0.80 1.09 -0.67 | 0 |
| 358 | 1 | 160 | 0.00 0.01 0.03 0.03 | 0.13 0.61 0.84 -0.52 | 0 |

Verdict: the error is the six cells that straddle the subject (r_eff 0.035, cells of 22 to 71 points with within-cell sd of
3 to 7 sqrt V). Collapsing each of them to a point erases the structure the subject is fitted to. The cell size doesn't
matter (corr with 1/n_c ~ 0). Even one-point cells far away are off, by up to 1.1 sqrt V, because their influence is taken
at the quantized problem's different subject.

**3. Step sensitivity** (10 cells across the error range): half, double and central steps agree with the default to 2.9e-3
(relative). The same forward difference of the cell mean on the FULL data matches the exact psi to 6e-3 sqrt V on all 10
cells, including cell 667, where the quantized value is off by 11.7 sqrt V. Verdict: the values are clean derivatives (no
basin or convergence jumps), and the step and normalisation are right; the error lies entirely in the quantized problem.

**4. Controls** (N = 10^4, draw 0, M = 1086 k-means++ cells, exact T.influence):

| case | output | sum p d^2 / V | median abs(d)/abs(exact) | corr |
|---|---|---|---|---|
| fp/all | a, b, c, scatter | 0.000, 0.000, 0.000, 0.002 | 0.008, 0.006, 0.000, 0.046 | 1.000, 1.000, 1.000, 0.999 |
| mvt/all | nu, P_tail | 93.1, 0.007 | 11.5, 0.019 | 0.976, 0.998 |

Verdict: the same code is exact for fp. It fails for mvt's nu, whose value depends on the spread of the radii, which
averaging within a cell removes. So the quantized pilot fails exactly where the statistic depends on within-cell structure.

**Overall:** fundamental, not a fault in the refit. The quantized refit is a correct derivative of a different problem,
accurate only when the statistic is insensitive to the within-cell spread (fp, P_tail), and badly wrong on shearmix's subject
and on mvt's nu.

## State of the code (2026-11-06, after the author's two pilot decisions)

- The full-data-pilot decision was implemented and then stopped (author: evaluations must come from quantized data). What
  stands in `method.py` / `core/pilot.py` now: the pilot is M evaluations at u = 1{cell c}/n_c through the one evaluation
  primitive (`_Evaluations`, weight vectors passed as (indices, weights) and built inside each task); E = E_full = every
  evaluation counted once (theta-hat, eta_full's three, M, splits, tests, J). With the exact evaluator this is the bed's pilot,
  and acceptance 1 still passes (same splits, K 179, Vhat 1.1e-14, U 1.0e-13). The real full-data run was not completed.
- `core/xvq.py` is off the method's path; only `scripts/pilot_diag.py` (the admissibility test) and `core/eta.py` use it.
- `scripts/port_check.log` still shows the earlier quantized-pilot run of item 2; it is stale for item 2.

## Inexact evaluations and dependent rows (blocks the quantized-evaluation design as specified)

The exact interpolant psi-hat must satisfy every constraint row exactly. When the active rows are linearly dependent, that is
possible only if their values satisfy the same linear relation. Exact influences always do (the bed); any evaluation with an
error does not.

Measured on the bed's case (exact evaluator, scale 1, the bed's run): the final 1371 rows have rank 1366. The 5 dependent rows
are the pilot rows of cells 459, 195, 924 and 988 (the cells around the subject, which the tree cuts into evaluated pieces;
each cell's row is then the n-weighted sum of its pieces' rows) and one large leaf-mean row. The same run with independent
noise added to every evaluation:

| noise per evaluation (units of sqrt V) | result |
|---|---|
| 0 (the bed) | runs, K 179 |
| 1e-6 (about the forward-difference noise of a full-data evaluation) | interpolant assert: residual 3.2e-6 against 1.7e-7 |
| 2.5e-3 (eps/4, the design's admissible quantization error) | interpolant assert: residual 8.2e-3 against 1.7e-7 |

So the method's step 2, as built, needs exact evaluations. A quantized R_S (or even a full-data forward difference) is not
exact, and the run fails as soon as the tree cuts a cell into evaluated pieces, which on shearmix it always does around the
subject. A calibration bound delta <= eps/4 does not help: the failure is at 1e-6. The fix is a change to the method (how
psi-hat treats a row that depends on earlier rows), which the design excludes ("nothing else in the method changes"). Options
for the author: (a) a new row found to lie in the span of the active rows (its Schur pivot is at the jitter; the row space is
already computed for the determined-leaf test) does not enter psi-hat; its value is kept as a check of consistency; (b) the
superseded row leaves instead (the cell's pilot row once its pieces cover it, a split leaf's test row once its children cover
it); (c) psi-hat becomes a jittered fit with the jitter set by the evaluation error (no longer an exact interpolant; against
the author's ruling). The author chose (a); see "Redundant rows".

## Redundant rows (option a, implemented 2026-11-06)

**The rule.** A new evaluation row whose weight pattern lies in the span of the active rows (squared projection residual
relative to the row's squared norm <= 1e-8, the determined-leaf test's threshold) does not enter psi-hat. Its value is kept:
after the next fit, value - row . psi-hat is recorded, and `MethodResult.eval_error` reports the count by kind and the max
and rms per output, in units of the output's scale. Such an evaluation is still counted in E. The tree's own rule (a parent's
row leaves when its children's rows enter) is unchanged. A redundant test row is not in psi-hat, so its value less the
implied value is already a cross-fitted misfit; it is used as one, and psi-tilde equals psi-hat on that leaf.

**Code.** `core/span.py` (and an identical copy, `bed/span.py`) keeps the span: the pilot rows implicitly (the cell
indicators), every other active row as its residual after the cell projection, with a Cholesky factor of their Euclidean
Gram matrix. Appending a row is one triangular solve; a leaving row is removed by a rank-one update. In `method.py` and
`bed/method.py` the changes are the same: `_Rows` keeps inactive rows, `enter()` tests each new row, and `resolve()` measures
redundant rows after each fit. A sibling's column is built as a combination of stored columns when its recipe (parent and
smaller child, recursively through redundant rows) reaches stored rows, and computed directly otherwise. When a smaller child
is redundant, its sibling takes the parent's position. Both children of an active row cannot be redundant; this is asserted.
`bed/estimators.py` and `core/interpolant.py` are unchanged. Bed backups: the scratchpad's `bed_backup/`.

**Acceptance 1** (exact evaluator, the bed's cells, k = 16, scale 1, against the bed's final record):

| | result |
|---|---|
| splits per round, n_taken, E per round, K 179, rounds 16, leaf sizes | identical |
| Vhat, max relative difference | 3.5e-10 |
| U, max relative difference | 5.1e-7 (above the 1e-9 tolerance) |
| delta | 1.1e-15 |
| redundant rows | 14: 13 siblings (the root's larger child is always redundant: it is the pilot rows minus the smaller child), 1 test row |
| relative squared residual | redundant rows at most 2.4e-16; rows that entered at least 3.6e-2 (a clean gap) |
| eval_error, exact evaluator | max 1.2e-7 .. 8.5e-7, rms 4e-8 .. 2.6e-7 (the interpolant's reproduction tolerance) |

Cause of the U difference: the bed record solved its dependent rows through the jitter. The unmodified bed, with only the
jitter changed, moves U by 6.2e-8 (jitter 1e-12) and 7.0e-6 (jitter 1e-9), so the record's U is itself known only to that
level. The modified bed reproduces the package's numbers exactly (U 5.14e-7 and Vhat 3.5e-10 from the record, the same as the
package): the two are still one method.

**bed/method_check.py A B C** (vqboot/fable, the usual env), before and after: line for line the same rounds, stops,
struct-diff counts and max relative differences. Both before and after, A and B fail against the stored reference records,
because those records predate the final code (the cut rule changed the partitions; METHOD_qij.md section 11). C passes: the
leave-one-out identity is checked against an explicit refit, combination columns to 1.6e-13, the factor to 1.8e-14. Fits:
94 and 75, against 97 and 78 before (no refit when every new test row of a round is redundant).

**Noise test** (`scripts/dependent_rows_check.py`, log beside it): the bed's case with independent noise on every evaluation
now completes.

| noise (sqrt V) | K | E | rounds | redundant | eval_error max | true R / delta, max | Vhat - (V - R), max abs |
|---|---|---|---|---|---|---|---|
| 0 | 179 | 1377 | 16 | 14 | 8.5e-7 | 0.094 | 0 |
| 1e-6 | 179 | 1377 | 16 | 14 | 1.2e-4 | 0.094 | 3e-6 |
| 2.5e-3 | 261 | 1556 | 58 | 60 (42 siblings, 17 tests, 1 leaf mean) | 1.34 | 0.14 | 6.6e-3 |

At 2.5e-3 the run is valid by its criterion (R <= delta on every output, R <= U on every output). Vhat is off by up to
6.6e-3, which is more than 2 x 2.5e-3: the report's evaluations carry the noise directly. The design's quantization bound
[Vhat (1 - 2 delta), Vhat (1 + 2 delta) + U] still holds on all 7 outputs. (The R <= U column is False at noise 0 and 1e-6:
U is the one-sided 84 % limit, and the bed's own record has R above U on 2 of 7 outputs at this stop.)

**Finding: noise amplification through the siblings.** A sibling's value is (n_p m_p - n_a m_a)/n_b. Along a chain of
siblings from the root, the error telescopes to about (N / n_leaf) times the evaluation error: about 500 x for a 20-point
leaf, which is the eval_error max of 1.34 at 2.5e-3. This holds whether the sibling's row is redundant or not. With exact
evaluations it is harmless; with quantized evaluations it multiplies the calibrated error on small leaves.

## Quantized evaluations (part 2): not built; the specification is circular

The calibration criterion is delta_j = max over pairs abs(quantized - full) / scale_j <= eps/4 with scale_j = sqrt(B) "from the
pilot". But the pilot's own M evaluations are, by the design, quantized evaluations on R_S (S = the cell), which need the
calibrated R first. So the order pilot -> scale -> calibration -> R -> pilot is circular. Options (not chosen by me):
(1) pilot on the initial R, take sqrt(B), calibrate, and redo the pilot on the final R if the calibration refined R. This
costs up to M extra evaluations, and the first pilot's sqrt(B) carries exactly the error the calibration then measures (on
shearmix the initial point-mass pilot was off by 0.14 to 1.33 V in sum_c p_c d^2); (2) calibrate first with a unit taken
from the k + 1 full-data calibration evaluations (biased high: they are the largest-mismatch cells), then the pilot on the
final R gives sqrt(B) for the ranking and the cut; (3) another unit for the calibration only (for example sqrt(F) of the
full-data base fit, if the statistic offers one). Everything else in the design is buildable as written: the per-point losses
exist for the three statistics (shearmix: minus log sum exp of `shearmix_model.log_joint`; mvt/all: the t log density in r^2
plus the Bernoulli log likelihood of the tail indicator; fp: the squared orthogonal distance to the fitted plane); adding them
means touching `cloudfil_u.py` and `estimators.py`.
