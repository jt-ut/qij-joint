# What qij_joint keeps: rulings sheet

25 September 2026. Built from the inventory of `qij@025613e` (details and
line numbers in `spec/inventory/inventory_C1..C4.md`; traced lines in
`spec/inventory/executed_lines.json`). Everything the paper's main path
executes is carried by default and is not listed here. Listed: everything
that needs a decision. **Rec** is the coordinator's recommendation; the
author rules. Mark each line K (keep), D (drop), or write a note.

The trace: six main cases at N = 2000 (draws 0, 1), the only failed draws
in the whole main study (MVT nu, s = 342 and 420; the third is 523; no
case ever failed a full-data fit), the one check, one GMM fit. 2,282
executed lines of 10,587.

## A. Code paths the paper never runs

| # | Item | Size | Rec |
|---|---|---|---|
| A1 | FITC sparse influence model (`influence_model='fitc'`, `fitc_rank`, the dispatcher, 8 functions, delegation branches) | ~770 lines, 37% of `influence_model.py` | D |
| A2 | Two-regime IMF (already ruled) | 596 lines; Chabrier shares only two constants and the box-rule helper | D (ruled) |
| A3 | `QIJ(eta=...)` override; every run uses the estimator's own `T.eta` | 4 lines | D |
| A4 | Result objects' `.variance`, `.interval()`, `.summary()` and `core/intervals.py` — never called by the pipeline, but they are the user-facing answer of `QIJ.fit` / `Bootstrap.fit` | ~110 lines | K `.variance` and `.interval()` (the user surface, ~25 lines); D `.summary()` |
| A5 | `estimators.weighted_mean` + its influence, `supports_for` (the check defines its own weighted mean) | ~30 lines | D |
| A6 | One-sided difference stencil (only when eta > 1/3; every estimator has eta ≤ 1e-6) | 5 lines | D |
| A7 | Structurally dead shortcuts: `kmeans_1d` M == 1, `bin_differences` M_used == 1 | 5 lines | D |
| A8 | GMM `reg_covar` ridge option (default 0, never used) and the `max_iter ≤ 25` short-circuit | ~15 lines | D |
| A9 | GMM `fit_and_influence` (never called) | 35 lines | K, and `ij` uses it (one fit, not two) |
| A10 | `influence_model` `denom_m <= 0` fallback: looks mathematically unreachable given the rule before it | 2 lines | D, after the build agent confirms the algebra |
| A11 | Multi-N directory logic, YAML configs, h5 resume I/O, `run_study` | replaced by the settled design | D (ruled) |

## B. Data-triggered failure handling (not hit in the trace; the accepted failure rules)

Stage-2 initial-bin evaluation failure (`_failed_result`, `bin_differences`
NaN branch); stage-1 collapse (hit by MVT nu); a failed prototype
(mass-centering over the finite ones); Cholesky jitter escalation;
degenerate length-scale bounds; the noise-floor upper-edge pin; the width
grid's lower endpoint; `rho` non-finite; `_resolve_bmu2`'s repair loop.
**Rec: K all** — they are the rulings of 18 September, and a clean package
still meets degenerate data.

## C. What the method computes beyond the interval

The interval is the normal interval on V_btw. These are computed alongside:

| # | Quantity | Paper read it? | Rec |
|---|---|---|---|
| C1 | V_win_hat, V_tot_hat | V_tot_hat yes | K (the declared tolerance's evidence; cheap) |
| C2 | rho and the refined field psi_hat | psi_hat yes (Figure B) | K |
| C3 | a_bca (skewness acceleration) | yes (T1) | your call: no BCa interval exists any more; K only if you want the skewness diagnostic |
| C4 | B_hat and per-bin d2T (constituents of the rejected second-order interval) | no | D storing them; the stencil still yields d2T for free, keep computing it only if refinement uses it |

## D. Stored fields the paper never read

| # | Fields | Rec |
|---|---|---|
| D1 | per-stage `evals_*`, `rows_*`, `wall_time_*` | K (cost story; the QIJ parallelism design needs the stage split) |
| D2 | `M_X`, `n_level_splits`, `n_adjacency_splits`, `n_refine_evals`, `gain_ratio` | K (one scalar each; describe what the refinement did) |
| D3 | `ell`, `lam`, `ell_bound`, `lam_bound` | K (influence-model health) |
| D4 | per-bin list columns `bin_mass`, `bin_influence`, `bin_d2T` | D (`bin_d2T` per C4); `bin_mass` and `bin_influence` K only if you want the bin-level record |
| D5 | partition product (whole thing; X-VQ vs psi0-quantile vs true-psi partitions over an M grid) | D from the first build; it is recomputable later from `ij` psi and `qij` points |
| D6 | points `psi0`, `sigma`, `bin_label`; the prototypes product | K for `--diag-draws` (small; the talk demo shows prototypes, receptive fields and bins drawn back on the plane) |
| D7 | `seed` columns | K (identity column of every row) |

## E. Efficiency (standards E1–E9)

No change to results:

* E-a. `ij` computes theta and psi from one fit (every `.influence()`
  currently refits; costly for MVT nu, Chabrier and the GMM).
* E-b. The double full-data evaluation per draw disappears with the method
  split.
* E-c. `fp_sdss.npz` loaded once, not twice.
* E-d. Vectorize the step-parameter list comprehension over prototypes
  (`influence_model.py:829`); elementwise, same bits.
* E-e. Cache each leaf's Var(psi0) and mean once in refinement instead of
  recomputing (`refine.py` `_variance`, `leaf_ubar`).

Changes floating-point order, so needs your approval under E8 (with a
measured saving first):

* E-f. `ivq.within_share` per-group loop → two `bincount` calls.
* E-g. `bin_posterior_variance` per-bin loop → one grouped pass.
* E-h. `sum_p_ubar2` maintained incrementally instead of re-summed.
  **Rec: measure all three on one IMF and one FP draw; approve only those
  that save more than a few percent of the draw.**

A design question, not a cleanup:

* E-i. Estimators recompute weight-independent quantities on every call
  (log-ratios, `r_sq` pieces, the GMM's fixed bases), 2,000 times per
  bootstrap draw. Fix: an optional `T.prepare(X)` whose result every call
  reuses. Same arithmetic, same bits, but it changes the estimator
  interface. **Rec: yes, optional, so a user's plain `T(X, w)` still works.**

Standards amendment:

* E-j. E5 bans module-level caches. The data loaders (`_fp_pool`,
  `_imf_pool`, `truth`) cache constant data read from the package's own
  files, which is deterministic and harmless in every process.
  **Rec: allow exactly that exception in E5.**

## F. Comments and stale references (no decision; noted)

History-carrying comments counted in every file (worst: `refine.py`'s
168-line module docstring). All are rewritten under C1–C3 during the port;
derivations move to the spec. Stale references disappear with the port:
`qij2_intervals_closed_form` (removed function still cited), `qij_interval`'s
removed `support` argument, three docstrings claiming Chabrier is "not
wired into the study".
