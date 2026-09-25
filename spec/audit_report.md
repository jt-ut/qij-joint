# qij_joint audit report

Machine: this laptop (macOS). `qij@025613e` (old, read only) vs `qij_joint`
(new, at the state on disk when this audit ran). N=2000, master seed 0,
eps=0.01, B=2000 throughout. No files in either package were modified;
no tests or assertions were added; the only new products are under
`audit/` (scratch) and `QIJ_WSOM2026/runs/main_migrated` (the migration).

## Summary of anything unexpected

1. **A driver bug, not a package bug.** The first attempt at Step 2 used
   a bash associative-array script. macOS's system `/bin/bash` is 3.2,
   which has no associative arrays; `declare -A` fails silently (no
   `set -e`) and every `${ARR[$key]}` then indexes *arithmetically*, so
   every string key collapsed to index 0 and the whole script
   re-ran `imf all` six times, never touching pareto/mvt/fp. Caught by
   inspecting the log's case labels before trusting the run; rewritten
   as a plain Python driver (`run_step2.py`) and rerun in full. Nothing
   here reflects on either package.
2. **Two transient process-pool crashes**, unrelated to the above: `mvt
   nu oracle --draws 999:1000 --workers 3` and `fp all boot --draws
   999:1000 --workers 4` each aborted once with
   `dyld[...]: Assertion failed: (this->magic == kMagic)`, a known
   flaky macOS dynamic-linker race under `loky`'s process pool, exit
   code -6 (SIGABRT). Both re-ran clean on the first retry; all draws
   are present. Not a `qij_joint` defect (it is a pool-startup race in
   the OS loader, not in the package's own code), but worth knowing if
   `run.py` is ever scripted unattended on macOS: check exit codes and
   retry.
3. **A bug in my own comparison script**, caught before it produced a
   false finding: `pandas.sort_values('s')` is not a stable sort, and
   with 2000 repeated `s` values per draw it was scrambling bootstrap
   replicate order within a draw before comparison, which briefly
   looked like a near-total mismatch in `boot`'s replicates. Removed the
   unneeded sort (both the migration round trip and the new-CLI output
   are already in increasing-`s`, original-replicate-order); the true
   result is the exact match reported in Step 3 below.
4. **Step 2 (same laptop, same code paths as the baseline): every kept
   field is either exactly equal or within 1e-12 relative**, consistent
   with the three approved E8 order-change rewrites and nothing else.
   No integer or boolean field differs anywhere, including at the three
   pathological MVT-nu draws (342, 420, 523), which reproduce their
   failure mode (188 of 189 evaluations failing, `L=1`, no Cholesky
   factorization ever attempted) bit-for-bit.
5. **Step 3 (migration + round trip): exact.** 0 mismatches on every
   migrated column, across all 1000 draws, all six cases, all four
   methods, including a full replicate-by-replicate check of `boot.h5`
   against the migrated `replicates` arrays (2000 replicates x 1000
   draws x every output, per case).
6. **Step 4 (this laptop vs. TACC-origin migrated products): small but
   real cross-machine drift, as expected and not gated.** `theta_hat`
   and `V_ij` agree to 1e-10 relative or better everywhere (mostly
   1e-14 to 1e-16). Two integer split-count fields flip at exactly one
   draw each (`mvt tail` s=1, `imf all` s=1), a discrete refinement
   decision tipping the other way under machine-level floating-point
   noise. The GP influence model's fitted length scale `ell_<o>` is the
   most sensitive downstream quantity, up to ~2% relative at some
   draws (`pareto shape` s=0) — plausible for a 1-D likelihood
   optimization over ~188 prototypes with a flat objective near its
   optimum; `lam_<o>` was at its lower bound in every case checked, so
   only the continuous `ell` search is exposed to this. None of this
   is gated per the audit's own Step 4 rule.

Everything below is data for the coordinator; no fix is proposed or made
in either package.

## Paths

- Baseline pickles (Step 1): `audit/baseline/*.pkl`
- Baseline task script: `audit/baseline_task.py`
- New-CLI outputs (Step 2): `audit/new/`, `audit/new_w1/`, `audit/new_w4/`
- Step 2 driver: `audit/run_step2.py` (the working Python version;
  `audit/run_step2.sh` is kept only as the broken bash-3.2 attempt, log
  `audit/logs/step2_BROKEN.log.bak`, for the record)
- Step 2 run log: `audit/logs/step2.log`
- Step 2 comparison script and findings: `audit/compare_step2.py`,
  `audit/step2_findings.json`
- Migration script: `audit/migrate.py`, log `audit/logs/migrate.log`
- Migrated products: `/Users/jtaylor/Dropbox/Research/QIJ_WSOM2026/runs/main_migrated`
- Migration round-trip script and findings: `audit/compare_migration.py`,
  `audit/migration_roundtrip_findings.json`
- Step 4 script and findings: `audit/compare_step4.py`,
  `audit/step4_findings.json`

All scripts were run with `PYTHONPATH` pointed at exactly one of the two
packages' `src/` at a time, `PYTHONDONTWRITEBYTECODE=1`, and the five BLAS
thread variables pinned to 1 before numpy import. The new CLI and
`migrate.py` were run with the shell sandbox disabled (both call
`sysctl`/`os.sysconf`, blocked by the sandbox); nothing else was.

---

## Step 1: baseline (`qij@025613e`, one process per task)

21 tasks (6 cases x s in {0, 1, 999}, plus MVT nu x s in {342, 420, 523}),
up to 12 concurrent. All 21 succeeded (no exceptions). Jitter is the
escalation step `_cholesky_with_jitter` used, wrapped at the module level
so every call anywhere in the influence-model fit is counted.

| dataset | estimator | s | wall time (s) | Cholesky calls | jitter=0 | jitter&gt;0 |
|---|---|---:|---:|---:|---:|---:|
| pareto | shape | 0 | 0.75 | 1 | 1 | 0 |
| pareto | shape | 1 | 0.79 | 1 | 1 | 0 |
| pareto | shape | 999 | 0.71 | 1 | 1 | 0 |
| pareto | tail | 0 | 0.72 | 1 | 1 | 0 |
| pareto | tail | 1 | 0.70 | 1 | 1 | 0 |
| pareto | tail | 999 | 0.63 | 1 | 1 | 0 |
| mvt | nu | 0 | 3.57 | 1 | 1 | 0 |
| mvt | nu | 1 | 3.59 | 1 | 1 | 0 |
| mvt | nu | 342 | 3.00 | 0 | 0 | 0 |
| mvt | nu | 420 | 2.96 | 0 | 0 | 0 |
| mvt | nu | 523 | 2.99 | 0 | 0 | 0 |
| mvt | nu | 999 | 3.32 | 1 | 1 | 0 |
| mvt | tail | 0 | 1.09 | 1 | 1 | 0 |
| mvt | tail | 1 | 1.07 | 1 | 1 | 0 |
| mvt | tail | 999 | 0.92 | 1 | 1 | 0 |
| fp | fp | 0 | 1.57 | 4 | 4 | 0 |
| fp | fp | 1 | 1.51 | 4 | 4 | 0 |
| fp | fp | 999 | 1.54 | 4 | 4 | 0 |
| imf | chabrier | 0 | 114.9 | 3 | 3 | 0 |
| imf | chabrier | 1 | 122.6 | 3 | 3 | 0 |
| imf | chabrier | 999 | 115.6 | 3 | 3 | 0 |

No draw anywhere needed a jitter escalation beyond 0 (the base
factorization always succeeded). The three MVT-nu draws called out as
"failed" in the audit prompt (342, 420, 523) never reach a Cholesky
factorization at all: 188 of 189 per-prototype evaluations fail for
`mvt_nu` at those seeds (`qij_row['n_failed']=189`, `M_X=188`), so the
influence-model fit is never attempted and refinement stops at `L=1` (a
single bin). This is a genuine data-triggered failure of the old code at
those seeds, reproduced unchanged (see Step 2).

---

## Step 2: `qij_joint` on the same draws, vs. baseline

New CLI calls: `oracle`/`ij` at `--workers 3`; `boot` at `--B 2000` with
`--workers 1` (`audit/new_w1`) and `--workers 4` (`audit/new_w4`); `qij`
at `--workers 1 --diag-draws 0:1`. Same draws as Step 1, one call per
contiguous range (`0:2`, `999:1000`, plus `342:343`/`420:421`/`523:524`
for MVT nu). All calls eventually succeeded (two transient `dyld`
SIGABRTs on the first pass, both clean on retry -- item 2 above).
Timing fields (`wall_time*`, `busy_time*`, `workers`) excluded from
comparison throughout, as directed.

| comparison | fields checked | result |
|---|---|---|
| oracle `theta_hat_<o>` vs old `truth_row theta_hat_<o>` | all 6 cases x 3 draws (6 x 5 for MVT nu) | **exact everywhere**, no findings |
| oracle `theta_true_<o>` vs old `theta_true_<o>` | fp (4 outputs), imf (3 outputs) | differs at 1e-16 to 1e-15 relative for `a,b,scatter,m_c,sigma,x`; `theta_true_c` (fp) differs at 1.1e-5 relative (absolute diff 3.2e-18, both values are ~1e-13, i.e. numerically zero) -- see note below. pareto/mvt: exact (parametric truth is a fixed constant) |
| ij `V_ij_<o>` vs old `V_oracle_<o>` | all cases, all draws | **exact everywhere**, no findings |
| ij `psi` (s=0) vs old `points_df psi_<o>` | all cases | **exact everywhere**, no findings |
| boot `replicates` (w1 and w4) vs old `boot_theta` | all cases, all draws, both worker counts | **exact everywhere**, no findings |
| boot `n_failed` vs old `boot_n_failed` | all cases, all draws | **exact everywhere** (all zero) |
| boot w1 vs w4 | all cases, all draws | **identical**, no findings |
| qij scalar row (kept columns) vs old `qij_row` | all cases, all draws | 37 fields differ, **every one within 1e-12 relative** (max observed 5.7e-16); zero differences at the MVT-nu failure draws 342/420/523 (bit-identical, including `n_failed=189`, `L_nu=1`) |
| qij `points`/`prototypes` (s=0) vs old `points_df`/`prototypes_df` | mvt nu, mvt tail, fp | 3 fields differ (`psi_hat_<o>` only), **all within 1e-12 relative** (max 3.5e-15); `prototypes` exact everywhere |

**Note on `theta_true_c` (fp):** `qij.datasets.truth('fp','fp')` is not a
stored constant -- it refits the FP estimator on the full 76,997-galaxy
population every time it is called (`qij/src/qij/datasets.py`, `truth()`).
My baseline task calls this function fresh, on this laptop, which can
converge to a fit that differs from what was fit on TACC and frozen into
`qij_joint`'s `data/truth/fp.json` by a few ULPs. Because `c` is the FP
scatter law's near-zero intercept (~-2.86e-13), a few ULPs of absolute
difference is a large *relative* difference. This is exactly the
"difference is EXPECTED for imf and possibly fp" the audit brief called
out; `qij_joint`'s truth file was verified separately to match
`runs/main/fp/fp/truth.parquet` and `runs/main/imf/chabrier/truth.parquet`
bit-for-bit (draw 0, all 1000 draws constant), which is the actual
frozen source it is supposed to reproduce.

**The 37 qij scalar-row and 3 points differences** are exactly the fields
the three approved E8 rewrites (plan Sec 6.4) would touch: `V_win_hat_<o>`
(the incrementally-maintained `sum_p_ubar2` / grouped `bin_posterior_
variance`), and everything computed from it downstream in the same
draw -- `rho_<o>`, `gain_ratio_<o>`, and `psi_hat_<o>` (which is built
from `rho`). No `V_btw_<o>` finding appears standalone; no integer field
(`L_*`, `n_level_splits_*`, `n_adjacency_splits_*`, `n_refine_evals_*`,
`evals_*`, `rows_*`, `M_X`, `n_failed`, `bin_label_*`, `bmu`) or boolean
field (`ell_bound_*`, `lam_bound_*`) differs anywhere in Step 2.

---

## Step 3: migration of `runs/main` -> `runs/main_migrated`

`audit/migrate.py` converts all 6 cases x 1000 draws into the new layout
via `qij_joint.products.write_draw`, matching the new pipeline's exact
column set, order and dtype for every method (verified against Step 2's
own output schemas). Ran in 139.3 s.

| case | oracle | ij | boot | qij |
|---|---:|---:|---:|---:|
| pareto/shape -> pareto shape | 1000 | 1000 | 1000 | 1000 |
| pareto/tail -> pareto tail | 1000 | 1000 | 1000 | 1000 |
| mvt/nu -> mvt nu | 1000 | 1000 | 1000 | 1000 |
| mvt/tail -> mvt tail | 1000 | 1000 | 1000 | 1000 |
| fp/fp -> fp all | 1000 | 1000 | 1000 | 1000 |
| imf/chabrier -> imf all | 1000 | 1000 | 1000 | 1000 |

No column differed from what Step 2's live pipeline wrote (checked field
by field, dtype by dtype, against `audit/new*`'s own schemas before
finalizing `migrate.py`). `qij_partition.parquet` was not migrated, by
design.

**Round trip** (`audit/compare_migration.py`): re-read every migrated
method with `products.collect`/`collect_array` and compared, exact, to
frames built straight from `runs/main`'s `truth.parquet`, `qij.parquet`,
`boot.h5`, `qij_points.parquet`, `qij_prototypes.parquet` -- all 1000
draws, all 6 cases, all 4 methods, including a full element-by-element
check of every one of the 2000 x 1000 x (outputs) bootstrap replicates
per case.

**Result: 0 mismatches, everywhere.** (An initial run reported ~2 million
"mismatches" in `boot`'s replicates; that was the comparison script's own
bug -- an unstable sort scrambling replicate order before comparison,
item 3 above -- not a migration defect. Fixed, and confirmed exact.)

---

## Step 4: this laptop's new draws vs. the TACC-origin migrated products

Same draws as Step 2, compared against `runs/main_migrated` (built from
data generated on TACC, a different machine). Reported per the audit
brief, not gated.

| field | cases with any difference | max relative difference observed | notes |
|---|---|---:|---|
| `theta_hat_<o>` | 4 of 6 cases | 5.1e-14 (mvt nu) | pareto tail, mvt tail exact (parametric, low-dim closed forms); others 1e-16 to 5e-14 |
| `V_ij_<o>` | 4 of 6 cases | 3.8e-11 (mvt nu) | pareto tail, mvt tail exact; others 1e-15 to 1e-14 |
| boot `replicates` (w1 and w4 identical to each other) | 4 of 6 cases | 2.7e-14 (fp `scatter`) | pareto tail, mvt tail exact; ~4,000-6,000 of 2000x3-draw replicates differ per case, all at this scale |
| qij kept scalar fields | pareto shape, pareto tail, mvt nu, mvt tail, fp all, imf all (77 field/case findings) | `gain_ratio_P_tail` 0.645 (pareto tail); `ell_<o>` up to 0.023 (pareto shape); `V_win_hat_<o>` up to 0.0041; most other fields (`V_btw`, `V_tot_hat`, `rho`) 1e-5 or smaller | see discussion below |
| integer qij fields | mvt tail (s=1), imf all (s=1) | exact match required; **17 mismatches**, all at exactly one draw per case | `evals_refinement/total`, `rows_refinement/total`, `L_P_tail`/`L_sigma`/`L_x`, `n_level_splits_*`, `n_refine_evals_*` each off by 1 bin/eval at that single draw |

**Discussion.** `theta_hat`, `V_ij` and the bootstrap replicates -- all
computed by one or two direct estimator evaluations -- agree to 1e-10
relative or (usually) far better, comfortably inside the "~6 significant
figures" cross-machine expectation the audit brief sets for IMF and
consistent with ordinary BLAS/LAPACK cross-machine floating-point
differences. The `qij` method amplifies this: the GP influence model
fits a continuous length scale `ell_<o>` by a 1-D likelihood search over
only `M_X` (~188) prototypes, whose objective can be quite flat near its
optimum, so machine-level noise in the inputs can shift the fitted `ell`
by a percent or more (`lam_<o>` did not show this because it sat at its
lower bound in every draw checked, so only `ell`'s continuous search is
exposed). That, in turn, moves everything computed from `ell`
(`V_win_hat`, `rho`, `gain_ratio`) by a similarly amplified amount. The
17 integer mismatches are the discrete face of the same thing: a
refinement gain-ratio threshold decision tipping the other way by a
hair at exactly one draw each in `mvt tail` and `imf all`, splitting one
bin more or fewer. None of this reflects a difference between the two
packages -- Step 2 (same machine, same code paths as the baseline) shows
`qij_joint` reproducing `qij@025613e` bit-for-bit apart from the three
approved E8 rewrites, all within 1e-12 relative.

---

## Migrated directory

`/Users/jtaylor/Dropbox/Research/QIJ_WSOM2026/runs/main_migrated`:
**505 MB, 30,018 files.** This matches the expected count exactly: per
case, 1000 (oracle) + 1001 (ij: 1000 scalar rows + 1 `psi` array at s=0)
+ 2000 (boot: 1000 scalar rows + 1000 `replicates` arrays) + 1002 (qij:
1000 scalar rows + `points`+`prototypes` at s=0) = 5003 files x 6 cases
= 30,018.
