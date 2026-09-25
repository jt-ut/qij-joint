# Inventory C1 — `src/qij/core/influence_model.py` (2078 lines)

Repo: `/Users/jtaylor/Dropbox/Software/JT_Py_Pkgs/qij` @ `025613e`. Trace source:
`<scratch>/inventory/executed_lines.json` and `executed_by_task.json` (16 tasks,
no exceptions — see `trace_notes.md`), cross-checked line-by-line against the
static code below. Main-path settings per the coordinator brief: `influence_model
= 'gp'`, `fitc_rank = None`, eps 0.01, N 2000, B 20 (trace)/2000 (production);
datasets/estimators pareto (shape, tail), mvt (nu, tail), fp (fp), imf
(chabrier). q (output count): 1 for shape/tail/nu, 3 for chabrier (m_c, sigma,
x), 4 for fp (a, b, c, scatter). d_z (Z's whitened dimension): 1 for pareto and
imf (X is scalar), 10 for mvt, 3 for fp.

One important trace artifact, noted once here rather than per row: every FITC
function's `def` line shows up as "executed" in `executed_lines.json` because
Python's tracer records the line that defines a function object at *import*
time, even when the function body is never called. Checked per-task in
`executed_by_task.json`: **all 16 tasks** hit exactly the same 8 `def` lines
(1447, 1481, 1557, 1582, 1590, 1912, 1943, 1985) and nothing else in those
functions' bodies. So every FITC function is classified `SWITCH`, not `MAIN`,
despite its `def` line appearing in the executed set.

## Table 1 — items

| item | file:lines | what it does | status | lines |
|---|---|---|---|---|
| module `qij.core.influence_model` | 1-2078 | Fits and queries the initial influence estimate: a per-coordinate GP (default) or FITC sparse alternative, affine mean, Matérn-3/2 kernel, declared noise floor, one cached per-draw posterior pass | MAIN | whole file; GP path MAIN, FITC path SWITCH (rows below) |
| `_SQRT3` | 197 | sqrt(3) factor in the Matérn-3/2 formula | MAIN | 197 |
| `_LOG_LAM_LO`, `_LOG_LAM_HI` | 198-199 | Absolute search bounds for log(lam): [1e-10, 1e2] | MAIN | 198-199 |
| `_N_WIDTH_GRID` | 200 | Number of log-spaced ell candidates (5) in the outer width-search grid | MAIN | 200 |
| `_UNCERTAINTY_BATCH_CAP` | 201 | Row-batch size (4096) for the psi0/uncertainty query pass over Z | MAIN | 201 |
| `_BPV_CHUNK` | 202 | Chunk size (2048) for `bin_posterior_variance`'s bin-summed and pairwise loops | MAIN | 202 |
| `_SELF_SUM_LOG_RANGE` | 203-206 | Max `\|c*(x-o)\|` span (200) per block in `_matern32_self_sum_1d`'s running-sum recursion | MAIN | 203-206 |
| `_FITC_D0_FLOOR` | 207 | Floor (1e-6) on FITC's D0 diagonal, avoiding a Woodbury blow-up at inducing points | SWITCH — `influence_model='fitc'` only | 207 |
| `_KERNEL_CACHE_BYTES` | 209-220 | Memory budget (256 MB) gating whether the per-draw kernel-row cache is kept | MAIN | 209-220 |
| `_PointTerms` (dataclass) | 223-258 | Cached pass of psi0/sigma/residuals/kernel rows over one Z, keyed on Z's identity | MAIN | 223-258 |
| `InfluenceModel` (dataclass) | 261-321 | Fitted full-rank GP model: per-coordinate design, width/lam/jitter, Cholesky factors, shared eigendecomposition | MAIN | 261-321 |
| `FITCInfluenceModel` (dataclass) | 324-418 | Fitted FITC model: rank-r objects (inducing points, W, E, K_mm/B factors) replacing every M_c-scale object | SWITCH — `influence_model='fitc'`; own docstring at line 1789 notes it is "built and committed but not adopted" | 324-418 |
| `_whitening_from` | 421-437 | PCA whitening `(mean, transform)` of Z | MAIN | 421-437 |
| `_matern32` | 440-443 | Matérn-3/2 kernel value at distance r, width ell | MAIN | 440-443 |
| `_matern32_self_sum_1d` (main body) | 446-524 | Sum of pairwise kernel values within one 1-D bin via O(n log n) running sums, blocked to avoid overflow | MAIN | 446-524 minus row below |
| `_matern32_self_sum_1d`: n<=1 early return | 483-484 | Degenerate bin of 0 or 1 point | DATA — trigger: a refinement bin with <=1 point | 484 |
| `_basis` (affine branch) | 527-534 | h(x) = (1, x) affine basis in whitened coordinates | MAIN | 527-534 minus row below |
| `_basis`: constant-only branch | 532-533 | Falls back to h(x) = 1 when the design is too small for an affine basis (M_g <= d_z+2) | DATA — never triggered at N=2000 in the traced draws; trigger: a coordinate's finite design shrinks to <= d_z+2 points | 533 |
| `_length_scale_bounds` (CONN-based bounds) | 537-569 | ell_min = median CONN-connected whitened distance; ell_max = 10x the largest inter-prototype distance | MAIN | 537-559, 565-566, 569 |
| `_length_scale_bounds`: no-CONN-edge fallback | 560-561 | ell_min = smallest positive inter-prototype distance when no CONN edge has positive length | DATA — trigger: CONN graph has no positive-length edges | 560-561 |
| `_length_scale_bounds`: fully-degenerate fallback | 562-563 | ell_min = 1.0 when every prototype coincides | DATA — trigger: all prototypes at the same point | 563 |
| `_length_scale_bounds`: ell_max<=ell_min fallback | 566-567 | Forces ell_max = 10x ell_min | DATA — trigger: degenerate distance spread | 567 |
| `_cholesky_with_jitter` (j=0 success) | 572-586 | One Cholesky of A, no jitter needed | MAIN | 572-586 |
| `_cholesky_with_jitter`: jitter escalation | 583-588 | Escalates jitter x10, x100 on `LinAlgError` | DATA — trigger: A ill-conditioned at zero jitter | 587-588 |
| `_cholesky_with_jitter`: final failure | 589-591 | Raises `RuntimeError` after 3 jitter levels fail; the study records the draw as failed | DATA — trigger: A still not PD after every escalation | 589-591 |
| `_within_1pct_log` (comparison) | 594-596, 598-600 | True when x is within 1% in log of either bound | MAIN | 594-596, 598-600 |
| `_within_1pct_log`: invalid-x guard | 596-597 | Returns False when x is non-finite or non-positive | DATA — defensive guard, not observed to trigger | 597 |
| `_floor_equation` | 603-609 | lam·s_c²(lam), the profiled closed-form quantity, as a function of log(lam) | MAIN | 603-609 |
| `_lambda_floor` (root-find + lower-edge pin) | 612-629, 631-632 | Declared-noise floor lam_floor,c via one `brentq` call, or pinned to the lower edge (1e-10) when already inactive | MAIN — the lower-edge pin (line 627) fires routinely for closed-form/low-eta estimators (module docstring); the interior `brentq` root (631-632) fires for eps=0.01 estimators | 612-629, 631-632 |
| `_lambda_floor`: upper-edge pin | 629-630 | Pins to the ceiling (1e2) when even the largest lam cannot reach the declared level | DATA — trigger: declared noise level swamps the whole projected response; never triggered in the traced draws | 630 |
| `fit_influence_model` (dispatcher, 'gp') | 635-652 | Dispatches to `_fit_gp_influence_model` | MAIN | 635-652 |
| `fit_influence_model`: 'fitc' branch | 653-654 | Dispatches to `_fit_fitc_influence_model` | SWITCH — `influence_model='fitc'` | 653-654 |
| `fit_influence_model`: invalid-value branch | 655-658 | Raises `ValueError` for any value other than 'gp'/'fitc' | UNREACHED — no caller in `src/qij` or `scripts` passes a value other than 'gp'/'fitc' (only `qij.py`/`study.py` set it, both from the same two-valued option) | 655-658 |
| `_fit_gp_influence_model` (main body) | 661-1034 | Fits the per-coordinate GP: per-coordinate finite design, shared width search per group, per-coordinate final Cholesky solve, shared eigendecomposition, diagnostics | MAIN | 661-1034 minus rows below |
| `_fit_gp_influence_model`: M_c==0 fallback | 751-752 | `psi_bar=0.0` when a coordinate has zero finite prototypes (every prototype evaluation failed for that output) | MAIN — confirmed on the two TACC-style failing draws traced (`mvt/nu` s=342, s=420) | 751-752 |
| `_fit_gp_influence_model`: lam ceiling already reached | 875-880 | Skips the inner lam search when the declared floor alone already pins lam_c at the ceiling | DATA — never triggered in the traced draws | 876-880 |
| `_fit_gp_influence_model`: best width is the grid's lower endpoint | 912-917 | Refinement interval taken from the first two grid points when ell_min is the best grid candidate | DATA — never triggered; the interior case (917) and upper-endpoint-skip case (913) both fired instead | 915-916 |
| `_fit_gp_influence_model`: denom_m<=0 fallback | 964-969 | `s2_c=0.0` when M_g-m_g<=0 | DATA — appears unreachable in practice given the M_c>=3 constant-path threshold and the m_g rule (M_g-m_g >= 2 always); a defensive fallback | 968-969 |
| `_coordinate_groups` | 1037-1050 | Groups non-constant coordinates by shared `centers` object identity | MAIN | 1037-1050 |
| `_point_terms` (GP path, cache hit + miss) | 1053-1177 | The one cached posterior pass (psi0, sigma, residuals, kernel rows) over Z, keyed on Z's identity; both a cache hit (line 1075) and a fresh computation were exercised | MAIN | 1053-1077, 1086-1177 |
| `_point_terms`: FITC delegation | 1077-1085 | Delegates to `_fitc_psi0`/`_fitc_uncertainty` when `model` is `FITCInfluenceModel` | SWITCH — `influence_model='fitc'` | 1078-1085 |
| `psi0` | 1180-1212 | Public accessor: `_point_terms(model, Z).psi0` | MAIN | 1180-1212 |
| `uncertainty` | 1215-1248 | Public accessor: `_point_terms(model, Z).sigma` | MAIN | 1215-1248 |
| `bin_posterior_variance` (GP path) | 1251-1400 | Within-bin posterior variance v_k per refinement bin, via bin-summed vectors and the shared eigen form; both the 1-D running-sum SS_k and the 2-D+ chunked pairwise SS_k were exercised (pareto/imf vs. mvt/fp) | MAIN | 1251-1333, 1335-1338, 1340-1363, 1365-1400 |
| `bin_posterior_variance`: FITC delegation | 1333-1334 | Delegates to `_fitc_bin_posterior_variance` | SWITCH — `influence_model='fitc'` | 1334 |
| `bin_posterior_variance`: constant-coordinate early return | 1338-1339 | Returns all-zero when the coordinate is on the constant path | DATA — never triggered in the traced draws (the failing `mvt/nu` draws' constant coordinate did not reach this call) | 1339 |
| `bin_posterior_variance`: degenerate-bin skip | 1363-1364 | Skips a bin of size <=1 | DATA — trigger: a refinement bin with 0 or 1 point | 1364 |
| FITC section banner/derivation comment | 1403-1444 | Notation and the Woodbury/Sylvester identities the FITC functions below implement | SWITCH | 1403-1444 |
| `_farthest_point_inducing` | 1447-1478 | Deterministic farthest-point selection of r inducing points from a coordinate group's design | SWITCH — `influence_model='fitc'` | 1447-1478 |
| `_fitc_profile` | 1481-1554 | FITC's analogue of `inner`: exact profiled REML criterion at one lam via the Woodbury identity | SWITCH — `influence_model='fitc'` | 1481-1554 |
| `_fitc_lambda_floor` | 1557-1579 | FITC's declared-noise floor via `brentq` on `_fitc_profile`'s s2_hat | SWITCH — `influence_model='fitc'` | 1557-1579 |
| `_fitc_default_rank` | 1582-1587 | Default FITC rank `min(M_c, max(32, ceil(M_c/6)))` when `fitc_rank` is None | SWITCH — `influence_model='fitc'`, `fitc_rank=None` | 1582-1587 |
| `_fit_fitc_influence_model` | 1590-1909 | Fits the FITC model: inducing points, shared outer width search via `_fitc_profile`, per-coordinate low-rank solve | SWITCH — `influence_model='fitc'` | 1590-1909 |
| `_fitc_psi0` | 1912-1940 | FITC's psi0 query, O(N·r) per coordinate | SWITCH — `influence_model='fitc'` | 1912-1940 |
| `_fitc_uncertainty` | 1943-1982 | FITC's uncertainty query, O(N·r²) per coordinate | SWITCH — `influence_model='fitc'` | 1943-1982 |
| `_fitc_bin_posterior_variance` | 1985-2078 | FITC's `bin_posterior_variance`, via bin-summed cross-kernel rows to the inducing points | SWITCH — `influence_model='fitc'` | 1985-2078 |
| OPTION `fit_influence_model(..., influence_model='gp')` | 636-637, 651-658 | Selects the full-rank GP (default) vs. FITC | MAIN value `'gp'` (used by every call in `qij.py`/`study.py`'s `main.yaml`-driven config, which never sets `influence_model`); other value `'fitc'` is SWITCH | see dispatcher rows above |
| OPTION `fit_influence_model(..., fitc_rank=None)` | 636-637 | FITC-only inducing-point-count override; ignored under `'gp'` | SWITCH — only meaningful when `influence_model='fitc'`; main path always passes `None` and it is never read on the GP path | 1590-1591, 1702-1704 |

## Table 2 — standards findings (MAIN and DATA code only)

| file:lines | rule | what |
|---|---|---|
| 829 | E1 | List comprehension `t_g = np.array([step_parameter(delta_f, float(pj)) for pj in p_g])` loops in Python over the group's M_g prototype points, calling `step_parameter` once per point instead of vectorizing over the `p_g` array |
| 829 | E7 | Same line: list-then-`np.array()` conversion rather than a preallocated/vectorized array |
| 1360-1398 | E2 | `bin_posterior_variance`'s `for gi, idx in enumerate(groups):` loop accumulates bin-summed kernel rows (`s_vec`), residuals (`R_vec`) and the raw kernel sum (`SS_k`) one refinement bin at a time, rather than one `np.bincount`/`np.add.reduceat` pass over a point-to-bin label array |
| 1-180 | C1 | 7 history-carrying passages in the module docstring: line 17 "(ruled 18 September; plan §4)"; line 29 "fields that used to be shared across coordinates"; line 63 "(plan §36.2(2), 19 September revision 8)"; line 103 "(20 September, the wall-time refactor)"; line 113 "before this, psi0 and sigma were each computed twice per draw"; line 129 "What used to be q triangular-solve pairs..."; line 165 "it is no longer O(N * M_X_used^2) per coordinate" |
| 262-321 | C1 | 3 history-carrying passages in `InfluenceModel`'s docstring/field comments: line 269-270 "(ruled 18 September; module docstring)"; line 273 "Every field below that used to be a single array..."; line 293 field comment "(module docstring, plan sec 36.2(2))" |
| 661-1034 | C1 | 4 history-carrying passages in `_fit_gp_influence_model`'s docstring/inline comments: line 672 "(module docstring, plan §36.2(2))"; lines 709-710 "(module docstring, plan §36.2(2))"; line 895 "Revision 10 (plan section 36.17(2), 20 September 2026): the bounded refinement is SKIPPED..."; line 1163 "the per-output `cho_solve(chol, Kc.T)` it replaces used to hold" |

Checked and not found in MAIN/DATA code: **E3** (no explicit inverse — every
solve goes through `cho_factor`/`cho_solve`/`solve_triangular`, one factor
reused for every right-hand side, e.g. `chol` reused for both `Hb_g` and
`psi_c` at lines 955/959); **E5** (the module's own history record, lines
102-113, documents the fix already made — one cached `_point_terms` pass per
`Z`, no invariant recomputed per call); **E6** (this module never evaluates
the estimator `T` itself, only queries the fitted GP/FITC posterior, so E6
does not apply). The FITC path's `Kmm_inv = cho_solve(cholKmm, eye_r)` /
`Binv = cho_solve(cholB, eye_r)` (lines 1834-1835) *is* an explicit-inverse
pattern (E3), but it is SWITCH code, out of this table's scope.

## Summary

**Lines by status** (line-range accounting; ~56 of 2078 lines are blank
separators between items and are not assigned a status):

| status | lines | share |
|---|---|---|
| MAIN | ~1226 | 59% |
| DATA | ~25 | 1% |
| SWITCH | ~767 | 37% |
| UNREACHED | 4 | <1% |
| (blank, unassigned) | ~56 | 3% |
| **total** | 2078 | |

SWITCH is almost entirely the FITC section: `FITCInfluenceModel` (95 lines),
the FITC banner + 8 functions (1447-2078, 618 lines), `_FITC_D0_FLOOR` (1
line), the dispatcher's `'fitc'` branch (2 lines), and `_point_terms`'s/
`bin_posterior_variance`'s FITC-delegation lines (9 lines) — 767 lines total,
i.e. this whole alternative influence model (selected only by
`influence_model='fitc'`, never used on the paper's main path) is over a
third of the file.

**Every option, main-path value, and other supported values:**
- `influence_model` (`fit_influence_model` keyword, threaded from
  `QIJ.__init__`/`study.py`'s `config.get('influence_model', 'gp')`):
  main-path value `'gp'`; other supported value `'fitc'` (SWITCH, the whole
  FITC section); any other string raises `ValueError` (UNREACHED).
- `fitc_rank` (`fit_influence_model` keyword): main-path value `None`
  (unread on the `'gp'` path); other supported values: `None` under `'fitc'`
  means "use `_fitc_default_rank` per group", or an explicit int clipped to
  each group's own M_g — both SWITCH, since they only take effect under
  `influence_model='fitc'`.
- `_KERNEL_CACHE_BYTES` (256 MB, module constant, not a call-site option):
  gates whether `_point_terms`/`bin_posterior_variance` cache kernel rows;
  at the paper's N=2000 the cache is always on (confirmed: line 1379, the
  cache-miss re-formation branch, never executed in the trace); it would
  turn off at the cost study's larger N (module docstring, N=76997 example).

**Comment/docstring lines vs. code lines:** the file opens with a 180-line
module docstring (lines 1-180, all prose) plus two large dataclass
docstrings (`InfluenceModel` 24 lines, `FITCInfluenceModel` 44 lines) and a
42-line FITC section banner (1403-1444) — roughly 290 lines of pure
docstring/comment-block prose before counting the many inline `#` comments
threaded through the function bodies (e.g. `_fit_gp_influence_model`'s
outer-search comments run 792-817, 822-827, 833-838, 895-911, ~60 lines on
their own). A rough count: docstrings/comment-only lines are on the order of
520-560 of the file's 2078 lines (25-27%); the remainder (~1520-1560 lines)
is executable code, blank separators, or field-annotation comments on
dataclass attribute lines (which this rough count folds in with code since
they share the line with a real field declaration).

## What I could not classify with full confidence

- **`_length_scale_bounds`'s three ell_min branches, `_within_1pct_log`'s
  guard, `_lambda_floor`'s edge pins, and `_cholesky_with_jitter`'s
  escalation**: classified DATA from line-level non-execution in the 16-task
  trace (2 draws each of 6 cases, 2 failing `mvt/nu` draws, `check`, `gmm`).
  Confirmed reachable by inspection (no upstream guard forecloses them), but
  "never observed to trigger in B=20/these particular seeds" is not the same
  guarantee as "reachable but rare at B=2000 production scale" — a wider
  seed range could exercise them.
- **`bin_posterior_variance`'s constant-coordinate early return (line 1339)**:
  did not fire even on the two failing `mvt/nu` draws where `nu`'s single
  coordinate *did* take the constant path (confirmed via line 752). That
  means `refine.py` apparently never calls `bin_posterior_variance` for a
  constant-path coordinate on those draws — plausible (the constant path
  needs no refinement-driving variance estimate) but I did not trace
  `refine.py`'s own call sites to confirm the mechanism; I looked only at
  this file's line coverage.
- **Whether a single vector estimator (fp, q=4; chabrier, q=3) ever splits
  into more than one coordinate group** (i.e., a genuine *partial* prototype
  failure — one output's evaluation failing while another's, at the same
  prototype, succeeds) **is not visible from line coverage**: the grouping
  code (lines 744-790, 1037-1050) runs identical lines whether `groups` holds
  one dict entry or several. The only confirmed complete-failure case in the
  trace is `mvt/nu` (q=1, so trivially one group of size 0). I cannot confirm
  or rule out multi-group behavior for fp/chabrier from the trace alone.
- **Line-range accounting in the Summary is a manual line-count partition
  by item, not a certified per-line audit** beyond the specific branches I
  individually checked against `executed_lines.json`/`executed_by_task.json`
  above; the ~56 "blank, unassigned" lines are believed to be blank
  separators between items but I did not verify each one.
