# qij_joint: the clean package

Coordinator's plan, 25 September 2026. Rulings are the author's unless
marked (C), a coordinator default the author may overrule.

## 0. What this is

`JT_Py_Pkgs/qij_joint` (import name `qij_joint`) is a clean distillation of
the QIJ method: only the logic the author accepts, none of the switches,
compatibility shims and dead designs left by the experiments. The old
`JT_Py_Pkgs/qij` repo is FROZEN as the WSOM paper's code: the paper,
its tables and figures stay there and are never brought forward; figure
code is ported later, as needed. The name's "joint" refers to a later
change, joint quantization of influence space for multi-output estimators
(currently marginal, one 1-D quantizer per output); it is NOT part of the
first build.

**Must hold:** on every quantity the new package keeps, a draw run by
`qij_joint` equals the same draw run by `qij@025613e` on the same machine,
bit for bit (timing fields excepted), at any worker count — except where
the author approves an efficiency rewrite that changes floating-point
order (Section 5, rule E8).

## 1. Sequence

1. **Inventory** (read-only agent, below): what exists in `qij`, what the
   paper's main path actually executes, what is switch- or data-dependent.
2. **Author rules** keep / drop / keep-as-diagnostic per item.
3. **Spec** (coordinator): Section 6 completed from the rulings, with a
   line budget per module.
4. **Build** (Sonnet, under Section 5), then the **old-vs-new audit**, then
   the **migration** of the TACC products into the new layout.

## 2. Settled design (carried from the 25 Sept discussion)

* **Unit of work** = (dataset, estimator, method, N, draws). Estimators:
  `pareto shape`, `pareto tail`, `mvt nu`, `mvt tail`, `fp all`,
  `imf all` (Chabrier), plus the 2-D Gaussian mixture from `qij/gmm.py`.
  The two-regime IMF is not carried (mis-specified).
* **Methods** `oracle` (theta_true, theta_hat), `ij` (analytic psi and
  V_ij = mean(psi^2)/N; QIJ never sees psi), `boot`, `qij`; each callable
  alone; draw s uses seed = master_seed + s everywhere, so methods join on s.
* **No YAML.** Every parameter is a flag. `runbook.sh` in the project folder
  is plain: variables and literal command lines, copy-pasted into a
  terminal; no case, functions or branching.
* **Products per draw:**
  `<out>/<dataset>_<estimator>_N<N>/<method>/s<SSSSS>.parquet` (one scalar
  row) and `s<SSSSS>.<kind>.parquet` (arrays: ij `psi`, boot `replicates`,
  qij `points`/`prototypes` for `--diag-draws`), plus `runs.log` (one JSON
  line per call: time, host, CPU, argv, parameters, draws written/skipped).
  The parent writes every file, arrays first, scalar row last, via
  `.tmp` + `os.replace`; the scalar row is the done marker; `--force`
  overwrites.
* **Timing:** raw only. `wall_time` (elapsed, what a user waits),
  `busy_time` (sum of each parallel task's own wall time plus the parent's
  serial time), `workers`. Ratios derived downstream.
* **Parallelism:** `oracle`, `ij` across draws; `boot`, `qij` a sequential
  draw loop, parallel within the draw (boot over replicate chunks; the
  parent draws the resamples in order from `default_rng(seed)` so results
  are identical at any worker count; qij serial within the draw in the
  first build). Processes, not threads, through one module
  `qij_joint/parallel.py` under the contract below.
* **Parallelism contract:** (1) only ints, floats, strings, numpy arrays
  cross the process boundary; estimators by registry key, a user's `T`
  once at pool start; (2) everything a worker imports is in the package,
  scripts have the `__main__` guard; (3) X memory-mapped once per draw,
  tasks carry only small arrays; (4) workers return result plus bookkeeping
  (evaluations, rows, failure flag, own wall time), the parent sums, no
  worker state; (5) workers catch estimator exceptions and return NaN plus
  a flag; (6) the worker initializer pins the five BLAS thread variables and
  builds the estimator; (7) one persistent pool per run; (8) tasks batched
  so dispatch cost is small beside them. `workers == 1` creates no pool.
* **Diagnostics** that combine methods (for example the old partition
  product, not carried in the first build) are post-processing over `ij`
  and `qij` products, never part of a method's draw.

## 3. Audit (after the build)

1. Baseline: the old `qij` working tree (at `025613e`) runs its
   `study._run_draw` for s in {0, 1, 999} and MVT nu's failed draws
   {342, 420, 523}, main parameters, counting the Cholesky jitter steps.
2. `qij_joint` runs the same draws (boot at 1 and 4 workers); every kept
   field equal with `==` (NaN == NaN), timing excluded; fields touched by
   an approved E8 rewrite compared at a stated tolerance instead.
3. Migration of `QIJ_WSOM2026/runs/main` (read only) into the new layout,
   kept fields only; round trip against the old frames exact.
4. The new laptop draws against the migrated products on disk: max
   relative difference per field, reported, not gated (the IMF agrees to
   about 6 significant figures across machines).

## 4. Carried project rules

No tests, no test files, no assertions, no input validation that raises.
Exactly one pass/fail check in the package: the weighted-mean scale
identity (`python -m qij_joint.check`). Nothing in the method is computed
by Monte Carlo. Terms follow `qij/spec/QIJ_glossary.md` (copied into this
repo at build time).

## 5. Code standards (the author's; canonical; every build prompt includes this section verbatim)

**Readability and size**

* R1. Every module has a line budget, set in Section 6 before the build.
  An agent that exceeds it reports why; the coordinator decides.
* R2. One way to do each thing. No option, flag or branch without a
  current use; no deprecated arguments, compatibility shims, or
  "accepted and ignored" parameters. A later need adds the option then.
* R3. No dead or commented-out code.
* R4. Names follow the spec's notation (`psi`, `p_k`, `U_k`, `V_btw`);
  the same quantity has the same name in every module.
* R5. Functions do one thing and fit on a screen; a function past ~60
  lines is split unless its steps only make sense together.
* R6. Public functions carry type hints and a docstring of at most a few
  lines: what it computes, argument and return shapes, and the formula
  or spec section it implements.
* R7. `try/except` only at a declared failure boundary: an estimator
  evaluation, or a library call that signals failure only by raising
  (the influence model's Cholesky); nowhere else.

**Comments**

* C1. A comment says what the code does and why, as it stands now. Never
  what it used to do, when it changed, who ruled it, or which revision,
  amendment, plan section or agent introduced it. History lives in git
  commit messages; lasting rationale lives in the spec.
* C2. Comment the non-obvious WHY (a numerical reason, a failure rule, a
  choice among equivalent forms). Do not restate what the line says.
* C3. A derivation longer than a few lines goes in the spec, and the code
  names the spec section once.

**Efficiency**

* E1. No Python loop over points (N) or over the rows of an array that
  numpy can process whole. Loops are allowed over small counts (outputs q,
  bins L, draws, refinement rounds) when the body is vectorized. A loop
  whose every iteration is one estimator evaluation (bootstrap replicates,
  prototype survey, bin stencils) is not an E1 finding: an evaluation of a
  black-box T cannot be vectorized; such loops are the parallelism targets
  (Section 2).
* E2. Grouped sums by `np.bincount` / `np.add.reduceat` over one argsort;
  never a per-group Python loop or a pandas groupby in the method.
* E3. Linear algebra by factorization: Cholesky (`cho_factor`/`cho_solve`)
  or `solve`, never an explicit inverse; factor once, reuse for every
  right-hand side; exploit structure (diagonal, low rank, Woodbury).
* E4. Memory is bounded: no array with N rows and M (prototype) or N
  columns is ever formed, cached or not; kernel products over the points
  are computed in row chunks of bounded size; nothing forms a B x N array.
* E5. Compute once, pass down: per-draw invariants are computed by the
  caller and passed in, never recomputed per call and never held in
  module-level caches (which break process parallelism and reproducibility).
  One exception: a module-level cache of CONSTANT data read from the
  package's own `data/` files (the FP and IMF population loaders).
* E6. Never evaluate the estimator twice at the same weights; the base
  evaluation is shared.
* E7. Preallocate; no `np.append` / `concatenate` / list-then-stack inside
  a loop; pandas only at the I/O boundary.
* E8. An efficiency rewrite that changes floating-point order (a loop sum
  replaced by `np.sum`, a reordered reduction, a different solver) changes
  results in the last bits and so breaks the audit's bit-identity. Such a
  rewrite needs the author's approval; approved ones are audited at a
  relative tolerance of 1e-12, with any change in an integer outcome (bin
  count, split counts) reported by draw. Approved: the three in Section 6.4.
* E9. A claimed saving is a measured one: time the function before and
  after, by direct timing, not by profiler attribution or arithmetic.

**Enforcement (coordinator, at review; not a package check).** Every diff
is read against R1 to E9. Mechanical screens run before reading: line
counts against budget; a grep for history words (`previously`, `used to`,
`was changed`, `revision`, `rev[0-9]`, `amend`, `ruled`, `plan §`,
`Sec [0-9]`, dates); a grep for `for .* in range(N`, `range(len(X`,
`iterrows`, `np.append`, `inv(`; pyflakes. A violation goes back to the
agent before anything is committed.

## 6. The package

Rulings: `spec/qij_inventory_rulings.md` as the coordinator recommended,
except a_bca is dropped, the three floating-point rewrites are approved
without measurement, truth is stored data, one file per dataset.

### 6.1 Layout and line budgets (whole file, docstrings included; R1)

```
qij_joint/
  pyproject.toml                       package metadata; deps numpy, scipy, pandas, pyarrow, h5py, joblib, vqlp
  spec/                                this plan, rulings, inventory, method_notes.md, glossary.md
  scripts/
    run.py                     75     CLI: one (dataset, estimator, method) over a range of draws
    make_truth.py              55     regenerates data/truth/{fp,imf}.json from the populations
  src/qij_joint/
    __init__.py                10     exports QIJ, Bootstrap
    core/
      counter.py               65     Counter: counts evaluations, rows, failures; holds prepare() state
      differences.py           80     step sizes, perturbed weights, the central stencil
      xvq.py                  200     data quantizer, prototype survey
      influence_model.py     1060     the GP influence model (psi0, sigma, bin posterior variance)
      ivq.py                  285     1-D influence quantizer, initial bins, bin differences, V_btw
      refine.py               520     gain-driven refinement, V_win_hat, rho, psi_hat
    qij.py                    240     QIJ.fit
    bootstrap.py              100     Bootstrap.fit, serial or on a pool
    result.py                 115     QIJResult, BootstrapResult (.variance, .interval)
    parallel.py               165     the one module that touches a process pool
    estimators.py             650     pareto_shape, pareto_tail, mvt_nu, mvt_tail, fp, Chabrier
    gmm.py                    800     GMM2D (not registered as a study case yet)
    datasets.py                90     data draws, mvt_vq_transform, population loaders
    registry.py                80     (dataset, estimator) -> Case; truth()
    pipeline.py               230     the four methods' draw loops
    products.py               130     the only module that knows the product layout
    check.py                   50     the one check
    data/
      fp_sdss.npz, stars.h5            populations (copied unchanged)
      truth/pareto.json, mvt.json, fp.json, imf.json
```

Total budget about 5,000 lines (budgets raised to the built sizes where the interface, an exact port, or a later wave set the size), against 7,600 in `qij` outside the
paper-only tables and figures.

### 6.2 Pinned interfaces (the three builders code against these)

**Estimator protocol.** `T(X, w) -> ndarray (q,)`, NaN on failure, never
raises by design; attributes `name: str`, `outputs: tuple[str, ...]`,
`eta: float`. Optional `T.influence(X, w) -> ndarray (N, q)`. Optional
`T.prepare(X) -> prep` returning the weight-independent quantities of X;
an estimator with `prepare` also accepts `T(X, w, prep=prep)` and must
return bit-for-bit what `T(X, w)` returns. `GMM2D` keeps its per-call
`**kwargs`.

**Counter** (`core/counter.py`). `Counter(T, N)`; `counter(X, w)` calls
`T`, passing `prep` from `T.prepare(X)` computed once per distinct array
object for the Counter's lifetime (one fit), and counts evaluations, rows
and NaN results as now. `snapshot() -> (evaluations, rows)`.

**QIJ** (`qij.py`). `QIJ(eps=0.01, seed=0, vq_transform=None).fit(X, T)
-> QIJResult`. No `eta`, `influence`, `influence_model` or `fitc_rank`
arguments. Arithmetic, RNG use and evaluation order as `qij@025613e`
with `influence_model='gp'`.

**QIJResult** (`result.py`): `outputs`, `N`, `theta_hat (q,)`, per output
`V_btw`, `V_win_hat`, `V_tot_hat`, `L`, `n_level_splits`,
`n_adjacency_splits`, `rho`, `gain_ratio`, `n_refine_evals`, `ell`, `lam`,
`ell_bound`, `lam_bound` (each an array (q,)); `M_X`, `n_failed`,
`evals_by_stage`, `rows_by_stage`, `wall_time_by_stage` (dicts over
`prototype`, `full_data`, `refinement`, `total`); `busy_time_total`,
`workers`; diagnostic arrays `psi0 (N,q)`, `sigma (N,q)`, `psi_hat (N,q)`,
`bin_label (N,q)`, `bmu (N,)`, `prototype_p (M,)`, `prototype_w (M,d)`
(native coordinates), `prototype_I (M,q)`. Methods `.variance` (V_btw)
and `.interval(level)` (normal interval on V_btw).

**Bootstrap** (`bootstrap.py`). `Bootstrap(B=2000, seed=0).fit(X, T,
pool=None) -> BootstrapResult(outputs, replicates (B,q), n_failed,
wall_time, busy_time, workers)`, `.variance`, `.interval(level)`
(percentile). Resamples drawn in replicate order from
`default_rng(seed).multinomial(N, full(N, 1/N))`, exactly as now; with a
pool, chunks of count vectors go to workers; results identical to serial.

**parallel.py.** `Pool(workers, T=None, case=None)`: a persistent
executor from `joblib.externals.loky.get_reusable_executor`, initialized
once with either a user's `T` (cloudpickled once) or a registry key
`case=(dataset, estimator)` resolved in the worker; the initializer pins
the five BLAS thread variables. `pool.share(X) -> handle`: the parent
writes X once per draw to a `.npy` in a temporary directory; workers
memory-map it and keep the map for that draw only. `pool.map(fn, tasks)
-> list` in task order; each result carries the task's own wall time;
workers catch every exception from `T` and return NaN plus a failure
flag. `workers == 1` means no executor: `map` runs in-process.

**registry.py.** `Case(dataset, estimator, draw: (N, seed) -> X,
make_T: () -> T, vq_transform: callable | None)`; `case(dataset,
estimator) -> Case`; `truth(dataset, estimator) -> ndarray (q,)` read from
`data/truth/<dataset>.json`. Cases: `pareto shape`, `pareto tail`,
`mvt nu`, `mvt tail`, `fp all`, `imf all`.

**Truth files.** `data/truth/<dataset>.json`:
`{"<estimator>": {"outputs": [...], "values": [...], "source": "..."}}`.
Pareto and MVT: the specified constants in `qij.datasets._PARAMETRIC_TRUTH`,
source `"specified by the data-generating process"`. FP and IMF: the
`theta_true_<o>` columns of `QIJ_WSOM2026/runs/main/{fp/fp,imf/chabrier}/
truth.parquet` (constant across draws; use draw 0, confirm all rows
equal), source naming the population file, "estimator fit with unit
weights", and the estimator's settings. `make_truth.py` recomputes the FP
and IMF entries the same way and rewrites the two files; rerun it after
any change to those estimators' settings.

**pipeline.py / products.py / run.py.** As Sections 2 and 4 of this plan:
per-draw scalar rows and arrays, parent writes atomically, `runs.log`,
`--force`. Methods: `oracle` (theta_true from `registry.truth`, theta_hat
= `T(X, ones)`), `ij` (`psi = T.influence(X, ones)`, `V_ij =
mean(psi**2, axis=0)/N`; no separate `T` call), `boot`, `qij`
(`--diag-draws` selects draws that also store `points` and `prototypes`).
`oracle` and `ij` parallel across draws; `boot` and `qij` a sequential
draw loop, `boot` using its pool within the draw, `qij` serial in this
build. Scalar fields per method: Section 2's table, with the qij row
being QIJResult's scalars under today's column names
(`V_btw_<o>`, `wall_time_<stage>`, `evals_<stage>`, `rows_<stage>`,
`normalized_rows`, ...) minus `B_hat_<o>`, `a_bca_<o>`, and the per-bin
list columns.

### 6.3 What is not carried

FITC; the two-regime IMF; `QIJ(eta=...)`; `.summary()`; the
`weighted_mean` estimator and `supports_for`; the one-sided stencil; the
two dead shortcuts in `ivq.py`; GMM `reg_covar` and its `max_iter <= 25`
short-circuit; the `denom_m <= 0` fallback in the influence model (after
confirming the algebra makes it unreachable); `a_bca` and
`core/outputs.py`; `B_hat` and per-bin `d2T` storage (d2T is still
computed if refinement uses it); the partition product; YAML, `run_study`,
multi-N directories, h5 resume I/O; the paper's tables and figures.
All data-triggered failure handling is carried (rulings sheet B).

### 6.4 Efficiency changes in the port

Same bits: `ij` uses one fit; one full-data evaluation per draw; the FP
population read once (its standardization constants derived from the
cached pool); the step-parameter list comprehension in the influence model
vectorized; each refinement leaf's Var(psi0) and mean computed once;
`prepare(X)` on every estimator whose calls recompute weight-independent
quantities (log-ratios, standardized X, the GMM's fixed bases), with
identical arithmetic.

Approved order changes (E8, audited at 1e-12): `ivq.within_share` as two
`bincount` calls; `bin_posterior_variance` as one grouped pass;
`sum_p_ubar2` maintained incrementally.

### 6.5 Build plan: three builders in parallel, then the audit

Disjoint files; the interfaces above are the only contract between them.
Each builder writes `<scratch>/build/B<k>_done.txt` when finished. B3's
end-to-end run waits for B1 and B2's markers (poll by file existence,
never by `ps`).

| Builder | Files | From `qij` |
|---|---|---|
| B1 method | `core/*`, `qij.py`, `result.py`, `check.py`, `spec/method_notes.md`, `spec/glossary.md` | `core/*`, `qij.py`, `result.py`, `intervals.py`, `check.py`, `spec/QIJ_glossary.md`, `spec/QIJ_method_spec.md` |
| B2 estimators and data | `estimators.py`, `gmm.py`, `datasets.py`, `registry.py`, `data/**`, `scripts/make_truth.py` | `estimators.py`, `gmm.py`, `datasets.py`, `data/`, the TACC `truth.parquet` files |
| B3 infrastructure | `parallel.py`, `bootstrap.py`, `pipeline.py`, `products.py`, `scripts/run.py`, `__init__.py`, `pyproject.toml` | `bootstrap.py`, `study.py`, `scripts/run.py` |

Then the audit agent (Section 3), then the coordinator writes the runbook
and commits.

---

## Build prompts

Each prompt below is dispatched as: **Common preamble** + **Section 5 of
this plan, verbatim** + the builder's own part.

### Common preamble

> You are one of three builders writing `qij_joint`, a clean distillation
> of the `qij` package's QIJ method. Repo:
> `/Users/jtaylor/Dropbox/Software/JT_Py_Pkgs/qij_joint` (new; create only
> the files your part names). The old package, READ ONLY:
> `/Users/jtaylor/Dropbox/Software/JT_Py_Pkgs/qij` at HEAD `025613e`
> (source `src/qij/`). Never read the `vqboot` repository, even though
> your shell may start inside it. Read first: `qij_joint/spec/
> qij_joint_plan.md` (all of it; Section 6.2 pins the interfaces you share
> with the other two builders, who are working at the same time on
> disjoint files), `qij_joint/spec/qij_inventory_rulings.md`, and the
> inventory rows for your files in `qij_joint/spec/inventory/`.
>
> The requirement: every number a kept quantity takes must equal what
> `qij@025613e` computes for the same inputs, bit for bit, except where
> Section 6.4 approves an order change. So port the arithmetic exactly:
> same operations, same order, same RNG calls in the same sequence, same
> estimator calls in the same sequence. Distill everything else: delete
> what Section 6.3 drops, rewrite every comment and docstring to the
> standards below (what the code does now and why; no history), and stay
> within the line budgets of Section 6.1. Where porting exactly and a
> standard conflict, port exactly and report the conflict.
>
> Environment: `python3.9`; `PYTHONPATH=<qij_joint>/src`;
> `PYTHONDONTWRITEBYTECODE=1`; `OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS`,
> `MKL_NUM_THREADS`, `VECLIB_MAXIMUM_THREADS`, `NUMEXPR_NUM_THREADS` = 1.
> `pip install` does not work; use `PYTHONPATH`. `loky` ships inside
> joblib as `joblib.externals.loky`.
>
> Rules: no tests, test files, assertions, or input validation that
> raises; no new checks. Do not modify the old `qij` repo or
> `/Users/jtaylor/Dropbox/Research/QIJ_WSOM2026`. Do not commit, do not
> `git init`. Verification is only what your part names. Scratch folder:
> `/private/tmp/claude-501/-Users-jtaylor-Dropbox-Software-JT-Py-Pkgs-vqboot-fable/c0fd00f7-ff54-4ae7-926b-df11406439fd/scratchpad/build/`.
> When finished, write `B<k>_done.txt` there (`done`, or `failed:
> <reason>`), then report: files with line counts against budget, every
> place you could not port exactly or had to interpret the spec, and
> every standards conflict.

### B1: the method

> Your files: `src/qij_joint/core/{counter,differences,xvq,
> influence_model,ivq,refine}.py`, `src/qij_joint/qij.py`,
> `src/qij_joint/result.py`, `src/qij_joint/check.py`,
> `spec/method_notes.md`, `spec/glossary.md`.
>
> 1. Port `qij/src/qij/core/*` and `qij.py` to the Section 6.2 `Counter`,
>    `QIJ` and `QIJResult` interfaces. The GP influence model only: delete
>    FITC and the `influence_model`/`fitc_rank` options entirely. Delete
>    the `denom_m <= 0` fallback only if you can show from the preceding
>    rule that it is unreachable; otherwise keep it and say why. `QIJ`
>    no longer takes `influence`; it never sees the analytic influence.
> 2. `QIJResult` carries the diagnostic arrays Section 6.2 lists, taken
>    from quantities the method already computes (psi_hat is
>    `CoordinateResult.field`; prototype positions in native coordinates
>    are today's `W_X`). `busy_time_total = wall_time_total` and
>    `workers = 1` in this build. `.interval(level)` and the bootstrap's
>    percentile interval come from `qij/src/qij/core/intervals.py`,
>    folded into `result.py`.
> 3. Apply Section 6.4's changes that fall in your files, including the
>    three approved order changes. `Counter` implements `prepare`
>    (Section 6.2); the prototype survey evaluates T on the prototypes,
>    a different array from X, so it gets its own `prepare`.
> 4. `check.py`: the weighted-mean scale identity, ported as is.
> 5. `spec/method_notes.md`: the derivations that today live in module
>    docstrings (refinement's split rule, gain pricing, closing rule,
>    V_win_hat; the influence model's noise floor and width search; the
>    stencil's step rule), stated as the method now is, no history; each
>    module names its section once. `spec/glossary.md`: copy of
>    `qij/spec/QIJ_glossary.md` with entries for dropped quantities
>    (a_bca, B_hat, FITC, the support clip) removed.
>
> Verification: `pyflakes` clean on your files; `python3.9 -m
> qij_joint.check` passes. Nothing else.

### B2: estimators and data

> Your files: `src/qij_joint/estimators.py`, `src/qij_joint/gmm.py`,
> `src/qij_joint/datasets.py`, `src/qij_joint/registry.py`,
> `src/qij_joint/data/**`, `scripts/make_truth.py`.
>
> 1. Port `pareto_shape`, `pareto_tail`, `mvt_nu`, `mvt_tail`, `fp`,
>    `Chabrier` with their analytic influences. Delete the two-regime IMF
>    and exactly its removal set (`spec/inventory/inventory_C3.md`
>    Table 3), `weighted_mean`, `_weighted_mean_influence`,
>    `supports_for`, and anything else only they use.
> 2. Port `GMM2D` with `fit_and_influence`; delete `reg_covar` and the
>    `max_iter <= 25` short-circuit. Its 243-line module docstring becomes
>    a short docstring plus a section in `spec/method_notes.md` titled
>    "GMM2D" (B1 creates that file; append your section at the end, after
>    B1's `B1_done.txt` exists, or create the file with only your section
>    if B1 has not finished when you are otherwise done, and say so).
> 3. Add `prepare(X)` (Section 6.2) to every estimator whose calls
>    recompute weight-independent quantities; `T(X, w, prep=T.prepare(X))`
>    must equal `T(X, w)` bit for bit. List each estimator and what its
>    `prepare` holds.
> 4. `datasets.py`: the four data draws and `mvt_vq_transform`, arithmetic
>    unchanged; the FP and IMF population loaders cached (the E5
>    exception); FP's standardization constants derived from the cached
>    pool, not a second read of the file. No truth logic here.
> 5. `registry.py` and `data/truth/{pareto,mvt,fp,imf}.json` per Section
>    6.2. Copy `fp_sdss.npz` and `stars.h5` unchanged.
> 6. `scripts/make_truth.py` per Section 6.2. Do not run it.
>
> Verification: `pyflakes` clean on your files; import every module; for
> each registered case, one call `T(X, ones)` on `case.draw(200, 0)`
> compared with the old package's estimator on the old package's draw,
> and one `T(X, w, prep=T.prepare(X))` against `T(X, w)`, all with `==`.
> Report those comparisons. Nothing else.

### B3: infrastructure

> Your files: `src/qij_joint/parallel.py`, `src/qij_joint/bootstrap.py`,
> `src/qij_joint/pipeline.py`, `src/qij_joint/products.py`,
> `src/qij_joint/__init__.py`, `scripts/run.py`, `pyproject.toml`.
>
> 1. `parallel.py` per Section 2's contract and Section 6.2.
> 2. `bootstrap.py` per Section 6.2: identical results at any worker
>    count; `busy_time` is the sum of task wall times plus the parent's
>    time drawing resamples; chunks of about `B / (4 * workers)`
>    replicates. Uses `prepare` when `T` has it, computed once per draw
>    (in each worker, for the shared X).
> 3. `products.py`, `pipeline.py`, `scripts/run.py` per Sections 2, 4 and
>    6.2: the CLI `run.py <dataset> <estimator> <method> --N --draws a:b
>    --seed --workers --out [--force]`, `--B` for boot, `--eps
>    --diag-draws a:b` for qij; `products.write_draw`, `collect`,
>    `collect_array`. `qij` with `--workers` other than 1 prints that
>    within-draw QIJ parallelism is not built yet and runs with 1.
> 4. `__init__.py` exports `QIJ` and `Bootstrap`; `pyproject.toml` with
>    the dependencies of Section 6.1.
>
> Until B1 and B2 finish, code against Section 6.2 only. Then, after both
> `B1_done.txt` and `B2_done.txt` exist (poll by file every 60 s in one
> long bash call), verification: `pyflakes` clean on your files, and ONE
> end-to-end execution into the scratch folder: all four methods on
> `pareto tail` at `--N 200 --draws 0:4`, `boot` with `--B 40 --workers
> 2`, `oracle` and `ij` with `--workers 2`, `qij` with `--diag-draws
> 0:1`; then `collect` each method and print the frames' shapes. If a B1
> or B2 marker says `failed`, stop and report.
