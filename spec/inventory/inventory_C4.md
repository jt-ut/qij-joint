# Inventory C4

Scope: `src/qij/study.py`, `src/qij/qij.py`, `src/qij/result.py`,
`src/qij/bootstrap.py`, `src/qij/check.py`, `src/qij/__init__.py`,
`src/qij/core/xvq.py`, every file in `scripts/` except `make_tables.py` and
`make_figures.py` (i.e. `scripts/run.py`), and every stored product field
(`truth.parquet`, `qij.parquet`, `boot.h5`, `qij_partition.parquet`,
`qij_points.parquet`, `qij_prototypes.parquet`).

Repo: `/Users/jtaylor/Dropbox/Software/JT_Py_Pkgs/qij` HEAD `025613e`.
Project data: `/Users/jtaylor/Dropbox/Research/QIJ_WSOM2026` (read only).

Trace: completed successfully (`trace_done.txt` = "done"). Used
`executed_lines.json` and `trace_notes.md` from
`<scratch>/inventory/`. 16 tasks: the 6 (dataset, estimator) main-study
cases at s=0,1 each (pareto/shape, pareto/tail, mvt/nu, mvt/tail, fp/fp,
imf/chabrier), plus `mvt/nu` s=342 and s=420 (the two TACC-failed draws
traced -- the only case with any failures in the whole main study: s in
{342, 420, 523} all have `n_failed>0` AND NaN `V_btw_*` together; no
`theta_hat` NaN occurs anywhere), plus `qij.check` and one `qij.gmm` fit.
Status below is stated from this trace where the item is in `_run_draw`'s
call graph; for `run_study`/`_run_dataset`/the flush/resume/h5 I/O
functions/`scripts/run.py`, which the trace never calls, status is
`MAIN (production I/O, not traced)`, confirmed by reading `run_study`'s
call chain under `configs/main.yaml`'s settings.

## Paper-only files (stay in the old repo; not inventoried)

| file | lines |
|---|---|
| `src/qij/tables.py` | 737 |
| `src/qij/figures.py` | 2288 |
| `scripts/make_tables.py` | 47 |
| `scripts/make_figures.py` | 109 |

## Configs other than `main.yaml` (retired; kept for the record)

`configs/main.yaml`: `master_seed=0, S=1000, B=2000, eps=0.01,
levels=[.50,.60,.70,.80,.90,.95,.99]`, datasets pareto[shape,tail] N=[2000],
mvt[nu,tail] N=[2000], fp[fp] N=[2000], imf[chabrier] N=[2000].
`levels` is never read by `study.py` (only by the excluded `tables.py`/
`figures.py`); `influence_model`/`fitc_rank` keys are absent (so `study.py`'s
own defaults, `'gp'`/`None`, apply).

| config | differs from `main.yaml` |
|---|---|
| `cost_vs_n.yaml` | S=100; datasets: fp only, N=[1000,3000,10000,30000,76997] (multi-N -> `run_study`'s `N<size>` subdirectory branch) |
| `cost_vs_n_imf.yaml` | S=100; datasets: imf[**imf**] (two-regime, not chabrier) only, N=[1000,2000,5000,10000,19857] (multi-N) |
| `mvt_tail_n1000.yaml` | S=200; datasets: mvt[tail] only, N=[1000] (not 2000) |
| `smoke.yaml` | S=3, B=50; otherwise same datasets/estimators as main (imf[chabrier]) |
| `timing.yaml` | S=20; datasets: imf[**imf**] not chabrier; run with `--workers 1` (not a config field) |
| `timing_draw.yaml` | S=1; same as timing.yaml otherwise; `master_seed` overwritten per SLURM array task |
| `validate.yaml` | S=200; datasets: pareto+mvt+fp only, no imf case at all |

None of the retired configs set `influence_model` or `fitc_rank` either --
`'fitc'` and any non-null `fitc_rank` are unused by every config on disk.

---

## Table 1 -- code items

### `src/qij/__init__.py` (7 lines)

| item | file:lines | what it does (one line) | status | lines |
|---|---|---|---|---|
| module | `__init__.py:1-7` | re-exports `QIJ`, `Bootstrap` as the package's public surface | MAIN | 7 |

### `src/qij/check.py` (49 lines)

| item | file:lines | what it does (one line) | status | lines |
|---|---|---|---|---|
| module | `check.py:1-49` | `python -m qij.check`: the one pass/fail check, the weighted-mean scale identity | MAIN | 49 (all executed per trace) |
| `_weighted_mean` | `check.py:19-20` | plain callable `T`, no `.outputs`/`.eta`/`.name`, so `QIJ.fit`'s `_wrap` takes its "plain callable" branch | MAIN | 2 |
| `main` | `check.py:23-45` | draws N(0,1) data, fits `QIJ()` on `_weighted_mean`, checks `S_btw+S_win == S_tot` to `1e-12` | MAIN | 23 |
| `__main__` guard | `check.py:48-49` | CLI entry | MAIN (module entry, executed as `python -m qij.check`) | 2 |

### `src/qij/bootstrap.py` (49 lines)

| item | file:lines | what it does (one line) | status | lines |
|---|---|---|---|---|
| module | `bootstrap.py:1-49` | `Bootstrap`, the comparator method | MAIN | 49 (all 27 executable lines executed per trace) |
| `Bootstrap.__init__` OPTION `B` | `bootstrap.py:20` | replicate count; main-path value 2000 (`main.yaml`'s `B`; traced with B=20, same code path) | MAIN | - |
| `Bootstrap.__init__` OPTION `seed` | `bootstrap.py:20-22` | RNG seed; main-path value `master_seed+s` | MAIN | - |
| `Bootstrap.fit` | `bootstrap.py:24-49` | draws B multinomial resamples, evaluates `T` on each, returns `BootstrapResult` | MAIN | 26 |
| loop `for b in range(self.B)` | `bootstrap.py:40-41` | one `T` evaluation per replicate, weights drawn one at a time (comment: avoids B*N floats live at once) | MAIN | 2 |

### `src/qij/qij.py` (194 lines)

| item | file:lines | what it does (one line) | status | lines |
|---|---|---|---|---|
| module | `qij.py:1-195` | `QIJ`, the influence-aligned variance method | MAIN | 105 executed lines (per trace); remaining are docstrings/blank |
| `_wrap` | `qij.py:32-40` | gives a plain callable `outputs=('theta',)`/machine-eps `eta`/name; passes through a `T` that already has `.outputs` | MAIN | 9 |
| `_wrap` branch: `hasattr(T,'outputs')` True | `qij.py:35-36` | used for all 6 study estimators (each already decorated with `.outputs`) | MAIN | 2 |
| `_wrap` branch: False (plain callable) | `qij.py:37-40` | used by `check.py`'s `_weighted_mean` | MAIN | 4 |
| `QIJ.__init__` OPTION `eps` | `qij.py:46,49` | cost-rule tolerance; main-path value 0.01 | MAIN | - |
| `QIJ.__init__` OPTION `eta` | `qij.py:46,50` | per-evaluation relative accuracy override; default `None` | see `fit` ternary below | - |
| `QIJ.__init__` OPTION `seed` | `qij.py:46,51` | drives the 𝒳-VQ only; main-path value `master_seed+s` | MAIN | - |
| `QIJ.__init__` OPTION `vq_transform` | `qij.py:47,52` | main-path value `datasets.mvt_vq_transform` for mvt, `None` otherwise | MAIN (both values used across the 6 cases) | - |
| `QIJ.__init__` OPTION `influence_model` | `qij.py:48,60` | `'gp'` (main-path, default) or `'fitc'` (sparse) | MAIN='gp'; `'fitc'`=SWITCH | - |
| `QIJ.__init__` OPTION `fitc_rank` | `qij.py:48,61` | FITC inducing-point count; main-path value `None` | MAIN=`None`; non-null=SWITCH | - |
| `QIJ.fit` | `qij.py:63-194` | runs stage 1 (𝒳-VQ) and stage 2 (refinement), returns `QIJResult` | MAIN | most of the method (see below) |
| ternary `eta = self.eta if self.eta is not None else T.eta` | `qij.py:73` | ONE line, both branches; `self.eta` is never set to a non-`None` value anywhere in the repo (`grep eta=` across `src/qij` and `scripts/` finds no `QIJ(..., eta=...)` call) | `self.eta is not None` branch: UNREACHED (referenced nowhere with a non-default value) -- for a caller override of accuracy that no caller ever supplies; `T.eta` branch: MAIN | - |
| `Z, inverse = vq_transform(X) if ... else (X, identity)` | `qij.py:84` | mvt gets its own whitening transform+inverse; every other dataset gets the identity | MAIN (both branches; 4/6 traced cases take the identity branch, mvt takes the transform) | - |
| `if Z.ndim == 1: reshape` | `qij.py:90-92` | 1-D datasets (pareto, imf) need Z promoted to 2 columns for the quantizer, then un-promoted on the way out | MAIN (line 90-92 in executed set; pareto/imf cases 1-D) | 3 |
| `M_requested = cost_rule_M(...)` | `qij.py:94` | requested first-stage prototype count | MAIN | 1 |
| `run_xvq` call | `qij.py:97` | stage 1 in full | MAIN | 1 |
| `W_X = inverse(xvq.centers)` | `qij.py:107` | prototypes in T's native coordinates, recomputed via the same pure `inverse`, not returned a second way | MAIN | 1 |
| `fit_influence_model` call | `qij.py:108-110` | fits the GP/FITC influence model (out of scope: `core/influence_model.py`) | MAIN | 3 |
| `theta_hat = counter(X, ones(N))` | `qij.py:117` | the shared full-data evaluation -- **also computed independently by `study._run_draw:139`'s `T(X, ones)` for the truth row; same weights, same X, evaluated twice** | MAIN -- **E6** (see Table 2) | 1 |
| refinement list comprehension `run_refinement` per output | `qij.py:120-130` | per-output gain-driven refinement (out of scope: `core/refine.py`) | MAIN | 11 |
| `if any(cr.failed for cr in coordinates): NaN-void` | `qij.py:142-149` | a failed initial-bin evaluation on any coordinate voids V_btw/V_win_hat/V_tot_hat/B_hat/a_bca on every coordinate | MAIN -- confirmed executed (mvt/nu s=342, s=420, the traced TACC-failed draws, hit lines 142-146) | 8 |
| `evals_by_stage`/`rows_by_stage`/`wall_time_by_stage` dicts | `qij.py:153-172` | per-stage bookkeeping | MAIN | 20 |
| `if influence is not None:` | `qij.py:176-183` | oracle variance/psi only computed when an analytic influence was passed | MAIN, both branches (`check.py` passes none -> False; all 6 study estimators carry `.influence` -> True) | 8 |
| `QIJResult(...)` construction | `qij.py:185-193` | assembles the result | MAIN | 9 |

### `src/qij/result.py` (135 lines)

| item | file:lines | what it does (one line) | status | lines |
|---|---|---|---|---|
| module | `result.py:1-136` | `QIJResult`/`BootstrapResult` dataclasses and their `.interval`/`.variance`/`.summary` surface | MAIN (fields only) | 48 executed lines are field declarations and `def` statements only |
| `QIJResult` dataclass fields | `result.py:43-62` | every field `study._run_draw`/`_qij_row` read off `res` | MAIN | 20 |
| `QIJResult.variance` property | `result.py:64-72` | V_btw per output | UNREACHED -- `def`/decorator lines execute at class-definition time, but the property body (66-72) is never in the executed set, and no `.variance` access appears anywhere in `src/qij` or `scripts/` outside `result.py` itself | 9 |
| `QIJResult.interval` | `result.py:74-80` | (q,2) interval at any level, via `core.intervals.qij_interval` | UNREACHED -- never called (body 75-80 not executed; no `.interval(` call site anywhere) | 7 |
| `QIJResult.summary` | `result.py:82-103` | one row per output, the "pinned user surface" | UNREACHED -- never called (body 83-103 not executed; no `.summary(` call site anywhere) | 22 |
| `BootstrapResult` dataclass fields | `result.py:106-113` | `outputs`, `replicates`, `n_failed`, `wall_time` | MAIN -- these ARE read directly by `study._run_draw` (`boot_res.replicates`/`.n_failed`/`.wall_time`) | 8 |
| `BootstrapResult.variance` property | `result.py:115-118` | `nanvar` over replicates | UNREACHED (same evidence as `QIJResult.variance`) | 4 |
| `BootstrapResult.interval` | `result.py:120-122` | percentile interval | UNREACHED | 3 |
| `BootstrapResult.summary` | `result.py:124-135` | one row per output | UNREACHED | 12 |

### `src/qij/core/xvq.py` (218 lines)

| item | file:lines | what it does (one line) | status | lines |
|---|---|---|---|---|
| module | `core/xvq.py:1-219` | the 𝒳-VQ: quantize, then measure influence at each prototype by forward differences | MAIN | 74 executed lines per trace |
| constant `_KAPPA_REF = 2.7` | `core/xvq.py:29` | the cost-rule's reference kappa | MAIN, single value, no alternate | 1 |
| constant `_BMU2_BATCH_CAP = 4096` | `core/xvq.py:34` | row-batch cap for the second-BMU repair | used only inside the DATA branch below | 1 |
| `XVQ` dataclass | `core/xvq.py:37-46` | the fitted codebook's structure | MAIN | 10 |
| `cost_rule_M` | `core/xvq.py:49-64` | `M_X = ceil(sqrt((1+2*q*M_ref)*N/2))`, floored at 20, capped at `N//2` | MAIN | 16 |
| `fit_xvq` | `core/xvq.py:67-104` | k-means via `vqlp.VQFitter`, drops empty receptive fields, resolves BMU/BMU2, builds CADJ | MAIN | 38 |
| `_resolve_bmu2` | `core/xvq.py:107-143` | repairs `bmu2` where the second BMU pointed at a dropped-empty prototype | function entry/early-return MAIN; repair body DATA (not triggered) | see below |
| `_resolve_bmu2`: `if not missing.any(): return bmu2` | `core/xvq.py:127-130` | early return when no point needs repair | MAIN -- confirmed: lines 127-130 executed, but the repair body (132-143) is NOT in the executed set on any of the 16 traced tasks | 4 |
| `_resolve_bmu2`: repair loop (batched) | `core/xvq.py:132-143` | for points needing repair, nearest live prototype other than own `bmu`, batched over `_BMU2_BATCH_CAP` rows | DATA -- not executed in the trace; triggered when the quantizer drops enough empty receptive fields that some point's stored second-BMU no longer refers to a live prototype (module docstring/comment, lines 113-125) | 12 |
| `prototype_influences` | `core/xvq.py:146-195` | 1 base evaluation + one forward difference per prototype, mass-centred | MAIN -- fully executed including the loop | 50 |
| loop `for j in range(M_used)` | `core/xvq.py:176-181` | one `counter` evaluation per prototype (perturbed weights, forward difference) | MAIN -- **E1** (see Table 2) | 6 |
| finite-mask centering (`finite`/`mass`/`psi_bar`) | `core/xvq.py:191-194` | centers `I_proto` over finite prototypes only, mass-weighted; a failed prototype's evaluation stays NaN (comment: "ruled 18 September") | MAIN if any prototype evaluation ever fails (comment implies this is a real, if rare, occurrence); on the traced draws all prototypes were finite so this is arithmetically a no-op-if-all-finite case, still executed every call | 4 |
| `run_xvq` | `core/xvq.py:198-218` | stage 1 in full: fit codebook, map centers via `inverse`, compute prototype influences | MAIN | 21 |

### `src/qij/study.py` (552 lines)

| item | file:lines | what it does (one line) | status | lines |
|---|---|---|---|---|
| module | `study.py:1-553` | the study loop: `run_study` writes truth/boot/qij(+partition/points/prototypes) products | MAIN | 158 executed lines (`_run_draw`'s graph); `run_study`/`_run_dataset`/I-O helpers MAIN (production I/O, not traced) |
| constant `PARTITION_S = 200` | `study.py:71` | draws `s < 200` get a `qij_partition` row | MAIN (both traced s=0,1 satisfy it; s=342/420 do not, since >=200) | 1 |
| constant `POINTS_DRAW = 0` | `study.py:72` | the one designated draw for `qij_points`/`qij_prototypes` | MAIN (s=0 traced for every case) | 1 |
| constant `PARTITION_M_GRID = (4,6,8,12,16,24,32,48,64)` | `study.py:73` | the M grid `_partition_rows` sweeps | MAIN | 1 |
| constant `_FLUSH_EVERY = 20` | `study.py:74` | draws between parquet flushes | MAIN (production I/O, not traced), single value | 1 |
| `_DATASET_ESTIMATORS` registry | `study.py:79-95` | (dataset,T.name) -> estimator callable | MAIN for the 6 main-path entries | 17 |
| `_DATASET_ESTIMATORS['imf']['imf']` (two-regime) | `study.py:91-92` | the retired two-regime gamma/generalized-Schechter IMF fit | SWITCH -- reachable only via `estimators: [imf]` in `timing.yaml`/`timing_draw.yaml`/`cost_vs_n_imf.yaml`, never `main.yaml` (which names `chabrier`) | 2 |
| `_VQ_TRANSFORM = {'mvt': mvt_vq_transform}` | `study.py:100` | mvt alone gets the MVT standardization; `.get()` returns `None` for every other dataset | MAIN, both outcomes exercised | 1 |
| `_pin_threads` | `study.py:103-110` | pins the 4 BLAS thread env vars to 1 | MAIN (production I/O, not traced -- called only from `run_study`) | 8 |
| `_run_draw` | `study.py:118-187` | one draw, every estimator, both methods; called directly by the tracer | MAIN | 70 (all lines in its body appear in the executed set) |
| `theta_hat = T(X, ones)` | `study.py:139` | the study's own full-data evaluation, for the truth row -- **duplicates `QIJ.fit`'s internal `counter(X, ones(N))` at `qij.py:117`, same X, same weights** | MAIN -- **E6** (see Table 2) | 1 |
| `Bootstrap(B, seed).fit(X, T)` | `study.py:140` | the comparator | MAIN | 1 |
| `QIJ(...).fit(X, T, influence=psi_fn)` | `study.py:142-144` | the method under study | MAIN | 3 |
| `truth_row` V_oracle block: `if qij_res.oracle_variance is not None` | `study.py:150-152` | oracle variance columns, only when the estimator carries an analytic influence | MAIN (True branch; all 6 estimators carry `.influence`, so the False branch of this specific check never fires within `_run_draw`) | 3 |
| partition-rows block: `if psi_oracle is not None and s < PARTITION_S` | `study.py:154-166` | builds `qij_partition` rows for early draws | MAIN | 13 |
| `Z.ndim==1` reshape inside partition block | `study.py:158-163` | promotes 1-D Z to 2 columns for `fit_xvq` (mirrors `qij.py`'s own promotion) | MAIN (pareto/imf cases) | 6 |
| points/prototypes block: `if psi_oracle is not None and s == POINTS_DRAW` | `study.py:168-175` | builds the one designated draw's points/prototypes frames | MAIN | 8 |
| `per_est[name] = dict(...)` | `study.py:177-186` | the per-estimator result bundle `_run_dataset` writes | MAIN | 10 |
| `_qij_row` | `study.py:190-239` | one `qij.parquet` row: shared diagnostics + per-output `CoordinateResult`/influence-model fields + `bin_mass`/`bin_influence`/`bin_d2T` list columns | MAIN, every field computed (confirmed: all lines 211-239 executed) | 50 |
| `_quantile_labels` | `study.py:242-248` | M equal-count bin labels by rank, via one argsort | MAIN | 7 |
| `_btw_over_tot` | `study.py:251-261` | V_btw/V_tot of a value under a partition, via `np.bincount` | MAIN | 11 |
| `_btw_over_tot`'s `else float('nan')` | `study.py:261` | guards `v_tot > 0`; degenerate case | DATA -- undistinguishable from the trace at line level (one ternary line); triggered if a coordinate's total variance underflows to exactly 0 | - |
| `_partition_rows` | `study.py:264-290` | builds the M-grid rows: X-VQ, psi0-quantile, true-psi-quantile partitions | MAIN | 27 |
| `_points_frame` | `study.py:293-335` | `qij_points.parquet` rows for the designated draw | MAIN | 43 |
| `_prototypes_frame` | `study.py:338-356` | `qij_prototypes.parquet` rows for the designated draw | MAIN | 19 |
| `_truth_done` | `study.py:363-368` | resume marker: draws already in `truth.parquet` | MAIN (production I/O, not traced) | 6 |
| `_append_parquet` | `study.py:371-385` | append+dedupe-on-flush by key | MAIN (production I/O, not traced); both `os.path.exists` branches used across a real multi-flush run | 15 |
| `_write_designated_frame` | `study.py:388-393` | outright write for points/prototypes (no dedup needed) | MAIN (production I/O, not traced) | 6 |
| `_flush` | `study.py:396-408` | flushes truth/qij/partition buffers to parquet | MAIN (production I/O, not traced) | 13 |
| `_open_boot_h5` | `study.py:415-433` | opens/creates the fixed-shape `boot.h5` | MAIN (production I/O, not traced); both branches (`'a'` on resume, `'w'` fresh) used across the actual production run | 19 |
| `_write_boot_slot` | `study.py:436-441` | writes one draw's slot into `boot.h5` | MAIN (production I/O, not traced) | 6 |
| `_run_dataset` | `study.py:448-502` | per-(dataset, sample size) draw loop, joblib `Parallel` dispatch | MAIN (production I/O, not traced) | 55 |
| `_run_dataset`: `if not pending: return` | `study.py:462-463` | early exit when a dataset is already fully done | DATA -- triggered on resuming a run where this dataset's draws are all already written | 2 |
| `run_study` | `study.py:509-552` | one loop over datasets/sample sizes/draws | MAIN (production I/O, not traced) | 44 |
| OPTION config `master_seed` | `study.py:515` | main-path value 0 | MAIN | - |
| OPTION config `eps` (`config.get('eps', 0.01)`) | `study.py:516` | main-path value 0.01; every config on disk sets `eps` explicitly | MAIN (value always supplied; the literal default `0.01` fallback is never exercised since no config omits the key) | - |
| OPTION config `S` | `study.py:517` | main-path value 1000 | MAIN | - |
| OPTION config `B` | `study.py:518` | main-path value 2000 | MAIN | - |
| OPTION config `influence_model` (`config.get('influence_model','gp')`) | `study.py:526` | no config on disk sets this key, so every run takes the `'gp'` default | MAIN='gp' (default always applies); `'fitc'` = SWITCH (supported, unused) | - |
| OPTION config `fitc_rank` (`config.get('fitc_rank')`) | `study.py:527-528` | no config sets this key, so `fitc_rank` is always `None` | MAIN=`None`; a non-null int = SWITCH | - |
| OPTION config `datasets` (list of dicts) | `study.py:531-537` | drives the dataset/estimator/N loop | MAIN | - |
| `multi_N = len(N_list) > 1` | `study.py:539` | inserts an `N<size>` subdirectory when a dataset config carries more than one N | MAIN=False (every `main.yaml` dataset has one N); True = SWITCH, used only by `cost_vs_n.yaml`/`cost_vs_n_imf.yaml` | - |
| config key `levels` | `main.yaml` (not read anywhere in `study.py`) | downstream interval levels | UNREACHED in this scope -- never read by any file in scope; read only by the excluded `tables.py`/`figures.py` | - |

### `scripts/run.py` (45 lines)

| item | file:lines | what it does (one line) | status | lines |
|---|---|---|---|---|
| module | `run.py:1-46` | CLI entry: pins BLAS threads before numpy import, copies the config, calls `run_study` | MAIN (production entry point, not traced -- the tracer calls `_run_draw` directly) | 45 |
| BLAS thread pin loop | `run.py:14-16` | `os.environ.setdefault` for 4 thread-count vars, before `numpy`/`yaml` import | MAIN | 3 |
| CLI arg `config` (positional) | `run.py:28` | path to a study config YAML | MAIN | - |
| CLI arg `out_dir` (positional) | `run.py:29` | output directory | MAIN | - |
| CLI flag `--workers` | `run.py:30-32` | default `cpu_count()-1`; production run used `--workers 14` explicitly (module docstring's own example) | MAIN=14 (explicit, production); default `cpu_count()-1` = SWITCH (used only when the flag is omitted) | - |
| `shutil.copy(config, out_dir/config.yaml)` | `run.py:36` | archives the exact config used | MAIN | 1 |
| `main()` | `run.py:26-42` | argparse + `run_study` call | MAIN | 12 |

---

## Table P -- stored product fields

Sizes below are the sum over the 6 main-run cases (`fp/fp`, `mvt/nu`,
`mvt/tail`, `pareto/shape`, `pareto/tail`, `imf/chabrier`), read from
`QIJ_WSOM2026/runs/main/<dataset>/<estimator>/`.

**`truth.parquet`** (total 264,070 bytes / 258 KiB)

| product | field | written at file:line | what it is | READ-BY-PAPER or UNREAD | read at file:line |
|---|---|---|---|---|---|
| truth | `s` | `study.py:146` | draw index, join key | READ-BY-PAPER | `tables.py`/`figures.py` merge on `s` throughout |
| truth | `seed` | `study.py:146` | `master_seed+s` | UNREAD | no `df["seed"]`/`truth["seed"]` access; the only "seed" hits in `tables.py`/`figures.py` are an unrelated `rng_seed` plotting parameter |
| truth | `theta_true_<o>` | `study.py:148` | the estimand's true value | READ-BY-PAPER | `tables.py:131`, `figures.py:350` |
| truth | `theta_hat_<o>` | `study.py:149` | the estimator's own full-data value | READ-BY-PAPER | `tables.py:132`, `figures.py:351` |
| truth | `V_oracle_<o>` | `study.py:152` | `mean(psi_oracle^2)/N`, the oracle asymptotic variance | READ-BY-PAPER (tables.py only) | `tables.py:128,136`; NOT read by `figures.py` (`figures.py:314` comment: "no figure in this file compares against V_oracle any more") |

**`qij.parquet`** (total 7,433,553 bytes / 7.09 MiB)

| product | field | written at file:line | what it is | READ-BY-PAPER or UNREAD | read at file:line |
|---|---|---|---|---|---|
| qij | `s` | `study.py:211` | join key | READ-BY-PAPER | merges throughout both files |
| qij | `M_X` | `study.py:211` | prototypes actually used | UNREAD | only mentioned in comments (`figures.py:1455,1583`), never `df["M_X"]` |
| qij | `n_failed` | `study.py:211` | evaluations that failed this draw | READ-BY-PAPER | `tables.py:146`, `figures.py:361` |
| qij | `evals_prototype` | `study.py:213` | stage-1 evaluation count | UNREAD | no access |
| qij | `evals_full_data` | `study.py:213` | stage-2 base evaluation count | UNREAD | no access |
| qij | `evals_refinement` | `study.py:213` | refinement evaluation count | UNREAD | no access |
| qij | `evals_total` | `study.py:213` | whole-fit evaluation count | READ-BY-PAPER | `tables.py:142`, `figures.py:362` |
| qij | `rows_prototype` | `study.py:214` | stage-1 row count | UNREAD | no access |
| qij | `rows_full_data` | `study.py:214` | stage-2 base row count | UNREAD | no access |
| qij | `rows_refinement` | `study.py:214` | refinement row count | UNREAD | no access |
| qij | `rows_total` | `study.py:214` | whole-fit row count | UNREAD | no direct access (only `normalized_rows`, a separate stored field, is read) |
| qij | `wall_time_prototype` | `study.py:215` | stage-1 wall time | UNREAD | no access |
| qij | `wall_time_full_data` | `study.py:215` | stage-2 wall time | UNREAD | no access |
| qij | `wall_time_refinement` | `study.py:215` | refinement wall time | UNREAD | no access |
| qij | `wall_time_total` | `study.py:215` | whole-fit wall time | READ-BY-PAPER | `tables.py:139`, `figures.py:357,2028` |
| qij | `normalized_rows` | `study.py:216` | `rows_total/N` | READ-BY-PAPER | `tables.py:143,605,731`, `figures.py:363,1262,1607,1612` |
| qij | `V_btw_<o>` | `study.py:221` | the method's measured between-bin variance (the reported estimate) | READ-BY-PAPER | `tables.py:133`, `figures.py:352` |
| qij | `V_win_hat_<o>` | `study.py:222` | estimated within-bin variance (diagnostic) | UNREAD | only in comments/docstrings (`tables.py:363`, `figures.py:405`, past-tense "no figure ... any more") |
| qij | `V_tot_hat_<o>` | `study.py:223` | estimated total variance (diagnostic) | READ-BY-PAPER | `tables.py:134`, `figures.py:354` |
| qij | `B_hat_<o>` | `study.py:224` | BCa bias estimate | UNREAD | only in one comment (`tables.py:363`) |
| qij | `a_bca_<o>` | `study.py:225` | BCa acceleration | READ-BY-PAPER | `tables.py:135`, `figures.py:355` |
| qij | `L_<o>` | `study.py:226` | final 𝓘-VQ bin count | READ-BY-PAPER | `tables.py:141`, `figures.py:353` |
| qij | `n_level_splits_<o>` | `study.py:227` | refinement diagnostic | UNREAD | no access |
| qij | `n_adjacency_splits_<o>` | `study.py:228` | refinement diagnostic | UNREAD | no access |
| qij | `rho_<o>` | `study.py:229` | within-bin shape scale | UNREAD | no access |
| qij | `gain_ratio_<o>` | `study.py:230` | refinement stopping diagnostic | UNREAD | no access |
| qij | `n_refine_evals_<o>` | `study.py:231` | refinement evaluation count | UNREAD | no access |
| qij | `ell_<o>` | `study.py:232` | influence-model GP width | UNREAD | no access |
| qij | `lam_<o>` | `study.py:233` | influence-model GP noise/lambda | UNREAD | no access |
| qij | `ell_bound_<o>` | `study.py:234` | width at its bound flag | UNREAD | no access |
| qij | `lam_bound_<o>` | `study.py:235` | lambda at its bound flag | UNREAD | no access |
| qij | `bin_mass_<o>` | `study.py:236` | per-bin mass `p_k` (list column) | UNREAD | only in comments (`tables.py:22,101-102`) |
| qij | `bin_influence_<o>` | `study.py:237` | per-bin centered influence `U_k` (list column) | UNREAD | only in comments |
| qij | `bin_d2T_<o>` | `study.py:238` | per-bin uncentered 2nd difference (list column) | UNREAD | only in comments |

**`boot.h5`** (total 176,636,288 bytes / 168.5 MiB)

| product | field | written at file:line | what it is | READ-BY-PAPER or UNREAD | read at file:line |
|---|---|---|---|---|---|
| boot | `theta` | `study.py:437` | (S,B,q) replicate array | READ-BY-PAPER | `tables.py:115`, `figures.py:333` |
| boot | `s` | `study.py:428,438` | draw index per slot | READ-BY-PAPER | `tables.py:116` (`s_bootstrap`, for row alignment), `figures.py:334` |
| boot | `seed` | `study.py:429,439` | `master_seed+s` per slot | UNREAD | never `h5f["seed"]` in either file |
| boot | `wall_time` | `study.py:430,440` | bootstrap wall time per draw | READ-BY-PAPER | `tables.py:117`, `figures.py:335` |
| boot | `n_failed` | `study.py:431,441` | failed-replicate count per draw | READ-BY-PAPER | `tables.py:118`, `figures.py:336` |
| boot | attr `outputs` | `study.py:432` | output name order | READ-BY-PAPER | `tables.py` (col_order logic ~123), `figures.py:338,342` |

**`qij_partition.parquet`** (total 561,386 bytes / 548 KiB) -- **entirely UNREAD**

| product | field | written at file:line | what it is | READ-BY-PAPER or UNREAD | read at file:line |
|---|---|---|---|---|---|
| partition | `s` | `study.py:277` | draw index | UNREAD | `grep -n qij_partition` on both files returns zero read sites (only the docstring reference at `figures.py` comment lines) |
| partition | `M` | `study.py:277` | grid value | UNREAD | " |
| partition | `xvq_<o>` | `study.py:280` | X-VQ receptive-field V_btw/V_tot | UNREAD | " |
| partition | `ivq_psi0_<o>` | `study.py:284` | model-estimate-quantile V_btw/V_tot | UNREAD | " |
| partition | `ivq_true_<o>` | `study.py:288` | true-psi-quantile V_btw/V_tot (achievable bound) | UNREAD | " |

**`qij_points.parquet`** (total 910,880 bytes / 889 KiB)

| product | field | written at file:line | what it is | READ-BY-PAPER or UNREAD | read at file:line |
|---|---|---|---|---|---|
| points | `s` | `study.py:327` | draw index (the designated draw only) | UNREAD | not accessed as a column (file itself is opened per-draw by path, not filtered by `s`) |
| points | `i` | `study.py:327` | point index | UNREAD | no access |
| points | `bmu` | `study.py:327` | nearest prototype | READ-BY-PAPER | `figures.py:952,955,1088` |
| points | `psi0_<o>` | `study.py:330` | influence-model posterior mean at the point | UNREAD | no access |
| points | `psi_<o>` | `study.py:331` | oracle (true) influence | READ-BY-PAPER | `figures.py:955,1088,1100` |
| points | `sigma_<o>` | `study.py:332` | influence-model posterior sd at the point | UNREAD | no access |
| points | `psi_hat_<o>` | `study.py:333` | the method's refined influence estimate | READ-BY-PAPER | `figures.py:955,1088,1100` |
| points | `bin_label_<o>` | `study.py:334` | final 𝓘-VQ bin label | UNREAD | mentioned only in an error message string (`figures.py:1076,1095`), never accessed as `pts["bin_label_<o>"]` |

**`qij_prototypes.parquet`** (total 114,930 bytes / 112 KiB) -- **effectively UNREAD (dead read path)**

| product | field | written at file:line | what it is | READ-BY-PAPER or UNREAD | read at file:line |
|---|---|---|---|---|---|
| prototypes | `s` | `study.py:351` | draw index | UNREAD | no access |
| prototypes | `j` | `study.py:351` | prototype index | referenced in `figures.py:956` (`protos["j"]`) but only inside `_b_axis`/its caller, see note | see note |
| prototypes | `p` | `study.py:351` | receptive-field mass | UNREAD | no `protos["p"]` access anywhere |
| prototypes | `w_<d>` | `study.py:353` | prototype position in T's native coordinates | referenced in `figures.py:938-945` (`_b_axis`) | see note |
| prototypes | `I_<o>` | `study.py:355` | measured, mass-centred prototype influence | UNREAD | no `I_<o>`/`protos["I_...` access anywhere |

Note on `qij_prototypes.parquet`: `figures.py` defines `_b_axis(protos, axis_kind)` at
`figures.py:931-958`, which does read `protos["w_0"]`/`w_cols`/`protos["j"]` --
but `_b_axis` is never called anywhere in `figures.py` (`grep -n "_b_axis("`
finds only its own `def` line); `fig_b` (the function that would use it) only
reads `qij_points.parquet` (`figures.py:1050-1100`). So although a grep for
column names finds a "read" inside `_b_axis`, that function is dead code and
`qij_prototypes.parquet` is not actually loaded (no `pd.read_parquet(...
"qij_prototypes.parquet")` call exists in either `tables.py` or `figures.py`).
Treated as UNREAD for all fields, with this caveat recorded.

---

## Table 2 -- standards findings (MAIN and DATA code only)

| file:lines | rule | what |
|---|---|---|
| `bootstrap.py:40-41` | E1 | `for b in range(self.B): replicates[b] = T(X, rng.multinomial(...))` -- a Python loop over B replicate rows (B=2000 on the main path), one estimator evaluation per iteration. E1 names "replicates' rows" explicitly as disallowed. The module's own comment defends it on memory grounds (avoids holding B*N resample weights at once), not vectorizability -- confirmed executed via the trace (lines 40-41 in the executed set). |
| `core/xvq.py:176-181` | E1 | `for j in range(M_used): ... I_proto[j] = (counter(W_X, omega) - theta_Q) / t_j` -- a Python loop over M_used prototypes ("prototypes' points" in E1's own wording), one `counter` evaluation per prototype for its forward difference. Confirmed executed (all 6 lines in the trace's union). |
| `study.py:139` + `qij.py:117` | E6 | The estimator is evaluated at the exact same `(X, ones(N))` weights twice per draw: `study._run_draw:139` computes `theta_hat = T(X, ones)` for the truth row, and `qij.QIJ.fit:117` independently computes `theta_hat = counter(X, np.ones(N))` for its own full-data stage. Both are on the confirmed MAIN path (every traced task hits both lines); neither reuses the other's value. |
| `study.py:1-53` (module docstring) | C1 | 8 lines carrying history: cites "plan Sec 3/6/7", "interface sheet Amendment 6" (config schema), "Amendment 6" (no partition/points config key), "plan Sec 5" (vq_transform), "interface sheet Sec 5, amended" (product paths). |
| `study.py:69-100` (constants section + `_DATASET_ESTIMATORS`/`_VQ_TRANSFORM` comments) | C1 | 4 lines: "(plan Sec 3, Sec 7)" above the constants; "interface sheet Amendment 2" above `_DATASET_ESTIMATORS`; "(20 September)" naming when the Chabrier estimator was added, inside the `imf` dict-entry comment; "plan Sec 5, interface sheet Amendment 4" above `_VQ_TRANSFORM`. |
| `study.py:103-110` (`_pin_threads`) | C1 | 1 line: "(plan Sec 6)" in the docstring. |
| `study.py:118-187` (`_run_draw`) | C1 | 1 line: "(Amendment 6)" in the docstring, on why partition/points are keyed off `T.influence` rather than a config flag. |
| `study.py:190-239` (`_qij_row`) | C1 | 1 line (3 markers): "(interface sheet Sec 5, Amendment 3)" in the docstring's opening sentence. |
| `study.py:264-290` (`_partition_rows`) | C1 | 1 line: "(plan Sec 3)" in the docstring. |
| `study.py:338-356` (`_prototypes_frame`) | C1 | 1 line: "(plan Sec 3 \"What to add\" 1)" in the docstring. |
| `study.py:411-433` (boot.h5 I/O section + `_open_boot_h5`) | C1 | 2 lines: "(interface sheet Sec 5)" as the section header comment, and again in `_open_boot_h5`'s own docstring. |
| `study.py:509-552` (`run_study`) | C1 | 2 lines: "(plan Sec 6, module docstring's config schema)" in the docstring; "Amendment 6's 'no partition/points config key' spirit extends here too" in the `influence_model` comment. |
| `qij.py:1-14` (module docstring) | C1 | 4 lines: "plan §1, §6" (twice), "interface sheet" + "plan §4", "plan §7". |
| `qij.py:32-40` (`_wrap`) | C1 | 1 line: "plan §4" in the docstring. |
| `qij.py:43-44` (class `QIJ` docstring) | C1 | 1 line: "(plan §1, §6)". |
| `qij.py:53-59` (`__init__`'s `influence_model` comment) | C1 | 1 line: "(FITC smoke-test addition)" naming when/why the option was added. |
| `qij.py:63-68` (`fit` docstring) | C1 | 1 line: "(plan §6)". |
| `qij.py:79-84` (`fit`'s `vq_transform` comment) | C1 | 1 line: "(amendment 4 -- the MVT standardization depends on this draw's own mean and sd...)". |
| `qij.py:131-141` (`fit`'s failed-coordinate comment) | C1 | 1 line: "(plan §4)". |
| `result.py:1-6` (module docstring) | C1 | 2 lines: "(plan §1)" (twice), one naming "plan §3 'qij.parquet'". |
| `result.py:21-42` (`QIJResult` docstring) | C1 | 5 lines: "(plan §1, §3 'qij.parquet')", "(interface sheet §1)", "plan §3", "interface sheet §4 core/counter.py", "plan §1" again for oracle fields. |
| `result.py:74-80` (`.interval`) | C1 | 1 line: "(interface sheet §4 core/intervals.py)". |
| `result.py:82-89` (`.summary`) | C1 | 1 line: "(interface sheet §1)". |
| `result.py:106-109` (`BootstrapResult` docstring) | C1 | 1 line: "(plan §1)". |
| `result.py:124-127` (`BootstrapResult.summary`) | C1 | 1 line: "(interface sheet §1)". |
| `bootstrap.py:1-6` (module docstring) | C1 | 2 lines: "(plan §1)" (twice). |
| `bootstrap.py:17-18` (class docstring) | C1 | 1 line: "(plan §1)". |
| `check.py:1-11` (module docstring) | C1 | 3 lines: "plan §10, amendment 6" (title line), "(old plan §5.3)", "amendment 6" again for the labels-source. |
| `__init__.py:1-2` (module docstring) | C1 | 1 line: "(plan §1)". |
| `core/xvq.py:1-16` (module docstring) | C1 | 1 line: "Amendment 1" naming `run_xvq`'s role, "interface sheet Amendment 1". |
| `core/xvq.py:146-195` (`prototype_influences`) | C1 | 2 lines: "(ruled 18 September)" naming when/who decided the finite-prototype centering rule. |
| `core/xvq.py:198-218` (`run_xvq`) | C1 | 2 lines: "(method outline steps 1-3, interface sheet Amendment 1)". |

No E2 (per-group loop/groupby), E3 (explicit inverse/repeated factorization)
or E7 (append/concatenate/list-then-stack in a loop) findings in this
scope's MAIN/DATA code: `study._append_parquet`'s `pd.concat` and
`_flush`'s list-building are one-shot, at the I/O boundary, not inside a
per-draw or per-point loop, matching E7's own carve-out ("pandas only at
the I/O boundary"). `core/xvq.py`'s `_resolve_bmu2` batch loop is a
memory-bounded chunk loop (E4's own pattern), not a per-point loop.

---

## Summary

### Lines by status, per file (of this scope)

| file | total lines | MAIN | MAIN (production I/O, not traced) | DATA | SWITCH | UNREACHED |
|---|---|---|---|---|---|---|
| `__init__.py` | 7 | 7 | - | - | - | - |
| `check.py` | 49 | 49 (all 27 executable lines traced) | - | - | - | - |
| `bootstrap.py` | 49 | 49 (27 executable lines traced incl. E1 loop) | - | - | - | - |
| `qij.py` | 194 | ~180 (105 executed + surrounding structure) | - | - | ~10 (`'fitc'`/non-null `fitc_rank` values; not separately numbered lines, they are option VALUES on existing lines) | ~4 (`self.eta is not None` ternary branch, unreachable) |
| `result.py` | 135 | ~28 (dataclass field lines, both classes) | - | - | - | ~86 (`.variance`/`.interval`/`.summary` bodies, both classes, 9+7+22+4+3+12=57 body lines, plus their surrounding defs/docstrings) |
| `core/xvq.py` | 218 | ~206 | - | 12 (`_resolve_bmu2`'s repair loop, `xvq.py:132-143`) | - | - |
| `study.py` | 552 | 158 (executed, `_run_draw`'s graph) | ~180 (`run_study`, `_run_dataset`, `_truth_done`, `_append_parquet`, `_write_designated_frame`, `_flush`, `_open_boot_h5`, `_write_boot_slot`) | ~3 (`_run_dataset`'s early return; `_btw_over_tot`'s NaN ternary) | ~5 (`_DATASET_ESTIMATORS['imf']['imf']`, `multi_N=True` path -- option values, not separately-numbered code) | - |
| `scripts/run.py` | 45 | 45 (production entry, not traced) | - | - | - | - |

(Line counts above are approximate where a status applies to a fraction of
a shared line, e.g. one branch of a ternary or one value of a config-driven
default; exact executed-line sets are in `executed_lines.json`.)

### Every option, main-path value, and other supported values

| option | where | main-path value | other supported value(s) |
|---|---|---|---|
| `QIJ(eps=...)` | `qij.py:46` | 0.01 | none other used by any config |
| `QIJ(eta=...)` | `qij.py:46` | `None` (falls back to `T.eta`) | any float override -- never called with one anywhere in the repo (UNREACHED) |
| `QIJ(seed=...)` | `qij.py:46` | `master_seed+s` | any int |
| `QIJ(vq_transform=...)` | `qij.py:47` | `datasets.mvt_vq_transform` (mvt) / `None` (pareto, fp, imf) | any `(X)->(Z,inverse)` callable |
| `QIJ(influence_model=...)` | `qij.py:48` | `'gp'` | `'fitc'` (SWITCH -- unused by any config) |
| `QIJ(fitc_rank=...)` | `qij.py:48` | `None` | any int (SWITCH -- unused by any config) |
| `Bootstrap(B=...)` | `bootstrap.py:20` | 2000 (main.yaml); other configs use 50 (smoke) | any int |
| `Bootstrap(seed=...)` | `bootstrap.py:20` | `master_seed+s` | any int |
| config `master_seed` | `study.py:515` | 0 | any int (every config on disk uses 0, except `timing_draw.yaml` which overwrites it per SLURM array task) |
| config `eps` | `study.py:516` | 0.01 | every config on disk uses 0.01; the `.get` default is never exercised |
| config `S` | `study.py:517` | 1000 | 100 (cost_vs_n*), 200 (mvt_tail_n1000, validate), 3 (smoke), 20 (timing), 1 (timing_draw) |
| config `B` | `study.py:518` | 2000 | 50 (smoke) |
| config `influence_model` | `study.py:526` | `'gp'` (default; key absent everywhere) | `'fitc'` (SWITCH, no config sets it) |
| config `fitc_rank` | `study.py:527` | `None` (key absent everywhere) | any int (SWITCH, no config sets it) |
| config `datasets[].N` (single vs list) | `study.py:539` | single-element list (no `N<size>` subdir) | multi-element list -> `N<size>` subdir (cost_vs_n.yaml, cost_vs_n_imf.yaml) |
| config `datasets[imf].estimators` | `study.py:531-536` | `[chabrier]` | `[imf]` (two-regime; timing.yaml, timing_draw.yaml, cost_vs_n_imf.yaml) |
| config `levels` | present in every config, read only outside this scope | `[.50,.60,.70,.80,.90,.95,.99]` | not varied in any config; not read by any file in this scope |
| `scripts/run.py --workers` | `run.py:30` | 14 (production run's own documented invocation) | `cpu_count()-1` default when omitted |
| module constant `PARTITION_S` | `study.py:71` | 200 | single value, no alternate in code |
| module constant `POINTS_DRAW` | `study.py:72` | 0 | single value, no alternate |
| module constant `PARTITION_M_GRID` | `study.py:73` | `(4,6,8,12,16,24,32,48,64)` | single value |
| module constant `_FLUSH_EVERY` | `study.py:74` | 20 | single value |
| module constant `_KAPPA_REF` | `core/xvq.py:29` | 2.7 | single value |
| module constant `_BMU2_BATCH_CAP` | `core/xvq.py:34` | 4096 | single value |

### Comment+docstring lines vs code lines per file (this scope)

Counted as: total lines minus blank lines minus lines that are pure code
(no `#`/inside a `"""..."""` block); a line with trailing/inline comment is
not double counted here (kept simple, approximate, from the same reads
used to build Table 1/2).

| file | total lines | comment/docstring lines (approx) | code lines (approx) |
|---|---|---|---|
| `__init__.py` | 7 | 2 | 5 |
| `check.py` | 49 | 11 | 27 (+ blanks) |
| `bootstrap.py` | 49 | 15 | 27 (+ blanks) |
| `qij.py` | 194 | ~70 | ~105 (+ blanks) |
| `result.py` | 135 | ~55 | ~48 (+ blanks) |
| `bootstrap.py`+`check.py`+`__init__.py` | (see above) | | |
| `core/xvq.py` | 218 | ~95 | ~74 (+ blanks) |
| `study.py` | 552 | ~230 | ~230 (+ blanks) |
| `scripts/run.py` | 45 | ~10 | ~30 (+ blanks) |

Precise per-function C1 counts (history-carrying lines specifically,
narrower than "all comment lines") are in Table 2 above; total C1 lines
found across this scope: study.py 13, qij.py 9, result.py 11, bootstrap.py
3, check.py 3, `__init__.py` 1, `core/xvq.py` 5 -- 45 history-carrying
comment/docstring lines total across the 553+194+135+49+49+7+218 = 1205
lines of the module/class/dataclass/function-level files in this scope
(excluding `scripts/run.py`, which has none).

### Retired configs' differences

See the table under "Configs other than `main.yaml`" above.
