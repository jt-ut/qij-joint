# Cleanup before the port (2026-11-06)

## Tag
`archive/pre-port-20261106` (lightweight, on HEAD `e6d6570`, not pushed). Every
deleted module is recoverable by name, e.g.
`git show archive/pre-port-20261106:src/qij_joint/core/tree.py`. No commits made.

## Deleted (the earlier forms of the variance-estimation loop, replaced by the port)

| module | lines before | after |
|---|---:|---:|
| qij.py | 692 | deleted |
| qijt.py | 545 | deleted |
| qijdt.py | 735 | deleted |
| ijfd.py | 572 | deleted |
| core/influence_model.py | 1411 | deleted |
| core/joint.py | 714 | deleted |
| core/affine_pilot.py | 260 | deleted |
| core/sigma_points.py | 198 | deleted |
| core/calibrate.py | 230 | deleted |
| core/tree.py | 1093 | deleted |
| core/ivq.py | 228 | deleted |

Kept: core/abc.py (the port's intervals; BootstrapResult.bc_interval), core/xvq.py,
core/differences.py, core/eta.py, core/counter.py.

## Changed

| module | before | after | change |
|---|---:|---:|---|
| pipeline.py | 716 | 223 | run_oracle / run_ij / run_boot byte-identical to HEAD; old run_qij, run_ijfd, run_qijt, run_qijdt and all their row/product writers removed; one new run_qij |
| result.py | 351 | 71 | QIJResult removed; BootstrapResult and `_normal_interval` kept |
| __init__.py | 6 | 15 | exports Bootstrap, run_qij; `run_method` exported only once `method.py` exists |
| check.py | 49 | 56 | rewritten as a smoke script (see below) |
| audit.py | 46 | 46 | kept unchanged: run_oracle and run_boot use `search_audit` |
| products.py | 280 | 185 | old-method `dir_tag` branches, `QIJ_CANONICAL` and `ensure_config`'s old-key shims removed (second pass) |

### New run_qij
`run_qij(dataset, estimator, N, draws, seed, out_dir, eps=0.01, level=0.84, M=None,
workers=None, tag=None, force=False)`. `force` was added (not in the brief) to keep the
other runners' resume behaviour: a draw whose row exists is skipped. The method is
imported lazily inside the function (`from .method import run as run_method`), so the
package imports before `method.py` exists. Pool: `Pool(workers, T=T)` as in run_boot
(`workers=None` -> 1, in-process), passed as `pool=`; `seed = seed + s`, as the other
runners. Products through `products.method_dir(..., 'qij', tag or '')` and
`products.write_draw` (parquet, as run_boot/run_ij):
- `s<draw>.parquet` row: dataset, estimator, N, s, seed, eps, level, M, K, rounds, E,
  E_full, wall_s, workers, `Vhat_/U_/delta_/V_lb_<out>`, `lo_/hi_<out>` when
  `intervals` is not None, and the five timings. The draw column is named `s`, not
  `draw`, because `products.collect` sorts on `s`.
- `s<draw>.rounds.parquet` and `s<draw>.leaves.parquet`: one row per dict; any
  per-output array value of length J is spread into `<key>_<out>` columns.

### check.py
Kept, rewritten: `python -m qij_joint.check` runs run_qij for one draw of pareto/shape
at N=200 into a temp dir and reads the three products back. If `qij_joint.method` is
absent it substitutes a stub `run` with the contract's fields (placeholder zeros), so
the runner and writers are exercised; the printout says whether the method or the stub
ran. Run today: stub (method.py not yet present); row 23 columns, rounds 8, leaves 5.

## Import fixes
- `__init__.py`: dropped `from .qij import QIJ`.
- `check.py`: dropped `from .qij import QIJ` (rewritten).
- `pipeline.py`: dropped `IJFD`, `QIJ`, `QIJDT`, `QIJT` imports.
- `result.py`: no longer referenced by any deleted module; `bootstrap.py` still imports
  `BootstrapResult` from it.

## spec/
- Moved to `spec/archive/` (41 files): every `QIJ_*` spec, proposal, interface, plan,
  draft, decision and check spec, `QIJ_mods_waves.md`, `QIJ_state_brief.md`,
  `QIJ_talk_handoff.md`, `waves_build_plan.md` (the mods-waves build plan) and
  `method_notes.md` (the superseded method's derivations). `QIJ_estimator_fit_spec.md`
  and `QIJ_mods_waves.md` had uncommitted edits; they moved with them. 22 of the 41
  were untracked and remain untracked in the archive.
- Added `spec/METHOD_qij.md`: verbatim copy of `vqboot/fable/bed/METHOD_qij.md`
  (281 lines) under a one-line origin comment.
- Left in place (unclear whether superseded design or package reference):
  `glossary.md` (old X-VQ/I-VQ notation), `qij_joint_plan.md`, `qij_inventory_rulings.md`,
  `audit_report.md`, `inventory/`.

## Second pass (coordinator, same day): the leftovers, finished

1. `scripts/run.py` (169 -> 82 lines): methods are `oracle`, `ij`, `boot`, `qij`;
   `qij` takes `--eps` (0.01), `--level` (0.84), `--M` (default: the method's own) and
   calls the new `run_qij(..., eps, level, M, workers, tag, force)`. All qijt / qijdt /
   ijfd choices and the old qij flags (`--diag-draws`, `--gptrend`, `--gpwidth`,
   `--M-X`, `--budget*`, `--survey`, `--quantized-start`, `--sigma-points`,
   `--fit-weights`, `--z`, `--n-min`, `--L-max`, `--pilot`, `--gp-floor`,
   `--point-curvature`) are removed. `--help` runs.
2. `products.dir_tag`: `qij` -> `eps<eps>`, plus `_level<level>` when not 0.84 and
   `_M<M>` when M is given; `boot`/`oracle`/`ij` -> `''`. The qijt/qijdt/ijfd branches
   and `QIJ_CANONICAL` are gone. `ensure_config` now compares the stored config
   exactly. The old shims (fit_weights, z/n_min/L_max, pilot/gp_floor, and ignoring
   unknown stored keys) are removed, so a folder written by a deleted method, e.g. an
   old default `qij_eps0.01`, raises instead of accepting new products beside old ones.
3. Scripts deleted. Each drove only a deleted method:
   - `oracle_hybrid_cadj_gp_assessment.py` (160): oracle study of the old GP influence model on selective CADJ cells.
   - `oracle_vor2_gp_assessment.py` (215): old GP influence model on Vor1 vs directed-Vor2 partitions.
   - `selective_cadj_oracle_draw1.py` (277): selective directed-CADJ peeling scored through the old GP pilot.
   - `oracle_cadj_contrast_conditioning.py` (257): conditioning the old GP posterior on CADJ contrasts.
   - `oracle_aggregate_moment_gp.py` (263): point- vs aggregate-moment observation model for the old GP.
   - `end_to_end_moment_gp_draw1.py` (172): the old QIJ end to end with the aggregate-moment GP mean swapped in.
   - `qijdt_validate_v1..v4.py` (233/570/153/163): validation runs of the deleted qijdt.
   - `qijt_validate_v1..v4.py` (220/499/163/105): validation runs of the deleted qijt.
     These eight were not among the seven import breakers, but they drove `run.py qijt/qijdt` with the old qij flags, so they were dead too.
   Fixed: `validate_gmm_fit.py`, `validate_gmm_search.py` (docstrings no longer name `ijfd`).
   Left: `oracle_metric_vq_test.py` imports QIJT from an external `--qijt-root`, not
   from this package, so it is unaffected.
4. `tests/` deleted (its only file was `tests/oracle_metric/test_oracle_metric_quantizer.py`,
   34 lines). It was a pytest file checking that `scripts/oracle_metric_quantizer.py`'s
   local pull-back metric is symmetric positive-definite and that its generalized Lloyd
   is deterministic for a fixed budget (12 centres, every weight non-zero). The script
   itself is kept.
5. Docstrings: `core/__init__.py` (module list), `cloudfil.py`/`cloudfil_u.py`
   (`periodic` no longer said to be read by `core/sigma_points.py`; `core/counter.py`
   still copies it), and `pipeline.run_ij` (the `ijfd` sentence removed; its code is
   unchanged). Every `spec/QIJ_*.md` / `spec/method_notes.md` path in a docstring now
   points to `spec/archive/` in bootstrap, cloudfil, cloudfil_u, datasets, gmm, gmm_param,
   registry, result, seeding, shearmix and shearmix_model.
   Not touched (the port agent's files): `core/abc.py` (names
   `core.influence_model`), and the old `spec/` paths in `core/abc.py`, `core/xvq.py`,
   `core/differences.py`, `core/eta.py`, `core/counter.py` and `parallel.py`.
6. `result._normal_interval` was left for the port agent to decide.

## Still open
- `spec/glossary.md`, `qij_joint_plan.md`, `qij_inventory_rulings.md`, `audit_report.md`
  and `inventory/` stay in `spec/`, as before.
