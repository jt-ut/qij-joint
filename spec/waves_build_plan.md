# Waves A and B: coordinator's build plan

Coordinator, 26 September 2026. Implements `spec/QIJ_mods_waves.md`
(the planner's spec, which governs). This file assigns the work to agents,
pins the interfaces between them, and holds their prompts. Code standards:
`qij_joint_plan.md` §5, included verbatim in every build prompt.

## Wave A

### Agents

Two builders in parallel on disjoint files, then the audit agent, then (on
the author's go) the measurement agent.

| Agent | Files | Spec items |
|---|---|---|
| A-GP | `core/influence_model.py`, `spec/method_notes.md` | A1 quadratic trend, A2 local width |
| A-PL | `core/counter.py`, `core/xvq.py`, `qij.py`, `result.py`, `parallel.py`, `pipeline.py`, `products.py`, `scripts/run.py`, `gmm.py` | A3 survey parallelism, A4 `M_X`, A6 GMM reference labelling, the switch plumbing |

A-PL writes its method-note text (survey parallelism, A6) to
`<scratch>/waves/A-PL_notes.md`; the coordinator merges it, so only A-GP
edits `method_notes.md`.

### Pinned interfaces

**Influence model** (`core/influence_model.py`, A-GP):

```
fit_influence_model(Z, xvq, I_proto, theta_Q, eta,
                    gptrend='affine', gpwidth='global') -> InfluenceModel
psi0(model, Z) / uncertainty(model, Z) / bin_posterior_variance(model, Z, coordinate, groups, sigma_c)
```

Signatures of the three query functions are unchanged: queries are always
at the draw's N points, so under `gpwidth='local'` the model stores each
point's length scale at fit time from `xvq.bmu` (A2: a point takes its
best-matching prototype's length). `InfluenceModel` gains `gptrend`,
`gpwidth` (the strings), `c` (q,) (the fitted global factor; NaN under
`global` and on the constant path), `h` (M_used,) (the local CONN spacing
per live prototype; computed under `local` only, NaN under `global`).
`width` stays the fitted ℓ under `global` and is NaN under `local`;
`at_bound[:, 0]` is the bound flag of whichever width parameter the switch
fits (ℓ or c). `m` becomes 1, d_z + 1 or 1 + d_z + d_z(d_z+1)/2.

**Counter** (`core/counter.py`, A-PL): `counter.add(evaluations, rows,
failed)` accounts for evaluations performed in workers; the parent calls it
with each task's returned bookkeeping (contract rule 4).

**Survey** (`core/xvq.py`, A-PL): `prototype_influences(W_X, counter, p,
eta, pool=None)` and `run_xvq(..., pool=None)`; with a pool the M_used
per-prototype evaluations are pool tasks (prototype index and its weights),
the base evaluation stays in the parent, `W_X` is shared once via
`pool.share`; `pool=None` is the ported loop. The survey's summed task
times feed `busy_time_total`.

**QIJ** (`qij.py`, A-PL): `QIJ(eps=0.01, seed=0, vq_transform=None,
gptrend='affine', gpwidth='global', M_X=None).fit(X, T, pool=None)`.
`M_X` given overrides `cost_rule_M`. `QIJResult` gains `gptrend`, `gpwidth`,
`M_X_source` ('rule' or 'argument'), `c` (q,), `c_bound` (q,),
`prototype_h` (M,); `busy_time_total` and `workers` become real.

**Products** (A-PL): the qij scalar row gains `gptrend`, `gpwidth`,
`M_X_source`, `c_<o>`, `c_bound_<o>` (with the ported `ell_<o>`,
`ell_bound_<o>` kept, NaN/False under `local`); the prototypes table gains
`h`. CLI: `--gptrend`, `--gpwidth`, `--M-X` on `qij`; `qij --workers W`
now runs the survey on a pool of W (the message saying it is not built
goes).

**GMM2D** (`gmm.py`, A-PL): `GMM2D(..., reference=None)`, `reference =
(means (K,2), covs (K,2,2))`; A6's Bhattacharyya assignment labels every
evaluation (`__call__`, `influence`, `fit_and_influence`) when given; no
reference is the ported μ_x order, bit for bit. A helper returns a fit's
`(means, covs)` in the reference format.

### Line budgets (whole file)

`influence_model.py` 1,100 (from 899); `gmm.py` 720; `xvq.py` 175;
`qij.py` 190; `result.py` 115; `counter.py` 65; `pipeline.py` 230;
others at their current budgets.

### Cost stated before dispatch

Builders: no runs beyond pyflakes and the one check. Audit: the audit
draws only, B = 20; the IMF QIJ fit is about 20 s per draw on this laptop,
run at 1 and 4 workers, so about 3 minutes; everything else seconds.
Measurement: dispatched separately on the author's go, under testing rule
3's 20-minute projection stop.

## Prompts

Each build prompt is: **Common preamble** + `qij_joint_plan.md` §5
verbatim + the testing rules of `QIJ_mods_waves.md` verbatim + the agent's
part.

### Common preamble

> You are one of two builders implementing wave A of
> `spec/QIJ_mods_waves.md` in the `qij_joint` package, repo
> `/Users/jtaylor/Dropbox/Software/JT_Py_Pkgs/qij_joint` (git, branch
> `main`, HEAD `589c803`; do not commit). Read first, in full:
> `spec/QIJ_mods_waves.md` (the spec; it governs), `spec/waves_build_plan.md`
> (your files and the interfaces you share with the other builder, who is
> working at the same time on disjoint files), and the parts of
> `spec/method_notes.md` your files implement. Never read the `vqboot`
> repository, even though your shell may start inside it; never read or
> write the old `qij` repo or anything under `~/Dropbox/Research`.
>
> The requirement: with every switch at its first value (`gptrend='affine'`,
> `gpwidth='global'`, no reference, no `M_X`), every number the package
> produces is bit-identical to HEAD at any worker count. Keep the ported
> code path unchanged for those values; add the new behaviour beside it.
>
> Environment: `python3.9`, `PYTHONPATH=<repo>/src`,
> `PYTHONDONTWRITEBYTECODE=1`, the five BLAS thread variables = 1. Edit
> with the Write/Edit tools. Scratch:
> `/private/tmp/claude-501/-Users-jtaylor-Dropbox-Software-JT-Py-Pkgs-vqboot-fable/c0fd00f7-ff54-4ae7-926b-df11406439fd/scratchpad/waves/`
> (use this path, never `$TMPDIR`). Verification is exactly: pyflakes clean
> on your files and `python3.9 -m qij_joint.check`; nothing else is run.
> When finished write `<scratch>/waves/<agent>_done.txt` (`done` or
> `failed: <reason>`) and report: files with line counts against budget,
> every interpretation of the spec you made, every standards conflict.

### A-GP: the regression

> Your files: `src/qij_joint/core/influence_model.py`,
> `spec/method_notes.md` (section 3 only). Implement A1 (`gptrend`) and A2
> (`gpwidth`) exactly as the spec states, to the interfaces pinned in
> `waves_build_plan.md`. Notes: under `gpwidth='global'` evaluate the
> ported kernel expression itself (A2's last paragraph); under `local` the
> 1-D running-sum shortcut for SS_k assumes a stationary kernel, so v_k and
> sigma go through the chunked pairwise path, batched so nothing forms an
> N x N array (standard E4); the per-candidate eigendecomposition shared
> across a group's outputs carries over to the search over c. The constant-
> only fallback uses the new basis size m. Add to method_notes section 3
> the quadratic basis, the local spacing, the non-stationary kernel and the
> search over c, as the method now is.

### A-PL: plumbing, survey parallelism, M_X, GMM labelling

> Your files: `src/qij_joint/core/counter.py`, `src/qij_joint/core/xvq.py`,
> `src/qij_joint/qij.py`, `src/qij_joint/result.py`,
> `src/qij_joint/parallel.py` (only if needed), `src/qij_joint/pipeline.py`,
> `src/qij_joint/products.py` (only if needed), `scripts/run.py`,
> `src/qij_joint/gmm.py`. Implement A3, A4 and A6 and the switch plumbing,
> to the interfaces pinned in `waves_build_plan.md`: `QIJ` passes
> `gptrend`/`gpwidth` to `fit_influence_model` (A-GP is adding those
> parameters now; code against the pinned signature). The survey on a pool
> follows the parallelism contract of `qij_joint_plan.md` §2 (workers get
> the estimator once at pool start, `W_X` once per draw via `share`, each
> task returns its outputs, own wall time and failure flag; the parent
> assembles in prototype order and calls `counter.add`). `qij.py`'s
> `lambda` for the inverse transform never crosses the pool (the survey
> ships native-coordinate `W_X`, not the transform). In `pipeline.run_qij`
> create one pool per run when `workers > 1` and pass it to every draw's
> fit. For A6 follow the spec's formula and `linear_sum_assignment`; the
> no-reference path is untouched. Write your method-note text (survey
> parallelism, A6 labelling) to `<scratch>/waves/A-PL_notes.md`; do not edit
> `spec/method_notes.md`.

### Audit (wave A, after both builders)

> Audit wave A of `qij_joint` (`spec/QIJ_mods_waves.md` A5 and testing
> rules 1-6; `spec/waves_build_plan.md`). Do not modify the package. Stored
> baseline (the products of `qij_joint@589c803`):
> `<scratch>/audit/new/` (oracle, ij, qij) and `<scratch>/audit/new_w1/`
> (boot, B 2000), scratch =
> `/private/tmp/claude-501/-Users-jtaylor-Dropbox-Software-JT-Py-Pkgs-vqboot-fable/c0fd00f7-ff54-4ae7-926b-df11406439fd/scratchpad`.
> Draws: s in {0, 1, 999} for pareto shape, pareto tail, mvt nu, mvt tail,
> fp all, imf all, plus mvt nu 342, 420, 523; N 2000, seed 0, eps 0.01.
> Run with the new CLI into `<scratch>/waves/audit_A/`: `oracle` and `ij`
> at `--workers 3`; `boot` at `--B 20 --workers 1`; `qij` with the default
> switches at `--workers 1` and again at `--workers 4` (two output dirs),
> `--diag-draws 0:1`. Compare every field present in both, bit-exact
> (`==`, NaN equal to NaN), excluding timing fields (`wall_time*`,
> `busy_time*`, `workers`); boot replicates against the first 20 stored
> replicates of each draw. List every new column with its values (they
> have no baseline). Run CLI commands with the shell sandbox disabled
> (the package calls `sysctl`; loky calls `os.sysconf`); use the explicit
> scratch path, never `$TMPDIR`; if a command dies at start-up with a
> macOS `dyld` assertion, rerun it once and note it. Nothing else is run.
> Report one table: field groups by case, exact or not, and every
> difference in full.
