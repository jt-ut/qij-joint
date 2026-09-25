# Inventory C3 — `src/qij/estimators.py`, `src/qij/datasets.py`, `src/qij/gmm.py`, `src/qij/data/`

Repo: `/Users/jtaylor/Dropbox/Software/JT_Py_Pkgs/qij` @ `025613e` (read-only). Classification
is grounded in `<scratch>/inventory/executed_lines.json`, `executed_by_task.json` and
`trace_notes.md` (tracer's run: main.yaml's 6 (dataset,estimator) cases at N=2000, s=0,1 each,
plus mvt/nu s=342 and s=420 — the only two of the three TACC-failing draws in the whole S=1000
main study, `n_failed>0` and NaN `V_btw_*` together — plus `qij.check` and one `qij.gmm`
fit+influence call on `ChaconMixGenerator(9).sample(2000, random_state=0)`, K=3). Per
`trace_notes.md`, **no other case (pareto shape/tail, mvt tail, fp, imf/chabrier) had any
failing draw anywhere in the full S=1000 main study** — mvt/nu's 3 failures (s=342,420,523) are
the only ones on record. Where a status below is a plain reading of `executed_lines.json`
membership it is not separately flagged "(unconfirmed)"; the few items resolved from
`main.yaml`/`study.py`/`check.py` alone (not exercised by the trace's own six-plus-two-plus-two
tasks) are noted as such inline.

Line counts in the `lines` column are inclusive span lengths (def-to-def or comment-block
spans), not just executed-line counts, so that Table 1 + Table 3 rows sum to each file's total.

---

## Table 1 — `src/qij/estimators.py` (1473 lines)

| item | file:lines | what it does (one line) | status | lines |
|---|---|---|---|---|
| module docstring + imports | estimators.py:1-52 | module contract (never raises, no state, `w` sums to len(X)); imports; `EPS` constant | MAIN | 52 |
| `estimator(outputs, eta, name, supports)` | estimators.py:55-82 | decorator attaching `name/outputs/eta/supports` to a closed-form fn | MAIN | 28 |
| — branch: `supports=None` default → all `(-inf,inf)` | estimators.py:77-78 | used by `weighted_mean` (only caller passing no `supports`) | MAIN (runs at import regardless of trace scenario) | 2 |
| — branch: `supports=[...]` given | estimators.py:79-80 | used by every other decorated fn | MAIN | 2 |
| `_weighted_quantile(x,w,q)` | estimators.py:85-91 | weighted linear-interpolated quantile, O(N log N) | SWITCH — only caller is `IMF._moment_start` (removed path); def-line executes at import, body never runs. For: seeding IMF's Mstar/p moment start | 7 |
| `_BOX_TOL` | estimators.py:108 | box-rule tolerance, fraction of box width | MAIN | 1 |
| `_at_bound(value,lo,hi,tol)` | estimators.py:111-121 | true if `value` within `tol`·width of either bound | MAIN — called by `mvt_nu` (main path) and by `IMF`/`Chabrier`'s own `_at_box` (shared helper; kept regardless of the IMF ruling since `mvt_nu` and `Chabrier` need it) | 11 |
| `weighted_mean(X,w)` | estimators.py:128-138 | `T=sum(wx)/sum(w)` | UNREACHED — not in `study._DATASET_ESTIMATORS` under any dataset; not called by `qij.check` (which defines its own local `_weighted_mean`, see below); only touched via `.outputs`/`.supports` attribute reads in the `SUPPORT_BY_OUTPUT` loop, never invoked. Decorator line 128 and `def` line 129 execute (import-time); body (130-138) never does. For: the "closed forms are exact algebraic evaluations" reference example named in the module docstring — never wired to a dataset | 11 |
| `_weighted_mean_influence(X,w)` | estimators.py:141-150 | `psi_i = x_i - theta` | UNREACHED — same as above | 10 |
| Pareto constants | estimators.py:157-161 | `PARETO_X_MIN`, `PARETO_ALPHA_TRUE`, fixed 99th-pct threshold `PARETO_TAIL_C` | MAIN | 5 |
| `pareto_shape(X,w)` | estimators.py:164-176 | Hill estimator `alpha=sum(w)/sum(w·log(x/x_min))` | MAIN (pareto/shape is a main-study case) | 13 |
| — `if not isfinite(alpha): NaN` | estimators.py:171-172 | fails when `sum(w·log_ratio)<=0` | DATA, not executed in the trace's 2 draws or their bootstrap/QIJ perturbations; trigger: a weight perturbation or draw pushing the weighted log-ratio sum to ≤0. Never fired in the full S=1000 pareto/shape run either (no known failures) | 2 |
| — `except Exception: NaN` | estimators.py:174-175 | catch-all failure boundary (R7) | DATA, not executed; trigger: malformed X (e.g. non-positive mass, not possible from `datasets.pareto`) | 2 |
| `_pareto_shape_influence(X,w)` | estimators.py:178-189 | `psi=1/alpha-log(x/x_min)`, `A=1/alpha^2`; **re-calls `pareto_shape(X,w)`** to get theta | MAIN | 12 |
| — `if not all(isfinite(theta))` | estimators.py:181-182 | NaN passthrough when the re-derived fit failed | DATA, not executed (never triggered) | 2 |
| `pareto_tail(X,w)` | estimators.py:192-201 | `T=sum(w·1{x>c})/sum(w)`, c fixed at true 99th pct | MAIN | 10 |
| — `except Exception: NaN` | estimators.py:200-201 | failure boundary | DATA, not executed | 2 |
| `_pareto_tail_influence(X,w)` | estimators.py:204-214 | `psi=1{x>c}-theta`; **re-calls `pareto_tail`** | MAIN | 11 |
| — NaN passthrough | estimators.py:207-208 | | DATA, not executed | 2 |
| MVT constants | estimators.py:221-228 | `MVT_D`, `MVT_NU_TRUE`, fixed `MVT_TAIL_C`, root-find bracket/extension constants | MAIN | 8 |
| `_mvt_score_terms`, `_mvt_score_weighted` | estimators.py:231-244 | per-observation profile score for nu, weighted mean | MAIN | 14 |
| `mvt_nu(X,w)` — nominal bracket + `brentq` | estimators.py:246-292 (core: 260-272, 280, 283-284, 286, 290) | root-find of the weighted profile score in log(nu) | MAIN — both the direct-`brentq` success path (mvt/nu s=0,1) **and** the failure paths below all executed, because `T` is evaluated at many bootstrap/QIJ-perturbed weight vectors per draw, not just `w=ones` | 47 |
| — bracket-extension while-loops | estimators.py:273-278 | geometrically widen `[log_lo,log_hi]` toward `[log(0.1),log(1e6)]` when the nominal `[0.5,200]` bracket doesn't contain a root | MAIN — executed on the two TACC-failing draws (s=342, 420) | (in 47) |
| — one-signed-score cap branch | estimators.py:280-281 | returns the appropriate cap when the score never changes sign over the whole admissible range | MAIN — executed on s=342/420 | (in 47) |
| — box-rule NaN | estimators.py:288-289 | counted failure when the root/cap rests on `[log(0.1),log(1e6)]`'s own boundary | MAIN — executed on s=342/420 (this is the mechanism behind the 3 known TACC failures) | (in 47) |
| — `except Exception: NaN` | estimators.py:291-292 | failure boundary | DATA, not executed | 2 |
| `_mvt_nu_influence(X,w)` | estimators.py:295-311 | `psi=g+h` at `nu_hat`; `A` by central difference; **re-calls `mvt_nu(X,w)`** (a full re-root-find) | MAIN | 17 |
| — NaN passthrough (295→298) | estimators.py:298-299 | when `mvt_nu` failed | MAIN — executed (paired with the s=342/420 box-rule failures above) | (in 17) |
| — `A` non-finite/near-zero NaN | estimators.py:309-310 | central-difference curvature degenerates | DATA, not executed | (in 17) |
| `mvt_tail(X,w)` | estimators.py:317-326 | `T=sum(w·1{‖x‖>c})/sum(w)` | MAIN | 10 |
| — `except Exception: NaN` | estimators.py:325-326 | | DATA, not executed | 2 |
| `_mvt_tail_influence(X,w)` | estimators.py:329-338 | `psi=1{‖x‖>c}-theta`; **re-calls `mvt_tail`** | MAIN | 10 |
| FP population load (module-level) | estimators.py:345-352 | loads `fp_sdss.npz` at import, computes `FP_POP_MEAN`/`FP_POP_STD`, deletes the raw array | MAIN (unconditional module-level code; runs on every import, i.e. every task) | 8 |
| `_fp_standardize(X)` | estimators.py:355-356 | `(X-FP_POP_MEAN)/FP_POP_STD` | MAIN | 2 |
| `_fp_tls_fit(X_std,w)` | estimators.py:359-380 | weighted total-least-squares plane via one 3×3 `eigh` | MAIN | 22 |
| — `if eigvecs[2,2]<0: flip sign` | estimators.py:370-371 | canonicalizes the TLS normal's sign, data-dependent (LAPACK's own sign is arbitrary) | MAIN — executed (occurred in at least one of the traced fp draws/perturbations) | (in 22) |
| `fp(X,w)` | estimators.py:383-392 | wraps standardize+TLS, returns `(a,b,c,scatter)` | MAIN | 10 |
| — `except Exception: NaN(4)` | estimators.py:391-392 | | DATA, not executed | 2 |
| `_fp_compute_if(X_std,fit)` | estimators.py:395-420 | closed-form implicit-function influence, ported from `vqboot.dgp.fundamental_plane` | MAIN | 26 |
| — `IF_s = .../scatter if scatter>1e-12 else 0` | estimators.py:418 | guards the scatter-influence division | MAIN (line executes; which side of the ternary fired is not separable from line coverage) | (in 26) |
| `_fp_influence(X,w)` | estimators.py:423-433 | re-derives `X_std`/TLS fit (**a second full TLS fit** of the same `(X,w)`), then `_fp_compute_if` | MAIN | 11 |
| — `except Exception: NaN` | estimators.py:429-430 | | DATA, not executed | 2 |
| **IMF removal-set items** | estimators.py:436-1057 (minus lines shared with Chabrier) | see **Table 3** | SWITCH (whole block) | see Table 3 |
| Chabrier section header + `CHABRIER_M_B_DEFAULT` | estimators.py:1052-1059 | comment + fixed default break mass | MAIN | 8 |
| `_chabrier_norm(a,sigma,slope,m_b,m_min)` | estimators.py:1062-1144 | `Z(theta)` and its 1st/2nd partials for the truncated Chabrier density | MAIN | 83 |
| `_chabrier_score_hessian(masses,w,m_b,m_min,m_c,sigma,slope)` | estimators.py:1147-1241 | `(psi, A, Z)` for Chabrier at fitted params, one M-estimator construction | MAIN | 95 |
| `Chabrier` class docstring + `name/outputs/supports/eta` | estimators.py:1244-1289 | class contract; `supports=((0,inf),(0,inf),(0,inf))` — unlike IMF, `x>0` is a genuine support (bare power-law tail) | MAIN | 46 |
| `Chabrier.__init__(m_min, m_b, bounds)` | estimators.py:1291-1295 | fixes `m_b`/`m_min`/optimizer box at construction | MAIN | 5 |
| `Chabrier._at_box(params)` | estimators.py:1297-1304 | box rule on `(m_c,sigma,x)` directly (no reparam) | MAIN | 8 |
| `Chabrier._moment_start(...)` | estimators.py:1306-1330 | lognormal mean/std below `m_b`, un-inverted Hill slope above (needs no inversion, unlike IMF's) | MAIN | 25 |
| `Chabrier.__call__(X,w)` | estimators.py:1332-1410 | trust-constr fit with exact grad/Hessian from `_chabrier_score_hessian` | MAIN | 79 |
| — `if n_lo<3 or n_hi<3: NaN` | estimators.py:1340-1341 | too few points on one side of `m_b` | DATA, not executed; trigger: a pathological resample with <3 points above/below the 1 M☉ break | 2 |
| — `nll`'s infeasibility/exception penalty branches | estimators.py:1363-1364, 1368-1369, 1370-1371 | `_NLL_PENALTY` returned when params ≤0, `_chabrier_score_hessian` raises, or `Z` degenerates | DATA, none executed; trigger: optimizer probing a boundary point or a near-singular normalizer | 6 |
| — `hess`'s analogous guards | estimators.py:1387-1388, 1392-1394 | `np.eye(3)` fallback | DATA, not executed | 5 |
| — `if not res.success: NaN` | estimators.py:1404-1405 | trust-constr non-convergence | DATA, not executed (0/1000 imf/chabrier failures in the main study) | 2 |
| — `if self._at_box(res.x): NaN` | estimators.py:1406-1407 | fit parks on Chabrier's own box (plan Sec 4 ruling, shared box-rule convention) | DATA, not executed | 2 |
| `Chabrier.influence(X,w)` | estimators.py:1412-1440 | `IF=A^-1 psi` from `_chabrier_score_hessian`; **re-calls `self(X,w)`**, i.e. a second full trust-constr fit, to get `theta` | MAIN | 29 |
| — NaN branches (1428-1429 at-box/non-finite theta; 1436-1437 non-finite IF) | estimators.py:1428-1429, 1436-1437 | | DATA, none executed | 4 |
| — `except Exception: NaN` | estimators.py:1439-1440 | | DATA, not executed | 2 |
| `SUPPORT_BY_OUTPUT` construction | estimators.py:1444-1459 | builds `{output_name: (lo,hi)}` once from every estimator's own declared supports, over `(weighted_mean, pareto_shape, pareto_tail, mvt_nu, mvt_tail, fp, IMF, Chabrier)` | MAIN (module-level loop always runs at import; it reads `IMF`/`weighted_mean`'s `.outputs`/`.supports` attributes only, never calls them) | 16 |
| `supports_for(outputs)` | estimators.py:1462-1473 | `(q,2)` support-bounds array for a list of output names, `.get(...,(-inf,inf))` fallback | UNREACHED — no caller anywhere in `src/` or `scripts/` (grepped; zero hits besides its own definition) | 12 |

## Table 1 — `src/qij/datasets.py` (143 lines)

| item | file:lines | what it does | status | lines |
|---|---|---|---|---|
| module docstring + imports | datasets.py:1-25 | dataset contract; `_DATA_DIR` | MAIN | 25 |
| Pareto/MVT constants | datasets.py:28-33 | `PARETO_ALPHA/X_MIN`, `MVT_D/NU` | MAIN | 6 |
| `IMF_TAU`, `IMF_BOUNDS` | datasets.py:35-39 | fixed breakpoint 0.176609 and (alpha,Mstar,p) optimizer box for the two-regime IMF | see **Table 3** (IMF removal set) | 5 |
| `CHABRIER_M_B`, `CHABRIER_M_MIN`, `CHABRIER_BOUNDS` | datasets.py:41-51 | Chabrier's fixed break, fixed pool minimum mass (hardcoded literal `0.0017582750879228115`, not computed at runtime from `_imf_pool().min()` despite the comment saying that's its origin), optimizer box | MAIN | 11 |
| `_PARAMETRIC_TRUTH` dict | datasets.py:53-58 | closed-form `theta_true` for pareto shape/tail, mvt nu/tail | MAIN | 6 |
| `pareto(N,seed)` | datasets.py:61-65 | inverse-CDF draw | MAIN | 5 |
| `mvt(N,seed)` | datasets.py:68-73 | normal/sqrt(chi²/nu) construction, d=10 | MAIN | 6 |
| `_fp_pool()` | datasets.py:76-79 | loads `fp_sdss.npz`'s `fp_data` (76,997×3), **`@functools.lru_cache(maxsize=1)`** | MAIN — carried state: process-lifetime cache of the whole population array | 4 |
| `fp(N,seed)` | datasets.py:82-87 | draw N rows with replacement from `_fp_pool()` | MAIN | 6 |
| `_imf_pool()` | datasets.py:90-93 | loads `stars.h5`'s `data/BH_Mass` (19,857,), **`@functools.lru_cache(maxsize=1)`** | MAIN — carried state, same pattern | 4 |
| `imf(N,seed)` | datasets.py:96-101 | draw N with replacement from `_imf_pool()` | MAIN | 6 |
| `mvt_vq_transform(X)` | datasets.py:104-119 | per-coordinate mean/sd whitening; returns `(Z, inverse)`, `inverse` closes over this call's own mean/std | MAIN — passed to `QIJ(vq_transform=...)` for MVT draws only | 16 |
| `truth(dataset, estimator)` | datasets.py:122-143 | `theta_true`: closed form or population-estimator fit; **`@functools.lru_cache(maxsize=None)`** | MAIN — carried state: unbounded process-lifetime memo, keyed on `(dataset, estimator)` | 22 |
| — `key in _PARAMETRIC_TRUTH` branch | datasets.py:129-130 | pareto/mvt closed forms | MAIN | (in 22) |
| — `('fp','fp')` branch | datasets.py:132-134 | fits `estimators.fp` on the whole pool, unit weights | MAIN | (in 22) |
| — `('imf','imf')` branch | datasets.py:135-138 | fits the two-regime `estimators.IMF` on the whole pool | see **Table 3** — condition (135) is checked every call (always False on the traced path) but the body (136-138) never executes; only reachable if a config's `imf` dataset requests the `imf` estimator (main.yaml requests only `chabrier`) | 4 (body) |
| — `('imf','chabrier')` branch | datasets.py:139-143 | fits `estimators.Chabrier` on the whole pool | MAIN | 5 |

## Table 1 — `src/qij/gmm.py` (973 lines)

Not on the paper's main path (`GMM2D` is never referenced by `study.py`/`qij.py`/`scripts/`); status below is against the tracer's own direct `fit`+`influence` calls on `GMM2D` (K=3, ChaconMix 9 sample, N=2000).

| item | file:lines | what it does | status | lines |
|---|---|---|---|---|
| module docstring | gmm.py:1-243 | full design rationale (two-phase multi-start, Louis's identity, `reg_covar`, per-call overrides, centering) | MAIN (string-statement counts as executed at import) | 243 |
| module constants | gmm.py:245-258 | `_LOG2PI`, `_ACCEPT_TOL`, `_COND_MAX=1e12`, `_MAX_NEWTON=20`, Newton tolerances, `_ETA_DEFAULT=1e-12` | MAIN | 14 |
| `_SHORT_ITERS=25`, `_PROMOTE_N=3` | gmm.py:260-269 | two-phase multi-start budget/promotion count, "chosen by measurement" | MAIN | 10 |
| `_D`, `_STYPES`, `_Cfg` namedtuple | gmm.py:271-278 | covariance-derivative basis matrices; per-call resolved config | MAIN | 8 |
| `_make_outputs(K)`, `_make_supports(K)` | gmm.py:285-296 | output names / support bounds, pure functions of K | MAIN | 12 |
| `_idx_mu(K,k)`, `_idx_S(K,k)` | gmm.py:299-306 | index layout helpers | MAIN | 8 |
| `_pack(K,...)`, `_unpack(K,...)` | gmm.py:309-332 | flatten/unflatten `(pis,mus,Ss)` ↔ `theta`; each has a `for k in range(K)` loop (K small, body scalar) | MAIN | 24 |
| `_starts(X,K,n_starts,seed)` | gmm.py:339-353 | deterministic multi-start init from `(X,seed)` only, **never sees `w`** | MAIN | 15 |
| `_build_features(X)` | gmm.py:356-374 | builds the shared `(N,6)` feature buffer `Q`; `XP=Q[:,:5]` is a view | MAIN | 19 |
| `_sigma_terms(Ss)` | gmm.py:377-388 | `(a,b,c,det)`, raises `LinAlgError` on non-PD | MAIN | 12 |
| `_log_density_coeffs(...)` | gmm.py:391-408 | `(6,K)` coefficient matrix for `Q@C` = log-density | MAIN | 18 |
| `_e_step_fast(...)` | gmm.py:411-424 | one `(N,6)@(6,K)` matmul E-step | MAIN | 14 |
| `_m_step(w,R,XP,reg_covar)` | gmm.py:427-450 | weighted moment M-step | MAIN | 24 |
| — `if reg_covar: S11+=reg_covar; S22+=reg_covar` | gmm.py:445-447 | ridge | SWITCH — option `reg_covar>0.0`; default (and the trace's) is `0.0`. For: keeps a near-collapsing component's fit from being a counted failure | 3 |
| `_run_em(...)` (phase 2, single start) | gmm.py:453-497 | sequential EM to `max_iter`/`tol` | MAIN — the tol-convergence early return (476-480) executed | 45 |
| — `LinAlgError`/non-finite-`ll` → `return None` (×2) | gmm.py:469-470, 473-474, 490-492, 495-496 | start degenerates mid-EM | DATA, none executed; trigger: a component's covariance goes non-PD or `ll` non-finite during phase-2 refinement | 8 |
| — budget-exhausted finalize | gmm.py:484-497 | `converged=False` return at `max_iter` | DATA, not executed (the promoted start converged via the tol check first) | 14 |
| `_sigma_terms_batched(Ss)` | gmm.py:500-513 | same monomials as `_sigma_terms`, never raises (NaN/Inf confined to a start's own columns) | MAIN | 14 |
| `_e_step_batched(...)`, `_m_step_batched(...)` | gmm.py:516-569 | batched `(N,6)@(6,K·S)` E-step and `(K·S,N)@(N,5)` M-step across all phase-1 starts at once | MAIN | 54 |
| — `_m_step_batched`'s `if reg_covar:` ridge | gmm.py:565-566 | | SWITCH, same option as above | 2 |
| `_phase1_batch(...)` | gmm.py:572-628 | runs every start `_SHORT_ITERS` iterations in lockstep, no convergence test; degeneracy checked once after the loop | MAIN | 57 |
| — per-start degeneracy `continue` | gmm.py:619-621, 623-625 | drops a start whose final point is non-finite or non-PD | DATA, not executed; trigger: a start collapsing within the short screen (module cites its own "mixture 12" as a case where this matters) | 8 |
| `_fit_em_multistart(...)` | gmm.py:631-657 | phase 1 (batched) → rank by ll → phase 2 (best `_PROMOTE_N`, sequential) | MAIN | 27 |
| — `if not screened: return None` | gmm.py:641-642 | every phase-1 start degenerated | DATA, not executed | 2 |
| — `if phase1_budget>=max_iter: return` short-circuit | gmm.py:643-646 | phase 1 alone is the whole budget | SWITCH — option: caller's `max_iter<=_SHORT_ITERS(25)`; default/traced `max_iter=500` | 4 |
| — `if not finished: return None` | gmm.py:655-656 | every promoted start failed phase 2 | DATA, not executed | 2 |
| `_canonical_sort(pis,mus,Ss)` | gmm.py:660-662 | relabel by ascending `mu_x` (ties on `mu_y`) | MAIN | 3 |
| `_score_info(X,Q,w,K,pis,mus,Ss)` | gmm.py:670-783 | `(psi,A,ll)` via Louis's identity; `for k in range(K)` loop (K small, per-component vectorized block) | MAIN | 114 |
| `_fit(X,w,cfg)` | gmm.py:790-856 | multi-start EM → canonical order → Newton polish → `resid` gate | MAIN | 67 |
| — polish loop's score-tol `break` | gmm.py:817-818 | `norm<_NEWTON_SCORE_TOL` | MAIN — executed | (in 67) |
| — lack-of-progress `break` | gmm.py:819-825 | plateaued residual, `_NEWTON_PROGRESS_FACTOR` | DATA, not executed | 7 |
| — step-rejection breaks (singular solve; infeasible `pis_new`; `LinAlgError` in `_score_info`; ll decrease) | gmm.py:829-830, 833-834, 837-838, 839-840 | Newton step not accepted | DATA, none executed (every attempted step in the traced fit was accepted) | 8 |
| — `score_history.append(...)` | gmm.py:844 | diagnostic list built inside the Newton loop | MAIN | (in 67) |
| — `resid > eta → return None` | gmm.py:852-853 | fit doesn't solve the score equation to claimed precision | DATA, not executed | 2 |
| `GMM2D.__init__` | gmm.py:868-881 | fixes `K,n_starts,seed,tol,max_iter,reg_covar`; builds `outputs/supports/eta/p` once | MAIN | 14 |
| `GMM2D._resolve(kwargs)` | gmm.py:883-893 | per-call kwarg overrides → `_Cfg`, never mutates `self` | MAIN | 11 |
| `GMM2D.__call__(X,w,**kwargs)` | gmm.py:895-912 | centers X, calls `_fit`, shifts means back | MAIN | 18 |
| — `except Exception: NaN` | gmm.py:911-912 | | DATA, not executed | 2 |
| `GMM2D.influence(X,w,**kwargs)` | gmm.py:914-938 | `IF=A^-1 psi`; **re-runs `_fit` in full** (a second complete multi-start EM + Newton polish at the same `(X,w)`) | MAIN | 25 |
| — `if cfg.reg_covar>0.0: NaN` | gmm.py:917-921 | ridged fit's influence is undefined (different objective) | SWITCH, same `reg_covar` option | 5 |
| — `if fit is None: NaN` | gmm.py:929 | | DATA, not executed | 1 |
| — `cond(A)>_COND_MAX → NaN` | gmm.py:931-932 | | DATA, not executed | 2 |
| — non-finite `IF → NaN` | gmm.py:934-935 | | DATA, not executed | 2 |
| — `except Exception: NaN` | gmm.py:937-938 | | DATA, not executed | 2 |
| `GMM2D.fit_and_influence(X,w,**kwargs)` | gmm.py:940-974 | one fit → `(theta, psi)`, exists specifically to avoid the double-fit above | UNREACHED — no caller anywhere in `src/`/`scripts/` (`study.py`/`qij.py` call `T(X,ones)` then `T.influence(X,ones)` separately, never `fit_and_influence`; the tracer's own gmm task likewise calls fit then influence separately per its instructions). `def` line (940) executes at class-body definition; body (941-974) never runs | 35 |

---

## Table 2 — standards findings (MAIN and DATA code only)

| file:lines | rule | what |
|---|---|---|
| estimators.py:169, 262, 321, 356/387, (every closed-form/model estimator) | E5 | Every estimator recomputes its own per-draw, `w`-independent invariant on **every single call**: `pareto_shape`'s `log_ratio=np.log(x/x_min)` (169), `mvt_nu`'s `r_sq=sum(X**2,axis=1)` (262, and again independently at 302 in `_mvt_nu_influence`), `mvt_tail`'s `r=sqrt(sum(X**2))` (321, again at 333), `fp`'s `X_std=_fp_standardize(X)` (387, again at 425 in `_fp_influence`). None of these depend on `w`, yet nothing passes them down across a draw's many bootstrap replicates / QIJ perturbations (which reuse the same `X`); each is recomputed from scratch on every one of the thousands of calls per draw. |
| datasets.py:76-79 (`_fp_pool`), 90-93 (`_imf_pool`), 122-143 (`truth`) | E5 | Three `functools.lru_cache`-decorated module-level caches (`maxsize=1`, `maxsize=1`, `maxsize=None`). Exactly the anti-pattern E5 names ("module-level caches, which break process parallelism and reproducibility"): correctness is unaffected here (each worker process gets its own cache of a deterministic, fixed-file result), but the caching is genuinely module-level state, not passed down by a caller. |
| estimators.py:345-352 | E5 (soft) | `FP_POP_MEAN`/`FP_POP_STD` computed once at import from `fp_sdss.npz`, then the raw pool array is `del`eted — a module-level constant rather than a caller-supplied invariant, though (unlike the `lru_cache`s above) it is a fixed constant identical in every process, so it does not itself threaten bit-identity. |
| gmm.py:356 (`_build_features`, called from `_fit` at 795) | E5 | `Q`/`XP` depend only on `X`, never `w`, yet `_fit` rebuilds them from scratch on every call. Because `GMM2D` holds no cross-call state by design (module docstring), `Q`/`XP` are rebuilt for every bootstrap replicate and every QIJ perturbation of the same draw's `X`, never computed once by the caller and passed in. |
| estimators.py:143+178-189 (pareto_shape/influence), 193+204-214 (pareto_tail), 265-284+297-311 (mvt_nu, expensive: a full bracket+brentq re-solve), 317-326+330-335 (mvt_tail), 385-392+424-428 (fp, a second `eigh`), 1332-1410+1427 (Chabrier, a second full `trust-constr` fit) | E6 | Every one of these estimators' `.influence(X,w)` re-derives `theta` by calling the base estimator (or redoing its fit) again at the same `(X,w)` instead of sharing the one evaluation `study._run_draw` already made (`theta_hat = T(X, ones)`) or that `QIJ.fit` makes internally. For the closed forms this is cheap; for `mvt_nu` it re-runs the whole bracket-extension/brentq search, and for `Chabrier` it re-runs the full bounded `trust-constr` optimization (the imf/chabrier trace tasks take 51-61s each, largely attributable to two full optimizations per draw instead of one). |
| gmm.py:895-912 (`__call__`) / 914-938 (`influence`) | E6 | Same pattern: `influence` calls `_fit` a second time in full (multi-start EM + Newton polish) rather than sharing the one fit `__call__` (or `fit_and_influence`) already computed. `fit_and_influence` exists in this file precisely to avoid this (module docstring says so explicitly), but nothing in the package calls it (see Table 1, UNREACHED). |
| gmm.py:352, 626, 654, 684, 688, 844 | E7 | `list.append()` inside a Python loop: `_starts` (352, over `n_starts`≈20), `_phase1_batch` (626, over `S`≈`n_starts`), `_fit_em_multistart` (654, over `_PROMOTE_N`=3), `_score_info` (684, 688, over `K` components), `_fit`'s Newton loop (844, `score_history`, ≤`_MAX_NEWTON`=20). None loop over N; all are small, fixed-count loops (starts/components/Newton steps), so this is the named pattern applied to small non-data counts rather than a per-N accumulation. |
| estimators.py: whole file | C1 | Module docstring (1-40) narrates lineage/history at length: what was ported from `vqboot`, what was dropped ("`Estimand` protocol", `check_scale_invariance`, warm-start caches, "the old `IMFDGPEstimand.psi`... both become the NaN-on-failure convention here"). `estimator()` decorator docstring (60) cites "plan Sec 3 / interface sheet Amendment 2". Box-rule comment (95-107) cites "revision plan Sec 36.2(3)" and "commit afbf169" by name. `mvt_nu` docstring (254-259) cites "revision plan Sec 36.2(3)" again. FP section comment (349) cites "plan Sec 4". `IMF` class docstring (623) cites "plan Sec 4". `IMF._at_box` docstring (652-678) is the heaviest instance: a full incident narrative — "commit afbf169", "seventeen IMF draws ran p to its upper bound... against a tolerance of 2.1e-5. The guard caught none of them" — describing a past bug and its fix. `IMF._moment_start` docstring (682-703) records "Coding-manager correction to Sec 4 (interface sheet Amendment 5)" and contrasts the old (wrong) formula with the corrected one. `IMF.influence` (811) and `Chabrier.influence` (1422) both cite "plan Sec 4 ruling". `SUPPORT_BY_OUTPUT`'s comment (1444) cites "Sec 1 of the batch of four, 19 September" — a date. Roughly 13 distinct docstring/comment locations carry this kind of material (≈19 raw lines matching the enforcement grep's history-word list); one further reference, `IMF.__call__`'s "(spec QIJ_dgp_imf.md Sec 0)" (746), is a spec-section citation of the kind C3/R6 explicitly permit, not a plan/revision reference — noted as a borderline case, not counted above. |
| datasets.py:1, 110 | C1 | Module docstring and `mvt_vq_transform`'s docstring each cite "plan Sec 5" by name. |
| gmm.py:36 | C1 | "module used to lay `Q` out `[x^2, xy, y^2, x, y, 1]`... this is the same six numbers, just one buffer" — explicitly describes a prior layout, in `_build_features`'s design section. |
| gmm.py:65-69 | C1 | "An earlier version of this module batched BOTH phases behind a Python-level active set that dropped starts as they converged or degenerated; its per-iteration bookkeeping cost more than it saved... so it was removed" — describes a removed prior design, in the two-phase multi-start section (`_phase1_batch`/`_fit_em_multistart`). |

---

## Table 3 — IMF removal set (two-regime IMF, ruled mis-specified)

Every function/constant used **only** by the two-regime `IMF` path, versus what it shares with
`Chabrier` (the other user on the main path) or with `mvt_nu` (also main path). "Chabrier
reuses gamma/Schechter/incomplete-gamma helpers?" — checked directly: **no**. `Chabrier` has its
own, separate lognormal/power-law machinery (`_chabrier_norm`, `_chabrier_score_hessian`); it
shares only the generic box-rule helper and two numeric constants with the IMF section, both
listed below as shared.

### Removable (IMF-only)

| item | file:lines | shared with | lines |
|---|---|---|---|
| `_weighted_quantile` | estimators.py:85-91 | nobody (only `IMF._moment_start` calls it) | 7 |
| `_EPS_MASS`, `_DENSITY_FLOOR` | estimators.py:440-441 | nobody — `_NLL_PENALTY`(442) and `_SQRT2PI`(443) on the same lines are **shared with Chabrier** (see below) and are NOT part of this removal set | 2 |
| `_gamma_density_scalar` | estimators.py:446-454 | nobody | 9 |
| `_gamma_deriv_scalar` | estimators.py:457-461 | nobody | 5 |
| `gamma_density` | estimators.py:464-469 | nobody | 6 |
| `_upper_incomplete_gamma` | estimators.py:472-486 | nobody | 15 |
| `_schechter_raw` | estimators.py:489-491 | nobody | 3 |
| `_schechter_norm` | estimators.py:494-506 | nobody | 13 |
| `schechter_density` | estimators.py:509-513 | nobody | 5 |
| `schechter_deriv` | estimators.py:516-519 | nobody | 4 |
| `_solve_gamma_constraints` | estimators.py:522-588 | nobody — the audited continuity solve ported from `vqboot.imf_fitter._solve_gamma_constraints` | 67 |
| `_imf_continuity` | estimators.py:591-602 | nobody | 12 |
| `_imf_joint_density` | estimators.py:605-619 | nobody | 15 |
| `IMF` class (docstring, `__init__`, `_at_box`, `_moment_start`, `__call__`, `influence`) | estimators.py:622-830 | `_at_box`'s body calls the shared `_at_bound` (module-level, kept) but is itself IMF-only | 209 |
| `_QUAD_Z/_QUAD_W/_QUAD_PANELS/_QUAD_MARGIN`, `_gamma_order_moments` | estimators.py:833-909 (consts 836-838) | nobody — quadrature for the incomplete gamma's order-derivatives, needed only by `_imf_score_hessian` | 77 |
| `_imf_score_hessian` (incl. nested `hi_grad`/`hi_hess`) | estimators.py:912-1049 | nobody | 138 |
| `IMF_TAU`, `IMF_BOUNDS` (+ their comment) | datasets.py:35-39 | nobody | 5 |
| `datasets.truth`'s `('imf','imf')` branch | datasets.py:135-138 | nobody (the `if` at 135 is a shared line — it's part of `truth`'s if-chain, always evaluated — but its body is IMF-only) | 4 |

**IMF-only line total (this scope): 596** (587 in estimators.py + 9 in datasets.py). `gmm.py`
has no IMF-related content.

### Shared — stays regardless of the IMF ruling

| item | file:lines | shared by |
|---|---|---|
| `_BOX_TOL`, `_at_bound` | estimators.py:108-121 | `mvt_nu` (main path, box-hit failure — the actual mechanism behind the 3 known TACC `mvt/nu` failures), `IMF._at_box`, `Chabrier._at_box` |
| `_NLL_PENALTY` | estimators.py:442 | `IMF.__call__`'s `nll` (752,755,762) **and** `Chabrier.__call__`'s `nll` (1364,1369,1371,1382) |
| `_SQRT2PI` | estimators.py:443 | `IMF` does not use it; it is used by `_chabrier_norm` (1113-1125, 8 uses) — kept for Chabrier regardless |
| the general two-tier "optimizer box + `_at_box` box-rule" convention | estimators.py:645/652-678 (IMF) vs 1288/1297-1304 (Chabrier) | design pattern shared between `IMF` and `Chabrier` (each has its own bounds/`_at_box`, not shared code, but the same rule) — not a removable line, noted for context only |

---

## Summary

### Lines by status, per file

| file | total | MAIN | DATA | SWITCH | UNREACHED |
|---|---|---|---|---|---|
| estimators.py | 1473 | 406 (executed) | ≈60 (failure/guard branches not hit by the trace's 2+2 draws or the gmm/check tasks — see Table 1 rows) | 596 (IMF removal set, Table 3) + `_weighted_quantile` already counted there | ≈40 (`weighted_mean`+`_weighted_mean_influence` bodies 21, `supports_for` 12, plus scattered UNREACHED lines) |
| datasets.py | 143 | 70 (executed) | 0 | 9 (IMF removal set) | 0 |
| gmm.py | 973 | 418 (executed) | ≈50 (unexecuted failure/degeneracy branches in `_run_em`, `_phase1_batch`, `_fit_em_multistart`, `_fit`, `GMM2D.influence`) | ≈10 (`reg_covar>0` branches, `phase1_budget>=max_iter` short-circuit) | 35 (`fit_and_influence`) |

(The MAIN figures are exact, taken directly from `executed_lines.json`; the DATA/SWITCH/UNREACHED
splits are derived from Table 1's row spans and are approximate where a row's span mixes an
executed condition line with an unexecuted body — e.g. `if X: <not executed>` counts its `if`
line under MAIN and only the body under DATA/SWITCH.)

### Every option, main-path value, and other supported values

| option | where | main-path value | other values seen in this scope |
|---|---|---|---|
| `estimator(...)`'s `supports` | estimators.py:55 | `None` (weighted_mean) or explicit list (all others) | both are exercised at import; no third form |
| box-rule tolerance `_at_bound(..., tol=_BOX_TOL)` | estimators.py:108-121 | `tol=_BOX_TOL=1e-4` (default, never overridden anywhere in this scope) | none — no caller passes a different `tol` |
| `IMF(tau, bounds)` | estimators.py:647-650 / datasets.py:38-39 | constructed once at `study.py` import with `tau=0.176609`, `bounds=((-10,-0.01),(1,200),(0.1,20))`, but never *called* on the traced/main path | reachable via a different study config (e.g. `cost_vs_n`/archive configs, per `figures.py`'s references to `dataset="imf", estimator="imf"`) selecting the `imf` estimator instead of `chabrier` for the `imf` dataset |
| `Chabrier(m_min, m_b, bounds)` | estimators.py:1291-1295 / datasets.py:49-51 | `m_min=0.0017582750879228115` (hardcoded), `m_b=1.0`, `bounds=((0.01,5),(0.02,5),(0.05,10))` | none seen — always constructed with these same module constants |
| `GMM2D(K, n_starts, seed, tol, max_iter, reg_covar)` | gmm.py:868-881 | traced call: `K=3` (ChaconMix 9), other args left at defaults `n_starts=20, seed=0, tol=1e-8, max_iter=500, reg_covar=0.0` | `reg_covar>0.0` (SWITCH — skips Newton polish and forces `influence`→NaN); per-call kwarg overrides to any of the six are supported (`_resolve`) but never exercised in the trace |
| `datasets.truth(dataset, estimator)` | datasets.py:122-143 | called (at least) for `('fp','fp')` and `('imf','chabrier')`, and for the parametric pareto/mvt keys | `('imf','imf')` supported by the code (line 135) but its condition never fires on the traced/main-study path |

### Comment+docstring lines vs code lines per file (approximate, via `tokenize`)

| file | total | comment/docstring lines | blank lines | code lines |
|---|---|---|---|---|
| estimators.py | 1473 | 516 | 194 | 763 |
| datasets.py | 143 | 56 | 26 | 61 |
| gmm.py | 973 | 414 | 94 | 465 |

`gmm.py` and `estimators.py` are both roughly 45-50% comment/docstring by line count once blanks
are excluded (516/763 ≈ 68% of *code+comment* lines are comment/docstring in estimators.py;
414/465 ≈ 89% in gmm.py, driven by its 243-line module docstring).

### Data files (`src/qij/data/`)

| file | size | loaded by |
|---|---|---|
| `fp_sdss.npz` | 5,545,524 bytes (~5.3 MB) | loaded **twice, independently**: (1) `estimators.py`'s module-level code (345-352) at import, to compute `FP_POP_MEAN`/`FP_POP_STD`, then the raw array is deleted; (2) `datasets._fp_pool()` (76-79, `lru_cache(maxsize=1)`), which keeps the full (76997,3) array cached for `datasets.fp()`'s resampling and for `datasets.truth(('fp','fp'))` |
| `stars.h5` | 114,681 bytes (~112 KB) | `datasets._imf_pool()` (90-93, `lru_cache(maxsize=1)`) — reads only `data/BH_Mass` (19857,); feeds `datasets.imf()`'s resampling and `datasets.truth(('imf','imf'))`/`datasets.truth(('imf','chabrier'))`. (The file also carries `data/sim_codes`, `data/sim_levels`, `data/snapshot`, `grid/M_grid`, none of which the package reads.) |
