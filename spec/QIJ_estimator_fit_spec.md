# The mixture estimator's fit, brought to standard practice (planner, 29 September 2026; held for the author's word)

Scope: `gmm.py`'s `_fit` and what it calls, for `GMM2D` and through it `P2Mixture`.
The ESTIMAND does not change: the Chen–Tan penalized likelihood with penalty
strength 1/Σω, weights summing to N, the analytic influence by Louis's
identity plus the penalty's weight sensitivity, a continuation labelled
against its start by minimum-Bhattacharyya assignment. Only the optimizer
around it changes. The mixture estimator appears in none of the paper's
audits, so no bit-identity constraint applies; every stored cloudfil product
changes and none had passed a gate.

## 1. What the investigation established (scratchpad `gmm_diag/`, four reports)

On draw 1's vor2 survey points, continued from θ̂ at η = 10⁻⁷:

- The failure is the control flow, not the objective, the tolerance, the
  weights, the penalty, or conditioning. The SQUAREM block stops on the
  likelihood's relative change (tol 10⁻⁸) at iteration 32 with the score
  norm still 3 × 10⁻³, on a plateau beside a saddle; the one-time
  positive-definiteness gate passes (smallest eigenvalue 3 × 10⁻³); the
  undamped full Newton step raises the likelihood by 7 × 10⁻⁶ and is
  accepted, landing where the Hessian is indefinite (eigenvalue −70); the
  polish never re-tests definiteness and its acceptance slack of 10⁻⁹
  admits 39 steps of length 3 × 10⁻⁷ that each lower the likelihood by
  6 × 10⁻¹⁰; exit at residual 1.05 after 586 Hessian evaluations and 9 s.
- Weights are handled correctly everywhere (duplicated points against
  integer weights agree to 3 × 10⁻¹² in the fit and 10⁻⁹ in the influence).
- η at 10⁻⁶, 10⁻⁵, 10⁻⁴ fails identically; once EM has converged the fit
  reproduces to 10⁻¹². Penalty off and at ten times strength both converge.
- The maximum exists and is reachable: plain EM in 290 iterations (0.4 s),
  Levenberg–Marquardt from the stalled iterate in 29 steps (0.56 s), the
  house EM at tol 10⁻¹⁰ in 75 steps; one Newton step then reaches 5 × 10⁻⁹.
- Near θ̂ on that draw there are two maxima and a saddle. A quasi-Newton
  run from θ̂ reaches the higher one; EM and a damped Newton from EM's
  endpoint reach the other. A continuation must follow one branch.
- `_measure_eta_Q` returns NaN, not a failure, when its continuations fail.

## 2. The fit, as it should be

**2.1 EM, with a convergence test that means convergence.** Weighted
penalized EM as built, SQUAREM allowed in its safeguarded form: a SQUAREM
step that lowers the penalized likelihood is discarded and the plain EM
step taken instead (monotone by construction). Stop by Aitken's
acceleration criterion (McLachlan and Krishnan, §4.9): with ℓ_k the
penalized log-likelihood per unit weight at iteration k and
c_k = (ℓ_k − ℓ_{k−1})/(ℓ_{k−1} − ℓ_{k−2}), the projected limit is
ℓ_∞ = ℓ_{k−1} + (ℓ_k − ℓ_{k−1})/(1 − c_k); stop when |ℓ_∞ − ℓ_k| < tol_A
AND the score norm (already formed for the gate) is below tol_g. tol_A =
10⁻¹⁰, tol_g = 10⁻³ (P): the case that failed had score 3 × 10⁻³ at its
premature stop and 5 × 10⁻⁴ at the point from which the finish converged.
Iteration cap 2 000; reaching it is a status, not a NaN.

**2.2 A trust-region Newton finish, no gate.** From EM's endpoint, in the
unconstrained parametrization (softmax for the weights, Cholesky factors
with log-diagonal for the covariances, means as they are), maximize the
penalized likelihood with `scipy.optimize.minimize(method='trust-exact')`
on its negative, supplying the analytic gradient and the exact Hessian
the code already forms (`_score_info_penalized`'s ψ̄ and A, mapped through
the parametrization's Jacobian; the chain rule for the Hessian is
standard and small). Initial trust radius 10⁻² in the whitened parameter
scale, so the finish stays on EM's branch; `gtol` = the scaled gradient
criterion of 2.3; `maxiter` 100. A trust-region step is accepted only on
sufficient increase (the method's own ratio test); indefiniteness is
handled inside the subproblem, so no gate, no halving loop, no
acceptance slack. If the finish ends by iteration cap or a trust radius
below 10⁻¹², the fit returns EM's endpoint with a status (2.4).

**2.3 Convergence declared on a scaled gradient.** Converged when
‖ψ̄‖_scaled ≤ η, with ‖·‖_scaled the norm of the gradient in the
unconstrained parametrization divided by max(1, |ℓ|) (Dennis and Schnabel's
relative gradient), η the value the caller passes (η_full or η_Q, both
measured). The declared η stays a measured reproducibility, not a solver
setting: `_measure_eta_Q` and `measure_eta_full` return a FAILURE flag
when any of their three continuations fails, never NaN through `nanmax`.

**2.4 Status, not NaN, at the estimator; NaN at the method's boundary.**
`_fit` returns (θ, status) with status in {converged, em_cap,
newton_cap, infeasible, linalg}; `GMM2D` exposes it as `last_fit_info`
together with the final score norm, EM iterations, Newton iterations and
the labelling assignment. `GMM2D.__call__` and `P2Mixture` return NaN for
any status other than converged, exactly as now, so the method's rule (a
failed evaluation is excluded and counted, never retried) is unchanged,
and the pipeline records the status in the failure log.

**2.5 What is removed.** `_cholesky_ok` as a gate, the second EM block on
gate failure, the damped-Newton polish loop with `_MAX_NEWTON`,
`_MAX_HALVINGS` and `_ACCEPT_TOL`, and the relative-ℓ stopping test in
`_run_em`/`_em_accelerated`. `_cholesky_ok` survives only where the
influence needs a positive-definite A (`_COND_MAX` check unchanged).

**2.6 Untouched.** The cold search (`search = multistart`, its screen and
starts); the E-step, M-step, penalty, `_weighted_cov`, Louis's identity,
`_penalty_influence_extra`, the labelling; `takes_start` semantics. A cold
fit uses 2.1 and 2.2 for each start's run and the same status.

## 3. Validation (no tests in the package; a measurement report)

Direct calls to the estimator only. NO method runs: no `qij`, `boot`,
`oracle`, `ij` or `ijfd` invocation through the pipeline, at any size.
One script, one report, on branch `clean`:

1. The five diagnostic cases of `gmm_diag` (draw 1 vor2 centroids, draw 1
   vor1 centroids, draw 0 vor2 centroids, draw 1 moments survey points,
   draw 1 full data), continued from θ̂: status, EM iterations, Newton
   iterations, final scaled score, wall time; the failing case must
   converge, the others must reach the same maximum as before to 10⁻⁷
   relative.
2. The survey's 1 094 continuations on draws 0 and 1, called directly
   (`xvq.prototype_influences` on the stored quantizer, or the equivalent
   loop of `T(rows, ω_j, start=θ_Q)` calls; not a `qij` run): failure
   count (was 0 on both after the weight fix), r² of I_j against the true
   receptive-field means from the stored `ij` product (was 0.97–0.99),
   wall time.
3. Fifty resampled-weight fits on draw 1, called directly as
   `T(X, ω_b, start=θ̂)` with multinomial weights (the bootstrap's own
   call, without the bootstrap pipeline): degenerate count (was 9 of 50
   under the old control on the first rehearsal), status breakdown.
4. Twenty cold fits, `T(X_s, 1)` on draws 0–19 called directly: failures
   (were 3 of 20), status breakdown.
5. The influence check the estimator has always had: finite-difference
   agreement of `T.influence` on draw 0's full data to the accuracy the
   step allows.
6. The duplicated-points weight test of `gmm_diag/weights` rerun once, to
   show the refactor kept the weight handling (3 × 10⁻¹² / 10⁻⁹).

Acceptance: item 1's failing case converges and the four others are
unchanged to 10⁻⁷; items 2–4 show no new failure class and fewer
failures than before; items 5 and 6 unchanged. Nothing else is run.

## 5. Constants restated relative (coordinator's questions, 29 September; supersedes the numbers in section 2)

The author's rule: no constant in the code is set from one draw of one
dataset; every threshold is relative to a quantity the fit measures or
the caller passes.

**Coordinate scales.** In the unconstrained parametrization each
coordinate i has a scale s_i: 1 for the weight logits and the covariance
log-diagonals; the data's column standard deviation (already computed by
`_weighted_cov`) for the means and the off-diagonal Cholesky entries.
"Scaled" below means divided by s_i. The scaled gradient norm is Dennis
and Schnabel's relative gradient, max_i |g_i| s_i / max(|ℓ|, 1), with ℓ
the penalized log-likelihood per unit weight.

1. **EM stop.** EM stops when the scaled gradient norm ≤ √η AND the
   Aitken-projected remaining gain |ℓ_∞ − ℓ_k| ≤ η · max(|ℓ_k|, 1). The
   first delivers an iterate from which a Newton finish converges
   quadratically to η in one or two steps; the second guards against
   declaring convergence on a plateau. Both are tied to the caller's η
   (η_full or η_Q, measured). The absolute tol_A = 10⁻¹⁰ and tol_g =
   10⁻³ of section 2.1 are withdrawn.
2. **EM iteration cap.** A safety limit guaranteeing termination; no
   converged result depends on it. 20 × p, p the parameter count; status
   `em_cap` when it binds.
3. **Initial trust radius.** The scaled length of EM's last step, floored
   at √η. Near convergence EM's steps are small, so the finish starts on
   EM's branch by construction; the trust-region algorithm enlarges the
   radius on successful steps. The 10⁻² of section 2.2 is withdrawn.
4. **Trust-region iteration cap.** Safety limit, 2 × p; status
   `newton_cap`.
5. **Give-up radius.** Δ_min = η in scaled coordinates: a step below the
   parameters' own reproducibility cannot improve the fit; the finish
   then returns EM's endpoint with status `newton_stalled`. The 10⁻¹² of
   section 2.2 is withdrawn.
6. **Convergence (section 2.3).** Scaled gradient norm ≤ η, unchanged in
   substance; the scale is the one defined above.

The four evidence reports of section 1 are to be copied from the
planner's session scratch into the Research folder and cited there.

## 4. Cost

EM to Aitken convergence is longer than to the old tolerance (75–290
iterations against 32 on the failing case, 0.3–0.4 s); the trust-region
finish is two to five Hessian evaluations against the old two on a
converging case and 586 on the failing one. Net per evaluation: about
unchanged on converging fits, ten times faster on the ones that used to
fail. Measured in item 1.

## 6. Continuations on a tiny perturbation: the finish's stop and acceptance (planner, 1 October 2026; one build; applies to the continuation path only, `start` given)

**What was found (coder, 1 October; the seeded-tree validation, the
step scan and two instrumented finishes on draw 7, η_full 1e-7).** A
continuation from θ̂ under a perturbation that moves the fit by a
displacement of order 1e-6 in scaled coordinates (a single-point bin at
the step rule's t ≈ 6e-8) can return a systematically PARTIAL response
— 43% of the exact displacement on one bin, 0.43 on log_reff — while
reporting `converged`. The step scan on 20 single-point bins is
bimodal: 4 shrunk 10–19%, the rest within 0.2%; the shrink depends on
the absolute step and not on η (η/100 leaves it). Two defects, both in
the finish's own logic, neither in the trust radius (200× the Newton
step) nor in the Hessian (PD, condition ~2e4):

A. **Callback order.** `_callback` tests "scaled gradient ≤ η →
   converged" BEFORE its "step == 0 → a rejected step, return". A
   perturbed problem's starting gradient g0 is itself of order η (the
   weight change times one point's score over W), so when the first
   Newton step is rejected the point has not moved, the gradient is
   still g0 ≤ η, and the callback raises converged: the fit stays where
   EM's few linearly-convergent rounds left it. That is the partial
   response.

B. **Why the step is rejected.** The exact Newton step, PD Hessian,
   well inside the radius, is rejected only when trust-exact's
   actual/predicted reduction ratio is noise: the predicted gain
   ½δᵀHδ ≈ 5e-13 lies inside the rounding error of ℓ, which is a
   weighted sum over N terms and so carries eps·|ℓ|·√N … eps·|ℓ|·N
   (6.5e-14 … 6.5e-12 here), not eps·|ℓ|. At 10t the gain is 5e-11,
   above the band, and the fit completes (response 1.001). **Measured
   directly (coder, 1 October, the author's go): bin 355 at 1t, in the
   finish's own scaled objective, fun(v0) = 2.9264457106644648,
   predicted reduction of the exact Newton step 3.77e-15, actual
   −3.11e-15, ρ = −0.82 → rejected; the noise of fun along the step (41
   points, a ∈ [0, 2], against the exact quadratic model) has sd
   3.5e-15 and max 1.4e-14, only 24 of 41 values distinct; eps·|fun| =
   6.5e-16. The predicted reduction sits at about one noise sd. An
   earlier "gain = 5e-13, 770× resolution" was mis-scaled (a raw-θ
   quadratic form, not the objective's own reduction); the resolution
   hypothesis was never refuted.**

**The rule, stated once (replaces an earlier three-rule draft of this
section whose thresholds on the objective — a rounding-resolution
r_ℓ, a floored relative test — were patches on the wrong instrument;
the author, 1 October: "do we need to stop and think about how
gradient-driven optimization works?").**

A continuation is a ROOT-FINDING problem started next to the root: the
base fit θ̂ solves the score equations under the base weights, the
perturbed weights move that root by a displacement that is, to second
order, one Newton step (minus the inverse Hessian times the perturbed
gradient at θ̂ — the influence function is exactly this
linearization), and Newton's method from there converges in one or two
steps if nothing stops it. The trust region judges a step by the
DECREASE OF THE OBJECTIVE; near a root that decrease is quadratic in
the displacement while the gradient is linear in it, so for the
perturbations the quantizer stages make, the objective's change falls
below the rounding noise of an N-term sum (measured: predicted
reduction 3.8e-15 against noise sd 3.5e-15 on bin 355) while the
gradient is still resolved to many digits (9.7e-14 reached on bin 381).
Judging the step by the objective is judging it with the one quantity
that cannot see it; every symptom above (the rejected correct step, the
callback reading a gradient still under η, the stall accepting) follows
from that. Therefore:

**On a continuation whose starting gradient is small — `start` given
AND g0 ≤ √η, g0 the scaled gradient norm at the finish's start v0
under the perturbed weights — the finish is Newton on the score,
judged on the gradient and never on the objective. A continuation with
g0 > √η, and every cold fit, keeps trust-exact on ℓ unchanged, byte
for byte.** (The scope is in the rule, settled with the coder 1
October: the failures sit at g0 ≈ 2–4e-8 against √η = 3e-4, and every
survey row, stencil, check split and the η measurement is a
continuation, so a rule on the whole continuation path would change
the last digits of everything already validated.)

* Acceptance: a step is accepted when it reduces the scaled gradient
  norm (the standard merit for nonlinear equations, ½‖g‖² in scaled
  coordinates); a step that does not reduce it is halved and retried
  (at most the trust-region iteration cap of section 5 item 4, which
  stays as the safety limit); the objective ℓ is not consulted for
  acceptance.
* Convergence: the scaled gradient norm ≤ max(√η · g0, g_floor) AND
  ≤ η (the absolute test of section 5 item 6), both required. √η is
  the same factor the EM stop and the stall rule use; η·g0 would demand
  a gradient below what can be computed (bin 381 ends at 9.7e-14 from
  g0 1.85e-8). g_floor is the scaled gradient's own float resolution,
  formed from the fit's terms: eps · √N · max_i |s_i · (score scale)_i|
  / max(|ℓ|, 1), the rounding random walk of the N-term score sums in
  the units of the scaled gradient — needed because without it a
  continuation whose √η·g0 lies below what the gradient can resolve
  could never converge, would halve to the cap, and would return as a
  FAILED evaluation (GMM2D.__call__ returns NaN unless converged);
  today's worst margin is ~60× (5.9e-12 against 9.7e-14 on bin 381) and
  shrinks with N and with smaller η.
* A rejected or halved step is never convergence: the convergence test
  is applied only to a point the finish has moved to.
* Stall: when a step below the give-up radius (section 5 item 5) fails
  to reduce the gradient, the finish returns the best point reached
  with status `newton_stalled`; a stall is never `converged`.
* The initial step is the exact Newton step from v0 with hess(v0)
  under the PERTURBED weights, as the finish's own `_eval` forms it
  (not θ̂'s base-weight information: the difference is O(t), harmless to
  second order, but reusing it would break the halving loop's
  consistency and the reproducibility reasoning); no trust-region
  subproblem; the trust radius plays no role on this path.
* Status: `last_fit_info` gains `g0` and `newton_halvings`; `converged`
  keeps its meaning; nothing else in the status set changes.

No threshold on the objective exists on this path; the only resolution
quantity is g_floor, on the gradient, formed from eps, N and the fit's
own terms. The √N random-walk form on the OBJECTIVE of the earlier
draft is withdrawn with the rule that needed it.

**Validation (author's machine; no sweep).** (i) Byte identity: every
cold fit (oracle, the full-data θ̂ of every method) and every
continuation whose g0 > √η (where the absolute test already implied the
relative one) on draw 1 and draw 7 at ε 0.01 — the whole qij product
set of today's pipeline identical to current main, new status fields
excepted. (ii) The step-scan bins of draws 7 and 1 (scratchpad
step_scan_d{1,7}.parquet) re-measured at 1t, judged by the scan's own
aggregate: the mass-weighted slope of measured U on the exact bin mean,
per group (single-point, smallest multi-point) and output, within 3e-4
of 1 at 1t (as the 3t and 10t rows are today), AND no bin with a
response below 0.99 (today 4 of 20 single-point bins sit at 0.81–0.91).
Per-bin agreement to 3e-4 is not required: unaffected bins already
scatter 0.9986–1.0038 per bin from forward-step truncation and from the
exact influence being the linearization. The 10t–100t responses
unchanged. (iii) The seeded branch's draw-7 ε 0.01
run repeated with the amended fit: the measured-between deficit
(1–3.2%) gone, shortfall ≤ max(ε, today's) — the number that motivated
this section. (iv) Fit cost: `wall_time_prototype` and the check-stage
wall unchanged within noise (the gradient-judged Newton path removes
the rejected trust-region iterations and adds no evaluation where the
absolute test was already sufficient).

Constants: none new. eps is machine epsilon; √η and η are the caller's
measured tolerance; g_floor is formed from eps, N and the fit's own
score terms; there is no threshold on the objective. The step rule of
the quantizer stages is unchanged by this section.
