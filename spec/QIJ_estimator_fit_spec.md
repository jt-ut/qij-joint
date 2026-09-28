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
