# The investigation plan: qijt with the affine field — one method, all measurements quantized, no patched machinery (planner, 6 October 2026; supersedes QIJ_within_variance_decision.md's recommendation)

## 0 (rewritten at the end of 6 October). The method, one rule

One X-VQ, grown as a tree. Every row is a prototype carrying its
field's count, so the weights sum to N at every level, and the
estimator is evaluated on the current prototypes only: the quantized
fit is T(W, ω) with ω_j = n_j. A prototype's influence is the response
of that fit to its field's mass, a forward difference at the step
rule. A field is split, adding a prototype, where its lack of fit
exceeds the tolerance's share, and the process stops where it no
longer does: in the bulk at the first survey, in the core at one or
two points per field, where the prototype is the data point. The
variance is the between-prototype term plus the affine within read
from measured gradients, and the stop is the tolerance. The survey and
the influence VQ are the same operation; there is no second stage, no
moment row, no raw-point category, no full-data evaluation.

What was shown on 6 October with real quantized evaluations (scratchpad
instrument_gate/, onerow_refined/): the quantized fit converges to the
full-data fit as fields split — a prototype's measured influence
matches the exact field mean to 1–2% once the influential fields reach
single points, the between-prototype variance matches the exact to
0.1%, the measurement is linear in the step (half/double ratios 1.000,
no sign flips) down to single points, the base fit is a fixed point to
1e-13, and a quantized evaluation costs 1/200 of a full fit in wall
time. Two package defects were found and are to be removed in any
build: the points survey hands the estimator weights summing to M
rather than N (core.xvq.prototype_influences, survey='points'), and the
moment rows exist to compensate for it. The representation's row count
does not grow with the data dimension.

What is not yet shown: the loop itself with the real instrument —
spec QIJ_one_rule_run_spec.md. Stages 1–3 below ran with a stand-in
instrument (Stage 3 void for that reason; Stages 1–2 stand as the
affine read's tests); Stage 4 waits on the one-rule run.

## 0 (original). What the method is, in one paragraph

All measurements are forward differences on the quantized
representation (the moments rows of the receptive fields), never on the
full data. The influence model between measurements is the affine field
over the receptive fields: each field's mean is its measured prototype
influence, its gradient is fitted from its connectivity neighbours, and
its lack of fit (the residuals of that regression) is both the within
variance the model cannot see and the indicator that the field is
under-resolved. The partition is the X-VQ itself, grown as a tree: a
field whose lack-of-fit energy exceeds the tolerance's share is split
along its gradient direction and the halves are measured, which is one
quantized evaluation each and is what a bin measurement in today's loop
becomes. The variance is the between-field term over the current fields
plus the affine within inside them, with the lack-of-fit energy as the
within's error bar; the stop is the tolerance. Nothing is scaled, shifted,
floored or counted: no κ̂, no scale-or-shift update, no χ² term, no round
floor, no n_min, no ρ, no root centring, no full-data central pairs.
This is the 29 September qijt decision with a model between its
measurements and a refinement rule in the tolerance's own units.

## 1. The evidence it rests on (all 6 October, draw 45 unless stated; scratchpad paths in qij-planner-state)

- The quantized survey influence at a field equals the full-data mean
  influence over the field up to a scale that goes to 1 with
  resolution: slope 0.72–0.96, R² 0.93–0.994 at 1094 prototypes;
  slope 0.996–1.007, R² 0.993–1.000 at 2188. The between-field variance
  from the survey reproduces the exact one to 0.993–1.000 at 2188.
- The affine field's within read on the final leaves, with no factor,
  is inside the tolerance on six outputs at 1094 prototypes; position
  angle is short (0.11) because the core is under-resolved (shortest-
  lag semivariance 1.4–2.8 × Var; the 2× survey fixed pa on 3 Oct).
- The gradient-pushed receptive-field variance reads the fields holding
  99% of the within at 1.0–2.4 on draw 45 and 0.97–1.17 on the
  Fundamental Plane, with nothing fitted.
- The affine model's point uncertainty from noise alone is 10–300× too
  small on the bins: the lack of fit is the missing term, untested.
- Every Gaussian-process variant failed on one case or another for a
  kernel hyperparameter; the affine field has none.

## 2. The rules this plan runs under

One draw per check (draw 45; draw 0 of the paper cases). Products
first: no estimator evaluation until Stage 4. Every spec carries a
worked example with real magnitudes and its explosion cases; every
agent prompt carries a sanity cap. Each stage ends with its deciding
number beside the baseline and the author's go before the next. No
respecification without a data look that names the cause. Terminology:
the glossary and the author's own words.

Baseline on draw 45, ε = 0.01: the loop as built, 1094 survey
evaluations (quantized) + 702 full-data central pairs, errors −0.25 to
−1.0ε on six outputs and −3.3ε on pa; the floored loop, 2038 full-data
central pairs, every output inside ±0.06ε. Cost unit: a full-data
evaluation = 1; a quantized evaluation on M rows = M/N.

## 3. The stages

**Stage 1 — the within read and its error bar on the refined survey
(products exist: the 2188-prototype diagnostic run of draw 45).**
Question: on the refined survey, does the affine field read the final
leaves' within inside the tolerance on every output including pa, and
does the lack-of-fit term, pushed into the bins and the leaves, cover
the realized error (ratio near 1 on the 351 bins; the leaves' exact
within inside the error bar)? Also: the lack-of-fit map — how many
fields exceed the tolerance's share at 1094 and at 2188 prototypes, and
where (this is the local refinement's cost estimate). Deciding numbers:
implied error/ε per output at 2188; the bin ratio with the lack-of-fit
term at 1094 and 2188; the count of flagged fields. Pass: inside ±1ε on
every output at 2188 and a bin ratio within a factor of three.

**Stage 2 — generality of the affine field and the lack of fit.** The
Stage 1 computation on draw 0 of the Fundamental Plane and the IMF
(products exist), and the multivariate t once the coder reproduces its
quantizer coordinates (the regenerated whitened draw matched the stored
labels on 85% of points; a products task). Deciding numbers as Stage 1.
Gate: a failure on a smooth case means the model is wrong and the plan
stops here.

**Stage 3 — the method emulated, no full-data fits.** Extend the
draw-45 emulator: start from the 1094-prototype X-VQ and its survey;
while any field's lack-of-fit energy exceeds its share of ε times the
between-field term, split it along its gradient direction and "measure"
the halves by the full-data field mean of the exact influence plus the
quantized-measurement scatter the 2188-prototype products showed (1–8%
of the field-mean spread, per output) — the oracle stand-in for a
quantized evaluation, stated; update the affine field (the halves are
new fields with new neighbours); stop when no field exceeds. Read
V = Σ p_j I_j²/N + the affine within, with the lack-of-fit bar. Deciding
numbers: error/ε per output against the exact V_ij; evaluations in
size-N units (Σ M_t/N over the evaluations made) beside 1094·M/N + 702
for the loop as built and the floor's 2038; the number of fields at the
stop. Pass: inside ±1ε on every output at a cost below the loop as
built. This is the emulation of the method itself and the test of the
single-channel claim: every measurement is a field, every field feeds
the model.

**Stage 4 — the build.** Only after Stages 1–3 pass: a build order to
the coder on the qijt branch — the affine field, the lack-of-fit
refinement, quantized-only measurement, the variance and the stop as
section 0 — with the code standards, one-draw validation against the
Stage 3 emulation (the one pass/fail: the emulated and built numbers
agree on draw 45), then the thousand-draw study on TACC against the
loop as built and the floor, in the frame of 4 October: each method
against its own target.

## 4. What is dropped, and the one thing kept from the old loop

Dropped: κ̂ and its sandwich error; the scale-or-shift update; the χ²
term; the round floor; n_min; ρ and the root centring; the full-data
central pairs; the two-means cut in influence space (the split is in
data space along the field's gradient); the Gaussian-process influence
model and its width search.

Kept: the X-VQ and the moments survey; forward differences and the
declared accuracy's step rule; conservation where a parent's measured
influence and one half's give the other half; the tolerance ε and the
stop's form (estimate plus error bar against ε times the total).

## 5. Out of scope, named

Any further repair of the loop as built; any Gaussian-process variant;
the score basis; the data-space extent and other per-leaf regressions;
the lineage-decay models; the round floor as a method. The paper's
frozen `qij` repo is untouched. The mvt quantizer-coordinate
reproduction is the coder's, not a model question.

## 6. Risks, stated

- Stage 3's oracle stand-in for a quantized evaluation is exact plus
  scatter; the real quantized evaluation at a small field has the
  forward-difference noise of a small mass (t_j ∝ p_j), which the step
  rule sets; the build's validation is where that is measured.
- The lack-of-fit term has not been scored anywhere yet; Stage 1 is its
  first test and the plan's first gate.
- In high dimension (mvt, 10-D) the connectivity neighbours may not
  determine a gradient well (188 prototypes); Stage 2's third case is
  the test, and the plan does not proceed to Stage 4 without it.
- Cost: the lack-of-fit refinement is local; its evaluation count on
  draw 45 is Stage 1's flagged-field count, not an assumption.
