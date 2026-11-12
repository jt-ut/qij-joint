# The seeded measured tree: one state vector, a Zador-sized minimax k-means seed, growth and check as one measured loop (planner, 1 October 2026; one build on a branch, not merged until the author rules on the validation)

Scope. A new joint second stage behind `joint_mode`: `'staged'` =
today's stage exactly (growth on the pilot → central stencils on every
initial bin → the measured check, `check_rule='measured'`), default,
byte-identical products; `'seeded'` = this document. Built on branch
`qij-seeded` in its own worktree; the default stays `'staged'` on main
and nothing is merged until section 7's report is in and the author
rules. Requires `ivqbins='joint'`, `pilot='gp'`; `check_rule` is inert
under `'seeded'` (the loop below is the check). Stage 1 (survey, pilot
fit with `gpwidth='local'`, `fit_weights='mass'`), the sigma stage,
the marginal path and every product downstream of the per-point
estimate are untouched. Code standards: `qij_joint_plan.md` section 5,
verbatim in the build prompt. No check is added. Terms as the
glossary: prototype cells, the pilot, points, bins; never "rows"/
"field".

## 0b. REVISION 3 NOT ADOPTED (1 October 2026, evening): the calibrated stop fails the selection test

The author ruled ε a BOUND: the within share allowed to be left on the
draw, not an expected target. A stop respects a bound only with a
conservative estimate of the remaining within. The calibrated stop of
revision 3 (item 2 below) is conservative only if the realized/predicted
ratio measured on the split leaves is at least the ratio on the unsplit
ones. Tested offline by the coder (scratchpad kappa_sel2.py; draw 1,
today's 512 leaves at 0bd71ad, re-anchored base updated once per leaf
with the stored measured means, exact ψ from the ij product, a leaf's
realized within standing in for a split's realized gain): the finished
partition's predicted within is already 0.0015–0.0035 of the total
(under ε everywhere), so the test was run with the top-ranked leaves
covering 25 / 50 / 75% of the predicted within. κ_top ≥ κ_rest FAILS on
5–6 of 7 outputs at every size; implied realized shortfall if the loop
stopped on κ̂: 1.29–1.47 ε (log_reff, logit_w, pa). Ranking on the
prediction selects leaves whose prediction is high relative to the
truth (the coder's objection), and that outweighs the tail
under-prediction (the planner's argument). The stop is unsafe in the
bound's direction by construction, not by a one-draw accident.

Consequence: today's pipeline's stop structure stays — its ~½ε landing
(the ε/L rule) is the margin that respects the bound, now with the
measured reason. The one saving revision 3 contained that carries no
risk to the bound is one forward evaluation per leaf in place of the
central pair (~390 of draw 1's 1266 size-N evaluations); its cost is
the forward bias (1.2–2% of V_btw at the Mac's η_full 1e-7… measured
on the Mac; expected ~⅓ at TACC's η), to be decided on one draw at
TACC's η. Items 1 (bisecting k-means start) and 3 (re-anchored vector)
are not adopted either: their value rested on the stop. Section 0a is
kept below as the record of the design and of why it fails.

## 0a. Revision 3 (1 October 2026, the author's rulings of the day; SUPERSEDES section 3 entirely and the parts of sections 4–5 named below; everything not named stays as built on qij-seeded)

**What the day established, offline, on draws 1 and 7 at ε 0.01 and
0.141 (scratchpad ivq_kmeans_check/, python3.9, no estimator calls).**
(a) The minimax k-means seed of section 3.2 is 30–40× worse than the
tree's partition at the same K: its PC1-quantile init is
one-dimensional, Lloyd from it converges to slabs; the routine itself
is correct (started from the tree's partition it improves it). (b) At
the tree's own count, k-means++ from scratch matches the tree's
partition within ±10% on the worst-output predicted share: the tree
(ending with a Lloyd pass) is as good a partition as k-means gives at
that K, and every search-free k-means K-finder (farthest-first seeding
to tolerance; incremental with warm Lloyd) needs 10–160% MORE bins to
reach the same target. (c) The pilot partition alone already met ε on
the exact influence in all four cases (realized worst share 0.0081,
0.0082, 0.112, 0.117 against ε 0.01, 0.01, 0.141, 0.141), and the check
then spent 83–155 splits for nothing the tolerance needed. The waste
is the per-bin rules: growth's ε/L flag lands the total at ~½ε, and the
check's ε/L flag and ε·V/L pay test keep splitting while no rule reads
the total. (d) The spike blind spot stands: a bin holding a spike can
pass a mean test while its within spread is under-predicted many
times; the pilot's tail is wrong in SCALE and right in ORDER, so a
split does isolate the spike once decided; and the survey's measured
prototype values carry the spike at full size where the smoothed
field does not (raw survey aggregated to bins recovers 0.90–1.17 of
the true bin value; the smoothing loses up to 1.40 on top).

**The design, stated once: one tree, one loop.**

1. **Stage 1, the start partition: bisecting k-means on the pilot
   (Steinbach, Karypis and Kumar 2000), replacing section 3.** Start
   from one bin. Repeatedly split, by two-means on the pilot's
   predicted values of its own points (today's `two_means_split`), the
   bin with the LARGEST within estimate W_k (defined in item 3), and
   STOP when the predicted total within share, Σ_k W_k,c / V̂_c, is
   under ε on every output. Then one Lloyd pass (today's `_lloyd_all`).
   No ε/L flag anywhere; the per-bin share only ranks. The ellipse
   count (3.1) and the minimax k-means (3.2) are deleted. The leaves
   are the loop's start partition; 3.3's seed measurement (one forward
   evaluation per leaf, all in one batch) and first update apply to
   them unchanged.

2. **The loop's stop: CALIBRATED (the author's ruling, 1 October,
   after two offline tests by the coder on draw 1's 512 measured
   leaves; replaces section 5's Termination and section 4's pay test).**
   The within-bin variance is unobservable without splitting, and no
   estimate built from the pilot is an upper bound on it: on draw 1
   the pilot under-predicts the realized within by 10–30% on five of
   seven outputs (re-anchored base 0.75–0.98 before, 0.79–0.98 after
   one level of 3.3's update; the twenty largest-within leaves, half
   the within, 0.4–0.9 on the worst outputs). A stop at ε on any
   predicted estimate therefore fails in the unsafe direction
   (≈1.1–1.3ε on the updated vector, ≈2.8ε with a noise-corrected
   estimate). The staged pipeline meets ε only because its ε/L rule
   lands at ~½ε predicted and the pilot's error on cloudfil is ≈2.
   The stop must be a measured statement. The measurement is the
   split: a split's PREDICTED gain G_s,c is the vector's between
   variance of its two children at the time of the split, its
   REALIZED gain Δ_s,c is the children's measured between (section 4,
   formed from re-centred values, not debiased by b_Δ), and their
   ratio is the vector's within error in that leaf along that cut.
   Per output, cumulative over the run's own splits so far,
   κ̂_c = Σ_s Δ_s,c / Σ_s G_s,c (a ratio of sums, mass-weighted by
   construction). The STOP: κ̂_c · ΣW_c ≤ ε · V̂_c on every output,
   V̂_c = measured between + κ̂_c·ΣW_c (the within share of the
   calibrated total). Because the under-prediction sits in the
   extreme leaves and those rank first, κ̂ from the splits made is at
   least the ratio on the remainder, so the stop errs safe without a
   margin; as rounds accumulate, κ̂ covers more of the cloud and the
   remainder shrinks. When the in-place update has made the vector
   correct, κ̂ reads 1 and the calibrated stop is the uncalibrated one
   at no extra cost. The per-split pay test (τ = ε·V_btw/L, paid/strike/
   close) is REMOVED. A leaf is CLOSED only when its W_k is below the
   split's noise floor n_Δ,c + b_Δ,c (section 4's terms, unchanged):
   below it a split cannot return a measurable gain; a split whose
   realized gain lands under the floor closes both children. Products:
   n_closed_noise and their summed W replace n_closed_unpaid /
   n_strike_closes / n_closed_hot / V_win_closed; κ̂_c per round and at
   termination is a product.

3. **The within estimate W_k, used by stage 1's stop, the loop's
   ranking, each split's predicted gain and the stop (CORRECTED twice
   on 1 October: the survey-only form is constant within a prototype
   cell and blind to within-cell variance, measured at 9–21% of V; the
   noise-corrected form subtracts a reproducibility BOUND and removes
   real within, 0.36 of the realized on two outputs).** Per point i
   and output c the RE-ANCHORED value
   v_i,c = I_{bmu(i),c} + ( psi0_i,c − psi0bar_{bmu(i),c} ):
   the survey's measured prototype influence of the point's own cell
   (`bmu`) plus the field's deviation of the point from the field's
   mass-weighted mean over that cell — the measured cell means with
   the field's within-cell shape. v is the loop's base vector
   (section 2's psi_hat starts from v, not from psi0), updated in
   place by 3.3's shift/scale as every leaf and child is measured.
   W_k,c = p_k · Var_{i∈k}(psi_hat_i,c), the mass-weighted variance
   over the bin's points of the CURRENT vector (so the update scale
   is already inside it; no noise correction). The field psi0 keeps
   the two-means cut inside a bin and sd_hat's initial value; m_k is
   the vector's mean over the bin. W = 0 only where the vector is
   constant over the bin; such a leaf is never ranked, never closed,
   contributes 0. Stage 1's total is V̂_c = Var_all(v_i,c) and its
   stop is ΣW_c ≤ ε·V̂_c (uncalibrated: nothing is measured yet, and
   stage 1 only sets the start).

4. **Rounds (section 4 'Open bins', amended).** Every stage-1 leaf is
   measured first (3.3, one forward evaluation each, one batch) and
   the vector updated; the leaf's measurement is what the ranking,
   the spike test (a measured mean departing from the vector's by more
   than its within sd) and the split's noise estimate read, at the
   price of one evaluation per split later made. THE FIRST ROUND IS
   UNCONDITIONAL: no κ̂ exists before a split has been measured, so it
   splits the top-ranked leaves whose summed W reaches ε·V̂_c on the
   binding output (sized by ε, no constant), measures both children of
   each, and forms κ̂. Every later round: the excess E_c = κ̂_c·ΣW_c −
   ε·V̂_c on the binding output; split the top-ranked leaves whose
   summed W reaches E_c/κ̂_c (the W-mass that, realized at the
   measured ratio, meets the target), one two-means each. When the
   top-ranked leaf's W alone covers that, bisect that leaf on the
   vector by item 1's rule until the leaf's predicted contribution is
   under the target and measure ALL its pieces in one batch — a
   dominant bin yields its k pieces in one round. After each round:
   update the vector in the children, recompute W, κ̂, the stop. Bit
   identity across worker counts as now.

5. **Unchanged from the seeded branch as built:** the state vector
   psi_hat/sd_hat and 3.3's scale/shift update; one forward evaluation
   per child; the per-split measured noise δ and the floor n_Δ + b_Δ;
   raw U with re-centering at every variance (section 4 'Centering');
   the cap seed + 2·splits ≤ 1 + 2·M_X_used (safety only; it should
   never bind under the total stop); products split_noise and
   tree_centering_drift; section 6 parallelism.

6. **One path.** After acceptance the `'staged'` path (growth with the
   ε/L flag, the separate central-stencil pass, `check_rule` and the
   check module's flag/pay machinery) is removed and `joint_mode` goes;
   the code has this tree and this loop. Until acceptance the staged
   path stays as the validation's baseline.

**Why ε·V/L is not the criterion (the author's question, answered
once).** "Every bin under ε/L" is sufficient for "the total under ε",
never necessary; an efficient partition has a few bins well over ε/L
and most well under, because within mass is never spread evenly.
Enforcing fairness per bin is what lands the total at a fraction of ε,
and as a split-acceptance rule it has the wrong sign: over the total
it refuses the largest available gain for being below a quota, under
the total it keeps splitting. The only per-bin threshold with a reason
is the measurement's noise floor, which comes from η, not from ε.

**Constants.** None new. ε and η are the caller's; the noise floor is
section 4's; bisecting k-means has no constant.

**Validation (author's machine; ONE draw, nothing else).** Draw 1,
ε 0.01, the study configuration, this branch against today's pipeline
(qij-fitfix 0bd71ad: L 512, size-N 1266, V_btw/V_ij 0.9957–0.9987,
realized shortfall 0.13–0.43%): report L at stage 1 and at
termination, seed + loop evaluations and size-N total, rounds, the
realized shortfall 1 − V_btw/V_ij per output, κ̂_c per round and at
termination beside the realized ratio, n_closed_noise, and wall. ACCEPTANCE: realized shortfall ≤ ε on every output AND size-N
total below 1266. Reported, not gated: how far under ε the total
landed (the number this revision exists to raise toward ε).

## 0. Revision 2 (1 October 2026): the design fixes from the first validation

The first build (branch qij-seeded, e43c844 + fd32dcc) passed V1,
saved 16–28% of evaluations at ε 0.141 and 1–7% at ε 0.01, and failed
the shortfall gate on draw 7 at ε 0.01 (1.2–3.4% against ε 1%). The
coder's attribution (ij exact influence, by bin status) showed the
partition did its job — within variance left 0.15–0.41% on both draws,
half in bins closed unpaid and half in bins never proposed — and that
the deficit was MEASUREMENT: the measured bin means under-state the
exact between variance by 1–3% on six outputs. The discriminating
diagnostic rejected truncation and a base offset (the zero-perturbation
continuation is 2e-11–4e-9 per output) and found a multiplicative
shrink at tiny steps: the final bins' measured U regress on their exact
means with slope 0.976–0.994 in the smallest-mass tercile (single-point
bins, step t ≈ 3e-7) and 1.000 in the largest. Mechanism: a
continuation stops when its scaled gradient is within η_full, so its
endpoint is resolved to about η_full·|θ̂|; a perturbation whose true
displacement t·|U| is only a hundred times that is read short by a
percent, in both directions alike (a central stencil at the same step
shares it), and worse the smaller the step. The author ruled these
design flaws, not method ones; they are fixed here, and the mode's
adoption is the author's decision after the re-validation.

The four rules of revision 2, which override any conflicting line
below:

R1 **Step floor (every forward and central measurement in this mode).**
    The displacement must clear the fit's resolution by a measured
    margin: for bin k, t_k = max( step_parameter(forward_step(η_full),
    p_k), max_c √η_full · |θ̂_c| / |m_kc| ), the inner max over the
    measured outputs whose predicted bin mean from the updated vector
    satisfies |m_kc| > s_kc + δU_kc (outputs with a mean indistinguishable
    from zero or from their own error are skipped; if none qualifies the
    rule's step stands). The relative shrink is then ≤ √η_full (3e-4 at
    η 1e-7) and the truncation at the raised step stays of order 1e-4.
    Products: `n_step_floored`, and per split the step used.
    **FORM AND CONSTANT PENDING MEASUREMENT (coder's objection, 1
    October, accepted):** the √η_full form assumes the shrink is
    η_full·|θ̂|/(t|U|); the draw-1 control contradicts it — draw 1's
    η_full is 10× draw 7's and its small bins show no shrink
    (0.9986–1.000), where the model predicts 3× more. Whatever governs
    the shrink is not η_full as measured (the optimizer's real stopping
    on draw 7 looser than η_full, or something draw-specific). Before
    R1 is built: the step scan — on draws 1 and 7, ~20 single-point and
    small bins of the final seeded partition, each measured forward at
    t, 3t, 10t, 30t (start θ̂, eta η_full, as production) against the
    exact mean (ij); ~160 evaluations per draw on the author's machine,
    on the author's go. It gives whether the shrink falls as 1/t, the
    multiple of t at which it drops below 1e-3, the truncation at the
    raised steps, and why draw 1 is clean. R1's floor then takes its
    form and constant from that table, not from this model.
    **STEP SCAN DONE (coder, 1 October; arrays scratchpad
    step_scan_d{1,7}.parquet).** The shrink is set by the ABSOLUTE step
    t, not by η: tightening the continuation's η 100× leaves it
    unchanged; it appears only below t ≈ 2e-7 (draw 7's single-point
    bins at 6.3e-8: slope 0.981–1.003; at 3t = 1.9e-7: 0.9992–1.0000;
    draw 1's smallest steps are already 2e-7 because its η_full is 10×
    larger — hence the clean control). The upper edge is truncation:
    from t ≈ 4e-6–1e-5 upward the error reaches 1e-3 and more,
    sign-varying by output. The clean window is t ≈ 2e-7 to 2e-6 on
    both draws, every output within ~3e-4. So R1 becomes

        t_k = max( step_parameter(forward_step(η_full), p_k),  t_min ),

    **WITHDRAWN as a method constant (author, 1 October): a t_min read
    off this window would be a constant from two draws of one
    estimator, and η does not drive it (the scan showed η/100 changes
    nothing).** The rule this mode needs is the general one: a measured
    step must RESOLVE — the fix belongs in the fitter's continuation
    path (below), stated in machine epsilon and the fit's own
    quantities, not in a number from this dataset. R1 is therefore: no
    step floor in this spec; the fit-spec amendment, once the mechanism
    is verified, makes every continuation resolve its perturbation, and
    the step rule stays as it is.
    The mechanism is not identified; η does not control it, so the
    candidates were an absolute tolerance inside the fitter's finish or
    float resolution. **Coder's read of gmm.py (1 October):** not
    scipy's gtol (passed as 0; the stop is the package's own callback,
    and every explicit stop scales with η). Hypothesis: float resolution
    of ℓ in the trust-region ACCEPTANCE — a displacement δ gains about
    ½δᵀHδ in ℓ, which for a single-point bin at t ≈ 6e-8 is far below
    eps·|ℓ|, so trust-exact's actual/predicted ratio is float noise,
    steps are rejected, the radius collapses, the stall rule accepts the
    point (resid ≤ √η trivially, the perturbed gradient being ~t·|score_i|/N),
    and the fit stays where EM's few linearly-convergent SQUAREM rounds
    left it: a systematic partial response. This predicts an ABSOLUTE
    floor on the DISPLACEMENT t·|U| in scaled θ, ≈ √(eps·|ℓ|/λ_H),
    independent of η and of N — so t_min is a displacement floor and
    transfers across N and estimators in those units. **REFUTED by the
    4-evaluation check (1 October): the ℓ-gain is 770–1100× above
    eps·|ℓ| even at 1t, so the acceptance test is informative. The
    actual defect is a premature "converged" on the continuation path:
    the finish takes ONE Newton iteration and stops at resid 3.7e-8 <
    η = 1e-7 having covered 43% of the displacement (bin 355,
    log_reff); a tiny perturbation's own starting gradient g0 is
    already of order η, so a partial step passes the ABSOLUTE stop; at
    10t the gradient is 10× larger and the fit completes (response
    1.001). The scan's bins are bimodal (4 of 20 shrunk 10–19%, the
    rest within 0.2%): a pass/fail stop event. The fix lives in the
    fit spec (the finish), η-driven and constant-free: on a
    continuation the stop and the stall acceptance are RELATIVE to the
    problem posed, resid ≤ √η·g0 (the same √η the EM stop and stall
    rule already use; η·g0 would ask for a gradient below float noise),
    floored at the gradient's own float resolution; whether the initial
    trust radius on continuations must also follow the perturbation
    (why one Newton step covered only 43%) awaits two instrumented
    evaluations (bins 355 and 381 at 1t) on the author's go. This spec's
    step rule stays as written.** The same shrink
    reaches today's pipeline: its check's small one-sided children sit
    at t ∝ √η·p, below 2e-7 at η_full 1e-7 (TACC's usual value) for
    single-point children at tight ε; an audit of the sweep's child
    steps against 2e-7 is warranted before any further tightening of ε.
    [The same shrink affects the current check's small one-sided
    children at tight ε; that is reported to the author separately and
    is not changed by this spec.]

R2 **Central seed.** The seed bins are measured by the central stencil
    (2 per bin) at the R1 step: V2 found the forward seed's effect on
    V_btw at 1.2–2.0%, over the 1% rule.

R3 **Close on evidence, not on noise.** A split's gain is debiased by
    the squared-noise term and compared with its share MINUS the
    first-order noise: the lineage CLOSES only when, for every measured
    output, Δ_c − b_Δ,c < τ_c − n_Δ,c; otherwise both children stay
    candidates for the open test. (The first build's τ' = max(τ, n_Δ +
    b_Δ) raised the bar where the noise was large and closed splits
    within the noise of paying — the premature closing the author
    named.)

R4 **The identity as the noise.** With R1 removing the step dependence
    of the shrink, the parent-versus-children identity δ (section 4) is
    a same-footing comparison; δU_c for a split = max(|δ_kc|, η_full·
    |θ̂_c|/t) feeds n_Δ and b_Δ in R3, the reproducibility floor as the
    fallback. (Revision 1 had made δ information-only; R1 is what makes
    it usable.)

Everything else (the state, the seed count, the minimax k-means, the
open test with /N, the re-centring, the scale/shift update with its
measurability condition, the cap, the products, bit identity) stands
as written. Validation (section 7) is rerun on the same draws and ε
against today's pipeline (growth → central stencils → measured check),
with the first build's numbers as the record of what R1–R4 changed.

## 1. Why (the measured record)

* The separation of growth from measurement is inherited from the
  paper's fixed-bin design, not derived: growth is free, then every
  initial bin costs a central stencil (2 evaluations), then the check
  splits with measured gains. On the study-lm sweep at ε 0.01 that is
  2·337 + ~200 evaluations.
* Measurements are never fed back: a split's gain decides only whether
  its lineage continues; the children's predicted shares, the flag and
  the proposal ranking read the pilot as if nothing had been measured.
  Unpaid splits were 43–68% of check splits on the validation draws.
* The tolerance is delivered at a fifth of itself: growth stops where
  the PILOT's predicted share meets ε; the realized share on the final
  bins is 0.16–0.38 ε at the median (study-lm), with a tail of 2–3× the
  predicted at p90–p99. The margin is structural, not chosen.
* The tree is narrow at the top (2, 4, 8 splits), so a within-draw pool
  idles there; the bootstrap has one synchronization point.
* The per-point estimate is assembled from scattered pieces (the pilot
  array, the bin means, a scale factor, a gain ratio, a reconstruction
  for the sigma stage).

Cost arithmetic of the mode: the seed costs K' forward evaluations
(against 2·L0 central stencils today) and every split costs 2 (both
children measured, against the check's 1 + mass balance). The second
is a deliberate purchase: it buys independent measurements in every
bin (no error chain down a lineage) and the parent-versus-children
identity as a product. The saving the mode is built for is in the
count: a tree that stops on measured gains should end nearer the
tolerance than the staged design's realized fifth of it; section 7
measures whether the total comes out lower.

The risk this mode carries, stated (1 October, the coder's review):
the open test reads the pilot's within-bin shape, scaled to the
measured mean, so where the pilot under-predicts within-bin variance
in a bin whose MEAN it gets right, the bin is never proposed and that
variance is never discovered — the same blind spot the measured
check's flag has. The staged design hid it behind growth's ~2×
overshoot in predicted share; a tree that stops on measured gains
spends that margin. Two things argue it fares better than LBG did: the
count overshoot that sank LBG (a doubling schedule) is absent, the seed
being a lower bound and every later split measured; and the scale
update raises a bin's predicted within spread by the same factor by
which its level was under-predicted, and in spike bins the two
under-predictions have one cause, the smoothing of a concentrated
influence. That is an argument, not a measurement. No margin factor is
built in: a factor is a constant, the author's to set with a number in
hand, and section 7 is where the number comes from — V3 may fail on a
spike output at loose ε for exactly this reason, and if it does the
report says by how much and the margin becomes the author's ruling.

Baseline for every number in section 7: the study-lm configuration on
draws 1 and 7 at ε 0.01 and 0.141 (the `'staged'` run of the same
code). Oracle inputs: none in the rule; V_ij enters the validation
only.

## 2. The state

Two arrays, created from stage 1 and updated in place:

* `psi_hat` (N, q_meas): the per-point influence estimate, initialized
  to the pilot's psi0 minus its offset (the centred pilot, as the joint
  stage uses it today);
* `sd_hat` (N, q_meas): its uncertainty, initialized to the pilot's
  posterior sd (`uncertainty`), updated alongside psi_hat and REPORTED
  only (a product on diagnostic draws; no decision in this mode reads
  it — it is carried so the estimate and its uncertainty travel
  together for later use).

Standardization for cutting: Ψ̃ = psi_hat / std_pilot, std_pilot the
pilot's per-output sd over the N points, fixed for the draw (the
updates change psi_hat, not the metric the cuts are made in).

Every quantity the stage reports at termination is a function of
these two arrays and the bin labels (section 5).

## 3. The seed

### 3.1 The count

d_eff = min(d_x, q_meas), d_x the data dimension (2 on cloudfil). With
λ_1 ≥ … ≥ λ_{d_eff} the top eigenvalues of the correlation matrix of
the standardized pilot (q_meas × q_meas, eigh), the Zador count for a
d_eff-dimensional Gaussian at per-coordinate distortion ε:

    K_Z = ceil( [ C_d · 2π · ((d+2)/d)^((d+2)/2) · (λ_1⋯λ_d)^(1/d) / (d·ε) ]^(d/2) ),
    C_1 = 1/12,  C_2 = 5/(36√3)  (d > 2 is not built; raise)

(d = 1 reduces to the cost rule's M_ref = ⌈√(2.7/ε)⌉; d = 2 to
⌈1.008·√(λ_1 λ_2)/ε⌉, ≈ 1.75/ε on cloudfil: 9 at ε 0.2, 180 at 0.01.)
Sanity bounds, both the method's own quantities:

    K = clip( K_Z,  1,  min( L_pilot(ε), M_X_used ) )

L_pilot(ε) = the count today's `grow` reaches on the pilot at ε (free,
no evaluation): the seed never exceeds what the pilot itself asks for.
K must not exceed the number of distinct Ψ̃ rows (assert). Products:
`seed_K_zador`, `seed_K`, `seed_L_pilot`, `seed_lambda` (the d_eff
eigenvalues).

### 3.2 The partition: minimax k-means at K

Deterministic, no seed: initial centroids at the mid-quantiles
(k + ½)/K of the distinct projections of Ψ̃ on its first principal
axis (eigh of the q × q covariance, sign fixed by the largest-magnitude
loading positive), each point to its nearest seed in projection
(`ivq.kmeans_1d`'s init rule). Then the reweighted loop: weights ω
(q_meas,) start at 1; Lloyd on Ψ̃·√ω from the current labels to
convergence or 100 iterations (`_lloyd_all`, emptied centroids
dropped); compute each output's within share S_c on the UNWEIGHTED Ψ̃;
if max_c S_c did not decrease, stop and keep the previous labels; else
ω ← ω·S/mean(S), ω ← ω/mean(ω), repeat (cap 10 passes, a convergence
backstop, reported as `seed_reweight_passes`). The result is K' ≤ K
bins (drops counted). Products: `seed_L` (= K'), `seed_S_pred` per
output (the pilot's predicted within share on the seed).

### 3.3 The seed measurement and the first update

Every seed bin k: one forward evaluation at t_k = step_parameter(
forward_step(η_full), p_k), start = θ̂ (`counter`/`call_T`'s rule), all
K' in one pool batch. (Forward, not central, so the seed costs K'
evaluations rather than 2K'; the forward step's first-order truncation
is (t_k/2)·T'' along the bin's direction, expected at ~5e-4 of U_k at
these steps — the check's own one-sided splits have carried it at the
1% agreement with V_ij — and V2 measures it on draw 1's seed bins,
central against forward, before the tree is judged; if it exceeds 1%
of any output's V_btw the seed goes central, 2K', by the author's
ruling.); U_k = (T(+t_k)[measured] − θ̂[measured])/t_k,
mass-centred by cr = Σ_k p_k U_k (today's centering); cr is the seed's
noise check, a product per output (`seed_centering_residual`, in units
of √(V_btw)). A failed evaluation fails the draw's stage (as a failed
initial stencil does today). V_btw,c = Σ_k p_k U_kc²/N.

Update inside each bin k, per output c (the vector's bin mean m_kc,
its within sd s_kc, over the bin's points):

* scale, if |m_kc| > s_kc + δU_kc, with δU_kc = η_full·|θ̂_c|/t_k the
  measurement's own error (so the ratio U/m is determined to better
  than the bin's spread — a scale is used only where it is itself
  measurable):  psi_hat_ic ← psi_hat_ic · (U_kc/m_kc),
  sd_hat_ic ← sd_hat_ic · |U_kc/m_kc|;
* shift, otherwise (the mean is small against the spread or against
  its own error — a sign change inside the bin, or a near-zero mean):
  psi_hat_ic ← psi_hat_ic + (U_kc − m_kc), sd_hat unchanged.

(Revised 1 October after the coder's review: the first form, |m| > s
alone, let a noisy U/m multiply a whole bin's spread.)

After the update the bin mean of psi_hat equals U_kc exactly, in both
cases. Products: counts of scale/shift updates per output
(`n_update_scale`, `n_update_shift`, accumulated over the whole stage).

## 4. The tree: one measured loop

Rounds, until no bin is open or the cap binds. At the start of round r:
L = current bin count; V_btw,c measured as above; V_win,c = (1/N) Σ_k
p_k Var_k(psi_hat_c) (the vector's within-bin variance); V̂_tot,c =
V_btw,c + V_win,c; τ_c = ε·V_btw,c/L.

**Termination (added 1 October after the first validation; a design
flaw of this spec, not of the build).** The loop STOPS when every
measured output's total predicted within share on the updated vector
is under the tolerance, max_c Σ_k (p_k Var_k(psi_hat_c)/N) / V̂_tot,c ≤
ε — growth's own stop. The first text terminated only when no bin was
over its share ε/L, which ends only when gains fall below ε·V/L
EVERYWHERE and so splits on long after the total is met: on draw 7 at
ε 0.01 the tree ran to 524 bins (today's pipeline 435) and landed at a
realized within of 0.2–0.4 ε, overshooting the tolerance exactly as the
per-bin schedule does, with 1243 evaluations against 1237 and no
saving at tight ε. The per-bin share selects WHICH bins to split; the
total share says WHEN to stop.

**Open bins.** While the loop runs, a bin is open when it is not closed
(below) and max_c (p_k Var_k(psi_hat_c)/N) / V̂_tot,c > ε/L — bin k's
own contribution to V_win,c (the /N as in V_win,c's definition above)
against its share of the predicted total; growth's own selection rule
on the updated vector. (Corrected 1 October: the first text omitted
the /N, which would open every bin; the coder built it as corrected.) (After the seed, every bin is tested by this; there is
no flag against the pilot: the pilot inside a measured bin IS the
measurement.)

**Proposal.** One split kind: two-means on the bin's own Ψ̃ rows
(`two_means_split`, principal-axis halves then Lloyd, deterministic).
Infeasible (fewer than two distinct rows) closes the bin. [The
adjacency split and its CADJ pricing are not in this mode; if the
author wants it back it is a separate ruling.]

**Measurement.** Both children: one forward evaluation each at their
own t (as 3.3), start = θ̂, every child of the round in one pool batch.
A failed evaluation cancels the split: the parent stays, closed, its
evaluations counted.

**Centering (revised 1 October after the coder's review).** With both
children measured independently, mass balance no longer holds and the
mass-weighted sum of the current bins' U drifts with the measurement
noise; left in, that drift enters V_btw squared, a one-sided upward
bias. So every U is kept RAW (uncentred) and the centering is applied
where a variance is formed: V_btw,c = Σ_k p_k (U_kc − ū_c)²/N with
ū_c = Σ_k p_k U_kc the current mass-weighted mean, at every round's
τ, in every realized gain (Δ_c formed from the re-centred parent and
children values), and at termination; the seed's cr is ū at the seed.
`tree_centering_drift` = ū per output at termination, in units of
√V_btw, is the product that shows the noise the re-centring removed.

**The per-split identity (measured, reported; revised 1 October after
the coder's review).** δ_kc = U_kc − (p_a U_ac + p_b U_bc)/p_k — the
parent's measured mean against its children's combination, zero in
exact arithmetic — is stored per split (`split_noise`, array product:
split id, parent, children masses, t_parent, t_children, δ per
output). It is NOT the rule's noise: the parent and its children are
forward differences at different steps (t ∝ p), so δ carries the
difference of their first-order truncation terms, systematic and of
the order (t_k − t_child)/2 · T'', which at these steps can exceed
the reproducibility scatter several-fold and would inflate the floor
and close splits that pay. The floor stays the validated one (the
measured check's 2.4), with δU_c = η_full·|θ̂_c|/t_small:

    n_Δ,c = (2/N)·(p_a|U_ac| + p_b|U_bc|)·δU_c,   b_Δ,c = ((p_a + p_b)/N)·δU_c²,
    τ'_c = max(τ_c, n_Δ,c + b_Δ,c).

V3 reports |δ| against n_Δ per split so a later ruling can move the
floor onto the identity once a same-step form of it exists.

**The realized gain and the decision.** Δ_c = (p_a U_ac² + p_b U_bc²
− p_k U_kc²)/N, V_btw,c += Δ_c. The split PAID if Δ_c ≥ τ'_c for some
measured output. Paid: both children stay candidates (the open test
above decides them next round). Not paid: both children are CLOSED —
one miss closes, the measured check's revision 2 rule, which its
validation chose over the paper's two strikes (revision 1's two
strikes wasted one split per lineage at tight ε). The even-split case
the second strike was meant for is reported instead: `n_closed_hot` =
closed bins whose updated predicted within share still exceeds ε/L at
termination, with their summed predicted within share per output
(`V_win_closed`), so the reader sees what one-miss closing left on the
table. A closed bin is final.

**The update.** psi_hat and sd_hat inside both children by 3.3's
scale/shift rule to their measured means. The parent's measurement is
retained only in `split_noise`.

**The cap.** Seed evaluations + 2·splits ≤ 1 + 2·M_X_used (today's
maximum: growth's M_X_used stencil pairs plus the check's 1 +
M_X_used), tested at the top of each round (`tree_capped`).

**Bit identity across workers.** Every decision of round r reads the
round-start state and the round's own measurements; the pool only
changes who evaluates what. The build's V1 covers this at workers 1
and 4.

## 5. Termination and products

From the two arrays and the final labels, per measured output:
V_btw = Σ_k p_k U_k²/N (= the variance of psi_hat's bin means, since
every bin mean is its measurement); V_win_hat = (1/N) Σ_k p_k
Var_k(psi_hat) over the final bins; V_tot_hat = V_btw + V_win_hat =
Var(psi_hat)/N over all points. The reported variance is V_tot_hat,
per the author's ruling of 30 September on the study-lm sweep. The
sigma stage, when on, reads psi_hat as its per-point influence (today's
`joint_psi_hat` reconstruction is not used under `'seeded'`). Arrays:
psi_hat and sd_hat on diagnostic draws (replacing psi_hat/sigma in
points.parquet under this mode), bin labels, bin U, `split_noise`.
Scalars: seed_* (3.1–3.3), L, n_rounds, n_splits, n_closed_unpaid,
n_closed_infeasible, tree_capped, evals_seed, evals_tree,
`n_open_per_round` (array: the parallel width of each round), the
update counts, tree_centering_drift, wall per stage. gain_ratio, a_c,
the flag counters and the eps-inflated variant are not produced under
`'seeded'`.

## 6. Parallelism

Rounds are the synchronization points: the seed is one batch of K'
evaluations, each round one batch of 2·(open bins). Wall time at W
workers ≈ (evaluations)/W + (rounds)·t_eval; the seed supplies the
width at the top. `n_open_per_round` is the product that shows it.

## 7. Validation (author's machine, draws 1 and 7, ε 0.01 and 0.1414213562373095, `--workers 3` and `1`, hard time cap; nothing on TACC; the branch, not main)

V1 (the one pass/fail): `joint_mode='staged'` at the new code vs
current main (study-lm configuration), draw 1 ε 0.01: every product
identical, new columns excepted; and `'seeded'` at workers 1 vs 4 on
the same draw: identical in every non-timing column.

V2 (the seed, from the diagnostic arrays, offline, no evaluation beyond
the seed's own): per draw × ε: K_Z, K, L_pilot, the seed's worst
predicted share S/ε, the reweighting passes, the seed's centering
residual, the scale/shift counts; and the seed's exact realized
shortfall on its bins (V_ij from the ij products) beside its predicted
share. Plus the forward-bias line (3.3): on draw 1 at both ε, every
seed bin measured ALSO by the central stencil (K' extra pairs, this
validation only), the relative difference (U_fwd − U_ctr)/U_ctr per bin
and output, and its effect on V_btw per output; the seed stays forward
if that effect is under 1% of V_btw on every output, else central.

V3 (the whole stage): `'seeded'` vs `'staged'`, per draw × ε, per
output: total size-N evaluations (seed + tree vs stencils + check), L,
n_rounds and max/median n_open_per_round, the realized shortfall
1 − V_btw/V_ij and its ratio to the vector's predicted within share at
termination, n_closed_unpaid as a share of splits, V_tot_hat/V_ij.
Acceptance: shortfall_seeded ≤ max(ε, shortfall_staged) per draw ×
output, AND total evaluations under `'seeded'` ≤ under `'staged'` at
both ε on both draws. Reported, not gated: where the realized share
lands relative to ε (the staged design lands at a fifth; the measured
stop should land nearer), the round count against the staged design's,
and the per-split noise against the reproducibility floor.

Report: one markdown file with V1–V3, the branch and commit, wall per
stage at workers 1 and 3 (counts are the record).

## 8. Questions a coder will ask

* Which V_btw in τ at round r? The measured one at the round's start,
  before the round's splits, per output — the check's convention.
* Does the open test use V̂_tot or the pilot's V̂? V̂_tot = measured
  between + the vector's within, per output, at the round's start; the
  pilot's own total is not used after the seed.
* The scale rule when U_kc and m_kc have opposite signs? Then |m_kc| >
  s_kc fails in practice (a sign flip means the bin's mean is small
  against its spread); if it does not, the scale is negative and
  |U/m| scales sd_hat — allowed; count it (`n_update_negative_scale`).
* η_full? The draw's measured value, as everywhere; only as the floor
  of δU_c.
* One split kind only? Yes in this build (section 4); the adjacency
  split's return is a separate ruling.
* Line budget: `core/joint.py` gains the seed (3.1–3.3) and the loop
  (4) behind the mode; `'staged'` code untouched. If the file exceeds
  the budget, the seed goes to `core/seed.py`.
* Nothing here changes the reported interval or the analysis layer's
  variants beyond the products listed.
