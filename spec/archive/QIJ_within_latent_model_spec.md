# NOT ADOPTED (1 October 2026, evening): validation item 1 failed by more than its se; nothing built

Offline, draw 1, the total+A18 partition (246 bins), exact influence
from the ij product, cuts by two_means_split on the stored a_c-
corrected pilot, two levels deep, estimator 3.1–3.4 as written (coder,
scratchpad latent_offline.py). (A) The 2.3 identity fails: observed
children/parent ratio 1.24, 1.23, 1.74, 1.49, 1.40, 2.07, 0.95 against
the predicted (1 − φf̄)/(1 − f̄) = 1.05, 0.91, 1.01, 0.99, 0.98, 1.21,
0.94, with the capture ratio φ measured directly at 0.98, 1.06, 0.99,
1.00, 1.01, 0.88, 1.04 — about one, not premise 3's 0.74–0.94 on this
partition. The ratio's growth with depth is not the capture mechanism:
the pilot's κ itself rises as bins shrink (children's κ > parent's),
which the model forces into φ (φ̂ 0.38–1.03). (B) V̂_win / Σ T = 1.15,
1.41, 2.42, 2.57, 1.04, 2.37, 0.68; |error|/se = 1.0, 1.9, 1.0, 1.5,
0.1, 0.5, 4.4 — log_contrast under-estimated by 4.4 se (unsafe), four
outputs over by 2.4–2.6×. (C) An emulated loop spends 180 evaluations
on the unconditional first round and then sits at those ratios.
Diagnosis: κ is SCALE-DEPENDENT on cloudfil (the pilot's relative error
grows as bins shrink; the same finding as the GP-on-bins and lineage
tests), not a per-bin quantity inherited down a lineage; any estimator
calibrated at one scale and applied at another carries the drift. Any
successor must be indexed by bin scale and pass this same offline gate
before it is written. The spec below is kept as the record.

# The within-bin variance as a latent variable: a population model fitted on the run's own splits, and a stop that reads an estimate with its uncertainty (planner, 1 October 2026; written on the author's instruction; defined on the a_c-corrected pilot vector of A18 and on `tree_rule='total'`'s growth as the only stage 1; to be sent once 'total' is the single path on main)

## 0. What this replaces and why

The within-bin variance is never observed: a bin's measurement gives
its mean, and only splitting a bin reveals part of its within as the
between of its children. Every second-stage rule so far either read
the pilot's prediction as if it were the within (growth's ε/L flag,
the check's flag, `tree_rule='total'`'s stop) or read one
realized/predicted ratio with no noise law, no spread and no account
of which bins had been chosen (the calibrated stop of the seeded
spec's revision 3, which failed its selection test). The pilot's
error is not a bias to be corrected but a skewed latent quantity to be
modelled, and the data to model it are produced by every split the
run makes. This spec states the model, its estimator (closed forms),
its stop, and what it replaces. One constant enters (section 6).

## 1. Premises, checked offline on draw 1 (a_c-corrected pilot against the exact influence; scratchpad ivq_kmeans_check, 1 October)

Notation, per bin k and output c, all in the units the code's
`win_hat` uses (variance over N):
  W_kc = p_k·Var_k(ψ̂_c)/N   the pilot's predicted within mass,
  T_kc = p_k·Var_k(ψ_c)/N    the same with the true influence (unknown),
  κ_kc = T_kc / W_kc.

1. **κ is a skewed random variable across bins, median-unbiased.** On
   both of draw 1's partitions (512 and 246 bins): unweighted median κ
   0.9–1.2 on every output; sd of log κ 0.3–1.1; heavy upper tail. The
   mass-weighted mean Σ_k T_kc / Σ_k W_kc = 1.1–1.9, which is the
   mean–median gap a log-normal predicts from that spread. The pilot's
   "under-prediction" is skew, not bias.
2. **κ depends weakly on W**, the covariate the ranking selects on:
   weighted slope of log κ on log W between −0.15 and +0.27; the top
   half of W-mass against the rest differs by −30% to +35%, sign
   varying by output. The selection effect is a covariate effect.
3. **A split measures κ, noisily and with a known-direction bias.**
   With D the children's between on the true influence and G on the
   pilot (definitions in 2.2): corr(log D/G, log κ) = 0.40–0.65 across
   bins; Σ D/Σ G = 0.88–1.29 against Σ T/Σ W = 0.97–1.60, because the
   pilot's cut captures 0.41–0.50 of the pilot's within but only
   0.32–0.46 of the true within; per-split scatter sd(log D/G)
   1.2–2.3.

## 2. The model (per output c; bins independent given W)

**2.1 Latent law.** log κ_kc = μ_c + β_c·(log W_kc − ℓ̄_c) + ζ_kc with
ζ ~ N(0, τ_c²); ℓ̄_c is a centering (3.3), not a parameter. The expected
within of a leaf given its W is W_kc·m_c(W_kc) with
m_c(W) = exp(μ_c + β_c(log W − ℓ̄_c) + τ_c²/2).

**2.2 One split, one observation.** A split s of bin k is a two-means
cut of the bin's points on the standardized pilot vector (today's
`two_means_split`), children a (the smaller by count) and b, masses
p_a, p_b, p_a + p_b = p_k. Measurement as today's level split: one
forward evaluation of child a at its own step → U_a; U_b by
conservation, U_b = (p_k U_k − p_a U_a)/p_b, U_k the parent's measured
mean (every stage-1 leaf has one from its stencil; every child of a
measured split has one from here). Then
  D_sc = ( p_a (U_ac − U_kc)² + p_b (U_bc − U_kc)² ) / N     realized gain
  G_sc = ( p_a (m_ac − m_kc)² + p_b (m_bc − m_kc)² ) / N     predicted gain,
m the pilot vector's means over the same point sets; f_s = G_sc/W_kc
the pilot's capture fraction of the cut (known before measuring).
(D_sc equals today's Δ = (p_a U_a² + p_b U_b² − p_k U_k²)/N exactly,
by conservation.) The observation law: D_sc = φ_c · κ_kc · G_sc · ε_sc
with E[ε] = 1, φ_c the population capture ratio (true captured fraction
over pilot captured fraction; premise 3: 0.74–0.94 on draw 1). ε
carries the split's sampling noise and the forward-difference noise
(the floor n_Δ + b_Δ of the measured-check spec, unchanged in form);
its law is not needed: the estimator uses E[ε] = 1 and the observed
scatter.

**2.3 Two levels identify φ.** Within a bin, T_k = D_k + T_a + T_b
(the true within of the parent is the gain plus the children's
within) and W_k = G_k + W_a + W_b. Taking expectations with E[D_k] =
φκ_k G_k and T_k = κ_k W_k: E[T_a + T_b] = κ_k (W_k − φ G_k) =
κ_k W_k (1 − φ f_k), while W_a + W_b = W_k (1 − f_k). So the children's
κ, pooled over the pair, is κ_k·(1 − φ f_k)/(1 − f_k) — larger than
the parent's when φ < 1 (the compression the cut did not capture is
still inside the children). A later split of a child observes φ times
the child's κ. Hence, for a parent split with ratio D_k/G_k and the
splits of its children with pooled ratio (D_a + D_b)/(G_a + G_b),
  E[children's ratio] / E[parent's ratio] = (1 − φ f_k) / (1 − f_k),
which is 1 when φ = 1 (self-similar) and (1 − 0.85·0.5)/0.5 = 1.15 for
f = 0.5, φ = 0.85. This is validation item 1.

## 3. The estimator (closed forms; milliseconds; no estimator call)

All sums are over splits made so far in the run; per output; a split's
"level" is its depth below a stage-1 leaf (a stage-1 leaf split =
level 1; a child of a level-1 split, split = level 2; and so on). A
split of ANY level is an observation of φκ for the bin it split.

**3.1 Pooled ratio.** R = Σ_s D_s / Σ_s G_s over ALL splits: the
G-weighted mean of φκ over the split bins. Its standard error, the
ratio-estimator sandwich: se(R)² = Σ_s G_s² (D_s/G_s − R)² / (Σ_s G_s)².

**3.2 Capture ratio φ̂.** Over the set P of splits whose parent bin was
itself split (parent split π(s)): R_child = Σ_{s∈P} D_s / Σ_{s∈P} G_s,
R_parent = Σ_{s∈P} D_{π(s)} / Σ_{s∈P} G_{π(s)} (a parent counted once
per child split that exists), f̄ = Σ_{s∈P} G_{π(s)} f_{π(s)} /
Σ_{s∈P} G_{π(s)}, r = R_child/R_parent, and
  φ̂ = (1 − r(1 − f̄)) / f̄.
se(φ̂) by the delta method from se(R_child) and se(R_parent) (3.1's
form on each set, treated as independent): se(φ̂) = r·(1 − f̄)/f̄ ·
sqrt( se(R_child)²/R_child² + se(R_parent)²/R_parent² ). φ̂ is
UNDEFINED while P is empty or φ̂ ≤ 0 (the pair data are insufficient:
r ≥ 1/(1 − f̄) is noise, not physics); the stop (4.1) is not evaluated
while φ̂ is undefined, and no clipping is applied.

**3.3 Selection slope β̂.** Weighted least squares of y_s = log(D_s/G_s)
on x_s = log W_{k(s)} − ℓ̄, weights G_s, over all splits with D_s > 0;
ℓ̄ = Σ_s G_s log W_{k(s)} / Σ_s G_s (the G-weighted mean log W of the
split bins; the same ℓ̄ is used in 3.4). se(β̂) from the weighted
residuals in the usual way. (Splits with D_s = 0 contribute to R and
φ̂ but not to β̂.)

**3.4 Mean law and within estimate.** m̂_c(W) = (R/φ̂)·exp(β̂ (log W −
ℓ̄)). The τ²/2 term of 2.1 is inside R already (a ratio of sums
estimates the mean of κ, not its median), so τ is never estimated.
Over the current leaves l (every bin not yet split, stage-1 leaves and
children alike), with their own W_l on the A18 vector:
  V̂_win,c = Σ_l W_lc · m̂_c(W_lc).
Standard error, delta method, the three sources independent:
  se(V̂_win)² = V̂_win² · ( se(R)²/R² + se(φ̂)²/φ̂² )
               + se(β̂)² · ( Σ_l W_l m̂(W_l) (log W_l − ℓ̄) )².
No resampling anywhere.

**3.5 Before any split exists.** R, φ̂, β̂ are undefined; m̂ ≡ 1 is used
only to rank and to size the first round (4.2); the stop is not
evaluated.

## 4. The stop, the rounds, what closes, what goes

**4.1 Stop.** Evaluated after every round once φ̂ is defined (3.2).
Every output c must satisfy
  V̂_win,c + z·se(V̂_win,c) ≤ ε · ( V_btw,c + V̂_win,c ),
V_btw the measured between over the current leaves (updated by +D_s at
every split, as today's check does). The reported variance is
V_btw + V̂_win; se(V̂_win), R, φ̂, β̂ and their se's are products.

**4.2 Rounds.** Let X_c = V̂_win,c + z·se(V̂_win,c) − ε (V_btw,c +
V̂_win,c) be the excess on output c (with m̂ ≡ 1 and se ≡ 0 before any
split). The binding output c* = argmax_c X_c / (ε (V_btw,c + V̂_win,c)).
Rank the open leaves by expected within W_lc*·m̂_c*(W_lc*),
descending, ties to the lowest id. For each leaf in rank order form
its two-means cut and its predicted gain G_l (2.2) and expected gain
g_l = f_l·W_lc*·m̂_c*(W_lc*); the round is the smallest prefix whose
Σ g_l ≥ X_c* (at least one leaf; the whole list if never reached). All
of a round's small-child evaluations go in one pool batch (today's
`_split_task`); bit identity across worker counts as today. After the
round: V_btw += D_s per split; the children become leaves with their
own W; recompute R, φ̂, β̂, V̂_win, se; test 4.1.

**4.3 Closing.** A leaf is closed (never ranked again) when its
expected gain g_l on every output is below the split's noise floor
n_Δ + b_Δ (the measured-check spec's terms, in gain units), when its
cut is infeasible (`two_means_split` returns None), or when its
evaluation fails (as today: the leaf stays as a final bin). The
evaluation cap 1 + M_X_used stays as the safety limit (`check_capped`).

**4.4 Stage 1, and what goes.** Stage 1 is `grow_total` as built
(one bin split per round, the largest predicted within on the binding
output, stop when every output's predicted total within share is
under ε, then one Lloyd pass) and the central stencils on its leaves;
untouched by this spec. Stage 2 is `_run_total`'s loop with three
edits: the ranking weight (4.2: W·m̂ in place of W), the stop (4.1: the
estimate with its uncertainty in place of the pilot's share), and the
estimator of section 3 feeding both, fitted on the splits the loop
already records (D, G, the parent split, the level). Everything else
in the loop stays: the two-means cut, the small-child forward
evaluation with the large child by conservation, V_btw += D per split,
one pool batch per round, closing on infeasible or failed splits, the
evaluation cap. Nothing in this spec reads the GP posterior. The
per-bin path remains behind `tree_rule='perbin'` (A19: 'total' is the
default at 2282b8b); its removal, so that `tree_rule` disappears and
the code has one stage 1 and one stage 2, is a separate item on this
spec's acceptance.

## 5. Cost

Per split: the same evaluations as today's level split (one forward
evaluation; the large child by conservation). The fit: four scalars
per output on at most ~100 observations in closed form, no estimator
call. Starting from 'total' growth on draw 1 (246 leaves, 854 size-N,
no measured decision, κ = 1.92 on log_reff four percent from the
bound), stage 2 spends its calibration round and whatever the
estimate then demands; the per-bin path spent 120 check splits on
quotas and learned nothing about the within. With sd(log D/G) ≈ 1.5,
about 40–100 splits per output put se(R)/R near a quarter; the stop
ends the loop where the estimate says ε is.

## 6. Constants

One: z, the quantile at which "estimate plus uncertainty" is read
against ε, the author's, set once for the method; it is the only
margin in the design and it is explicit. Structural minima, not
constants: φ̂ must be defined (at least one parent–child split pair per
output, with φ̂ > 0) before the stop is evaluated. Everything else is
estimated from the run; ε and η are the caller's.

## 7. Validation (author's machine; ONE draw)

Draw 1, ε 0.01, the study configuration, on the A18 vector.
1. **Offline first, no estimator evaluations** (exact influence from
   the ij product; today's 0bd71ad partition of 512 bins): split every
   bin with ≥ 4 points by two-means on the pilot, and each child with
   ≥ 4 points again; form D and G on the exact influence and the
   pilot for every split (2.2); check 2.3's identity: the pooled
   children's/parent's ratio against (1 − φf̄)/(1 − f̄) with φ computed
   directly from the capture fractions (premise 3); then run 3.1–3.4
   on these splits and compare V̂_win per output with Σ T over the
   leaves, and se(V̂_win) with the actual error. If the identity fails
   by more than its se, or V̂_win misses Σ T by more than its se on
   most outputs, STOP: the model is wrong before any build.
2. **One run**, against the stored 'total' run under A18 (draw 1,
   runs/scratch/total_a18_check_mac: L 246, size-N 854, reported-
   variance error 0.13–0.66 ε, realized within share 0.80–1.38%) and,
   for the record, the per-bin run (0bd71ad: L 512, size-N 1266):
   L, splits by level, evaluations, size-N; per output V̂_win against
   the true within (exact influence), se against the actual error, the
   reported-variance error |1 − (V_btw + V̂_win)/V_ij| against ε, the
   realized within share, R, φ̂, β̂ with se's. ACCEPTANCE: reported-
   variance error ≤ ε on every output, and the actual error within
   se(V̂_win) on most outputs; the evaluation count is reported, not
   gated.
