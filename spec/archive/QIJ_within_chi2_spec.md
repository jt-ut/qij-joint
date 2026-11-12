# STATUS (2 October 2026): validation item 1 RUN OFFLINE before any build — the calibration under-reads the within term by up to 30–45% at 20–60 splits on draw 1, in a known direction, NOT covered by the standard error; no sampling scheme repairs it. NOT built. The author decides whether a measured-but-under-reading within term is worth its evaluations.

Result (scratchpad, draw 1, total+A18 partition, exact influence; κ̂ from
the selected level-1 splits applied to the leaves after those splits):
estimate/truth, [rank by W | equal-W-mass strata | every k-th leaf] —
20 splits: 0.60–1.34 | 0.58–1.10 | 0.46–1.43; 40: 0.73–1.04 |
0.70–1.11 | 0.70–1.20; 60: 0.69–1.00 | 0.56–1.10 | 0.79–1.25; all 180:
0.73–1.14. Reported se/κ̂ at 60: 0.08–0.25, against errors of 0.2–0.45:
the miss is a BIAS, not variance. It is not selection (stratified and
systematic sampling land in the same place); it is the capture ratio
(a split reads 0.77–1.0 of the within ratio it stands for) plus the
children being finer bins than their parents. In the bound's terms a
within term at ε read at 0.7 of its size is a reported-variance error
≈ 0.4ε — inside ε — but "estimate + z·se" does not describe it;
"estimate, known to under-read by a factor measured at 0.7–1.0 on one
draw" does. The text below is corrected where it claimed otherwise
(premise 5, 3.2) and kept as the record.

# The within-bin variance as a χ² quantity parameterized by bin count: one scale per output measured by the loop's own splits, a stop that reads the estimate with its error bar (planner, 2 October 2026; the author's model; built on `tree_rule='total'` and the A18 vector; a build order)

## 0. What this is

A bin's within-bin variance is a variance over n points. The standard
model for a variance over n points is a scaled χ² with n − 1 degrees of
freedom. This spec makes that the model of the second stage: for every
bin, the true within is one scale κ (per output) times the pilot's
predicted within, times a χ²_{n−1}/(n−1) factor. κ is measured by the
loop's own splits. The stop reads the within estimate with its error
bar against ε. No regimes, no fitted dependence on n, no lineage
bookkeeping; one constant (section 5) and one structural minimum.

It replaces, inside `_run_total`, the ranking weight (pilot within →
pilot within × κ̂) and the stop (pilot share ≤ ε → estimate + z·se ≤ ε
of the estimated total). Everything else in `_run_total` stays: the
two-means cut on the standardized vector, the small-child forward
evaluation with the large child by conservation, V_btw += D per split,
one pool batch per round, closing on infeasible or failed splits, the
evaluation cap. Stage 1 (`grow_total`, the Lloyd pass, the central
stencils) is untouched.

## 1. Premises, measured on draw 1 (exact influence from the ij product; the A18 vector; scratchpad ivq_kmeans_check, 1–2 October)

Notation, per bin k and output c, in the units `win_hat` uses:
W_kc = p_k·Var_k(ψ̂_c)/N (pilot's predicted within mass), T_kc the same
on the true influence, n_k the bin's point count.

1. **The χ² scatter is there, by count.** Across bins grouped by n, the
   observed sd of log(T/W) against the χ²_{n−1} prediction
   sqrt(trigamma((n−1)/2)): n 3–4: 0.8–2.0 observed vs 1.2 predicted;
   n 17–32: 0.2–0.8 vs 0.31; n ≥ 65: 0.10–0.33 vs 0.13. On both of
   draw 1's partitions.
2. **One pooled scale predicts the finer scale within the bound's
   tolerance.** κ̂ = ΣD/ΣG over the 180 level-1 splits of the 'total'
   partition, applied to the 186 level-2 bins: estimate/truth per
   output 0.95, 1.14, 0.98, 1.01, 1.03, 0.73, 1.10. The worst (logit_w,
   0.73) is a 27% error on a within term that is ~ε of the total, i.e.
   ≈ 0.4ε in the reported variance.
3. **The pilot under-predicts small bins and the pooled κ absorbs it.**
   Bins under the survey's cell size (N/M_X ≈ 9 points) hold 53–84% of
   the true within on the 'total' partition with T/W 1.3–2.5; bins
   above it have T/W 0.8–1.2. The pooled κ over the partition is
   1.2–1.9. This is why κ is not 1 and why it must be measured, not
   assumed. (A two-regime model was considered and rejected by the
   author: one scale, and the under-prediction lives in κ and in the
   error bar.)
4. **Split observations.** A split of a large bin is far more stable
   than a one-degree-of-freedom contrast (sd of log D/G 0.3–1.1 for
   n ≥ 17, because the cut follows the pilot's structure); a split of a
   small bin is a genuine one-dof contrast (sd 1.2–2.5). The cut's
   capture ratio (realized fraction of the true within over the
   pilot's fraction of its own) is 0.77–1.0 at level 1; its effect is
   inside premise 2's numbers.
5. **Selection and bias (corrected 2 October).** Ranking by predicted
   within selects bins whose ratio differs from the rest by −30% to
   +35%; but the under-read of the leaves' within at 20–60 splits
   (0.6–1.0 of the truth, rank order) is NOT selection — stratified
   and systematic sampling give the same — and is NOT inside the
   standard error (se/κ̂ 0.08–0.25 vs errors 0.2–0.45). It is the
   capture ratio (premise 4) and the finer scale of the children.
6. **The realized total's own χ² uncertainty** given κ, over the 246
   leaves: relative sd 0.08–0.12.

## 2. The model

**2.1 Bins.** For every bin with n_k ≥ 2: T_kc = κ_c · W_kc · ξ_kc,
ξ ~ χ²_{n_k−1}/(n_k−1), independent across bins, E[ξ] = 1,
Var[ξ] = 2/(n_k−1). A 1-point bin has W = T = 0. One κ_c per output,
unknown, measured (section 3).

**2.2 Splits.** A split s of bin k: the two-means cut of the bin's
points on the standardized pilot vector (`two_means_split`, as in
`_run_total`), children a (the smaller by count) and b, masses p_a,
p_b. Measurement as `_run_total` does it: one forward evaluation of a
at its own step → U_a; U_b = (p_k U_k − p_a U_a)/p_b with U_k the
parent's measured mean. Then, per output,
  D_sc = ( p_a (U_ac − U_kc)² + p_b (U_bc − U_kc)² ) / N   (= today's Δ)
  G_sc = ( p_a (m_ac − m_kc)² + p_b (m_bc − m_kc)² ) / N
with m the pilot vector's means over the same point sets. The
observation law: D_sc = κ_c · G_sc · ε_sc with E[ε] = 1; the scatter of
ε is observed (premise 4), its law is not needed. Each split is
recorded with (bin id, n_k, p_a, p_b, W_kc, G_sc, D_sc) per output; the
children become leaves with their own W and n.

## 3. The estimator (closed forms; no estimator call; no resampling)

All sums over the splits made so far in the run; per output.

**3.1 κ̂.** κ̂_c = Σ_s D_sc / Σ_s G_sc. A RATIO OF SUMS, which weights
each split by the within mass it represents (G ∝ W × capture). NOT a
precision-weighted mean: the χ² law makes large-bin splits precise and
small-bin splits noisy, and precision weighting would let large bins
(T/W ≈ 1) dominate while most of the within sits in small bins (T/W ≈
2) — biased low, the unsafe side (premise 3). This sentence is the
reason the code must not "improve" 3.1.
Standard error, the ratio-estimator sandwich:
  se(κ̂_c)² = Σ_s G_sc² (D_sc/G_sc − κ̂_c)² / (Σ_s G_sc)².
(The χ² noise of each observation is inside this empirical scatter; no
separate noise model.) Edge: an output with Σ_s G_sc = 0 (the pilot
predicts no within anywhere it was split) takes κ̂_c = 1, se = 0; its
V̂_win is then Σ_l W_lc, which is 0 or negligible by the same fact.

**3.2 The within estimate and its error bar.** Over the current leaves
l (every bin not yet split; stage-1 leaves and children alike), on the
A18 vector:
  V̂_win,c = κ̂_c · Σ_l W_lc
  se(V̂_win,c)² = (Σ_l W_lc)² · se(κ̂_c)²  +  κ̂_c² · Σ_{l: n_l ≥ 2} W_lc² · 2/(n_l − 1).
The first term is the calibration's VARIANCE (the split scatter); the
second is the χ² variability of the realized total around its
expectation given κ (premise 6), summed over the leaves by their
counts. Neither carries the BIAS of premise 5: κ̂ under-reads the
within ratio by the capture ratio (0.77–1.0 on draw 1) and the
children's finer scale, measured together at 0.6–1.0 of the truth at
20–60 splits. That under-read is a known-direction assumption the
bound must carry explicitly if this is built; it is not in se.

**3.3 Before any split.** κ̂ is undefined; κ̂ ≡ 1 is used only to rank
leaves (4.1); the stop is not evaluated (4.2).

## 4. Rounds, stop, closing, products

**4.1 Ranking and a round.** Rank the open leaves by expected within
W_lc*·κ̂_c* on the binding output c*, descending, ties to the lowest
id. A round splits the smallest top-ranked prefix whose expected gains
Σ f_l·W_lc*·κ̂_c* reach the excess X_c* (4.2), f_l = G_l/W_l the pilot's
own capture fraction of the leaf's cut (known before measuring); at
least one leaf; the whole list if never reached. All of a round's
small-child evaluations go in one pool batch (`_split_task`); bit
identity across worker counts as today. After the round: V_btw += D_s
per split; recompute κ̂, se, V̂_win on the new leaves; test 4.2.

**4.2 Stop.** A split is a split of a bin and yields D and G on every
output, so the split count is one number. The stop is evaluated only
once the run has at least n_min = 20 splits (the structural minimum
below which the sandwich standard error of a ratio is not usable; the
author may set it otherwise, see section 5); until then every round
is a calibration round sized to bring the split count to n_min, by
rank on the binding output with κ̂ as it stands (≡ 1 before the first
round). The binding
output is c* = argmax_c X_c / (ε (V_btw,c + V̂_win,c)) with
  X_c = V̂_win,c + z·se(V̂_win,c) − ε · ( V_btw,c + V̂_win,c ).
STOP when X_c ≤ 0 on every output. The reported variance is
V_btw + V̂_win; se(V̂_win), κ̂ and se(κ̂) are products beside it.

**4.3 Closing and the cap.** A leaf closes (never ranked again) when its
expected gain f_l W_l κ̂ is below the split's noise floor n_Δ + b_Δ (the
measured-check spec's terms, in gain units) on every output, when its
cut is infeasible, or when its evaluation fails (the leaf stays as a
final bin). The evaluation cap 1 + M_X_used stays as the safety limit
(`check_capped`).

**4.4 Products.** Per output: kappa_hat, se_kappa, V_win_hat (=
V̂_win), se_V_win_hat, V_tot_hat, share_final = V̂_win/(V_btw+V̂_win),
n_splits; the per-split table (bin, n, p_a, p_b, W, G, D per output).
`gain_ratio`, `S_pred` (growth's) and `joint_a` as today.

## 5. Constants

z: the number of standard errors of the within estimate added before
the comparison with ε — the author's, set once, an argument with no
default; it is the one margin in the design and it is explicit.
n_min = 20: the number of splits before a standard error is read. A
CHOICE, the author's (not a structural fact): below ~20 observations
the sandwich se of a ratio is itself unreliable, but the number is a
ruling, and the alternative — a target relative se of κ̂ — is another
constant. ε and η are the caller's. Nothing else.

## 6. Cost

Per split one evaluation, as today's level split. The estimator is a
few sums. On draw 1 under 'total' (854 size-N, no splits today), the
calibration is n_min = 20 splits per output at least, and the stop's
own demand beyond that: with se(κ̂)/κ̂ ≈ sqrt(2/n) + heterogeneity
(premise 4), 20–60 splits put the within term's error bar at 20–50%
of itself, i.e. 0.2–0.5ε in the reported variance. Expect 50–100
evaluations per draw: 'total' + this ≈ 900–950 against the per-bin
pipeline's 1266, with a measured bound in place of the κ ≤ 2 bet.

## 7. Validation (author's machine; ONE draw)

Draw 1, ε 0.01, the study configuration, on main (2282b8b+).
1. **Offline first, no estimator evaluations** (exact influence; the
   total+A18 partition of 246 bins; the splits of the premises): run
   3.1–3.2 on the level-1 splits in RANK ORDER (the loop's own
   selection, top-W first), at 20, 40, 60 and all 180 splits; for each
   count report κ̂ ± se, V̂_win ± se on the level-2 bins against Σ T,
   and |error|/se. If at 60 splits the estimate misses Σ T by more
   than 2 se on more than one output, STOP and report; nothing is
   built. (Premise 2 says it passes at 180; this item says at what
   count.)
2. **One run**, against runs/scratch/total_a18_check_mac (L 246, size-N
   854, reported-variance error 0.13–0.66ε): L, splits, evaluations,
   size-N; per output κ̂ ± se, V̂_win ± se against the true within
   (exact influence), the reported-variance error |1 − (V_btw +
   V̂_win)/V_ij| against ε, share_final, and the stop's margin X_c at
   termination. ACCEPTANCE: reported-variance error ≤ ε on every
   output and |error of V̂_win| ≤ 2 se on every output; the evaluation
   count reported, not gated.

Spec-vs-code: every site where the code must differ from this text is
reported by the coder before merge and the text corrected.
