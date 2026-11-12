# STATUS (2 October 2026): BUILT AND MERGED TO MAIN at 3c26023 (= tag study-unified; fast-forward from 284b762; main is the only branch; local, not pushed). The previous method is tag study-total (284b762) with its products in runs/total_s1000. ONE RUN PASSED acceptance on every output; the reviewer's one-vector audit complete (one defect fixed: a_bca over the final leaves, B_hat NaN); the TACC two-version study below is the adoption evidence.

Draw 1, ε 0.01, z 2, n_min 30, L_max 1094 (runs/scratch/unified_check_
mac), unified | A21 'total' | the emulation: leaves 463 | 246 | 412;
loop evaluations 924 | 493 | 823; size-N total 1286 | 854 | 1184; n_min
and L_max did not bind. κ̂ ± se 1.08–1.33 ± 0.02–0.07 (stable from ~7
splits); V̂_win/true on the final leaves 0.65–0.96; V_btw/V_ij
0.9936–0.9955; V_tot/V_ij 0.9984–0.9997; reported-variance error
0.03–0.16% (A21 0.11–0.53%); realized within share 0.45–0.64%, under ε
on every output (A21 0.80–1.38%, over on four); margin at the stop
−0.39…−0.63; scale updates 772–905 of ~930 per output. Leaves +12% vs
the emulation: the real loop's vector starts in SURVEY units (the root
is a centring; the scale enters bin by bin) whereas the emulation's
started a_c-corrected, so κ̂ settles at ≈ a_c² (1.04–1.39) rather than
≈ 1; κ̂ absorbs the gap as designed, inflating V̂_win and adding splits.
As built (judgement calls, accepted): the root centring is not routed
through _apply_update (U/m = 0 would wipe the vector) and is not
counted; std0 = the populated vector's std (survey units); the noise
floor from the measured-check spec §4 on ψ̂'s predicted child means;
f·W·κ̂ = G·κ̂ directly; every open leaf's cut re-formed each round, the
prefix sized from the non-closed candidates; L_max truncates an
over-sized round to its top prefix; B_hat/a_bca full width from every
split's central pair; rounds logged with start-of-round κ̂/V̂_win/
margin; bins/splits/rounds tables written every draw.

# The unified measured loop: one tree from one bin, every split measured, one state vector, one calibrated stop (planner, 2 October 2026; the author's go; a BUILD ORDER on a branch off main 284b762; merge on acceptance; z exposed to the user)

**TACC two-version study (coder, 2 October; cloudfil p1, N 10000,
draws 0–999, 10 ε, 414 in-basin draws; study-total = main 284b762,
study-unified = 3c26023 with z 2, n_min 30, L_max M_X_used; report
Research/QIJ_joint/analysis/summaries/compare_total_unified/REPORT.md,
figures analysis/figs/frontier_compare_{sizeN,seconds}.pdf).** Fraction
of draws with ANY output over ε, total | unified — realized within
share: ε 0.01 100% | 11%; 0.0316 98% | 14%; 0.1 99% | 27%; 0.2 96% |
46%; reported-variance error: 17% | 3%; 20% | 1%; 30% | 4%; 52% | 9%.
'total' fails on log_reff, log_axis_ratio and pa at every ε (81–99% of
draws); unified's per-output failure at tight ε is 1–8%, rising at
loose ε where the stop lands near ε (the structural under-read, not in
se). Cost at the same nominal ε: unified 1.4–1.7× total (ε 0.01: 1120 vs
788 size-N; ε 0.2: 237 vs 136). Frontier vs the bootstrap at matched
size-N: both beat it (−0.3 to −0.5% Winkler excess vs boot's +0.3 to
+0.75%) and tie with each other at matched cost. Coverage: unified
94.4–94.9% at every ε; total falls to 93.4% at ε 0.2. Reading: the same
interval quality per evaluation; unified spends what an honest ε
requires, total's ε is a label it does not meet on three outputs.
Merge is the author's decision.

## 0. The method in one paragraph

Quantize the data, survey the prototypes, fit the pilot; these populate
the one per-point vector ψ̂ (A20's architecture rule). Then ONE loop:
split the leaf whose measured mean was most wrong, measure both
children, correct the vector in them, record the split's realized and
predicted gain, and stop when the within term read from the vector,
calibrated by the splits made so far and with its error bar, is under
ε of the total. V_btw is the variance of the vector's leaf means (all
measured); V_win is κ̂ times the vector's within over the leaves. The
reported variance is their sum. Levers: ε (the caller's), z (exposed,
default 2; the margin against sampling scatter AND the structural
under-read, see 4.4), n_min (exposed, default 30), L_max (exposed,
default M_X_used), M_X (the survey rule), the GP pilot's settings.

## 1. What it replaces

Everything between the pilot and the report in today's second stage:
stage-1 growth (`grow`, the pilot-share stop), the stencil pass, a_c as a
separate step (A18), the pilot-share stop of `_run_total`, the floor and
calibration-round language of the χ² spec. What it keeps, verbatim:
the one-vector rule and the update rule (A20), the measured-error
ranking (A20), the two-means cut on the current vector, conservation
for the large child, the noise floor for closing, the evaluation cap,
one pool batch per round, bit identity across worker counts, the
sigma stage, the products' conventions.

## 2. Premise, checked (scratchpad unified_emulation.{py,report}; draw 1, ε 0.01, the exact influence standing in for the measurements)

z = 2: 412 leaves, 411 splits, 823 tree evaluations (today 493), total
1184 with the survey (today 854; the per-bin pipeline 1266);
reported-variance error 0.09–0.22% (today 0.11–0.53%); realized within
share 0.57–0.83%, under ε on every output (today 0.80–1.38%, over on
four); κ̂ 1.01–1.05 (the in-situ updates do the calibration's work at
the cut); V̂_win/true within on the final leaves 0.62–0.87 (the
structural under-read below the leaves, unseen by κ̂; the stop still
clears because it stops at 0.45–0.80 of the allowance). z = 1.645: 386
leaves, 1132 total, one output at ε. The floor of 30 splits and the
cap never bound. Missing from the emulation: finite-difference noise.

## 3. State and units

* ψ̂ (N × q): populated by the pilot in survey units (the GP's point
  predictions; nothing else exists at population), then updated in
  place by every measurement. There is no other per-point array.
* Leaves: a partition of the points; initially one leaf, all points.
  Per leaf: n, p = n/N, the measured mean U (q,), ē = U − m^pre (q,)
  where m^pre is ψ̂'s mean over the leaf before its own update.
* Splits: per split s, per output, G_s the predicted gain and D_s the
  realized gain (3.3), the parent id, the children ids.
* Units: W_l = p_l·Var_l(ψ̂)/N; D_s = (n_a(U_a − U_k)² + n_b(U_b − U_k)²)/N²,
  G_s the same with ψ̂'s means over the children before their update;
  V_btw = the mass-weighted variance of the leaves' U about their mass-
  weighted mean (the same quantity as today's, with every leaf measured
  — exactly today's V_btw += D_s per split, from V_btw = 0 at the root);
  V_ij, V_tot in the same units.

## 4. The loop

**4.0 The root (corrected after the coder's review).** No evaluation:
U_root ≡ 0 exactly (influences sum to zero over the data; scaling every
weight equally leaves the weight-normalized estimator unchanged), so a
root measurement would return θ̂ on both sides. The root's update is the
CENTRING shift ψ̂ ← ψ̂ − mean(ψ̂) per output. The survey-to-full-data
scale (A18's a_c, 1.02–1.18 on draw 1) is NOT applied globally — nothing
measured exists at the root; it enters bin by bin through the scale
updates of the first split's children (each scaled by its own U/m), so
after the first round the vector is on the full-data scale where it has
been measured and in survey units elsewhere, until measured. The
emulation's root was exactly this (U_root = 0, ē_root = −mean(pilot)),
so its κ̂ 1.01–1.05 includes the effect.

**4.1 Ranking.** Per leaf and output, r_lc = p_l·ē_lc²/N if |ē_lc| > δU_lc
(the update rule's noise at the leaf's step), else W_lc. Rank on the
binding output c* (4.4), descending, ties to the lowest id. One-point
leaves are never ranked; a leaf whose cut is infeasible or whose
expected gain f_l·W_lc·κ̂_c is below the noise floor n_Δ + b_Δ on every
output is closed.

**4.2 A round.** For candidates in rank order, form the cut (needed for
the closing test of 4.1 and for f_l = G_l/W_l); the round is the
smallest prefix whose summed EXPECTED GAIN f_l·W_lc*·κ̂_c* reaches the
excess X_c* (4.4) — a split removes about f·W of a leaf's within, not
W, so sizing by W undershoots and adds rounds; at least one leaf; the
whole list if never reached. (The emulation sized rounds by W; that
changes when the stop is tested between rounds, not the ranking, so
the validation reports the split count beside the emulation's 411.) For each leaf in the round: cut by `two_means_split` on
(ψ̂ − mean(ψ̂))/std0 over the leaf's points, std0 the per-output std of
the populated vector over the whole cloud (a unit, fixed at population);
children a (smaller by count) and b. All of the round's evaluations in
one pool batch; bit identity across worker counts.

**4.3 Measurement and update.** The small child a by a CENTRAL pair at
its own step (two evaluations) → U_a; the large child by conservation,
U_b = (p_k U_k − p_a U_a)/p_b. (Central, not forward: the forward bias
measured at 1.4–1.8% of V_btw on draw 1 fails the 1% rule; the central
pair is the stencils' own form, so cost per leaf is unchanged from
today's stencils.) Record ē_a, ē_b against ψ̂'s means before the update;
update ψ̂ in both children by the update rule; record G_s (computed
before the update) and D_s; V_btw += D_s.

**4.4 Stop.** After every round, per output:
  κ̂_c = Σ_s D_sc / Σ_s G_sc (ratio of sums over ALL splits; κ̂ = 1 and
    se = 0 if Σ G = 0);
  se(κ̂_c)² = Σ_s G_sc² (D_sc/G_sc − κ̂_c)² / (Σ_s G_sc)²;
  V̂_win,c = κ̂_c · Σ_leaves W_lc;
  se(V̂_win,c)² = (Σ_l W_lc)² se(κ̂_c)² + κ̂_c² Σ_{l: n_l ≥ 2} W_lc² · 2/(n_l − 1);
  V_tot,c = V_btw,c + V̂_win,c;
  X_c = V̂_win,c + z·se(V̂_win,c) − ε·V_tot,c;  c* = argmax_c X_c/(ε V_tot,c).
STOP when X_c ≤ 0 on every output and at least n_min splits have been
made (n_min: a USER-EXPOSED parameter like z — a QIJ argument and CLI
flag, default 30, the author's ruling; a ratio's standard error read
from fewer observations is itself unreliable; reported whether it
binds — in the emulation it did not). Before any
split κ̂ ≡ 1 and se ≡ 0, so the first rounds are sized by the pilot's
own excess. CAP (the author's ruling): on the number of LEAVES, L ≤ L_max, default
L_max = M_X_used — M_X prototypes in data space, up to as many leaves
in influence space; exposed to the user like z; `capped` flagged when
it binds. Every leaf beyond the first is one split at two evaluations,
so the loop costs at most 2·(L_max − 1) evaluations. On draw 1: 1094
allowed, 412 used. The old 1 + M_X_used evaluation cap goes.

**On z (what it covers).** se(V̂_win) reports the scatter of the splits
and the χ² variability of the leaves; it does NOT report the
structural under-read of the within below the final leaves (V̂_win/true
0.62–0.87 on draw 1), which κ̂ cannot see. The stop clears inside ε on
draw 1 because z·se lands the estimate at 0.45–0.80 of the allowance,
and that margin is what covers the under-read: z is the method's
margin against BOTH, not a pure confidence level. A user lowering z
lowers that margin (z = 1.645 put one output at ε in the emulation);
the parameter's documentation says so.

**4.5 Report.** V_btw, V̂_win, se(V̂_win), V_tot = V_btw + V̂_win (the
primary variance; the interval as today), κ̂ ± se, L, splits, rounds,
evaluations, the stop's margin X_c/(ε V_tot) at termination, the
update counts (scale/shift/negative), the per-leaf table (n, U, ē, W),
the per-split table (parent, children, G, D), and `z`. `joint_psi_hat`
= the final vector. S_pred, growth rounds, the stencil counters, a_c
and share_final no longer exist. ABC ingredients (ruling after the
reviewer's audit, 2 October): a_bca is formed over the FINAL LEAVES —
a true partition, every leaf having a U (measured or by
conservation); B_hat is NaN — its summand D2 exists only for measured
small children (a central pair's second difference is not conservable
to the large child), so no partition carries it, and the ABC bias is
parked by the author's standing ruling (normal intervals only; b̂ found
broken 10–200×); abc_interval therefore returns NaN, consistent with
the analysis's exclusion. Accumulating over every split's small child
at every depth (nested regions counted repeatedly, Σp ≠ 1) is wrong
and is removed. Cosmetic, fixed with it: `rank_rule` reported
consistently by _failed_result and _joint_defaults; `n_evals` is two
per split ATTEMPT.

## 5. Constants

z (default 2; EXPOSED to the user as a parameter of QIJ and the CLI,
the author's ruling; it covers the structural under-read as well as
the scatter, 4.4). n_min (default 30; EXPOSED to the user like z; reported when it
binds). L_max (default M_X_used; exposed; the cap on leaves).
ε and η the caller's; δU, the noise floor and the cap from η and M_X
as today. Nothing else.

## 6. Cost

Two evaluations per split (one central pair), so 2(L − 1); no root
evaluation.
Draw 1 estimate (section 2): 823 tree, 1184 total, against 493/854
today and the per-bin pipeline's 1266; the within then respects ε on
every output. The loop's own fit: sums, milliseconds.

## 7. Validation (author's machine; ONE run; the branch vs the stored A21 run of the same draw)

Draw 1, ε 0.01, the study configuration, z = 2. Report: L, splits,
rounds, evaluations and size-N total against 246 / 854; per output
κ̂ ± se, V̂_win ± se against the true within on the final leaves (exact
influence), V_btw/V_ij, V_tot/V_ij and the reported-variance error
against ε (today 0.11–0.53%), the realized within share 1 − V_btw/V_ij
(today 0.80–1.38%), the margin at termination, the update counts, and
whether the floor or the cap bound. Acceptance: reported-variance
error ≤ ε AND realized within share ≤ ε on every output, and the
emulation's leaf count within a factor the finite-difference noise can
explain (report the κ̂ trajectory beside the emulation's). Spec-vs-code
list before merge, as always: every site that reads anything but ψ̂,
the per-leaf ē and U, or the per-split G and D is a defect.
