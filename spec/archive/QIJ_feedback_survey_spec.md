# NOT ADOPTED (3 October 2026): the author's verdict — "methods with rules like 'after about 30 splits' will likely not pass muster for a well-posed method (why not 10 instead of 30? 1000 instead of 30?); if the method were deciding for itself when it should stop and reassess, that would be acceptable. But this smells like a hack job." Also: the author intended a fix confined to the PILOT STAGE (before any bins or splits), which this is not. Not a build order; no further refinement. Kept as the record (the trigger forms tried, the premise checks, the localisation problem). The open direction: a self-determined densification of the survey at the pilot stage, with the method's own measurements deciding when to stop.

# The feedback survey: the loop's measurements refine the survey where the pilot was too coarse (planner, 3 October 2026; the author's go; a BUILD ORDER on a branch off main 3304046; merge on acceptance)

## 0. What this is

The unified loop's measurements show where the pilot's within is wrong:
a split whose realized gain exceeds what the model with its margin
predicted has found structure below the survey's cell scale. Under
this spec that signal, once per run, re-quantizes the flagged cells
locally with more prototypes, measures the new prototypes with the
survey's own quantized fits, refits the pilot, repopulates the state
vector in those cells with the measured leaves' updates re-applied,
and continues the loop. Nothing else in the method changes: one
vector, the same ranking, the same stop (z, n_min, L_max), the same
products plus a few new ones. No new constant: the trigger reuses z,
the refinement factor is read from the loop's own leaf sizes.

Why: on draw 45 of cloudfil_u the pilot's within below the cell scale
was wrong by ×3–150 (×10⁴ on the angle), the stop under-read by up to
4ε, and a global 2× survey fixed most of it (all outputs inside ε at
ε 0.1; one at −1.2% at ε 0.01; loop 694 → 508 evaluations; survey
+360 size-N). The error was regional (filament leaves held 72–98% of
the missing within), so a local refinement should buy the 2× survey's
effect for a fraction of its prototypes. The alternative safety, the
floor on the stop, costs ×2–3 in full-data evaluations on such draws.

## 1. Premise check (OFFLINE, before building; the coder runs it)

Done for 2.1(a) (the bead never fires; draw 45 fires on the
under-reading outputs — recorded in 2.1). Remaining, for 2.1(b): on
the stored bead draw-1 run and the stored draw-45 1× run (splits in
order, parents' point sets from the tree, `bmu` from the points file,
κ̂ from the scalar product, W_parent from the final psi_hat over the
parent's points — exact for parents whose points were never updated
again, approximate otherwise; state it): count the splits with an
intra-cell parent and D > κ̂·W_parent on any output; the distinct
cells they flag; those cells' component majority; and the round at
which the first such split occurs. Expected: few on the bead; many on
draw 45, in filament cells, in later rounds. If the bead flags more
than a handful of cells, STOP and report.

## 2. The pass

**2.1 Trigger and localisation (corrected twice, 3 October; the
record of the first two forms is in the planner's notes).**
(a) TRIGGER, round-level: after round r, once n_min splits exist,
output c fires when the round's realized gains exceed the model's
within term AS IT STOOD GOING INTO THE ROUND:
  ΣD_rc  >  V̂_win,c(r−1)  = κ̂_c(r−1) · ΣW over the leaves standing before round r.
A round cannot realize more than what truly remained, so exceeding
the model's figure for it is a measured statement that the model
under-reads. Checked on stored products: the bead (draw 1, ε 0.01)
never fires (max ratio 0.77 over rounds 5–9); draw 45 fires from round
5 on exactly the under-reading outputs (pa 4.6 → 128 at ε 0.01;
log_axis_ratio 2.9 at ε 0.1). Any output firing triggers. (The same
before-round comparison is the correct form of the stop's floor; the
after-round form fires on the bead by construction.)
(b) LOCALISATION, per split at or below the cell scale: a data-space
cell is FLAGGED when a split s whose parent lies inside that one cell
(all its points share one `bmu`) realized
  D_sc  >  κ̂_c · W_parent,c   on any output,
a gain larger than κ̂ times the parent's ENTIRE predicted within —
possible only when the model is wrong for that parent, since a gain
never exceeds the parent's true within. Flags accumulate over rounds.
(The first localisation rule — every cell of the leaves split in the
triggering round — refined everything, because the triggering round
is early and splits every leaf.)

**2.2 When.** ONCE per run: the first round after n_min splits in which
(a) has fired on some output AND at least one cell is flagged by (b)
triggers the pass; later flags are
counted (products) but do not re-trigger (a second pass is a future
item, not this one). If no round ever flags, the run is byte-identical
to today's.

**2.3 Refinement factor per flagged cell.** k_c = ceil(n_cell / n_min_leaf,c),
where n_min_leaf,c is the point count of the SMALLEST leaf the loop
has produced inside cell c at the time of the trigger (the scale at
which the loop found structure there); k_c ≥ 2; cap k_c at n_cell (one
prototype per point at most). Report the distribution of k_c.

**2.4 Local re-quantization.** For each flagged cell, run the X-VQ's own
k-means (`xvq.py`'s routine, same init rule, seeded as the survey is)
on the cell's points alone with k_c centres; the sub-cells replace the
cell: new prototypes, new `bmu` for the cell's points, the moments
representation per sub-cell as the survey builds it (`--survey
moments`). Every other cell is untouched. Prototype ids: append.

**2.5 Re-survey.** Measure the new prototypes' influences exactly as the
survey measures the originals (the quantized fits of `xvq.prototype_
influences`, same step rule, same batch machinery, one pool batch);
the originals are NOT re-measured. Count these evaluations in the
survey's size-N accounting (`rows_by_stage`/`evals_prototype`), under
a new key `evals_prototype_refine`.

**2.6 Pilot refit.** Refit the GP (`influence_model`) on ALL prototypes,
original plus new, with the study's settings (gptrend, gpwidth local,
fit_weights mass) — a full refit, hyperparameters re-selected as in
the first fit. Predict ψ0 for the points of the refined cells only.

**2.7 Repopulation of the one vector (the ordering that keeps the
one-vector rule).** For the points of the refined cells, set ψ̂ to the
new ψ0 (survey units, as at population; the root centring re-applied
on the whole vector so the mean stays zero per output). Then re-apply,
in the order the leaves were measured, the update rule of every
measured leaf that intersects a refined cell, using each leaf's STORED
measured mean U (not re-measured) and the vector's CURRENT mean and
within sd over the leaf's points. Leaves outside the refined cells
are untouched. std0 (the cut's unit) stays the original population's.
After this, every measured leaf's mean equals its U again, and the
within shape inside the refined cells is the finer pilot's.

**2.8 Continue.** The loop resumes with the same leaves, the same splits
list (κ̂, se unchanged by the pass — they are measured quantities), the
recomputed W on the repopulated vector, the ranking, rounds and the
stop as before. The leaf cap L_max stays at the ORIGINAL M_X_used (the
pass adds prototypes, not leaf budget; report if it binds).

## 3. Products (added)

refine_triggered (bool), refine_round, n_cells_flagged, n_prototypes_
added, evals_prototype_refine (and its size-N), the k_c distribution
(min/median/max), the flagged cells' component majority if the
dataset carries components (diagnostic only), κ̂/se/V̂_win/the margin
per output at the round before and the round after the pass, the
count of later flags that did not re-trigger. The per-leaf and
per-split tables unchanged.

## 4. Cost

The new prototypes' quantized fits (cheaper than full-data fits) plus
one GP refit at the enlarged prototype count. On draw 45 the expected
scale: a few hundred quantized fits in the filament cells against the
floor's ~1300 full-data evaluations, and against the global 2× survey's
+1094 prototypes.

## 5. Constants

None new. z, n_min, L_max as exposed; the X-VQ's and the GP's settings
as the study's.

## 6. Validation (author's machine; ONE draw)

Draw 45 of cloudfil_u p2, ε 0.01, the study configuration, the branch
against the stored 1× run (runs/scratch/shearmix_diag_d45: L 348, loop
694, reported-variance error −0.4…−4.1%, pa κ̂ 12.3) and the stored 2×
run (…_mx2: L 255, loop 508, survey +1094 prototypes, errors +0.07…−1.2%,
pa κ̂ 1.5). Report: whether and at which round the pass triggered,
cells flagged and their components, prototypes added and their size-N,
loop evaluations and leaves, total size-N; per output the reported-
variance error against ε, the realized within share, κ̂ before/after.
ACCEPTANCE: reported-variance error ≤ ε on every output AND total
size-N (survey + refinement + loop) below the 2× run's total AND the
premise check of section 1 passed on the bead's stored products.
Spec-vs-code list before merge; every site reading anything but ψ̂,
per-leaf ē/U, per-split G/D, or the fixed units is a defect.
