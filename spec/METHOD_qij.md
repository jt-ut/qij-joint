<!-- Ported from vqboot/fable/bed/METHOD_qij.md on 2026-11-06; brought to the package's method on 2026-11-12. -->
# QIJ: derivative-free estimation of the influence variance with a certified error

Status: the method as implemented in `src/qij_joint/method.py` (`run`), described once, in its parallel form. It is the
test bed's method (`bed/method.py`, 2026-11-03) with the changes ruled since the port: every evaluation a refit on the
full data (step 1); M from the budget rule (step 1, `spec/METHOD_pilot_budget.md`); the bound of a row, the offset Ω in
the entry criterion and the stop, and the repair (steps 3, 5, 7). The plain account for a reader of the paper is
`spec/METHOD_as_built.md`. Measured numbers are in section 11.

## 0. Notation (proposed; cleaned up before the port)

| symbol | meaning | was |
|---|---|---|
| N, d | data points and their dimension | — |
| J (the paper's q) | number of outputs of the statistic; j indexes outputs | q |
| ψ (N × J) | the influence function of the statistic at the data points, never computed | Z in the bed |
| V_j = (1/N) Σ_i ψ_ij² | the target: the variance of the influence, per output | — |
| ε | the required relative precision of V̂ | — |
| α = 0.16 | the one-sided miss level, the bootstrap's own: 2/ε² replicates miss by more than ε one time in six | level 0.84 |
| M | number of pilot cells, M = 3 (κ_r / ε)^(r/2), r = min(d, J) (the budget rule, `METHOD_pilot_budget.md`) | C |
| evaluation | one refit of the statistic on the full data with the weights of a group perturbed; it returns Σ_i u_i ψ_i for the chosen weight vector u, up to the step term. The unit of cost. | read, contrast, look |
| η | the estimator's relative accuracy, measured once on the full data at the start (η_full) | — |
| the step term | the amount by which a forward-difference evaluation differs from Σ_i u_i ψ_i, of relative order √η | — |
| the bound of a row | the allowance carried by a row of ψ̂ for its step term: √η for a measured row (0 with exact evaluations); for an inferred sibling the parent's and the measured child's bounds combined as their means are | — |
| Ω_j = Σ_l p_l bound_lj² | the offset: the allowance added to the entry criterion and the stop for the inferred leaves | — |
| repair | the direct evaluation of an inferred leaf when Ω alone blocks the stop | — |
| L | number of leaves of the tree; l indexes leaves, n_l points, p_l = n_l/N, m_l the leaf's mean influence | K |
| ψ̂ | the model: the kernel interpolant through every evaluation made so far | influence model |
| ψ̃ | the reported model (section 8) | certified field |
| R = (1/N) Σ_i (ψ_i − ψ̃_i)² | the error of the reported model; the certified quantity | R, Rcf |
| r_l | the error term measured on leaf l by its test evaluation | v̂R_l |
| R̂ = Σ_l r_l, q_α, U = R̂ / q_α | the estimate of R, the exact quantile of its law, the upper limit | — |
| δ_j = ε · V_lb,j | the passing value of U_j; V_lb a lower bound on V_j known exactly (section 7) | delta |
| V̂_j = 2 P_j − ‖ψ̃_j‖²/N | the reported estimate | model-corrected estimate |
| k | the cap on splits per round = the number of workers | BED_BATCH |
| round | one pass: selection, a batch of splits, the test | — |

Dropped from the description: W (the within-leaf variance of ψ, recorded for diagnostics only; nothing in the method
depends on it), "look", "read", "contrast", "certification step", "trigger", "field", the sequential loop.

## 1. What the method does

It estimates V to relative precision ε using evaluations of the statistic only, and certifies the error of the
estimate with a one-sided statement at level 1 − α. The only randomness is in weights the method draws itself; the
influences are fixed numbers and the statement is conditional on the data. Every round's evaluations are independent
of one another and run on k workers at once, so the wall time is (number of rounds) × (one refit) plus overhead, the
same shape as a bootstrap's B/k refits.

## 2. The eight steps

**Step 1 — pilot.** Quantize the data into M cells by k-means with greedy k-means++ seeding (sklearn's rule,
2 + ln M candidates per step) on the clustering coordinates (the case's own, else the standardized data); section 11
shows why the seeding matters. M is set before anything else by the budget rule (`METHOD_pilot_budget.md`): the pilot
is given the cost the tree will spend, M = 3 K_r(ε), K_r(ε) = (κ_r / ε)^(r/2) the number of leaves Zador's law gives
for a normal influence of dimension r = min(d, J) at within variance ε; 300 at r = 2 and ε = 0.01, 1319 at r = 3,
independent of N. Make M evaluations, one per cell, in parallel: evaluation c has weight 1/n_c on the points of cell c
and returns m_c, the mean influence of the cell. Every evaluation, here and below, is one refit of the statistic on the
full data with the weights of the group perturbed (weights ω = 1 − t Σu + t N u, t = δ_step / max|N u − Σu|,
δ_step = 2 √η), a forward difference continued from θ̂. The bed fed exact cell means; the package's forward difference
agrees with the exact mean to the step term (section 4). Evaluations on the quantized data (a refit on the M prototypes
with their masses) were measured and rejected: they return the influence of a different problem and cost the same wall
time as a full refit (`METHOD_as_built.md` section 4; `spec/archive/quantized_run_20261107.log`).

The kernel scale of every point is its cell's scale: the mean distance from the cell's prototype to the prototypes it
shares a Voronoi boundary with (the CONN graph), or the nearest prototype when the cell has no neighbour.

**Step 2 — the model ψ̂.** ψ̂ is the minimum-norm interpolant of the exponential kernel
k(x_i, x_j) = (2 σ_i σ_j / (σ_i² + σ_j²))^{d/2} exp(−√(2 |x_i − x_j|² / (σ_i² + σ_j²))) through every evaluation made so
far: each evaluation is one linear constraint Σ_i u_i ψ̂_i = (returned value), and ψ̂ satisfies all of them exactly. A
refit is one linear solve with the Gram matrix of the constraint rows. The M pilot rows never change, so their block
is factored once; each refit factors only the border and the Schur complement of the tree and test rows. ψ̂ is the
method's memory. Its residual ψ − ψ̂ is what no evaluation has determined; for every constraint row u,
Σ_i u_i (ψ − ψ̂)_i = 0 exactly.

**Step 3 — a round: selection and the batch.** The tree starts as one leaf holding all N points. Each round ranks the
splittable leaves (n_l ≥ 2) by the model's own within-leaf variance, w_l = p_l Var_l(ψ̂) summed over outputs, in
descending order. Selection uses ψ̂ only, never a returned value. The round takes the first j leaves of the ranking,
where j is the smallest number such that the stop criterion of the current phase would hold if those leaves
contributed nothing (section 4), capped at k and at least 1. For each taken leaf the cut is the two-means partition of
the leaf's ψ̂ rows (J-dimensional), started from the sign of the projection on their leading principal direction and
iterated until no point moves; it is deterministic and library-free. The j child-mean evaluations run in parallel (one
per split: weight 1/n_a on the smaller child a); the sibling's mean follows exactly from the parent's,
m_b = (n_l m_l − n_a m_a)/n_b, and its row is the same combination of the parent's and the child's rows. The parent row
is replaced by the two child rows and ψ̂ is refitted once. Each row carries a bound for its step term: √η for a measured
row (0 with exact evaluations), and for the inferred sibling (n_l bound_l + n_a bound_a)/n_b, the combination its mean
was formed by, so a leaf inferred through several generations carries the step terms of every measurement it was formed
from. The bound is an allowance in the criteria (steps 5 and 7), not a change to any value: a measured leaf's mean is
the evaluation's result, and an inferred leaf's mean is the exact consequence of the measured ones.

**Step 4 — the batch count.** Before the test has begun, the forecast weights are w_l per output. Remove the taken
leaves' weights in ranking order; stop taking when Σ_remaining w_l ≤ ε q̃ V_lb on every output, where q̃ is the exact
quantile of section 6 computed with the remaining weights. Any j splits reduce the forecast by at most the j largest
weights, so this count never exceeds the number of splits the method would make one at a time. Once the test has begun,
the measured error terms r_l replace the forecast weights: remove them in descending order of max_j r_lj/δ_j until
Σ_remaining r_l / q_α(remaining) ≤ δ on every output. The leaves split remain the top j of the ψ̂ ranking (a leaf chosen
by its measured term would bias the surviving sum low). At k = 1 the method is the sequential one exactly.

**Step 5 — entry.** The test begins at the first round where the forecast itself passes:
Σ_l w_l + Ω ≤ ε q̃ V_lb on every output, with q̃ the exact quantile for weights w_l and Ω_j = Σ_l p_l bound_lj² the
offset for the step terms the leaf means carry. Until then no leaf carries a test
evaluation. The forecast is biased low (the interpolant is smoother inside a leaf than the influence), so the test
begins early rather than late; a test that does not pass costs only the evaluations it adds to ψ̂.

**Step 6 — the test.** Every leaf without a test evaluation receives one (after entry: the 2j new children of the
round), all in parallel. For leaf l draw independent standard normal multipliers on its points, centre them
(v_i = w_i − mean w, so Σ_i v_i = 0), and evaluate with weight u = v/n_l; it returns t_l = Σ_i v_i ψ_i / n_l. The row
enters ψ̂ and ψ̂ is refitted once. The leaf's misfit is e_l = t_l − (the same weights applied to the model fitted with
every row except this one), obtained from the full factorisation by the leave-one-out identity, with an explicit refit
for rows involved in a linear dependency. Because the multipliers are fresh and the model without this row never saw
them, n_l e_l is, given the leaf, exactly normal with variance Σ_i d_i² where d = ψ − ψ̂_{−l} on the leaf, so
r_l = p_l n_l e_l² is unbiased for p_l Var_l(ψ − ψ̂_{−l}) and r_l / E r_l is exactly χ²₁, on every leaf, of every size,
for every output. A leaf's term stays with it until the leaf is split; the split replaces it by its children's.

A leaf is determined when the constraints fix its influences exactly: every point's indicator lies in the row space of
the active constraint rows (projection residual 1 − h ≤ 10⁻⁸; measured, determined points sit at ≤ 10⁻¹² and the rest
at ≥ 10⁻³). This is an algebraic statement about the rows, not about the kernel, so it holds for any ψ. A determined
leaf's term is set to zero and it leaves the quantile's weights. Only leaves met by at least n_l rows can qualify.

**Step 7 — the limit and the stop.** R̂_j = Σ_l r_lj. The law of R̂ / E R̂ is Σ_l (E r_l / Σ E r) χ²₁,l; its α-quantile
q_α,j is computed exactly by inverting the characteristic function (Imhof), with the measured terms in place of their
expectations. U_j = R̂_j / q_α,j exceeds the error in all but a fraction α of multiplier draws. The method stops at the
first round with U_j + Ω_j ≤ δ_j = ε V_lb,j on every output, where V_lb,j = max(B_j, F̃_j /(1 + √ε)²), B_j = Σ_l p_l m_lj²
(the leaf means are exact, so B ≤ V exactly) and F̃_j = ‖ψ̃_j‖²/N. Since q_α ≤ 1, any output with R̂_j > δ_j fails
without the quantile being computed. When every leaf is determined, U = 0. The offset is there because the test cannot
see a constant shift of ψ̂ over a leaf whose mean was inferred: the centred contrasts sum to zero on the leaf. When U
alone passes and only Ω blocks the stop, the round repairs instead of splitting: the inferred leaves with the largest
p_l bound_l² / δ are evaluated directly (one evaluation each, at most k per round), which replaces their inferred row and
sets their bound to √η. Measured on the suite of 2026-11-07: Ω below a thousandth of δ on every draw, no repair made.

**Step 8 — the report.** The reported model ψ̃ equals, on each tested leaf that is not determined, the model fitted
without that leaf's own test row, and ψ̂ elsewhere. One evaluation per output with weight ψ̃_j /N returns
P_j = (1/N) Σ_i ψ̃_ij ψ_ij. The estimate is V̂_j = 2 P_j − ‖ψ̃_j‖²/N. From ψ_i² = 2 ψ̃_i ψ_i − ψ̃_i² + (ψ_i − ψ̃_i)²,
V̂_j = V_j − R_j exactly, so the statement is

    V̂_j ≤ V_j ≤ V̂_j + U_j, one-sided, level 1 − α, for every output j, with U_j ≤ ε V_lb,j ≤ ε V_j.

Cost: M (pilot) + (number of splits) + (number of test evaluations) + (number of repairs) + J evaluations, each one
refit on the full data; plus the 4 of the start (θ̂ and the three of η_full).

## 3. Why these choices

- Selection by ψ̂'s within-leaf variance: the model's own statement of where it varies most, which is where a
  piecewise-constant reading of the influence loses most variance. It uses no returned value, so the surviving test
  terms are not selected on their size.
- The batch count as a lower bound: the method never splits a leaf that the one-at-a-time method would not also have
  split before its stop. The price of a batch is only that its j splits are ranked from one ψ̂ and its 2j test
  evaluations meet one ψ̂, so each evaluation carries slightly less information than in sequence. Measured: zero at
  N = 10⁴ with k = 16 (section 11); 1 to 17 % when k is a third of the tree.
- Centred Gaussian multipliers: they make r_l / E r_l exactly χ²₁ for any pattern of deviations. Balanced random signs
  do not: on a two-valued pattern such as (a, a, −a, −a) the sign evaluation returns exactly zero one time in three, and
  on the two-valued tail-probability output of mvt that passed the test falsely.
- The cross-fitted misfit: a leaf is assessed by a model that never saw its own test row, so the row cannot flatter
  itself; the reported model uses the same cross-fitted values on the tested leaves, which is why V̂'s error is the R
  the test measures.
- One exact interpolant: every evaluation reaches every later decision. A compactly supported kernel would make the
  columns sparse and the overhead smaller, but at the same constraints its error of ψ̂ was five times larger
  (bed/ASSESS_overhead.md), which is repaid in evaluations, the expensive unit.
- M from the budget rule: the pilot is given the cost the tree will spend, with the tree's leaf count taken as Zador's
  count for a normal influence of dimension r = min(d, J) at within variance ε. It depends on ε and not on N, as the
  rest of the cost does. The bed's M = N / ln N (1086 at N = 10⁴) was an occupancy rule for exact cell means; with every
  pilot evaluation a full refit it cost more than the tree it served (section 11). The tree reaches ε from either pilot;
  the pilot changes the route, not the stop.

## 4. What is exact and what is approximate

Exact, given the data: the conditional law of each term (χ²₁ times its expectation); the identity V̂ = V − R; the
sibling means; the leaf means as constraints (so the residual has zero mean in every leaf and R equals the within-leaf
variance of the residual); the quantile of a weighted sum of χ²₁ variables.

Three approximations, all stated and measured rather than derived:

1. The weights of the quantile are the unknown expectations E r_l, replaced by the measured terms r_l (the standard
   plug-in). With many leaves q_α is near 1 (0.93 at a thousand leaves) and the plug-in matters little; with few leaves
   it is well below 1 and the limit widens accordingly.
2. The test is run every round after entry, each at level 1 − α, and the method stops at the first pass; consecutive
   tests share most of their standing terms. The run-level miss rate is therefore not α by construction. Measured:
   section 11.

3. The step term. An evaluation is a forward difference of a refit solved to relative accuracy η at step δ_step = 2 √η:
   the solver's part of the difference is of order η / δ_step and the second-order part of order δ_step, both of
   relative order √η, which is why the bound of a measured row is √η. The constants (the second derivative of the
   statistic along the group's direction; the solver's tolerance against its true error) are not derived, and the
   method's own measurements say they are not 1: on rows the redundancy rule finds in the span of the active rows, the
   discrepancy between the evaluation and the value the active rows imply is 10 to 50 times √η (shearmix, 2026-11-07).
   The consequence is in the stop's statement, not in the interval: V − V̂ exceeded U on 4 to 12 of 20 draws per output
   on shearmix while every output was within ε on 20 of 20 draws. Open (2026-11-12): whether the bound stays √η with
   a justified constant or is taken from those measured discrepancies.

One empirical component: a standing term r_l refers to the model at the time of its evaluation, not the current one;
adding rows need not shrink the residual on other leaves, so the two can differ. Measured agreement within a few
percent on the mvt draws at N = 10⁴ (bed/report_look8_stale_mvt{0,3}.txt).

Not assumed: any distribution of the influence; any smoothness beyond what the interpolant uses for its guesses (a
wrong guess costs evaluations, not validity); any relation between leaves. The entry criterion and the batch count
affect cost only.

## 5. Parallel structure

Every evaluation of a round is independent of the others: the M pilot evaluations; the j child-mean evaluations of a
batch; the 2j test evaluations; the J report evaluations. With k workers a round's evaluations take one refit of wall
time. Rounds at N = 10⁴ with k = 16: 16 (shearmix), 17 (mvt), 89 (fp), against 177, 55 and 1297 one at a time. The
overhead between rounds is the refit of ψ̂ (one block factorisation and one solve), the ranking, the quantiles, and
the kernel columns of the new rows; the kernel columns of a batch are themselves independent and belong on the same
workers in the package.

## 6. The quantile

Q = Σ_l a_l χ²₁,l with a_l = r_l / Σ r ≥ 0, Σ a_l = 1. Its distribution function is inverted numerically from the
characteristic function Π_l (1 − 2 i a_l t)^{−1/2} (Imhof's integral); the α-quantile is found by a bracketed root
search started from the Welch–Satterthwaite value. The same routine with the forecast weights w_l gives q̃.

## 7. The passing value δ

The stop is relative: U_j ≤ ε V_lb,j. V_lb is the larger of two quantities known at the time of the test: B_j = Σ_l
p_l m_lj², which is at most V_j because the leaf means are exact, and F̃_j /(1 + √ε)², where F̃_j = ‖ψ̃_j‖²/N; if the
test passes, V_j ≥ V̂_j = 2 P_j − F̃_j and the bound follows. In the bed the influences are scaled so that V = 1 and δ
reads as ε times a number close to 1.

## 8. The reported model and its evaluation

ψ̃ differs from ψ̂ only on the tested, non-determined leaves, where it takes the cross-fitted values
ψ̂_i − (K Xⁱ_l) e_l from the leave-one-out solve. The report evaluation is one refit per output with the weight vector
ψ̃_j /N: a linear functional of the influence, like every other evaluation. Its cost is J.

## 9. Cost accounting

| item | evaluations (each one refit on the full data) |
|---|---|
| start | 4: θ̂ and the three of η_full |
| pilot | M |
| splits | one per split (the smaller child; the sibling is inferred) |
| test | one per leaf that reaches the test, plus two per split after entry |
| repairs | one per repaired leaf |
| report | J |

Package, N = 10⁴, ε = 0.01, M = 300 (2026-11-07): shearmix 676 (median of 20; 300 pilot, about 250 leaves), mvt 727
(median of 5). Bed, M = 1086 and exact influences: shearmix 1376 (402 beyond the pilot), mvt 1134 (167 beyond the
pilot), fp 2458 beyond the pilot. The bootstrap at the same ε is 2/ε² = 20000.

## 10. Implementation notes for the port

- One column path, no kernel stored (audit of 2026-11-06, bed/AUDIT_method.md): each constraint row's column K·a is
  computed from the points once, when the row enters, in fixed tiles of kernel entries. A split's larger child takes its
  column as the exact linear combination of its parent's and sibling's columns (the root's children from the pilot
  cells), so the kernel work per run is a few N² entries (2.8 N² on shearmix, 5.4 N² on fp, about 13 ns per entry). A
  stored kernel (0.75 GB at N = 10⁴, 75 GB at 10⁵) was measured and rejected.
- The kernel's prefactor (2 σ_i σ_j / (σ_i² + σ_j²))^{d/2} is computed by repeated multiplication and one square root for
  odd d; a general power per entry costs six times as much.
- The pilot block of the Gram matrix is factored once; each refit factors only the border and the Schur complement of
  the tree and test rows. The jitter is 10⁻¹⁰ times the largest diagonal entry of the pilot block, fixed at the first
  fit for the whole run. The Gram matrix itself is not stored; reproduction is checked as A ψ̂ − y. No eigen fallback.
- A split's smaller child takes its parent's position in the row order, so the column store and the Gram matrix update
  in place and nothing is moved; ψ̂ at all N points is one matrix–vector product per refit.
- Memory: 0.7 GB peak on shearmix and 1.1 GB on fp at N = 10⁴, most of it the column store (N × m) and the three Gram
  buffers. Fresh allocations of changing size every refit are what had held gigabytes before; buffers are reused.
- The posterior-variance check runs on candidate leaves only (those met by at least n_l rows).
- Refinement step on the Cholesky solve before any eigen fallback.
- The quantizer is sklearn's k-means++ and nothing else; 0.6 to 1.3 s at N = 10⁴. Section 11 states which cells each
  suite result used.
- No pytest; checks are plain scripts.

## 11. Measured

**Package (2026-11-07), real evaluations, one child per split, M = 300 by the budget rule, ε = 0.01, level 0.84, k = 16,
N = 10⁴.** Shearmix, 20 draws: every output within ε on 20 of 20; median evaluations 676, leaves 249, rounds about 20;
Ω below a thousandth of δ, no repair; V − V̂ above U on 4 to 12 draws per output (section 4, item 3). mvt, 5 draws: within
ε on 3 of 5 (misses at 1.3 ε and 2.9 ε on the tail output; the design level is 84 %); median evaluations 727, leaves 220.
Against the bed's certified run at M = 1086 with exact influences: shearmix 1376 evaluations, 179 leaves; mvt 1134, 25.
Wall: 63 s per shearmix draw alone on one thread; 180 to 345 s per draw with 10 to 16 at once on one machine (the refit
itself 3 to 5 times slower under that load).

**Bed.** Suite D (N = 2000) and suite V (N = 10⁴), pilot cells by k-means++ at M = round(N / ln N) (263 and 1086), ε = 0.01,
level 0.84, k = 16; cost in full-refit equivalents, median (90 %). The suite D rows at 222 cells are the earlier
development baseline.

| suite | dataset | draws | valid | cost | rounds |
|---|---|---|---|---|---|
| D (222 cells) | shearmix | 20 | 20/20 | 154 (177) | 18 |
| D (222 cells) | mvt | 5 | 5/5 | 104 (115) | 15 |
| D (222 cells) | fp | 5 | 5/5 | 956 (989) | 42 |
| V (1086 cells) | shearmix | 20 | 20/20 | 408 | 16 |
| V (1086 cells) | mvt | 20 | 20/20 | 166 | 17 |
| V (1086 cells) | fp | 5 | 5/5 | 2489 | 89 |

Suite V rows are the final code (2026-11-06: algebraic determined-leaf test, owned two-means cut, pilot-block
factorisation). The previous code's rows were 399 (439), 167 (173) with one miss at 1.11 ε, and 2458 (2506); the cut
rule changed the partitions, so the miss did not recur, which is a different draw of the same design rate, not a
robustness gain. Those records are kept beside the new ones with the suffix `.standing_20261106`.

The one mvt miss is at R = 1.11 ε (draw 11). The two-valued tail-probability output of mvt, which with 588 pilot cells
hid its error in small mixed leaves and missed on 2 of 5 draws, does not miss with 1086 cells on 20 draws.

Pilot cells from FAISS's own (random) seeding instead of k-means++, same settings otherwise:

| suite | dataset | draws | valid | cost | k-means++ for comparison |
|---|---|---|---|---|---|
| D (263 cells) | shearmix | 20 | 20/20 | 130 (144) | 20/20, 148 |
| D (263 cells) | mvt | 5 | 4/5 | 144 (181) | 5/5, 88 |
| D (263 cells) | fp | 5 | 5/5 | 963 (991) | 5/5, 922 |
| V (1086 cells) | shearmix | 20 | 20/20 | 387 (413) | 20/20, 402 (444) |
| V (1086 cells) | mvt | 20 | 13/20 | 306 (344) | 19/20, 167 (173) |

On mvt the random-start cells cost twice as much and miss on 7 of 20 draws, with errors up to 3.4 ε on the tail
output. Plain squared-distance seeding (one candidate per step, then Lloyd by FAISS or sklearn, 20 iterations or to
convergence) is not enough either: suite V mvt 12/20 valid at 196 (224), fp 5/5 at 2451 (2528), shearmix 20/20 at
394 (438). What the k-means++ cells have and the others lack is the greedy step of the seeding: several candidates per
step drawn in proportion to squared distance, keeping the one that lowers the total squared distance most. On mvt draw
7 the share of the tail output's influence variance left unresolved inside the pilot cells is

| seeding | unresolved share | cells in the 1 % tail | cells mixing both values of the tail output |
|---|---|---|---|
| FAISS random start | 33.6 % | 3.4 % | 4.1 % |
| squared distance, 1 candidate per step | 11.7 to 15.8 % | 7.2 % | 1.7 % |
| squared distance, 3 candidates per step | 5.0 % | | |
| squared distance, 9 candidates per step (sklearn's rule 2 + ln M) | 0.6 % | | |
| sklearn k-means++ | 1.4 % | 9.0 % | 0.2 % |
| scalable k-means++ with a weighted greedy reduction | 1.1 % | | |

The Lloyd iterations change nothing; the seeds decide. The pilot cells are therefore greedily seeded (k-means++ as
sklearn implements it, 0.6 s at N = 10⁴, about a minute at 10⁵, against thousands of pilot evaluations at that N).

Wall time, one process, one BLAS thread, N = 10⁴, k = 16, final code: shearmix 8.4 s (16 rounds), against 32 s for
the sequential loop and 78 s before the performance passes; fp 24 s for 87 rounds and 88 refits of up to 3400 rows.
Peak memory 0.33 GB and 0.93 GB. What the method must hold is the column store, m × N doubles, which grows like
N²/ln N: about 9 GB at N = 10⁵. The refits parallelise through BLAS threads in the package; the block factorisation
matters only when the pilot block dominates the rows (N = 10⁵, M = 8686). Without the cache (N above the memory
budget) fp's columns cost 23 s on one thread and go on the worker pool. The bootstrap's 20000 refits at 0.13 s each: 162 s on 16 workers, 16 s on 80. The method on 16
workers: about 11 s of overhead plus 400 × 0.13 / 16 ≈ 3 s. For statistics whose refit costs under a millisecond
(mvt, fp) the bootstrap's whole run is a few seconds and the method's gain is in evaluations only.
