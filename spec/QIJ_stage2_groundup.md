# The second stage from the ground up (planner, 27 September 2026)

Terms. A **receptive field** is the set of data points whose nearest
prototype is a given prototype of the first-stage quantizer (the 𝒳-VQ);
there are 1 094 of them here, holding 1–26 points each, median 9. The
**prototype influence** I_j is the derivative of the estimator with respect
to the total weight of receptive field j; its target is the mean of the
true influence over that receptive field's points, called below the
**true receptive-field mean**. **Within-receptive-field variance** is the
variance of the true influence among the points of one receptive field.
**Share** means Σ_k p_k mean_k(ψ_c)² / ((1/N) Σ_i ψ_ic²): the fraction of
the influence variance a partition captures; V_btw/V_ij is the same number.

Scope: why QIJ recovers 57–93% of the influence variance on the demo
estimand (cloudfil, P2, N = 10 000, ε = 0.01) and what would have to change.
Everything below is measured on the two rehearsal draws (s = 0, 1) from the
coder's diagnostic pickle, the stored `ij` product (the analytic influence ψ,
N × 66), and the stored oracle fit θ̂. No estimator was built; all runs are
offline, in the planner's scratch (`stage2_review/`).

## 1. What the second stage is, as built

`core/ivq.py`: 17 bins by 1-D k-means on ψ̂₀_c; each bin measured by a
central stencil on the full data (2 evaluations); V_btw = (1/N) Σ p_k U_k².
`core/refine.py`: every open leaf carries a proposed split (two-means on
ψ̂₀ inside the leaf, or the CADJ adjacency split) priced by the model:
level gain from ψ̂₀'s bin means, adjacency gain from the GP posterior
within-bin variance v_k; the kind rule is Var_k(ψ̂₀) > v_k. The leaf with
the largest predicted gain is split, one evaluation on the smaller child,
the larger child by mass balance, until the best gain falls below
τ = ε V_btw/L (two strikes) or the guard 1 + M_𝒳 binds. `core/rounds.py`
batches the qualifying leaves of all outputs per round. A15's measured
trigger flags initial bins whose measured U_k disagrees with the model and
splits those lineages geometrically. Nothing measured during refinement
ever changes ψ̂₀, σ, or v_k: the model is frozen at the end of stage 1.

## 2. What is right

**The stencils are exact.** On both draws, for every output, the final
bins' measured U_k regress on the true bin means of ψ with slope 1.000 and
r² 1.000. V_btw equals Σ p_k mean_k(ψ)² for the partition the code built.
Every shortfall is the partition, none is the measurement.

**An exact offline simulator exists.** Replacing T by its linearization
T(ω) = θ̂ + (ω − 1)Ψ/N and running `run_refinement_rounds` unchanged
reproduces the stored V_btw (three decimals) and every per-output
evaluation count on both draws. Any stage-2 policy can now be evaluated
at zero cost.

## 3. The causal chain, measured

### 3.1 The prototype survey is biased, and the bias is not local

The survey's finite differences equal the analytic influence of the
estimator FITTED TO THE SURVEY ROWS (slope 0.995–0.999, r² ≥ 0.993, all
outputs): the code computes exactly what it is defined to compute, the
step is clean (step ratios 1.98–2.30), the continuation converges. But
the quantized fit's influence is not the true receptive-field mean:

| draw 0 | slope of true receptive-field mean on I_j | r² | share of V, I_j | share, true receptive-field means |
|---|---|---|---|---|
| p2_x | 1.29 | 0.93 | 0.51 | 0.92 |
| p2_y | 1.31 | 0.93 | 0.49 | 0.91 |
| p2_log_reff | 1.50 | 0.91 | 0.36 | 0.89 |
| p2_log_axis_ratio | 1.63 | 0.94 | 0.31 | 0.88 |
| p2_pa | 0.65 | 0.20 | 0.41 | 0.86 |
| p2_logit_w | 1.26 | 0.85 | 0.49 | 0.92 |
| p2_log_contrast | 1.07 | 0.84 | 0.66 | 0.89 |

Draw 1 is worse: slopes 0.89–2.13, r² 0.57–0.91, I_j's share 0.11–0.94
against true 0.64–0.87.

Decomposition (draw 0, share of V inside P2's receptive fields): placing
the survey rows at simplex vertices instead of the points costs
0.001–0.004; the quantized fit's information matrix costs 0.03–0.14; its
parameter shift costs 0.02–0.05 (0.30 for pa). Representing P2's own
receptive fields, or every receptive field within 0.4 of P2's centre (89
receptive fields, 3 829 rows), by their native points leaves the slopes
at 1.14–1.30 and pa's r² at 0.19. Only the full data (slope 1.000) repairs
it. The mixture's likelihood is shared: a surrogate of 3 269 rows fits a
105-point core sitting on a filament bead over the cloud differently from
the data, wherever the far rows are placed. Assumption A2 fails for this
estimand, and no local change to the quantizer's representation restores
it. The A9 acceptance ("corr above 0.9") passed on draw 0 because
correlation is blind to a scale factor.

### 3.2 The influence model then loses more

The GP's fitted global width is 0.098 in whitened units on both draws;
P2's semi-axes are 0.05 × 0.025 in data units. Initial 17-bin share on:

| | ψ̂₀ (built: survey, global) | survey values, no GP | ψ̂₀ (survey, local) | ψ̂₀ (exact means, local) | oracle ψ |
|---|---|---|---|---|---|
| draw 0 | 0.64–0.90 (pa 0.008) | 0.78–0.86 (pa 0.32) | 0.87–0.92 (pa 0.30) | 0.92–0.98 | 0.99 |
| draw 1 | 0.11–0.58 | 0.47–0.74 | 0.49–0.86 (pa 0.32) | 0.53–0.92 | 0.98 |

The global-width GP delivers a worse partition than the raw survey
values it was fitted to. `gpwidth=local` is the study's setting (A11) but
has never been run on cloudfil: the rehearsal and the diagnostics were
global. The posterior v_k is 0.004–0.5 of the true within-bin variance as
built, 0.2–0.3 with local width on the survey, 0.6–1.9 with local width
on exact receptive-field means.

### 3.3 The floor set by the receptive fields

The variance ψ has INSIDE receptive fields is 8–15% of V on draw 0 and
13–36% on draw 1, and 95–99% of it sits in P2's own receptive fields
(P2's 104 points occupy 15–35 receptive fields). No partition built from
receptive-field-level information can pass 0.85–0.92 (draw 0) or
0.64–0.87 (draw 1) without splitting receptive fields. One split of a P2
receptive field removes 72–82% of its variance with the oracle direction,
20–75% with two-means on ψ̂₀, 28–63% along the receptive field's principal
axis, 30–66% along the neighbouring prototypes' gradient.

### 3.4 What the refinement does with all this

Priced by a model that is scaled wrong, scattered, and blind inside P2,
the queue spends 2–13% of its evaluations on bins holding P2's points
(the coder's report). The exact simulator, stage-2 code unchanged, with
stage-1 inputs swapped:

| inputs → policy | draw 0: V_btw/V_ij, evaluations | draw 1 |
|---|---|---|
| survey, global (as built) | 0.76–0.92, 2 140 | 0.57–0.93, 3 188 |
| survey, local, gain | 0.57–0.94, 598 | 0.62–0.91, 1 150 |
| survey, local, measured trigger | 0.79–0.96, 839 | 0.74–0.97, 1 280 |
| exact receptive-field means, local, gain | 0.94–0.99, 301 | 0.84–0.94, 663 |
| exact receptive-field means, local, measured trigger | 0.95–0.99, 309 | 0.84–0.95, 809 |
| oracle 17 bins on ψ | 0.99, 34 | 0.98, 34 |

Queue and rounds differ by ≤ 0.02. Given exact receptive-field means and
local width the as-built refinement is within a few percent of ε on draw
0 at one seventh of the cost; on draw 1 it stalls at 0.84–0.95 because
the within-receptive-field variance (13–36%) must be removed by splits the
model cannot aim, and the model never learns from the stencils it pays for.

## 4. The structural conclusion

1. Stage 2's arithmetic and measurements are correct. Its policy is a
   greedy search steered entirely by a frozen stage-1 model. On this
   estimand that model is wrong in scale (survey), smoothed past the
   structure (global width), and below the resolution the structure
   needs (receptive fields larger than P2's variation scale). Every A15
   round patched the steering; none touched the model, which is why they
   moved the numbers by hundredths.
2. The quantized survey cannot be repaired locally for this estimand. The
   demo's cost story rests on it.
3. Even with exact receptive-field means the receptive-field partition
   leaves up to a third of the variance inside P2's receptive fields on
   draw 1. Reaching ε there needs either finer receptive fields inside P2
   (one evaluation per new receptive field) or a within-P2 model that
   improves as stencils are measured.

## 5. Repairs, ranked by what is measured

A. **Local width.** Zero build, one switch, already the study setting.
   Alone: 0.57–0.94 / 0.62–0.91. Necessary, not sufficient.

B. **Survey on the full data.** Each prototype influence as a forward
   difference of T on all N points with respect to its receptive field's
   total weight (a warm continuation, the same call the stencils make):
   1 094 evaluations, one pool batch, exact receptive-field means (slope
   1.000). With A and the code as built: 0.94–0.99 at 301 evaluations on
   draw 0, 0.84–0.95 at 663–809 on draw 1. Total 1 400–1 900 full-data
   evaluations against 2 140–3 188 full-data plus 1 094 quantized today.
   Cheaper and closer, and it changes the method's account of itself:
   stage 1 stops being cheap.

C. **B plus adaptive receptive-field splitting.** Split the receptive
   fields whose neighbouring prototypes' measured gradient times the
   receptive field's extent predicts within-receptive-field variance above
   the tolerance, one evaluation per split, before any bins are formed.
   With exact receptive-field means the bins add no information: V_btw =
   Σ_j p_j I_j² at receptive-field level needs no I-VQ, no stencils, no
   refinement queue. The oracle bound (17 bins on ψ, 34 evaluations) says
   tens of evaluations should suffice; the measured-criterion cost is NOT
   yet measured (the script failed silently and was not chased).

D. **Let the model learn from the stencils.** A bin mean is a linear
   functional of ψ; a GP conditions on it exactly. Re-predicting ψ̂ inside
   P2 after each round would aim the next splits. Untested; heavier to
   build than C, and C may make it unnecessary.

Not proposed: widening ε, changing the dataset, or another steering rule
in `refine.py`.
