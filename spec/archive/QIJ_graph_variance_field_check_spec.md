# The graph variance-field check on draw 45 (planner, 6 October 2026; an offline check, not a build)

## 0. The question

The amplitude-field check (spec QIJ_amplitude_field_check_spec.md, run
6 Oct) showed that a prior whose variance varies with position is
calibrated for the loop's bin influences and the stationary prior is
not, but its one global correlation length collapsed to the grid floor,
so the smooth part of the field carried nothing between prototypes. This
check makes three changes and asks whether the smooth part now carries
information: (1) the correlation length follows the prototype spacing
(the product's local width); (2) the log-variance is one value per
prototype, smooth along the connectivity graph; (3) that field is shared
across the seven outputs. Same draw, same data, same metric.

## 1. Data (as the amplitude-field check; copy its script)

Prototypes w_j, p_j, h_j (local spacing), I_jc for 7 outputs; the draw's
points X (raw coordinates, bmu-verified); the emulation's 351 measured
bins (small children) with members and exact bin influences U.
New: the connectivity graph CONN from the points: for every point its
nearest and second-nearest prototype; an undirected edge between every
such pair that occurs at least once; report the edge count and the
degree range.

## 2. The kernel: Matérn-3/2 with a length per prototype

ℓ_j = c · h_j. For two locations with lengths ℓ_i, ℓ_j (a point takes
the length of its nearest prototype), the Paciorek–Schervish form in 2-D
with isotropic lengths:

    k(x_i, x_j) = [2 ℓ_i ℓ_j / (ℓ_i² + ℓ_j²)] · (1 + √3 r_ij) exp(−√3 r_ij),
    r_ij² = 2 ‖x_i − x_j‖² / (ℓ_i² + ℓ_j²)

This is positive definite for any positive lengths (it is the
product's `_matern32_nonstationary` up to notation; reuse that function
from `qij_joint.core.influence_model` if its signature fits, read-only,
else implement the formula above). One factor c for everything.

## 3. The two models

Responses y_jc = I_jc; trend h(w) = (1, w_0, w_1); noise e ~ N(0, s_c² λ_c).

- (C) LOCAL WIDTH, STATIONARY AMPLITUDE: f_c ~ GP(0, s_c² k), per output.
  Fit: c on 8 log-spaced values in [0.3, 10], λ_c on the 13-value grid
  in [1e-4, 1e2]; s_c² profiled, β_c by GLS, REML form as before. c is
  chosen ONCE by the SUM over outputs of the profiled log marginal
  likelihood at each output's best λ_c, and then fixed for (D).
- (D) LOCAL WIDTH + GRAPH VARIANCE FIELD, SHARED: K_c,ij =
  exp(a_i + a_j + 2 b_c) k(w_i, w_j), with a_j one value per prototype
  shared by all outputs and b_c a per-output offset (b_1 = 0 fixed; s_c²
  still profiled, so b_c is identified relative to the kernel's
  shape, not its scale — report it). Penalty on a: ½ Σ_{(i,j)∈CONN}
  (a_i − a_j)² / τ² + ½ Σ_j a_j² / 9, with τ = 1 (a check parameter: a
  log-variance change of order one per graph edge; stated, not
  searched). Objective: Σ_c (profiled REML log ML)_c − penalty.
  Unknowns: a (1094), b_2..b_7, log λ_c (7). Analytic gradient (the same
  trace identities as the amplitude check, now summed over outputs;
  ∂K_c/∂a_j = (e_j 1ᵀ + 1 e_jᵀ) ∘ K_c; ∂K_c/∂b_c = 2 K_c). L-BFGS-B,
  bounds a ∈ [−12, 12], b ∈ [−12, 12], log λ ∈ [log 1e-4, log 1e2],
  at most 150 iterations; start a = 0, b = 0, λ_c at (C)'s optimum.
  Verify the gradient on 5 coordinates (two a_j, one b, two log λ)
  against central differences before use; report the discrepancy.

## 4. Predictions and metric (as the amplitude check, section 3)

For each measured bin and output, μ_S and σ_S² under (C) and (D), the
exact GP formulas for a linear functional, with the local-width kernel
at the bin's points (each point's ℓ from its nearest prototype, its a
from its nearest prototype under (D)); double sums chunked. Floor σ_S²
at 1e-12 · s_c².

Report per output for (C) and (D): the ratio Σ(U − μ)²/Σσ² overall, by
bin-size class (n = 1, 2, 3–4, 5–8, 9–16, 17–32, 33–64, ≥ 65) and by
round; RMS z, median |z|, frac |z| > 2, frac |z| > 3 overall and by
size class; the chosen c and each output's λ_c (flag bounds), s_c²,
log ML of (C) and (D) per output; under (D) the range of â over
prototypes, b̂_c, and a table of the 20 prototypes with the largest â
(j, w, h, â, and log |I_jc| for the output with the largest |I|) beside
the 20 with the smallest. Also reprint, for reference, the amplitude
check's overall ratios for (A) and (B) from its report.txt (read the
file; do not recompute).

## 5. Explosion cases, stated

- The local-width kernel's prefactor is ≤ 1 and the form is PD for any
  positive lengths; jitter 1e-8·trace/M before every Cholesky.
- exp(a_i + a_j + 2b) ≤ e^{48} at the bounds in principle; the graph
  penalty and ridge keep a within a few units in practice (the
  amplitude check's range was [−3, 6.3]); report the range. If any
  Cholesky fails after jitter or any σ_S is non-finite: SANITY CAP,
  stop and report which.
- c at a grid bound: report it; if c sits at 0.3 the length is again
  trying to go to the floor and the smooth part is not carrying
  information, which is a finding, not a defect.
- Compute budget: (C) grid 8 × 13 × 7 Choleskys of 1094² ≈ 10 s; (D)
  ≤ 150 evaluations × 7 Choleskys ≈ 60–90 s; bin predictions ≈ 10 s.
  Hard cap 5 minutes; fit pa first, then x, then the rest, and report
  what finished if the cap trips.

## 6. Worked example

Output x, the dipole across the core: lobes about 0.1 pc apart, the
core's prototypes spaced about 0.01–0.03 (h min 0.0097, median 0.106
over all prototypes; the core is dense). With c ≈ 2 the length in the
core is 0.02–0.06, below the lobe separation, so the GP interpolates the
dipole from the core's prototypes; outside the core h is 0.1–1.5 and
the length follows, so the field is smooth where the influence is flat.
Under (C) the amplitude is one number per output, so the uncertainty
inside the core (influence ±5) and outside (±0.05) is the same, and the
ratio of realized to predicted error should again be ≫ 1 on the one-
and two-point bins (the amplitude check's stationary ratio there was
9–10 on x). Under (D) the core's prototypes carry â of several units
(the amplitude check's region with the large influences had â ≈ 6 on x)
and the ratio on those bins should return toward 1, while the largest
bins' ratio should move up from the amplitude check's 0.16 if the
interpolation now contributes, because a correlated field averaged over
a large bin has a smaller predictive variance than independent values.

What would count as success: (D) with ratios near 1 on every size class
AND c away from the lower bound. What would count as the smooth part
still failing: (D) matching (B) with c at 0.3.

## 7. Output

`graph_variance_field_check.py` and `report.txt` in this session's
scratchpad under `graph_variance_field_check/`. Plain text, verbatim,
no interpretation.

---

## 8. Revision 2 (6 Oct, after the first run): the noise floor, and no per-output offset

The first run (scratchpad graph_variance_field_check/report.txt) showed
the smooth part carrying information (c = 2.2, interior) and the 1–2-point
bins calibrated within ×3.6 on every output, but the largest bins badly
mis-calibrated (76–87% of residuals beyond 3σ): the per-output offset b_c
was redundant with the profiled scale s_c² and traded against λ_c, the
optimiser pushed b to its bound and the noise to zero, and the model then
interpolated the forward-difference error of the prototype influences as
signal. Two changes:

1. **No b_c.** Model (D) is K_c = s_c² · exp(a_i + a_j) k(w_i, w_j) + ν_c I
   with a shared across outputs as before, and s_c² and ν_c (the ABSOLUTE
   noise variance of output c's prototype influences) optimised
   explicitly, NOT profiled: unknowns a (1094), log s_c² (7), log ν_c (7).
   REML log marginal likelihood and its analytic gradient as before with
   ∂A_c/∂log s_c² = s_c² exp(a_i + a_j) k and ∂A_c/∂log ν_c = ν_c I.
2. **The declared noise floor.** ν_c ≥ n_c² = 2 η_Q² θ_c² · median_j(1/t_j²),
   the product's own declared-noise level (influence_model.py docstring;
   glossary "declared noise floor"), with η_Q = 4.318328622856778e-06 from
   the run's scalar record, t_j = δ_f p_j/(1 − p_j), δ_f = 2√η_Q, p_j the
   prototype masses (median_j(1/t_j²) = 7.134e10, so n_c² = 2.661 · θ_c²),
   and θ_c the draw's estimate of output c, taken as the MEDIAN of
   `theta_p2_<output>` over the bootstrap replicates in
   `/Users/jtaylor/Dropbox/Research/QIJ_joint/runs/shearmix4/cloudfil_u_p2_N10000/boot/s00045.replicates.parquet`
   (the product would use the prototype-stage value; for a floor the
   difference is immaterial; state it). Bounds: log ν_c ∈ [log n_c², log(100·Var(y_c))],
   log s_c² ∈ [log(1e-6·Var(y_c)), log(1e6·Var(y_c))], a ∈ [−12, 12].
   REPORT n_c² beside Var(y_c) for every output: where n_c² exceeds
   Var(y_c) the declared accuracy says the prototype influences of that
   output are noise-dominated outside the core, which is a finding about
   the prototype stage, not about the prior (the stored product's
   λ_pa = 100 at its bound is this).

(C) under revision 2: the c choice is unchanged (the profiled form of
the first run); then at the chosen c, (s_c², ν_c) per output on a 13 × 13
log grid with the floor applied (≈ 1200 Choleskys, ≈ 25 s).
(D): L-BFGS-B, at most 300 function evaluations (the first run hit 150
iterations); start a = 0, (s_c², ν_c) at (C)'s refit (ν_c at the floor
if (C)'s optimum is below it). Gradient check on 5 coordinates (two a_j,
one log s², two log ν) before use. Everything else as sections 1–7.
Hard cap 5 minutes.

Worked example of the floor: a prototype's forward-difference step is
t_j ≈ 4.2e-6 at p_j ≈ 1e-3, so the declared noise on its influence is
√2 η_Q |θ_c| / t_j ≈ 1.45 |θ_c|. For an output whose estimate is of
order tens, the floor is of order 10³ against a response variance of 74
outside the core: the prototype influences are then noise-dominated
except in the core, and a calibrated prior must report that as large
predictive variance on bins outside the core — the largest bins' ratio
should then fall toward 1 from the first run's values, with z no longer
in the tens. For an output whose estimate is near zero (x, y: the core's
position near the origin) the floor is near zero and ν_c is free, as
the product itself has it.

Output: `graph_variance_field_check_v2.py` and `report_v2.txt` in the
same directory.
