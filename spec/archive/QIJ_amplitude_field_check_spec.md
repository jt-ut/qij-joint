# The amplitude-field calibration check on draw 45 (planner, 6 October 2026; an offline check, not a build)

## 0. The question

Is a Gaussian process prior with a LEARNED, spatially varying amplitude
calibrated for the loop's bin-mean measurements where the stationary
prior is not? Calibrated means: the prior's predictive distribution for a
bin mean, conditioned on the survey alone, puts the realized bin mean
where it says it will, as judged by the standardized residuals over the
loop's measured bins. Draw 45, the stored survey, the exact-influence
emulation's bins. Under five minutes, four threads, `python3.9`.

This is the premise of the Bayesian-quadrature framing: the posterior
on V is only trustworthy if the prior is calibrated for the measurements
the loop makes. If the stationary prior fails this and the amplitude
field passes, the latent-field model is the thing to develop; if both
fail, the prior family is wrong; if both pass, the amplitude field was
not the problem.

## 1. Data

- Survey: `/Users/jtaylor/Dropbox/Research/QIJ_joint/runs/scratch/shearmix_diag_d45/cloudfil_u_p2_N10000/qij_eps0.01_widthlocal_massfit_nosigma/s00045.prototypes.parquet`:
  1094 rows, `w_0`, `w_1` (prototype coordinates, the DATA coordinates —
  the earlier alignment check found the raw draw coordinates match `bmu`
  at 100%), `p` (cell mass), `h` (local spacing), `I_<output>` (the
  surveyed cell-mean influence, 7 outputs). These are the pilot's
  training data, with forward-difference noise.
- Points: the draw regenerated as `datasets.cloudfil_G_U_P3_v1(10000, 45)`
  (raw coordinates), verified against `points.parquet`'s `bmu` by nearest
  prototype (must agree on ≥ 99% of points; stop otherwise).
- Bins: the emulation (`worked_example/emu.py`, variant A, ε 0.01, run to
  its stop): the 351 measured SMALL children with their member indices
  and exact means U (noise-free). These are the loop's measurements. No
  sequential conditioning on them: every prediction is from the survey
  alone (stated limitation; it isolates the prior).
- Outputs: p2_x, p2_y, p2_log_reff, p2_log_axis_ratio, p2_pa, p2_logit_w,
  p2_log_contrast. Fit each output separately.

## 2. The two priors (same kernel, same trend, same noise model)

For output c, responses y_j = I_jc at w_j:

    y_j = h(w_j)ᵀβ + f(w_j) + e_j,   h(w) = (1, w_0, w_1),   e_j ~ N(0, s² λ)

- (A) STATIONARY: f ~ GP(0, s² k_ℓ), k_ℓ the Matérn-3/2 at one global
  length ℓ in the w units.
- (B) AMPLITUDE FIELD: f(w) = exp(a(w)) f₀(w), f₀ ~ GP(0, s² k_ℓ), so
  K_ij = exp(a_i + a_j) k_ℓ(w_i, w_j). a is piecewise constant on R = 32
  regions of the prototypes (deterministic k-means on w with a fixed
  seed; a check parameter, not a method rule), with a weak ridge
  Σ_r a_r²/(2·3²) that bounds it to about ±9 (a factor of 8000, above
  the survey's own spike of ×100). The noise is NOT amplified: it is the
  forward-difference error, the same everywhere.

Fitting. s² is profiled analytically and β by generalised least squares
inside the marginal likelihood. (A): ℓ on 12 log-spaced values from
0.3·median(h) to 3·range(w), λ on 13 log-spaced values in [1e-4, 1e2]
(the product's own bound); pick the maximum. (B): ℓ fixed at (A)'s
optimum; optimise (a_1..a_32, log λ) by L-BFGS-B with the ANALYTIC
gradient of the profiled log marginal likelihood (∂/∂a_r through
∂K/∂a_r = (1_r 1ᵀ + 1 1_rᵀ) ∘ K; standard trace identities with K⁻¹ and
α = K⁻¹(y − Hβ)); start at a = 0 and (A)'s λ; at most 150 iterations;
report status. One Cholesky of 1094 × 1094 per evaluation. Verify the
gradient against a central finite difference on 5 coordinates before use
(report the discrepancy).

## 3. The predictions and the metric

For each measured bin S (its n points x_i in data coordinates) and
output c, the predictive distribution of the bin mean under each fitted
prior: mean μ_S = (1/n) Σ_i [h(x_i)ᵀβ̂ + k(x_i, W)ᵀ α], variance
σ_S² = (1/n²)[Σ_{i,j∈S} K(x_i, x_j) − sᵀ A⁻¹ s + trend term], the exact
GP formula for a linear functional, with K at the points carrying
exp(a) of each point's nearest-prototype region under (B). The
double sum over the bin's points is formed in chunks, never an n × n
matrix held for the two root children (n ≈ 5000). Floor σ_S² at
1e-12 · s².

Standardized residual z_Sc = (U_Sc − μ_Sc)/σ_Sc.

Report per output, for (A) and (B):

- the DECIDING NUMBER: the energy ratio Σ_S (U_S − μ_S)² / Σ_S σ_S² over
  the 351 bins — 1 for a calibrated prior; ≫ 1 under-states the
  uncertainty, ≪ 1 over-states it;
- RMS(z), median |z|, fraction |z| > 2 (0.046 if calibrated) and > 3
  (0.003), overall and by bin-size class n = 1, 2, 3–4, 5–8, 9–16,
  17–32, 33–64, ≥ 65, and by emulation round 1–10;
- the fitted ℓ, λ, s², the log marginal likelihood of (A) and (B), and
  for (B) the range of â_r and, per region, â_r beside log of the RMS of
  the region's survey responses (whether the learned amplitude tracks
  the measured local magnitude).

Plus the alignment rate, the gradient check, wall time per fit,
`python3.9 --version`.

## 4. Explosion cases, stated

- λ at a bound (the product has λ_pa = 100, the upper bound): under (A)
  the survey is then "all noise", μ_S collapses to the trend and σ_S² to
  ≈ s²λ/n for big bins, so z is enormous for core bins — that is the
  failure the check is meant to show, not a defect. Report λ bounds hit.
- exp(a) ≤ e⁹ by the ridge; K stays positive definite because it is a
  Hadamard product of a PD kernel with a rank-one positive matrix. Add
  the jitter 1e-8·trace(K)/M to the diagonal before every Cholesky.
- σ_S² can be negative by rounding for large bins; the floor handles it;
  report how many bins hit the floor.
- SANITY CAP: if any fit's Cholesky fails after jitter, or any σ_S is
  non-finite, stop and report which.

## 5. Worked example (the magnitudes the spec was checked against)

Position angle. Survey responses: sd 8.6, range −96 to +135, the spike
in ~1% of cells. The stored stationary fit chose λ = 100 (bound) — it
treated the spike as noise, which is why the pilot's pa is flat and its
within read is 0.001 of the truth. Under (A) here, the same will happen:
for a core bin of 2 points with U ≈ ±100 in survey units, μ_S ≈ trend
≈ 0, σ_S² ≈ s²(λ/2 + k-terms) with s² ≈ Var(y) ≈ 74 → σ_S ≈ 60 at λ=100
— so z ≈ 1.5: the stationary prior is NOT obviously miscalibrated on
single core bins when λ is at the bound; it is miscalibrated on the
BULK bins (U ≈ 0.1, σ_S ≈ 60/√n: z ≈ 0) and on the energy ratio, since
Σσ_S² is dominated by λ/n terms that are far too large for the bulk and
too small for the core. The energy ratio by size class is the number
that separates the two priors: (A) ≪ 1 on big bins and ≫ 1 on small
core bins; (B) near 1 on both if the amplitude field is learned.
x (smooth): both priors should be near 1 on every class; if (B) is
worse than (A) on x, the region count or the ridge is the cause and the
report's â_r table will show it.

## 6. Output

`amplitude_field_check.py` and `report.txt` in this session's scratchpad
under `amplitude_field_check/`. Plain text, the tables of section 3,
verbatim, no interpretation.
