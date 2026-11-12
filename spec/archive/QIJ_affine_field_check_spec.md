# The affine receptive-field model: calibration check on draw 45 (planner, 6 October 2026; an offline check, not a build)

## 0. The question

The data look (scratchpad data_look/look.txt) showed that the gradient
of the influence fitted from a prototype's connectivity neighbours,
pushed through the receptive field's own covariance, predicts the
within-receptive-field variance of the exact influence at 1.0–2.4 on
the fields that hold 99% of it on draw 45 (position angle 2.0) and at
0.97–1.17 on the Fundamental Plane, with nothing fitted. This check
asks whether the influence model built on that, a piecewise-affine
field over the receptive fields, is calibrated for the loop's bin
influences and reads the final leaves' within-bin term, on draw 45,
beside the Gaussian-process results (stationary prior 8–57; variance
field 0.75–3.7 on the bins; the pilot's within read 0.001–0.97 on the
leaves). Products only; no truth used in any prediction; the truth
enters only as the score.

## 1. Data (all in hand)

As the data look: prototypes w_j, p_j, I_jc (draw 45, 1094 prototypes,
7 outputs); the draw X = datasets.cloudfil_G_U_P3_v1(10000, 45) (raw
coordinates match the prototypes; bmu agreement 100%); the exact
influence psi (score only); the connectivity graph CONN from nearest and
second-nearest prototypes (2243 edges); the receptive-field statistics
n_j, x̄_j, Σ_j (count, mean, covariance of the points nearest prototype
j); the emulation's 351 measured bins (small children, members, exact
bin influences U) and its final leaves with exact within T (score only).
Start from scratchpad data_look/look.py (the graph, the gradients, Σ_j
are already built there) and worked_example/emu.py.

## 2. The model

For a point x in receptive field j, output c:

    ψ̂_c(x) = I_jc + g_jcᵀ (x − x̄_j)

- Mean: the prototype influence I_jc (a cell mean; the affine model's
  mean over the field is I_jc because the correction term averages to
  zero about x̄_j).
- Gradient g_jc: ordinary least squares of (I_kc − I_jc) on (w_k − w_j)
  over j's CONN neighbours k. A prototype with fewer than d neighbours
  uses its neighbours' neighbours as well (report how many).
- Noise ν_c: the smallest of the five edge-length-quintile semivariances
  ½·mean(I_i − I_j)² over CONN edges (the nugget as the data show it: the
  shortest-lag quintile on a smooth field, the longest-lag quintile on
  draw 45 where the outskirts carry zero influence at both ends). No η.
- Uncertainty: Var(I_jc) = ν_c; Cov(g_jc) = 2 ν_c (A_jᵀ A_j)⁻¹ with A_j
  the neighbour-offset matrix (each difference has noise 2ν; the common
  I_j term's correlation across differences is ignored, stated).

Predictive distribution of a bin influence for bin S (n points,
n_Sj of them in field j, s_Sj = Σ_{i∈S∩j} (x_i − x̄_j)):

    μ_S  = (1/n) Σ_j [ n_Sj I_j + g_jᵀ s_Sj ]
    σ_S² = (1/n²) Σ_j [ n_Sj² ν + s_Sjᵀ Cov(g_j) s_Sj ]

Predicted within of a final leaf ℓ (n ≥ 2), per point, two forms:

    raw:        τ̂_ℓ = Var_{i∈ℓ}( ψ̂(x_i) )
    corrected:  τ̂_ℓ − (1/n) Σ_i (x_i − x̄_ℓ,j(i))ᵀ Cov(g_{j(i)}) (x_i − x̄_ℓ,j(i)),  floored at 0

(the raw within of the fitted field over-states the true within by the
gradient-noise term; the corrected form subtracts its expectation). Its
predictive sd: the square root of twice the subtracted term plus
ν(1 − Σ_j (n_ℓj/n)²), stated as an approximation. Singletons are 0.

## 3. Scores

(a) The 351 bins, as the Gaussian-process checks: ratio Σ(U − μ)²/Σσ²
overall, by bin-size class (1, 2, 3–4, 5–8, 9–16, 17–32, 33–64, ≥ 65)
and by round; RMS z, median |z|, frac |z| > 2, frac |z| > 3 overall and
by class. Print beside each output the stationary prior's and the graph
field's overall ratios from the earlier reports (amplitude_field_check/
report.txt (A); graph_variance_field_check/report_v2.txt (D)).

(b) The final leaves: Σ T̂/Σ T overall and by size class for the raw and
the corrected forms; the implied variance error (V_btw + Σ T̂ − V_ij)/
V_ij/ε beside the BASELINE κ̂·ΣW read (the emulator's own stats); the
predictive relative spread sd/Σ T̂ of the leaf sum (leaves independent).

Plus: ν_c beside Var(I_c) and the five quintile semivariances per
output; the count of prototypes needing two-hop neighbours; the
distribution of the gradient-noise term over leaves; identity check
Σ T + Σ D = V_ij; wall time; `python3.9 --version`.

## 4. Explosion cases

- ν_c ≈ 0 on an output whose longest-lag edges are exactly zero: σ_S² is
  then only the gradient term and the bins' ratio will be > 1 where the
  affine model's second-order residual is the real error — that is the
  finding the check is for (the model's uncertainty omits the
  second-order residual), not a defect. Report ν_c.
- A field with fewer than d+1 points has a singular Σ_j; it contributes
  no within (singletons) or a rank-deficient one (two points: a line) —
  Var over the points handles it; no inversion of Σ_j is needed anywhere.
- (A_jᵀA_j) singular (collinear neighbours): use `np.linalg.pinv`; count.
- No optimisation, no grid, no cap to trip; compute is seconds.

## 5. Worked example (from the data look)

Two-point leaf 668 in the core: its two points lie in one or two core
receptive fields; the within is (g_jᵀ(x_1 − x_2))²/4 per point. The
data look's sum over the core fields of exactly this quantity against
the exact within gave 1.0–2.4 per output, so Σ T̂/Σ T on the n = 2 class
should land near that band, with pa at about 2, against today's 0.001.
Bins: a core bin of n points has μ_S from the core fields' means and
gradients; its error is the affine model's second-order residual, which
the per-field scatter of ×5–20 in the data look says is large per
field; the ratio on small bins may exceed 1 by that residual — the
number says how much a second-order term or a directional-derivative
measurement would have to carry.

## 6. Output

`affine_field_check.py` and `report.txt` in this session's scratchpad
under `affine_field_check/`. Plain text, verbatim, no interpretation.
