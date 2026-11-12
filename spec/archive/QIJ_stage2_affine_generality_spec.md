# Stage 2 of QIJ_qijt_affine_plan.md: the affine field and its lack-of-fit error bar on the Fundamental Plane and the IMF (planner, 6 October 2026; an offline check, not a build)

## 0. The question and the gate

The Stage 1 computation (QIJ_stage1_affine_refined_spec.md), unchanged,
on draw 0 of the Fundamental Plane and the IMF: does the affine field,
as the pilot and with no factor, read the final leaves' within-bin term
inside the tolerance, and does the lack-of-fit error bar at the
neighbour scale (L1; L2 was rejected in Stage 1 and is not run)
calibrate the bins within a factor of three? Gate: a failure on either
of these smooth cases means the model is wrong and the plan stops.
The multivariate t waits for the coder to reproduce its quantizer
coordinates (85% bmu agreement with the regenerated whitened draw).

## 1. The cases (one survey size each; no refined products exist)

As QIJ_stage1_generality_check_spec.md section 1: fp_all_N2000 (d = 3,
outputs a, b, c, scatter, 371 prototypes) and imf_all_N2000 (d = 1,
outputs m_c, sigma, x, 321 prototypes), draw s00000, N = 2000,
`datasets.fp(2000, 0)` and `datasets.imf(2000, 0)` reshaped to (N, 1),
raw coordinates = quantizer coordinates (bmu agreement ≥ 99% required).
Baseline per output = the stored run's `V_tot_hat_<output>` from its
scalar record against the exact V_ij from the ij psi file; L_max = M_X.
No `h` column: not needed (the affine field has no width).

## 2. The computation

Exactly Stage 1's at one survey size per case: CONN; g_j by least
squares on CONN neighbours (two hops where fewer than d); ν_c = the
smallest edge-length-quintile semivariance; λ_jc = max(mean_k r_jkc² −
2ν_c, 0); the emulator (variant A, ε 0.01, exact measurements) with the
affine field's prediction as the pilot; the measured bins scored with
the noise-only and the L1 variance; the final leaves' corrected within
read, Σ T̂/Σ T overall and by size class, the implied error/ε beside the
baseline, the L1 leaf-sum bar and whether |Σ T̂ − Σ T| ≤ 2 sd; the flag
counts and the within share they hold. Report per case and output; the
alignment rates; two-hop/pinv counts; the identity check; L of the run
and of the stored run; wall; `python3.9 --version`.

## 3. Worked example and explosion cases

IMF, 1-D: a prototype has at most two connectivity neighbours (the
chain), so the gradient is a one-dimensional slope from one or two
differences and λ is the residual of a line through at most three
points — small where the influence is smooth in log mass, which is
everywhere but the ends of the range. Expect the within read inside ε
(the earlier generality check with the GP field read x at 0.96 and the
other two outputs at 3.8 and 8.9 of their within; the affine read has
no amplitude to misjudge) and L1 near 1 on the bins. Explosion: a field
with ONE neighbour gives an exact fit (λ = 0 by construction) — count
those fields and report the share of the within they hold; a field with
no CONN neighbour after two hops: report.
Fundamental Plane, 3-D: the influence is a regression's leverage times
residual, smooth over the whole space; the variogram's shortest-lag
share was 2–9% (the noise). Expect λ small everywhere, few or no flagged
fields, the within read inside ε, the bins calibrated by ν alone or
with a small L1 term. The earlier emulation on FP ran to the cap
L = 371 = M_X with the GP pilot; with the affine pilot report whether it
caps and where it stops. Explosion: none expected; the sanity cap on
non-finite σ stands.

## 4. Output

`stage2_affine_generality.py` and `report.txt` in this session's
scratchpad under `stage2_affine_generality/`. Plain text, one block per
case, a two-line summary (case; error/ε range over outputs; L1 bin ratio
range; flagged union; L), verbatim, no interpretation.
