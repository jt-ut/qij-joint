# Stage 1 of QIJ_qijt_affine_plan.md: the affine field's within read and its lack-of-fit error bar on the refined survey of draw 45 (planner, 6 October 2026; an offline check, not a build)

## 0. The questions (three, one computation)

1. On the 2188-prototype survey of draw 45, does the affine field read
   the final leaves' within-bin term inside the tolerance on every
   output, position angle included? (At 1094 prototypes it was inside on
   six and 0.11 of the truth on pa.)
2. Does the lack-of-fit term, pushed into the bins and the leaves, cover
   the realized error: a bin ratio near 1 where the noise-only bar gave
   12–516, and the leaves' exact within inside the error bar?
3. How many fields does the lack-of-fit indicator flag at 1094 and at
   2188 prototypes, and do the flagged fields hold the within? (The
   flagged count at 1094 is the local refinement's evaluation cost.)

## 1. Data (products in hand)

- 1094-prototype survey: as QIJ_affine_field_check_spec.md.
- 2188-prototype survey: `/Users/jtaylor/Dropbox/Research/QIJ_joint/runs/scratch/shearmix_diag_d45_mx2/cloudfil_u_p2_N10000/qij_eps0.01_widthlocal_massfit_nosigma_MX2188/s00045.{prototypes,points,bins}.parquet` and the scalar record (M_X = L_max = 2188). Same columns as the 1094 files; `bmu` at 2188.
- The draw X (raw coordinates; verify bmu agreement ≥ 99% at both sizes), the exact influence (score only), CONN at each size from the points.
- Outputs: the seven p2 outputs.

## 2. The model, as the affine check, plus the lack of fit

Per field j, output c: mean I_jc; gradient g_jc by least squares on CONN
neighbours; noise ν_c = the smallest edge-length-quintile semivariance.
NEW — the lack of fit:

    r_jkc = I_kc − I_jc − g_jcᵀ (w_k − w_j)      over j's neighbours k
    λ_jc  = max( mean_k r_jkc² − 2 ν_c , 0 )      the unmodelled variance at the neighbour scale

Two treatments of how λ enters a prediction, both reported (this is
the lack of fit's first test; the calibration number chooses):

- (L1) at the neighbour scale: a point in field j carries an unmodelled
  deviation of variance λ_jc, perfectly correlated within the field and
  independent across fields.
- (L2) scaled to the field: the same with λ_jc · (tr Σ_j / mean_k ‖w_k − w_j‖²)²,
  the quadratic scaling of a second-order deviation from the neighbour
  distance to the field's own extent (a stated modelling choice; it is
  smaller than L1 by the fourth power of the ratio of field radius to
  neighbour distance).

Bin prediction: μ_S as before; σ_S² = the noise-only form of the affine
check + Σ_j (n_Sj/n)² λ_jc (L1 or L2).
Leaf within: τ̂_ℓ as before (corrected form); its error bar
sd² = Σ_j (n_ℓj/n_ℓ) λ_jc (L1 or L2) — the unmodelled within per point,
mass-weighted over the leaf's fields — plus the affine check's own
spread term.

The indicator: field j is FLAGGED on output c when its unmodelled
within energy exceeds its mass share of the tolerance,
p_j λ_jc / N > ε · V_btw · p_j, i.e. λ_jc > N ε V_btw with V_btw the
between-field term Σ_j p_j I_jc² / N (survey units; at 2188 the units
coincide with the full data to 1%). Report the count per output and the
union over outputs, at 1094 and at 2188 (L1 and L2), and the share of
the exact within-field variance held by the flagged fields.

## 3. The computation

At each survey size: build CONN, g, ν, λ; run the emulator (variant A,
ε 0.01, exact measurements) with the PILOT = the affine field's
prediction at every point, ψ̂(x_i) = I_{j(i)} + g_{j(i)}ᵀ(x_i − x̄_{j(i)})
(the plan drops the Gaussian process; state that the stored run's
leaves differ); score the 351-or-so measured bins and the final leaves
as in the affine check, now with σ_S² and the leaf error bar under
noise-only, L1 and L2. Report, per output and per size: the bin ratio
(overall, by size class) under the three bars; the leaves' Σ T̂/Σ T
(corrected) overall and by size class; the implied error/ε beside the
baseline (the stored run's κ̂·ΣW read at that size, from its scalar
record V_tot_hat); the leaf-sum error bar under L1 and L2 as sd/ΣT̂ and
whether |Σ T̂ − Σ T| ≤ 2 sd; the flagged counts and shares. Plus the
alignment rates, two-hop/pinv counts, the identity check, wall time,
`python3.9 --version`.

## 4. Worked example and explosion cases

pa at 2188: the quantized–full-data residual energy is 0.9% of the
within and the 3 October run read pa at −0.02% with the GP pilot on this
survey, so the affine read should be inside the tolerance; at 1094 it
was 0.11. The error bar: at 1094 the shortest-edge semivariance in the
core is 2.8 × Var for pa, so λ in the core is of order the field's whole
variance, L1 puts a bar of that size on every core bin, and the bin
ratio should fall from 348 toward 1 or below (L1 may over-cover; L2
under); at 2188 λ in the core should fall by the resolution.
Flagged fields at 1094: the core's prototypes, expected of order
100–300 (the variance-field check lit 111–153 in its core region); at
2188 fewer or none on the smooth outputs.
Explosion: λ = 0 on every field of an output (noise correction eats the
residuals) → L1 = noise-only, reported as such; fields with < d
neighbours use two hops (count); the emulator with the affine pilot may
stop at a different L than the stored run (report both L's); no
optimisation, no cap to trip; compute under a minute.

## 5. Output

`stage1_affine_refined.py` and `report.txt` in this session's scratchpad
under `stage1_affine_refined/`. Plain text, verbatim, no interpretation.
