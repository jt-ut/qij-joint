# Stage 1 of QIJ_within_variance_decision.md: the generality check — the graph variance-field calibration on one draw each of the Fundamental Plane, the IMF and the multivariate t (planner, 6 October 2026; an offline check, not a build)

## 0. The question

Is the influence model that was calibrated on draw 45 of the sheared
mixture (QIJ_graph_variance_field_check_spec.md, revision 2: local
width, one log-variance per prototype smooth along the connectivity
graph and shared across outputs, the declared noise floor) calibrated on
cases with no cores and a different influence support? Same script,
same metric, one draw per case, nothing tuned. If the ratio of realized
squared error to predicted variance lands in the same band as on draw
45 (0.75–3.7 overall; 0.7–4.1 on the one- and two-point bins) on all
three cases, the model is generic. If it does not on any case, the
model family is wrong and nothing in Stage 2 should be built.

## 1. The three cases (products exist, draw 0 of each, N = 2000)

`/Users/jtaylor/Dropbox/Research/QIJ_joint/runs/paper_n2000/<case>/`:

| case | draw | data dimension | outputs (prototype-influence columns) | M_X | data regeneration | quantizer coordinates |
|---|---|---|---|---|---|---|
| fp_all_N2000 | s00000 | 3 (w_0..w_2) | I_a, I_b, I_c, I_scatter | 371 | `datasets.fp(2000, 0)` | identity (vq_transform None) |
| imf_all_N2000 | s00000 | 1 (w_0) | I_m_c, I_sigma, I_x | 321 | `datasets.imf(2000, 0)` | identity; a 1-D sample is promoted to (N, 1) |
| mvt_nu_N2000 | s00000 | 10 (w_0..w_9) | I_nu | 188 | `datasets.mvt(2000, 0)` | `datasets.mvt_vq_transform` (per-coordinate mean/sd standardisation of this draw) |

Per case: `qij/s00000.prototypes.parquet` (j, p, w_*, I_*), `qij/s00000.points.parquet`
(i, bmu, psi0_*, sigma_*, …), `ij/s00000.psi.parquet` (psi_* exact
influence), `qij/s00000.parquet` (scalars; M_X), `oracle/s00000.parquet`
(theta_hat_* per output, for the noise floor). The draw's seed is
base_seed + s = 0 + 0 = 0 (pipeline rule). Verify alignment by nearest
prototype against `bmu` (≥ 99%; stop otherwise). These prototypes files
carry NO `h` column: define h_j = the distance from prototype j to its
nearest other prototype, in the quantizer coordinates, and say so.

Declared accuracy for the floor: these older records do not store
eta_Q. Read the default declared accuracy and the prototype-stage
accuracy rule from `qij_joint/qij.py` (`QIJ.__init__`'s `eta` default and
how `eta_Q` is derived; read-only) and report the values used; if the
rule cannot be read with confidence, use the draw-45 value
4.318328622856778e-06 and SAY SO. The floor n_c² = 2 η_Q² θ_c² ·
median_j(1/t_j²) with t_j = 2√η_Q · p_j/(1 − p_j) and θ_c =
`theta_hat_<output>` from the oracle record. Report n_c² beside
Var(y_c).

## 2. The model and the metric: exactly revision 2 of the graph check

Copy `scratchpad/graph_variance_field_check/graph_variance_field_check_v2.py`
and generalise its loading only: outputs from the psi columns,
dimension from the w_* columns (the kernel formula is dimension-free:
r² = 2‖x_i − x_j‖²/(ℓ_i² + ℓ_j²), trend h(w) = (1, w_0, …, w_{d−1})),
M_X and L_max from the scalar record (L_max = M_X, as the product uses),
the emulator's `run('A', 0.01, L_max, exact, pilot)` unchanged (it is
case-independent: it takes the exact influence and the pilot vector).
CONN from the regenerated points as before. (C) and (D) as revision 2
(no per-output offset; explicit s_c², ν_c with the floor; the c choice
by the summed profiled log ML on the 8-value grid; τ = 1; a shared
across outputs — for mvt_nu, one output, "shared" is trivial). Bins =
the small children of the emulation's splits. Metric = the ratio of
realized squared error to predicted variance, overall, by bin-size
class and by round; RMS z, median |z|, frac |z| > 2 and > 3. Report
exactly the revision-2 tables per case, plus for each case the
emulation's stop L, splits, and the a-hat range with the ten largest
â prototypes (j, w, h, â, log|I|).

## 3. Explosion cases and budget

- A prototype with no CONN edge (possible in 10-D with 188 prototypes):
  the graph penalty leaves its a_j to the ridge alone; report the count
  of isolated prototypes.
- In 10-D the local width's nearest-prototype distances are large and
  the kernel may be near-white between prototypes; c at the grid's
  lower bound is then a finding (the smooth part not carrying
  information in 10-D), not a defect — report it.
- SANITY CAP as before: Cholesky failure after jitter or non-finite σ_S
  → stop and report which.
- Budget: M_X ≤ 371 so every Cholesky is cheap; the emulation at
  N = 2000 is well under a second; expected under 60 s per case. Hard
  cap 5 minutes total; run the cases in the order imf, fp, mvt.

## 4. Worked example

IMF, 1-D data: the influence of a stellar mass on the Chabrier
parameters is smooth in log mass with its largest values at the ends of
the mass range (the slope x is driven by the high-mass tail). The
learned log-variance should rise toward the ends; the one- and
two-point bins will sit there. With 321 prototypes on a line, h_j is
tiny and the local width will resolve any smooth variation; c should
sit in the interior. If the ratio on the small bins is in the 0.7–4
band, the model generalised from a 2-D compact support to a 1-D
tail-dominated influence with nothing changed.

MVT, 10-D, one output: the influence of a point on ν is a function of
its Mahalanobis radius alone, so in the 10 whitened coordinates it is
a radial function — smooth, no compact support, largest in the tails.
188 prototypes in 10-D are far apart; the local width may not carry
information between them (c at the bound) and the variance field
must then do the whole job as it did with the global length on draw
45 (ratio 0.9–4.1). That outcome would still be a pass for the
variance field and a finding against the smooth part in high
dimension.

## 5. Output

`stage1_generality_check.py` and `report.txt` in this session's
scratchpad under `stage1_generality_check/`. Plain text, one block per
case, then a three-line summary (case, overall ratio range over outputs,
one-and-two-point-bin ratio range, c and whether at a bound). Verbatim,
no interpretation.
