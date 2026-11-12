# The instrument gate: does a forward difference on the quantized fit measure a field's influence reliably as fields get small? Draw 45, real quantized evaluations (planner, 6 October 2026; an offline test by an agent, no package code)

## 0. The question and why it is the gate

qijt (29 Sept – 1 Oct) was parked because its measured contrasts at fine
nodes carried outliers thousands of noise units off: a sign flip between
the step and twice the step, a node exact at half the step but not at
the step, false outliers where the declared noise scale η·|θ̂| was near
zero, and a base fit that was not a fixed point of its own continuation.
Everything built today on quantized measurements (the between-field
term converging; the affine within read; the data-space tree) replaced
a quantized evaluation by the exact field mean plus scatter. This test
puts the real instrument in: the package's own survey code path,
one row per prototype (the qijt design, no moment rows), on draw 45,
at the field sizes a refinement produces. Three numbers decide it, per
field-size class: (1) the slope and scatter of the measured influence
against the exact field mean; (2) the agreement of the measurement at
half, base and double step; (3) the fraction of outliers, fields whose
error exceeds three times the class's scatter. Pass: scatter and
outlier fraction that do not blow up as fields shrink to two points,
and step agreement within the scatter. If it fails below some field
mass, that mass is the limit of the quantized instrument and the plan's
branch B applies.

## 1. Setup (the package's own machinery, read-only; main at e6d6570)

- Case ('cloudfil_u', 'p2') from `qij_joint.registry`; the draw
  `datasets.cloudfil_G_U_P3_v1(10000, 45)`; the oracle influence
  `runs/shearmix4/cloudfil_u_p2_N10000/ij/s00045.psi.parquet` (the
  p2 columns; SCORE ONLY).
- Stage 1 as `qij.py` runs it: follow `QIJ.fit`'s own calls — the full
  fit for `theta_hat` (≈ 9 s), `core.xvq.fit_xvq(Z, M=1094, seed=0)`
  (verify ≥ 99% bmu agreement with the stored points file), then
  `core.xvq.prototype_influences(W_X, counter, p, eta, survey='points',
  quantized_start='full-data', theta_hat=theta_hat, ...)` with the
  counter and eta exactly as `QIJ.fit` builds them. Do NOT reimplement
  the forward difference; the instrument under test is the package's.
- Timing on record: 1108 quantized evaluations in 77 s on one worker
  (0.07 s each); the full fit 8.6 s. Budget below ≈ 8000 evaluations ≈
  10 min on one worker; use the package's pool at 4 workers if
  `QIJ.fit` shows how, else one worker and the 10-minute cap.

## 2. The field sets (four sizes, nested)

Level 0: the 1094 prototypes (≈ 9.1 points per field). Choose the 32
fields with the largest max-over-outputs |I_j|/sd(I_c) from the stored
prototypes file (the fields that carry the influence). Levels 1–3:
split each chosen field recursively by deterministic two-means on its
own points in data space (one Lloyd pass from the two farthest points),
into 2, 4, 8 sub-fields (≈ 4.5, 2.3, 1.1 points each; stop splitting a
sub-field of one point). At each level the prototype set is: the 1062
unchosen fields' prototypes unchanged + the sub-fields' centroids, with
masses n/N. Rows = one per prototype (survey='points').

## 3. The measurements

- At levels 0, 1, 2, 3: one full survey at the base step
  (δ_f = 2√η_Q as the package sets it) → I_j for every prototype.
- At level 3 only: two more full surveys with the step halved and
  doubled (pass eta/4 and 4·eta to the same call; δ_f ∝ √eta).
- The fixed-point check at each level: evaluate T once at the base
  weights from the base start and report ‖θ − θ_Q‖/‖θ_Q‖.
- Evaluation count ≈ 1094 + 1126 + 1190 + 1318 + 2·1318 ≈ 8000.

## 4. The scores (per output; the sub-fields grouped by points per field: ≥ 9 (level-0 unchosen fields as the reference class), 4–8, 2–3, 1)

- (1) Measured I_j against the exact field mean F_j (mean of the oracle
  ψ over the field's points): mass-weighted slope and intercept, R²,
  residual sd relative to sd(F) within the class, and the residual
  relative to the DECLARED noise scale √2·η_Q·|θ_c|/t_j (how many
  declared-noise units the residuals are).
- (2) Step agreement at level 3: I_j(half)/I_j(base) and
  I_j(double)/I_j(base) per field — median, 10/90 pct, and the fraction
  with a sign flip or a ratio outside [0.5, 2].
- (3) Outliers: the fraction of fields per class with
  |I_j − b·F_j| > 3 × (class residual sd), and the five worst (j, n_j,
  I_j, F_j, the three step values).
- The fixed-point residual per level; wall time per survey;
  `python3.9 --version`.

## 5. Worked example and explosion cases

Level 0 is the stored survey redone with one row per prototype instead
of three: expect slope 0.7–1.0 and residual 8–26% of sd(F) as the
moments survey gave, or worse if one row per field is a poorer
representation at nine points (the moments rows were introduced for
exactly that; report the difference, it is the "over-quantizing"
question). Level 3 fields have one or two points, mass ≈ 1–2e-4,
t_j ≈ δ_f·p ≈ 4e-3·1.5e-4 ≈ 6e-7, declared noise √2·η_Q|θ|/t_j ≈ 10·|θ|
— of the order of the influence itself or larger: if the measurement is
no better than the declared noise there, the instrument does not reach
two-point fields and the limit is the mass where the residual sd
crosses the influence's own spread. The sign-flip signature of qijdt is
the step test's job. Explosion: a sub-field of one point whose weight
step makes the estimator's fit fail (NaN) — count and skip; a
continuation that does not converge — count; never retry with a
different step or start.

## 6. Output

`instrument_gate.py` and `report.txt` in this session's scratchpad
under `instrument_gate/`. Plain text, the tables of section 4 by class
and output, verbatim, no interpretation.
