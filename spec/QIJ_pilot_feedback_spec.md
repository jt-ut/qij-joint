# Pilot feedback: the loop's measurements refine the survey where the pilot is contradicted

Draft 3 October 2026 (coder's spec, replaces the planner's QIJ_feedback_survey_spec.md, which the author
judged ad hoc: fixed timing, once per run). STATUS: NOT BUILT. Section 6 lists the decisions the author
must make before a build; my recommended answers are marked.

## 0. Idea (the author's design)

The unified loop measures bin means. When a round realizes more within-variance than the model said
remained, the pilot is contradicted. The offending influence bins hold data points; each point belongs to
one X-VQ cell (its `bmu`; under `survey='moments'` a point's activation on the survey IS its bmu field, so
this is exact, not an approximation). Attribute the round's excess to cells through those memberships,
refine the cells that carry it, re-survey them, refit the pilot, rebuild the state vector, and continue.
The method decides when (whenever contradicted) and where (where the excess is); it repeats until the
pilot agrees with the measurements.

Evidence (offline, stored products; draw 45 of cloudfil_u at eps 0.01): the excess of the first firing
round is covered by 10 of 1094 cells (0.9%: core2 6, filament 4); rounds 6-7 need 1-3% of the cells,
all core2/filament. A global 2x survey (all 1094 cells) fixed this draw's under-read.

## 1. Code map (what exists, where)

- Stage 1, `qij.py:QIJ.fit`: `run_xvq` (core/xvq.py) builds the codebook (`fit_xvq`: centers, p, bmu,
  bmu2, conn = vqlp CADJ restricted to live prototypes) and the survey (`prototype_influences`: moments
  rows per field = d_x+1 simplex vertices reproducing the field's mean and covariance, or the field's own
  rows when n_j <= d_x+1; weights sum to N; theta_Q = T(rows, omega0) continued from theta_hat; I_j = forward
  difference raising field j's mass, step t_j = step_parameter(delta_f, p_j), start theta_Q, eta_rows;
  then mass-centred). `fit_influence_model(Z, xvq, I_proto[:, measured], theta_Q[measured], eta, gptrend,
  gpwidth, pool, fit_weights)` (core/influence_model.py) uses xvq.centers, xvq.p, xvq.conn (local spacing
  h_j) and xvq.bmu (per-point length under gpwidth='local'). `psi0_all = _psi0(model, Z)`.
- Stage 2, `core/joint.py:run_joint`: psi_hat = psi0_all - offset, root centring; loop of rounds; each
  split: two-means cut on psi_hat/std0, small child measured by a central pair, large by conservation,
  D_s, G_s (from pre-update child means), `_apply_update` (scale or shift so the leaf mean equals U),
  children's var_hat stored at creation; kappa, se from `calibrate.kappa_hat(D, G)`; V_win, se_V_win from
  `calibrate.vwin_hat(W, n, kappa, se)`; the stop `calibrate.stop_test`; `round_V_win` is recorded at the
  START of each round (it is the "before the round" quantity of section 2.1).
- Products: rounds table (per round: kappa, se_kappa, V_win, margin before the round), splits table (G, D,
  and since 3 Oct W_parent, n_parent at split time).

## 2. The pass

2.1 TRIGGER (per round r, after the round's splits are measured): output c fires when
  sum_{s in r} D_sc > V_win_c(r) + z * se(V_win_c)(r),
V_win(r), se(r) the loop's own values at the start of round r (the stop's own estimate and margin, z the
stop's own z). Any firing output triggers the pass. SEE DECISION D1 (the early-round problem).

2.2 WHERE: for every split s of round r and each firing output c, its excess
  e_sc = max(0, D_sc - kappa_c(r) * W_parent,sc).
Cell score S_jc = sum_s e_sc * (n of parent(s)'s points in cell j) / n_parent(s).
Per firing output, rank cells by S_jc and take the smallest prefix whose cumulative score reaches
E_c = min(sum_s D_sc - V_win_c(r), sum_j S_jc). The refined set is the union over firing outputs.

2.3 REFINE each selected cell j: k-means (the X-VQ's own `VQFitter`, same seed rule) on Z restricted to
the cell's points with k = 2 sub-centres (DECISION D4); the sub-cells replace cell j (new centers
appended, j's slot reused for one of them), points' bmu updated inside the cell only; bmu2 recomputed for
every point (nearest live prototype other than bmu, `_resolve_bmu2`'s distance routine); conn rebuilt
from the (bmu, bmu2) pairs exactly as vqlp's CADJ is defined (the build must first VERIFY that this
reconstruction reproduces `fit_xvq`'s conn at fit time; if it does not, stop and report); p recomputed.

2.4 RE-SURVEY (DECISION D3): rebuild the moments rows for the refined cells only (other fields' rows
unchanged); theta_Q' = T(new rows, new omega0, start=theta_Q, eta_rows) (one quantized fit); measure the
NEW prototypes' influences against theta_Q' exactly as `prototype_influences` does (same delta_f, steps,
start, pool batch); keep the old prototypes' I as measured; re-centre all I by the new masses. Count the
evaluations under a new key `evals_prototype_refine` (and rows) in the size-N accounting.

2.5 REFIT the pilot: `fit_influence_model` on all prototypes (hyperparameters re-selected as at stage 1),
psi0_new = _psi0(model_new, Z) for ALL points.

2.6 REBUILD THE STATE VECTOR by FULL REPLAY (DECISION D2): psi_hat <- psi0_new - offset_new, root
centring; then every measured split in its original order: its stored cut (child index sets), its stored
measured U_small/U_large and steps, `_apply_update` on both children exactly as at measurement. Recompute
along the replay each split's G (pre-update child means of the replayed vector) and W_parent, and each
leaf's m, ebar, var_hat. No evaluation. After the replay every measured leaf's mean equals its U again and
the within shape everywhere is the new pilot's.

2.7 CONTINUE: kappa, se from the replayed G with the stored D (DECISION D5), recomputed W, same ranking,
rounds, stop, the same tree (no cut is undone). L_max stays the ORIGINAL M_X_used (DECISION D6). The
pass may fire again at any later round (no once-only rule).

## 3. Products (added)

Per run: n_passes; per pass: round, firing outputs and their ratio sum D / (V_win + z se), cells refined
(count, ids), prototypes added, evals_prototype_refine and rows, wall time; kappa/se/V_win/margin before
and after each pass. The rounds/splits/bins/points tables keep their meaning (replayed G/W_parent overwrite
the stored ones; a column `replayed` marks rows rewritten by a pass). A pass log table `passes.parquet`.

## 4. Constants

None new: z (the stop's), the coverage rule (the round's own excess), k = 2 per refinement (repeat firing
refines further). D1 may require one.

## 5. Validation (ONE case first, then the author decides)

Draw 45 of cloudfil_u p2, eps 0.01, study settings: against the stored 1x run (L 348, loop 694 evals,
reported-variance error -0.4..-4.1%) and the 2x run (L 255, 508 evals, +1094 prototypes, +0.07..-1.2%).
Report: passes (rounds, outputs, cells and their components), prototypes added and their size-N, loop
evals and leaves, total size-N, per output the reported-variance error vs exact IJ (runs/shearmix4/
cloudfil_u_p2_N10000/ij/s00045.parquet). Acceptance: error <= eps on every output AND total size-N below
the 2x run's AND the bead draw-1 eps-0.01 run is unchanged bit-for-bit (the pass never fires there).

## 6. Decisions for the author (my recommendation first)

D1 EARLY ROUNDS / KAPPA FROM FEW SPLITS (the real open question). Offline, the trigger of 2.1 fires on the
bead at round 2 (sum D / V_win = 739 on log_contrast): kappa after ONE split is that split's own ratio,
and `kappa_hat` reports se = 0 with one split, so the margin vanishes exactly when kappa is least
reliable. The bead also fires marginally at its last round (1.04 with the z se margin). Options:
 (a) Use kappa = 1 (the raw pilot) and se from the pilot's own spread until kappa_hat's se is defined
     (>= 2 splits) — no count, but untested; I would test it offline first.
 (b) Keep the stop's existing n_min gate for the trigger (the same lever the stop already uses) — simple,
     but it is the count the author objected to.
 (c) A trigger that does not use kappa at all (raw pilot within, kappa = 1 throughout): test offline.
 TESTED (c) offline (sum D_r / sum W(r), sum W = V_win/kappa from the rounds table, kappa = 1): bead d1
 eps 0.01 rounds 1-9: 0.63, 0.47, 0.50, 0.85, 0.68, 0.88, 0.82, 0.81, 0.83 (never fires); round 10 (its
 last): 1.53 (log_axis_ratio). Draw 45 eps 0.01: fires from round 1 (61, 83, 1.6, 4.6, 22, 191, 344, 309,
 1425, 813); eps 0.1: from round 1 (12 ... 555). No kappa, no count, so the early-round problem is gone.
 The bead's last-round 1.53 is where the splits are smallest and D approaches measurement noise; the loop
 already computes each split's expected noise n_Delta + b_Delta (its closing floor): the principled
 correction is sum_s (D_sc - n_Delta_sc - b_Delta_sc) > sum W_c(r). Not testable offline (the floor terms
 are not in the products; a products-only column, as W_parent was, would make it testable).
 RECOMMENDATION: (c) with the noise-floor correction; record the floor per split, rerun bead d1 and d45
 (one case each, local) to confirm the bead never fires, then build.
D2 REPLAY SCOPE: full replay everywhere with the refitted pilot (recommended: one pilot, exact,
 no-evaluation, and the GP refit changes psi0 everywhere anyway) vs repopulating only the refined cells
 (mixes two pilots in one vector).
D3 SURVEY CONSISTENCY: the refined rows change the quantized dataset, so theta_Q' differs slightly from
 theta_Q and the old prototypes' I were measured on the old rows. Recommended: one new quantized fit,
 measure only the new prototypes, keep the old I (an approximation to state) vs re-measuring every
 prototype (exact, ~M_X extra quantized fits per pass, i.e. the global re-survey's cost).
D4 REFINEMENT FACTOR: k = 2 per selected cell per pass (recommended; repeated firing refines further where
 needed) vs the planner's ceil(n_cell / smallest leaf in the cell).
D5 KAPPA AFTER A PASS: recompute every past split's G under the replayed vector (recommended: kappa then
 measures the CURRENT pilot) vs keep the historical G (kappa mixes the old and new pilots).
D6 LEAF CAP: L_max stays the original M_X_used (recommended; prototypes are not leaf budget) vs grows with
 the added prototypes.
