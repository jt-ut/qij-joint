# The one-rule method run end to end with real quantized evaluations: draw 45 and the Fundamental Plane (planner, 6 October 2026; an offline test by an agent, no package code)

## 0. The question

The method of QIJ_qijt_affine_plan.md section 0 (rewritten today): one
X-VQ grown as a tree, every row a prototype carrying its field's count
(weights sum to N), the estimator evaluated on the current prototypes
only, a field split where its lack of fit exceeds the tolerance's
share, the variance read as the between-prototype term plus the affine
within, the stop at the tolerance. Every piece has been exercised in
isolation today (instrument_gate, onerow_refined, Stage 1). This runs
the loop itself with the real instrument in place of the stand-in that
voided Stage 3. Deciding numbers per case: error/ε per output at the
stop against the exact V_ij; evaluations and wall time to the stop,
beside the loop as built (draw 45: 1094 quantized + 702 full-data
evaluations, errors −0.25…−1.0ε on six outputs and −3.3ε on pa; in
wall time ≈ 77 s + 702 × 7 s); the rows at the stop; the trajectory by
round with the between and within terms each against their exact value
on the same partition.

## 1. The instrument (from scratchpad onerow_refined/onerow_refined_N.py; NOT core.xvq.prototype_influences, whose survey='points' weights sum to M)

Rows W (R × d), masses p (counts/N, Σp = 1), weights ω0 = N·p.
θ_Q = T(W, ω0) continued from the previous θ_Q (the full-data θ̂ for the
first survey), eta 1e-7. For each prototype j: t_j = step_parameter(δ_f, p_j)
with δ_f = forward_step(T.eta); ω_j = perturbed_weights(ω0, mask_j, t_j);
I_j = (T(W, ω_j; start θ_Q) − θ_Q)[measured]/t_j; mass-centre I over the
rows. Assert Σω0 = N before every survey. One evaluation per prototype
per survey; every survey re-measures every prototype (stale values from
an earlier fit are not reused; the gate showed untouched fields'
measured values change when the core is refined). Report also the
count of evaluations that a "new prototypes only" policy would have
spent, as the lower bound.

## 2. The state and the loop

Initial fields: the stored X-VQ (draw 45: the 1094 stored centres,
nearest-centre assignment, bmu agreement 100%; FP: its 371 stored
centres, draw 0). Per round:

1. SURVEY all prototypes (section 1).
2. CONN from every point's nearest and second-nearest current
   prototype; per field: gradient g_j by least squares on CONN
   neighbours (two hops where fewer than d), noise ν_c = the smallest
   edge-length-quintile semivariance, lack of fit λ_jc = max(mean_k
   r²_jkc − 2ν_c, 0), affine within τ̂_jc = Var over the field's points
   of I_j + g_jᵀ(x − x̄_j) minus the gradient-noise term (floored at 0;
   0 for singletons), the L1 bar on the within sum as Stage 1.
3. READ: V_btw,c = Σ_j p_j (I_jc − Ī_c)²/N; V̂_win,c = Σ_j p_j τ̂_jc/N;
   V̂ = V_btw + V̂_win; margin_c = (V̂_win,c + 2 sd_c)/(ε V̂_c).
4. STOP when margin ≤ 1 on every output.
5. FLAG fields with n_j ≥ 2 and λ_jc > N ε V_btw,c on any output; if
   none while unstopped, flag the fields with the largest p_j(τ̂_jc +
   λ_jc) on the binding output until their sum covers the excess
   (report when this fires).
6. SPLIT each flagged field at the median of its points' projection on
   the leading eigenvector of Σ_c g_jc g_jcᵀ/Var(I_c) (the gradient
   itself if the field has fewer than d+1 points); the halves become
   two prototypes at their centroids with their counts; the parent is
   removed.
7. Next round.

Caps: 8 rounds; rows ≤ N/2; 12 minutes of compute in total (draw 45
first, then FP); report the margins and the state if a cap binds.

## 3. Scores

Per round and output: rows, flagged, evaluations this round and
cumulative, wall this round, V̂/V_ij, error/ε, margin,
V_btw/V_btw_exact(partition), V̂_win/V_win_exact(partition), the
mass-weighted slope of I_j against the exact field mean (the units
scale; expected → 1 with refinement), ν_c, the identity check
V_btw_exact + V_win_exact = V_ij. At the stop or cap: error/ε per
output beside the loop as built; total evaluations, the lower-bound
count, total wall, rows, singleton share; whether the fallback fired.
Plus `python3.9 --version`.

## 4. Worked example (today's numbers)

Draw 45, round 1: 1094 evaluations ≈ 33 s; the between term from the
unrefined survey is 1.10–1.78 of the exact (the units scale 0.72–0.96
inverted); ≈ 132 fields flagged. Round 2: ≈ 1400 rows, ≈ 45 s; the
gate's level-1 result says the whole survey is then at R² ≥ 0.996 and
the slope 0.97–1.05, so the between term is within a few percent and
the affine within read is near the Stage 1 value inside 0.14ε. Expect
the stop in 2–4 rounds at 4000–6000 evaluations and 2–4 minutes of
wall, against 82 minutes of wall for the 702 full fits of the loop as
built. FP: 371 → ≈ 500 rows, under a minute.

## 5. Explosion cases

- The within read from neighbour gradients blew up in Stage 3 under
  fake scatter; with real measurements at 1–2% error it may hold. If
  V̂_win/V_win_exact exceeds 10 on any output in any round, say so in
  the round table; the between term is reported regardless and is the
  method's main claim.
- The margin may stay above 1 on outputs where the L1 bar is wide (the
  smooth outputs); the cap then binds — report the margins.
- A NaN evaluation: count and skip the prototype that round.
- Never reuse the package's survey='points' path; assert Σω0 = N.

## 6. Output

`one_rule_run.py` and `report.txt` in this session's scratchpad under
`one_rule_run/`. Plain text, per case the round table and the stop
table, verbatim, no interpretation.
