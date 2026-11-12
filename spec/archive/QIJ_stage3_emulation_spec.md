# Stage 3 of QIJ_qijt_affine_plan.md: the method emulated — the X-VQ grown by lack of fit, quantized measurements only, the affine field, the tolerance stop — on draw 45 and the Fundamental Plane (planner, 6 October 2026; an emulation on products, not a build)

## 0. The question

Does the method of the plan's section 0 reach the tolerance on every
output, and at what cost, starting from the stored survey, with no
full-data evaluation? Two cases: draw 45 of the sheared mixture from
its 1094-prototype survey (the within short by up to 3.2ε there, inside
0.14ε at 2188), and draw 0 of the Fundamental Plane from its
371-prototype survey (the within read short on every output, 76 fields
flagged). Deciding numbers per case: error/ε per output at the stop
against the exact V_ij; the cost in size-N units; the number of fields
and rounds at the stop; the trajectory of both by round. Baselines: the
stored run's `rows_total/N` (survey + full-data central pairs) and its
error; for draw 45 also the floored loop (2038 full-data pairs, inside
±0.06ε).

## 1. The state

A set of fields, each with its member points, mass p_j, mean x̄_j,
covariance Σ_j, a measured influence I_jc per output, and the
connectivity neighbours rebuilt from every point's nearest and
second-nearest field after each round. From these: the gradient g_jc
(least squares on the neighbours; two hops where fewer than d), the
noise ν_c (the smallest edge-length-quintile semivariance, recomputed
each round), the lack of fit λ_jc = max(mean_k r_jkc² − 2ν_c, 0), the
affine within τ̂_jc (the corrected form of the affine check over the
field's own points; 0 for singletons), and the L1 bar.

Initial state: the stored survey's fields and I_j (1094 and 371).

## 2. The measurement stand-in (stated, not a method rule)

A quantized evaluation at a field returns the full-data mean of the
exact influence over the field's points, times a bias b_c, plus
Gaussian scatter with sd s_c times the spread of the field means. The
bias and scatter are what the quantized survey showed against the
full-data field means (scratchpad data_look/qvsfull*.py): on draw 45,
slope 0.72–0.96 and residual 8–26% at ~9 points per field, slope
0.996–1.007 and 1–8% at ~4.6 points per field; both interpolated
linearly in log(points per field) between those two sizes, and held at
the finer size's values for smaller fields (no further improvement
assumed). For the Fundamental Plane compute the slope and scatter once
at its 371 prototypes (the only resolution on disk) and use them for
every field (conservative). The ORIGINAL survey values are kept as
measured (they carry their own bias); a re-measurement of a parent is
never made (the halves replace it).

## 3. The loop

Each round: (1) rebuild neighbours, g, ν, λ, τ̂, the bar; (2) read
V_btw,c = Σ_j p_j (I_jc − Ī_c)²/N (mass-centred), V̂_win,c =
Σ_j p_j τ̂_jc/N, sd_c the L1 bar on the within sum (Stage 1's leaf-sum
form with the fields as the leaves), V̂_c = V_btw,c + V̂_win,c; (3) STOP
when V̂_win,c + 2 sd_c ≤ ε V̂_c on every output; (4) otherwise flag every
field with n_j ≥ 2 and λ_jc > N ε V_btw,c on any output; if none is
flagged while the stop is unmet, flag the fields holding the largest
p_j·(τ̂_jc + λ_jc) until their sum covers the excess (report that this
fallback fired); (5) split every flagged field by the median of its
points' projection on the field's split direction — the leading
eigenvector of Σ_c g_jc g_jcᵀ / Var(I_c), the direction along which the
influences change fastest, standardised across outputs — into two new
fields; (6) measure both halves by the stand-in (two evaluations per
split; report the cost also with conservation, one per split, as the
alternative); (7) next round.

Cost: every evaluation costs (d + 1)·M_t/N size-N units, M_t the field
count at the time (the moments survey's d + 1 rows per field; state
it), plus the initial survey's `rows_prototype/N` from the stored
record. Report the cumulative cost per round and at the stop.

Caps (sanity): at most 12 rounds and at most N/2 fields; if a cap
binds, stop and report the per-output margins (V̂_win + 2sd)/(ε V̂) at
that point — a margin stuck above 1 on an output whose bar is wide
(x, y at 2188: sd/Σ T̂ ≈ 100) is a finding about the L1 bar, not a
defect.

## 4. Scores and report

Per case, per round: fields, flagged, evaluations, cumulative cost,
V̂_c/V_ij,c and the error/ε per output, the margin per output. At the
stop (or cap): the error/ε per output beside the baselines; the cost
beside `rows_total/N` of the stored run (and 2038 + survey for the
floored loop on draw 45); the field count; the share of fields that
are singletons; whether the fallback fired; the identity check
V_btw + Σ_j p_j Var_j(exact)/N = V_ij at every round (exact, since the
fields partition the points; max relative deviation). Plus the
stand-in's bias/scatter tables used, the alignment rates, wall time,
`python3.9 --version`.

## 5. Worked example and explosion cases

Draw 45, round 1: 132 fields flagged at 1094 (Stage 1), 264
evaluations at (3·1226)/10000 ≈ 0.37 units each ≈ 97 units, plus the
survey's stored rows_prototype/N. After the split the core's fields are
at about the 2188 survey's resolution where the affine read was inside
0.14ε, so by round 2–3 the within on pa should be near the truth; the
stop then depends on the bar: on log_axis_ratio and pa the L1 bar at
2188 was 0.3–0.5 of the within (the stop can close); on x, y, logit_w
it was 65–100× the within (the stop cannot close on those outputs
without many more rounds — the cap will bind there and the margin
table will say so; this is the L1 bar's over-width on smooth outputs,
known from Stage 1). Expected total: 2–4 rounds, a few hundred units,
against the stored run's rows_total/N (≈ 360 + 702) and the floor's
≈ 360 + 2038.
Fundamental Plane: 76 fields flagged at 371; the fields are 5–6 points;
splitting them gives 2–3-point halves whose affine within is near
exact; expect the error on output a to close from −2.07ε within one or
two rounds; cost 152 evaluations at (4·447)/2000 ≈ 0.9 units each ≈
135 units against the stored run's rows_total/N.
Explosion: a two-point field splits into singletons (τ̂ = 0, never
flagged again); a field with fewer than d + 1 points has a singular Σ_j
— the split direction is then the gradient itself (no eigendecomposition
needed); the stand-in's scatter at tiny fields is held at the finer
measured value; the bar on smooth outputs may hold the stop open (the
cap); the fallback (step 4) may flag fields repeatedly on an output
whose bar is the problem — report its count.

## 6. Output

`stage3_emulation.py` and `report.txt` in this session's scratchpad
under `stage3_emulation/`. Plain text; per case the round table, the
stop table, the baselines; verbatim, no interpretation.
