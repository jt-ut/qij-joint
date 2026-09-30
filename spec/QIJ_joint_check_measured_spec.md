# The joint check with a measured continuation rule (planner, 30 September 2026; one build; "change A")

Scope. A new argument `check_rule` on the joint second stage
(`ivqbins='joint'`, `core/joint.py run_joint`), `pilot='gp'` only in
this build (`'measured'` with `pilot='affine'` raises a clear error;
the affine pricer of 2.2 is specified so a later build can lift the
restriction after its own validation): `'predicted'` = today's check,
byte-identical products; `'measured'` = this document. The change is confined to how the check decides whether
a bin's children are split again, to how a split's kind is priced, and
to what the within-term products contain. Growth, the initial
measurement, the split evaluation itself, the mass-balance
conservation, the evaluation cap, the sigma stage, the marginal path
and every stage before the check are untouched. Code standards:
`qij_joint_plan.md` section 5, verbatim in the build prompt. No check
is added (the package's one check stays the base fit's).

Terms as the glossary: the quantized data, the prototype points, bins;
never "rows"/"field".

## 1. Why (the measured record; the author's three questions)

Baseline = the current code on the P1 ε sweep (in-basin medians, from
the products): at ε 0.01, L0 333, 252 bins flagged at the first check
(76%), 205 check fits (16% of the draw's evaluations); at ε 0.0447,
L0 156, 118 flagged, 292 check fits (38%); at ε 0.2, L0 34, 22
flagged, 356 check fits (69%). A split's two children are today
re-flagged against the SAME pilot means that were already wrong for the
parent, so a flagged lineage splits round after round until the
evaluation cap or the bin is exhausted: at ε 0.2 the 34 initial bins
take 356 fits. Meanwhile the realized shortfall 1 − V_btw/V_ij
(interval study, 414 in-basin draws) is a fifth to a quarter of ε at
every ε, so the check is buying variance well past the tolerance at the
price of most of the method's cost at loose ε. Oracle inputs: none
in the rule below; V_ij enters only the validation as the reference.
The number before the build: section 5's V2 reads it from the sweep
products for the two validation draws.

The posterior variance v_kc cannot decide splits: it is the posterior
variance of the within-bin component (`influence_model.
bin_posterior_variance`, v_k = mean diag Σ_k − mean Σ_k), set by the
kernel width against the prototype spacing, and on the sweep it is
0.2–0.38 of V_btw wherever the noise floor is small against a realized
within share of 0.005–0.036. So under `'measured'` no rule reads v.

## 2. The rule

### 2.1 Seeds (unchanged)

The first check's flags are today's `_flag_mask` on the L0 measured
bins: `'gp'` with the posterior allowance u_kc, `'affine'` with u ≡ 0.
A bin flagged here is an open bin. Every unflagged initial bin is
final. `_flag_mask` is not called again under `'measured'`.

### 2.2 Proposal for an open bin: kind and geometry from predicted gains, no v

For each open bin k (n_k ≥ 2), both kinds are priced per output at
rho2 = 1 from the pilot's mean only, and the larger summed normalized
gain Σ_c g_c / V_btw,c (V_btw the current per-output values) is taken:

* level: `two_means_split` on the bin's own Ψ̃ rows, priced by
  `level_gain_value(p_a, ubar_a, p_b, ubar_b, p_k, ubar_k, N, 1)` per
  output with ubar from `psi_centered` — exactly `decide_split_affine`'s
  level branch, for both pilots;
* adjacency: geometry from `_try_adjacency_split(idx, I_proto[:, c],
  bmu, bmu2)` on output c*, the remaining outputs tried in descending
  score if c*'s split is infeasible. Its price: under `'affine'`,
  `bridge_gain_value(leaf['bridge'], 1)` per output, as built
  (QIJ_affine_pilot_spec.md 3.2), c* = argmax_c of that score /
  V_btw,c; under `'gp'`, `level_gain_value` applied to the adjacency
  geometry's own two children (their `psi_centered` means), per output,
  c* = the output whose adjacency geometry gives the largest
  level_gain_value / V_btw,c over the feasible outputs. `adjacency_
  gain_value` (v-based) is not called under `'measured'`.

An infeasible chosen kind falls back to the other; both infeasible
closes the bin (final, no evaluation). The chosen proposal's per-output
predicted gain vector g is kept for gain_ratio (2.5) and for nothing
else. `decide_split_gp` (the Σ Var_k/V_btw ≥ Σ v_k/V_btw kind rule) is
not used under `'measured'`; it stays for `'predicted'`.

### 2.3 Measurement (unchanged)

All open bins' proposals of a round are evaluated in one pool batch:
the smaller child by one forward evaluation at t_small =
step_parameter(delta_f, p_small), delta_f = forward_step(eta_full),
start = θ̂; the larger child by mass balance; Δ_c = (p_small
U_small,c² + p_large U_large,c² − p_k U_k,c²)/N; V_btw,c += Δ_c. A
failed evaluation cancels the split (parent final, its evaluation
counted). The cap: the check stops when check evaluations reach
1 + M_X_used (`check_capped`), tested at the top of each round as today.

### 2.4 Continuation (revision 2, 30 September): the pilot proposes, the measurement can only close

Revision 1 of this section let the measurement re-propose children
regardless of the pilot and closed a lineage only after two consecutive
below-τ splits. Built and run on draws 1 and 7 it cut the check fits at
ε 0.141 (308 → 214) but raised them at ε 0.01 (174 → 218) and, at a
coarser reproducibility, pushed V_btw above V_ij by up to 5%: at tight
ε the tolerance share τ ≈ ε·V_btw/L is ~2e-5·V_btw at L ≈ 500, so
almost every split cleared it and the second strike wasted one split
per lineage; and the finite-difference noise enters a realized gain
squared, as a positive bias, which revision 1's first-order floor did
not see. Revision 2 keeps the marginal path's two gates in their proper
roles and adds the squared term.

**The rule.** A child bin is OPEN (proposed in the next round) if and
only if

  (a) today's predicted flag holds for it — `_flag_mask` at the current
      bin count, `'gp'` with the posterior allowance u_kc, exactly as
      the seeds (2.1) and as the `'predicted'` rule flag children; AND
  (b) its parent's split PAID: Δ_c ≥ τ'_c for SOME measured output c,
      with Δ the parent's realized gain (2.3) and τ' the floor below.

Both children of a split share (b); each child has its own (a). A
child that fails either is FINAL. Initial bins have no parent and are
open iff (a), as today. A bin with one point, or with no feasible
proposal (2.2) when its turn comes, is final. There is no strike flag
and no second chance: one unpaid split closes both children. The
measurement never opens a child the pilot would not have opened, so
every split this rule makes is one the `'predicted'` rule makes at the
same bin count or a larger per-bin allowance (a smaller L raises
`_flag_mask`'s allowance ε·V̂_c/L, so fewer bins are flagged, never
more); the one path dependence is the kind pricing's normalization by
the current V_btw (2.2), which can change WHICH split a bin gets, not
whether it is split.

**τ.** At the start of round r, with L_r the current bin count and
V_btw,c^(r) the current between-bin variance, τ_c^(r) = ε ·
V_btw,c^(r) / L_r — the marginal path's τ at selection (method_notes
section 4), per output. Every split evaluated in round r is judged
against τ^(r); a child's (b) is therefore fixed when its parent's split
is evaluated and never recomputed.

**The finite-difference floor, both orders.** η_full is measured as a
RELATIVE reproducibility of one continuation, each output's difference
scaled by the larger of the two values compared (`xvq._measure_eta_Q`),
no floor at 1. The error of the one-shot evaluation's U_small is
therefore δU_c = η_full · |θ̂_c| / t_small, with |θ̂_c| floored at
machine epsilon only (an output estimated at zero gets a zero floor and
τ_c alone decides it; a floor at 1 would overstate the noise a
thousandfold on x or y at 1e-3). Mass balance carries −(p_small/p_large)
of that error into U_large. In the realized gain Δ_c the error appears
to first order as scatter and to second order as a BIAS (the squares
are always positive):

    n_Δ,c = (2 · p_small / N) · (|U_small,c| + |U_large,c|) · δU_c       (scatter)
    b_Δ,c = (p_small / N) · (1 + p_small / p_large) · δU_c²             (bias, = E[Δ_c] from noise alone)

and the floor a gain must clear is

    τ'_c = max( τ_c^(r),  n_Δ,c + b_Δ,c ) ,

per split (p_small, p_large, U_small, U_large are the split's own) and
per output. b_Δ scales as η_full · θ̂_c² · (1 + p_small/p_large) /
(4 N p_small) (t_small = 2√η_full · p_small/(1 − p_small)), so it is
largest for one-point children on small-variance outputs: at η_full
1e-6 on x or y it is about 1% of V per such split, which is how V_btw
overshot V_ij by 5% under revision 1. b_Δ is a floor only: V_btw is NOT
debiased by it (η_full is a power-of-ten measurement; subtracting it
could over-correct). The summed b_Δ over the splits actually accepted
is a product (2.5), so the residual noise bias left in V_btw is
visible. No new constant: η_full is measured per draw and is already a
product. η_full is machine-dependent (draw 1: 1e-6 on the author's
machine, 1e-7 on TACC), so every comparison in section 5 is made on
the machine that produced its baseline.

**What this does and does not save.** Lineages whose pilot means are
wrong but whose within-bin variance is small — the cascade — close
after the one split that shows no gain, instead of splitting until
exhausted or capped. Lineages with real within-bin variance keep paying
and stay open under both gates. At tight ε, where τ is small and most
splits pay, the rule is neutral: the cost is today's, no higher, and
the saving is at loose ε. That is the honest expectation and it is what
section 5 measures per ε.

### 2.5 Products under `'measured'`

* V_win_hat,c = (1/N) Σ_k p_k Var_k(psi0_c) over the final bins with
  more than one point — the pilot's own within-bin spread, no v;
  V_tot_hat = V_btw + V_win_hat. Both are model quantities and are
  labelled so in `products.py`'s column docstrings. Under
  `'predicted'` today's definitions (with v) stand, for byte identity.
* gain_ratio,c = Σ realized Δ_c / Σ predicted g_c over every check
  split, unchanged: it is now the calibration record of the pilot's
  predicted gains.
* New scalar products: `check_rule`; `n_closed_unpaid` (splits whose
  two children were closed because the split did not pay, 2.4 (b));
  `n_closed_unflagged` (children of a paying split closed by the
  pilot's flag alone, 2.4 (a)); `n_noise_floored` (splits where the
  deciding floor was n_Δ + b_Δ rather than τ on every output, i.e. the
  split was judged against the noise); per output `sum_b_delta_<o>`
  (Σ b_Δ,c over the splits accepted into V_btw: the residual noise bias
  left in V_btw,c, to be read against V_btw,c). `n_flagged`, `n_check_rounds`,
  `n_check_evals`, `n_level_splits`, `n_adjacency_splits`,
  `check_capped`, `bin_*`, `L0`, `L` as today. `bin_flagged` under
  `'measured'` = the bins still open when the cap stopped the check.
* `config.json` records `check_rule`.

## 3. Interface

`QIJ(..., check_rule='predicted')`; `scripts/run.py --check-rule
{predicted,measured}`; passed to `run_joint` as one argument; the two
continuation functions and the two kind pricers are selected ONCE
where the check is set up (R2), never by a flag inside the loop. Under
`ivqbins='marginal'` the argument is inert and recorded.

Folders and config. `products.dir_tag` and `run.py`'s config dict
carry `check_rule` and `gp_update` (section 7) as non-canonical
suffixes, so a measured run never targets a predicted folder:
`qij_gp_eps<ε>` (predicted, unchanged), `qij_gp_eps<ε>_measured`,
`qij_gp_eps<ε>_measured_gpupd`; `config.json` records both keys in
every qij folder from this build on, and the config guard compares
them. A folder without the keys is read as `predicted`/`False`.

## 4. Cost

Bounded above by today's cap. Expected: the first round is unchanged
(the seeds), every later round splits only lineages whose last split
realized a gain; at ε 0.2 (34 initial bins, 356 fits today) most of the
cascade should not happen; at ε 0.01 (205 fits, 16% of cost) the saving
is at most that share. Reported per ε, never as one number.

## 5. Validation (the author's allowance: 2 draws at ε 0.01 and ε 0.141421; ALL ON THE AUTHOR'S MACHINE — the author ruled no exploration on TACC; η_full is machine-dependent, so baseline and candidate must share one machine)

Runs are local, `--workers 3` each, under a hard time cap; nothing on
TACC. The TACC sweep products (V2) remain the record of the current
rule at TACC's η_full; the acceptance compares the two rules run on
the same machine.

Draws: the first two in-basin draws of the P1 sweep in index order (the
oracle's `search_failed` False); the coder records the two ids.

V1 (the one pass/fail): draw 1 at ε 0.01 run on the author's machine
twice — at the pre-change commit 8a93888 and at the new code with
`check_rule='predicted'` — compared column by column; every column
identical (the new columns of 2.5 excepted). A local run is compared
with a local run: the TACC products cannot serve here because η_full
differs by machine.

V2 (baseline, from products, before any run): for both draws at both
ε, per output: L0, L, n_flagged, n_check_evals, V_btw, V_ij (from the
ij products, P1-derived as the summary does), the realized shortfall
1 − V_btw/V_ij, and the draw's total evaluations.

V3 (the build): `check_rule='measured'` on the same two draws at both
ε, same table plus n_closed_unpaid, n_closed_unflagged,
n_noise_floored, sum_b_delta per output, gain_ratio.
Acceptance, per draw and output, with shortfall = 1 − V_btw/V_ij on
that draw and output (V_ij from the ij products, the validation's
reference only; the rule itself never reads it):

    shortfall^measured  ≤  max(ε, shortfall^predicted)     for every output,

AND n_check_evals at or below the baseline's on both draws at both ε
(revision 2 makes no split the `'predicted'` rule does not, so equality
at tight ε is the expected outcome, not a failure; the per-ε saving is
reported and the author judges it). The baseline for this test is
`check_rule='predicted'` run on the author's machine on the same draws
at the same ε (not the TACC products: η_full is machine-dependent, 2.4);
V_ij from the ij products is the common reference for both. At loose ε
this caps the new rule at ε against V_ij, the method's own promise (V2:
the current rule sits at 2–6% at ε 0.141, so a purely relative test
would have allowed a 20% shortfall); at ε 0.01 it lets the new rule
lose no more than the current code does on the same draw (V2: the
current rule is already at 0.14–0.96% there, within V_ij's own per-draw
scatter, so an absolute "< ε" test would fail on noise). ε itself is
the only constant. A predicted share is not evidence. If the test fails
anywhere, the report says on which output and by how much, and stops.

No other run: no smoke, no other method, no other draw (the author's
rule: checks run only the method under test on 1–2 draws).

Report: one markdown file with V1's verdict, V2/V3 side by side, the
two draw ids, commit, and wall time per stage.

## 6. Answers to questions a coder will ask

* Which τ for a child measured in round r? The τ'_c computed at the
  start of round r (L and V_btw before that round's splits), the same
  for every split of the round, as `core.rounds` does.
* Ties in the kind choice? Level wins at equality, as
  `decide_split_affine` does.
* psi_centered under joint? psi0_all − offset, as `run_joint` already
  forms it.
* η_full? The draw's measured `eta_full` product (A15); the
  reproducibility floor, not the declared eta.
* Does the gain test (2.4 (b)) use the measured outputs only? Yes, the
  `measured` index set `run_joint` already carries.
* Line budget: `joint.py` gains one continuation function and one
  v-free gp pricer and loses nothing under `'predicted'`; keep it within
  the standing budget by moving the three `decide_split_*` pricers into
  a small `core/joint_pricing.py` if needed (R2 applies there too).
* Nothing here changes the reported interval; that ruling is separate.

## 7. Change B: conditioning the pilot on the measured bin means (separate flag, default off; built and tested after A)

Requires `pilot='gp'` and `check_rule='measured'` (raise a clear error
otherwise). `QIJ(..., gp_update=False)`; `--gp-update`; recorded in
config.json and the products. Its only use is where to cut and how to
rank proposals (2.2). It never enters the gain test (2.4 (b)), the
pilot's flag (2.4 (a)), the
seeds (2.1), V_btw, or any reported variance; the posterior VARIANCE
is never recomputed and never read.

### 7.1 The observation model

The influence model's latent is the GP at the M_X_used prototype
points (the survey's design), from which every point prediction is
formed. A measured bin k with points S_k is observed through its
prototype composition: with n_kj the number of bin k's points whose
best-matching prototype is j, w_kj = n_kj / n_k (one `bincount` of
(bin label, bmu) per update, an L × M_X_used sparse matrix W), the
observation is

    y_kc = U_kc / a_c  ≈  Σ_j w_kj f_c(z_j) + e_kc ,   Var(e_kc) = (δU_c / a_c)²

with U_kc the bin's mass-centred measured derivative (the current
bins: measured or by mass balance), a_c the check's per-output scale
(fitted once at the first measurement, as today; it brings the measured
scale to the pilot's), and δU_c the finite-difference error of 2.4 for
the evaluation that produced the bin (for a mass-balance bin, the
error carried from its measured sibling, (p_small/p_large)·δU_c). This
prototype-weighted form is the design: it costs no kernel sum over N
points or over point pairs, it is consistent with how the survey
represents a receptive field, and its residual (the within-cell
variation of ψ) is the quantity the X-VQ already declares below its
tolerance. The exact bin-mean functional (kernel sums over the bin's
points) is NOT built.

### 7.2 The update

Hyperparameters (width, amplitude, noise floor λ, trend) fixed at the
fitted values: conditioning only, no refit. The prototype-level
posterior given the survey observations is the existing model; the
update conditions it further on the L observations of 7.1 — a standard
Gaussian update on the M_X_used-dimensional latent (one solve of size
M_X_used + L, or the equivalent rank-L update). Observations are the
CURRENT bins only (each bin once; a split parent's observation is
replaced by its two children's). The update runs after the initial
measurement and after every check round that measured at least one
split; its wall time is measured per update and summed
(`gp_update_wall`), and the number of updates is a product
(`n_gp_updates`). The conditioned prototype-level mean is pushed to the
N points with the same point-prediction machinery the pilot uses
(`_point_terms`), giving psi0_all^(cond) (q columns, measured outputs),
re-centred by the same offset.

### 7.3 Where the conditioned mean is used

For the NEXT round's proposals (2.2) and nothing else:
Ψ̃, psi_centered, each open bin's `ubar`/`m`/`var` are recomputed from
psi0_all^(cond); `two_means_split` cuts on the conditioned Ψ̃; the
adjacency geometry orders prototypes by the conditioned prototype-level
mean in place of I_proto; `level_gain_value` prices from the
conditioned means; the predicted gain g recorded for gain_ratio is the
conditioned one. The seeds (2.1) are computed BEFORE the first update.
V_win_hat (2.5) under `gp_update=True` is the conditioned mean's
within-bin spread, labelled as such.

### 7.4 Acceptance (after A has passed; the same two draws at the same two ε; the same machine as A's validation)

Per ε, per draw: check fits under A+B plus the update cost in
fit-equivalents (Σ gp_update_wall divided by the draw's own measured
wall time of one full-data evaluation) must be BELOW the check fits
under A alone, AND shortfall^(A+B) ≤ max(ε, shortfall^(A)) on every
output, shortfall = 1 − V_btw/V_ij on the same draw and output (the
same form as V3, with A in the baseline's place). Reported per ε with both terms shown separately;
pass only if it holds at both ε. If B passes at one ε and fails at the
other, it is reported so, and the author decides whether it is worth a
per-ε default; it is not made the default by the build. The revisit
criterion the author set: B is worth this test only if, under A, many
of the splits realize below τ' (n_closed_unpaid and the share of
below-τ' splits are in A's report).

### 7.5 Questions a coder will ask about B

* Which noise for a mass-balance bin? (p_small/p_large)·δU_c of its
  sibling's evaluation; every current bin then has a δ.
* What if a_c is 0 or non-finite for an output? That output's
  observations are skipped in the update (its column of the latent is
  left at the survey posterior); count it in `n_gp_update_skipped`.
* Does conditioning change the sigma stage's directions? No: the sigma
  stage reads the joint path's own per-point ψ̂ (`joint_psi_hat`, from
  bin U), which is untouched.
* Bit identity: with `gp_update=False` every product is byte-identical
  to A's.
