# LBG in the joint growth: a Lloyd run after every round (planner, 30 September 2026; one build; validated AFTER the measured check, QIJ_joint_check_measured_spec.md)

**NOT ADOPTED (30 September 2026; reverted on main at 17363f0, code
identical to 79d98c5; record Research/QIJ_joint/analysis/
growth_lbg_validation.txt).** Built as specified (skip-Lloyd guard,
M_ref seeded start) and validated on the author's machine under
`check_rule='measured'`, tree → lbg: ε 0.01 draw 1 L0 312 → 393, net
cost 828 → 1006 fit-equivalents; draw 7 375 → 356, 870 → 913; ε 0.141
draw 1 55 → 40, 193 → 145 but the position angle's realized shortfall
15% > ε; draw 7 54 → 39, 207 → 155. Lloyd time was 1–4.5 s, not the
issue. Why: Lloyd equalizes the bins' shares, the split rule "every
bin over ε/L" then flags nearly every bin, and L doubles per round —
an overshoot in COUNT at the same predicted share; Lloyd minimizes the
SUM of shares while the tolerance is on the MAX, and the tree's
selective, unequal splits are the better optimizer for that
criterion. The section-1 measurement (one draw of the older P2 pilot,
473 → 339) did not carry to P1 — one draw of a stale pilot, against
the author's three questions. The finding that decides it: at ε 0.141
the realized shortfall on the position angle was ~1.3× the pilot's
predicted worst share; the tree lands at ~½ ε in predicted share, and
that margin is what makes the realized tolerance hold on every draw
so far. Under the measured check an under-predicted bin is never
flagged and never repaired, so any growth that lands nearer the
tolerance spends the margin. The greedy schedule (split only as many
of the largest-share bins as the predicted shares need) is NOT
pursued; if ever revisited it is measured offline first, growth alone
on the current pilot of draws 1 and 7 at both ε against the tree, and
built only if it wins there. The `'tree'` removal once written into
section 5 is withdrawn; growth stays as coded. The rest of this
document is the record of what was built.

Scope. One change to `core/joint.py grow` behind a new argument
`growth`: `'tree'` = today's growth, byte-identical products; `'lbg'` =
this document. Nothing else in the joint stage, the check, the sigma
stage, the marginal path or stage 1 changes. Code standards:
`qij_joint_plan.md` section 5, verbatim in the build prompt. No check
is added. Terms as the glossary: bins, the pilot, points; never
"rows"/"field".

## 1. Why (measured)

Growth as coded splits, in each round, every bin whose worst-output
predicted share exceeds ε/L, and runs Lloyd once, after the last round.
On the P1 sweep (500 draws per ε) it stops at a worst-output predicted
share of 0.4–0.6 ε at the median (0.6–0.8 at p90) — the last round
overshoots the tolerance by about two in bins, and each initial bin
costs two full-data stencil evaluations. The tree is a greedy
initializer; the object it approximates is the partition that meets
the tolerance at the smallest count, which is what Linde–Buzo–Gray
finds by running Lloyd at every count, not only at the end. Measured
on this machine, draw 0 of the older P2 pilot (the geometry only), no
evaluations:

| ε | growth as coded | Lloyd after every round |
|---|---|---|
| 0.01 | 15.6 s, L0 473, worst share 0.38 ε | 40.1 s, L 339 (−28%), worst share 0.72 ε |
| 0.141 | 3.8 s, L0 29, worst share 0.65 ε | 4.4 s, L 23 (−21%), worst share 0.68 ε |

One Lloyd run at L0 473 takes 0.4 s; the seconds in both columns are
the per-bin two-means loop growth already pays. At ε 0.01 the change
is 25 s of numpy for 268 fewer stencil evaluations (about 18 minutes
at the measured full-data rate). The partition ends nearer the
tolerance, so the check inherits less headroom; section 4 measures
that with the measured check in force.

## 2. The rule

Under `growth='lbg'` growth does not start from one bin. The start is
L_init = M_ref = ⌈√(κ_ref/ε)⌉ (κ_ref = 2.7, the same M_ref the cost rule
and the marginal path's initial bins use: the scalar quantizer's count
at the tolerance, 17 at ε 0.01, 5 at ε 0.141), a lower bound on the
count the joint partition needs since every output's own within share
must meet ε. Seeding, deterministic and the marginal path's own
(`ivq.kmeans_1d`'s init): the projection of every point's Ψ̃ row on
the first principal axis of Ψ̃ (eigh of its q × q covariance; sign
fixed by the largest-magnitude loading positive), the L_init
mid-quantiles (k + ½)/L_init of the DISTINCT projected values as
initial centroids on that axis (a point's centroid = its nearest
seed in projection), then one `_lloyd_all` run over all points in the
full Ψ̃ space. Product: `L_init`. Then the loop below, from that
partition. Under `'tree'` the start is one bin, as today.

A growth round is then today's round with one line added at its end:

1. (w, S) from the current labels; stop if every S_c ≤ ε; the cap test
   as today (L ≥ M_X_used → `growth_capped`).
2. Split every bin with max_c w_kc > ε/L by `two_means_split` on its
   own Ψ̃ rows (the cap's `room` rule as today); L ← L + successful
   splits.
3. NEW: `labels, L = _lloyd_all(psi_tilde, labels, L)` — one full Lloyd
   run of all L centroids over every point, from the current labels,
   to convergence or 100 iterations, an emptied centroid dropped and
   the rest relabelled (exactly the function today's final pass calls).
4. Next round.

Termination guard (coder's finding, 30 September). Step 3 can empty
centroids the round's splits just created, so L can fall back to or
below the round's starting count, and grow's loop has no round limit:
a round Lloyd undoes could repeat forever. Rule: if after step 3 L ≤
the round's starting L (no net growth), the NEXT round skips step 3
(a tree round, whose splits are never undone within it); Lloyd resumes
in the round after. Growth then strictly increases L at least every
other round until the tolerance or the cap, so it terminates, growth
never ends above the tolerance for this reason, and no new stopping
condition or "stalled" state exists. Product: `n_lloyd_skipped` (int;
0 under `'tree'`, where the guard cannot trigger).

The stop test of the next round therefore sees the Lloyd-optimized
partition (or, after a skipped round, the tree partition, which the
following round's Lloyd then optimizes). The end-of-growth Lloyd pass of today is then the last
round's step 3: it is NOT run a second time. `S_pred_pre_lloyd` under
`'lbg'` is the share before the last round's Lloyd run; `S_pred` after
it, as today. Deterministic, no seed, in-process numpy, never on the
pool — as growth is today. The reweighted (minimax) Lloyd variant and
any search over the count are NOT in this build.

Under `'tree'` the function is today's, byte for byte.

## 3. Interface and products

`QIJ(..., growth='tree')`; `scripts/run.py --growth {tree,lbg}`; passed
to `run_joint` → `grow` as one argument; inert and recorded under
`ivqbins='marginal'`. `products.dir_tag` carries `_lbg` as a
non-canonical suffix (after the `_measured` / `_measured_gpupd`
suffixes of the check spec, e.g. `qij_gp_eps0.01_measured_lbg`);
`config.json` records `growth`; a keyless folder reads as `'tree'`.
Products: `growth` (str), `growth_wall` (seconds spent in `grow`, new),
and today's `L0`, `n_growth_rounds`, `growth_capped`, `S_pred`,
`S_pred_pre_lloyd`.

## 4. Validation (after the measured check has passed; the same two draws, 1 and 7, at ε 0.01 and 0.1414213562373095; all on the author's machine, `--workers 3`, hard time cap; nothing on TACC)

V1 (the one pass/fail): draw 1 at ε 0.01 with `growth='tree'` at the
new code vs the pre-change commit, column by column, the new columns
excepted; identical.

V2 (growth alone, no evaluation): for both draws at both ε, `'tree'`
and `'lbg'`: L0, n_growth_rounds, S_pred per output (as a fraction of
ε), growth_wall. Reported first; the expected picture is L0 lower by
20–30% and the worst share nearer ε.

V3 (the whole joint stage with the measured check in force,
`check_rule='measured'`): `'lbg'` vs `'tree'` on both draws at both ε.
Acceptance, per draw × output, with shortfall = 1 − V_btw/V_ij (V_ij
from the ij products as the common reference):

    shortfall_lbg ≤ max(ε, shortfall_tree)                                   on every output,
    2·L0 + n_check_evals + growth_wall / s_fit   lower under 'lbg'           on both draws at both ε,

with s_fit the draw's own measured seconds per full-data evaluation
(the stencil stage's wall time over its evaluation count, on the same
run), so the extra numpy time of the Lloyd runs is charged in
fit-equivalents: at this estimator's ~0.25 s per fit, the 25 s measured
in section 1 is ~100 fit-equivalents, the same order as the 2·134
stencils it saved at ε 0.01, so the test must be net or it is not a
cost test. The wall-time totals of the whole joint stage are reported
beside the counts. The second line is the whole point: the initial
stencils saved must not be handed back as check fits or as growth
time. The per-ε saving is reported and the author judges it. If either line fails anywhere, the report says where
and by how much, and stops.

Report: one markdown file with V1's verdict, V2, V3 side by side, the
two draw ids, commit, wall time per stage.

## 5. Questions a coder will ask

* Is the M_ref start ever too many bins? Only if the joint partition
  needed fewer than the scalar quantizer's count at the same tolerance,
  which cannot happen: every output's own within share must meet ε.
  If a seed's cell is empty after the projection assignment, Lloyd
  drops it and L_init is recorded as what survived.
* (Withdrawn, 30 September: the post-acceptance removal of `'tree'`
  and its `room` rule and `S_pred_pre_lloyd` product is void, since
  `'lbg'` was not adopted; growth stays as coded.)
* Can Lloyd undo a split? It can empty a centroid, which is dropped;
  L then falls and the next round's flags decide. That is the
  optimizer working; `growth_capped` logic is unchanged since the cap
  is tested at the round's start.
* Does the cap's `room` rule change? No: it limits the round's splits
  as today; Lloyd runs after them.
* Interaction with the measured check (2.4 (b) of the check spec)?
  None in code; only the initial bins differ. V3 measures the effect.
* Line budget: one call added inside the loop and one branch on
  `growth`; no new module.
