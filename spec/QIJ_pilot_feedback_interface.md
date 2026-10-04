# Pilot feedback — shared build interface (5 parallel agents)

Branch `pilot-feedback`, worktree `JT_Py_Pkgs/qij_joint_feedback` (off main 1efe43e). Read this file whole and
`spec/QIJ_pilot_feedback_spec.md` (design; this file OVERRIDES it where they differ). Signatures, names and file
ownership here are FIXED. Need something not here: state the assumption in your report; never edit another
agent's file. No commits. Python3.9, `PYTHONPATH=src`, `OMP_NUM_THREADS=1`; Bash writes in the worktree need
`dangerouslyDisableSandbox: true`. Any check: ONE draw, hard cap 10 min per run, never run oracle/boot/ij.

## 0. The method (as decided by the author)

After each round of the unified loop (`core/joint.py:run_joint`), with the round's splits measured:
1. TRIGGER (per output c): excess_c = sum_{s in round} (D_sc - floor_sc) - SW_c, where SW_c = sum over the
   leaves standing at the START of the round of W_lc (RAW pilot within, kappa NOT applied; W_all at round start)
   and floor = the split's n_Delta + b_Delta (already in `info['floor']`). ratio_c = excess_c / (eps * Vtot_c),
   Vtot_c = V_btw_c + V_win_c at the START of the round (the loop's own `V_tot`). Fire if ratio_c > 1 for ANY c.
2. WHERE (CHILD-LEVEL, author's ruling 3 Oct): per split s of the round and per FIRING output c, its excess
   e_sc = max(0, D_sc - floor_sc - W_parent_sc) (W_parent: the parent's raw W at split time) is shared between the
   split's two children by how far each child's REALIZED term exceeds its PREDICTED one:
   R_{s,a,c} = n_a (U_a - U_k)^2 / N^2 (measured child mean U_a, parent's U_k: the root's 0 or the parent's recorded U),
   P_{s,a,c} = n_a (m_a - U_k)^2 / N^2 (m_a = the child's pre-update mean of psi_hat, the prediction G was built from),
   weight w_{s,a,c} = max(0, R - P); if both children's weights are 0, use w = R instead (both by realized term);
   share_{s,a,c} = w_a / (w_a + w_b) (if that is also 0, split equally by n). Then within each child by prototype
   membership: cell score A_jc = sum_s sum_{a in children(s)} e_sc * share_{s,a,c} * n_{s,a,j} / n_{s,a}.
   Per firing c: sort cells by A_jc descending, take the smallest prefix whose cumulative score >=
   min(excess_c, sum_j A_jc). Selected cells = union over firing c. (Why: at round 1 the only split is the root,
   whose parent is every point; parent-level attribution would select all cells. The excess comes from the
   children's measured means missing their predictions, so it is located in the child that missed.)
3. REFINE each selected cell: split it in 2 by k-means on Z restricted to its points (k = 2). Cell j keeps
   prototype id j for its first sub-cell; the second sub-cell is APPENDED (id M_old, M_old+1, ... in ascending
   order of the selected j). Only the selected cells' points change bmu.
4. RE-SURVEY: rebuild the moments survey rows for the new codebook; ONE new quantized fit
   theta_Q' = T(rows', omega0', start=theta_Q_old, eta=eta_rows); measure influences ONLY for the prototypes of
   refined cells (both sub-cells of each, i.e. old id j and its appended sibling) against theta_Q', exactly as
   `xvq.prototype_influences` measures (same delta_f = forward_step(sv.eta_Q) when takes_start else
   forward_step(eta), same step_parameter, same `_field_step_weights`, start=theta_Q', eta=eta_rows); all other
   prototypes keep their old I; then re-centre I mass-weighted over finite entries with the NEW p (as
   `prototype_influences` does at its end).
5. GP: re-solve the influence model with FROZEN hyperparameters on the new design (section 3).
6. STATE: psi0_new = `_psi0(model_new, Z)`; rebuild psi_hat by FULL REPLAY (section 4).
7. Continue the loop: kappa/se from the stored D and the REPLAYED G; recomputed W; same ranking, rounds, stop;
   std0 and L_max stay the ORIGINAL values. The pass may fire again at any later round.
`feedback=False` (the default) must leave every product BIT-IDENTICAL to today (no extra evaluations, no RNG use).

## 1. `src/qij_joint/core/feedback.py` — AGENT A (new file; pure numpy, no estimator calls)

```python
def round_trigger(D_round, floor_round, SW, Vtot, eps):
    """D_round, floor_round: (S_r, q); SW, Vtot: (q,). Returns (fire (q,) bool, ratio (q,), excess (q,))."""
def attribute_cells(D_round, floor_round, W_parent_round, child_indices, child_R, child_P, bmu, fire, excess):
    """child_indices: list (S_r) of (idx_small, idx_large) int arrays; child_R, child_P: (S_r, 2, q) the children's
    realized and predicted terms (section 0 step 2), child order (small, large); bmu: (N,) current prototype of every
    point. Returns (cells: sorted int array, scores: dict output_index -> (M,) score array)."""
```
Self-check (report numbers): rebuild the inputs from the stored products of
`Research/QIJ_joint/runs/scratch/feedback_check2/cloudfil_u_p2_N10000/qij_eps0.01_widthlocal_massfit_nosigma/s00045.*`
(splits: D_*, floor_*, W_parent_*, n_parent, parent/children ids, round; rounds: V_win_*, kappa_* at round start
— SW_c = V_win_c / kappa_c of the round's row r-1, Vtot from cumulative earlier D + V_win; points: bmu, bin_label;
bin_label k = rank of the final leaf's node id; a node's points = union of its final leaves) and reproduce: draw-45
fires at every round (round 1 ratio ~6048 on p2_pa); the bead `.../feedback_check2/cloudfil_p1_N10000/...s00001.*`
never fires (max ratio 0.32 at round 10); and report draw-45's selected cell count at round 1 and at round 5.

## 2. `src/qij_joint/core/xvq_refine.py` — AGENT B (new file)

```python
def rebuild_conn(bmu, bmu2, M):
    """(M, M) csr adjacency built from the (bmu, bmu2) pairs exactly as vqlp's CADJ is defined. VERIFY on one draw that
    it reproduces fit_xvq's conn (after its live-prototype reindexing) exactly; if not, find vqlp's definition."""
def refine_cells(Z, xvq, cells, seed):
    """k = 2 k-means on Z[xvq.bmu == j] for each j in cells (deterministic in seed; a cell with < 2 distinct points is
    skipped and reported). Returns (xvq_new: XVQ with centers/p/bmu/labels/bmu2/conn/M_used updated, M_requested
    unchanged; refined: list of (j, j_new) prototype id pairs). bmu2 recomputed for EVERY point (nearest live prototype
    other than its bmu); conn = rebuild_conn(...)."""
def resurvey(counter, X, xvq_new, refined, I_proto_old, theta_Q_old, sv_old, eta, pool, eta_rows):
    """Section 0 step 4. I_proto_old: (M_old, q_full) as stage 1 returned it (already centred). Returns
    (theta_Q_new (q_full,), I_proto_new (M_new, q_full), sv_new: SurveyRows, n_evals, n_rows, busy_delta)."""
```
Self-check: on draw 45 of cloudfil_u (X = datasets.cloudfil_G_U_P3_v1(10000, 45), stage-1 via `xvq.run_xvq` with the
study settings: survey='moments', quantized_start='full-data' (theta_hat from a cold fit), M from
`xvq.cost_rule_M(N, 7, 0.01)`), refine 10 cells of your choice; report rebuild_conn's equality check, the new
M_used, theta_Q' vs theta_Q max |diff|, the n_evals, and that unrefined prototypes' I are unchanged before re-centring.

## 3. `src/qij_joint/core/influence_model.py` — AGENT C (add ONE public function; touch nothing else in the file)

```python
def refit_frozen(model, Z, xvq_new, I_proto_new, theta_Q_new, eta):
    """The same InfluenceModel re-solved on xvq_new's prototypes with FROZEN hyperparameters: gptrend, gpwidth,
    fit_weights, whitening, width (global) or c (local), lam, s2, jitter, constant_path and const_value are kept;
    rebuilt: centers, h (CONN spacing from xvq_new.conn, as at fit), h_design, bmu (xvq_new.bmu), alpha, beta,
    g_chol and every other per-design solve product the predictors read; offset/median_sigma/p95_sigma recomputed
    as at fit. I_proto_new: (M_new, q) at the model's own (measured) width. No width search, no lam profiling."""
```
KEY self-check: `refit_frozen(model, Z, xvq, I_proto, theta_Q, eta)` with the ORIGINAL inputs must reproduce the
original model's `_psi0`/`_uncertainty` over Z to <= 1e-10 relative (draw 45, study settings, measured = the 7 p2
outputs). Then refine nothing else; report timing.

## 4. `src/qij_joint/core/joint.py` — AGENT D

- `run_joint(..., feedback: bool = False, refiner=None)` (keyword-only, after L_max). `refiner(cells, bmu_current)`
  is supplied by qij.py (agent E) and returns `dict(psi0=(N,q) new pilot predictions, survey units, measured width;
  offset=(q,); bmu=(N,) the new codebook's bmu; info=dict)` or None (failure: log it, keep the current state,
  continue without refining).
- Store per split a REPLAY RECORD: parent id, child ids, idx_small, idx_large, U_small, U_large, t_small, t_large,
  and for the attribution the parent's U (U_k) and the children's pre-update means m_pre_small, m_pre_large; the
  hook builds child_R / child_P (section 0 step 2) from them and calls attribute_cells with child_indices.
- `replay_state(psi0, offset, records, N, q, eta, theta_abs) -> (psi_hat, per_split dict(G=(S,q), W_parent=(S,q)),
  per_leaf dict(id -> m, ebar, var_hat))`: psi_hat = psi0 - offset; root centring; then for each record in original
  order: m_pre_small/large = psi_hat means over the children BEFORE update -> recompute G exactly as at split time
  (the parent's U: the root's 0 or the parent's recorded U), W_parent = (n_parent/N) * var(psi_hat over the parent's
  points) / N BEFORE the children's update; then `_apply_update` on small then large with the stored U and t, the
  same n_update counters (use scratch counters; the run's own counters keep their original values); then the
  children's m (pre-update means), ebar = U - m, var_hat after update.
- After each round when `feedback`: build the trigger inputs (section 0), call feedback.round_trigger; if any fires:
  cells = feedback.attribute_cells(...); out = refiner(cells, current bmu); if out: replay, overwrite split_G /
  split_W_parent and every live leaf's m/ebar/var_hat from the replay, set the current bmu to out['bmu'], append a
  pass record (round, ratio (q,), fired outputs, cells, out['info']) to a `passes` list. kappa/se next iteration from
  stored D and replayed G (they are recomputed from split_D/split_G every iteration already).
- JointResult gains `passes: list` (empty when feedback is off) and `feedback: bool`. `feedback=False` path:
  byte-identical (verify: bead cloudfil p1 draw 1 eps 0.01 L 463 n_evals 924 and identical V_tot_hat).
- Self-check replay_state WITHOUT a refiner: after round k of a normal run, replay with the ORIGINAL psi0 must
  reproduce psi_hat to <= 1e-12 and the recorded G / W_parent exactly (draw 45, eps 0.01).

## 5. `qij.py`, `result.py`, `pipeline.py`, `products.py`, `scripts/run.py` — AGENT E

- `QIJ(..., feedback: bool = False)`; `scripts/run.py --feedback` (qij only); `products` folder tag appends
  `_feedback` when on (config.json records it).
- In `QIJ.fit`, when feedback: build `refiner(cells, bmu)` as a closure over stage-1 state (Z, X, xvq, I_proto
  (full width), theta_Q, sv, model, measured, counter, pool, eta, eta_rows, seed): calls
  `xvq_refine.refine_cells(Z, xvq, cells, seed)` -> `xvq_refine.resurvey(...)` -> `influence_model.refit_frozen(model,
  Z, xvq_new, I_proto_new[:, measured], theta_Q_new[measured], eta)` -> psi0 = `_psi0(model_new, Z)`; updates the
  closure's own xvq/I_proto/theta_Q/sv/model for the next pass; returns dict(psi0, offset=model_new.offset,
  bmu=xvq_new.bmu, info=dict(n_cells, cells, prototypes_added, evals, rows, wall)). Wrap failures -> None.
- Accounting: the refiner's evaluations happen inside run_joint's window; report them as `evals_prototype_refine`
  / `rows_prototype_refine` and SUBTRACT them from the second stage's count, so `n_evals` keeps meaning the loop's
  own measurements. Total size-N includes them.
- Products: scalar row adds `feedback`, `n_passes`, `evals_prototype_refine`, `rows_prototype_refine`, `M_X_final`;
  a new per-draw table `<s>.passes.parquet` (one row per pass: round, ratio_<o>, fired_<o>, n_cells, prototypes_added,
  evals, rows, wall); the `M_X` scalar stays the stage-1 value.
- Self-check: with feedback off, run.py on bead cloudfil p1 draw 1 eps 0.01 produces an s00001.parquet whose every
  non-wall-time column equals the committed main's (run main in its own worktree? no — compare against
  `Research/QIJ_joint/runs/scratch/feedback_check2/cloudfil_p1_N10000/qij_eps0.01_widthlocal_massfit_nosigma/s00001.parquet`).
  The full feedback=True run is the coordinator's validation, not yours (you may smoke the wiring with a stub).

## 6. REVISION 2 (4 Oct, author-approved) — INTRA-CELL WHERE rule. Supersedes section 0 steps 2–3 and the
## `attribute_cells` / `refine_cells` / `refiner` signatures above. Trigger (step 1), resurvey's measurement and
## centring (step 4, with the M3 fix), GP (5), replay (6), continuation (7) are unchanged.

Why: child-level shares are not identifiable (conservation gives R_s/P_s = R_l/P_l = D/G, so shares = sizes), and
any attribution spread over parents larger than a cell selects every cell at round 1. A split localizes its excess
only to its PARENT; that is evidence about the SURVEY only when the parent lies inside ONE X-VQ cell.

2'. WHERE. Consider the splits made SINCE THE LAST PASS (all splits so far, for the first pass; a pass consumes its
    evidence). For split s with parent point set P_s (= idx_small ∪ idx_large) under the CURRENT bmu: if all of P_s
    share one prototype j, it is INTRA-CELL; for every output c that FIRED this round,
    e_sc = max(0, D_sc − floor_sc − W_parent_sc) (W_parent as stored/replayed; the trigger's own per-split term), and
    A_jc += e_sc. Splits whose parent spans ≥ 2 cells are attributed to no cell. Cell j is SELECTED iff
    Σ_c A_jc > 0. For each selected j also record n_min_child_j = the smallest child (min(n_small, n_large)) over its
    intra-cell splits with e > 0 on a fired output.
3'. REFINE selected cell j. Q_j = points with bmu j that lie in a CURRENT leaf (live or closed, i.e. a node with no
    children) of ≥ 2 points (singleton leaves are measured exactly; they get no prototype of their own).
    k_j = min(ceil(|Q_j| / n_min_child_j), number of distinct rows of Z[Q_j]). If k_j < 2: skip j (reported).
    k-means with k_j on Z[Q_j] (vqlp, deterministic in seed); ALL points with bmu j (Q_j and the singletons) are then
    assigned to the nearest of the k_j new centres. Sub-cell 0 keeps id j; sub-cells 1..k_j−1 are APPENDED (ids in
    ascending j, then sub-cell order). Any empty sub-cell or k-means failure → skip j, no mutation. Only cell-j points
    move; bmu2 recomputed for every point; conn = rebuild_conn. Resurvey measures ALL k_j sub-cells of every refined
    j (ids j and its appended siblings): cost Σ k_j evaluations on the new survey rows.

Signatures (replacing sections 1/2/4/5 where they differ):
```python
# core/feedback.py
def attribute_intracell(splits, bmu, fire):
    """splits: list of dict(idx_small, idx_large, D (q,), floor (q,), W_parent (q,)) for the splits since the last
    pass; bmu (N,); fire (q,) bool. Returns plan: list of (j, n_min_child_j) sorted by j, and scores dict j -> (q,) A_j."""
# core/xvq_refine.py
def refine_cells(Z, xvq, plan, seed):
    """plan: list of (j, Q_j int array, k_j). Returns (xvq_new, refined: list of (j, [j, j_new1, ...]), skipped: list)."""
# resurvey(..., refined, ...) takes refined in that form (all ids of each refined j are measured).
# joint.py: refiner(plan_cells, bmu_current, leaf_size) where plan_cells = list of (j, n_min_child_j) and
#   leaf_size (N,) = size of each point's current leaf; the refiner builds Q_j and k_j (rule 3').
```
`attribute_cells` (child-level) is removed. Pass record adds: n_intracell_splits, cells selected, sum k_j, skipped.
