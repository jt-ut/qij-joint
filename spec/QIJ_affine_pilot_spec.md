# The affine pilot for QIJ (planner, 29 September 2026; one build)

Scope: the paper's method `qij` gains a second pilot estimate of the
influence, `pilot='affine'`, which becomes the default; the Gaussian
process pilot stays as `pilot='gp'`, bit-identical to today, as the
control. Under `'affine'` there is no posterior variance anywhere, so
the refinement's adjacency proposals are priced by the bridge test on
the connectivity graph's second-order cells instead. Nothing else in
the method changes: survey, I-VQ bins, full-data stencils, level-split
pricing, realized gains, split_gamma, the two-strike lineage close, the
ABC ingredients and the products are as built. Code standards:
`qij_joint_plan.md` section 5, verbatim in the build prompt.

Measured basis (exact offline simulator on the stored production survey,
cloudfil draws 0 and 1, scratchpad fvpilot/{maps,sim,tiers}): the
affine pilot's 17-bin share 0.89–0.95 (d0) / 0.79–0.89 (d1) against the
stored point GP's 0.31–0.89 / 0.11–0.56; with the refinement below it
reaches V_btw/V_ij 0.92–0.96 (d0) / 0.82–0.92 (d1) at about 11–13
refinement evaluations per output, against the production run's
0.82–1.00 / 0.79–0.92 at 600–1000. The bridge score ranks cells by true
within-cell variance at Spearman 0.68–0.80 on every output and draw.

Terms: **cell** = a receptive field of the X-VQ; **second-order cell**
(jk) = the points of cell j whose second-nearest prototype is k; **bridge
score** = the disagreement between neighbouring cells' affine fields on
their shared second-order cell. No other new terms.

## 1. Interface

`QIJ(..., pilot='affine')`, `pilot ∈ {'affine', 'gp'}`, default
`'affine'`. `gptrend`/`gpwidth` keep their meaning under `'gp'` and are
unused under `'affine'` (their products NaN/False there, as under a
failed fit today). `scripts/run.py` gains `--pilot {affine,gp}` for
`qij`, default `affine`. No other argument changes.

## 2. The affine pilot (new module `core/affine_pilot.py`, ≤ 250 lines)

Inputs: the X-VQ (`centers` in Z, `labels`, `p`, `conn`, `bmu`, `bmu2`),
the draw's points Z, the survey's prototype influences I_proto (M, q_full)
mass-centred as stored (the moments survey; the weights summing to N as
built).

### 2.1 The reconstruction

For each cell j, per output: μ_j = the mean of the cell's points in Z;
neighbours N(j) = the cells adjacent to j in `conn` (either direction);
the gradient g_j = the weighted least-squares solution of

    I_k − I_j ≈ g_jᵀ (μ_k − μ_j),   k ∈ N(j),   weight p_k,

(normal equations Σ_k p_k (μ_k−μ_j)(μ_k−μ_j)ᵀ g = Σ_k p_k (μ_k−μ_j)(I_k−I_j);
g_j = 0 when fewer than d neighbours or the normal matrix is singular at
relative tolerance 1e-12 · its largest eigenvalue). The pilot at point i
in cell j:

    ψ̂₀(z_i) = I_j + g_jᵀ (z_i − μ_j).

The cell average of ψ̂₀ over the cell's own points equals I_j exactly.
Implementation per E1/E2: neighbour sums by `conn`'s CSR structure and
`bincount`, one small solve per cell (vectorized over cells with
`numpy.linalg.solve` on a stacked (M, d, d) array, singular cells masked).

### 2.2 The bridge score

For each CADJ pair (j, k) with a non-empty second-order cell (jk) (points
i with `bmu[i] = j`, `bmu2[i] = k`), per output:

    Δ_jk = | mean_{i ∈ (jk)} ψ̂₀^{(j)}(z_i) − mean_{i ∈ (jk)} ψ̂₀^{(k)}(z_i) |,

where ψ̂₀^{(j)} is cell j's field evaluated at those points and ψ̂₀^{(k)}
cell k's field. Its mass m_jk = |(jk)|/N. Stored per pair (sparse, keyed
like `conn`), both Δ_jk and m_jk; the (jk) and (kj) cells are distinct
and both scored.

### 2.3 Products of the module

`psi0` (N, q_full); the pair table (j, k, m_jk, Δ_jk per output); per-cell
`g_j` (M, d, q_full). No uncertainty of any kind.

## 3. What changes in `qij.py` and the refinement

### 3.1 `qij.fit`

Under `'affine'`: after the survey, `psi0_all = affine_pilot.psi0(...)`
replaces `fit_influence_model` + `_psi0`; `sigma_all` is not computed
(the products that carry it are NaN); the GP width search does not run.
The I-VQ bins, the initial stencils, `rho`, the ABC curvature stage, and
everything downstream are unchanged and take `psi0_all` as they do
today. Under `'gp'` the code path is today's, bit-identical.

### 3.2 Refinement pricing (`core/refine.py`, `core/rounds.py`)

Today a bin's proposal is a level split when Var_k(ψ̂₀) > v_k, else an
adjacency split priced by `adjacency_gain_value(p_k, v_k, N, rho2)`,
with v_k from `batch_v` (the GP's within-bin posterior variance). Under
`'affine'` there is no v_k. Rule: for each open bin, BOTH proposals are
priced and the larger predicted gain is taken:

* level split: `level_split_gain` unchanged (priced from ψ̂₀ and the
  measured centred means, as today);
* adjacency split: the geometry of `_try_adjacency_split` unchanged;
  its predicted gain

      g_adj = rho2 · Σ_{(jk) ⊂ bin} m_jk · Δ_jk² / 4,

  the sum over the second-order cells whose points all lie in the bin
  (per output c of the bin's own coordinate, as level gains are); the
  bin's `v` field is not used.

Everything after the proposal is unchanged: the queue order by predicted
gain, the realized gain from the stencils, `split_gamma` = realized /
predicted, the two consecutive below-τ strikes, the evaluation guard,
the `'queue'` and `'rounds'` schedules. `batch_v` is not called under
`'affine'` (R2: two pricing functions, `adjacency_gain_value` for `'gp'`
and a new `bridge_gain_value` for `'affine'`, selected once where the
refinement is set up, never a flag inside the loop).

### 3.3 The joint path

`ivqbins='joint'` runs under either pilot. A bin's flag rule and its
check's split-kind rule (`core.joint.run_joint`) are priced by the same
two pricers as the marginal path, selected once where the check is set
up (R2), never re-chosen inside the loop:

* **Flag rule.** Under `'affine'`, `u` (the GP posterior's own
  contribution) is 0 throughout, so a bin is flagged when for some
  output c, `p_k*(U_kc - a_c*m_kc)^2 > eps*V_hat_c/L`; `a_c` is fitted
  once from every bin after the first measurement and held fixed, as
  today.
* **Kind rule.** A flagged bin under `'affine'` prices BOTH kinds, per
  output, at rho2 = 1, and compares the summed normalized gains: level
  from the bin's own `two_means_split` on Ψ̃, priced by
  `level_gain_value`; adjacency from `g_adj,c = bridge_gain_value
  (bridge_kc, 1) = bridge_kc/4`, `bridge_kc` = Sum over the bin's own
  second-order cells (jk) of `m_jk*Delta_jk^2` (the same per-leaf
  bridge sum the marginal path's `batch_bridge` computes, batched over
  every measured output at once). Level is chosen when its summed
  normalized gain is at least the adjacency one; the adjacency geometry
  itself is `_try_adjacency_split` on whichever output c* has the
  largest `g_adj,c/V_btw,c`, trying the remaining outputs in descending
  order if c*'s split is infeasible. An infeasible chosen kind falls
  back to the other; both infeasible closes the bin, as today.
* **V_win_hat.** With no posterior variance under `'affine'`,
  `V_win_hat_c = (1/N)*Sum_k p_k*Var_k(psi0_c)` over the final bins
  (the same expression as today's `p_k*(Var_k + v_k)` with `v_k` = 0
  throughout), and `V_tot_hat = V_btw + V_win_hat`.

Growth, the check's evaluation cap, mass balance, `split_gamma`, and
`joint_psi_hat` (which reads only `V_btw`/psi0, independent of the
pilot's posterior) are unchanged under either pilot.

## 4. Products

`QIJResult` and the `qij` rows gain `pilot` (string). Under `'affine'`:
`ell, lam, ell_bound, lam_bound, c, c_bound` NaN/False; `sigma` and
`bin_U`'s posterior columns NaN; new array product `bridge` (every draw:
j, k, m_jk, Δ_jk per measured output) and `cells` (every draw: j, p_j,
μ_j, g_j per measured output). Under `'gp'` every product is
byte-identical to today's.

## 5. Validation (small, per the author's rule)

1. `qij --pilot gp` on cloudfil draw 0: byte-identical products to the
   stored production run at runs/cloudfil_clean (after dropping timing
   columns and the new `pilot` column). The one pass/fail item.
2. `qij --pilot affine` on cloudfil draws 0 and 1 and on (pareto, shape)
   draw 0, at the defaults (the moments survey, `eps` 0.01): report
   V_btw/V_ij per output, refinement evaluations per output, rounds,
   the initial 17-bin share (from `rho` and the bins), wall time, beside
   the stored `gp` values for the same draws. Expected from the exact
   simulator: cloudfil d0 0.92–0.96, d1 0.82–0.92, about 11–13
   refinement evaluations per output; Pareto near 1.0.
3. The all-methods cloudfil smoke with `qij` under both pilots (the
   smoke's own draws only).

Nothing else. Reports to Research/QIJ_joint/qij_affine_validation/.

## 6. Questions a coder will ask

* **Which Z?** The quantizer's coordinates, exactly what `fit_xvq` was
  fitted on (identity for cloudfil; `vq_transform`'s output where one
  exists).
* **Is I_proto per measured output or full width?** Full width as
  stored; the pilot is computed for every output (cheap) and sliced by
  `measured` where `qij.fit` slices today.
* **Which I_j when a prototype's evaluation failed (NaN)?** That cell's
  g_j = 0 and its ψ̂₀ = NaN; downstream treats NaN ψ̂₀ as today's model
  does for a failed prototype (the existing `finite/mass_c` convention).
* **Does rho change?** No: rho is the FD-to-prediction scale from the
  initial bins' stencils against the pilot's bin means, computed as
  today with the new ψ̂₀.
* **Where do Δ_jk live during refinement?** Precomputed once per draw in
  the pilot module, passed into the refinement as an array keyed by
  pair; the per-bin sum is a masked sum over pairs whose (jk) points are
  all in the bin (bin membership by point index, as the bins are stored).
* **What about second-order cells straddling two bins?** They contribute
  to neither bin's adjacency gain.
* **Line budgets:** `core/affine_pilot.py` ≤ 250; `refine.py` grows only
  by `bridge_gain_value` and the pricer selection; `qij.py` by the
  switch.
