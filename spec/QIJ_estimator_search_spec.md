# The mixture estimator's cold search, canonized (planner, 29 September 2026; one build)

Scope: the cold fit of `GMM2D` (and through it `P2Mixture`): the search
that produces θ̂ from the data with no start. The local optimizer of
`QIJ_estimator_fit_spec.md` (EM with Aitken stopping, safeguarded
acceleration, trust-region finish, status) is UNCHANGED and is what
"run to convergence" means below. Continuations (`start` given) are
unchanged and never run the search or the split-and-merge stage. The
search audit of commit 0e18d8f is unchanged and is this document's
acceptance instrument. Notation: K components, p = (K − 1) + 5K
parameters (59 at K = 10), N points with weights ω summing to N,
ℓ = the penalized log-likelihood per unit weight (`_penalized_ll`), η =
the estimator's reproducibility (`GMM2D.eta`, √machine-ε by default),
r_ik the responsibility of component k for point i at the current
parameters.

## 1. What is established

The audit (eight draws, commit 0e18d8f) shows the cold multistart
converging, on every audited draw, to a likelihood 17–177 nats per
10 000 points below the maximum reached by continuing from the truth.
The wrong basin is the same on every draw: two adjacent filament beads
merged into one component and the freed component spent on a spurious
tiny one. P2's outputs are nearly unaffected on most draws and moved
strongly on some (draws 45, 3, 39). Every fit is converged; the search
discards, or never forms, the start that separates the beads.

Best practice for the global search of a mixture likelihood is a fixed
set of three things: (i) many starts compared after runs long enough
for the comparison to mean something (Biernacki, Celeux and Govaert
2003); (ii) a structural move that repairs the characteristic failure
of merged and spurious components, split-and-merge EM (Ueda, Nakano,
Ghahramani and Hinton 2000, Neural Computation 12:2109); (iii) an audit
that records when the search fell short. The code has (iii) and a
weakened (i): a fixed 25-iteration screen. This document adds (ii) and
replaces the fixed budget by a rule. There is no fourth item.

## 2. The search

### 2.1 Starts

As built: the k-means starts and the greedy residual-insertion starts
(A12), their number and construction unchanged. Each start is a full
parameter vector (pis, mus, Ss).

### 2.2 Racing screen (replaces the fixed 25-iteration screen)

All starts run PLAIN EM steps (`_em_step`, no acceleration: the Aitken
projection below assumes EM's linear convergence, which acceleration
breaks) in lockstep, one iteration each per round, on the full data at
the base weights. After each round, for each live start with at least
three recorded values ℓ_{k−2}, ℓ_{k−1}, ℓ_k:

    c = (ℓ_k − ℓ_{k−1}) / (ℓ_{k−1} − ℓ_{k−2})
    ℓ_∞ = ℓ_{k−1} + (ℓ_k − ℓ_{k−1}) / (1 − c)   if 0 < c < 1
    ℓ_∞ = +∞                                    if c ≥ 1  (not yet in the linear regime: cannot be discarded)
    ℓ_∞ = ℓ_k                                   if c ≤ 0  (no monotone trend: no extrapolation)
    u   = ℓ_∞ − ℓ_k                             (the start's own claimed remaining gain; +∞ or 0 in the two special cases)

Let ℓ_best = max over live starts of ℓ_k (the best ACHIEVED value, not a
projection). A start is DISCARDED after the round when

    ℓ_∞ + u < ℓ_best,   i.e.  2 ℓ_∞ − ℓ_k < ℓ_best,

its most optimistic finish, credited twice, still below what another
start has already reached. A start with fewer than three values, or
with c ≥ 1, is never discarded. Discarded starts are not resumed.

The screen ENDS when the number of live starts is at most k_keep = 3
(the number the code already runs to convergence), or when 20 × p
rounds have been run (safety limit; status `screen_cap` recorded, the
k_keep best by ℓ_k kept). The survivors then run to convergence by the
local optimizer (accelerated EM, Aitken stop, finish), each labelled as
built, and the one with the largest ℓ is the winner. Products:
`n_starts_screened` (starts entered), `screen_rounds` (rounds run),
`n_survivors`.

### 2.3 Split-and-merge finish (new; cold fits only)

Input: the winner of 2.2, converged, with responsibilities r_ik and
penalized likelihood ℓ_cur. Repeat the following cycle.

**Candidates.**

- MERGE candidates: every pair (i, j), i < j, ranked by
  J_merge(i, j) = (r_i · r_j) / (‖r_i‖ ‖r_j‖), with r_k the length-N
  responsibility vector of component k (Ueda et al., eq. 12): largest
  first.
- SPLIT candidates: every component k ranked by the local
  Kullback–Leibler divergence (Ueda et al., eq. 13, on the sample):
  f_k(x_i) = r_ik / Σ_i r_ik (the empirical density around component k),
  p_k(x_i) = φ(x_i; μ_k, Σ_k) / Σ_i φ(x_i; μ_k, Σ_k) (the component's
  density normalized over the sample), J_split(k) = Σ_i f_k(x_i)
  log( f_k(x_i) / p_k(x_i) ): largest first.
- PROPOSALS: the triples (i, j, k) with k ∉ {i, j}, ordered merge rank
  first and split rank second (for the top merge pair, the split
  candidates in order; then the next merge pair). The first C = 5 are
  tried per cycle. C is the published breadth (Ueda et al. §3), not a
  tolerance; recorded in the products.

**A proposal (i, j, k).** From the current parameters:

- Merge i and j into i′: π_{i′} = π_i + π_j; μ_{i′} = (π_i μ_i + π_j μ_j) / π_{i′};
  Σ_{i′} = [π_i (Σ_i + μ_i μ_iᵀ) + π_j (Σ_j + μ_j μ_jᵀ)] / π_{i′} − μ_{i′} μ_{i′}ᵀ.
- Split k into k′ and k″: π = π_k / 2 each; μ = μ_k ± (√λ₁ / 2) v₁ with
  (λ₁, v₁) the leading eigenpair of Σ_k; Σ = Σ_k for both.
- K is unchanged (one pair merged, one component split). A proposal
  whose merged covariance is not positive definite, or whose λ₁ is 0,
  is skipped.
- PARTIAL EM (Ueda et al. §2.2): with every component other than the
  three new ones held fixed, and each point's responsibility mass for
  those others frozen at its pre-proposal value, run EM on the three
  new components only: E-step assigns the mass 1 − Σ_{others} r_i to
  the three in proportion to π φ; M-step (penalized, as `_m_step`)
  updates only the three. Stop by the local optimizer's Aitken rule on
  the partial likelihood, cap 20 × p iterations.
- Then FULL EM on all K components to convergence, then the finish
  (the local optimizer, unchanged), giving ℓ_new. The penalty is in
  force throughout.

**Acceptance.** The proposals are tried in order; the FIRST with
ℓ_new > ℓ_cur + η is accepted: its parameters become the current ones,
`n_smem_accepted` increments, and a new cycle begins from step
"Candidates". If none of the C proposals is accepted, the stage ends.
Safety limit: p acceptances (status `smem_cap`).

**Cost note.** Each tried proposal costs one partial EM plus one full
EM to convergence; at most C per cycle. Recorded: `n_smem_tried`,
`n_smem_accepted`, `smem_wall_time`.

### 2.4 The returned fit

The current parameters after 2.3, labelled as built (canonical sort, or
`reference` when given), with the fit status of the last finish. The
cold fit's own wall time INCLUDES the screen and the split-and-merge
stage: they are the estimator. Products, per cold fit:
`n_starts_screened`, `screen_rounds`, `n_survivors`, `n_smem_tried`,
`n_smem_accepted`, `smem_wall_time`, plus the existing `fit_status` and
the audit's `search_gap`, `search_failed`, `search_status`.

### 2.5 Untouched

The local optimizer, continuations (a fit with `start` runs neither
2.2 nor 2.3), labelling, the influence, the audit, the penalty, the
start generators. No new switch: the cold search is 2.1–2.4, always.

## 3. Validation (direct calls; one method run)

1. **The eight audited draws** (45, 3, 39, 0, 1, 2, 10, 20): cold fit,
   then the audit. PASS when `search_gap ≥ −η` on all eight. The
   product fields say which mechanism did it on each draw:
   `n_smem_accepted = 0` means the racing screen alone reached the
   basin; `n_smem_accepted > 0` means the split-and-merge stage was
   needed. No validation-only switch is built.
2. **Cost on the eight draws:** cold-fit wall time before (the stored
   ~90 s) and after, with `screen_rounds`, `n_survivors`,
   `n_smem_tried`, `n_smem_accepted` per draw.
3. **The five diagnostic continuation cases** of the fit spec:
   unchanged to 10⁻⁷ relative (continuations do not run 2.2 or 2.3).
4. ~~The 100-draw oracle run~~ — dropped by the author (29 September):
   validation ends with items 1–3. The `search_failed` rate over many
   draws is read from the study's own products on TACC, where the audit
   runs on every cold fit. After items 1–3 the estimator is DEFINED as
   the best of this search, the audit is in every product, and no
   further mechanism is added.

## 5. Author's rulings on the build (29 September)

- The split displacement (½ standard deviation along the leading
  eigenvector) and the proposal breadth C are constructor arguments of
  the estimator, `split_offset` (default 0.5) and `breadth` (default
  5), not module constants; both recorded in the products.
- k_keep = 3 in the racing screen is the old promotion count carried
  over, to be reviewed in the constants audit with the others.

## 4. What is not promised

That the global maximum is found on every draw; no search promises
that for this likelihood. What is promised: the search is the
standard one, its budget is set by the starts rather than by a number,
the merged-component failure has its standard remedy, and every
product says whether the search fell short on its draw.
