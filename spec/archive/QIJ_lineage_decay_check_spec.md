# The within-variance check on draw 45, version 3: a per-leaf within model trained on the nodes whose within the loop has already measured (planner, 6 October 2026; an offline check, not a build)

STATUS: versions 1 and 2 (the split energies as a decay process on the
tree) ran on 6 Oct. v1 blew up (my spec). v2 was sound and taught two
things: the tree latent predicts the next round's split energies within a
factor of two with no reference to the truth, and a model of the energy
BELOW a leaf extrapolated from the energies of selected splits
over-estimates the within of the large bulk leaves by 12–128× and of the
final leaves overall by 3–13× (pa: 1.2, the first model to read it).
Both versions ignored the training data this version is built on.

## 0. The question

Does a per-leaf model of the within variance, trained on the split nodes
whose within the loop has already measured and using only quantities the
loop carries for every leaf, predict the within of the unsplit leaves
better than today's read? Draw 45, the exact-influence emulation,
ε = 0.01, variant A, run to its own stop (L = 352). Under five minutes,
four threads, `python3.9`. The truth T of the unsplit leaves is used ONLY
as the final score (section 5).

## 1. The training data the loop itself produces (measured, not oracle)

For a split node k the identity T_k = D_k + T_a + T_b telescopes down its
sub-tree. If every leaf below k is a singleton, T_k equals the sum of the
realized gains in k's sub-tree: its within is MEASURED exactly. If some
leaves below k are not singletons, that sum is a LOWER BOUND B_k ≤ T_k,
with T_k − B_k the within of those leaves. On draw 45 at the stop:

| split nodes | 351 |
| fully resolved (T_k measured) | 153: 73 of size 2, 39 of 3–4, 26 of 5–8, 10 of 9–16, 4 of 17–32, 1 of 33–64 |
| partially resolved (B_k ≤ T_k) | 198; B_k/T_k = 0.77–0.96 at sizes 3–32, 0.89–0.99 at ≥ 33 |

So the within of every split node is either measured or bounded from
below within 1–25%. This is the censoring that is real: a bound from
below on an unobserved variance. The unsplit leaves are the prediction
set (142 with n ≥ 2; singletons are exactly 0).

## 2. The covariates, all available for every node including unsplit leaves

Per node k and output c, in per-point units (variance over N):

- w_kc = W_kc/p_k: the pilot's spread inside the node at its creation
  (after its own mean update), i.e. today's within read before κ̂.
- g_kc = e_{π(k)c} = D_{π(k)c}/p_{π(k)}: the parent's per-point split
  energy, the last coefficient measured on the node's lineage.
- u_k = log(n_k/N), the node's log size.

What the products say about them (scratchpad worked_example/dump*.txt;
sums over the class, predicted over true; pa in the last column):

Unsplit leaves, where the within sits — the 33 two-point leaves hold
36–96% of the final leaves' within (pa 96%, log_axis_ratio 88%,
log_reff 72%, x 57%, y 51%, log_contrast 40%, logit_w 36%); the 26
leaves of n ≥ 65 hold 25–38% on x, y, logit_w, log_contrast.

| class | ΣW/ΣT (x, y, reff, axis, w, contrast) | pa | Σ(g·p)/ΣT | pa |
|---|---|---|---|---|
| n = 2 (33) | 0.21 0.18 0.05 0.05 0.33 0.27 | 0.000 | 1.28 0.72 0.97 1.26 1.84 1.08 | 0.40 |
| n ≥ 33 (46) | 0.78–1.0 (both sub-classes) | 0.003 | 0.3–0.9 on half, 2–19 on the rest | 0.3–40 |

Fully resolved nodes (training set, T exact):

| class | ΣW/ΣT (six smooth outputs) | pa | Σ(g·p)/ΣT (six) | pa |
|---|---|---|---|---|
| n = 2 (73) | 0.71 0.80 0.18 0.18 0.94 0.22 | 0.004 | 1.37 1.34 0.67 0.83 1.68 0.56 | 1.35 |
| n = 3–4 (39) | 0.72 0.78 0.25 0.18 0.67 0.32 | 0.006 | 0.94 0.35 0.52 0.57 0.33 0.53 | 1.35 |
| n = 5–16 (36) | 0.68–0.87, 0.13–0.20, 0.72–0.84, 0.36 | 0.006–0.03 | 0.1–0.3 | 0.2–0.65 |

Two facts decide the model's form. (i) The pilot's spread w reads the
bulk leaves within 0.8–1.0 on the six smooth outputs and the small leaves
at 0.05–0.3; on pa it reads nothing anywhere (0.000–0.004). (ii) The
parent's split energy g reads the two-point leaves within 0.7–1.8 on the
six smooth outputs and 0.4 on pa, and over-shoots or under-shoots the
large leaves by up to 40×. So neither covariate alone will do, their
relative weight depends on where the node is, and the per-output
calibration of each is learnable from the resolved nodes. The model is
additive in the two covariates, with a per-output weight on g and a
local amplitude on the tree.

## 3. The model

For every split node k (training) and every unsplit leaf ℓ (prediction),
per output c, with τ_kc = T_kc/p_k the per-point within:

    τ_kc ~ Gamma( shape (n_k − 1)/2,  mean m_kc )         (the χ²_{n−1} law of a variance over n points)
    m_kc = exp(b_c + a_k) · ( w_kc + θ_c · g_kc )
    a_k = a_{π(k)} + η_k,  η_k ~ N(0, ν²),  a_root = 0;   θ_c = exp(t_c) ≥ 0

- b_c: the output's calibration of the pilot's spread (κ̂'s role).
- θ_c: how much of the last measured coefficient on the lineage to carry
  into the node's within, per output; near 1 on the two-point leaves by
  the table above, learned.
- a_k: the local log-amplitude shared across outputs, a random walk on
  the tree; it lets the core carry a different calibration from the bulk
  without a size rule.
- The shape is the sampling model, not a rule: the within of n points
  is a scaled χ² with n − 1 degrees of freedom, so a two-point node's
  within is one squared difference (shape ½) and a 100-point node's is
  tightly determined (shape 49.5).

Likelihood: for a fully resolved node, the gamma log-density of τ_kc at
mean m_kc. For a partially resolved node, the log survival
log P(τ ≥ B_kc/p_k) under the same gamma (`scipy.stats.gamma.logsf` with
shape (n_k − 1)/2 and scale m_kc/shape). Penalty: Σ_edges (a_k −
a_{π(k)})²/(2ν²). Unknowns: b (7), t (7), a (one per non-root node).
Maximise by `scipy.optimize.minimize` (L-BFGS-B) with an ANALYTIC
gradient (the gamma log-density and log-survival derivatives with respect
to log m are closed form; the derivative of log m with respect to b, t,
a is immediate). Start at b = log(ΣT/ΣW) over resolved nodes per output,
t = 0, a = 0. Posterior covariance of the unknowns: the inverse of the
Hessian of the negative penalised log-likelihood at the optimum, formed
by central finite differences of the analytic gradient (716 gradient
evaluations; `pinv` if singular, and say so).

ν grid: {0.1, 0.3, 1, 3, 10}, chosen once by the out-of-sample score of
section 4 (the pooled log-ratio RMS), never by the truth; every grid
value's score is reported.

### Explosion cases, stated

- m_kc ≤ exp(b_c + a_k)·(w + θ g): bounded by the covariates times an
  amplitude. a_k is penalised to its parent; b_c is a global constant;
  θ_c is a global per-output weight. The worst case is a_k at the edge of
  the prior, e^{±3} at ν = 1: a factor of 20, not 10^40.
- w_kc = 0 and g_kc = 0 together (a node whose parent's split had zero
  gain and whose pilot spread is zero): m = 0 and the gamma likelihood is
  undefined. Floor m at 1e-12·V_ij/N (report how many rows hit the floor).
- The log-survival for a partially resolved node with B ≪ m is ≈ 0 and
  carries no information; with B ≫ m it is very negative and pulls m up:
  that is the censoring doing its job.
- SANITY CAP (agent): if for any output Σ_ℓ T̂_ℓc over the unsplit leaves
  exceeds V_ij,c, or exceeds 100 × the baseline κ̂·ΣW, stop and report the
  offending numbers instead of the full tables.

## 4. Scoring (a): time-ordered, without the truth

At each round r = 2, …, 10: the training set is every node that is fully
or partially resolved by the END of round r − 1 (its sub-tree gains
summed over rounds ≤ r − 1; "fully resolved" likewise at that time). Fit
the three predictors below on it. The held-out set is the leaves that
exist at the end of round r − 1 and are split in round r; their target is
their within as measured by the END of the run (T_k if fully resolved by
the stop, else B_k at the stop, which is ≥ 0.77 T_k). Score, per output
and pooled over rounds: Σ predicted / Σ target over the held-out leaves,
and the RMS of log(predicted/target) over held-out leaves with n ≥ 2.

- P0: m = exp(b_c) · w   (today's read: one factor per output on the pilot's spread).
- P1: m = exp(b_c) · (w + θ_c g)   (no tree latent).
- P2: the full model at each ν.

## 5. Scoring (b): the unsplit leaves against the exact within, once

Fit P0, P1 and P2 (chosen ν) on all resolved and bounded nodes at the
stop. Predict T̂_ℓc = p_ℓ · m_ℓc for every unsplit leaf with n_ℓ ≥ 2
(a_ℓ from the prior: its parent's value). Report, per output, for P0,
P1, P2 and the BASELINE (the emulator's own κ̂·ΣW at the stop):

- Σ_ℓ T̂_ℓc / Σ_ℓ T_ℓc overall and by size class n = 2, 3–4, 5–8, 9–16,
  17–32, 33–64, ≥ 65;
- the implied variance error (V_btw + Σ T̂ − V_ij)/V_ij divided by ε;
- for P2 the predictive relative spread sd/Σ T̂ of the leaf sum: parameter
  part by the delta method on log m_ℓc through the posterior covariance
  (design row: 1 at b_c, ∂log m/∂t_c = θ g/(w + θ g) at t_c, 1 at a_ℓ),
  plus the sampling part Σ_ℓ p_ℓ² m_ℓc² · 2/(n_ℓ − 1); both parts and the
  total.

Plus: b̂_c, θ̂_c, the chosen ν, min/median/max of â over the unsplit
leaves, the number of floored rows, optimiser status and iteration count
for every fit, the identity Σ_ℓ T + Σ_s D = V_ij, stop L, counts of
resolved/bounded/unsplit nodes.

## 6. Worked example (the numbers the spec was checked against)

Two-point leaf 668 in the core (parent of size 3), pa: w = 1.1e-11 per
point·mass, g·p = 2.8e-5, T = 1.66e-4. With θ_pa ≈ 1/1.35 from the
resolved two-point nodes and a = 0: T̂ ≈ 2.1e-5, one eighth of this
leaf's truth. That is the single-leaf scatter (the 10th–90th percentile
of log10(g·p/T) over the two-point leaves is −1.6 to +2.1); the SUM over
the 33 two-point leaves is what the stop reads, and at the sum level the
same calibration gives predicted/true = (Σg·p/ΣT on unsplit) /
(Σg·p/ΣT on resolved) = 0.93, 0.54, 1.45, 1.52, 1.10, 1.9 on the six
smooth outputs and 0.30 on pa. The tree latent and the bounded nodes can
move these; the arithmetic cannot explode.

Bulk leaf 654 (n = 3309), x: w·p = 7.6e-9, g·p = 7.4e-11, T = 5.9e-9.
The g term is 1% of the w term; the prediction is exp(b_x)·w·p with
b_x calibrated on the resolved nodes where ΣW/ΣT ≈ 0.7 → T̂ ≈ 1.1e-8,
1.8× this leaf's truth, with a_ℓ free to pull it down toward the bulk's
own partially resolved ancestors, whose W/B is 0.6–0.65 at n ≥ 33 on x.

What would count as success: P2's implied error over ε inside ±1 on the
six smooth outputs (the baseline is −0.25 to −1.0) and nearer to 0 than
−3.3 on pa, with a spread the stop could read. What would count as
failure: P2 no better than P0 out of sample in section 4, which would
say the resolved nodes do not transfer to the unsplit leaves.

## 7. Output

`within_model_check.py` and `report.txt` in this session's scratchpad
under `within_model_check/`; the node table and the resolved/bounded
sums are built as in scratchpad `worked_example/dump2.py` (copy it; the
emulator copy `emu.py` there returns the final vector). Plain text, the
tables of sections 4 and 5, wall time, `python3.9 --version`. Verbatim,
no interpretation.
