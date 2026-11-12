# The within-bin term and the influence model: the evidence of 6 October, the options, and a recommendation (planner, 6 October 2026)

SUPERSEDED the same day by QIJ_qijt_affine_plan.md: Stage 1 of the
recommendation below (the Gaussian-process variance field's generality)
ran and failed on the Fundamental Plane; the data looks that followed
replaced the Gaussian process by the affine receptive-field model and
the full-data loop by quantized-only measurement. Kept as the record of
the evidence in sections 2.1–2.4.

Everything below is from draw 45 of the sheared mixture (cloudfil_u, p2
estimator, N = 10000, ε = 0.01), the worst-twentieth draw of the
999-draw study, unless a line says otherwise. Every number has a report
in this session's scratchpad; the specs for the checks are in this
folder (QIJ_lineage_decay_check_spec.md, QIJ_amplitude_field_check_spec.md,
QIJ_graph_variance_field_check_spec.md).

## 1. The decision to make

Two coupled questions:

1. How the loop reads the within-bin term V_win, which today is κ̂ times
   the initial influence estimate's spread inside the bins, with a χ²
   term in the error bar and an optional round-level floor.
2. What the influence model is, which today is a per-output Gaussian
   process with a stationary amplitude, a local width, and the declared
   noise floor, fitted once to the prototype influences and never
   conditioned on the bin influences the loop measures.

They are coupled because the within-bin term is the influence model's
uncertainty inside the bins. A model whose uncertainty is right makes
question 1 disappear.

## 2. What the day established

### 2.1 The within-bin term cannot be read from any per-bin quantity the loop carries

Six models of V_win were fitted and scored (the pilot's spread, the GP
posterior variance, the measured-mean miss, the data-space extent, the
lineage's split energies two ways, and a model trained on the 153 nodes
whose within the loop had resolved exactly). The finding is a ceiling,
not a failure of form: the 33 two-point bins hold 36–96% of the
within-bin term, each one's within is a single squared difference, and
the best predictor of their sum scatters by a factor of forty per bin.
The oracle fits, which see the truth, read that sum within a factor of
two at best. A factor of two on a term of order ε leaves the error of
order ε. Today's κ̂ read sits at that ceiling on every output
(0.3–0.97 of the truth; pa 0.001).

### 2.2 The stop as built cannot be fixed by its error bar

The χ² term is already in the stop. It is a relative error on the
estimate, and the estimate for a two-point bin in the core is the
pilot's spread, near zero there. A near-zero centre carries a near-zero
bar. The two stops that work on this draw: the round-level floor,
every output inside ±0.06ε at 2038 evaluations (2.9× the 702 of the
loop as built); or the loop as built, −0.25 to −1.0ε on six outputs
and −3.3ε on pa at 702. The floor's cost on a typical draw is not
known (it fires only when the last round's gains exceed the modelled
within; a thousand-draw question for TACC).

### 2.3 The influence model's uncertainty can be made right, and the smooth part can carry information

The test: condition on the 1094 prototype influences alone, predict the
351 bin influences the loop measured, score the ratio of realized
squared error to predicted variance (1 = calibrated).

| model | overall ratio, 7 outputs | one- and two-point bins | bins ≥ 65 points |
|---|---|---|---|
| stationary GP, global length (today's family) | 8–57 | 10–60 | 0.4–19 |
| local width, stationary amplitude, floor | 10–714 | 10–1200 | 0.07–15 |
| + variance field on 32 regions, global length | 0.9–4.1 | 1.0–5 | 0.08–1.1 |
| + variance field on the connectivity graph, local width, floor, shared across outputs | 0.75–3.7 | 0.7–4.1 | 0.1–5.7 |

With the local width the fitted length factor sits in the interior of
its grid (2.2), so the field interpolates between prototypes; with one
global length it collapsed to white. The variance field learned on the
graph placed its twenty largest values within a few hundredths of a
parsec of core 2's fitted centre and near zero elsewhere, from the
prototype influences alone, with nothing in the model that knows what a
core is.

What is still wrong: on the one- and two-point bins the variance is
the right size but 7–33% of residuals lie beyond three standard
deviations (0.3% for a Gaussian); on the largest bins three outputs
(y, logit_w, log_contrast) keep 21–63% beyond three. The scale of the
uncertainty is learned; its distributional shape is not Gaussian.

### 2.4 A finding about the prototype stage

The declared noise floor, 2 η_Q² θ_c² median_j(1/t_j²), exceeds the
total variance of the prototype influences on x and y (4.7 vs 0.24;
0.80 vs 0.41). By the declared accuracy, those prototype influences are
noise everywhere but the core; the stored pilot's noise ratio at its
upper bound on pa is the same fact. The author's reading from the
analytic maps: any prototype away from the core has no influence, so
its measured influence is all noise.

## 3. The options

**A. Keep the loop; adopt the round-level floor.** Works on this draw
at 2.9× cost. No model change. Unknown typical cost. Leaves κ̂, the
scale/shift update and the χ² term in place, all of which the author
has called arbitrary. Not recommended as the method; acceptable as the
safeguard while B is built.

**B. Replace the within-bin read and the stop by the influence model's
posterior, with the model made calibrated.** The Bayesian-quadrature
framing: the influence is a latent field, every measurement (prototype
or bin) is a linear functional of it, V is an integral of its square,
the posterior mean and variance of V are closed-form moments of the
field, the next bin to measure is the one whose measurement most
reduces the posterior variance of V, the stop is posterior sd(V) ≤ εV.
Replaces: the scale/shift update (by exact conditioning), κ̂, the χ²
term, the floor, n_min, the ranking rule and the round sizing. Keeps:
the two-means cut as the proposal mechanism, the central pair, the
X-VQ and prototype stage. The model it needs is the last row of the
table in 2.3, plus what 2.3 says is still missing.

**C. Per-bin within models trained on resolved nodes.** Exhausted;
the ceiling of 2.1 applies. Not recommended.

**D. The score basis (influence linear in the per-point score).**
Exact for M-estimators and cheaper than QIJ for them; undercuts the
black-box premise. Withdrawn.

**E. A survey-free tree with cuts along the measured leaf-mean field.**
Possible once B exists; premature now.

## 4. Recommendation: B, in three stages, the first of which is a test of generality and costs minutes

**Stage 1 — the generality test, before anything is built.** The
calibration check of 2.3, unchanged, on ONE draw each of the paper's
other cases (Fundamental Plane, IMF, multivariate t), where there are
no cores and the influence's support is different. Same script, same
metric. If the learned variance field is calibrated there too, the
model is generic and the author's concern about tuning to one Gaussian
mixture is answered with numbers. If it is not, the model family is
wrong and nothing else should be built. Compute: about 90 s per case.
This is the first thing to do because its failure stops everything
else and its cost is nothing.

**Stage 2 — the model, in its generic form.** The influence model
becomes: a Gaussian process per output with (i) the product's local
width, (ii) one log-variance per prototype, smooth along the
connectivity graph, shared across outputs, learned by marginal
likelihood from the prototype influences, (iii) the declared noise
floor, (iv) exact conditioning on every bin influence as it is measured,
(v) the within-bin posterior variance by the product's existing
bin-summed vectors. Two further pieces, to be added only if Stage 1
shows the heavy tails are general: a learned correlation-length field
beside the variance field (the influence's own rate of change, not the
data's density), and a heavy-tailed distributional family (a scale
mixture of Gaussians) so that a rougher-than-expected region is a mild
surprise and not a ten-sigma one. Nothing in (i)–(v) references cores,
mixtures or multipoles; the earlier suggestion of a low-order basis
inside the support was estimator-specific and is withdrawn.

**Stage 3 — the loop on the posterior, emulated before built.** Extend
the draw-45 emulator so that the ranking, the round sizing and the stop
come from the posterior of Stage 2's model (expected reduction of
var(V) per evaluation; stop at sd(V) ≤ εV), with the cuts unchanged.
Two numbers decide it: evaluations to the tolerance and the resulting
error per output, beside the loop as built (702; −3.3ε on pa) and the
floor (2038; within ε). The emulator measures the cost of the posterior
updates at the same time (the bin predictions for 351 bins took 11 s;
the field fit 54 s on 1094 prototypes; the per-update cost of the
posterior moments of V is of order N × M² and is not yet measured).

Only after Stage 3 is a build order written.

## 5. What is dropped, and why, if B is adopted

- The scale/shift update of the vector: replaced by exact conditioning
  of the model on the measured bin influence; the author showed the
  scale branch rests on a multiplicative error model the children
  themselves refute.
- κ̂: a global patch for the scale branch's local defect; sits at the
  ceiling of 2.1 and cannot read pa.
- The χ² term as a bolt-on: the posterior variance carries the
  degrees-of-freedom effect by construction.
- The round-level floor and n_min: the posterior's own uncertainty
  drives measurement where the learned variance is large.
- The ranking by measured error and the round sizing by expected gain:
  replaced by one criterion, the expected reduction of the posterior
  variance of V per evaluation.

## 6. Risks, stated

- One draw of one dataset underlies every number here. Stage 1 is the
  only answer to that.
- The heavy tails (2.3) are unexplained. A heavy-tailed family is the
  generic remedy; a learned length field is the other; neither is
  tested.
- The declared noise floor says the prototype stage measures noise on
  x and y away from the core at η_Q = 4.3e-6. Whether the prototype
  stage should use a larger step or a different accuracy is a separate
  question the author may want to take up; B does not depend on it.
- Cost of the posterior updates at N = 10000 is unmeasured; Stage 3
  measures it before anything is built.
