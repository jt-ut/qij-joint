# QIJ conference-talk handoff

Last updated: 1 October 2026

## Purpose of this file

This is the durable context for a new chat or model helping JT write the
conference talk about **QIJ**.  The immediate task is to develop the
background and warm-up slides while one final method modification is being
built and tested elsewhere.

The talk is about the paper's two-stage QIJ method and its current,
recognizably related implementation.  It is **not a talk about `qijdt` or the
standalone Haar/influence tree**.  That tree was explored and retained in the
package, but it required more than 4,000 estimator evaluations on cloudfil
and is not the headline method.  Do not let the long tree discussion in the
old chat take over the talk.

## What the talk is trying to accomplish

QIJ estimates uncertainty for an expensive, weighted black-box estimator
without paying for a pointwise finite-difference influence calculation over
all N observations or for a large bootstrap.  The paper's basic idea is:

1. Quantize the data space with X-VQ.
2. Evaluate derivatives for weighted prototype cells to survey influence.
3. Fit a pilot influence model (currently a GP) and predict influence over
   the observations.
4. Quantize the predicted influence (IVQ) to form a small number of groups.
5. Measure group derivatives on the full data and refine/check the grouping
   until the desired unresolved-variance tolerance is met.

The statistical identity underneath the method is the law of total
variance.  For influence values psi and a partition into bins,

    V_IJ = V_between + V_within.

Measured bin-mean influences give V_between.  QIJ designs and checks a
partition intended to make the undiscovered V_within no more than a declared
fraction epsilon.  The interval must be based on quantities justified by the
accepted method; model-predicted variance must not silently substitute for
measured influence variance.

The presentation should emphasize the idea, the scientific use, empirical
performance, and honest limitations.  It need not be a bookkeeping account
of every code switch.  JT has explicitly said that small-to-medium changes
from the accepted paper are acceptable in the talk, provided the method
remains recognizably QIJ.

## Scientific motivation and example

JT's applied setting is astronomy: young stellar objects (YSOs), including
STARFORGE-like simulations and radio observations.  The important rehearsal
example is `cloudfil_G_B6_P3_v1` (usually called **cloudfil**): N = 10,000
two-dimensional points containing filamentary structure and a compact source.
It resembles the geometry and estimation difficulty of the intended applied
work better than the paper's easier benchmark examples.

The cloudfil estimator is a Gaussian-mixture fit.  The principal reported
component has seven measured outputs: x, y, log effective radius, log axis
ratio, position angle, logit weight, and log contrast.  Its influence can be
highly localized.  This is the stress test that exposed weak pilot
interpolation and expensive refinement in the original implementation.

Do not describe cloudfil as evidence that geometry and influence are
unrelated.  The working view is subtler: geometry is informative, but the
mapping from data geometry to influence is nonlinear and estimator-specific;
a coarse geometric cell average can conceal a sharp influence pattern.

## Current method state: read before writing algorithm/results slides

The specifications are moving while the final modification is validated.
Before stating the finalized algorithm, evaluation count, tolerance rule, or
results, read these files in this order and check their modification dates:

1. `spec/QIJ_method_and_code_spec.md` -- canonical foundation and code map.
2. `spec/QIJ_mods_waves.md` and `spec/method_notes.md` -- accepted amendments
   and derivations.
3. `spec/QIJ_mass_weighted_fit_spec.md` -- mass-weighted GP/pilot fit.
4. `spec/QIJ_joint_check_measured_spec.md` -- measured continuation/check.
5. `spec/QIJ_seeded_measured_tree_spec.md` -- the final modification now being
   built/tested.  Despite the word "tree," this is a proposed replacement for
   QIJ's second-stage growth/check loop, seeded by the GP pilot; it is not the
   standalone `qijdt` method.  As of this handoff it is on branch
   `qij-seeded`, is not merged, and is not accepted until its validation is
   reported and JT rules on it.

The old `spec/QIJ_state_brief.md` is useful history but is dated 29 September
and foregrounds a tree direction that JT has since declined as the talk's
headline.  It must not override the newer specifications above.

When the background build finishes, ask JT for the validation result or read
the resulting report before freezing any method or results slides.  Never
turn a projected cost or acceptance criterion in a specification into a
measured result.

## Settled technical context worth retaining

- X-VQ construction itself is cheap on the cloudfil rehearsal (about one
  second).  The important costs have been GP fitting/prediction and full-data
  IVQ measurement/refinement.  Do not frame prototype construction as the
  bottleneck.
- The X-VQ survey uses weighted cell representations.  In the moments path,
  the representations preserve each receptive field's first two moments.
  A previously serious normalization bug was found: survey weights must sum
  to N, not to the reduced row count.  After correction, prototype survey
  influences agreed much better with true receptive-field means.
- The existing GP's predictive role has nevertheless been imperfect on
  cloudfil, particularly for localized influence.  Better grouping from the
  pilot can reduce costly refinement, but the final measured bin derivatives
  are what establish V_between.
- A mass-weighted pilot likelihood is a current method modification.  Its
  motivation is that the GP fit should serve the mass-weighted influence
  objective rather than give every prototype cell equal leverage regardless
  of its probability mass.  Consult its current validation before claiming
  gains.
- A moment-aware aggregate-observation GP was separately investigated in
  `spec/QIJ_moment_aware_GP_proposal.md`.  It uses the same survey derivatives
  and conditions on a cell-average linear functional rather than pretending
  that a cell's measured mean is a point observation at its centroid.  On
  cloudfil draw 1, its controlled mean-only experiment improved pointwise
  R-squared and oracle 17-bin variance retention for all seven outputs.
  However, a consistent posterior covariance, stopping rule, V_within
  estimate, and coverage validation were not completed.  The exploratory
  hybrid reused uncertainty from the old GP and is not valid evidence for a
  complete moment-aware method.  Do not present it as an accepted feature
  unless later code/specifications explicitly adopt and validate it.
- CADJ/Voronoi-2 subdivision was tested as a possible pilot repair.  It did
  not consistently align with influence variation and should not be sold as
  the solution.
- The standalone qijdt/Haar tree's coefficients are measured contrasts
  between geometrically defined child groups.  It was informative as a
  diagnostic, but its cloudfil evaluation cost made it unsuitable as the
  headline method.

## Bootstrap comparison: use the correct equivalence

An archived 30.5-second bootstrap run used only B = 50.  It is not a valid
production wall-time comparison.

For a normal bootstrap interval, the usual relative Monte Carlo standard
error of the bootstrap **variance** is approximately

    sqrt(2 / (B - 1)).

Thus, if QIJ's epsilon is interpreted as a fractional variance-computation
tolerance, a nominal comparison sets approximately

    epsilon = sqrt(2 / B),    or    B = 2 / epsilon^2.

Examples: epsilon 0.01 corresponds to about B = 20,000; 0.02 to 5,000;
0.05 to 800.  This equates numerical precision in estimating variance under
normal assumptions.  It does not equate statistical bias, interval coverage,
or all sources of approximation error.  Keep those claims separate.

The limiting bootstrap variance estimate as B tends to infinity is the
conditional variance of the estimator under the empirical resampling
distribution.  Increasing B removes Monte Carlo noise from approximating
that bootstrap distribution; it does not remove bootstrap bias or finite-N
statistical error.

## Immediate slide-writing task

Begin with background/warm-up slides that do not depend on the pending final
modification.  A useful initial narrative is:

1. Applied problem: uncertainty for an expensive fitted object, motivated by
   identifying and characterizing structure in astronomical point clouds.
2. Why ordinary resampling is costly: the estimator must be fit many times.
3. Infinitesimal jackknife intuition: ask how the fit changes when each
   observation's weight is nudged; variance comes from the spread of those
   influences.
4. Computational obstacle: exact pointwise finite differences replace B
   bootstrap fits with N perturbation fits, which is still too much.
5. Quantization insight: nearby/behaviorally similar observations can be
   perturbed together; between-group influence variance is measured, and the
   within-group remainder is the approximation error.
6. QIJ promise: use data geometry for a cheap influence survey, use the
   predicted influence to build efficient groups, then verify/repair those
   groups with measured full-data perturbations.
7. Transition to cloudfil: a scientifically relevant case where influence is
   localized and therefore tests whether the survey and repair really work.

For these opening slides, favor one idea and one visual per slide.  Define
influence verbally before using equations.  A small variance-decomposition
picture or animation is likely more useful than early GP details.  Avoid
qijdt, CADJ, moment-GP covariance, and implementation switches in the warm-up.

## Claims discipline for the talk

- Distinguish measured results, validation criteria, and projections.
- Compare evaluation cost and wall time separately.  Wall-time advantages can
  be estimator- and hardware-specific.
- Do not claim that a GP posterior alone recovers missing V_within unless the
  finalized method and coverage study establish that result.
- Do not claim that a better pilot R-squared automatically implies valid
  confidence intervals.  The relevant endpoint is the final variance and
  coverage after the measured grouping/check procedure.
- Explain epsilon in plain language as the tolerated unresolved fraction of
  influence variance, but use the final accepted implementation's exact
  stopping rule when giving details.
- Preserve the main conceptual identity: QIJ uses quantization to reduce the
  number of expensive perturbation evaluations while measuring enough
  between-bin influence variance to construct uncertainty intervals.

## Suggested first prompt in a fresh chat

> Read `spec/QIJ_talk_handoff.md` completely, then read the current source
> specifications it names.  We are writing the background/warm-up slides for
> my QIJ conference talk, not a qijdt talk.  Start by proposing a concise
> opening slide arc; do not freeze the final algorithm or results until I give
> you the outcome of the pending seeded-stage validation.

