# Proposed moment-aware GP for QIJ

## 1. Purpose

This document proposes replacing QIJ's current centroid-conditioned
influence GP with a moment-aware, aggregate-observation GP.  The change is
motivated by the cloudfil failure: the X-VQ geometry and prototype survey
define receptive fields, but the current GP discards most of that information
when it treats each measured receptive-field influence as a point observation
at the prototype centroid.

This is a replacement for the stage-1 influence model, not a new derivative
survey.  It uses the same prototype derivative evaluations already made by
the `moments` survey.  It must eventually replace both the current posterior
mean and the current posterior covariance.  Replacing only the mean produces
an internally inconsistent model and is not an acceptable final method.

## 2. Current model and its mismatch

For output coordinate c, let f_c(z) denote the latent pointwise influence
field in whitened data coordinates.  The current GP treats prototype response
I_jc as

    I_jc = f_c(w_j) + e_jc,

where w_j is the X-VQ prototype/centroid and e_jc is the GP observation noise.

Under the moments survey, however, I_jc is obtained from a weighted
representation of receptive field j.  If z_jr and q_jr are that field's
existing moment nodes and normalized quadrature weights, the appropriate
observation equation is

    I_jc = L_j f_c + e_jc
         = sum_r q_jr f_c(z_jr) + e_jc,

with sum_r q_jr = 1.  Thus I_jc is an aggregate or linear-functional
observation, not a measurement of f_c at w_j.

The distinction is immaterial for a tiny cell over an almost affine field. It
is important for cloudfil, where influence can be sharply curved or spiked
inside a receptive field.  A centroid need not carry the cell-average value.
The current model can consequently be confident about the wrong observation
semantics; ordinary GP posterior uncertainty does not diagnose this form of
model misspecification.

## 3. Proposed aggregate-observation GP

Let the R moment rows across all M receptive fields be stacked in Z_R.  Define
an R by M sparse matrix P by

    P[r,j] = q_jr

when row r belongs to field j, and zero otherwise.  Each column sums to one.
For the current cloudfil draw 1 experiment, M = 1,094 and R = 3,259, or about
three existing moment rows per field.

Use the same latent GP family as QIJ:

    f_c(z) = h(z)^T beta_c + g_c(z),
    g_c ~ GP(0, s_c^2 k_ell),

with the requested affine or quadratic trend and Matern-3/2 kernel.  If K_RR
is the kernel on the moment rows and H_R is their trend matrix, the observation
kernel and trend are

    K_oo = P^T K_RR P,
    H_o  = P^T H_R.

For query points Z_X,

    K_Xo = K_XR P.

The generalized least-squares posterior mean is then fitted from

    A_c     = K_oo + lambda_c I,
    G_c     = H_o^T A_c^-1 H_o,
    beta_c  = G_c^-1 H_o^T A_c^-1 I_c,
    alpha_c = A_c^-1 (I_c - H_o beta_c),

and evaluated as

    psi_hat_0c(X) = H_X beta_c + K_Xo alpha_c.

The existing shared-width REML search can be retained, but its likelihood must
use K_oo and H_o.  The experimental implementation did this for a global
width and quadratic trend.

## 4. Posterior covariance that must accompany the mean

The moment-aware posterior mean must not be combined with uncertainty from
the old centroid-conditioned GP.  For query points x and x', define

    k_xo = Cov(f(x), Lf),
    r_x  = h(x) - H_o^T A_c^-1 k_ox.

The consistent universal-kriging covariance is

    Sigma_c(x,x') = s_c^2 [
        k(x,x')
        - k_xo A_c^-1 k_ox'
        + r_x^T G_c^-1 r_x'
    ].

Pointwise posterior standard deviations use Sigma_c(x,x).  QIJ's refinement
also requires, for every candidate bin B,

    v_Bc = mean_{i in B} Sigma_c(x_i,x_i)
            - mean_{i,j in B} Sigma_c(x_i,x_j).

This is the same quantity the current `bin_posterior_variance` computes, but
all cross-covariances must be taken against the aggregate observations.  Bin
sums can still avoid materializing a dense within-bin covariance matrix.  In
particular, the observation-space cross sum is obtained from

    sum_{i in B} k_oi = P^T sum_{i in B} k_Ri.

The raw within-bin kernel sum and universal-mean correction then enter the
same `mean(diag) - mean(all)` calculation used today.

The consistent covariance must supply all of the following:

1. pointwise `sigma`;
2. `bin_posterior_variance`;
3. refinement split gains and priorities;
4. the refinement stopping decision;
5. the predicted V_win contribution.

## 5. What the completed experiments establish

### 5.1 Controlled posterior-mean comparison

The clean experiment used cloudfil/P2 draw 1, N = 10,000 and M = 1,094.  Both
models received exactly the same stored prototype responses I_j.  No new
derivative or estimator evaluation was added.  The only change was the
observation equation used by the GP.

The point-observation control treated I_j as f(w_j).  The aggregate model
treated I_j as the weighted average of f at the existing moment rows.  Against
the analytic pointwise influence, the results were:

| P2 output | Point GP R^2 | Aggregate GP R^2 | Point 17-bin share | Aggregate 17-bin share |
|---|---:|---:|---:|---:|
| x | 0.831 | 0.904 | 0.860 | 0.899 |
| y | 0.896 | 0.929 | 0.907 | 0.941 |
| log radius | 0.719 | 0.785 | 0.757 | 0.777 |
| log axis ratio | 0.698 | 0.806 | 0.726 | 0.806 |
| position angle | 0.515 | 0.765 | 0.612 | 0.786 |
| logit weight | 0.807 | 0.855 | 0.823 | 0.856 |
| log contrast | 0.737 | 0.814 | 0.787 | 0.822 |

Here “17-bin share” forms 17 one-dimensional k-means groups from the predicted
field and then evaluates the variance of the analytic influence means in those
groups.  It measures the partition information supplied to IVQ; it is not an
end-to-end QIJ variance estimate.

The aggregate observation model improved pointwise R^2 and 17-bin variance
retention for all seven outputs.  The largest repair was position angle.  Its
prediction-to-truth slope also improved from about 117 under the point model
to 1.80 under the aggregate model.  These results are direct evidence that
discarding receptive-field extent in the observation equation is one source
of the cloudfil prediction failure.

They do not demonstrate that the aggregate model is already sufficient.  Its
17-bin shares still range from 0.777 to 0.941, below the method's desired
near-complete capture for some outputs.

### 5.2 Component timing

One local profile on the same draw gave:

| Component | Wall time |
|---|---:|
| X-VQ geometry | 0.66 s |
| Existing point GP fit plus full point posterior | 5.91 s |
| Experimental aggregate GP fit plus mean prediction only | 10.33 s |

The point and aggregate timings are not equivalent: the point-GP time includes
its full posterior uncertainty calculation, whereas the aggregate-GP prototype
currently computes only the posterior mean.  Therefore 10.33 seconds is not a
timing estimate for the completed replacement.  Exact aggregate uncertainty
has neither been implemented nor timed.

### 5.3 Hybrid end-to-end experiment

An exploratory end-to-end run supplied the aggregate posterior mean to IVQ
but retained the old point-GP uncertainty in refinement.  At 14 workers it
reduced the observed total wall time from roughly 316 seconds for the matched
point-GP run to 201 seconds, and refinement evaluations from 1,410 to 844.

This result must not be interpreted as the performance of a moment-aware GP.
The hybrid used incompatible mean and covariance models.  It stopped too
early on some outputs; against the freshly generated analytic IJ reference,
its V_btw/V_ij ratios were approximately

    [0.819, 0.905, 0.973, 0.639, 0.859, 0.792, 0.790].

In particular, the axis-ratio result was unacceptable.  This may be caused by
the mismatched uncertainty rather than by the aggregate mean.  The experiment
shows that a better initial field can change and reduce refinement work.  It
does not establish a valid stopping rule, final variance, interval, or
coverage result for the proposed model.

Serial and parallel survey runs also landed in different estimator basins and
produced different prototype responses.  Results from different worker modes
must not be mixed.  Future comparisons must cache one X-VQ, theta_Q and I_j
survey and feed those identical artifacts to both GP alternatives.

## 6. Other caveats

1. Only the controlled mean comparison on cloudfil/P2 draw 1 has been
   completed.  Improvement across draws and datasets is unknown.
2. The experiment used a global stationary Matern-3/2 kernel and quadratic
   trend.  A local-width aggregate kernel has not been implemented or tested.
3. The existing declared-noise floor was reused.  Its derivation should be
   checked for an aggregate observation, because quadrature and estimator
   error need not have the same covariance as independent point noise.
4. The naive prototype constructs K_RR explicitly.  With R about 3M this is
   feasible at M near 1,000 but is unnecessary.  K_oo can be accumulated from
   the small per-cell moment supports or in blocks.
5. Exact aggregate posterior uncertainty may cost appreciably more than the
   current mean-only prototype.  Its cost must be measured, not inferred from
   the 10.33-second result.
6. Better initial 17-bin retention does not by itself guarantee correct final
   confidence intervals.  V_btw, V_win, stopping, and empirical coverage all
   require end-to-end validation.
7. The archived 30.5-second bootstrap comparison used B = 50, not B = 2,000.
   It cannot support a wall-time conclusion about QIJ versus a production
   bootstrap.

## 7. Required implementation contract

A production implementation should make the observation operator explicit in
`InfluenceModel`, rather than attaching a replacement prediction to the old
model.  At minimum it must store the moment-row design, sparse operator P,
aggregate trend, fitted aggregate covariance factors, and enough cached terms
to answer point and bin covariance queries.

The implementation should proceed in this order:

1. Add an aggregate-observation fit that returns a complete influence model,
   not merely predictions.
2. Implement aggregate posterior mean, pointwise variance, and bin posterior
   variance from the same fitted factors.
3. Make `psi0`, `uncertainty`, and `bin_posterior_variance` dispatch through
   the model's observation type so mixing point and aggregate posteriors is
   impossible.
4. Preserve the point-observation path as a bit-identical control.
5. Optimize K_oo construction only after the mathematical tests pass.

Required unit tests include:

- one moment node per field with P = I reproduces the point-observation GP;
- aggregate means computed by the posterior agree with direct application of
  P to latent-row predictions;
- posterior covariance is symmetric and positive semidefinite to tolerance;
- point variances and bin variances agree with a small brute-force covariance
  calculation;
- a constant field follows the existing constant path;
- no additional estimator calls occur relative to the current moments survey.

## 8. Validation and acceptance criteria

The first end-to-end comparison must reuse exactly the same cached X-VQ,
moment rows, theta_Q and prototype I_j for both models.  It should report:

- pointwise R^2, calibration slope and normalized error against analytic
  influence where available;
- initial 17-bin V_btw/V_ij;
- final V_btw/V_ij and V_tot_hat/V_ij;
- refinement evaluations and rounds;
- empirical interval coverage across draws;
- separately timed kernel construction, hyperparameter search, mean
  prediction, point uncertainty, bin uncertainty and estimator refinement.

The replacement should not be accepted merely because it improves the mean.
It must satisfy all of the following on the agreed rehearsal draws:

1. no new derivative/estimator evaluations in stage 1;
2. no material loss for any measured output relative to the point GP;
3. final variance and interval coverage meet the existing QIJ gate;
4. refinement evaluations are materially reduced;
5. aggregate posterior computation does not erase the saved refinement cost.

## 9. Present status

The aggregate-observation mean is a promising and well-motivated replacement
for the current centroid-conditioned mean.  Its controlled improvement across
all seven cloudfil influence fields is the strongest positive GP repair found
so far.  The complete moment-aware GP, however, does not yet exist: posterior
uncertainty, bin variance, V_win, stopping and coverage remain untested.  The
hybrid end-to-end results must not be presented as results of the proposed
replacement.
