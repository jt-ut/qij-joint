# QIJ method notes

Reference for the derivations behind `qij_joint`'s code, stated as the
method now is. Terms and symbols follow `glossary.md` in this folder;
subscripts i for data points, j for X-VQ prototypes, k for I-VQ bins,
c for estimator outputs. Each module names the section here it
implements, once.

## 1. The weight constructor and step rule

For a member set K of mass p in base weights omega0 (of any
total), the one weight constructor is

    omega_i(t) = (1 - t) * omega0_i + t * omega0_i * 1{i in K} / p

Every omega_i(t) sums to R for every t. A relative step delta on K --
the fraction of K's OWN mass the step moves it by -- is the weight
parameter t = delta * p / (1 - p): under it, every member weight is
multiplied by (1 + delta) and every other weight by (1 - t), so K's
mass becomes p*(1 + delta). Steps used: delta_f = 2*sqrt(eta) for a
single forward difference (the
prototype survey, and a refinement split's one-shot measurement);
delta = (3*eta)^(1/3) for the central stencil that measures the
initial bins.

The central stencil, at t = step_parameter(delta, p):

    U = [T(+t) - T(-t)] / (2*t)

is always used: every registered estimator's eta keeps delta <= 1, so
the downward step never drives a member weight negative (the one-sided
fallback for delta > 1 has no current use and is not carried).

`Counter` counts every evaluation, its rows, and any evaluation whose
result carries a NaN, and offers `T.prepare(A)` -- computed once per
distinct array object A over the Counter's lifetime -- to any
estimator that has it, so an estimator's weight-independent
quantities (log-ratios, standardized X, fixed design bases) are not
recomputed on every call. The prototype survey evaluates T on the
prototypes, a different array from the full data X, so it gets its
own prepared state.

## 2. The X-VQ: prototype count and prototype survey

The requested prototype count balances the cost of the two stages:

    M_ref = ceil(sqrt(kappa_ref / eps)),  kappa_ref = 2.7
    M_X   = ceil(sqrt((1 + 2*q*M_ref) * N / 2))

floored at 20 (a regression needs points) and capped at N // 2, beyond
which the X-VQ degrades toward a subsample jackknife -- the X-VQ is
given the same row budget as the finite differences of stage 2.

The quantizer (`vqlp.VQFitter`, Euclidean, two best-matching units) is
fit with no estimator evaluation, at `workers` FAISS OpenMP threads
(the pipeline's own worker count, 1 without a pool): the codebook fit
and recall give identical centers, bmu and bmu2 at 1, 4 and 8 threads,
so this is bit-identical, and the process's previous thread count is
restored afterward. The prototype survey then evaluates
T once on the prototypes, weights M_used*p, for theta_Q; for each
prototype j, one forward difference at t_j = step_parameter(delta_f,
p_j) gives I_j = [T(omega(t_j)) - theta_Q] / t_j. A failed prototype
evaluation is a missing response, per output: I_proto is mass-centred
over the finite prototypes only, and a prototype left NaN by its own
evaluation stays NaN, neither filled in nor dropped, since a vector
estimator can fail one output and not another at the same prototype.

**The prototype count as an argument.** `QIJ(M_X=...)` replaces the rule
above when given; the result records `M_X_source`, `'rule'` or
`'argument'`.

**The survey's representation of a receptive field: `survey`.** Under
`survey='points'` (ported, bit-identical) each live field j is one
survey row, the prototype, weight M_used*p_j, as above. Under
`survey='moments'` (spec/QIJ_mods_waves.md A8) field j is instead
represented by rows built from its own n_j NATIVE points (the rows T
is called on, not the whitened prototype): its own n_j rows, each
weight M_used*p_j/n_j, when n_j <= d_x+1; otherwise the d_x+1 vertices
of a regular simplex centred at the field's native mean, unweighted
second moment exactly I_{d_x} (the Helmert contrast matrix, scaled),
mapped through the eigendecomposition-based factor V*sqrt(Lambda) of
the field's native population covariance (so a rank-deficient field --
collinear points -- needs no regularization: a zero eigenvalue maps
every vertex's component along it to zero), first vertex along the
field's leading eigenvector by a fixed reflection of the canonical
simplex, each row weight N*p_j/(d_x+1) = n_j/(d_x+1), so the rows' weights
sum to N, the number of points they stand for. Either representation
reproduces the field's native mean and covariance exactly. A forward
step at prototype j scales EVERY row of field j by the same factor
(1-t_j) + t_j/p_j, every other row by (1-t_j) -- the points path's own
weight constructor (section 1), generalized to a stacked row array
whose weights do not sum to the row count, so raising the field's mass
scales all its rows alike and I_j, theta_Q and the GP's input (the
field means) keep their definitions. `rows`, the field index of each
row, and each row's base weight are built once per draw, vectorized
over fields: the field moments (grouped sums over the field index) and
the simplex map (a batched eigendecomposition and matrix product over
every field needing one) are each one call over every field, not a
per-field loop. Counter rows per evaluation are the stacked row count.
`X` is unused under `survey='points'`.

**The survey on a pool.** With a pool, the M_used per-prototype forward
differences are pool tasks; the base evaluation theta_Q stays in the
parent. The survey's own row array (W_X under `survey='points'`, the
stacked field rows under `survey='moments'`) is shared with the
workers once per draw; each task carries its prototype index and
perturbed weights over that same array, computed by the parent, and
returns the raw evaluation, a failure flag and its own wall time. The
parent assembles I_j in prototype order and adds each task's evaluation,
rows and failure to the counter, so I_proto and every count are the same
at any worker count. A worker computes `T.prepare` on the shared array
once per draw. `busy_time_total` replaces the pool's elapsed time within the survey by
the tasks' summed wall times; without a pool it equals the elapsed total.

**theta_hat on a pool.** The full-data base evaluation theta_hat =
T(X, ones(N)), used in stage 2, does not depend on stage 1's result. With
a pool, X is shared once and theta_hat is submitted as a pool task
immediately, before stage 1 runs, and collected once stage 1 finishes;
the collecting call adds its evaluation, rows and failure to the counter
at that point, so counts are the same as the serial call regardless of
worker count. `wall_time_full_data` becomes the time spent waiting for
the already-running task after stage 1, rather than the time to run it;
its task time is added to `busy_time_total`. A draw whose stage 1 fails
never collects this task: it is left uncounted, exactly as the serial
code never reaches its own theta_hat call on that path. With `pool=None`
the call is made in place, as before.

## 3. The influence model: trend, width search and noise floor

Per output c, a Gaussian process through that output's finite
prototypes: I_j = h(w_j)^T beta + f(w_j) + e_j, f ~ GP(0, s^2 k),
e_j ~ N(0, s^2 lam) homoscedastic. Non-constant coordinates are grouped
by shared finite design so the width (or c) search runs once per
group.

**Trend.** h(x) is the affine basis (1, x) under `gptrend='affine'`,
or the quadratic basis (1, x, {x_a x_b}_{a<=b}) under
`gptrend='quadratic'`, both in whitened coordinates; m, the basis
size, is d_z+1 (affine) or 1+d_z+d_z(d_z+1)/2 (quadratic). Constant-
only (m=1) when the finite design has at most m+1 points, m the size
of the REQUESTED basis for the trend in use.

**The kernel.** Under `gpwidth='global'` k is the stationary
Matern-3/2, k_ell(r) = (1 + sqrt(3) r/ell) exp(-sqrt(3) r/ell), one
ell shared by every output in the group. Under `gpwidth='local'` k is
the non-stationary Paciorek-Schervish kernel: for two locations of
lengths ell, ell' at whitened distance r,

    k(r; ell, ell') = (2*ell*ell' / (ell^2+ell'^2))^(d_z/2)
                      * kappa( r*sqrt(2) / sqrt(ell^2+ell'^2) ),
    kappa(t) = (1 + sqrt(3)*t) * exp(-sqrt(3)*t),

d_z the dimension of the GP's input z (equal to d_x unless a
`vq_transform` changes it). This equals k_ell(r) mathematically when
ell = ell', but not bit-equal to it, since sqrt(2)/sqrt(2*ell^2) is
not bit-equal to 1/ell -- `gpwidth='global'` evaluates k_ell itself
for that reason, so its results do not depend on the local-width code.

**Local spacing.** Under `gpwidth='local'`, live prototype j's length
is ell_j = c*h_j, one factor c shared by the group, h_j the median
over j's CONN neighbours k (CONN, the symmetrized CADJ graph on live
prototypes) of the whitened distance ||z_j - z_k||. Every live
prototype has at least one CONN neighbour, so h_j is defined
everywhere: no floor, no minimum degree, no fallback (with one
neighbour the median is that distance). h is computed once, over
every live prototype (not per group), so a group's own h_j values are
a slice of it. A query point (a data point at any of fit, prediction
or bin-variance time) takes the length of its best-matching prototype,
ell(x) = ell_{bmu(x)}; the model stores bmu at fit time, since every
query in this package is at the draw's own N points.

**Width/c search.** Searched jointly for the group on
[param_min, param_max]: under `gpwidth='global'` directly on ell,
param_min the median whitened distance between CONN-connected
prototypes, param_max ten times the largest inter-prototype distance
(both over the group's own finite design); under `gpwidth='local'` on
c instead, with the same two bounds converted through the group's own
h: c_min = param_min / median_j(h_j) (so c_min is about 1), c_max =
param_max / min_j(h_j). Either way: five log-spaced candidates, then
one bounded refinement between the best grid point's neighbours,
SKIPPED when the best grid point is the upper endpoint. With the mean
basis projected out (Q, an orthonormal basis of its complement), the
Matern-3/2 expansion loses its low-order terms exactly, so past about
param_max the family collapses to one fixed kernel; the width is not
identified there and the refinement would only climb a shallow log-
determinant tilt. The outer objective at each candidate is the sum,
over the group's outputs, of the profiled restricted negative log
marginal likelihood, computed from one shared eigendecomposition of
the projected kernel per candidate: Q^T K Q's spectrum comes out of
A = (I-P) K (I-P) + tau*P (P = W W^T, W spanning the mean basis,
I-P = Q Q^T) in O(m_g M_g^2) rather than the O(M_g^3) of forming
Q^T K Q directly, since A's range-Q eigenpairs are exactly those of
Q^T K Q and its range-P eigenvalues are tau repeated m_g times;
tau = 2*M_g puts the m_g structural eigenvalues above every eigenvalue
of Q^T K Q (which lies in [0, M_g] since k(0,.) = 1), so `eigh`'s
first M-m pairs are the ones profiling needs. This search (and the
per-candidate eigendecomposition it shares across the group's outputs)
is the same one `gpwidth='global'` runs over ell; over c it is the
identical procedure with the non-stationary kernel in place of k_ell.
The grid's five candidates, across every group, run as pool tasks when
a pool is given; the bounded refinement between neighbours stays serial.

**Noise floor.** The prototype influences are finite differences of
accuracy eta, so I_proto carries real evaluation noise: noise sd
sqrt(2)*eta*|theta_Q,c|/t_j at prototype j. The declared homoscedastic
level is n_c^2 = 2*eta^2*theta_Q,c^2 * median_j(1/t_j^2) over
coordinate c's own finite design. lam_c is searched by REML on
[max(lam_floor,c, 1e-10), 1e2], where lam_floor,c is the unique root
of lam*s_c^2(lam) = n_c^2 -- s_c^2(lam) the profiled closed form
(1/(M-m)) sum_i z_i^2/(Lambda_i + lam), increasing in lam -- found by
one bracketed root find on log lam, pinned to the nearer domain edge
when the equation has no root inside it. The floor is recomputed at
every candidate width, since Lambda and z depend on it. For an estimator
with eta at machine precision the floor lies below 1e-10 and the
search is unchanged.

**Factorization failure.** Each per-output system A = K + lam I is
factored by Cholesky, retried with a ridge of 0, then 1e-10, 1e-9 and
1e-8 times the mean diagonal of K. If every attempt fails the draw's
stage 1 has failed: `QIJ.fit` returns a result with every variance
NaN and no stage-2 evaluations, a failed draw like any other.

**Point queries.** psi0(x) = h(x)^T beta + k(x)^T alpha from the
per-coordinate Cholesky solve. sigma and the within-bin posterior
variance v_k both read the group's shared eigendecomposition K = V
Lambda V^T instead of a per-output Cholesky solve of A_c = K +
(lam_c+jit_c) I: P = K_n V is formed once per group and every output
in it takes its own weighted row sums from it -- an equivalent
factorisation of the same quantity, agreeing with the Cholesky form to
rounding, not bit for bit, but never touching psi0 itself. No N x M
array of kernel rows k(x_i, w_j) is ever formed, cached or not: K_n's
rows are formed fresh in row chunks of bounded size (both here and at
`bin_posterior_variance`'s own s = sum_{i in bin} k_i below), used
within the chunk that formed them, and never held past it. v_k =
mean(diag Sigma_k) - mean(Sigma_k) is formed from bin-summed vectors,
never an n_k x n_k matrix: mean(diag Sigma_k) is read from the
already-computed sigma; mean(Sigma_k) needs s^T A^-1 s (through the
shared eigendecomposition, kept on the same side of the subtraction as
sigma's own factorisation, since v_k is a difference of two nearly
equal quantities) and SS_k = sum_{i,j in bin} k(x_i,x_j), the one term
no chunking shrinks the cost of. Under `gpwidth='global'`, for a one-dimensional
design, SS_k comes from running sums over the sorted points
(O(n_k log n_k)) rather than the pairwise double sum: with
c = sqrt(3)/ell and u_i = x_i - o against a local origin o,

    S_i = e^{-c u_i} * sum_{j<i} e^{c u_j},   T_i = u_i*S_i - e^{-c u_i} * sum_{j<i} u_j e^{c u_j}

are exclusive cumulative sums, re-based every block of about 200/c
points so no exponential overflows; the carried sums transform exactly
across a rebase, so no pair is dropped or approximated -- only the
summation order differs from the pairwise form. This shortcut assumes
a stationary kernel, so it does not apply under `gpwidth='local'`
(non-stationary): there, and in two dimensions or more under either
switch, SS_k stands as the pairwise double sum, chunked on both sides
so no block larger than 2048 x 2048 is ever materialized; the same
chunked pass also forms s = sum_{i in bin} k_i against the design, at
each bin point's own length ell(x_i) = c*h_{bmu(i)} under `gpwidth=
'local'`.

## 4. The I-VQ, bin stencil, and refinement

**Initial bins.** 1-D k-means (`kmeans_1d`) on psi0_c at M_init =
min(ceil(sqrt(2.7/eps)), n_distinct): quantile init, `searchsorted`
assignment against prototype midpoints, update by two `bincount`s
(counts, sums); an empty bin is dropped and the rest relabelled.
`bin_differences` then central-stencils each bin (section 1) against
the shared base value theta_hat; a NaN stops the loop at the failing
bin. V_btw = (1/N) sum_k p_k U_k^2 over the initial bins.

**Initial bins on a pool.** With a pool, every bin's +t and -t stencil
evaluations run as pool tasks against X (shared once, before this
coordinate's stencils run); each task carries its bin index, its signed
step, and the bin's own membership mask, and returns its evaluation, a
failure flag and its own wall time. The parent assembles them in bin
order into the same U the serial loop computes, and counts the same
evaluations the serial loop would have: every bin before the first
failure, plus that failing bin's own two evaluations (the central
stencil always evaluates both +t and -t, never stopping between them).
A later bin's task still ran on the pool and its wall time is added to
`busy_time_total` regardless, since the work was done; it is simply not
counted as an evaluation the draw needed. The refinement loop's own
split evaluations below always stay serial.

**Split rule.** Each open leaf is priced once at creation (its
Var(psi0) and mean of centered psi0 depend only on its own fixed point
indices, so both are cached rather than recomputed at proposal time
and again in the final gather) and proposed a split: a LEVEL split
(two-means on psi0, at M=2) when Var_k(psi0) > v_k, expected gain

    g_hat = rho^2 * (p_a*ubar_a^2 + p_b*ubar_b^2 - p_k*ubar_k^2) / N

otherwise an ADJACENCY split -- the bin's points whose second-nearest
X-VQ prototype carries a higher prototype influence than their
nearest, against the rest (falling back to a level split if one side
is empty) -- expected gain g_hat = rho^2 * p_k * v_k / N. A leaf with
one point, or no valid proposal, is closed. `v_k` is priced once for
every leaf as it is created, in one batched call to
`bin_posterior_variance` covering a whole split's two children (or all
the initial bins at once), never one call per leaf per round.

**Queue and measurement.** The open leaf with the largest g_hat is
taken (ties: lower leaf id); the loop stops when g_hat < tau =
eps*V_btw/(current leaf count), or the cost guard (1 + M_X_used
refinement evaluations) binds. The smaller child is measured by one
forward-differenced evaluation at t_small = step_parameter(delta_f,
p_small); the larger child's U is derived by mass balance:

    U_large = (p_parent*U_parent - p_small*U_small) / p_large

Realized gain Delta = (p_small*U_small^2 + p_large*U_large^2 -
p_parent*U_parent^2)/N; V_btw += Delta.

**Gain pricing (gamma).** Both children of a split take gamma =
clip(Delta/g_hat, 0, 1) when g_hat is strictly positive and both
quantities are finite, else 1 (no usable information, so the within
term is not down-weighted). gamma scales the whole V_win_hat bracket
for that leaf: a low gain ratio is measured evidence the bin held no
variation the model successfully predicted, and the model's own
uncertainty v_k is part of the prediction being discounted.

**Closing rule.** Each leaf carries a strike flag, clear on the
initial bins. If Delta >= tau (the tau at selection), both children's
flags are clear and both are re-proposed. If Delta < tau and the
parent's flag was clear, both children's flags are set and they are
still re-proposed. If Delta < tau and the parent's flag was already
set, both children are closed outright. One below-tolerance split
cannot distinguish a bin with constant influence from one whose
variation split evenly between its children; a second consecutive one
can. A failed split evaluation cancels the split: the parent stays a
closed bin, and the rest of the refinement proceeds.

**V_win_hat.** Over the final bins with more than one point:

    V_win_hat = rho^2 * (1/N) * sum_k p_k * gamma_k * [Var_k(psi0) + v_k]

using each final bin's cached v_k from the round that created it --
never a fresh `bin_posterior_variance` call over the final bin set.
rho^2 = V_btw / ((1/N) sum_k p_k * ubar_k^2), recomputed over the
current bins after every accepted non-closing split; ubar_k is the
mean of centered psi0 over the bin. V_tot_hat = V_btw + V_win_hat.

## 5. GMM2D

`gmm.py`'s 2-D Gaussian-mixture estimator: K components, free full
covariances, fit by multi-start weighted EM (SQUAREM-accelerated),
polished with a damped Newton step gated on the observed information's
positive-definiteness, with an analytic influence via Louis's (1982)
identity.
`GMM2D(K, n_starts=20, seed=0, tol=1e-8, max_iter=500, reference=None)`;
`T(X, w) -> ndarray(p,)` never raises (a failed fit is NaN); no state
between calls; `T.influence(X, w) -> ndarray(N, p)`;
`T.fit_and_influence(X, w) -> (ndarray(p,), ndarray(N, p))` from one
fit.

**The estimand.** The free-covariance mixture likelihood is
unbounded (a component can collapse onto a few points), so `GMM2D`'s
estimand is the maximizer of the penalized log-likelihood

    ell_p(theta; X, w) = ell(theta; X, w) - a * sum_k[tr(S Sigma_k^-1) + log det Sigma_k],
    a = 1 / sum(w)  (= 1/N at unit weights),
    S = the w-weighted covariance of the rows passed to T,

the penalty function of Chen and Tan (2009, *Inference for multivariate
normal mixtures*, J. Multivariate Anal. 100(7):1367-1383, arXiv:0805.3906,
their eq. 2 with their `a_n = n^-1`, one of their two recommended
choices; the M-step below is their closed form). It is always on, for
every call and every fit -- no conditional switching -- since it is a
Bayesian Wishart-prior-on-Sigma_k device that makes the likelihood
surface bounded, not a diagnostic that should come and go with `w`. It
is of relative order `1/N`, so it moves a well-behaved fit by less than
its sampling error, and it vanishes as `N` grows. `S` is defined from
the rows and weights `T` is actually called with, so the quantized-data
fit and the full-data fit use the same definition, and `S`'s own
w-weighted mean is generally not the zero that `X`'s UNWEIGHTED mean is
(see **Centering** below) -- it is recomputed, from scratch, every fit.
Every closed form below (M-step, score, Hessian, influence) is that of
`ell_p`, not of the ordinary mixture log-likelihood `ell`; internally,
`ll` always means `ell_p` at the current point, per unit weight
(`ell_p / sum(w)`).

**Layout.** Parameter vector length `p = (K-1) + 5K`: the first `K-1`
mixing weights (`pi_K = 1 - sum`), then every component's mean
(`mu_1x, mu_1y, ..., mu_Kx, mu_Ky`), then every component's three free
covariance entries (`S_1_11, S_1_12, S_1_22, ...`), each component's
means-then-covariance grouped, component-major. `_pack`/`_unpack` are
the only two places that need this order.

**E-step.** With `Sigma_k = [[a,b],[b,c]]`, `det_k = a*c - b^2`,
`log pi_k + log phi_k(x)` is a fixed linear combination of the six
monomials `[x, y, x^2, xy, y^2, 1]`, so one `(N,6)` feature buffer `Q`
(built once per fit, from the centered X) against a `(6,K)` coefficient
matrix rebuilt each iteration gives the whole `(N,K)` log-density array
in a single matmul (`_log_density_coeffs`, `_e_step_fast`). `Q`'s first
five columns, in the same order, are the M-step's `(N,5)` moment matrix
`XP = Q[:, :5]`, a view, not a second array built over the same data.

**Multi-start**, drawn once from `X` alone, deliberately blind to `w` so
a weight perturbation cannot move the starting points. Each of the
`n_starts` starts is a k-means clustering of `X` into `K` clusters
(`scipy.cluster.vq.kmeans2`, `minit='++'` seeding, Lloyd iterations run
well past the point the assignments stop changing): a start's means are
the cluster centroids, its mixing weights the cluster fractions, and
each component's covariance is that cluster's own population covariance.
A cluster with fewer than three distinct points, or whose points are
collinear, has a singular covariance; that component's start covariance
falls back to the unweighted data covariance scaled by `1/K` instead, so
no start is degenerate at birth. Successive starts draw their k-means
seeding from the same `np.random.default_rng(seed)`, so they differ from
each other and the whole sequence is reproducible in `(X, seed)` alone.

Two phases, shaped differently on purpose. Both use the penalized
M-step (below) for Sigma_k -- there is no unpenalized code path.
Phase 1 (`_phase1_batch`) runs every start for a short, fixed
`_SHORT_ITERS` budget in lockstep, batched as one `(N,6) @ (6, K*S)`
E-step matmul and one batched M-step per iteration (`S = n_starts`),
with no per-iteration convergence test, degeneracy check, or `ll_p`
computation (only the trailing one, after the loop, for ranking): a
start whose covariance goes non-PD mid-phase just turns its own
coefficient columns to NaN/Inf, confined to that start's own block by
the batched matmul's column layout. Degeneracy is checked once, after
the loop, per start; any start that fails is dropped before ranking.

Phase 2 (`_run_em`, via `_fit_em_multistart`) takes the best
`_PROMOTE_N` of phase 1's survivors, by penalized weighted log-
likelihood (`ell_p / sum(w)`), and runs each one at a time,
sequentially, to the caller's own `max_iter` EM steps (or convergence,
whichever comes first) via SQUAREM acceleration (below), checking
`ell_p`'s own relative-tolerance convergence after every SQUAREM round
-- EM's monotonicity guarantee (below) is a monotonicity in `ell_p`, so
that is what its own stopping rule must watch. The selection pool is every surviving start, converged or not:
hitting the iteration budget means the log-likelihood surface is
locally flat there, not that the start found a bad point, and treating
a budget-exhausted start as a failure would make pool membership a
step function of `w`, incompatible with `T` needing to be
differentiable in `w`. The pool is ranked by penalized weighted log-
likelihood; the best is canonically relabeled (`_canonical_sort`,
ascending `mu_x`, ties on `mu_y`) BEFORE Newton polish, so `psi`/`A`
need no further permutation -- without this, two starts landing on the
same optimum with swapped labels would make `T` discontinuous in `w`.
`_SHORT_ITERS = 25`, `_PROMOTE_N = 3` are chosen by measurement (not
taste): validated to not move the winning start or `theta_hat` relative
to running every start to full budget.

**The penalized M-step** (`_m_step`, `_m_step_batched`). `pi`, `mu` are
the ordinary weighted-mean M-step, untouched by the penalty. `Sigma_k` is Chen
and Tan's closed-form inverse-Wishart MAP,

    Sigma_k <- (sum_i r_ik w_i (x_i - mu_k)(x_i - mu_k)^T + 2a S) / (sum_i r_ik w_i + 2a),

a convex blend of the ordinary weighted covariance about the just-
updated `mu_k` (weight `n_k = sum_i r_ik w_i`) and `S` (weight `2a`);
`a -> 0` as `N -> infinity` recovers the ordinary M-step exactly. EM
with this M-step is monotone increasing in `ell_p` (Chen and Tan 2009):
their argument is the ordinary EM monotonicity argument applied to
`ell_p`'s own complete-data form, since the added term is a fixed
(theta-independent-in-the-E-step) log-Wishart-density constant during
the E-step and a concave function of `Sigma_k` whose maximizer, given
the E-step's responsibilities, is exactly the closed form above.

**SQUAREM acceleration** (`_squarem_round`, `_em_accelerated`,
`_run_em`; Varadhan and Roland 2008, *Simple and globally convergent
methods for accelerating the convergence of any EM algorithm*, Scand.
J. Statist. 35:335-353, their steplength scheme S3). From the current
point `theta0`, two plain EM steps (`_em_step`, one E-step then the
penalized M-step above) give `theta1`, `theta2`; `r = theta1 - theta0`,
`v = (theta2 - theta1) - r`; step length `alpha = -norm(r)/norm(v)`,
held at most `-1` so the step is never shorter than two EM steps (at
`alpha = -1` exactly, `theta0 - 2*alpha*r + alpha^2*v` reduces
algebraically to `theta2`), and at least `-m`, with `m` starting at 4
for each start's EM and multiplied by 4 each time that limit binds. The
extrapolated point followed by one more EM step is the candidate;
discarded for `theta2` if it leaves the feasible set (a mixing weight
<= 0, a component covariance not PD) or its `ell_p` is below `theta2`'s
-- so the accepted point's `ell_p` is always >= `theta2`'s, hence
monotone in `ell_p` exactly as plain EM is, with no other constant or
retry. A round costs 2 EM steps (candidate rejected or infeasible) or 3
(accepted); `max_iter` is a budget in EM steps, not rounds, so a round
near the end of the budget skips the candidate trial rather than
overshoot it. On a slowly, near-linearly converging trajectory (the EM
map close to critical, `v` nearly proportional to `r`) the raw `alpha`
is large in magnitude, which is where the long step pays; the limit
`-m` grows as it binds, and the monotonicity safeguard rejects any
candidate that does not improve on `theta2`. Phase 1's batched screen is not
accelerated (it has no per-start convergence test to accelerate against
-- a fixed lockstep budget, used for ranking, not for finding a root).

**Newton gate** (`_cholesky_ok`, `_fit`). After phase 2's winning start
is labelled and its penalized score/information `(psi_bar, A)` formed,
the polish below runs only if `A` (`-H` of `ell_p`) is positive
definite (`np.linalg.cholesky` succeeds): a start that is not there yet
is not one the local quadratic model of Newton's step should be trusted
on. If `A` is not PD, one more block of SQUAREM-accelerated EM (the
same `max_iter` cap, continuing from the current point) runs, the start
is re-labelled and its `(psi_bar, A)` re-formed, and the PD test is
repeated once. Still not PD: the fit is NaN (below), no third block, no
other retry. Enforcing this before the FIRST Newton step matters: a
start whose EM has not actually reached a stationary point can have a
PD `A` and yet leave PD-ness after just one full Newton step (the local
quadratic model is only locally valid), producing exactly the
plateaued-score, many-halvings symptom the gate is meant to keep out of
a WINNING start's polish; a per-step re-test of `A`'s own PD-ness is not
part of the polish itself (only a mixing weight <= 0, a component
covariance not PD, or `ell_p` falling triggers a halving, all below).

**Newton polish** (`_fit`) takes up to `_MAX_NEWTON = 20` damped Newton
steps `theta <- theta + t * A^-1 (penalized weighted mean score)`, using
the same analytic score/information the influence needs, both the
penalized ones (below). At each iteration the
full step (`t = 1`) is tried first; it is accepted if the new point is
valid (every mixing weight positive, every component's covariance
positive-definite, penalized log-likelihood finite) and the penalized
weighted log-likelihood does not decrease, otherwise `t` is halved and the trial
repeated, up to `_MAX_HALVINGS = 30` halvings (`t` as small as
`2^-30 ~ 1e-9`). A full step from an unconverged EM point is a step from
outside the region the Newton model is locally accurate in, and
routinely overshoots a small mixing weight past zero or a covariance
past positive-definiteness; damping recovers a usable step there while
leaving a polish that already accepts the full step at every iteration
identical to before. If no step at any damping level is admissible, the
polish stops and the current point is kept. The loop otherwise stops
when the penalized weighted mean score's norm falls to `self.eta` or
below.

**Residual and `eta`.** `resid = norm(s_bar)` at the point the polish
loop exits, `s_bar` the penalized weighted mean score -- the same
quantity, at the same scale, the loop's own stopping test and the Newton
gate above both watch. `resid > self.eta` (or `A` singular at any point
along the way, so the gate or a Newton step cannot even be evaluated)
makes the whole evaluation fail (`T` and `influence` both NaN): a
`theta_hat` whose score is not this small is not one QIJ should treat as
an exact root of `ell_p`. `eta` is declared, not derived: `self.eta =
1e-12`, from the score norm the polish reliably reaches on the demo
mixture (`datasets.mix11`, K=9) once a start clears the gate and
actually converges (module report: two of its three phase-2 starts
converge to a `resid` of a few times `1e-13`; the third clears the gate
but does not converge within `_MAX_NEWTON` steps and would itself be
NaN, which is not "reliably reached" and is excluded from the
declaration), rounded up to `1e-12`. A fit whose Newton polish does not
converge at all fails `resid > eta` by orders of magnitude (`1e-3` or
larger), not by a margin that would call `eta`'s value into question.

**The score and Louis's identity** (`_score_info`). Per observation,
with responsibility `r_k`, residual `res = x - mu_k`,
`P_k = Sigma_k^-1`, `G_k = 0.5*(P_k res res^T P_k - P_k)`, all of the
RAW (unpenalized) mixture likelihood -- the penalty's terms are added
separately, below, since they are functions of `(Scov, a)` alone, not
of any one observation:

    d/dpi_j  = r_j/pi_j - r_K/pi_K                    (j = 1..K-1)
    d/dmu_k  = r_k * P_k @ res
    d/dS_k11 = r_k * G_k[0,0];  d/dS_k22 = r_k * G_k[1,1]
    d/dS_k12 = r_k * 2 * G_k[0,1]           (E_12 + E_21 duplication)

`A = (1/N) sum_i w_i I_i` by Louis's identity, `I_i = sum_k r_ik B_ik -
sum_k r_ik s_ik s_ik^T + s_i s_i^T`, with `s_ik` the complete-data score
(`r_k` replaced by an indicator) and `B_ik` its complete-data observed
information. `B_ik` is block-diagonal across components but, within
component `k`, carries the textbook `P_k` mu-block, the textbook
diag/all-ones pi-block, AND a mu-Sigma cross block
(`d(P_k res)_a / dS_b = -(P_k @ D_b @ P_k @ res)_a`) -- included because
it is genuinely nonzero at any one observation (only its EXPECTATION
over `res` vanishes, which is not what an observed-information Hessian
evaluates); omitting it costs Newton's quadratic convergence.

`sum_k r_ik B_ik` is never materialized as a per-observation
`(N,K,p,p)` tensor: every entry of `B_ik`'s nonzero (pi-pi and
component-`k`'s own 5x5) blocks is at most quadratic in `res`, so
`sum_i w_i r_ik B_ik` reduces exactly to `n_k` times a small matrix
built from the weighted moments `n_k = sum_i w_i r_ik`,
`rbar_k = (1/n_k) sum_i w_i r_ik res_i`,
`R2_k = (1/n_k) sum_i w_i r_ik res_i res_i^T` -- no `(N,p,p)` array, at
any K. The `-sum_k r_ik s_ik s_ik^T` term keeps a small per-observation
array per component (not `(N,p)`) because it is NOT reducible to
moments (it is the outer product of the full per-observation score, not
its expectation); it is assembled with one matmul per component.

**The penalty's terms** (`_penalty_terms`, `_score_info_penalized`). The
raw `(psi, A, ll)` above are combined with the penalty, which touches
only the Sigma_k blocks (pi, mu are untouched: the penalty is not a
function of them). With `P_k = Sigma_k^-1`, `G_k = P_k - P_k Scov P_k`
(the penalty's own gradient w.r.t. `Sigma_k`, packed into the
`(S11,S12,S22)` parametrization the same `d/dS_k12 = ... + ...`
duplication way as the raw score above):

    penalty_sum  = sum_k[tr(Scov P_k) + log det Sigma_k]
    g            = -a * G_k                       (p,, zero outside Sigma blocks)
    H[a,b]       = -a * tr(D_a @ dG_k/dD_b),  dG_k/dD_b = -P_k D_b P_k + P_k D_b P_k Scov P_k + P_k Scov P_k D_b P_k

(`D_b` the same three basis matrices `_D['S11'/'S12'/'S22']` the raw
Hessian's `PDP` already uses; `H` is block-diagonal across `k`, since
`Scov` does not depend on theta and different components' Sigma_k's
don't interact through the penalty). Then

    ll     = ll_raw - a*penalty_sum/W        (ell_p / sum(w))
    psi_bar = psi_bar_raw + g/W              (d(ell_p/sum(w))/dtheta)
    A      = A_raw - H/N                     (the penalized observed information / N)

matching the raw code's own normalization convention exactly (`psi_bar`
by `sum(w)`, `A` by `N`) so Newton, the residual, and `influence`'s
sandwich all stay self-consistent with how they already combined with
the raw terms.

**The influence's extra term** (`_penalty_influence_extra`). The
penalty depends on `w` not just through `theta_hat` (already captured
by the raw `psi`) but directly, through `S(w)` and `a(w)`: the M-
estimator's implicit-function-theorem sensitivity `d(penalized
score)/dw_i`, at fixed theta, therefore has a term beyond the raw
`psi_i`. With `d_i = x_i - xbar_w` (the row's residual from `Scov`'s OWN
w-weighted mean, not `prepare`'s unweighted `xmean`), `da/dw_i = -a^2`
(constant in `i`) and `dScov/dw_i = (d_i d_i^T - Scov)/W` (both closed
form, the second because `sum_i w_i d_i = 0` exactly by definition of
the weighted mean), the extra per-point, per-component term is

    Delta_k(w_i) = a^2 * G_k + (a/W) * P_k (d_i d_i^T - Scov) P_k,

packed into the theta vector the same way `g` is, and `influence`'s
per-observation term is `psi_full_i = psi_raw_i + sum_k Delta_k(w_i)`
(its weight sum(w) is not what `influence` returns -- `psi_i = N *
dT/dw_i` is, per the package's convention, and `A^-1 psi_full_i` gives
exactly that for the SAME reason `A^-1 psi_raw_i` already did for the
unpenalized estimator: `A` is `-Hessian(ell_p)/N`, and `psi_full_i` is
`d(SCORE_p_total)/dw_i` at the unnormalized `ell_p` scale). The
detector for a missing or wrong term here is the same as for the raw
score: Newton stops converging quadratically, or (checked separately,
by direct perturbation of `w`) the influence's first-order prediction
of `T(X, w + t*e_i) - T(X, w)` stops improving by the expected ~100x
per 10x shrink in `t`.

**Centering.** `X` is shifted by its own UNWEIGHTED mean once on entry
(fit and influence both computed in the shifted frame; component means
shifted back before `T` returns them) -- `Sigma = E[xx^T] - mu mu^T`
loses digits catastrophically once the data sits far from the origin,
and this module is Newton-polished to ~1e-12. The shift must be a
constant of `w` alone, or it would become part of what the influence
function measures. Covariances (and therefore `psi`, `A`, the
influence) are exactly invariant under a constant shift, so `influence`
never needs to shift anything back. `Scov` (the penalty's `S`) is a SEPARATE
quantity, recomputed fresh every fit from `(Xc, w)` about `Scov`'s own
w-weighted mean (`_weighted_cov`) -- unlike `xmean`, that mean is not
zero in general (it moves with `w`) and is not the shift `T` centers
on; it exists only to build `Scov` and the influence's `d_i` residuals.

**`prepare(X)`** holds the unweighted centering shift `xmean`, the
centered data `Xc = X - xmean`, and the feature buffer `(Q, XP)` built
from `Xc` -- all independent of `w`, and otherwise rebuilt from scratch
on every one of a draw's many bootstrap replicates / QIJ perturbations
of the same X.

**Failure convention.** The evaluation fails (NaN) when: every one of
the `n_starts` starts degenerates, so nothing survives phase 1 to rank;
or the winning start's `A` is not PD even after the Newton gate's one
retry block; or, after polish, `resid > self.eta` (nothing retried
beyond the polish's own `_MAX_NEWTON` iterations) or `A` becomes
singular along the way; or (`influence`/`fit_and_influence` only) `A`'s
condition number exceeds `_COND_MAX = 1e12`. A bad OTHER start's mid-EM
degeneracy just drops that one start -- multi-start's whole purpose. A
start whose own EM never reaches `_run_em`'s relative-tolerance
convergence within `max_iter` EM steps is not itself a failure (it is
finalized `converged=False` and still ranked, above); but if the
winning start then fails the Newton gate or its polish cannot bring
`resid` under `eta`, the whole evaluation is NaN -- a real outcome for a
hard mixture (K too large for N, near-empty or near-collinear clusters),
not an artifact of the penalty.

**Component labelling.** Without a reference, the winning start's
components are labelled in ascending mu_x order. With
`reference=(means (K,2), covs (K,2,2))`, they are labelled by the K x K
assignment (`scipy.optimize.linear_sum_assignment`) minimizing the total
Bhattacharyya distance to the reference components,

    D_B(a, b) = (1/8)(mu_a - mu_b)^T Sigma_bar^-1 (mu_a - mu_b)
                + (1/2) ln(det Sigma_bar / sqrt(det Sigma_a det Sigma_b)),
    Sigma_bar = (Sigma_a + Sigma_b)/2.

The covariance term separates a spike from the blob sharing its mean; the
mean term separates equal-covariance components. The labelling is applied
where the components are ordered, before the score, Hessian and Newton
polish, so it fixes the whole parameterization (including which
component's weight is implicit), not a permutation of a finished theta.
The reference is fixed at construction and every call uses it. A caller
wanting output c to be the same component throughout a draw fits once on
the full data without a reference, converts that fit with
`reference_from_theta(theta, K)`, and builds the estimator for every other
evaluation of the draw with it; the naive ordering remains the default.

## 6. The joint second stage

Under `ivqbins='joint'` (`core/joint.py`), stage 2 replaces the marginal
path's per-output 1-D quantizer and refinement queue by one partition
shared across every output: grown to the tolerance (`grow`), measured on
the full data, then corrected by a measured check against that
measurement. `ivqbins='marginal'` is section 4 in full and is untouched
by this section; stage 1 (sections 2-3) is untouched by the switch.

**Standardization and the shared failure rule.** Ψ̃ = psi0_all with
each output divided by its own standard deviation sd_c over the N
points, V_hat_c = sd_c^2 the predicted total variance of output c. An
output on the constant path (section 3) has no usable psi0 spread; since
every output shares one partition, such an output fails the whole draw
(`failed=True`, no growth attempted), rather than voiding only its own
coordinate as the marginal path does.

**Growth.** From one bin holding every point, a round computes, for
every current bin k and output c, w_kc = p_k * Var_k(psi0_c) / V_hat_c
(population variance over the bin's own points, 0 for a bin of one
distinct row) and S_c = sum_k w_kc, the predicted within share of
output c. Growth stops when every S_c <= eps. Otherwise every bin with
max_c w_kc > eps/L (L the current bin count) is split, in the same
round, by two-means on Ψ̃ (`two_means_split`): initial centroids are the
means of the two halves of the bin's rows split at the median of their
projection onto the bin's own first principal axis (an eigh of the
bin's q x q covariance), refined by Lloyd iterations to convergence or
100 iterations, deterministically. A bin whose two-means is infeasible
(fewer than two distinct rows) is left whole for the round; the
tolerance test guarantees this never blocks progress, since a bin
selected for splitting has w_kc > 0 for some c and therefore at least
two distinct rows. Growth also stops, `growth_capped` recorded, once L
reaches M_X_used -- checked only after confirming the tolerance is
still unmet, so a round that both converges and reaches the cap counts
as converged, not capped. A Lloyd pass then reassigns every point to
its nearest of the L centroids at once, to convergence or 100
iterations, dropping any centroid this empties; it lowers the predicted
within share at no evaluation cost and brings it close to the within
share the true influence has on the same bins. S_pred is read after
this pass, S_pred_pre_lloyd before.
Growth spends no evaluations and never runs on `pool`.

**Measurement.** The grown partition is measured by the existing
`ivq.bin_differences` central stencil, one task pair per bin on `pool`,
giving each bin's centered U_k in R^q from the same two evaluations for
every output; a NaN anywhere fails the whole draw (section 4's rule,
ported as is since every output already shares these bins).
V_btw,c = sum_k p_k U_kc^2 (`ivq.between_terms` per output).

**The measured check.** For every bin k and output c: m_kc = mean of
psi0_c over the bin's points (the model's predicted bin mean); u_kc =
mean(Sigma_k), the posterior variance of that mean
(`influence_model.bin_posterior_variance(..., with_mean=True)`, the
same chunked pairwise pass that prices v_kc, the marginal path's
expected within-bin variance -- both returned together, since u_kc is
exactly the `mean_Sigma` term v_kc's own formula already computes
before its diag-mean subtraction). A single scale a_c per output,

    a_c = sum_k p_k * U_kc * m_kc / sum_k p_k * m_kc^2,

is fitted once, after this first measurement, from every bin, and held
fixed through every check round: a model whose scale is off but whose
ordering is right (section 3's noise floor can leave psi0 on the wrong
scale by orders of magnitude while still recovering most of the oracle
variance) must not read that uniform error as per-bin disagreement.
Bin k is flagged when, for some output c,

    p_k * (U_kc - a_c*m_kc)^2  >  eps * V_hat_c / L  +  p_k * a_c^2 * u_kc,

the measured derivative differing from the rescaled prediction by more
than the bin's share of the tolerance plus what the posterior allows
for the error of a bin mean. There is no tau, no closing-strike flag, no
expected-gain queue, no rho^2 rescaling here: gain_ratio (below) is a
reported diagnostic, not a stopping rule.

Each check round, every currently flagged bin is priced and split in
one batch: the split kind is level when
sum_c Var_k(psi0_c)/V_btw,c >= sum_c v_kc/V_btw,c, else adjacency,
falling back to the other kind when the first choice is infeasible
(both infeasible closes the bin, with no further evaluation). A level
split is `two_means_split` on the bin's own Ψ̃ rows, exactly as growth's
own split. An adjacency split evaluates, for every output c, the ported
CADJ proposal (`refine.adjacency_split_gain`, the prototype-influence
ordering shared with the marginal path) and takes the output whose
proposal has the largest expected gain divided by that output's V_btw,c
(both proposal and gain use rho^2 = 1, since the joint path has no
rho). This ordering needs I_proto, the survey's per-prototype
influence; `run_joint` takes it as an optional argument that `QIJ.fit`
does not yet supply, so until that wiring is added every adjacency
proposal is infeasible and the fallback routes every split through the
level kind. The smaller of the chosen split's two children is measured
by one ported forward evaluation; the larger's U follows by mass
balance, exactly as the marginal path's own split (section 4); every
flagged bin's evaluation in a round runs through `pool` together.
V_btw,c is then updated (the split parent's term replaced by the sum of
its two children's) and each new child is checked by the same flag
test at the new, larger L; a failed evaluation cancels its split (the
parent stays a final bin, closed, its one evaluation still counted).
The check stops when no bin is flagged, or when check evaluations reach
1 + M_X_used (`check_capped`).

**Products (B6).** Per output: V_btw, V_win_hat, V_tot_hat = V_btw +
V_win_hat; V_win_hat = (1/N) sum_k p_k (Var_k(psi0_c) + v_kc) over the
final bins, v_kc cached from the round that priced it (never a fresh
posterior-variance pass over the final bin set), with no rho^2 and no
gamma_k -- the joint path has neither. S_pred, S_pred_pre_lloyd, a_c as
above. gain_ratio,c = sum of realized Delta_c over sum of expected g_c
across every check split (both at the split's own chosen partition and
kind, g_c using rho^2 = 1; NaN when output c had no check splits).
Shared: L0 (bins after growth), L (final), n_growth_rounds,
growth_capped, n_flagged (bins flagged by the first check), n_check_
rounds, n_check_evals, n_level_splits, n_adjacency_splits, check_capped,
failed. Bin constituents bin_mass, bin_U, bin_m (predicted means),
bin_flagged (true only for a bin still flagged when check_capped
stopped the check, since a bin closed for any other reason is never
revisited); per point, the shared bin_label. The q x q between-bin
matrix is recoverable from bin_mass and bin_U and is not stored.

## 7. Continuations and the ABC interval

**The estimator protocol.** An estimator whose fit depends on restarts
(a multi-start search, `gmm.GMM2D`) carries `takes_start = True` and
accepts `T(X, w, start=theta)`, theta in its own output layout (q,):
its usual iterative fit continued from `start` at the given weights,
then its usual polish and acceptance rule, with no multi-start. Without
`start`, T is unchanged. `Counter(X, w, start=None)` and
`parallel.call_T(T, X, w, start=None)` both pass `start` to T only when
it is given and `T.takes_start`; every other estimator (Pareto, MVT,
the Fundamental Plane, Chabrier) never sees it and runs exactly as
before.

**Why a continuation.** A finite difference of T needs its two
evaluations on ONE branch: for an estimator with restarts, "the best of
n_starts starts" is a discontinuous function of the weights wherever
the winning start changes, which on the quantized survey rows can
happen at every perturbation. Every perturbed survey evaluation, and
every full-data stencil evaluation of stage 2 (the initial bin
measurement, a refinement split, a joint check split), therefore
continues from a base fit rather than restarting: `start=theta_Q` on
the survey and the curvature stage, `start=theta_hat` on the full data,
passed down from `QIJ.fit` and threaded unchanged through
`ivq.bin_differences`, `refine.run_refinement` and `joint.run_joint`.
`start` is never passed to the base fits theta_Q/theta_hat themselves
(both keep their own multi-start search) or to the bootstrap (not a
derivative).

**theta_Q's own starting point: `quantized_start`.** Two values.
`'multistart'` (default): theta_Q is fit from scratch on the survey
rows, as every other quantity. `'full-data'`: theta_hat is evaluated
first (on `pool` when given, so it overlaps with the rest of stage 1's
own work) and theta_Q is its continuation onto the survey rows,
`start=theta_hat`. Either value leaves everything downstream of theta_Q
-- I_proto, the influence model, stage 2 -- exactly as built; the
switch only changes the point theta_Q's own fit starts from.

**Reproducibility of the continuation.** Before the prototype survey
runs, for an estimator with `takes_start`, eta_Q is measured directly
on the survey rows: two fits from theta_Q at the base weights (which
agree exactly for a deterministic continuation) and one at the
largest-mass prototype's weights perturbed by a relative 1e-6 and back,
`start=theta_Q` throughout; eta_Q is the largest relative difference
among the three returned parameter vectors, floored at T's own declared
eta. The survey's forward step then uses delta_f = 2*sqrt(eta_Q) in
place of eta (section 1). A self-check, reported and not gated, repeats
the step on the five largest-mass prototypes at delta_f and 2*delta_f
(`survey_step_ratio`, (5,q)): the ratio of the two raw responses per
output is near 2 for a response that is genuinely a derivative. For an
estimator without `takes_start` neither eta_Q nor the self-check is
measured (both NaN): the survey step stays exactly eta's own delta_f,
and no extra evaluation is spent, so every count and value elsewhere is
unaffected.

**The curvature stage.** After stage 1, before the survey's shared rows
are replaced by the full data for stage 2, `core.abc.curvature` prices
the ABC interval's curvature ingredient c_q, for every estimator, in
its own evaluation stage (2q evaluations, on the survey rows). It works
entirely within the survey's own M_X-cell problem: sigma_Q,c =
sqrt((1/M_X) sum_j p_j*I_jc^2), the least-favorable direction u_j =
I_jc / (sqrt(M_X)*sigma_Q,c), a step eps_c = min(eta_Q^(1/4),
0.5/max_j|u_j|), and two evaluations at weights omega0*(1 +/- eps_c*u_j)
on every row of field j, `start=theta_Q`. The raw central second
difference at that step, divided by (eps_c^2 * M_X), gives
2*sigma_Q,c*c_q, so c_q = raw / (2*M_X*sigma_Q,c). Because this is a
second-order correction (an error of a few percent moves an ABC
endpoint by a few percent of a 1/sqrt(N) term) it is priced on the
survey's cheaper problem rather than the full data.

**Bias and acceleration.** `ivq.bias_and_acceleration(binset, N)` reads
the ABC interval's other two ingredients off a measured `BinSet`'s
centered U and its (uncentered) central-stencil second difference d2T:

    B_hat_c = (1/(2N)) * sum_k p_k*(1-p_k)*d2T_k
    a_c     = (1/(6*sqrt(N))) * sum_k p_k*U_kc^3 / (sum_k p_k*U_kc^2)^(3/2)

(a_c NaN when an output's bins carry no between-bin variance). Computed
once per draw and coordinate, right after the first full-data
measurement of the relevant bins succeeds -- the marginal path's own
initial bins, before any refinement split; the joint path's shared bins
as measured right after growth, before any check split -- since only a
central, three-point stencil has a genuine second difference: a later
split measures only its smaller child by a one-shot forward evaluation
and derives the other by mass balance, with no third point to
difference against. `QIJResult.a`/`.b_hat` collect a_c/B_hat_c per
output (from `CoordinateResult.a_bca`/`.B_hat` under
`ivqbins='marginal'`, from `JointResult.a_bca`/`.B_hat` under
`'joint'`); both are voided to NaN wherever V_btw/V_win_hat/V_tot_hat
are (a failed coordinate, output, or draw).

**The ABC_q interval.** `QIJResult.abc_interval(level)` calls
`core.abc.abc_interval(theta_hat, sqrt(V_btw), a, b_hat, c_q, level)`:
per output, z0 = a + c_q - b_hat/sigma; at each tail's standard-normal
quantile z_alpha, w = z0+z_alpha, lambda = w/(1-a*w^2), xi = lambda +
c_q*lambda^2, endpoint = theta_hat + sigma*xi. At a = b_hat = c_q = 0
this reduces exactly to the normal interval `QIJResult.interval`
already computes from theta_hat and V_btw alone.

**The bootstrap's bias-corrected intervals.** `Bootstrap.fit` evaluates
theta_hat = T(X, ones(N)) once, on the pool when given, outside the
replicate loop's own RNG stream, and stores it on `BootstrapResult`.
`.bc_interval(level)` calls `core.abc.bc_interval(replicates, theta_hat,
level)`: z0 = Phi^-1(share of replicates below theta_hat, per output),
and the endpoint at tail z_alpha is the replicates' own
Phi(2*z0+z_alpha) quantile. `core.abc.bca_interval(replicates,
theta_hat, a, level)` is the same construction with a GIVEN
acceleration in place of 0 (endpoint percentile Phi(z0 + w/(1-a*w)),
w = z0+z_alpha); it never estimates its own acceleration (no jackknife
runs anywhere in this package) and is used only by a caller that
already has one, such as an analytic influence's own a_c.
