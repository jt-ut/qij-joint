# ShearMix2D and cloudfil_G_U_P3_v1: shared build interface

Status: build interface for three parallel agents (A, B, C), 3 Oct 2026.
Branch `shearmix`, worktree `JT_Py_Pkgs/qij_joint_shearmix`. Every agent
reads this file whole; signatures, names and layouts here are FIXED. If an
agent needs something not here, it states the assumption in its report
rather than changing another agent's file.

## 0. Why

The demo dataset `cloudfil_G_B6_P3_v1` builds its curved filament from six
straight Gaussian beads that flare off the curve at the bends. The author's
replacement: ONE smooth sheared-Gaussian filament, with an estimator whose
model CONTAINS that component (well-specified, so the truth is the
generating parameters; the author rejected fitting a plain GMM to it
because that "conflates interval issues with model misspecification").
The cloud and the three cores are unchanged.

## 1. The model

Density of one row x = (x, y):

    f(x) = sum_{k=1..Kg} pi_k N(x | mu_k, Sigma_k)  +  pi_f f_fil(x)
    f_fil(x, y) = N(x | m, s2x) * N(y - h(x) | 0, s2p)
    h(x) = b0 + b1 sin(omega x) + b2 cos(omega x),   b(x) = (1, sin omega x, cos omega x)

Kg = 4 free full-covariance Gaussians (cloud and three cores), pi_f =
1 - sum pi_k. omega is FIXED (not estimated): constructor argument,
default `OMEGA = np.pi / 2.64`. The basis uses RAW data x (no centering
anywhere in the filament; do not center X for this estimator).

The estimand maximizes the penalized log-likelihood per unit weight

    ell_p(theta) = (1/W) sum_i w_i log f(x_i) - a * P(theta),   W = sum w_i,  a = 1/W,
    P = sum_k [tr(S Sigma_k^-1) + log det Sigma_k]  +  [S11/s2x + log s2x + S22/s2p + log s2p]

S the w-weighted covariance of the rows (as GMM2D: Chen and Tan 2009; the
Gaussian part is exactly GMM2D's penalty, `gmm._weighted_cov`). The
filament term is the same Chen–Tan form applied to its two variances with
the diagonal of S. NOTE: check how GMM2D normalizes ell_p (per unit
weight or total) in `gmm._penalized_ll` / spec/method_notes.md section 5
and match it EXACTLY, including the scaling of the penalty, so the two
estimators' conventions agree; if this file's formula disagrees with
GMM2D's convention, GMM2D's convention wins and the report says so.

## 2. Parameter layout (theta, data coordinates), p = 6*Kg + 6 = 30

| block | entries | names (outputs) |
|---|---|---|
| Gaussian weights | Kg | pi1..pi4 (pi_f is implied, NOT stored) |
| Gaussian means | 2*Kg | mu1x, mu1y, ..., mu4x, mu4y |
| Gaussian covs | 3*Kg | S1_11, S1_12, S1_22, ..., S4_11, S4_12, S4_22 |
| filament | 6 | fil_m, fil_s2x, fil_b0, fil_b1, fil_b2, fil_s2p |

Feasible iff all pi_k > 0, pi_f > 0, every Sigma_k positive definite,
s2x > 0, s2p > 0.

## 3. Module `src/qij_joint/shearmix_model.py` (AGENT B owns)

Pure functions, numpy only, no fitting loops. All take/return theta in
section 2's layout. `Kg` is always an explicit argument.

```python
OMEGA = np.pi / 2.64

Prep = namedtuple('Prep', ['X', 'Bx'])          # X (N,2) float; Bx (N,3) = b(x) at raw x
Pen  = namedtuple('Pen',  ['W', 'a', 'Scov'])   # sum w, 1/W (or GMM2D's convention), weighted cov (2,2)

def n_params(Kg) -> int
def make_outputs(Kg) -> tuple                    # section 2 names, in order
def pack(Kg, pis_g, mus, covs, fil) -> ndarray  # pis_g (Kg,), mus (Kg,2), covs (Kg,2,2),
                                                 # fil = (m, s2x, b0, b1, b2, s2p)
def unpack(Kg, theta) -> (pis_g, pi_f, mus, covs, fil)   # fil as ndarray(6,)
def feasible(Kg, theta) -> bool
def prepare(X, omega=OMEGA) -> Prep
def penalty_setup(prep, w) -> Pen
def log_joint(prep, Kg, theta) -> ndarray (N, Kg+1)   # log(pi_k f_k(x_i)); column Kg = filament
def e_step(prep, Kg, theta) -> (R (N,Kg+1), logf (N,))  # responsibilities, log f(x_i)
def m_step(prep, w, R, pen, Kg) -> theta         # closed-form PENALIZED M-step: Gaussians as GMM2D;
                                                 # filament: m, s2x weighted mean/var of x (+ penalty);
                                                 # b by weighted least squares of y on Bx; s2p from
                                                 # weighted residuals (+ penalty)
def penalized_ll(prep, w, Kg, theta, pen) -> float    # NaN if infeasible
def grad(prep, w, Kg, theta, pen) -> ndarray (p,)     # analytic d ell_p / d theta
def score_rows(prep, w, Kg, theta, pen) -> ndarray (N,p)
    # the per-row psi whose solve against A gives the influence, EXACTLY
    # GMM2D's definition (gmm._score_info / _score_info_penalized /
    # _penalty_influence_extra, method_notes section 5): including the
    # penalty's own data dependence through S. Contract: the w-weighted
    # mean of the rows equals grad(...) at any theta.
def info(prep, w, Kg, theta, pen, rel_step=1e-5) -> ndarray (p,p)
    # A = -(d grad / d theta), central differences of the ANALYTIC grad,
    # step h_j = rel_step * max(1, |theta_j|), symmetrized ((A+A.T)/2).
    # Same normalization as GMM2D's A (so IF = solve(A, psi.T).T).
# unconstrained parametrization for the Newton finish (mirror gmm_param.py):
def to_u(Kg, theta) -> u                          # weights: log-ratio to pi_f; Gaussian covs: as
def from_u(Kg, u) -> theta                        #   gmm_param (Cholesky/log); s2x, s2p: log;
def jac_u(Kg, u) -> ndarray (p,p)                 #   m, b: identity.  d theta / d u
def coordinate_scales(Kg, Scov) -> ndarray (p,)   # as gmm_param.coordinate_scales, extended
```

Gradient checks Agent B must run itself (in its report): grad vs central
FD of penalized_ll at a random feasible theta (rel err < 1e-6);
weighted mean of score_rows == grad; m_step increases penalized_ll from
any feasible theta (EM monotonicity, 50 iterations, 3 random starts);
from_u(to_u(theta)) == theta; jac_u vs FD of from_u.

## 4. Module `src/qij_joint/shearmix.py` (AGENT C owns)

The estimator, mirroring `gmm.GMM2D`'s public contract EXACTLY (read
gmm.py's module docstring and class; spec/QIJ_estimator_fit_spec.md
sections 2.4 and 7; spec/QIJ_estimator_search_spec.md):

```python
class ShearMix2D:
    name = 'shearmix2d'
    takes_start = True
    def __init__(self, Kg=4, omega=OMEGA, n_starts=20, seed=0, reference=None,
                 eta=gmm._ETA_DEFAULT, cond_max=gmm._COND_MAX_DEFAULT)
    # attributes: Kg, omega, p, outputs, eta, cond_max, reference, last_fit_info (gmm.FitInfo or None)
    def prepare(self, X) -> Prep
    def loglik(self, X, w, theta, prep=None) -> float       # penalized, per unit weight, no fit
    def __call__(self, X, w, prep=None, start=None, eta=None, **kw) -> ndarray (p,)
    def influence(self, X, w, prep=None, start=None, eta=None, **kw) -> ndarray (N,p)
    def fit_and_influence(self, X, w, prep=None, start=None, eta=None, **kw) -> (theta, IF)
```

Same semantics as GMM2D: never raises (NaN on failure); statuses and
`last_fit_info` as GMM2D (`gmm.FitInfo`, `search` dict may hold the
search's own diagnostics); `start` = continuation (skip search, EM from
start, same Newton finish), labels against start's own Gaussians;
otherwise against `reference` (means (Kg,2), covs (Kg,2,2)) by min total
Bhattacharyya (`gmm._bhattacharyya`, `linear_sum_assignment`), else the
canonical sort (`gmm._canonical_sort` logic on the Gaussians). The
filament is NEVER relabelled (always the last block). Influence =
`solve(A, psi.T).T` with A = `info(...)`, psi = `score_rows(...)`,
refused (NaN) when cond(A) > cond_max. Convergence test and trust-region
finish: mirror `gmm._fit` (scaled gradient via coordinate_scales,
`gmm._trust_region_step` reused directly; the Hessian in u is obtained
by central FD of the u-gradient `jac_u(u).T @ grad(theta(u))`).

Search (cold fit, no start): this is the risky part; keep it simple,
deterministic in `seed`, and documented:
1. Filament seed by trimmed least squares: regress y on Bx (weights w);
   5 rounds, each refitting on the 40% of rows with the smallest |residual|.
   m, s2x from those rows' x; s2p from their residuals; pi_f = 0.35.
2. Gaussian seeds for the remaining structure: mirror `seeding.py`'s
   residual-driven greedy insertion (Verbeek et al.) but with the
   filament seed's density included in the current mixture density, so
   insertion targets the excess the filament does not explain (the cores
   are tiny; k-means alone will not find them). Also k-means starts on
   the rows outside the filament's 2-sigma band, `n_starts` of them.
3. Screen all starts with short EM runs (mirror GMM2D's racing screen or
   a fixed short budget; say which), keep the best 3 by penalized ll, run
   each to EM convergence + Newton finish, return the best.
Report wall time per cold fit on one N = 10,000 draw.

## 5. Dataset, subject wrappers, registry, truth (AGENT A owns)

Files: `src/qij_joint/data/make_cloudfil_G_U_P3_v1.py` (writes the .npz
and .txt beside it), `src/qij_joint/data/cloudfil_G_U_P3_v1.npz`,
`.txt`, a new function in `datasets.py`, new module
`src/qij_joint/cloudfil_u.py`, two entries in `registry.py`, new
`src/qij_joint/data/truth/cloudfil_u.json`.

- Components, in this file order: cloud, P1 embedded, P2 on-filament, P3
  off-filament (all four copied EXACTLY from `cloudfil_G_B6_P3_v1.npz` by
  role: weight, mean, covariance), then the filament: weight 0.38,
  m = 0, s2x = 1.25**2, b = (0, 0.55, 0), s2p = 0.08**2, omega = pi/2.64.
  npz keys: weights (5,), means (4,2), covs (4,2,2), role (5,),
  fil (6,) = (m, s2x, b0, b1, b2, s2p), omega, name, N_design, subject.
  Weights sum to 1.
- `datasets.cloudfil_G_U_P3_v1(N, seed)`: counts =
  default_rng(seed).multinomial(N, weights) then draws per component in
  file order from the same rng, mirroring `datasets.cloudfil_G_B6_P3_v1`
  exactly (read it); the filament: x ~ N(m, s2x), y = h(x) + sqrt(s2p)*eps.
- `cloudfil_u.py`: `_SubjectShearMixture` mirroring `cloudfil._SubjectMixture`
  (same constructor args, outputs = ShearMix2D outputs + 7 derived, same
  `measured`, `periodic`, `loglik`, `last_fit_info`, `prepare`), wrapping
  `shearmix.ShearMix2D(Kg=4, reference=reference, ...)`. Derived outputs:
  identical closed forms to `cloudfil._derived_outputs` for the subject
  Gaussian s, EXCEPT the peak contrast's denominator sums the other
  Gaussians AND the filament's density at mu_s, all on one consistent
  normalization (fully normalized densities on both sides is simplest:
  ln C = ln(pi_s N(mu_s|mu_s,Sigma_s)) - ln(sum_{k != s} f_k(mu_s))).
  Subject selection among the 4 Gaussians by Bhattacharyya to the truth
  (as `cloudfil._select_subject`). Jacobian of the 7 derived outputs by
  central FD as `cloudfil._jacobian`. Classes `P2ShearMixture` (name
  'p2shearmix', role 'P2 on-filament', prefix 'p2', pa_lo 0) and
  `P1ShearMixture` (name 'p1shearmix', role 'P1 embedded', prefix 'p1',
  pa_lo -pi/2).
- `registry.py`: `('cloudfil_u', 'p2')` and `('cloudfil_u', 'p1')` cases.
- `data/truth/cloudfil_u.json`: `{"p2": {"outputs": [...], "values":
  [...]}, "p1": {...}}` exactly like `truth/cloudfil.json`: the packed
  true theta (Gaussians in FILE order: cloud, P1, P2, P3; filament last)
  + the 7 derived outputs at the truth. Compute it with the wrapper's own
  derived-output function, not by hand.

Agent A can write `cloudfil_u.py` against section 4's interface before
`shearmix.py` exists; it imports `shearmix_model` for unpack/densities.

## 6. Rules for all agents

- python3.9; run from the worktree root with `PYTHONPATH=src`.
- Touch ONLY your own files (section 3/4/5). Do not edit gmm.py,
  cloudfil.py, seeding.py, pipeline, core/, spec/ (except your report),
  other agents' files.
- No commits; the coordinator commits after review.
- Every check is ONE draw (seed 1, N = 10,000) at most; hard time cap 10
  minutes on any run; never run oracle/bootstrap/ij/qij pipelines.
- Report: files, signatures as built, every deviation from this file,
  and the checks' numbers.
