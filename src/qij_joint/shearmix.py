"""`ShearMix2D`: the estimator built on `shearmix_model.py`'s pure math
(spec/QIJ_shearmix_interface.md section 4). `Kg` free full-covariance
Gaussians (cloud and cores) plus ONE sheared-Gaussian filament component,
fit by the same multi-start-EM / trust-region-Newton design as
`gmm.GMM2D`, with the filament seeded and never relabelled.

`ShearMix2D(Kg=4, omega=shearmix_model.OMEGA, n_starts=20, seed=0,
reference=None, eta=gmm._ETA_DEFAULT, cond_max=gmm._COND_MAX_DEFAULT)`
follows `GMM2D`'s own public contract exactly: `T(X, w) -> ndarray(p,)`
never raises (NaN on any status but 'converged'); `last_fit_info`
(`gmm.FitInfo`) is the one piece of state between calls, overwritten by
every `__call__`/`influence`/`fit_and_influence`; `takes_start = True`
makes `start` a continuation (skip the cold search, EM from `start` at
the given weights, then the same trust-region finish), labelled against
`start`'s OWN Gaussians by minimum-Bhattacharyya assignment (never
`reference`, never the canonical sort); without `start`, labelling is
against `reference` (means (Kg,2), covs (Kg,2,2)) when given, else the
canonical sort (ascending (mu_x, mu_y)) -- in both cases ONLY the Kg
Gaussian blocks are permuted; the filament block is never touched, never
relabelled, always last. `influence = solve(A, psi.T).T`, `A =
shearmix_model.info(...)`, `psi = shearmix_model.score_rows(...)`,
refused (NaN) when `cond(A) > cond_max`. No centering anywhere (module
docstring of `shearmix_model.py`): `prepare` is `shearmix_model.prepare`
verbatim, unlike `GMM2D.prepare`'s unweighted shift.

Reused from `gmm.py` by import, unchanged: `_trust_region_step` (the
exact Moré-Sorensen trust-region subproblem solve, model- and
layout-agnostic -- it only ever sees a gradient, a Hessian and a
radius), `_bhattacharyya` (also layout-agnostic: it takes (means, Ss)
pairs, so this module converts its own (Kg,2,2) covariance blocks to
the (Kg,3) `Ss` triple it expects), `FitInfo`, `_ETA_DEFAULT`,
`_COND_MAX_DEFAULT`. Also reused: `gmm_param.scaled_gradient_norm`, the
one-line Dennis-and-Schnabel relative-gradient formula
(`max_i |g_i| s_i / max(|ell|, 1)`) that both `GMM2D` and this module
apply to their own (differently-shaped) gradient/scale vectors --
nothing about it is Gaussian-mixture-specific. And `seeding._cell_stats`,
the over-segmentation cell geometry/statistics helper (count, weight
sum, area, weighted mean, weighted covariance per k-means cell) used
unchanged by the greedy insertion below, since nothing about computing
a cell's own statistics depends on what density explains the rest of
the data.

EVERYTHING ELSE is new (written here, not imported), because the
filament changes it: the EM step (`shearmix_model.e_step`/`m_step`
operate on one flat `theta`, not `GMM2D`'s own (pis, mus, Ss) triple,
and the filament's own M-step is a weighted least-squares fit, not a
Gaussian moment update); the SQUAREM/Aitken-accelerated EM loop (mirrors
`gmm._em_accelerated`/`_squarem_round`, restated over `theta` vectors
instead of (pis, mus, Ss) tuples, since `shearmix_model.feasible`/
`to_u`/`from_u` all take the single packed layout); the trust-region
Newton finish (mirrors `gmm._fit`'s section-7 loop exactly -- same
acceptance rule, same radius update, same convergence/stall tests --
but forms `H_u` by CENTRAL FINITE DIFFERENCE of the u-gradient
`jac_u(u).T @ grad(theta(u))`, per the build interface, rather than
`gmm_param.chain`'s closed-form curvature correction, which this
module's parametrization does not supply); and the cold search (below),
which has no `GMM2D` analogue at all.

THE COLD SEARCH, kept deliberately simple and deterministic in `seed`
alone (never in `w`):

1. FILAMENT SEED, trimmed least squares. Fit `y` on `Bx` (weighted
   least squares at the caller's own `w`) on the current row subset,
   starting from ALL rows; after each fit, keep the 40% of rows (by
   count, `round(0.4*N)`) with the smallest |residual| AGAINST THAT
   FIT, recomputed over all N rows each round (not a shrinking subset
   of the subset) -- 5 rounds total, so round 1 fits on everything and
   rounds 2-5 each refit on a fresh 40% selected by the previous round's
   own fit. `m`, `s2x` are the final 40%'s own weighted mean/variance of
   x; `s2p` is their weighted residual variance against the final `b`;
   `pi_f` is fixed at 0.35 for the whole cold search (never re-estimated
   until the joint EM that follows takes over all parameters, filament
   included).
2. GAUSSIAN SEEDS. Two independent constructions, both deterministic in
   `seed`:
   (a) Greedy LIKELIHOOD-GAIN insertion (`_greedy_insert`; Verbeek,
       Vlassis and Kroese 2003's own greedy-EM criterion, as
       `seeding.greedy_em_start`'s round-6 design, but reimplemented
       here rather than called -- `seeding.py`'s own mixture density
       and EM calling convention are pure-Gaussian and take no
       filament): one Gaussian (the whole draw's weighted mean and
       covariance, at weight 1 - pi_f) grown to Kg by inserting, one at
       a time, a new component at the `_GREEDY_CELL_FACTOR*Kg`-cell
       k-means over-segmentation's cell with the largest likelihood
       GAIN (`_insertion_gains`): tentatively add that cell's own
       weighted mean/covariance as a new Gaussian at a small mixing
       weight (the cell's own weight share), rescale the rest of the
       CURRENT mixture -- Gaussians AND the fixed filament seed -- by
       (1 - that weight) with everything else held fixed, and measure
       the resulting penalized-ll gain over the current mixture's own
       ll; the cell with the largest gain wins. 20 accelerated-EM steps
       follow every insertion (`_INSERT_EM_STEPS`, `seeding.py`'s own
       constant) so the model has settled before the next cell is
       judged. The over-segmentation itself is built in the filament
       seed's own UNSHEARED coordinates z = (x, y - h_hat(x))
       (`_unsheared_coords`), not the raw data: the shear has Jacobian
       1 and is exactly invertible (it only ever shifts y by a function
       of x, so no cell's row count changes), but in z the filament is
       a straight band at z2 ~ 0 instead of a curved ridge, so a round
       k-means cell there tracks the band's own width instead of
       straddling its curve -- candidate Gaussians are still
       initialised from each chosen cell's own rows in ORIGINAL (x)
       coordinates (`_cell_stats` called on `X`; only cell MEMBERSHIP
       comes from the z-space k-means). This is the one piece of the
       search that sees the filament at all while choosing WHERE to put
       a Gaussian, which is the whole point (a Gaussian-only criterion
       would re-target the filament's own structure everywhere along
       its length instead of the cores) -- see the root-cause audit
       below for why an EARLIER version of this rule (a density times a
       circular cell-area estimate of the expected count, instead of a
       likelihood gain) failed on exactly this point, and why the
       unsheared coordinates are needed alongside it.
   (b) `n_starts` k-means starts (`scipy.cluster.vq.kmeans2`,
       `minit='++'`, same `_KMEANS_ITER` as `gmm.py`), but clustering
       only the rows OUTSIDE the filament seed's own 2-sigma band
       (|x - m| > 2*sqrt(s2x) OR |y - h(x)| > 2*sqrt(s2p)) into Kg
       clusters: the filament band is overwhelmingly filament rows, so
       clustering it along with everything else would waste cluster
       budget on filament structure the Gaussians are not meant to
       explain. Each start's weights are its cluster fractions of the
       off-band rows scaled by (1 - pi_f); covariances are each
       cluster's own population covariance (falling back to the
       off-band data's own covariance / Kg on a cluster with under 3
       members, as `gmm._starts`).
3. SCREEN: a FIXED short EM budget (25 plain-SQUAREM-accelerated-EM
   steps, `_SCREEN_BUDGET` -- not the racing screen of
   spec/QIJ_estimator_search_spec.md 2.2, which assumes PLAIN
   unaccelerated EM's own linear convergence for its Aitken projection
   and would need its own re-derivation for a filament whose M-step is
   a regression, not a moment update; a fixed budget needs none of
   that, and the interface permits either) on every candidate from step
   2, ranked by penalized ll; the best 3 (`_K_KEEP`, `GMM2D`'s own
   constant) are each run to full EM convergence and the trust-region
   finish; the one with the highest FINAL (post-finish) penalized ll
   among those that reach a non-'infeasible' status is returned --
   ranking after the finish, not before it, since the finish can reorder
   two close EM endpoints (the lesson of
   spec/QIJ_estimator_fit_spec.md section 6's split-and-merge
   discussion, section "6": a winner picked before the finish is not
   always the winner after it).
No split-and-merge stage is implemented: the build interface's search
(section 4) does not ask for one, unlike `GMM2D`'s cold search (which
gained it from spec/QIJ_estimator_search_spec.md 2.3 after its own
multistart-alone search was shown to miss the beaded-filament basin on
every audited draw). The filament seed already supplies what
split-and-merge existed to repair for `GMM2D` -- a model-driven
explanation of the elongated structure the Gaussians-only multistart
could not find -- so there's a priori less for it to fix here FOR THE
FILAMENT ITSELF.

ROOT-CAUSE AUDIT (cloudfil_G_U_P3_v1, N=10000, seed=1): an EARLIER
version of item 2(a) scored each over-segmentation cell by
`wsum - W*area*f(centre)` -- a circular cell-area (`pi*r_median^2`)
times the current mixture's own density at the cell's centre, built
for the roughly round clusters a plain GMM produces. That version's
cold fit converged 0.0048/unit weight (~48 nats total) below the
continuation from truth, never finding the tiny off-filament core P3
(50 rows, weight 0.005, at (-1.2,-1.1)): tracing its insertion
decisions (EM budgets 20-400 steps/round, over-segmentations at every
cell_factor 10-200) showed the same wrong cell winning every time, and
evaluating the EXACT insertion score at the TRUE theta (not the fitted
one) showed why: cells whose centre sits on the filament's own ridge
show a spurious circular-area excess of 60-130, bigger than P3's own
true excess of 17, PURELY from the area/density mismatch against the
filament's narrow, strongly curved density -- a property of the
ridge's geometry, not a convergence, resolution, or seed-bias artifact,
present even at the exact true parameters and not fixable by tuning
that rule itself.

The decisive diagnostic that replaced it: at the true parameters, with
one core removed at a time (P2, P3, P1 in turn; weights renormalised),
score every cell of the SAME over-segmentation by four combinations --
{x-space, z-space (unsheared)} cell geometry x {circular-area excess
(old), likelihood gain (new)} scoring -- and rank the removed core's
own cell against the largest score anywhere else. Old circular-area
scoring ranked the missing cell 1st for P1 and P2 but 18th/188 for P3
(behind ridge cells scoring up to 130 against P3's own 64), in BOTH
cell geometries: the area/density mismatch is a property of the
SCORING rule, not of which space the cells are built in. Likelihood
gain ranked the missing cell 1st/188 (x-space) or 1st/192 (z-space) for
EVERY removed core, with a clear margin over the next-best cell in
every case -- gain alone (even on the unmodified, ridge-straddling
x-space cells) already fixes the P3 failure, because it measures the
fit's own improvement rather than approximating an expected count.
`_greedy_insert` combines gain scoring WITH the unsheared cell geometry
regardless (the module docstring above), since the two are
independent, complementary fixes -- gain removes the scoring bias
itself; the unsheared cells additionally let the over-segmentation's
cells track the ridge's curve rather than cut across it, which widens
the margin between the correct cell and its runner-up for every core
tested (e.g. P3's margin grew from 2.8x to 7.7x) and gives P2 (the
on-ridge core; the subject this dataset's talk estimates) a better-
conditioned initial covariance besides. On the audited draw, the fixed
`_greedy_insert` alone -- no further repair of any kind -- places all
three cores and the cloud within 0.01-0.02 of their true means on its
first pass, and the full cold search's final penalized ll matches the
continuation-from-truth's to ~1e-15 (far inside the `eta` tolerance),
WITH OR WITHOUT the off-band k-means starts (item 2(b)) in the
candidate pool. A previous agent's patch for this same failure (a
second, filament-excluding greedy insertion plus a split-and-merge
repair combining its result with item 2(a)'s) is consequently NOT
needed on top of the fixed scoring rule and unsheared cells, and has
been removed (simpler code wins): neither the second insertion nor the
merge repair improves on `_greedy_insert` alone once ITS OWN scoring
rule is the actual problem being fixed, and the one draw audited here
no longer has a gap for either of them to close.

Labelling happens AFTER the EM-converge-and-finish pipeline, as a final
permutation of the Kg Gaussian blocks, rather than before it as
`gmm._fit` does: relabelling is a bijective reindexing of parameter
blocks the optimizer treats independently (`to_u`/`from_u`/`jac_u` act
coordinate-by-coordinate within each Gaussian's own block), so applying
it before or after a fit that converges to the same point by the same
path gives identical numbers, only reordered -- doing it once at the
end, uniformly for the cold search and the continuation path, is
simpler than threading a label order through every intermediate EM/
finish call.
"""

from collections import namedtuple
import math
import warnings

import numpy as np
from scipy.cluster.vq import kmeans2
from scipy.optimize import linear_sum_assignment

from . import gmm
from . import shearmix_model as sm
from .gmm_param import scaled_gradient_norm
from .seeding import _cell_stats

# ----------------------------------------------------------------------
# Constants. None fitted to any one draw (feedback_constants.md):
# the trim fraction/rounds and pi_f seed are the search's own stated
# design (build interface section 4 item 1); the EM/Newton iteration
# caps and FD step mirror gmm.py's/shearmix_model.py's own; the screen
# budget and k_keep are explicit choices documented in the module
# docstring above, not measurements.
# ----------------------------------------------------------------------
_TRIM_ROUNDS = 5
_TRIM_FRACTION = 0.4
_PI_F_SEED = 0.35
_OFFBAND_SIGMA = 2.0

_GREEDY_CELL_FACTOR = 50
_MIN_CELL_COUNT = 5
_MIN_INSERT_SHARE = 5.0
_INSERT_EM_STEPS = 20
_KMEANS_ITER = 300

_SCREEN_BUDGET = 25
_K_KEEP = 3

_EM_ITER_CAP_FACTOR = 20
_NEWTON_ITER_CAP_FACTOR = 2
_HESS_REL_STEP = 1e-5


# ======================================================================
# Thin wrappers over shearmix_model: feasibility/NaN -> None sentinels,
# so the EM/search code below never has to special-case an exception
# versus a NaN return from the pure-math module.
# ======================================================================

def _ll_at(prep, w, Kg, theta, pen):
    try:
        return sm.penalized_ll(prep, w, Kg, theta, pen)
    except Exception:
        return float('nan')


def _em_step(prep, w, pen, Kg, theta):
    """One penalized-EM update (E-step responsibilities, closed-form
    M-step) -> theta_new, or None if `theta` is infeasible, the E-step's
    own density is singular (`log_joint`/`e_step` raise
    `np.linalg.LinAlgError` on a non-PD covariance or non-positive
    filament variance -- `shearmix_model.py`'s module docstring), or the
    M-step's own output is not finite."""
    if not sm.feasible(Kg, theta):
        return None
    try:
        R, _ = sm.e_step(prep, Kg, theta)
    except np.linalg.LinAlgError:
        return None
    if not np.all(np.isfinite(R)):
        return None
    theta_new = sm.m_step(prep, w, R, pen, Kg)
    if not np.all(np.isfinite(theta_new)):
        return None
    return theta_new


# ======================================================================
# SQUAREM-accelerated EM to the joint Aitken/scaled-gradient stop test
# (mirrors gmm._squarem_round / _em_accelerated over theta vectors).
# ======================================================================

def _squarem_round(prep, w, pen, Kg, theta0, budget, m):
    """One SQUAREM round spending at most `budget` (>=1) EM steps.
    Returns (theta, ll, n_em, m) at the round's own output, or
    (None, nan, 0, m) if either of the two REQUIRED EM steps (theta0 ->
    theta1, theta1 -> theta2) degenerates -- there is no fallback for
    those, unlike the optional SQUAREM-extrapolated candidate, whose own
    failure or non-improvement simply falls back to theta2 (gmm.py's own
    rule)."""
    theta1 = _em_step(prep, w, pen, Kg, theta0)
    if theta1 is None:
        return None, float('nan'), 0, m
    if budget == 1:
        return theta1, _ll_at(prep, w, Kg, theta1, pen), 1, m

    theta2 = _em_step(prep, w, pen, Kg, theta1)
    if theta2 is None:
        return None, float('nan'), 0, m
    ll2 = _ll_at(prep, w, Kg, theta2, pen)
    if budget == 2:
        return theta2, ll2, 2, m

    r = theta1 - theta0
    v = (theta2 - theta1) - r
    vnorm = np.linalg.norm(v)
    alpha = min(-np.linalg.norm(r) / vnorm, -1.0) if vnorm > 0.0 else -1.0
    if alpha < -m:
        alpha = -m
        m *= 4.0
    theta_sq = theta0 - 2.0 * alpha * r + alpha ** 2 * v

    if sm.feasible(Kg, theta_sq):
        theta3 = _em_step(prep, w, pen, Kg, theta_sq)
        if theta3 is not None:
            ll3 = _ll_at(prep, w, Kg, theta3, pen)
            if np.isfinite(ll3) and ll3 >= ll2:
                return theta3, ll3, 3, m

    return theta2, ll2, 2, m


def _em_accelerated(prep, w, pen, Kg, theta, ll, eta, budget=None):
    """SQUAREM rounds from `theta` with its own `ll`, until BOTH the
    scaled-gradient test and the Aitken-projected remaining-gain test
    pass, or the `20*p` iteration cap (`budget`, when given, replaces
    it -- the greedy seeding's and the screen's own fixed step budgets).
    Mirrors gmm._em_accelerated exactly except the gradient is formed as
    `jac_u(u).T @ grad(theta)` (shearmix_model's own chain rule) rather
    than gmm_param.chain's closed-form-plus-curvature Hessian machinery
    (only the gradient is needed here; the Hessian is the finish's job).
    Returns (theta, ll, n_used, status, score_scaled, last_step_u), or
    None if a required EM step degenerated anywhere along the
    trajectory (`_squarem_round`'s own sentinel) or the gradient itself
    could not be formed (a non-PD covariance at an otherwise-feasible
    theta -- `_em_step`'s same defensive case)."""
    p = sm.n_params(Kg)
    iter_cap = _EM_ITER_CAP_FACTOR * p if budget is None else int(budget)
    sqrt_eta = math.sqrt(eta)
    eps = np.finfo(float).eps
    s = sm.coordinate_scales(Kg, pen.Scov)

    n_used = 0
    m = 4.0
    ll_hist = [ll]
    u_prev = sm.to_u(Kg, theta)
    last_step_u = np.zeros(p)
    score_scaled = None
    status = 'em_cap'
    theta_cur = theta

    while n_used < iter_cap:
        budget_round = iter_cap - n_used
        theta_new, ll_new, n_em, m = _squarem_round(prep, w, pen, Kg, theta_cur, budget_round, m)
        if theta_new is None or not np.isfinite(ll_new):
            return None
        theta_cur = theta_new
        n_used += n_em
        ll_hist.append(ll_new)
        if len(ll_hist) > 3:
            ll_hist.pop(0)

        u_new = sm.to_u(Kg, theta_cur)
        last_step_u = u_new - u_prev
        u_prev = u_new
        ll = ll_new

        if len(ll_hist) == 3:
            ell_km2, ell_km1, ell_k = ll_hist
            denom = ell_km1 - ell_km2
            aitken_ok = False
            tol_ell = eta * max(abs(ell_k), 1.0)
            if abs(denom) > eps * max(abs(ell_km1), abs(ell_km2), 1.0):
                c_k = (ell_k - ell_km1) / denom
                one_minus_c = 1.0 - c_k
                if abs(one_minus_c) > eps * max(abs(c_k), 1.0):
                    ell_inf = ell_km1 + (ell_k - ell_km1) / one_minus_c
                    aitken_ok = abs(ell_inf - ell_k) <= tol_ell
            else:
                aitken_ok = abs(ell_k - ell_km1) <= tol_ell
            if aitken_ok:
                try:
                    g_theta = sm.grad(prep, w, Kg, theta_cur, pen)
                except np.linalg.LinAlgError:
                    return None
                g_u = sm.jac_u(Kg, u_new).T @ g_theta
                score_scaled = scaled_gradient_norm(g_u, s, ell_k)
                if score_scaled <= sqrt_eta:
                    status = 'converged'
                    break

    if score_scaled is None:
        try:
            g_theta = sm.grad(prep, w, Kg, theta_cur, pen)
        except np.linalg.LinAlgError:
            return None
        g_u = sm.jac_u(Kg, u_prev).T @ g_theta
        score_scaled = scaled_gradient_norm(g_u, s, ll)

    return theta_cur, ll, n_used, status, score_scaled, last_step_u


def _run_em(prep, w, pen, Kg, theta0, eta):
    """Weighted EM for a single start to the joint stop test, or None
    the moment the start degenerates (mirrors gmm._run_em). Returns
    dict(theta, ll, converged, n_iter, status, score_scaled,
    last_step_u)."""
    if not sm.feasible(Kg, theta0):
        return None
    ll0 = _ll_at(prep, w, Kg, theta0, pen)
    if not np.isfinite(ll0):
        return None
    result = _em_accelerated(prep, w, pen, Kg, theta0, ll0, eta)
    if result is None:
        return None
    theta, ll, n_used, status, score_scaled, last_step_u = result
    return dict(theta=theta, ll=ll, converged=(status == 'converged'), n_iter=n_used,
                status=status, score_scaled=score_scaled, last_step_u=last_step_u)


# ======================================================================
# Trust-region Newton finish (mirrors gmm._fit's section-7 loop exactly;
# H_u by central FD of the u-gradient, per the build interface).
# ======================================================================

def _g_u_at(prep, w, pen, Kg, u):
    theta = sm.from_u(Kg, u)
    g_theta = sm.grad(prep, w, Kg, theta, pen)
    J = sm.jac_u(Kg, u)
    return J.T @ g_theta


def _hess_u_fd(prep, w, pen, Kg, u, rel_step=_HESS_REL_STEP):
    """H_u (p,p) = d(jac_u(u).T @ grad(theta(u))) / du, central
    differences, step h_j = rel_step * max(1, |u_j|) (shearmix_model.info's
    own step rule), symmetrized."""
    p = len(u)
    H = np.empty((p, p))
    for j in range(p):
        h = rel_step * max(1.0, abs(u[j]))
        up = u.copy()
        up[j] += h
        um = u.copy()
        um[j] -= h
        gp = _g_u_at(prep, w, pen, Kg, up)
        gm = _g_u_at(prep, w, pen, Kg, um)
        H[:, j] = (gp - gm) / (2.0 * h)
    return 0.5 * (H + H.T)


def _finish(prep, w, pen, Kg, theta_em, last_step_u, eta):
    """The trust-region Newton finish, one loop for every fit (cold
    search survivors and continuations alike): mirrors gmm._fit's
    section-7 algorithm -- exact Moré-Sorensen subproblem
    (`gmm._trust_region_step`, reused unchanged), ACCEPT on rho >= 1/4
    or (interior Newton step AND the scaled gradient falls), radius
    update 2x/4x/unchanged, CONVERGED at an accepted point with scaled
    gradient <= sqrt(eta)*g0, STALL when the radius falls below eta.
    Returns (dict(theta, ll, resid, g0, n_iter_newton), status) with
    status in {'converged', 'newton_stalled', 'newton_cap'}; raises
    np.linalg.LinAlgError if the finish's own score/gradient cannot be
    formed at some point along the way (propagated to the caller, which
    treats it as a 'linalg' fit)."""
    p = sm.n_params(Kg)
    s = sm.coordinate_scales(Kg, pen.Scov)
    cache: dict = {}

    def _eval(v):
        key = v.tobytes()
        if cache.get('key') != key:
            u = v * s
            theta = sm.from_u(Kg, u)
            ll = _ll_at(prep, w, Kg, theta, pen)
            g_theta = sm.grad(prep, w, Kg, theta, pen)
            J = sm.jac_u(Kg, u)
            g_u = J.T @ g_theta
            H_u = _hess_u_fd(prep, w, pen, Kg, u)
            cache['key'] = key
            cache['val'] = dict(theta=theta, ll=ll, g_u=g_u, H_u=H_u)
        return cache['val']

    def fun(v):
        return -_eval(v)['ll']

    def jac(v):
        return -_eval(v)['g_u'] * s

    def hess(v):
        H_uv = _eval(v)['H_u']
        return -(H_uv * s[:, None]) * s[None, :]

    def _score_v(v):
        e = _eval(v)
        return scaled_gradient_norm(e['g_u'], s, e['ll'])

    u0 = sm.to_u(Kg, theta_em)
    v0 = u0 / s
    radius0 = max(float(np.linalg.norm(last_step_u / s)), math.sqrt(eta))
    n_cap = _NEWTON_ITER_CAP_FACTOR * p
    radius_max = max(1e3, 10.0 * radius0)

    g0 = _score_v(v0)
    g_conv = math.sqrt(eta) * g0

    def _converged(v, score, n_iter_newton):
        e = _eval(v)
        return dict(theta=e['theta'], ll=e['ll'], resid=score, g0=g0,
                    n_iter_newton=n_iter_newton), 'converged'

    v = v0
    fun_v = fun(v)
    score_v = g0
    g_v, H_v = jac(v), hess(v)
    radius = radius0
    n_iter_newton = 0

    while n_iter_newton < n_cap:
        n_iter_newton += 1
        step, m, interior, _ = gmm._trust_region_step(g_v, H_v, radius)
        v_trial = v + step
        fun_trial = fun(v_trial)
        score_trial = _score_v(v_trial)
        rho = (fun_v - fun_trial) / m if m > 0.0 else -float('inf')

        if rho >= 0.25:
            accept = True
            if rho > 0.75 and not interior:
                radius = min(radius * 2.0, radius_max)
        elif interior and score_trial < score_v:
            accept = True
        else:
            accept = False
            radius = float(np.linalg.norm(step)) / 4.0

        if accept:
            v, fun_v, score_v = v_trial, fun_trial, score_trial
            if score_v <= g_conv:
                return _converged(v, score_v, n_iter_newton)
            g_v, H_v = jac(v), hess(v)

        if radius < eta:
            if score_v <= eta:
                return _converged(v, score_v, n_iter_newton)
            e = _eval(v)
            return dict(theta=e['theta'], ll=e['ll'], resid=score_v, g0=g0,
                        n_iter_newton=n_iter_newton), 'newton_stalled'

    e = _eval(v)
    return dict(theta=e['theta'], ll=e['ll'], resid=score_v, g0=g0,
                n_iter_newton=n_iter_newton), 'newton_cap'


def _converge_and_finish(prep, w, pen, Kg, theta0, eta):
    """EM to the joint stop test, then (only if EM itself converged) the
    trust-region finish -- mirrors gmm._fit's "the finish runs only when
    EM's own status is 'converged'" rule. Returns (dict(theta, ll,
    resid, g0, n_iter_em, n_iter_newton), status) or (None, status) on
    'infeasible'/'linalg'."""
    run = _run_em(prep, w, pen, Kg, theta0, eta)
    if run is None:
        return None, 'infeasible'
    if run['status'] != 'converged':
        return dict(theta=run['theta'], ll=run['ll'], resid=run['score_scaled'],
                    g0=run['score_scaled'], n_iter_em=run['n_iter'],
                    n_iter_newton=0), 'em_cap'
    try:
        fit, status = _finish(prep, w, pen, Kg, run['theta'], run['last_step_u'], eta)
    except np.linalg.LinAlgError:
        return None, 'linalg'
    fit['n_iter_em'] = run['n_iter']
    return fit, status


# ======================================================================
# Cold search: filament seed, Gaussian seeds, fixed-budget screen,
# best 3 to convergence + finish (build interface section 4; see the
# module docstring for the full design and what was simplified).
# ======================================================================

def _filament_seed(prep, w):
    """Trimmed weighted least squares of y on Bx: 5 rounds, each
    refitting on the 40% of ALL rows with the smallest |residual|
    against the PREVIOUS round's own fit (round 1 fits on every row, as
    there is nothing to trim yet). Returns fil = (m, s2x, b0, b1, b2,
    s2p) from the final round's own 40% subset: m, s2x its weighted
    mean/variance of x; s2p its weighted residual variance against the
    final b. pi_f itself is not part of this seed (fixed at
    `_PI_F_SEED` by the caller)."""
    X, Bx = prep.X, prep.Bx
    y = X[:, 1]
    N = X.shape[0]
    n_keep = max(4, int(round(_TRIM_FRACTION * N)))
    cur_idx = np.arange(N)
    b = None
    for _ in range(_TRIM_ROUNDS):
        Bsub, ysub, wsub = Bx[cur_idx], y[cur_idx], w[cur_idx]
        Wmat = Bsub * wsub[:, None]
        AtA = Bsub.T @ Wmat
        rhs = Wmat.T @ ysub
        b = np.linalg.solve(AtA, rhs)
        resid_all = y - Bx @ b
        order = np.argsort(np.abs(resid_all))
        cur_idx = order[:n_keep]
    Xf, wf = X[cur_idx], w[cur_idx]
    resid_f = y[cur_idx] - Bx[cur_idx] @ b
    Wf = float(wf.sum())
    m = float((wf @ Xf[:, 0]) / Wf)
    s2x = float((wf @ (Xf[:, 0] - m) ** 2) / Wf)
    s2p = float((wf @ resid_f ** 2) / Wf)
    return np.array([m, s2x, b[0], b[1], b[2], s2p])


def _trial_insertion_theta(Kg_cur, theta_cur, new_w, new_mu, new_cov):
    """`theta_cur` (Kg_cur Gaussians + filament) with one further
    Gaussian appended at (`new_mu`, `new_cov`, weight `new_w`), the
    existing Kg_cur Gaussians rescaled by (1 - new_w) with their own
    means/covariances held FIXED, the filament block untouched (its
    share of 1 - sum(pis_g) is implied, not stored, so rescaling the
    Gaussian weights alone also rescales pi_f correctly). Used by
    `_insertion_gains` to score a candidate cell and, after the best
    candidate is chosen, by `_greedy_insert` itself to build the
    accepted insertion -- the SAME construction either way."""
    pis_g, _, mus, covs, fil = sm.unpack(Kg_cur, theta_cur)
    pis_g_new = np.concatenate([pis_g * (1.0 - new_w), [new_w]])
    mus_new = np.concatenate([mus, new_mu[None, :]], axis=0)
    covs_new = np.concatenate([covs, new_cov[None, :, :]], axis=0)
    return sm.pack(Kg_cur + 1, pis_g_new, mus_new, covs_new, fil)


def _insertion_gains(prep, w, pen, Kg_cur, theta_cur, ll_cur, eligible, wsum, mean, cov3, W):
    """Likelihood-GAIN score for every eligible over-segmentation cell
    (Verbeek, Vlassis and Kroese 2003's own greedy-EM criterion, in
    place of a density/area approximation): for cell `j`, tentatively
    append a new Gaussian initialised from that cell's own weighted mean
    and covariance (`mean[j]`, `cov3[j]`, `_cell_stats`) at a small
    mixing weight (the cell's own share of the total weight, `wsum[j] /
    W`, floor-protected by `_MIN_INSERT_SHARE / W` -- the SAME weight
    `_greedy_insert` uses for the insertion it actually accepts), with
    every existing component (and the filament) rescaled by (1 - new
    weight) but otherwise held fixed (`_trial_insertion_theta`), and
    measure the resulting penalized-ll gain over `ll_cur` (the CURRENT
    mixture's own ll, at `Kg_cur` components). No density evaluated at a
    cell centre, no cell-area term anywhere: a cell's score is exactly
    how much the fit improves if a Gaussian is planted there, which is
    well-defined for a cell of any shape against a mixture of any shape
    (in particular, a cell straddling the filament's narrow, curved
    ridge needs no special handling, unlike the circular-area excess
    this replaced -- the cold-fit audit, module docstring). Returns
    gain (M_cells,), -inf at every ineligible cell or one whose trial
    theta is itself infeasible/non-finite."""
    gains = np.full(eligible.shape[0], -np.inf)
    for j in np.nonzero(eligible)[0]:
        new_w = max(wsum[j] / W, _MIN_INSERT_SHARE / W)
        cov_j = np.array([[cov3[j, 0], cov3[j, 1]], [cov3[j, 1], cov3[j, 2]]])
        theta_trial = _trial_insertion_theta(Kg_cur, theta_cur, new_w, mean[j], cov_j)
        ll_trial = _ll_at(prep, w, Kg_cur + 1, theta_trial, pen)
        if np.isfinite(ll_trial):
            gains[j] = ll_trial - ll_cur
    return gains


def _unsheared_coords(X, Bx, fil):
    """z = (x, y - h_hat(x)), h_hat the trimmed-LS filament seed `fil`
    (`_filament_seed`'s own b0, b1, b2 against the model's fixed-omega
    basis `Bx`). The shear has Jacobian 1 and is exactly invertible (x
    is untouched; y is shifted by a function of x alone), so it
    preserves every cell's row count -- only the over-segmentation's OWN
    geometry changes: in z the filament is a straight band at z2 ~ 0 of
    width ~sigma_perp, so a round k-means cell there tracks the ridge
    instead of straddling its curve (`_greedy_insert`'s own docstring)."""
    m, s2x, b0, b1, b2, s2p = fil
    h_hat = Bx @ np.array([b0, b1, b2])
    return np.column_stack([X[:, 0], X[:, 1] - h_hat])


def _greedy_insert(prep, w, Kg, fil, pi_f, eta, seed, cell_factor=_GREEDY_CELL_FACTOR):
    """Residual-driven greedy insertion (module docstring item 2(a)):
    one Gaussian (the weighted mean/covariance of the whole draw, at
    weight 1 - pi_f) grown to Kg by inserting, one at a time, a new
    component at the `cell_factor*Kg`-cell k-means over-segmentation
    cell with the largest likelihood-GAIN candidate (`_insertion_gains`;
    Verbeek, Vlassis and Kroese 2003), 20 accelerated-EM steps after
    each insertion (`_INSERT_EM_STEPS`). The over-segmentation is built
    in the filament seed's own UNSHEARED coordinates (`_unsheared_coords`)
    rather than the raw data, so its cells follow the ridge instead of
    straddling it; candidate Gaussians are still initialised from each
    chosen cell's own rows in ORIGINAL (x) coordinates (`_cell_stats`
    called on `X` itself -- only cell MEMBERSHIP comes from the z-space
    k-means). Returns theta (Kg components) or None if the seeding
    degenerates anywhere (too few rows, no eligible cell, a non-PD
    covariance along the way)."""
    X, Bx = prep.X, prep.Bx
    N = X.shape[0]
    M = min(cell_factor * Kg, max(Kg, N - 1))
    if M < 1:
        return None
    Z = _unsheared_coords(X, Bx, fil)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        centres_z, bmu = kmeans2(Z, M, iter=_KMEANS_ITER, minit='++',
                                  seed=np.random.default_rng(seed))
    count, wsum, _, mean, cov3 = _cell_stats(X, w, centres_z, bmu, M)
    eligible = count >= _MIN_CELL_COUNT
    if not eligible.any():
        return None

    pen = sm.penalty_setup(prep, w)
    W = pen.W
    xbar = (w @ X) / W
    d = X - xbar
    c0 = (d * w[:, None]).T @ d / W

    pis_g = np.array([1.0 - pi_f])
    mus = xbar[None, :].copy()
    covs = c0[None, :, :].copy()
    theta = sm.pack(1, pis_g, mus, covs, fil)
    ll0 = _ll_at(prep, w, 1, theta, pen)
    if not np.isfinite(ll0):
        return None
    result = _em_accelerated(prep, w, pen, 1, theta, ll0, eta, budget=_INSERT_EM_STEPS)
    if result is None:
        return None
    theta, ll_cur = result[0], result[1]

    for k_new in range(1, Kg):
        Kg_cur = k_new
        gains = _insertion_gains(prep, w, pen, Kg_cur, theta, ll_cur, eligible, wsum, mean, cov3, W)
        j = int(np.argmax(gains))
        if not np.isfinite(gains[j]):
            return None
        new_w = max(wsum[j] / W, _MIN_INSERT_SHARE / W)
        cov_new = np.array([[cov3[j, 0], cov3[j, 1]], [cov3[j, 1], cov3[j, 2]]])
        theta_next = _trial_insertion_theta(Kg_cur, theta, new_w, mean[j], cov_new)

        Kg_next = k_new + 1
        ll0 = _ll_at(prep, w, Kg_next, theta_next, pen)
        if not np.isfinite(ll0):
            return None
        result = _em_accelerated(prep, w, pen, Kg_next, theta_next, ll0, eta,
                                  budget=_INSERT_EM_STEPS)
        if result is None:
            return None
        theta, ll_cur = result[0], result[1]

    return theta


def _offband_mask(X, Bx, fil):
    """Rows outside the filament seed's own 2-sigma band, used by
    `_kmeans_starts_offband`: `|y - h(x)| > 2*sqrt(s2p)` OR `|x - m| >
    2*sqrt(s2x)`."""
    m, s2x, b0, b1, b2, s2p = fil
    h = Bx @ np.array([b0, b1, b2])
    resid = X[:, 1] - h
    return ((np.abs(resid) > _OFFBAND_SIGMA * math.sqrt(s2p))
            | (np.abs(X[:, 0] - m) > _OFFBAND_SIGMA * math.sqrt(s2x)))


def _kmeans_starts_offband(prep, w, Kg, fil, pi_f, n_starts, seed):
    """`n_starts` k-means starts (module docstring item 2(b)), clustering
    only the rows outside the filament seed's own 2-sigma band into Kg
    clusters; weights are cluster fractions of the off-band rows scaled
    by (1 - pi_f), covariances each cluster's own population covariance
    (falling back to the off-band data's own covariance / Kg on a
    cluster under 3 members, as gmm._starts). Returns a list of theta
    (possibly empty, if fewer than Kg rows sit outside the band)."""
    X = prep.X
    Bx = prep.Bx
    offband = _offband_mask(X, Bx, fil)
    Xo = X[offband]
    No = Xo.shape[0]
    if No < Kg:
        return []
    rng = np.random.default_rng(seed)
    data_cov = np.cov(Xo.T, bias=True) if No > 1 else np.eye(2)
    S0 = data_cov / Kg
    starts = []
    for _ in range(n_starts):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            centroids, labels = kmeans2(Xo, Kg, iter=_KMEANS_ITER, minit='++', seed=rng)
        counts = np.bincount(labels, minlength=Kg).astype(float)
        pis0 = (counts / No) * (1.0 - pi_f)
        mus0 = centroids.astype(float).copy()
        covs0 = np.empty((Kg, 2, 2))
        for k in range(Kg):
            pts = Xo[labels == k]
            if pts.shape[0] < 3:
                covs0[k] = S0
                continue
            c = np.cov(pts.T, bias=True)
            det = c[0, 0] * c[1, 1] - c[0, 1] ** 2
            covs0[k] = c if (np.isfinite(det) and det > 0.0) else S0
        theta0 = sm.pack(Kg, pis0, mus0, covs0, fil)
        starts.append(theta0)
    return starts


def _cold_search(prep, w, Kg, n_starts, seed, eta):
    """The cold search (module docstring): filament seed, the
    likelihood-gain greedy insertion (`_greedy_insert`) + off-band
    k-means starts (`_kmeans_starts_offband`), a fixed-budget screen
    (`_SCREEN_BUDGET` plain-SQUAREM-accelerated EM steps), the best
    `_K_KEEP` by screened ll each run to full convergence + the Newton
    finish, the best of THOSE (by final ll, among non-'infeasible'
    statuses) returned. Returns dict(fit, status, search) or None if
    nothing survived even the screen."""
    pen = sm.penalty_setup(prep, w)
    fil = _filament_seed(prep, w)
    pi_f = _PI_F_SEED

    candidates = []
    greedy_theta = _greedy_insert(prep, w, Kg, fil, pi_f, eta, seed)
    if greedy_theta is not None:
        candidates.append(greedy_theta)
    candidates.extend(_kmeans_starts_offband(prep, w, Kg, fil, pi_f, n_starts, seed))

    if not candidates:
        return None

    screened = []
    for theta0 in candidates:
        if not sm.feasible(Kg, theta0):
            continue
        ll0 = _ll_at(prep, w, Kg, theta0, pen)
        if not np.isfinite(ll0):
            continue
        result = _em_accelerated(prep, w, pen, Kg, theta0, ll0, eta, budget=_SCREEN_BUDGET)
        if result is None:
            continue
        theta_s, ll_s = result[0], result[1]
        screened.append(dict(theta=theta_s, ll=ll_s))

    if not screened:
        return None

    screened.sort(key=lambda d: -d['ll'])
    survivors = screened[:_K_KEEP]

    results = []
    for surv in survivors:
        fit, status = _converge_and_finish(prep, w, pen, Kg, surv['theta'], eta)
        if fit is not None:
            results.append((fit, status))
    if not results:
        return None

    def _rank(item):
        fit, status = item
        return (1 if status == 'converged' else 0, fit['ll'])

    results.sort(key=_rank, reverse=True)
    best_fit, best_status = results[0]
    search_info = dict(n_candidates=len(candidates), n_screened=len(screened),
                        n_survivors=len(survivors))
    return dict(fit=best_fit, status=best_status, search=search_info)


# ======================================================================
# Labelling: only the Kg Gaussian blocks are ever permuted; the
# filament block is always last and untouched (mirrors
# gmm._canonical_sort / _reference_sort, restated over (Kg,2,2)
# covariances rather than gmm.py's own (K,3) Ss triple).
# ======================================================================

def _covs_to_Ss(covs):
    Kg = covs.shape[0]
    Ss = np.empty((Kg, 3))
    Ss[:, 0] = covs[:, 0, 0]
    Ss[:, 1] = covs[:, 0, 1]
    Ss[:, 2] = covs[:, 1, 1]
    return Ss


def _label_gaussians(pis_g, mus, covs, mode, ref):
    if mode == 'canonical' or ref is None:
        order = np.lexsort((mus[:, 1], mus[:, 0]))
    else:
        ref_mus, ref_covs = ref
        Ss = _covs_to_Ss(covs)
        dist = gmm._bhattacharyya(mus, Ss, ref_mus, ref_covs)
        _, col = linear_sum_assignment(dist)
        order = np.argsort(col)
    return pis_g[order].copy(), mus[order].copy(), covs[order].copy(), order


# ======================================================================
# ShearMix2D.
# ======================================================================

class ShearMix2D:
    """See module docstring."""

    name = 'shearmix2d'
    takes_start = True

    def __init__(self, Kg: int = 4, omega: float = sm.OMEGA, n_starts: int = 20,
                 seed: int = 0, reference=None, eta: float = gmm._ETA_DEFAULT,
                 cond_max: float = gmm._COND_MAX_DEFAULT):
        self.Kg = int(Kg)
        self.omega = float(omega)
        self.n_starts = int(n_starts)
        self.seed = int(seed)
        self.reference = reference
        self.p = sm.n_params(self.Kg)
        self.outputs = sm.make_outputs(self.Kg)
        self.eta = float(eta)
        self.cond_max = float(cond_max)
        self.last_fit_info = None

    def prepare(self, X: np.ndarray):
        return sm.prepare(np.asarray(X, dtype=float), omega=self.omega)

    def loglik(self, X: np.ndarray, w: np.ndarray, theta: np.ndarray, prep=None) -> float:
        """The penalized log-likelihood per unit weight at `theta`, no
        fit. NaN when `theta` is infeasible."""
        w = np.asarray(w, dtype=float)
        prep_ = prep if prep is not None else self.prepare(X)
        theta = np.asarray(theta, dtype=float)
        if not sm.feasible(self.Kg, theta):
            return float('nan')
        pen = sm.penalty_setup(prep_, w)
        try:
            return float(sm.penalized_ll(prep_, w, self.Kg, theta, pen))
        except Exception:
            return float('nan')

    def _fit_impl(self, prep, w, eta, start):
        """(dict_or_None, status). `start` given: EM-continue + finish
        from `start`, labelled against start's OWN Gaussians. `start`
        None: the cold search, labelled against `self.reference` or the
        canonical sort."""
        pen = sm.penalty_setup(prep, w)
        Kg = self.Kg
        if start is not None:
            theta0 = np.asarray(start, dtype=float)
            fit, status = _converge_and_finish(prep, w, pen, Kg, theta0, eta)
            if fit is None:
                return None, status
            _, _, ref_mus, ref_covs, _ = sm.unpack(Kg, theta0)
            mode, ref = 'start', (ref_mus, ref_covs)
            search_info = None
        else:
            search = _cold_search(prep, w, Kg, self.n_starts, self.seed, eta)
            if search is None:
                return None, 'infeasible'
            fit, status = search['fit'], search['status']
            search_info = search['search']
            if self.reference is not None:
                mode, ref = 'reference', self.reference
            else:
                mode, ref = 'canonical', None

        pis_g, pi_f, mus, covs, fil = sm.unpack(Kg, fit['theta'])
        pis_g_l, mus_l, covs_l, label_order = _label_gaussians(pis_g, mus, covs, mode, ref)
        theta_labelled = sm.pack(Kg, pis_g_l, mus_l, covs_l, fil)

        out = dict(theta=theta_labelled, status=status, resid=fit['resid'], g0=fit['g0'],
                    n_iter_em=fit['n_iter_em'], n_iter_newton=fit.get('n_iter_newton', 0),
                    label_order=label_order, search=search_info, ll=fit['ll'])
        return out, status

    @staticmethod
    def _fit_info(fit, status) -> gmm.FitInfo:
        if fit is None:
            return gmm.FitInfo(status=status, score=float('nan'), n_iter_em=0,
                                n_iter_newton=0, label_order=None, search=None,
                                g0=float('nan'))
        return gmm.FitInfo(status=status, score=fit['resid'], n_iter_em=fit['n_iter_em'],
                            n_iter_newton=fit['n_iter_newton'], label_order=fit['label_order'],
                            search=fit.get('search'), g0=fit['g0'])

    def __call__(self, X: np.ndarray, w: np.ndarray, prep=None, start: np.ndarray = None,
                 eta: float = None, **kw) -> np.ndarray:
        try:
            w = np.asarray(w, dtype=float)
            prep_ = prep if prep is not None else self.prepare(X)
            start_ = None if start is None else np.asarray(start, dtype=float)
            eta_use = self.eta if eta is None else eta
            fit, status = self._fit_impl(prep_, w, eta_use, start_)
            self.last_fit_info = self._fit_info(fit, status)
            if status != 'converged':
                return np.full(self.p, np.nan)
            return fit['theta'].copy()
        except Exception:
            self.last_fit_info = self._fit_info(None, 'linalg')
            return np.full(self.p, np.nan)

    def influence(self, X: np.ndarray, w: np.ndarray, prep=None, start: np.ndarray = None,
                  eta: float = None, **kw) -> np.ndarray:
        N = len(X)
        try:
            w = np.asarray(w, dtype=float)
            prep_ = prep if prep is not None else self.prepare(X)
            start_ = None if start is None else np.asarray(start, dtype=float)
            eta_use = self.eta if eta is None else eta
            fit, status = self._fit_impl(prep_, w, eta_use, start_)
            self.last_fit_info = self._fit_info(fit, status)
            if status != 'converged':
                return np.full((N, self.p), np.nan)
            pen = sm.penalty_setup(prep_, w)
            A = sm.info(prep_, w, self.Kg, fit['theta'], pen)
            if np.linalg.cond(A) > self.cond_max:
                return np.full((N, self.p), np.nan)
            psi = sm.score_rows(prep_, w, self.Kg, fit['theta'], pen)
            IF = np.linalg.solve(A, psi.T).T
            if not np.all(np.isfinite(IF)):
                return np.full((N, self.p), np.nan)
            return IF
        except Exception:
            self.last_fit_info = self._fit_info(None, 'linalg')
            return np.full((N, self.p), np.nan)

    def fit_and_influence(self, X: np.ndarray, w: np.ndarray, prep=None,
                           start: np.ndarray = None, eta: float = None, **kw):
        N = len(X)
        nan_theta = np.full(self.p, np.nan)
        nan_psi = np.full((N, self.p), np.nan)
        try:
            w = np.asarray(w, dtype=float)
            prep_ = prep if prep is not None else self.prepare(X)
            start_ = None if start is None else np.asarray(start, dtype=float)
            eta_use = self.eta if eta is None else eta
            fit, status = self._fit_impl(prep_, w, eta_use, start_)
            self.last_fit_info = self._fit_info(fit, status)
            if status != 'converged':
                return nan_theta, nan_psi
            theta = fit['theta'].copy()
            pen = sm.penalty_setup(prep_, w)
            A = sm.info(prep_, w, self.Kg, theta, pen)
            if np.linalg.cond(A) > self.cond_max:
                return theta, nan_psi
            psi = sm.score_rows(prep_, w, self.Kg, theta, pen)
            IF = np.linalg.solve(A, psi.T).T
            if not np.all(np.isfinite(IF)):
                return theta, nan_psi
            return theta, IF
        except Exception:
            self.last_fit_info = self._fit_info(None, 'linalg')
            return nan_theta, nan_psi
