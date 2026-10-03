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
and forms `H_u` analytically, `shearmix_model.hessian`'s own Louis's-
identity Hessian pushed through `shearmix_model.chain`'s closed-form
curvature correction (that module's own mirror of `gmm_param.chain`);
the earlier central-finite-difference construction survives as
`_hess_u_fd`, for the validation check only); and the cold search (below),
which has no `GMM2D` analogue at all.

THE FAILURE this redesign fixes (measured on a TACC oracle run,
cloudfil_G_U_P3_v1, N=10,000, seed = draw index): an EARLIER version of
the cold search below -- gain-scoring insertion against the CURRENT
mixture, exactly as here, but growing straight from a single
all-row Gaussian with only `_INSERT_EM_STEPS = 20` accelerated-EM steps
after EVERY insertion, including the very first -- converged to a worse
optimum than the continuation from the truth on 352 of 628 draws (56%),
gap -0.0002 to -0.015/unit weight (median -0.0045); P2 (weight 0.01, at
(1.32, 0.55), sitting ON the filament) was missing in 97% of the
failing draws, P1 in 89%, with a near-empty 4th Gaussian (weight
~0.0006) typically dumped somewhere irrelevant. Cause: 20 EM steps is
not EM convergence -- scoring every insertion, including the first,
against a background (cloud + filament) that had NOT yet converged let
the half-fitted filament's own ridge residual look bigger than P2's
real signal (P2 sits ON the ridge, so it is the core most sensitive to
the filament's own state), so a Gaussian got spent elsewhere and local
EM never recovered it.

THE DESIGN (SCALE SEPARATION): the model is one large Gaussian (cloud)
plus `Kg - 1` small Gaussians (cores) plus one filament, and the cold
search is organised around that scale gap rather than treated as one
undifferentiated Kg-component search:

1. BACKGROUND FIRST (`_converge_background`): fit ONLY the large-scale
   structure -- one Gaussian plus the filament, no cores -- to FULL
   penalized-EM convergence (the same joint Aitken/scaled-gradient stop
   test as every other EM run in this module, `_em_accelerated` with
   `budget=None`, never a fixed step count). Seeded with the trimmed-LS
   filament seed (`_filament_seed`, unchanged from the design this
   replaces) and a Gaussian at the weighted mean/covariance of ALL rows
   (`_background_seed`) -- not just the rows outside the filament band:
   the three cores together are only ~3.5% of the mass, so they barely
   move the all-row mean/covariance, and using them needs no further
   choice of which rows to exclude. THIS is the fix for THE FAILURE
   above: by the time any core is scored against this model, the
   filament has nothing left to settle, so it cannot masquerade as a
   core's own signal.
2. CORES AGAINST THE CONVERGED BACKGROUND (`_grow_cores`/`_insert_best`):
   the over-segmentation -- the `_GREEDY_CELL_FACTOR*Kg`-cell k-means
   partition, unchanged in its own geometry from the design this
   replaces -- is now built in the CONVERGED background's own unsheared
   coordinates z = (x, y - h_hat(x)) (`_unsheared_coords`, `_core_cells`;
   the shear has Jacobian 1 and is exactly invertible, so it preserves
   every cell's row count, but in z the filament is a straight band
   instead of a curved ridge, so a round k-means cell there tracks the
   ridge's own width instead of straddling its curve -- unchanged
   reasoning from the earlier design, just anchored to the converged
   filament rather than the seed). Every eligible cell is FIRST refined
   by partial EM before it is scored (`_refine_candidates`: Verbeek,
   Vlassis and Kroese 2003's own partial-EM step, not just their gain
   criterion) -- starting from the cell's own weighted mean/covariance,
   update ONLY that candidate's own weight/mean/covariance from its own
   responsibility against the FULL trial mixture, every existing
   component and the filament held fixed, to the same relative-ll
   convergence test used everywhere else in this module. This is
   needed because a cell's raw weighted statistics can be too broad to
   win on a first try -- in particular P2's own cell, which straddles
   the filament, includes enough ridge rows in its raw covariance that
   the UNREFINED candidate's gain against a converged background was
   measured slightly NEGATIVE (-0.00006) on one draw (see "Draw 1" below)
   -- but refinement lets a real core tighten onto the rows that
   actually belong to it before being judged. THEN candidates are
   scored by penalized-likelihood GAIN against the CONVERGED current
   model (`_insertion_gains`, now scoring the REFINED candidate: still
   Verbeek, Vlassis and Kroese 2003's own greedy-EM criterion, just no
   longer skipping their paper's own refinement step); the best is
   inserted AT ITS REFINED PARAMETERS and the WHOLE model is
   re-converged by full EM (never a fixed budget); this repeats until
   `Kg - 1` cores are present. A cell already spent is not excluded from
   later rounds -- once absorbed into the model its own gain drops well
   below the next real core's (the gate check below measured this
   directly: after P1 is added and the model re-converges, P1's own
   cell drops out of the top 2 and the remaining two cores stay there),
   so no separate bookkeeping is needed.
3. FINAL RELOCATION (`_relocate_cores`): for each of the `Kg - 1` cores
   in turn (the cloud is never removed), remove it, re-converge the rest
   to full EM convergence, re-insert by the SAME gain rule against that
   converged rest, re-converge again, and keep the relocation only if
   the penalized ll rises (strictly) over the model at the start of this
   core's own attempt; otherwise the core is left where it was. Repeats
   passes (one sweep over all `Kg - 1` cores each) until a pass accepts
   no relocation, capped at `_RELOC_PASS_CAP` sweeps as a safety net
   (that constant's own comment). Then the usual EM-converge + Newton
   finish (`_converge_and_finish`), which, started from an
   already-converged `theta`, settles immediately.
No split-and-merge stage, and no off-band k-means multistart: the build
interface's search (section 4) asks for neither as a hard requirement
("keep the existing k-means/other starts only if they still add
something... simpler is better"), and the validation below (both draws
checked) shows the deterministic path above alone reaches the
continuation-from-truth's own penalized ll to ~1e-15 -- far inside the
`eta` tolerance -- with no competing candidate pool of any kind. The
earlier design's off-band k-means starts (`_kmeans_starts_offband`) and
fixed-budget racing screen (`_SCREEN_BUDGET`/`_K_KEEP`, borrowed from
`GMM2D`'s own multistart convention) are REMOVED for the same reason the
build interface gives for GMM2D's own split-and-merge needing no analog
here: the filament seed already supplies the model-driven explanation of
the large-scale structure that a Gaussians-only multistart could not
find, and now that scoring happens against a CONVERGED background
instead of an immature one, the single gain-driven growth path finds
every core directly -- there is nothing left for a second candidate pool
to add, and `n_starts` k-means starts clustering rows that are mostly
explained by one converged Gaussian already was always a weaker
construction than the gain rule itself.

THE GATE CHECK (`scripts/shearmix_checks/scale_gate.py`; the task's own
required check before any of this was built): on draw 2 (a FAILING draw
under the OLD design, gap -0.0040, P2 missed), scoring all 192 eligible
cells of the over-segmentation against the CONVERGED background
(`_converge_background`, 9 EM iterations) ranked P1's cell 1st
(gain 0.00417), P3's cell 2nd (0.00393), P2's cell 3rd (0.00049) --
EXACTLY the three core cells as the top 3, with the best non-core cell
at 0.00015 (3x smaller than P2's own gain, the smallest of the three).
After adding P1 (the top-ranked cell) and re-converging to Kg=2, the
remaining two cores (P3, P2) stayed the top 2 of the re-ranked cells
(0.00394, 0.00120), confirming the design's core claim: against a
CONVERGED background, the three true cores -- even the weak, on-ridge
P2 -- dominate every other cell, with no further repair needed.

PARTIAL-EM REFINEMENT, AND WHY `_MIN_CELL_COUNT` IS 3, NOT A TUNED
FLOOR: an earlier version of this module raised `_MIN_CELL_COUNT` from
5 to 30 after draw 1's own cold fit chose a 5-row cell ~4 sigma out in
the cloud's own tail at its third insertion (gain 0.00040) OVER P2's
own cell (gain -0.00006, rank 15/193) -- the author's ruling (one
constant must never be set from one draw, feedback_constants.md) is
that this was the wrong fix: raising the floor hid the real problem
rather than solving it. The real problem is that P2's own RAW cell gain
was NEGATIVE: P2 sits ON the filament's ridge, so its over-segmentation
cell's raw weighted mean/covariance (unrefined, straight off `_cell_stats`)
includes enough ridge rows to make its candidate Gaussian too broad to
beat a converged background on a first try. `_refine_candidates` (THE
DESIGN, item 2, above) fixes this directly: every candidate, P2's own
included, is now refined by partial EM -- against the FULL trial
mixture, every existing component held fixed -- before it is scored,
letting it tighten onto the rows that actually belong to it. Measured
on draw 1 at the exact same point as the -0.00006 failure above (Kg_cur
= 1, scoring against the converged background alone): P2's own cell's
REFINED gain is +0.005260 (still rank 15/197, since P1's cell wins this
round at 0.008406, but now a real, positive, order-of-magnitude-larger
signal rather than a sign flip away from a spurious 5-row outlier
winning). `_MIN_CELL_COUNT` is restored to 3 -- the mathematical
minimum for `_cell_stats` to supply any covariance at all (that
constant's own comment) -- and no second, higher floor is layered back
on top: refinement, not a bigger floor, is what makes a genuinely
compact core's cell win over a coincidentally tight handful of
stragglers, because refinement can expand OR shrink a candidate's own
support from the full-row responsibilities, not just accept whatever
rows the over-segmentation happened to assign it.

VALIDATION (`scripts/shearmix_checks/scale_validate_refine.py`; the
task's own required check after building): on draw 2, the cold fit
converges in 580.3s wall time to penalized ll -2.850559, matching the
continuation from the truth (ll -2.850559) to a gap of -8.9e-16/unit
weight (task's own pass threshold >= -1e-8); P1/P2/P3 recovered within
0.003-0.02 of their true means (P2 at (1.3228, 0.5478) vs truth
(1.32, 0.55)). On draw 1 (seed 1): the cold fit converges in 532.0s to
ll -2.858917, gap 0.0 against the same continuation, P1/P2/P3 again
recovered within 0.003-0.02 of truth. Both gaps are at the SAME ~1e-15
numerical-identity scale as the pre-refinement design's own validation,
i.e. refinement does not just fix P2's sign on one draw, it still finds
the exact global optimum the continuation-from-truth finds on both.

Every accepted insertion's own ROW COUNT, on both draws, is far above
the `_MIN_CELL_COUNT = 3` floor -- draw 2: 224, 67, 46; draw 1: 217, 60,
20 -- and every accepted candidate's refined covariance determinant is
O(1e-6) to O(1e-7) on BOTH draws, for all three (genuine) cores: this is
the real cores' own physical scale, not a refinement artifact, confirmed
by the exact match to the continuation-from-truth optimum above. The
WATCH FOR case the task asked to be alert to -- a few coincidentally
tight tail-outlier rows at the `_MIN_CELL_COUNT = 3` floor winning on
gain through refinement -- was NOT observed on either draw: no accepted
insertion is anywhere near the floor, and relocation (which would
revert any such spurious insertion once the rest of the model had
re-converged around it) found nothing to revert on either draw (`ll`
unchanged bit-for-bit through relocation, both draws -- a single
accepted pass, consistent with growth alone already reaching the global
optimum). Cost: refinement is the dominant cost of the cold search --
each `_insertion_gains` call (one per growth insertion, one per
relocation attempt) costs ~20-30s at N=10,000 with ~195-197 eligible
candidates refined together in the one vectorised (N, M) pass per
refinement iteration that function's own docstring describes (never one
candidate at a time); growth itself (3 insertions) is ~140-155s per
draw, relocation (one pass, both draws) ~370-435s, pushing total cold-
search wall time from the pre-refinement design's 8-15s to 530-580s --
still comfortably inside the task's own 10-minute-per-run cap, but the
single largest cost in this module by a wide margin.

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
# design (build interface section 4 item 1); the cell factor/min insert
# share/k-means iteration cap are the over-segmentation's own geometry
# choices (unchanged from the design this module replaces); the EM/
# Newton iteration caps and FD step mirror gmm.py's/shearmix_model.py's
# own. `_RELOC_PASS_CAP` is new (module docstring, "THE DESIGN", item 3):
# a safety cap on the number of final-relocation SWEEPS over the Kg-1
# cores, never expected to bind -- a relocation is only kept when it
# raises the penalized ll (strictly), so with only Kg-1 = 3 cores to
# cycle through, a handful of sweeps is ample margin before the process
# necessarily settles; mirrors the safety-cap convention already used by
# `_EM_ITER_CAP_FACTOR`/`_NEWTON_ITER_CAP_FACTOR` (a generous bound on a
# process that terminates on its own, not a tuned threshold).
#
# `_MIN_CELL_COUNT` is the MATHEMATICAL MINIMUM to estimate a 2-D
# covariance at all, not a tuned value (author's ruling, after an
# earlier version of this module raised it 5 -> 30 on the strength of
# one draw's own failure -- feedback_constants.md: never set a constant
# from one draw). A symmetric 2x2 covariance has 3 free parameters
# (S11, S12, S22); the w-weighted sample covariance about a cell's own
# mean has rank <= min(members - 1, 2), so it is only (generically)
# positive-definite -- i.e. an actual 2-D covariance, not a degenerate
# one -- once a cell has >= 3 members. `seeding._cell_stats` (reused
# unchanged below) already bakes this exact threshold in on its own
# terms: a cell's covariance is computed only for `hi - lo >= 3`
# members (3 or fewer gives the zero placeholder its own docstring
# describes as "never read"); `_MIN_CELL_COUNT = 3` is the eligibility
# floor that matches what that helper can actually supply, nothing
# more. Any residual risk from a handful of coincidentally tight
# straggler rows winning the gain rule at this floor is now the partial-
# EM refinement's own concern (`_refine_candidates`, below), not a
# second, higher floor layered on top of this one -- see that
# function's docstring and VALIDATION in the module docstring for what
# was actually observed on the two draws checked.
# ----------------------------------------------------------------------
_TRIM_ROUNDS = 5
_TRIM_FRACTION = 0.4
_PI_F_SEED = 0.35

_GREEDY_CELL_FACTOR = 50
_MIN_CELL_COUNT = 3
_MIN_INSERT_SHARE = 5.0
_KMEANS_ITER = 300

_RELOC_PASS_CAP = 5

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
    own step rule), symmetrized. SUPERSEDED in the Newton finish by the
    analytic `_hess_u_analytic` (below); kept only for the validation
    check against it."""
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


def _hess_u_analytic(prep, w, pen, Kg, u, g_theta, theta):
    """H_u via `shearmix_model.hessian`'s analytic Louis's-identity
    Hessian, pushed through `shearmix_model.chain` (mirrors
    `gmm_param.chain`'s own curvature-correction term). Replaces
    `_hess_u_fd` (~2p gradient evaluations per call) with one `hessian`
    call. `g_theta`, `theta` are the caller's own (already computed at
    this `u`), so no redundant `grad`/`from_u` evaluation."""
    H_theta = sm.hessian(prep, w, Kg, theta, pen)
    _, H_u = sm.chain(Kg, u, g_theta, H_theta)
    return H_u


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
            H_u = _hess_u_analytic(prep, w, pen, Kg, u, g_theta, theta)
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
# Cold search: converged background first, cores grown one at a time by
# likelihood-gain insertion against the CONVERGED current model, then a
# final relocation pass (build interface section 4; see the module
# docstring, "THE DESIGN", for the full design and the gate check it was
# validated against).
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


def _background_seed(prep, w, pen, fil):
    """Kg=1 theta: the trimmed-LS filament seed `fil` plus ONE Gaussian
    at the weighted mean/covariance of ALL rows (`pen.Scov`, about its
    own weighted mean), weight `1 - _PI_F_SEED`. ALL rows, not just the
    rows outside the filament band: the three cores together are only
    ~3.5% of the mass (THE DESIGN, item 1), so they barely move the
    all-row mean/covariance, and using them needs no further choice of
    which rows to exclude."""
    X = prep.X
    xbar = (w @ X) / pen.W
    pis_g = np.array([1.0 - _PI_F_SEED])
    mus = xbar[None, :]
    covs = pen.Scov[None, :, :].copy()
    return sm.pack(1, pis_g, mus, covs, fil)


def _converge_background(prep, w, pen, eta):
    """THE DESIGN, item 1 (BACKGROUND FIRST): the filament seed
    (`_filament_seed`) plus the single all-row Gaussian (`_background_seed`),
    run to FULL penalized-EM convergence (`_em_accelerated`, `budget=None`
    -- the same joint Aitken/scaled-gradient stop test as every other EM
    run in this module, not a fixed step count) before any core is
    scored against it. This is the fix for THE FAILURE (module
    docstring): the earlier design scored every insertion, including the
    FIRST, against a background only 20 EM steps into convergence, and
    the half-fitted filament's own ridge residuals could then outscore a
    real core (P2, sitting ON the ridge) at a wrong cell. Returns
    (theta, ll, fil) -- `fil` the CONVERGED filament, used by
    `_core_cells` to build the over-segmentation in unsheared
    coordinates -- or None if the background itself degenerates."""
    fil = _filament_seed(prep, w)
    theta0 = _background_seed(prep, w, pen, fil)
    ll0 = _ll_at(prep, w, 1, theta0, pen)
    if not np.isfinite(ll0):
        return None
    result = _em_accelerated(prep, w, pen, 1, theta0, ll0, eta, budget=None)
    if result is None:
        return None
    theta, ll = result[0], result[1]
    _, _, _, _, fil_bg = sm.unpack(1, theta)
    return theta, ll, fil_bg


def _trial_insertion_theta(Kg_cur, theta_cur, new_w, new_mu, new_cov):
    """`theta_cur` (Kg_cur Gaussians + filament) with one further
    Gaussian appended at (`new_mu`, `new_cov`, weight `new_w`), the
    existing Kg_cur Gaussians rescaled by (1 - new_w) with their own
    means/covariances held FIXED, the filament block untouched (its
    share of 1 - sum(pis_g) is implied, not stored, so rescaling the
    Gaussian weights alone also rescales pi_f correctly). Used by
    `_insert_best` to build the actually-accepted insertion at the
    WINNING candidate's refined (weight, mean, covariance)
    (`_insertion_gains`/`_refine_candidates`); scoring itself no longer
    builds this full `Kg_cur + 1`-component theta for every candidate --
    `_refine_candidates`'s own docstring explains the two-term-mixture
    shortcut that replaces it during refinement/gain-scoring."""
    pis_g, _, mus, covs, fil = sm.unpack(Kg_cur, theta_cur)
    pis_g_new = np.concatenate([pis_g * (1.0 - new_w), [new_w]])
    mus_new = np.concatenate([mus, new_mu[None, :]], axis=0)
    covs_new = np.concatenate([covs, new_cov[None, :, :]], axis=0)
    return sm.pack(Kg_cur + 1, pis_g_new, mus_new, covs_new, fil)


def _refine_candidates(prep, w, pen, Kg_cur, theta_cur, logf_old, eligible,
                        wsum, mean, cov3, W, eta):
    """Partial-EM refinement of EVERY eligible candidate cell (Verbeek,
    Vlassis and Kroese 2003), run BEFORE gain-scoring: starting from the
    cell's own weighted mean/covariance (`mean[j]`, `cov3[j]`,
    `_cell_stats`) at weight `max(wsum[j]/W, _MIN_INSERT_SHARE/W)` (the
    SAME initial weight `_insert_best` used to use directly), update
    ONLY that candidate's own (weight, mean, covariance) from its own
    responsibility against the FULL trial mixture -- the `Kg_cur`
    existing Gaussians and the filament, all of `theta_cur`, held FIXED
    throughout, their total mass rescaled by `(1 - new weight)` -- to
    the SAME relative-ll convergence test `_em_accelerated` uses
    elsewhere in this module (`|ell_k - ell_{k-1}| <= eta *
    max(|ell_k|, 1)`, reusing `eta`, no new tuned number), capped at
    `_EM_ITER_CAP_FACTOR * sm.n_params(Kg_cur + 1)` iterations -- the
    SAME `_EM_ITER_CAP_FACTOR * p` safety-cap convention `_em_accelerated`
    itself uses, `p` here the full trial model's own parameter count
    (plain, unaccelerated EM steps, so this candidate-only refinement
    needs more of them than a SQUAREM-accelerated full-model run to hit
    the same tolerance; measured empirically never to bind -- see this
    function's own cost report in the module docstring). The new covariance's own update is the model's
    usual penalized M-step blend (Chen-Tan with `pen.Scov`, exactly
    `m_step`'s per-Gaussian formula) -- the weight and mean are
    unpenalized, also exactly as `m_step`.

    Why this is cheap to VECTORISE across all `M` eligible candidates at
    once rather than running `M` separate single-candidate EM loops:
    because the existing components and the filament are frozen, their
    combined contribution to the trial density at any row is exactly
    `(1 - new_w) * f_old(x)`, `f_old` the CURRENT (`theta_cur`)
    mixture's own density -- `logf_old`, passed in from the caller's own
    E-step at `theta_cur`, computed ONCE and reused for every candidate
    and every refinement iteration, never recomputed. The full trial
    density is therefore the two-term mixture

        f_trial(x) = (1 - new_w) * f_old(x) + new_w * N(x | new_mu, new_cov),

    collapsing what would otherwise be a `(Kg_cur + 2)`-column E-step
    (`Kg_cur` existing Gaussians, the candidate, the filament) into one
    `logsumexp` of two terms, and likewise collapsing the penalized ll
    into `ll_cur`'s own existing-component penalty total (`pen_existing`,
    `_penalty_value` on `theta_cur` -- also computed ONCE, constant
    across every candidate and iteration) plus the candidate's own
    covariance penalty term alone. What is vectorised is this: one
    (N, M) quadratic-form/logsumexp/responsibility pass per refinement
    iteration, across every still-active candidate at once, rather than
    M separate (N, 1) passes -- see the module docstring for the
    measured cost on both draws checked.

    A candidate is frozen (dropped from the active set, its last theta
    kept) the iteration its own relative-ll test first passes, or the
    moment it degenerates (non-PD trial covariance, non-finite or
    non-positive responsibility-weighted count) -- the latter scored
    -inf, same convention as an ineligible cell.

    Returns (new_w, new_mu, new_cov, ll_trial), each (M,)/(M,2)/(M,2,2)/
    (M,): the refined candidate's own parameters and penalized ll
    (-inf, and a `new_w`/`new_mu`/`new_cov` not meant to be read, at
    every ineligible or degenerated cell)."""
    X = prep.X
    Wtot, a_pen, Scov = pen
    pen_existing = sm._penalty_value(Kg_cur, theta_cur, Scov)
    M = eligible.shape[0]

    # FIX (A): GMM2D's own feature-matrix pattern (gmm._build_features/
    # _log_density_coeffs) -- QX built ONCE for this call; every
    # candidate's log N(x|mu_j,cov_j) across all N rows becomes the one
    # matmul QX @ C below, and the weighted second moments become the
    # one matmul wr.T @ XP, replacing the (N, m, 2) diff/diff2 arrays and
    # the einsum this function used to build every iteration.
    QX, XP = gmm._build_features(X)

    new_w = np.where(eligible, np.maximum(wsum / W, _MIN_INSERT_SHARE / W), np.nan)
    new_mu = mean.copy()
    new_cov = np.empty((M, 2, 2))
    new_cov[:, 0, 0] = cov3[:, 0]
    new_cov[:, 0, 1] = new_cov[:, 1, 0] = cov3[:, 1]
    new_cov[:, 1, 1] = cov3[:, 2]

    ll_trial = np.full(M, -np.inf)
    ll_prev = np.full(M, np.nan)
    ll_prev2 = np.full(M, np.nan)
    active = eligible.copy()

    iter_cap = _EM_ITER_CAP_FACTOR * sm.n_params(Kg_cur + 1)
    for _ in range(iter_cap):
        idx = np.nonzero(active)[0]
        if idx.size == 0:
            break
        w_j, mu_j, cov_j = new_w[idx], new_mu[idx], new_cov[idx]
        a_, b_, c_ = cov_j[:, 0, 0], cov_j[:, 0, 1], cov_j[:, 1, 1]
        det = a_ * c_ - b_ * b_
        feasible_j = (np.isfinite(det) & (det > 0.0) & np.isfinite(w_j)
                      & (w_j > 0.0) & (w_j < 1.0))
        if not feasible_j.all():
            bad = idx[~feasible_j]
            ll_trial[bad] = -np.inf
            active[bad] = False
            idx = idx[feasible_j]
            w_j, mu_j, cov_j = w_j[feasible_j], mu_j[feasible_j], cov_j[feasible_j]
            a_, b_, c_, det = a_[feasible_j], b_[feasible_j], c_[feasible_j], det[feasible_j]
            if idx.size == 0:
                continue

        inv00, inv01, inv11 = c_ / det, -b_ / det, a_ / det
        logdet = np.log(det)
        C = gmm._log_density_coeffs(np.ones(idx.size), mu_j, a_, b_, c_, det)
        logN = QX @ C  # (N, m), one matmul in place of the old (N, m, 2) quad form
        a_term = np.log1p(-w_j) + logf_old[:, None]
        b_term = np.log(w_j) + logN
        hi = np.maximum(a_term, b_term)
        logf_tr = hi + np.log1p(np.exp(np.minimum(a_term, b_term) - hi))  # (N, m)
        r = np.exp(b_term - logf_tr)

        pen_cand = (Scov[0, 0] * inv00 + Scov[1, 1] * inv11
                    + 2.0 * Scov[0, 1] * inv01 + logdet)
        ll_now = (w @ logf_tr) / Wtot - a_pen * (pen_existing + pen_cand) / Wtot

        prev = ll_prev[idx]
        prev2 = ll_prev2[idx]
        converged_now = (np.isfinite(prev)
                          & (np.abs(ll_now - prev) <= eta * np.maximum(np.abs(ll_now), 1.0)))
        ll_trial[idx] = ll_now
        ll_prev2[idx] = prev
        ll_prev[idx] = ll_now
        active[idx[converged_now]] = False

        # FIX (B): racing via GMM2D's own Aitken projection
        # (gmm._aitken_screen, reused unchanged). Once a still-active
        # candidate has three recorded ll values, project its limit;
        # drop (freeze at its current ll, the score it keeps) any whose
        # projected limit cannot beat the leader's CURRENT ll (the best
        # ll among every candidate scored so far, active or already
        # frozen) -- it cannot win the gain race, so no further
        # partial-EM spend on it is needed.
        not_converged = ~converged_now
        has_hist = np.isfinite(prev2)
        check = not_converged & has_hist
        if check.any():
            leader_ll = np.max(ll_trial[np.isfinite(ll_trial)])
            check_pos = np.nonzero(check)[0]
            ell_inf = np.array([gmm._aitken_screen(prev2[k], prev[k], ll_now[k])[0]
                                 for k in check_pos])
            drop_idx = idx[check_pos[ell_inf < leader_ll]]
            if drop_idx.size:
                active[drop_idx] = False

        still = active[idx]
        if still.any():
            idx_still = idx[still]
            wr = w[:, None] * r[:, still]
            nk = wr.sum(axis=0)
            ok = np.isfinite(nk) & (nk > 0.0)
            upd_idx = idx_still[ok]
            if upd_idx.size:
                wr_ok = wr[:, ok]
                nk_ok = nk[ok]
                M5 = wr_ok.T @ XP  # (m_upd, 5): sum w*[x,y,x^2,xy,y^2]
                mux = M5[:, 0] / nk_ok
                muy = M5[:, 1] / nk_ok
                Exx = M5[:, 2] / nk_ok
                Exy = M5[:, 3] / nk_ok
                Eyy = M5[:, 4] / nk_ok
                S11_raw = Exx - mux * mux
                S12_raw = Exy - mux * muy
                S22_raw = Eyy - muy * muy
                denom_pen = nk_ok + 2.0 * a_pen
                S11 = (nk_ok * S11_raw + 2.0 * a_pen * Scov[0, 0]) / denom_pen
                S12 = (nk_ok * S12_raw + 2.0 * a_pen * Scov[0, 1]) / denom_pen
                S22 = (nk_ok * S22_raw + 2.0 * a_pen * Scov[1, 1]) / denom_pen
                new_w[upd_idx] = nk_ok / Wtot
                new_mu[upd_idx] = np.stack([mux, muy], axis=-1)
                new_cov[upd_idx, 0, 0] = S11
                new_cov[upd_idx, 0, 1] = new_cov[upd_idx, 1, 0] = S12
                new_cov[upd_idx, 1, 1] = S22
            dead_idx = idx_still[~ok]
            if dead_idx.size:
                ll_trial[dead_idx] = -np.inf
                active[dead_idx] = False
    return new_w, new_mu, new_cov, ll_trial


def _insertion_gains(prep, w, pen, Kg_cur, theta_cur, ll_cur, eligible, wsum, mean, cov3, W, eta):
    """Likelihood-GAIN score for every eligible over-segmentation cell
    (Verbeek, Vlassis and Kroese 2003's own greedy-EM criterion, in
    place of a density/area approximation), AFTER refining each
    candidate's own (weight, mean, covariance) by partial EM against
    the CURRENT mixture (`_refine_candidates` -- its own docstring for
    why this is needed: a cell's raw weighted mean/covariance can be too
    broad, e.g. including filament rows at a core that sits ON the
    ridge, to beat the current mixture on a first try; refinement lets
    it tighten onto the rows that actually belong to it before it is
    judged). The existing components (and the filament) are rescaled by
    (1 - new weight) but otherwise held fixed throughout the refinement.
    Returns (gains, new_w, new_mu, new_cov): `gains` (M_cells,), -inf at
    every ineligible cell or one whose refinement itself degenerates;
    `new_w`/`new_mu`/`new_cov` the REFINED candidate parameters (what
    `_insert_best` actually inserts for the winner), None/None/None if
    `theta_cur`'s own E-step could not be formed."""
    M = eligible.shape[0]
    try:
        _, logf_old = sm.e_step(prep, Kg_cur, theta_cur)
    except np.linalg.LinAlgError:
        return np.full(M, -np.inf), None, None, None
    new_w, new_mu, new_cov, ll_trial = _refine_candidates(
        prep, w, pen, Kg_cur, theta_cur, logf_old, eligible, wsum, mean, cov3, W, eta)
    gains = np.where(np.isfinite(ll_trial), ll_trial - ll_cur, -np.inf)
    return gains, new_w, new_mu, new_cov


def _unsheared_coords(X, Bx, fil):
    """z = (x, y - h_hat(x)), h_hat the CONVERGED background's own
    filament (`fil`, `_converge_background`'s output; `b0, b1, b2`
    against the model's fixed-omega basis `Bx`). The shear has Jacobian
    1 and is exactly invertible (x is untouched; y is shifted by a
    function of x alone), so it preserves every cell's row count --
    only the over-segmentation's OWN geometry changes: in z the filament
    is a straight band at z2 ~ 0 of width ~sigma_perp, so a round k-means
    cell there tracks the ridge instead of straddling its curve
    (`_core_cells`'s own docstring)."""
    m, s2x, b0, b1, b2, s2p = fil
    h_hat = Bx @ np.array([b0, b1, b2])
    return np.column_stack([X[:, 0], X[:, 1] - h_hat])


def _core_cells(prep, w, fil_bg, Kg, seed, cell_factor=_GREEDY_CELL_FACTOR):
    """THE DESIGN, items 2-3's shared over-segmentation: the
    `cell_factor*Kg`-cell k-means over-segmentation, built ONCE in the
    CONVERGED background's own unsheared coordinates (`_unsheared_coords`,
    `fil_bg` from `_converge_background` -- not the raw filament seed, so
    the cells track the ridge the model has actually settled on), and
    reused unchanged by every insertion/relocation round that follows:
    only the theta/ll being scored against changes round to round, never
    the cells themselves (candidate Gaussians are still initialised from
    each chosen cell's own rows in ORIGINAL (x) coordinates -- `_cell_stats`
    called on `X` itself; only cell MEMBERSHIP comes from the z-space
    k-means). Returns (wsum, mean, cov3, eligible), `eligible` the cells
    with at least `_MIN_CELL_COUNT` rows."""
    X, Bx = prep.X, prep.Bx
    N = X.shape[0]
    M = min(cell_factor * Kg, max(Kg, N - 1))
    Z = _unsheared_coords(X, Bx, fil_bg)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        centres_z, bmu = kmeans2(Z, M, iter=_KMEANS_ITER, minit='++',
                                  seed=np.random.default_rng(seed))
    count, wsum, _, mean, cov3 = _cell_stats(X, w, centres_z, bmu, M)
    eligible = count >= _MIN_CELL_COUNT
    return wsum, mean, cov3, eligible


def _insert_best(prep, w, pen, Kg_cur, theta_cur, ll_cur, wsum, mean, cov3, eligible, W, eta):
    """Score every eligible cell against the CONVERGED `theta_cur`/
    `ll_cur`, each FIRST refined by partial EM (`_insertion_gains`/
    `_refine_candidates`), insert the single best cell AT ITS REFINED
    (weight, mean, covariance) (`_trial_insertion_theta`), and
    re-converge the WHOLE model to full EM convergence
    (`_em_accelerated`, `budget=None` -- THE DESIGN, items 2 and 3:
    're-converge the whole model'/'re-converge the rest', never a fixed
    step budget). Returns (theta_new, ll_new, cell_index) or None if no
    eligible cell scores finite or the resulting insertion/reconvergence
    degenerates."""
    gains, new_w, new_mu, new_cov = _insertion_gains(
        prep, w, pen, Kg_cur, theta_cur, ll_cur, eligible, wsum, mean, cov3, W, eta)
    j = int(np.argmax(gains))
    if not np.isfinite(gains[j]):
        return None
    theta_next0 = _trial_insertion_theta(Kg_cur, theta_cur, new_w[j], new_mu[j], new_cov[j])
    Kg_next = Kg_cur + 1
    ll0 = _ll_at(prep, w, Kg_next, theta_next0, pen)
    if not np.isfinite(ll0):
        return None
    result = _em_accelerated(prep, w, pen, Kg_next, theta_next0, ll0, eta, budget=None)
    if result is None:
        return None
    theta_next, ll_next = result[0], result[1]
    return theta_next, ll_next, j


def _grow_cores(prep, w, pen, Kg, theta_bg, ll_bg, wsum, mean, cov3, eligible, W, eta):
    """THE DESIGN, item 2 (CORES AGAINST THE CONVERGED BACKGROUND): grow
    the converged background (Kg_cur=1) to `Kg` components by calling
    `_insert_best` `Kg-1` times in a row -- each insertion scored
    against, and followed by a full reconvergence of, the model the
    PREVIOUS insertion already converged (never the background alone,
    never a partial-EM intermediate). A cell already spent is not
    excluded from later rounds: once absorbed into the model its own
    gain drops well below the next real core's (the gate check this
    module's design was built against measured this directly -- module
    docstring -- so no separate bookkeeping is needed). Returns
    (theta, ll) at `Kg` components, or None if growth degenerates at any
    step."""
    theta, ll = theta_bg, ll_bg
    for Kg_cur in range(1, Kg):
        out = _insert_best(prep, w, pen, Kg_cur, theta, ll, wsum, mean, cov3, eligible, W, eta)
        if out is None:
            return None
        theta, ll, _ = out
    return theta, ll


def _remove_gaussian(Kg_cur, theta, k):
    """`theta` with Gaussian `k` (1 <= k < Kg_cur; the cloud, index 0,
    is never removed -- `_relocate_cores`) dropped and the remaining
    Gaussians rescaled by `1 / (1 - pi_k)` so they (and the untouched
    filament) again sum to `1 - pi_f`: the exact inverse of
    `_trial_insertion_theta`'s own rescale-by-`(1 - new_w)` construction."""
    pis_g, _, mus, covs, fil = sm.unpack(Kg_cur, theta)
    pi_k = pis_g[k]
    keep = [i for i in range(Kg_cur) if i != k]
    pis_rest = pis_g[keep] / (1.0 - pi_k)
    return sm.pack(Kg_cur - 1, pis_rest, mus[keep], covs[keep], fil)


def _relocate_cores(prep, w, pen, Kg, theta, ll, wsum, mean, cov3, eligible, W, eta):
    """THE DESIGN, item 3 (FINAL RELOCATION): for each of the `Kg - 1`
    cores in turn (component 0, the cloud, is never touched), remove it
    (`_remove_gaussian`), re-converge the rest to full EM convergence,
    re-insert by the SAME likelihood-gain rule against that converged
    rest (`_insert_best` -- the core may legitimately land back in its
    own cell, or move to a different one), and keep the relocation only
    if it raises the penalized ll (strictly) over `ll` (the model at the
    START of this core's own attempt, with the core still in its old
    place); otherwise this core is left untouched. Repeats PASSES (one
    sweep over all `Kg - 1` cores each) until a pass accepts no
    relocation at all, capped at `_RELOC_PASS_CAP` sweeps as a safety net
    that is not expected to bind (see that constant's own comment).
    Returns (theta, ll) at the (possibly unchanged) `Kg`-component
    model."""
    for _ in range(_RELOC_PASS_CAP):
        changed = False
        for k in range(1, Kg):
            theta_rest0 = _remove_gaussian(Kg, theta, k)
            ll_rest0 = _ll_at(prep, w, Kg - 1, theta_rest0, pen)
            if not np.isfinite(ll_rest0):
                continue
            result = _em_accelerated(prep, w, pen, Kg - 1, theta_rest0, ll_rest0, eta, budget=None)
            if result is None:
                continue
            theta_rest, ll_rest = result[0], result[1]
            out = _insert_best(prep, w, pen, Kg - 1, theta_rest, ll_rest,
                                wsum, mean, cov3, eligible, W, eta)
            if out is None:
                continue
            theta_new, ll_new, _ = out
            if ll_new > ll:
                theta, ll = theta_new, ll_new
                changed = True
        if not changed:
            break
    return theta, ll


def _cold_search(prep, w, Kg, n_starts, seed, eta):
    """The cold search (module docstring, THE DESIGN): background first
    (`_converge_background` -- ONE Gaussian over all rows plus the
    trimmed-LS filament seed, run to full EM convergence before anything
    is scored against it); the shared over-segmentation built in the
    CONVERGED background's own unsheared coordinates (`_core_cells`);
    `Kg - 1` cores grown one at a time by likelihood-gain insertion
    against the converged current model, each insertion followed by a
    full reconvergence (`_grow_cores`); final relocation, each core in
    turn removed/reconverged/reinserted/reconverged and kept only if the
    penalized ll rises (`_relocate_cores`); then the usual EM-converge +
    Newton finish (`_converge_and_finish`) -- which, started from an
    already-converged `theta`, settles immediately. `n_starts` is
    accepted but UNUSED: the earlier design's off-band k-means multistart
    has been removed (module docstring: it added nothing once THE
    FAILURE's actual cause -- scoring against an unconverged background
    -- was fixed, and simpler is better), but the constructor's public
    signature (build interface section 4) is unchanged, so this function
    keeps the parameter for interface compatibility. Returns
    dict(fit, status, search) or None if the background itself
    degenerates."""
    pen = sm.penalty_setup(prep, w)
    bg = _converge_background(prep, w, pen, eta)
    if bg is None:
        return None
    theta_bg, ll_bg, fil_bg = bg

    W = pen.W
    wsum, mean, cov3, eligible = _core_cells(prep, w, fil_bg, Kg, seed)
    if not eligible.any():
        return None

    grown = _grow_cores(prep, w, pen, Kg, theta_bg, ll_bg, wsum, mean, cov3, eligible, W, eta)
    if grown is None:
        return None
    theta, ll_grown = grown

    theta, ll_reloc = _relocate_cores(prep, w, pen, Kg, theta, ll_grown, wsum, mean, cov3,
                                       eligible, W, eta)

    fit, status = _converge_and_finish(prep, w, pen, Kg, theta, eta)
    if fit is None:
        return None
    search_info = dict(ll_background=ll_bg, ll_grown=ll_grown, ll_relocated=ll_reloc,
                        n_cells=int(eligible.shape[0]), n_eligible=int(eligible.sum()))
    return dict(fit=fit, status=status, search=search_info)


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
