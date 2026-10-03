"""Validate the Monte-Carlo-excess fix (shearmix.py, `_greedy_insert` /
`_mc_expected_counts`) on ONE draw (cloudfil_G_U_P3_v1, N=10000, seed=1,
unit weights). Two cold-search configurations:

  (A) WITHOUT the previous agent's off-band greedy insertion
      (`_greedy_insert_offband`) and merge repairs (`_merge_candidates`)
      -- a local copy of `_cold_search`'s own orchestration that omits
      those two candidate sources (a temporary local variable here, not
      a new public option in shearmix.py).
  (B) WITH them (the actual `ShearMix2D.__call__`, no start).

Both are compared, by final penalized ll, against the continuation from
the truth (start = the first 30 values of
data/truth/cloudfil_u.json's 'p2' entry).

Run from worktree root: PYTHONPATH=src python3.9 scripts/shearmix_checks/search_validate.py
"""
import json
import os
import time

import numpy as np

from qij_joint import datasets, shearmix as sx, shearmix_model as sm

N = 10000
SEED = 1
Kg = 4

X = datasets.cloudfil_G_U_P3_v1(N, SEED)
w = np.ones(N)

here = os.path.dirname(os.path.abspath(__file__))
truth_path = os.path.join(here, '..', '..', 'src', 'qij_joint', 'data', 'truth', 'cloudfil_u.json')
with open(truth_path) as f:
    truth = json.load(f)
theta_truth = np.array(truth['p2']['values'][:30], dtype=float)
assert len(theta_truth) == sm.n_params(Kg) == 30

est = sx.ShearMix2D(Kg=Kg, seed=SEED)

print("=== Continuation from truth ===")
t0 = time.time()
theta_cont = est(X, w, start=theta_truth)
t1 = time.time()
fi_cont = est.last_fit_info
prep = est.prepare(X)
pen = sm.penalty_setup(prep, w)
ll_cont = sm.penalized_ll(prep, w, Kg, theta_cont, pen) if np.all(np.isfinite(theta_cont)) else float('nan')
print("status=%s  wall=%.2fs  ll=%.8f" % (fi_cont.status, t1 - t0, ll_cont))


def _report_recovered(label, theta, ll, status, wall, search_info=None):
    print("\n--- %s ---" % label)
    print("status=%s  wall=%.2fs  ll=%.8f  gap_vs_cont=%.3e"
          % (status, wall, ll, (ll - ll_cont) if np.isfinite(ll) else float('nan')))
    if search_info is not None:
        print("search:", search_info)
    if np.all(np.isfinite(theta)):
        pis_g, pi_f, mus, covs, fil = sm.unpack(Kg, theta)
        print("pis_g:", np.round(pis_g, 5), " pi_f:", round(pi_f, 5))
        print("mus:\n", np.round(mus, 4))
        print("fil (m, s2x, b0, b1, b2, s2p):", np.round(fil, 5))
    else:
        print("theta is NaN (fit failed)")


# ----------------------------------------------------------------------
# (A) Cold search WITHOUT the off-band greedy insertion / merges.
# ----------------------------------------------------------------------
def _cold_search_no_offband(prep, w, Kg, n_starts, seed, eta):
    pen = sm.penalty_setup(prep, w)
    fil = sx._filament_seed(prep, w)
    pi_f = sx._PI_F_SEED

    candidates = []
    greedy_theta = sx._greedy_insert(prep, w, Kg, fil, pi_f, eta, seed)
    if greedy_theta is not None:
        candidates.append(greedy_theta)
    candidates.extend(sx._kmeans_starts_offband(prep, w, Kg, fil, pi_f, n_starts, seed))

    if not candidates:
        return None

    screened = []
    for theta0 in candidates:
        if not sm.feasible(Kg, theta0):
            continue
        ll0 = sx._ll_at(prep, w, Kg, theta0, pen)
        if not np.isfinite(ll0):
            continue
        result = sx._em_accelerated(prep, w, pen, Kg, theta0, ll0, eta, budget=sx._SCREEN_BUDGET)
        if result is None:
            continue
        theta_s, ll_s = result[0], result[1]
        screened.append(dict(theta=theta_s, ll=ll_s))

    if not screened:
        return None

    screened.sort(key=lambda d: -d['ll'])
    survivors = screened[:sx._K_KEEP]

    results = []
    for surv in survivors:
        fit, status = sx._converge_and_finish(prep, w, pen, Kg, surv['theta'], eta)
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


print("\n=== (A) Cold search WITHOUT off-band insertion / merges ===")
t0 = time.time()
search_a = _cold_search_no_offband(prep, w, Kg, est.n_starts, SEED, est.eta)
t1 = time.time()
if search_a is None:
    print("search FAILED (no survivors)")
else:
    fit_a, status_a = search_a['fit'], search_a['status']
    pis_g, _, mus, covs, fil = sm.unpack(Kg, fit_a['theta'])
    theta_labelled_a, _, _, _, order_a = (fit_a['theta'], None, None, None, None)
    _report_recovered("(A) no-offband", fit_a['theta'], fit_a['ll'], status_a, t1 - t0, search_a['search'])

# ----------------------------------------------------------------------
# (B) Cold search WITH the off-band insertion / merges (actual production path).
# ----------------------------------------------------------------------
print("\n=== (B) Cold search WITH off-band insertion / merges (production) ===")
est_b = sx.ShearMix2D(Kg=Kg, seed=SEED)
t0 = time.time()
theta_b = est_b(X, w)
t1 = time.time()
fi_b = est_b.last_fit_info
ll_b = sm.penalized_ll(prep, w, Kg, theta_b, pen) if np.all(np.isfinite(theta_b)) else float('nan')
_report_recovered("(B) with-offband", theta_b, ll_b, fi_b.status, t1 - t0, fi_b.search)

print("\n=== Truth (for reference) ===")
pis_g_t, pi_f_t, mus_t, covs_t, fil_t = sm.unpack(Kg, theta_truth)
print("pis_g:", np.round(pis_g_t, 5), " pi_f:", round(pi_f_t, 5))
print("mus:\n", np.round(mus_t, 4))
print("fil:", np.round(fil_t, 5))
