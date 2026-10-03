"""Diagnostic for the ShearMix2D cold-search failure on cloudfil_G_U_P3_v1
(N=10000, seed=1): P3 (weight 0.005, mean (-1.2,-1.1), tiny covariance)
is not found. Read-only: reimplements the greedy insertion with tracing
rather than editing shearmix.py. Run from worktree root:

    PYTHONPATH=src python3.9 scripts/shearmix_checks/search_diag.py
"""
import math
import warnings

import numpy as np
from scipy.cluster.vq import kmeans2

from qij_joint import datasets, shearmix as sx, shearmix_model as sm
from qij_joint.seeding import _cell_stats

N = 10000
SEED = 1
Kg = 4
P3_MU = np.array([-1.2, -1.1])
P3_WEIGHT = 0.005

X = datasets.cloudfil_G_U_P3_v1(N, SEED)
w = np.ones(N)
prep = sm.prepare(X, omega=sm.OMEGA)
pen = sm.penalty_setup(prep, w)

fil = sx._filament_seed(prep, w)
pi_f = sx._PI_F_SEED
print("filament seed:", fil, "pi_f seed:", pi_f)

near_P3 = X[np.linalg.norm(X - P3_MU, axis=1) < 0.15]
print("rows near P3 (|x-mu|<0.15):", near_P3.shape[0])

# ----------------------------------------------------------------------
# (a) Scan every candidate theta (greedy insertion result + all k-means
# off-band starts) for a component near P3's mean with small covariance.
# ----------------------------------------------------------------------
def describe_candidate(theta, Kg, label):
    pis_g, pi_f_, mus, covs, fil_ = sm.unpack(Kg, theta)
    d = np.linalg.norm(mus - P3_MU, axis=1)
    j = int(np.argmin(d))
    print("  [%s] closest-to-P3 comp %d: mu=%s dist=%.4f pi=%.5f "
          "cov_diag=%s" % (label, j, mus[j], d[j], pis_g[j], np.diag(covs[j])))
    return d[j] < 0.2 and pis_g[j] < 0.02


print("\n=== (a) Scanning starts for a P3-like component ===")
greedy_theta = sx._greedy_insert(prep, w, Kg, fil, pi_f, sm.OMEGA, sx.gmm._ETA_DEFAULT, SEED)
found_greedy = False
if greedy_theta is None:
    print("greedy insertion FAILED (returned None)")
else:
    found_greedy = describe_candidate(greedy_theta, Kg, "greedy")

kmeans_starts = sx._kmeans_starts_offband(prep, w, Kg, fil, pi_f, 20, SEED)
print("n kmeans-offband starts:", len(kmeans_starts))
found_any_kmeans = False
for i, th in enumerate(kmeans_starts):
    ok = describe_candidate(th, Kg, "kmeans-offband[%d]" % i)
    found_any_kmeans = found_any_kmeans or ok
    # also report P1 (mean (0,0)) is found, since cloud and P1 share the mean
    pis_g, pi_f_, mus, covs, fil_ = sm.unpack(Kg, th)
    d0 = np.linalg.norm(mus, axis=1)

print("\nAny start (greedy) with P3-like component:", found_greedy)
print("Any k-means-offband start with P3-like component:", found_any_kmeans)

# ----------------------------------------------------------------------
# (b) Trace the greedy insertion step by step: where it inserted, and
# the residual excess in P3's own over-segmentation cell vs the chosen
# cell at each step. Also track whether a component near (0,0) (P1,
# sharing the cloud's mean) appears.
# ----------------------------------------------------------------------
print("\n=== (b) Greedy insertion trace ===")


def traced_greedy_insert(prep, w, Kg, fil, pi_f, omega, eta, seed,
                          cell_factor=sx._GREEDY_CELL_FACTOR):
    Xd = prep.X
    Nd = Xd.shape[0]
    M = min(cell_factor * Kg, max(Kg, Nd - 1))
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        centres, bmu = kmeans2(Xd, M, iter=sx._KMEANS_ITER, minit='++',
                                seed=np.random.default_rng(seed))
    count, wsum, area, mean, cov3 = _cell_stats(Xd, w, centres, bmu, M)
    eligible = count >= sx._MIN_CELL_COUNT

    # Which over-segmentation cell is P3's (closest centre to P3_MU)?
    dist_to_P3 = np.linalg.norm(centres - P3_MU, axis=1)
    j_p3 = int(np.argmin(dist_to_P3))
    print("  P3's own cell: idx=%d centre=%s count=%d wsum=%.4f eligible=%s dist=%.4f"
          % (j_p3, centres[j_p3], count[j_p3], wsum[j_p3], eligible[j_p3], dist_to_P3[j_p3]))

    pen_ = sm.penalty_setup(prep, w)
    W = pen_.W
    xbar = (w @ Xd) / W
    d = Xd - xbar
    c0 = (d * w[:, None]).T @ d / W
    pis_g = np.array([1.0 - pi_f])
    mus = xbar[None, :].copy()
    covs = c0[None, :, :].copy()
    theta = sm.pack(1, pis_g, mus, covs, fil)
    ll0 = sx._ll_at(prep, w, 1, theta, pen_)
    result = sx._em_accelerated(prep, w, pen_, 1, theta, ll0, eta, budget=sx._INSERT_EM_STEPS)
    theta = result[0]

    for k_new in range(1, Kg):
        Kg_cur = k_new
        f = sx._mixture_density_at(centres, omega, Kg_cur, theta)
        e = np.where(eligible, wsum - W * area * f, -np.inf)
        j = int(np.argmax(e))
        print("  step %d: chosen cell idx=%d centre=%s e=%.4f  |  P3 cell e=%.4f (eligible=%s)"
              % (k_new, j, centres[j], e[j], e[j_p3], eligible[j_p3]))
        pis_g, pi_f_cur, mus, covs, fil_cur = sm.unpack(Kg_cur, theta)
        new_w = max(e[j] / W, sx._MIN_INSERT_SHARE / W)
        pis_g_new = np.concatenate([pis_g * (1.0 - new_w), [new_w]])
        mus_new = np.concatenate([mus, mean[j][None, :]], axis=0)
        cov_new = np.array([[cov3[j, 0], cov3[j, 1]], [cov3[j, 1], cov3[j, 2]]])
        covs_new = np.concatenate([covs, cov_new[None, :, :]], axis=0)
        Kg_next = k_new + 1
        theta = sm.pack(Kg_next, pis_g_new, mus_new, covs_new, fil_cur)
        ll0 = sx._ll_at(prep, w, Kg_next, theta, pen_)
        result = sx._em_accelerated(prep, w, pen_, Kg_next, theta, ll0, eta, budget=sx._INSERT_EM_STEPS)
        theta = result[0]
        pis_g2, _, mus2, covs2, _ = sm.unpack(Kg_next, theta)
        print("    post-EM(20) means:\n", mus2, "\n    post-EM(20) pis:", pis_g2)
    return theta, j_p3, eligible[j_p3], count[j_p3]


theta_tr, j_p3, p3_eligible, p3_count = traced_greedy_insert(
    prep, w, Kg, fil, pi_f, sm.OMEGA, sx.gmm._ETA_DEFAULT, SEED)

# ----------------------------------------------------------------------
# (c) Did the screen drop a P3-holding start, or did EM move it away?
# Reconstruct the full candidate list + screen ranking as _cold_search
# does, and report where the greedy start (and any kmeans start with a
# P3-like init) ranks.
# ----------------------------------------------------------------------
print("\n=== (c) Screen ranking ===")
candidates = []
labels = []
if greedy_theta is not None:
    candidates.append(greedy_theta)
    labels.append('greedy')
for i, th in enumerate(kmeans_starts):
    candidates.append(th)
    labels.append('kmeans[%d]' % i)

screened = []
for lbl, theta0 in zip(labels, candidates):
    if not sm.feasible(Kg, theta0):
        screened.append((lbl, None, float('nan')))
        continue
    ll0 = sx._ll_at(prep, w, Kg, theta0, pen)
    if not np.isfinite(ll0):
        screened.append((lbl, None, float('nan')))
        continue
    result = sx._em_accelerated(prep, w, pen, Kg, theta0, ll0, sx.gmm._ETA_DEFAULT, budget=sx._SCREEN_BUDGET)
    if result is None:
        screened.append((lbl, None, float('nan')))
        continue
    theta_s, ll_s = result[0], result[1]
    screened.append((lbl, theta_s, ll_s))

ranked = sorted([s for s in screened if s[1] is not None], key=lambda t: -t[2])
print("Screen ranking (top 6):")
for lbl, theta_s, ll_s in ranked[:6]:
    pis_g, _, mus, covs, _ = sm.unpack(Kg, theta_s)
    d = np.linalg.norm(mus - P3_MU, axis=1)
    j = int(np.argmin(d))
    print("  %-14s ll=%.6f  closest-to-P3: mu=%s dist=%.4f pi=%.5f"
          % (lbl, ll_s, mus[j], d[j], pis_g[j]))

greedy_rank = [i for i, (lbl, _, _) in enumerate(ranked) if lbl == 'greedy']
print("greedy start rank (0-indexed) among screened:", greedy_rank, "of", len(ranked))
print("_K_KEEP =", sx._K_KEEP)

# ----------------------------------------------------------------------
# (d) Over-segmentation cell resolution vs P3's own size.
# ----------------------------------------------------------------------
print("\n=== (d) Cell resolution vs P3 size ===")
M = sx._GREEDY_CELL_FACTOR * Kg
print("M = cell_factor*Kg =", M, " N =", N, " mean cell pop ~ N/M =", N / M)
print("P3: weight %.4f -> expected rows ~ %.1f, ellipse ~0.05 x 0.025 pc" % (P3_WEIGHT, P3_WEIGHT * N))
print("P3's own over-segmentation cell: count=%d eligible(>=%d)=%s"
      % (p3_count, sx._MIN_CELL_COUNT, p3_eligible))
