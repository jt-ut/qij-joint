"""DECISIVE DIAGNOSTIC (run before any fit): at the TRUE parameters, with
one core removed at a time (P2, then P3, then P1; weights renormalised),
compare FOUR combinations for the greedy-insertion candidate score:

  cell geometry:  (x) original-space k-means  vs  (z) unsheared-space
                  k-means, z = (x, y - h_hat(x)), h_hat the trimmed-LS
                  filament seed the search already computes (shear has
                  Jacobian 1, exactly invertible, preserves counts)
  candidate score: (old) circular-area density excess (the retired
                  `_greedy_insert` rule)  vs  (new) likelihood GAIN of
                  tentatively inserting the cell's own Gaussian
                  (Verbeek, Vlassis and Kroese 2003)

Candidate Gaussians are always initialised from the cell's own rows in
ORIGINAL (x) coordinates, whichever space the cells were built in.
Read-only: does not edit shearmix.py. Run from worktree root:

    PYTHONPATH=src python3.9 scripts/shearmix_checks/search_gain_space_diag.py
"""
import warnings

import numpy as np
from scipy.cluster.vq import kmeans2

from qij_joint import datasets, shearmix as sx, shearmix_model as sm
from qij_joint.seeding import _cell_stats

N = 10000
SEED = 1
Kg = 4

X = datasets.cloudfil_G_U_P3_v1(N, SEED)
w = np.ones(N)
prep = sm.prepare(X, omega=sm.OMEGA)
pen = sm.penalty_setup(prep, w)
W = pen.W

d = np.load('src/qij_joint/data/cloudfil_G_U_P3_v1.npz', allow_pickle=True)
weights_t = d['weights'].astype(float)
means_t = d['means'].astype(float)
covs_t = d['covs'].astype(float)
fil_t = d['fil'].astype(float)
roles = ['cloud', 'P1', 'P2', 'P3']

# The data-driven trimmed-LS filament seed (search's own quantity,
# independent of which core is "removed" below -- it only sees X, w).
fil_seed = sx._filament_seed(prep, w)
_, _, b0, b1, b2 = fil_seed[0], fil_seed[1], fil_seed[2], fil_seed[3], fil_seed[4]


def h_hat(x):
    return b0 + b1 * np.sin(sm.OMEGA * x) + b2 * np.cos(sm.OMEGA * x)


Z = np.column_stack([X[:, 0], X[:, 1] - h_hat(X[:, 0])])

Mcells = sx._GREEDY_CELL_FACTOR * Kg

with warnings.catch_warnings():
    warnings.simplefilter('ignore')
    centres_x, bmu_x = kmeans2(X, Mcells, iter=sx._KMEANS_ITER, minit='++',
                                seed=np.random.default_rng(SEED))
    centres_z, bmu_z = kmeans2(Z, Mcells, iter=sx._KMEANS_ITER, minit='++',
                                seed=np.random.default_rng(SEED))

count_x, wsum_x, area_x, mean_x, cov3_x = _cell_stats(X, w, centres_x, bmu_x, Mcells)
eligible_x = count_x >= sx._MIN_CELL_COUNT

count_z, wsum_z, area_z, _mean_z, _cov3_z = _cell_stats(Z, w, centres_z, bmu_z, Mcells)
_, _, _, mean_x_of_z, cov3_x_of_z = _cell_stats(X, w, centres_z, bmu_z, Mcells)
eligible_z = count_z >= sx._MIN_CELL_COUNT

centres_x_of_z = np.column_stack([centres_z[:, 0], centres_z[:, 1] + h_hat(centres_z[:, 0])])


def _density_at(points, Kg_cur, theta, omega):
    prep_pts = sm.prepare(points, omega=omega)
    L = sm.log_joint(prep_pts, Kg_cur, theta)
    Lmax = L.max(axis=1)
    return np.exp(Lmax) * np.sum(np.exp(L - Lmax[:, None]), axis=1)


def _cov_from3(c3):
    return np.array([[c3[0], c3[1]], [c3[1], c3[2]]])


def old_excess(wsum, area, f, eligible):
    return np.where(eligible, wsum - W * area * f, -np.inf)


def new_gain(Kg_cur, theta_cur, ll_cur, eligible, wsum, mean, cov3):
    gains = np.full(eligible.shape[0], -np.inf)
    for j in np.nonzero(eligible)[0]:
        new_w = max(wsum[j] / W, sx._MIN_INSERT_SHARE / W)
        cov_j = _cov_from3(cov3[j])
        theta_trial = sx._trial_insertion_theta(Kg_cur, theta_cur, new_w, mean[j], cov_j)
        ll_trial = sx._ll_at(prep, w, Kg_cur + 1, theta_trial, pen)
        if np.isfinite(ll_trial):
            gains[j] = ll_trial - ll_cur
    return gains


def report(label, score, centres_used, j_truth, n_eligible):
    order = np.argsort(-score)
    rank = int(np.where(order == j_truth)[0][0]) + 1
    val_here = float(score[j_truth])
    top_cell = int(order[0])
    top_val = float(score[top_cell]) if top_cell != j_truth else float(score[order[1]])
    print("  [%-22s] missing-core cell idx=%d score=%.5f  rank=%d/%d; "
          "best elsewhere=%.5f" % (label, j_truth, val_here, rank, n_eligible, top_val))


print("Mcells=%d  eligible_x=%d  eligible_z=%d\n" % (Mcells, int(eligible_x.sum()), int(eligible_z.sum())))

for removed in ('P2', 'P3', 'P1'):
    j_rm = roles.index(removed)
    keep = [k for k in range(4) if k != j_rm]
    p_rm = weights_t[j_rm]
    scale = 1.0 / (1.0 - p_rm)
    pis_g = weights_t[keep] * scale
    mus = means_t[keep]
    covs = covs_t[keep]
    Kg_cur = len(keep)
    theta_minus = sm.pack(Kg_cur, pis_g, mus, covs, fil_t)
    assert sm.feasible(Kg_cur, theta_minus)
    ll_cur = sm.penalized_ll(prep, w, Kg_cur, theta_minus, pen)
    assert np.isfinite(ll_cur)

    mu_rm = means_t[j_rm]
    j_truth_x = int(np.argmin(np.linalg.norm(centres_x - mu_rm, axis=1)))
    z_rm = np.array([mu_rm[0], mu_rm[1] - h_hat(mu_rm[0])])
    j_truth_z = int(np.argmin(np.linalg.norm(centres_z - z_rm, axis=1)))

    print("=== core removed: %s (weight %.4f, ~%.0f rows) ===" % (removed, p_rm, p_rm * N))

    f_x = _density_at(centres_x, Kg_cur, theta_minus, sm.OMEGA)
    e_a = old_excess(wsum_x, area_x, f_x, eligible_x)
    report("(a) x-space / old-excess", e_a, centres_x, j_truth_x, int(eligible_x.sum()))

    g_b = new_gain(Kg_cur, theta_minus, ll_cur, eligible_x, wsum_x, mean_x, cov3_x)
    report("(b) x-space / gain     ", g_b, centres_x, j_truth_x, int(eligible_x.sum()))

    f_xz = _density_at(centres_x_of_z, Kg_cur, theta_minus, sm.OMEGA)
    e_c = old_excess(wsum_z, area_z, f_xz, eligible_z)
    report("(c) z-space / old-excess", e_c, centres_z, j_truth_z, int(eligible_z.sum()))

    g_d = new_gain(Kg_cur, theta_minus, ll_cur, eligible_z, wsum_z, mean_x_of_z, cov3_x_of_z)
    report("(d) z-space / gain     ", g_d, centres_z, j_truth_z, int(eligible_z.sum()))
    print()
