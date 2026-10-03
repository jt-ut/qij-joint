"""REVIEWER check (not part of the build): direct-perturbation influence
test for ShearMix2D on cloudfil_G_U_P3_v1, ONE draw (seed=1, N=10000),
per spec/QIJ_shearmix_interface.md's review task 2.

Convention (method_notes.md section 5, "The influence's extra term" and
gmm.GMM2D.influence/fit_and_influence): psi_i = N * dT/dw_i (the influence
row IS the normalized-score-scale sensitivity; A^-1 psi_i gives it, A the
penalized observed information / W). With unit weights here W = N, so
dtheta/dw_i ~= IF_i / N exactly (no ambiguity between the two
conventions at unit weights).

Run: PYTHONPATH=src python3.9 scripts/shearmix_checks/review_influence.py
"""
import os
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.join(_HERE, '..', '..', 'src')
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

import json  # noqa: E402

from qij_joint import datasets  # noqa: E402
from qij_joint.shearmix import ShearMix2D  # noqa: E402
from qij_joint import shearmix_model as sm  # noqa: E402

_TRUTH = os.path.join(_SRC, 'qij_joint', 'data', 'truth', 'cloudfil_u.json')


def main():
    import sys as _sys
    eta_override = float(_sys.argv[1]) if len(_sys.argv) > 1 else None
    N = 10000
    seed = 1
    X = datasets.cloudfil_G_U_P3_v1(N, seed)
    w = np.ones(N)

    with open(_TRUTH) as f:
        truth = json.load(f)
    truth_raw = np.array(truth['p2']['values'][:30], dtype=float)
    Kg = 4

    est = ShearMix2D(Kg=Kg, seed=0)
    prep = est.prepare(X)
    print('eta_override =', eta_override, '(None -> default)')

    t0 = time.time()
    theta_hat, IF = est.fit_and_influence(X, w, prep=prep, start=truth_raw, eta=eta_override)
    t1 = time.time()
    print('fit_and_influence (base) wall time: %.2f s' % (t1 - t0))
    print('last_fit_info status:', est.last_fit_info.status,
          'resid:', est.last_fit_info.score)
    assert np.all(np.isfinite(theta_hat)), 'base fit did not converge'
    assert np.all(np.isfinite(IF)), 'base influence is NaN'

    pis_g, pi_f, mus, covs, fil = sm.unpack(Kg, theta_hat)
    print('theta_hat pis_g:', pis_g, 'pi_f:', pi_f)
    print('theta_hat mus:', mus)
    pen_diag = sm.penalty_setup(prep, w)
    A_diag = sm.info(prep, w, Kg, theta_hat, pen_diag)
    print('cond(A) at theta_hat:', np.linalg.cond(A_diag), 'cond_max:', est.cond_max)

    # ------------------------------------------------------------------
    # Row selection.
    # ------------------------------------------------------------------
    prep_t = est.prepare(X)
    pen_t = sm.penalty_setup(prep_t, w)
    R, logf = sm.e_step(prep_t, Kg, theta_hat)
    # P2 rows: distance to (1.32, 0.55) < 0.05 (task's own criterion).
    d_p2 = np.hypot(X[:, 0] - 1.32, X[:, 1] - 0.55)
    p2_candidates = np.where(d_p2 < 0.05)[0]
    rows_p2 = list(p2_candidates[np.argsort(d_p2[p2_candidates])[:2]])

    # Filament rows: high filament responsibility (column Kg) AND far
    # from every Gaussian mean (generic filament points, not the P2
    # bead sitting on the curve).
    means_all = mus  # (Kg,2)
    dist_to_any_mean = np.min(
        [np.hypot(X[:, 0] - means_all[k, 0], X[:, 1] - means_all[k, 1]) for k in range(Kg)],
        axis=0)
    print('R[:,Kg] (filament resp.) stats: max=%.4f, n>0.99=%d, n>0.9=%d, n>0.5=%d' %
          (R[:, Kg].max(), (R[:, Kg] > 0.99).sum(), (R[:, Kg] > 0.9).sum(),
           (R[:, Kg] > 0.5).sum()))
    print('dist_to_any_mean stats: max=%.3f, n>0.3=%d' %
          (dist_to_any_mean.max(), (dist_to_any_mean > 0.3).sum()))
    fil_thresh = 0.99
    while True:
        fil_mask = (R[:, Kg] > fil_thresh) & (dist_to_any_mean > 0.3)
        fil_candidates = np.where(fil_mask)[0]
        if len(fil_candidates) >= 2 or fil_thresh < 0.3:
            break
        fil_thresh -= 0.1
    assert len(fil_candidates) >= 2, 'could not find 2 filament rows'
    # two well-separated filament rows (by x) for variety
    order = np.argsort(X[fil_candidates, 0])
    rows_fil = [int(fil_candidates[order[len(order) // 4]]),
                int(fil_candidates[order[3 * len(order) // 4]])]

    # Core-1 (P1 embedded) row: index 1 in file order -> which theta_hat
    # label does P1 end up at? theta_hat is labelled against truth_raw's
    # own Gaussians (continuation), so label order should match file
    # order (cloud=0, P1=1, P2=2, P3=3) to first approximation; identify
    # robustly by highest responsibility to the component nearest truth
    # P1 mean (0,0) with small covariance (truth S2_11 ~ 0.00228).
    from qij_joint.cloudfil_u import _true_subject
    mu_p1_true, Sigma_p1_true = _true_subject('P1 embedded')
    # Among Kg=4 fitted components, the one nearest truth P1 by mean+cov
    # (Bhattacharyya-like crude match: smallest ||mu_k - mu_p1_true|| +
    # ||cov_k - Sigma_p1_true||).
    best_k, best_score = None, np.inf
    for k in range(Kg):
        score = (np.hypot(*(means_all[k] - mu_p1_true))
                 + np.linalg.norm(covs[k] - Sigma_p1_true))
        if score < best_score:
            best_score, best_k = score, k
    k_p1 = best_k
    row_p1 = int(np.argmax(R[:, k_p1]))

    # Cloud row far from everything: high cloud responsibility (the
    # component nearest truth mean (0,0) with LARGE covariance), AND
    # far (> 1.5) from every other mean and off the filament band.
    mu_cloud_true, Sigma_cloud_true = _true_subject('cloud')
    best_k, best_score = None, np.inf
    for k in range(Kg):
        if k == k_p1:
            continue
        score = (np.hypot(*(means_all[k] - mu_cloud_true))
                 + abs(np.trace(covs[k]) - np.trace(Sigma_cloud_true)))
        if score < best_score:
            best_score, best_k = score, k
    k_cloud = best_k
    far_mask = (dist_to_any_mean > 1.5) & (R[:, k_cloud] > 0.9)
    cloud_candidates = np.where(far_mask)[0]
    # pick the single farthest such row
    row_cloud = int(cloud_candidates[np.argmax(dist_to_any_mean[cloud_candidates])])

    rows = dict(fil1=rows_fil[0], fil2=rows_fil[1],
                p2_1=rows_p2[0], p2_2=rows_p2[1],
                p1=row_p1, cloud=row_cloud)
    print('\nSelected rows:')
    for name, i in rows.items():
        print('  %-6s row=%6d  x=%.4f y=%.4f  R=%s' %
              (name, i, X[i, 0], X[i, 1], np.round(R[i], 4)))

    # ------------------------------------------------------------------
    # Derived-output wrapper (P2ShearMixture) for the 7-output check.
    # ------------------------------------------------------------------
    from qij_joint.cloudfil_u import P2ShearMixture
    p2mix = P2ShearMixture()
    theta37_hat = p2mix(X, w, prep=prep, start=truth_raw, eta=eta_override)
    IF37 = p2mix.influence(X, w, prep=prep, start=truth_raw, eta=eta_override)
    assert np.all(np.isfinite(theta37_hat)) and np.all(np.isfinite(IF37))
    outs37 = p2mix.outputs
    print('\nP2ShearMixture derived outputs at theta_hat:', theta37_hat[30:])

    outs30 = sm.make_outputs(Kg)
    blocks = {
        'pi': list(range(Kg)),
        'mu': list(range(Kg, 3 * Kg)),
        'S': list(range(3 * Kg, 6 * Kg)),
        'fil': list(range(6 * Kg, 6 * Kg + 6)),
    }

    # ------------------------------------------------------------------
    # FD vs IF, per row, per h.
    # ------------------------------------------------------------------
    results = {}
    for name, i in rows.items():
        row_result = {}
        for h in (1e-3, 1e-4):
            w2 = w.copy()
            w2[i] += h
            theta_pert = est(X, w2, prep=prep, start=theta_hat, eta=eta_override)
            if not np.all(np.isfinite(theta_pert)):
                row_result[h] = None
                continue
            FD = (theta_pert - theta_hat) / h
            IF_pred = IF[i] / N
            # Per-block max relative error, with a NOISE FLOOR on the
            # denominator: a coordinate where both FD and IF/N are far
            # below the parameter's own scale (<1e-6 absolute, i.e. a
            # row this far from a block has essentially zero true
            # sensitivity there) is excluded from the relative-error
            # max -- otherwise two tiny, noise-level numbers blow up
            # the ratio with no bearing on whether psi/A are correct
            # (confirmed by direct inspection: every coordinate with
            # |FD| or |IF/N| >= 1e-6 agrees to 3-4 significant figures;
            # the "outliers" below that floor are this artifact, not a
            # formula error).
            NOISE_FLOOR = 1e-6
            block_err = {}
            block_err_floored = {}
            for bname, idx in blocks.items():
                num = np.abs(FD[idx] - IF_pred[idx])
                den = np.maximum(np.abs(FD[idx]), np.abs(IF_pred[idx]))
                mask = den >= NOISE_FLOOR
                den_safe = np.where(den < 1e-12, 1.0, den)
                block_err[bname] = float(np.max(num / den_safe))
                block_err_floored[bname] = (float(np.max((num / den_safe)[mask]))
                                             if mask.any() else float('nan'))
            row_result[h] = dict(block_err=block_err, block_err_floored=block_err_floored,
                                  FD=FD, IF_pred=IF_pred)

            # derived outputs (37-vector) via P2ShearMixture
            theta37_pert = p2mix(X, w2, prep=prep, start=theta37_hat, eta=eta_override)
            if np.all(np.isfinite(theta37_pert)):
                FD37 = (theta37_pert - theta37_hat) / h
                IF37_pred = IF37[i] / N
                num = np.abs(FD37[30:] - IF37_pred[30:])
                den = np.maximum(np.abs(FD37[30:]), np.abs(IF37_pred[30:]))
                mask = den >= NOISE_FLOOR
                den_safe = np.where(den < 1e-12, 1.0, den)
                rel = num / den_safe
                row_result[h]['derived_max_rel_err'] = float(np.max(rel))
                row_result[h]['derived_max_rel_err_floored'] = (
                    float(np.max(rel[mask])) if mask.any() else float('nan'))
                row_result[h]['derived_err_per_output'] = dict(
                    zip(outs37[30:], rel.tolist()))
                row_result[h]['derived_abs_per_output'] = dict(
                    zip(outs37[30:], np.maximum(np.abs(FD37[30:]), np.abs(IF37_pred[30:])).tolist()))
            else:
                row_result[h]['derived_max_rel_err'] = None
                row_result[h]['derived_max_rel_err_floored'] = None
        results[name] = row_result

    print('\n==== FD vs IF/N: max relative error per block ====')
    print('(raw = unfloored max rel err over ALL coords; floored = same,'
          ' excluding coords where both |FD| and |IF/N| < %.0e -- see'
          ' module docstring note on the noise-floor artifact)' % NOISE_FLOOR)
    for name, i in rows.items():
        print('\nRow %s (idx %d):' % (name, i))
        for h in (1e-3, 1e-4):
            r = results[name][h]
            if r is None:
                print('  h=%.0e: perturbed fit FAILED (NaN)' % h)
                continue
            print('  h=%.0e RAW    : pi=%.3e mu=%.3e S=%.3e fil=%.3e | derived=%.3e' %
                  (h, r['block_err']['pi'], r['block_err']['mu'],
                   r['block_err']['S'], r['block_err']['fil'],
                   r['derived_max_rel_err'] if r['derived_max_rel_err'] is not None else float('nan')))
            fe = r['block_err_floored']
            de = r.get('derived_max_rel_err_floored')
            print('  h=%.0e FLOORED: pi=%.3e mu=%.3e S=%.3e fil=%.3e | derived=%.3e' %
                  (h, fe['pi'], fe['mu'], fe['S'], fe['fil'],
                   de if de is not None else float('nan')))

    print('\n==== Derived-output per-output relative error (h=1e-4) ====')
    for name, i in rows.items():
        r = results[name][1e-4]
        if r is None or r.get('derived_err_per_output') is None:
            continue
        print('Row %s:' % name)
        for oname, err in r['derived_err_per_output'].items():
            print('  %-18s %.3e' % (oname, err))


if __name__ == '__main__':
    main()
