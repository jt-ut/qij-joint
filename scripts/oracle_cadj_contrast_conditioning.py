#!/usr/bin/env python3.9
"""Condition the production Vor1 GP on selected directed-CADJ contrasts.

Cloudfil draw 1 oracle feasibility experiment.  Selection uses only the
stored Vor1 survey and CADJ geometry (the already-audited deployable order).
For a selected directed child A, R is its parent's then-current residual;
the revealed observation is the analytic full-data contrast mean_A(psi) -
mean_R(psi).  These regional linear functionals condition the EXISTING GP
posterior.  Hyperparameters are held fixed and no estimator derivative is
evaluated.

The posterior covariance includes both the latent Matern term and universal-
kriging trend uncertainty.  With K contrasts, only a K x K eigensolve is new.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np
import pandas as pd
from scipy.linalg import cho_solve
from scipy.spatial.distance import cdist

from qij_joint import registry
from qij_joint.core.influence_model import (
    _basis, _matern32, fit_influence_model, psi0,
)
from qij_joint.core.ivq import kmeans_1d
from qij_joint.core.xvq import _moments_survey_rows, fit_xvq


RUN_ROOT = Path(
    "/Users/jtaylor/Dropbox/Research/QIJ_joint/runs/"
    "cloudfil_clean/cloudfil_p2_N10000"
)
SELECTION = Path("results/selective_cadj_draw1/selection.npz")
OUTPUTS = (
    "p2_x", "p2_y", "p2_log_reff", "p2_log_axis_ratio", "p2_pa",
    "p2_logit_w", "p2_log_contrast",
)


def level_share(prediction: np.ndarray, truth: np.ndarray, bins: int = 17):
    out = np.empty(truth.shape[1])
    for c in range(truth.shape[1]):
        labels, _ = kmeans_1d(prediction[:, c], bins)
        count = np.bincount(labels).astype(float)
        mean = np.bincount(labels, weights=truth[:, c]) / count
        out[c] = np.sum((count / len(truth)) * mean ** 2) / np.mean(truth[:, c] ** 2)
    return out


def prediction_r2(prediction: np.ndarray, truth: np.ndarray):
    x = prediction - prediction.mean(axis=0)
    y = truth - truth.mean(axis=0)
    xy = np.sum(x * y, axis=0)
    return xy ** 2 / (np.sum(x ** 2, axis=0) * np.sum(y ** 2, axis=0))


def build_contrasts(bmu, pair_label, pair_parent, order, Kmax):
    """Sequential child-vs-current-residual functionals on native points."""
    selected_by_parent: dict[int, set[int]] = {}
    pieces = []
    for g in np.asarray(order[:Kmax], dtype=int):
        parent = int(pair_parent[g])
        already = selected_by_parent.setdefault(parent, set())
        A = np.flatnonzero(pair_label == g)
        residual_mask = bmu == parent
        if already:
            residual_mask &= ~np.isin(pair_label, np.fromiter(already, dtype=int))
        residual_mask &= pair_label != g
        R = np.flatnonzero(residual_mask)
        if len(A) == 0 or len(R) == 0:
            raise RuntimeError(f"invalid sequential split g={g}, parent={parent}")
        pieces.append((int(g), parent, A, R))
        already.add(int(g))

    support = np.unique(np.concatenate([np.concatenate((A, R)) for _, _, A, R in pieces]))
    inverse = np.full(len(bmu), -1, dtype=int)
    inverse[support] = np.arange(len(support))
    B = np.zeros((len(support), Kmax), dtype=float)
    meta = []
    for k, (g, parent, A, R) in enumerate(pieces):
        B[inverse[A], k] = 1.0 / len(A)
        B[inverse[R], k] = -1.0 / len(R)
        meta.append((g, parent, len(A), len(R)))
    if not np.allclose(B.sum(axis=0), 0.0, atol=1e-14):
        raise RuntimeError("contrast weights do not sum to zero")
    return support, B, np.asarray(meta, dtype=int)


def stable_solve(S: np.ndarray, rhs: np.ndarray):
    """Symmetric pseudoinverse with a declared numerical eigenvalue floor."""
    S = (S + S.T) / 2.0
    value, vector = np.linalg.eigh(S)
    vmax = max(float(value[-1]), 1.0)
    floor = 1e-10 * vmax
    keep = value > floor
    solution = vector[:, keep] @ ((vector[:, keep].T @ rhs) / value[keep])
    cond = float(value[-1] / value[keep][0]) if np.any(keep) else float("inf")
    return solution, value, floor, cond


def posterior_contrast_terms(model, c, Zw, support, B):
    """Return Cov_post(f(Zw), Lf) and Cov_post(Lf,Lf), without s2.

    s2 multiplies both matrices and cancels in noiseless conditioning.
    """
    W = model.centers[c]
    ell = float(model.width[c])
    m = int(model.m[c])
    Zs = Zw[support]
    Hs = _basis(Zs, m)
    Hx = _basis(Zw, m)
    Hb = _basis(W, m)

    Ksw = _matern32(cdist(Zs, W), ell)
    Kss = _matern32(cdist(Zs, Zs), ell)
    C = B.T @ Ksw                       # K x M: Cov(Lf, f(W))
    HL = B.T @ Hs                       # K x m
    KLL = B.T @ Kss @ B                 # K x K

    AinvC = cho_solve(model.chol_A[c], C.T)  # M x K
    RL = HL - C @ model.ainv_hb[c]           # K x m
    GinvRL = cho_solve(model.g_chol[c], RL.T)  # m x K
    SLL = KLL - C @ AinvC + RL @ GinvRL

    # Query in chunks so no N x M or N x support block is retained beyond
    # the returned N x K cross-covariance itself.
    N = len(Zw)
    K = B.shape[1]
    cross = np.empty((N, K), dtype=float)
    chunk = 1024
    for start in range(0, N, chunk):
        sl = slice(start, min(start + chunk, N))
        Kxw = _matern32(cdist(Zw[sl], W), ell)
        Kxs = _matern32(cdist(Zw[sl], Zs), ell)
        Rx = Hx[sl] - Kxw @ model.ainv_hb[c]
        cross[sl] = Kxs @ B - Kxw @ AinvC + Rx @ GinvRL
    return cross, (SLL + SLL.T) / 2.0


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path,
                    default=Path("results/oracle_cadj_contrast_conditioning"))
    ap.add_argument("--budgets", default="0,25,50,100")
    args = ap.parse_args(argv)
    budgets = sorted({int(v) for v in args.budgets.split(",")})
    Kmax = max(budgets)
    args.out.mkdir(parents=True, exist_ok=True)

    t_all = time.perf_counter()
    case = registry.case("cloudfil", "p2")
    T = case.make_T()
    X = case.draw(10000, 1)
    Z = np.asarray(X, dtype=float)
    N = len(Z)
    xvq = fit_xvq(Z, 1094, 1, workers=1)

    # Recreate the original moments-survey base only to supply theta_Q to
    # the existing GP's declared-noise floor.  Stored I_proto is the actual
    # production survey response used as the GP training data.
    oracle = pd.read_parquet(RUN_ROOT / "oracle/s00001.parquet")
    theta_hat = np.array([float(oracle[f"theta_hat_{o}"].iloc[0]) for o in T.outputs])
    qij = pd.read_parquet(RUN_ROOT / "qij/s00001.parquet")
    eta_rows = float(qij["eta_full"].iloc[0])
    rows, _, omega0 = _moments_survey_rows(X, xvq.bmu, xvq.p, xvq.M_used)
    theta_Q = np.asarray(T(rows, omega0, prep=T.prepare(rows), start=theta_hat,
                           eta=eta_rows), dtype=float)
    if not np.all(np.isfinite(theta_Q)):
        raise RuntimeError("failed to recreate moments theta_Q")

    proto = pd.read_parquet(RUN_ROOT / "qij/s00001.prototypes.parquet").sort_values("j")
    I = np.column_stack([proto[f"I_{o}"].to_numpy() for o in OUTPUTS])
    model, _ = fit_influence_model(
        Z, xvq, I, theta_Q[np.asarray(T.measured, dtype=int)], float(T.eta),
        gptrend="quadratic", gpwidth="global", pool=None,
    )
    base_prediction = np.asarray(psi0(model, Z))

    truth_frame = pd.read_parquet(RUN_ROOT / "ij/s00001.psi.parquet")
    truth = np.column_stack([truth_frame[f"psi_{o}"].to_numpy() for o in OUTPUTS])

    pair_key = xvq.bmu.astype(np.int64) * xvq.M_used + xvq.bmu2.astype(np.int64)
    keys, pair_label = np.unique(pair_key, return_inverse=True)
    pair_parent = (keys // xvq.M_used).astype(int)
    selection = np.load(SELECTION)
    order = np.asarray(selection["order"], dtype=int)
    if not np.array_equal(pair_parent, selection["child_parent"]):
        raise RuntimeError("selection artifact does not match rebuilt directed cells")
    support, B, meta = build_contrasts(xvq.bmu, pair_label, pair_parent, order, Kmax)

    mean, transform = model.whitening
    Zw = (Z - mean) @ transform.T
    prediction_by_budget = {K: base_prediction.copy() for K in budgets}
    diagnostics = []
    t_condition = time.perf_counter()
    for c in range(len(OUTPUTS)):
        cross, S = posterior_contrast_terms(model, c, Zw, support, B)
        old_support = base_prediction[support, c]
        predicted_L = B.T @ old_support
        observed_L = B.T @ truth[support, c]
        residual = observed_L - predicted_L
        coord_diag = {"output": OUTPUTS[c], "contrast_residual_rms": float(np.sqrt(np.mean(residual ** 2))),
                      "budgets": {}}
        for K in budgets:
            if K == 0:
                coord_diag["budgets"][str(K)] = {"condition": None, "rank": 0,
                                                   "eigen_min": None, "eigen_max": None}
                continue
            coef, eig, floor, cond = stable_solve(S[:K, :K], residual[:K])
            prediction_by_budget[K][:, c] += cross[:, :K] @ coef
            coord_diag["budgets"][str(K)] = {
                "condition": cond, "rank": int(np.sum(eig > floor)),
                "eigen_min": float(eig[0]), "eigen_max": float(eig[-1]),
                "floor": floor,
            }
        diagnostics.append(coord_diag)
    conditioning_wall = time.perf_counter() - t_condition

    result = {
        "draw": 1, "N": N, "M_vor1": int(xvq.M_used),
        "outputs": list(OUTPUTS), "budgets": budgets,
        "gp_eta": float(T.eta), "eta_rows_for_theta_Q": eta_rows,
        "selected_support_points": int(len(support)),
        "selection_rule": "stored deployable selective_cadj_draw1 order; no analytic influence",
        "observation_rule": "analytic mean(child)-mean(current residual), revealed after selection",
        "hyperparameters": "fixed production Vor1 GP",
        "conditioning_wall_seconds": conditioning_wall,
        "total_wall_seconds": time.perf_counter() - t_all,
        "cases": {}, "diagnostics": diagnostics,
    }
    for K in budgets:
        pred = prediction_by_budget[K]
        result["cases"][str(K)] = {
            "r2": prediction_r2(pred, truth).tolist(),
            "oracle_17bin_Vbtw_over_Vij": level_share(pred, truth).tolist(),
            "prediction_min": pred.min(axis=0).tolist(),
            "prediction_max": pred.max(axis=0).tolist(),
        }
        np.savez_compressed(args.out / f"K{K}.npz", prediction=pred,
                            selected_order=order[:K], contrast_meta=meta[:K])
    (args.out / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
