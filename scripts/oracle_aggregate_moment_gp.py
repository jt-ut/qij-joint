#!/usr/bin/env python3.9
"""Point-observation versus aggregate-moment GP, cloudfil draw 1.

Both models use the SAME stored prototype derivatives from the production
moments survey.  The control is qij_joint's existing GP, which treats I_j as
f(w_j).  The experiment treats I_j as the weighted average of f over the
cell's existing moment-survey rows.  No new estimator derivative is taken.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np
import pandas as pd
from scipy.linalg import cho_factor, cho_solve
from scipy.optimize import minimize_scalar
from scipy.sparse import csr_matrix
from scipy.spatial.distance import cdist

from qij_joint import registry
from qij_joint.core.differences import forward_step, step_parameter
from qij_joint.core.influence_model import (
    _basis, _lambda_floor, _length_scale_bounds, _matern32,
    _whitening_from, fit_influence_model, psi0,
)
from qij_joint.core.ivq import kmeans_1d
from qij_joint.core.xvq import _moments_survey_rows, fit_xvq


RUN_ROOT = Path(
    "/Users/jtaylor/Dropbox/Research/QIJ_joint/runs/"
    "cloudfil_clean/cloudfil_p2_N10000"
)
OUTPUTS = (
    "p2_x", "p2_y", "p2_log_reff", "p2_log_axis_ratio", "p2_pa",
    "p2_logit_w", "p2_log_contrast",
)


def level_share(prediction, truth, bins=17):
    out = np.empty(truth.shape[1])
    for c in range(truth.shape[1]):
        label, _ = kmeans_1d(prediction[:, c], bins)
        count = np.bincount(label).astype(float)
        mean = np.bincount(label, weights=truth[:, c]) / count
        out[c] = np.sum((count / len(truth)) * mean ** 2) / np.mean(truth[:, c] ** 2)
    return out


def metrics(prediction, truth):
    x = prediction - prediction.mean(axis=0)
    y = truth - truth.mean(axis=0)
    xy = np.sum(x * y, axis=0)
    r2 = xy ** 2 / (np.sum(x ** 2, axis=0) * np.sum(y ** 2, axis=0))
    slope = xy / np.sum(x ** 2, axis=0)
    nmse = np.sum((x - y) ** 2, axis=0) / np.sum(y ** 2, axis=0)
    return {"r2": r2.tolist(), "slope_to_truth": slope.tolist(),
            "nmse": nmse.tolist(),
            "oracle_17bin_Vbtw_over_Vij": level_share(prediction, truth).tolist()}


def aggregate_kernel(Drr, P, ell):
    Krr = _matern32(Drr, ell)
    return np.asarray(P.T @ (Krr @ P))


def fit_aggregate_gp(Zw, roww, P, Hobs, y, p, theta_Q, eta, bounds):
    """Production-style shared-width REML on aggregate observation kernels."""
    M, q = y.shape
    m = Hobs.shape[1]
    Mminus = M - m
    Qfull, _ = np.linalg.qr(Hobs, mode="complete")
    W = Qfull[:, :m]
    Q = Qfull[:, m:]
    psi_proj = [Q @ (Q.T @ y[:, c]) for c in range(q)]
    tau = 2.0 * M
    delta = forward_step(eta)
    t = np.array([step_parameter(delta, float(v)) for v in p])
    inv_t2_median = float(np.median(1.0 / t ** 2))
    noise2 = [2.0 * eta ** 2 * float(theta_Q[c]) ** 2 * inv_t2_median
              for c in range(q)]
    Drr = cdist(roww, roww)
    trace = []

    def evaluate(log_ell):
        ell = math.exp(float(log_ell))
        K = aggregate_kernel(Drr, P, ell)
        C = W.T @ K
        E = C @ W
        E[np.diag_indices_from(E)] += tau
        WC = W @ C
        Aproj = K - WC - WC.T + W @ (E @ W.T)
        eig, vec = np.linalg.eigh(Aproj)
        eig = np.maximum(eig[:Mminus], 0.0)
        vec = vec[:, :Mminus]
        per = []
        total = 0.0
        for c in range(q):
            z = vec.T @ psi_proj[c]
            lam_floor = _lambda_floor(z, eig, Mminus, noise2[c])
            lo = max(math.log(lam_floor), math.log(1e-10))

            def inner(log_lam):
                denom = eig + math.exp(float(log_lam))
                s2 = max(float(np.sum(z ** 2 / denom) / Mminus), 1e-300)
                return (Mminus / 2.0) * math.log(s2) + 0.5 * float(np.sum(np.log(denom)))

            if lo >= math.log(1e2):
                log_lam = math.log(1e2)
                nll = inner(log_lam)
            else:
                opt = minimize_scalar(inner, bounds=(lo, math.log(1e2)), method="bounded")
                log_lam, nll = float(opt.x), float(opt.fun)
            per.append({"lam": math.exp(log_lam), "lam_floor": lam_floor, "nll": nll})
            total += nll
        entry = {"ell": ell, "K": K, "per": per, "nll": total}
        trace.append(entry)
        print(json.dumps({"ell": ell, "nll": total,
                          "lambda": [v["lam"] for v in per]}), flush=True)
        return total

    lo, hi = map(math.log, bounds)
    grid = np.linspace(lo, hi, 5)
    grid_nll = [evaluate(v) for v in grid]
    best = int(np.argmin(grid_nll))
    if best == 4:
        bracket = None
    elif best == 0:
        bracket = (grid[0], grid[1])
    else:
        bracket = (grid[best - 1], grid[best + 1])
    if bracket is not None:
        minimize_scalar(evaluate, bounds=bracket, method="bounded",
                        options={"xatol": 2e-3})
    chosen = min(trace, key=lambda z: z["nll"])

    K = chosen["K"]
    pred = np.empty((len(Zw), q))
    alpha = []
    beta = []
    for c in range(q):
        lam = chosen["per"][c]["lam"]
        chol = cho_factor(K + lam * np.eye(M), lower=True)
        AinvH = cho_solve(chol, Hobs)
        AinvY = cho_solve(chol, y[:, c])
        Gchol = cho_factor(Hobs.T @ AinvH, lower=True)
        b = cho_solve(Gchol, Hobs.T @ AinvY)
        a = AinvY - AinvH @ b
        alpha.append(a)
        beta.append(b)

    alpha = np.column_stack(alpha)
    beta = np.column_stack(beta)
    Hx = _basis(Zw, m)
    chunk = 1024
    for start in range(0, len(Zw), chunk):
        sl = slice(start, min(start + chunk, len(Zw)))
        Kxr = _matern32(cdist(Zw[sl], roww), chosen["ell"])
        Kxo = np.asarray(Kxr @ P)
        pred[sl] = Hx[sl] @ beta + Kxo @ alpha
    return pred, chosen, trace


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path,
                    default=Path("results/oracle_aggregate_moment_gp"))
    args = ap.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()

    case = registry.case("cloudfil", "p2")
    T = case.make_T()
    X = case.draw(10000, 1)
    Z = np.asarray(X, dtype=float)
    N = len(Z)
    tx = time.perf_counter()
    xvq = fit_xvq(Z, 1094, 1, workers=1)
    timing_xvq = time.perf_counter() - tx
    proto = pd.read_parquet(RUN_ROOT / "qij/s00001.prototypes.parquet").sort_values("j")
    y = np.column_stack([proto[f"I_{o}"].to_numpy() for o in OUTPUTS])

    oracle = pd.read_parquet(RUN_ROOT / "oracle/s00001.parquet")
    theta_hat = np.array([float(oracle[f"theta_hat_{o}"].iloc[0]) for o in T.outputs])
    qij = pd.read_parquet(RUN_ROOT / "qij/s00001.parquet")
    eta_rows = float(qij["eta_full"].iloc[0])
    rows, row_field, omega0 = _moments_survey_rows(X, xvq.bmu, xvq.p, xvq.M_used)
    tx = time.perf_counter()
    theta_Q = np.asarray(T(rows, omega0, prep=T.prepare(rows), start=theta_hat,
                           eta=eta_rows), dtype=float)
    timing_theta_Q = time.perf_counter() - tx
    theta_measured = theta_Q[np.asarray(T.measured, dtype=int)]

    truth_df = pd.read_parquet(RUN_ROOT / "ij/s00001.psi.parquet")
    truth = np.column_stack([truth_df[f"psi_{o}"].to_numpy() for o in OUTPUTS])

    # Exact current-code control from the same I and theta_Q.
    tx = time.perf_counter()
    control_model, _ = fit_influence_model(
        Z, xvq, y, theta_measured, float(T.eta),
        gptrend="quadratic", gpwidth="global", pool=None)
    timing_control_gp = time.perf_counter() - tx
    control = np.asarray(psi0(control_model, Z))

    mean, transform = _whitening_from(Z)
    Zw = (Z - mean) @ transform.T
    roww = (rows - mean) @ transform.T
    R = len(rows)
    qweight = omega0 / (N * xvq.p[row_field])
    P = csr_matrix((qweight, (np.arange(R), row_field)),
                   shape=(R, xvq.M_used))
    if not np.allclose(np.asarray(P.sum(axis=0)).ravel(), 1.0):
        raise RuntimeError("moment quadrature weights do not sum to one per cell")
    Hrows = _basis(roww, 6)  # quadratic in d=2: 1+2+3
    Hobs = np.asarray(P.T @ Hrows)
    bounds = _length_scale_bounds(cdist((xvq.centers - mean) @ transform.T,
                                        (xvq.centers - mean) @ transform.T))
    tx = time.perf_counter()
    aggregate, chosen, trace = fit_aggregate_gp(
        Zw, roww, P, Hobs, y, xvq.p, theta_measured, float(T.eta), bounds)
    timing_aggregate_gp = time.perf_counter() - tx

    result = {
        "draw": 1, "N": N, "M": int(xvq.M_used), "moment_rows": R,
        "outputs": list(OUTPUTS), "gp_eta": float(T.eta),
        "eta_rows_for_theta_Q": eta_rows,
        "aggregate_width": chosen["ell"],
        "aggregate_lambda": [v["lam"] for v in chosen["per"]],
        "aggregate_width_evaluations": len(trace),
        "component_wall_seconds": {
            "xvq_geometry": timing_xvq,
            "one_theta_Q_fit_for_reconstruction": timing_theta_Q,
            "production_point_gp_fit_plus_full_posterior": timing_control_gp,
            "aggregate_gp_fit_plus_mean_prediction_only": timing_aggregate_gp,
        },
        "control_width": control_model.width.tolist(),
        "control_lambda": control_model.lam.tolist(),
        "control_point_observation": metrics(control, truth),
        "aggregate_moment_observation": metrics(aggregate, truth),
        "wall_seconds": time.perf_counter() - t0,
        "meaning": (
            "Same stored I_j in both cases. Aggregate case conditions on "
            "cell-average operators over existing moment rows; no new derivative fits."
        ),
    }
    (args.out / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    np.savez_compressed(args.out / "predictions.npz", control=control,
                        aggregate=aggregate, truth=truth, row_field=row_field,
                        quadrature_weight=qweight)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
