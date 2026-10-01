#!/usr/bin/env python3.9
"""Fixed-budget oracle-metric VQ experiment for cloudfil draw 1.

This is deliberately a stand-alone feasibility experiment, not production
QIJ code.  It compares ordinary Euclidean VQ with generalized Lloyd VQ under
the local pullback metric

    G_i = I + lambda * J_i.T @ J_i / median(trace(J.T @ J)),

where J_i is a k-nearest-neighbour local linear estimate of the gradient of
the seven *stored analytic* P2 influences (each standardized by its full-data
RMS).  The prototype budget is fixed across cases.  A successful oracle case
would establish that the existing budget can carry the estimator-relevant
geometry; it would not by itself give a deployable way to learn G.

The script imports the current QIJT implementation from ``--qijt-root`` but
only calls its dataset, estimator, and X-VQ APIs.  It does not alter or invoke
the production tree.  Results include purely geometric/influence-codebook
diagnostics and, unless ``--skip-estimator`` is used, fits P2Mixture to each
weighted centroid measure and audits its analytic influence there.

Example:

  python3.9 scripts/oracle_metric_vq_test.py --lambdas 0,0.0001,0.001,0.01 \
      --out /tmp/oracle_metric_vq_draw1
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys
import time

for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
              "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_name, "1")

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree


DEFAULT_QIJT_ROOT = pathlib.Path(
    "/Users/jtaylor/Dropbox/Software/JT_Py_Pkgs/qij_joint_wt_Tint")
DEFAULT_RUN_ROOT = pathlib.Path(
    "/Users/jtaylor/Dropbox/Research/QIJ_joint/runs/cloudfil_clean/"
    "cloudfil_p2_N10000")
P2_NAMES = ("p2_x", "p2_y", "p2_log_reff", "p2_log_axis_ratio", "p2_pa",
            "p2_logit_w", "p2_log_contrast")


def _load_package(root: pathlib.Path):
    src = str(root.resolve() / "src")
    if src not in sys.path:
        sys.path.insert(0, src)
    from qij_joint import registry
    from qij_joint.core.xvq import fit_xvq
    return registry, fit_xvq


def _stored_products(run_root: pathlib.Path, draw: int, outputs: tuple[str, ...]):
    oracle = pd.read_parquet(run_root / "oracle" / f"s{draw:05d}.parquet")
    qij = pd.read_parquet(run_root / "qij" / f"s{draw:05d}.parquet")
    psi_df = pd.read_parquet(run_root / "ij" / f"s{draw:05d}.psi.parquet")
    theta = np.array([float(oracle[f"theta_hat_{o}"].iloc[0]) for o in outputs])
    psi = np.column_stack([psi_df[f"psi_{o}"].to_numpy() for o in P2_NAMES])
    return theta, float(qij["eta_full"].iloc[0]), psi


def _local_pullback_metric(Z: np.ndarray, psi: np.ndarray, k: int,
                           ridge: float) -> tuple[np.ndarray, dict]:
    """Estimate J=d standardized-psi/d quantizer-coordinate and return J'J."""
    psi0 = psi - psi.mean(axis=0)
    psi_scale = np.sqrt(np.mean(psi0 ** 2, axis=0))
    psi_scale = np.where(psi_scale > 0, psi_scale, 1.0)
    Y = psi0 / psi_scale
    _, nn = cKDTree(Z).query(Z, k=min(k + 1, len(Z)), workers=1)
    nn = nn[:, 1:]
    n, d = Z.shape
    G = np.empty((n, d, d))
    gradients = np.empty((n, Y.shape[1], d))
    for i in range(n):
        dx = Z[nn[i]] - Z[i]
        dy = Y[nn[i]] - Y[i]
        gram = dx.T @ dx
        penalty = ridge * max(float(np.trace(gram)) / d, np.finfo(float).eps)
        # B is d x q in dy ~= dx @ B.
        B = np.linalg.solve(gram + penalty * np.eye(d), dx.T @ dy)
        gradients[i] = B.T
        G[i] = B @ B.T
    trace = np.trace(G, axis1=1, axis2=2)
    positive = trace[trace > 0]
    scale = float(np.median(positive)) if positive.size else 1.0
    return G / scale, {
        "psi_rms": psi_scale.tolist(),
        "raw_metric_trace_median": scale,
        "raw_metric_trace_quantiles": np.quantile(trace, [0, .1, .5, .9, .99, 1]).tolist(),
        "gradient_frobenius_quantiles": np.quantile(
            np.linalg.norm(gradients, axis=(1, 2)), [0, .1, .5, .9, .99, 1]).tolist(),
    }


def _centroids(X: np.ndarray, labels: np.ndarray, M: int) -> tuple[np.ndarray, np.ndarray]:
    counts = np.bincount(labels, minlength=M).astype(float)
    centers = np.column_stack([
        np.bincount(labels, weights=X[:, a], minlength=M) for a in range(X.shape[1])
    ])
    live = counts > 0
    centers[live] /= counts[live, None]
    return centers, counts


def _generalized_lloyd(Z: np.ndarray, G_oracle: np.ndarray, initial: np.ndarray,
                       lam: float, max_iter: int, batch: int) -> tuple[np.ndarray, np.ndarray, dict]:
    """Minimize sum_i (z_i-c_label)'(I+lam*G_i)(z_i-c_label)."""
    n, d = Z.shape
    centers = np.array(initial, dtype=float, copy=True)
    eye = np.eye(d)[None, :, :]
    metrics = eye + float(lam) * G_oracle
    old_labels = None
    objective = []
    for iteration in range(max_iter):
        labels = np.empty(n, dtype=np.int32)
        loss = np.empty(n)
        for lo in range(0, n, batch):
            hi = min(n, lo + batch)
            diff = Z[lo:hi, None, :] - centers[None, :, :]
            dist = np.einsum("bmd,bdk,bmk->bm", diff, metrics[lo:hi], diff,
                             optimize=True)
            labels[lo:hi] = np.argmin(dist, axis=1)
            loss[lo:hi] = dist[np.arange(hi - lo), labels[lo:hi]]
        objective.append(float(loss.sum()))
        if old_labels is not None and np.array_equal(labels, old_labels):
            break

        new = np.empty_like(centers)
        counts = np.bincount(labels, minlength=len(centers))
        dead = np.flatnonzero(counts == 0)
        for j in np.flatnonzero(counts):
            take = labels == j
            A = metrics[take].sum(axis=0)
            b = np.einsum("nij,nj->i", metrics[take], Z[take])
            new[j] = np.linalg.solve(A, b)
        # Deterministic farthest-point repair.  Usually unused because the
        # Euclidean VQ initialization has no empty receptive fields.
        if dead.size:
            order = np.argsort(-loss, kind="stable")
            used = set()
            for j, point in zip(dead, (i for i in order if i not in used)):
                new[j] = Z[point]
                used.add(int(point))
        centers, old_labels = new, labels
    return centers, labels, {
        "iterations": iteration + 1,
        "objective": objective[-1],
        "objective_path": objective,
        "converged_labels": bool(iteration + 1 < max_iter),
    }


def _weighted_center(A: np.ndarray, w: np.ndarray) -> np.ndarray:
    return A - np.average(A, axis=0, weights=w)


def _r2_slope(x: np.ndarray, y: np.ndarray, w: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x, y = _weighted_center(x, w), _weighted_center(y, w)
    cov = np.sum(w[:, None] * x * y, axis=0)
    xx = np.sum(w[:, None] * x * x, axis=0)
    yy = np.sum(w[:, None] * y * y, axis=0)
    r2 = np.divide(cov * cov, xx * yy, out=np.full_like(cov, np.nan), where=(xx * yy) > 0)
    slope = np.divide(cov, xx, out=np.full_like(cov, np.nan), where=xx > 0)
    return r2, slope


def _codebook_diagnostics(X: np.ndarray, Z: np.ndarray, psi: np.ndarray,
                          labels: np.ndarray, centers_z: np.ndarray) -> tuple[dict, np.ndarray, np.ndarray]:
    M = len(centers_z)
    centers_x, counts = _centroids(X, labels, M)
    means, _ = _centroids(psi, labels, M)
    psi0 = psi - psi.mean(axis=0)
    means0 = means - np.average(means, axis=0, weights=counts)
    total = np.mean(psi0 ** 2, axis=0)
    between = np.sum(counts[:, None] * means0 ** 2, axis=0) / len(X)
    top = np.argsort(-np.sum((psi0 / np.sqrt(total)) ** 2, axis=1), kind="stable")[:100]
    resid = Z - centers_z[labels]
    result = {
        "M_used": int(np.count_nonzero(counts)),
        "cell_size_quantiles": np.quantile(counts[counts > 0], [0, .1, .5, .9, .99, 1]).tolist(),
        "euclidean_distortion": float(np.mean(np.sum(resid ** 2, axis=1))),
        "oracle_between_share": np.divide(between, total).tolist(),
        "oracle_between_share_min": float(np.min(between / total)),
        "oracle_between_share_mean": float(np.mean(between / total)),
        "top100_unique_cells": int(np.unique(labels[top]).size),
        "top100_cell_size_quantiles": np.quantile(counts[labels[top]], [0, .25, .5, .75, 1]).tolist(),
    }
    return result, centers_x, counts


def _estimator_diagnostics(T, rows: np.ndarray, weights: np.ndarray,
                           theta_hat: np.ndarray, eta: float, psi: np.ndarray,
                           labels: np.ndarray, outputs: tuple[str, ...]) -> dict:
    theta_q = np.asarray(T(rows, weights, start=theta_hat, eta=eta), dtype=float)
    info_q = getattr(T, "last_fit_info", None)
    IF = np.asarray(T.influence(rows, weights, start=theta_q, eta=eta), dtype=float)
    info_if = getattr(T, "last_fit_info", None)
    measured = np.array([outputs.index(name) for name in P2_NAMES])
    proto = IF[:, measured]
    field_mean, _ = _centroids(psi, labels, len(rows))
    r2, slope = _r2_slope(field_mean, proto, weights)
    proto0 = _weighted_center(proto, weights)
    psi0 = psi - psi.mean(axis=0)
    variance_ratio = ((weights[:, None] * proto0 ** 2).sum(axis=0) / weights.sum()) / np.mean(psi0 ** 2, axis=0)
    # Oracle diagnostic only: the variance ratio after removing the fitted
    # scalar response error.  A deployable method would have to estimate this
    # scale from reserved full-data contrasts, not from the analytic field.
    scaled_variance_ratio = np.divide(
        variance_ratio, slope * slope, out=np.full_like(variance_ratio, np.nan),
        where=slope != 0.0)

    # Contrast agreement on a deterministic 4-nearest-neighbour graph of
    # prototype locations: a cheap surrogate for the coarse tree coefficients.
    _, nn = cKDTree(rows).query(rows, k=min(5, len(rows)), workers=1)
    a = np.repeat(np.arange(len(rows)), nn.shape[1] - 1)
    b = nn[:, 1:].reshape(-1)
    edge_true, edge_proto = field_mean[a] - field_mean[b], proto[a] - proto[b]
    edge_w = np.sqrt(weights[a] * weights[b])
    edge_r2, edge_slope = _r2_slope(edge_true, edge_proto, edge_w)

    def info_dict(info):
        if hasattr(info, "_asdict"):
            return info._asdict()
        return info if isinstance(info, dict) else str(info)

    return {
        "theta_Q_status": info_dict(info_q),
        "influence_fit_status": info_dict(info_if),
        "theta_Q_derived": theta_q[measured].tolist(),
        "theta_hat_derived": theta_hat[measured].tolist(),
        "theta_Q_minus_full": (theta_q[measured] - theta_hat[measured]).tolist(),
        "prototype_vs_field_mean_r2": r2.tolist(),
        "prototype_vs_field_mean_slope": slope.tolist(),
        "prototype_variance_over_full": variance_ratio.tolist(),
        "prototype_variance_over_full_after_oracle_scale": scaled_variance_ratio.tolist(),
        "neighbor_contrast_r2": edge_r2.tolist(),
        "neighbor_contrast_slope": edge_slope.tolist(),
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--qijt-root", type=pathlib.Path, default=DEFAULT_QIJT_ROOT)
    ap.add_argument("--run-root", type=pathlib.Path, default=DEFAULT_RUN_ROOT)
    ap.add_argument("--out", type=pathlib.Path, default=pathlib.Path("/tmp/oracle_metric_vq_draw1"))
    ap.add_argument("--draw", type=int, default=1)
    ap.add_argument("--N", type=int, default=10000)
    ap.add_argument("--M", type=int, default=1094)
    ap.add_argument("--seed", type=int, default=1,
                    help="VQ seed; defaults to the draw, matching existing validation")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--knn", type=int, default=32)
    ap.add_argument("--ridge", type=float, default=1e-4)
    ap.add_argument("--lambdas", default="0,0.0001,0.001,0.01")
    ap.add_argument("--max-iter", type=int, default=20)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--skip-estimator", action="store_true")
    args = ap.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)

    registry, fit_xvq = _load_package(args.qijt_root)
    case = registry.case("cloudfil", "p2")
    X = case.draw(args.N, args.draw)
    T0 = case.make_T()
    outputs = tuple(T0.outputs)
    theta_hat, eta, psi = _stored_products(args.run_root, args.draw, outputs)
    if len(psi) != len(X):
        raise ValueError(
            f"stored influence has {len(psi)} rows but generated X has {len(X)}; "
            "--N must match the stored run (normally 10000)")
    # Match the production quantization coordinates exactly.  Cloudfil's
    # registered transform is None, hence Z is raw X (not standardized).
    if case.vq_transform is None:
        Z = np.asarray(X, dtype=float)
    else:
        Z, _inverse = case.vq_transform(X)
        Z = np.asarray(Z, dtype=float)
        if Z.ndim == 1:
            Z = Z.reshape(-1, 1)

    t0 = time.perf_counter()
    baseline = fit_xvq(Z, args.M, args.seed, args.workers)
    fit_vq_seconds = time.perf_counter() - t0
    # Use exact means in the production quantizer coordinates as the common
    # initialization.
    initial, _ = _centroids(Z, baseline.bmu, baseline.M_used)
    t0 = time.perf_counter()
    G, metric_diag = _local_pullback_metric(Z, psi, args.knn, args.ridge)
    metric_seconds = time.perf_counter() - t0

    results = []
    saved_arrays = {"X": X, "Z": Z, "psi": psi, "metric": G}
    for lam in [float(x) for x in args.lambdas.split(",")]:
        t0 = time.perf_counter()
        if lam == 0.0:
            # Exact current-method control: do not silently perform extra
            # Lloyd iterations using this script's independent solver.
            centers_z, labels = initial, baseline.bmu.copy()
            resid = Z - centers_z[labels]
            opt = {"iterations": 0,
                   "objective": float(np.sum(resid ** 2)),
                   "objective_path": [], "converged_labels": True,
                   "production_control": True}
        else:
            centers_z, labels, opt = _generalized_lloyd(
                Z, G, initial, lam, args.max_iter, args.batch)
        lloyd_seconds = time.perf_counter() - t0
        codebook, rows, weights = _codebook_diagnostics(X, Z, psi, labels, centers_z)
        tag = ("lambda_" + format(lam, ".6g")).replace(".", "p").replace("-", "m")
        saved_arrays[f"{tag}_centers_z"] = centers_z
        saved_arrays[f"{tag}_rows_x"] = rows
        saved_arrays[f"{tag}_weights"] = weights
        saved_arrays[f"{tag}_labels"] = labels
        row = {
            "lambda": lam,
            "kind": "euclidean" if lam == 0 else "oracle_pullback",
            "lloyd_seconds": lloyd_seconds,
            "optimization": opt,
            "codebook": codebook,
        }
        if not args.skip_estimator:
            t1 = time.perf_counter()
            row["estimator"] = _estimator_diagnostics(
                case.make_T(), rows, weights, theta_hat, eta, psi, labels, outputs)
            row["estimator_seconds"] = time.perf_counter() - t1
            row["estimator_row_equivalents"] = 2.0 * len(rows) / args.N
        results.append(row)
        with open(args.out / "results.partial.json", "w") as f:
            json.dump(results, f, indent=2, default=str)

    payload = {
        "design": {
            "draw": args.draw, "N": args.N, "M_requested": args.M,
            "seed": args.seed, "knn": args.knn, "ridge": args.ridge,
            "qijt_root": str(args.qijt_root.resolve()),
            "run_root": str(args.run_root.resolve()),
            "outputs": list(P2_NAMES),
            "warning": "Oracle feasibility test: stored full-data analytic influence constructs the metric.",
        },
        "timing": {"ordinary_vq_seconds": fit_vq_seconds,
                   "metric_learning_seconds": metric_seconds},
        "metric": metric_diag,
        "results": results,
    }
    with open(args.out / "results.json", "w") as f:
        json.dump(payload, f, indent=2, default=str)
    np.savez_compressed(args.out / "codebooks.npz", **saved_arrays)
    pd.DataFrame([{
        "lambda": r["lambda"], "kind": r["kind"],
        "share_min": r["codebook"]["oracle_between_share_min"],
        "share_mean": r["codebook"]["oracle_between_share_mean"],
        "top100_unique_cells": r["codebook"]["top100_unique_cells"],
        "euclidean_distortion": r["codebook"]["euclidean_distortion"],
    } for r in results]).to_csv(args.out / "summary.csv", index=False)
    print(json.dumps(payload, indent=2, default=str))


if __name__ == "__main__":
    main()
