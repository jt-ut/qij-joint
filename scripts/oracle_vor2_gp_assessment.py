#!/usr/bin/env python3.9
"""Cloudfil draw-1 oracle Vor1 versus directed-Vor2 GP assessment.

This is a read-only, standalone experiment.  The directed second-order
membership is exactly the ordered pair ``(xvq.bmu, xvq.bmu2)`` produced by
the package quantizer; pair order is intentionally not sorted.  For each
partition, the existing QIJ influence model is trained on the analytic
cell-mean P2 influences located at arithmetic cell centroids.  Its posterior
mean is then queried at the original points.

The existing cloudfil setting is retained: quadratic mean trend, global
Matérn-3/2 width, and the stored full-data eta.  No estimator evaluation is
performed and no production module is modified.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from qij_joint import registry
from qij_joint.core.influence_model import fit_influence_model, psi0
from qij_joint.core.ivq import kmeans_1d
from qij_joint.core.xvq import fit_xvq


RUN_ROOT = Path(
    "/Users/jtaylor/Dropbox/Research/QIJ_joint/runs/"
    "cloudfil_clean/cloudfil_p2_N10000"
)
OUTPUTS = (
    "p2_x", "p2_y", "p2_log_reff", "p2_log_axis_ratio", "p2_pa",
    "p2_logit_w", "p2_log_contrast",
)


def grouped_means(values: np.ndarray, labels: np.ndarray, m: int):
    count = np.bincount(labels, minlength=m).astype(float)
    means = np.column_stack([
        np.bincount(labels, weights=values[:, c], minlength=m) / count
        for c in range(values.shape[1])
    ])
    return means, count


def cell_design(Z: np.ndarray, Y: np.ndarray, labels: np.ndarray):
    m = int(labels.max()) + 1
    centers, count = grouped_means(Z, labels, m)
    influence, count_y = grouped_means(Y, labels, m)
    assert np.array_equal(count, count_y)
    pseudo_xvq = SimpleNamespace(
        centers=centers,
        p=count / len(Z),
        # global-width fitting does not inspect these, but including them
        # makes the intentionally minimal stand-in explicit.
        conn=None,
        bmu=labels,
        M_used=m,
    )
    return pseudo_xvq, influence, count


def variance_capture(Y: np.ndarray, labels: np.ndarray) -> np.ndarray:
    Yc = Y - Y.mean(axis=0, keepdims=True)
    means, count = grouped_means(Yc, labels, int(labels.max()) + 1)
    return np.sum(count[:, None] * means ** 2, axis=0) / np.sum(Yc ** 2, axis=0)


def levelset_share(prediction: np.ndarray, truth: np.ndarray, bins: int = 17):
    q = truth.shape[1]
    share = np.empty(q)
    used = np.empty(q, dtype=int)
    truth = truth - truth.mean(axis=0, keepdims=True)
    for c in range(q):
        labels, _ = kmeans_1d(prediction[:, c], bins)
        used[c] = int(labels.max()) + 1
        means, count = grouped_means(truth[:, [c]], labels, used[c])
        share[c] = np.sum(count * means[:, 0] ** 2) / np.sum(truth[:, c] ** 2)
    return share, used


def prediction_metrics(prediction: np.ndarray, truth: np.ndarray):
    q = truth.shape[1]
    r2 = np.empty(q)
    slope = np.empty(q)
    nmse = np.empty(q)
    scaled_nmse = np.empty(q)
    for c in range(q):
        x = prediction[:, c] - prediction[:, c].mean()
        y = truth[:, c] - truth[:, c].mean()
        slope[c] = np.dot(x, y) / np.dot(x, x)
        r2[c] = np.corrcoef(x, y)[0, 1] ** 2
        denom = np.dot(y, y)
        nmse[c] = np.dot(x - y, x - y) / denom
        residual = slope[c] * x - y
        scaled_nmse[c] = np.dot(residual, residual) / denom
    return r2, slope, nmse, scaled_nmse


def fit_one(name, Z, Y, labels, theta, eta):
    xvq, cell_y, count = cell_design(Z, Y, labels)
    t0 = time.perf_counter()
    model, _ = fit_influence_model(
        Z, xvq, cell_y, theta, eta,
        gptrend="quadratic", gpwidth="global", pool=None,
    )
    prediction = np.asarray(psi0(model, Z))
    wall = time.perf_counter() - t0
    raw_capture = variance_capture(Y, labels)
    r2, slope, nmse, scaled_nmse = prediction_metrics(prediction, Y)
    share17, bins_used = levelset_share(prediction, Y)
    return {
        "name": name,
        "n_cells": int(len(count)),
        "cell_size_min": int(count.min()),
        "cell_size_median": float(np.median(count)),
        "cell_size_max": int(count.max()),
        "fit_predict_wall_seconds": wall,
        "raw_cell_capture": raw_capture.tolist(),
        "gp_r2": r2.tolist(),
        "gp_scale_slope": slope.tolist(),
        "gp_nmse": nmse.tolist(),
        "gp_scale_adjusted_nmse": scaled_nmse.tolist(),
        "oracle_17bin_Vbtw_over_Vij": share17.tolist(),
        "bins_used": bins_used.tolist(),
        "gp_width": model.width.tolist(),
        "gp_lambda": model.lam.tolist(),
        "gp_width_at_bound": model.at_bound[:, 0].tolist(),
        "gp_lambda_at_bound": model.at_bound[:, 1].tolist(),
    }, prediction, cell_y, xvq.centers, count


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, default=RUN_ROOT)
    parser.add_argument("--out-dir", type=Path,
                        default=Path("results/oracle_vor2_gp_draw1"))
    parser.add_argument("--draw", type=int, default=1)
    parser.add_argument("--M", type=int, default=1094)
    parser.add_argument("--only", choices=("vor1", "vor2", "both"), default="both")
    args = parser.parse_args(argv)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    case = registry.case("cloudfil", "p2")
    T = case.make_T()
    X = case.draw(10000, args.draw)
    Z = np.asarray(X, dtype=float)
    xvq = fit_xvq(Z, args.M, args.draw, workers=1)

    psi_frame = pd.read_parquet(
        args.run_root / "ij" / f"s{args.draw:05d}.psi.parquet"
    )
    Y = np.column_stack([psi_frame[f"psi_{name}"].to_numpy() for name in OUTPUTS])
    oracle = pd.read_parquet(
        args.run_root / "oracle" / f"s{args.draw:05d}.parquet"
    )
    theta = np.array([float(oracle[f"theta_hat_{name}"].iloc[0]) for name in OUTPUTS])
    qij = pd.read_parquet(
        args.run_root / "qij" / f"s{args.draw:05d}.parquet"
    )
    eta = float(qij["eta_full"].iloc[0])

    pair_key = xvq.bmu.astype(np.int64) * xvq.M_used + xvq.bmu2.astype(np.int64)
    _, directed_vor2 = np.unique(pair_key, return_inverse=True)
    partitions = {"vor1": xvq.bmu, "directed_vor2": directed_vor2}
    selected = partitions.items() if args.only == "both" else [
        ("directed_vor2" if args.only == "vor2" else "vor1",
         partitions["directed_vor2" if args.only == "vor2" else "vor1"])
    ]

    summary = {
        "draw": args.draw,
        "N": len(X),
        "M_requested": args.M,
        "M_vor1": int(xvq.M_used),
        "n_directed_vor2": int(directed_vor2.max()) + 1,
        "cadj_nnz": int(xvq.conn.nnz),
        "outputs": list(OUTPUTS),
        "gptrend": "quadratic",
        "gpwidth": "global",
        "eta": eta,
        "cases": {},
    }
    summary_path = args.out_dir / "summary.json"
    if summary_path.exists():
        previous = json.loads(summary_path.read_text())
        invariant = ("draw", "N", "M_requested", "M_vor1",
                     "n_directed_vor2", "cadj_nnz", "outputs",
                     "gptrend", "gpwidth", "eta")
        if all(previous.get(k) == summary.get(k) for k in invariant):
            summary["cases"].update(previous.get("cases", {}))
    for name, labels in selected:
        result, prediction, cell_y, centers, count = fit_one(
            name, Z, Y, labels, theta, eta
        )
        summary["cases"][name] = result
        np.savez_compressed(
            args.out_dir / f"{name}.npz", labels=labels, prediction=prediction,
            cell_influence=cell_y, centers=centers, counts=count,
        )
        # Checkpoint after each potentially expensive exact-GP fit.
        summary_path.write_text(
            json.dumps(summary, indent=2) + "\n"
        )
        print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
