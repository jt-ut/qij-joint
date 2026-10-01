#!/usr/bin/env python3.9
"""Offline selective-directed-CADJ experiment, cloudfil draw 1.

Selection is strictly blind to analytic point influences.  It starts with the
stored, actually measured Vor1 survey values ``I_j``.  A candidate is one
directed second-order cell ``A=(bmu=j,bmu2=k)`` inside the current unpeeled
residual ``R_j``.  Its influence proxy is the neighbouring survey value
``I_k``.  In standardized output coordinates, its predicted Haar gain is

  score(A | R_j) = n_A n_R / [N (n_R-n_A)]
                   * mean_c ((I_kc-I_Rjc)/s_c)^2,

where ``s_c^2=sum_j p_j(I_jc-Ibar_c)^2``.  This is the exact between-child
gain implied by assigning ``I_k`` to A and enforcing mass balance against
the current residual proxy.  After selection, the *ranking-only* residual is
updated with I_k, hence it never observes oracle influence.

Only after a cell is purchased is its analytic cell mean revealed.  The
measured residual is then updated by mass balance against the stored parent
survey value.  Prefixes K=0,25,... therefore form a deterministic nested
one-child-versus-residual peel.

This research script changes no production code.
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


def grouped_mean(values, labels, m):
    count = np.bincount(labels, minlength=m).astype(float)
    mean = np.column_stack([
        np.bincount(labels, weights=values[:, c], minlength=m) / count
        for c in range(values.shape[1])
    ])
    return mean, count


def variance_capture(truth, labels):
    y = truth - truth.mean(axis=0, keepdims=True)
    m = int(labels.max()) + 1
    mean, count = grouped_mean(y, labels, m)
    return np.sum(count[:, None] * mean ** 2, axis=0) / np.sum(y ** 2, axis=0)


def bins_share(field, truth, n_bins=17):
    y = truth - truth.mean(axis=0, keepdims=True)
    out = np.empty(y.shape[1])
    for c in range(y.shape[1]):
        labels, _ = kmeans_1d(field[:, c], n_bins)
        mean, count = grouped_mean(y[:, [c]], labels, int(labels.max()) + 1)
        out[c] = np.sum(count * mean[:, 0] ** 2) / np.sum(y[:, c] ** 2)
    return out


def blind_peel_order(parent, second, child_count, survey, max_k, N):
    """Greedy nested order using survey values and masses only."""
    M, q = survey.shape
    parent_count = np.bincount(
        parent, weights=child_count, minlength=M
    ).astype(float)
    p = parent_count / N
    survey_bar = p @ survey
    scale2 = np.sum(p[:, None] * (survey - survey_bar) ** 2, axis=0)
    scale2 = np.where(scale2 > 0.0, scale2, 1.0)

    # Shadow residuals use neighbour survey values, never purchased truth.
    residual_n = parent_count.copy()
    residual_I = survey.copy()
    selected = np.zeros(len(child_count), dtype=bool)
    order = []
    score_at_purchase = []
    for _ in range(max_k):
        best = None
        for a in range(len(child_count)):
            if selected[a]:
                continue
            j, k = int(parent[a]), int(second[a])
            na, nr = float(child_count[a]), float(residual_n[j])
            if na >= nr:  # the last directed child remains as residual
                continue
            delta = survey[k] - residual_I[j]
            # Child-vs-residual Haar gain after enforcing the parent's mean.
            score = (na * nr / (N * (nr - na))) * float(np.mean(delta ** 2 / scale2))
            key = (score, -int(j), -int(k))
            if best is None or key > best[0]:
                best = (key, a, score)
        if best is None:
            break
        _, a, score = best
        selected[a] = True
        order.append(a)
        score_at_purchase.append(score)
        j, k = int(parent[a]), int(second[a])
        na, nr = float(child_count[a]), float(residual_n[j])
        residual_I[j] = (nr * residual_I[j] - na * survey[k]) / (nr - na)
        residual_n[j] = nr - na
    return np.asarray(order, dtype=int), np.asarray(score_at_purchase), np.sqrt(scale2)


def hybrid_prefix(
    K, order, point_child, child_parent, child_mean_x, child_mean_y,
    child_count, vor1, survey,
):
    """Build K peeled rows and one mass-balanced residual per parent."""
    M, q = survey.shape
    chosen = order[:K]
    chosen_mask = np.zeros(len(child_count), dtype=bool)
    chosen_mask[chosen] = True

    # Every selected directed child gets its own row.  Every parent retains
    # exactly one residual row containing all its unselected directed cells.
    row_of_child = np.full(len(child_count), -1, dtype=int)
    rows_x, rows_I, rows_n = [], [], []
    for a in chosen:
        row_of_child[a] = len(rows_x)
        rows_x.append(child_mean_x[a])
        rows_I.append(child_mean_y[a])       # revealed only because purchased
        rows_n.append(child_count[a])

    residual_row = np.empty(M, dtype=int)
    for j in range(M):
        kids = np.flatnonzero((child_parent == j) & ~chosen_mask)
        nr = float(child_count[kids].sum())
        assert nr > 0.0
        # Geometry is free: exact centroid of the residual directed cells.
        xr = np.sum(child_count[kids, None] * child_mean_x[kids], axis=0) / nr
        bought = np.flatnonzero((child_parent == j) & chosen_mask)
        # Measured influence is inferred only by mass balance against the
        # parent's stored actual survey response.
        numerator = np.sum(child_count[child_parent == j]) * survey[j]
        if len(bought):
            numerator = numerator - np.sum(
                child_count[bought, None] * child_mean_y[bought], axis=0
            )
        Ir = numerator / nr
        residual_row[j] = len(rows_x)
        rows_x.append(xr)
        rows_I.append(Ir)
        rows_n.append(nr)

    rows_x = np.asarray(rows_x)
    rows_I = np.asarray(rows_I)
    rows_n = np.asarray(rows_n)
    assert len(rows_n) == M + K
    assert rows_n.sum() == len(point_child)
    labels = np.empty(len(point_child), dtype=int)
    for i, a in enumerate(point_child):
        labels[i] = row_of_child[a] if chosen_mask[a] else residual_row[vor1[i]]
    field = rows_I[labels]
    return labels, field, rows_x, rows_I, rows_n


def gp_prediction(Z, labels, centers, influences, counts, theta, eta):
    xvq = SimpleNamespace(
        centers=centers, p=counts / len(Z), bmu=labels,
        conn=None, M_used=len(counts),
    )
    t0 = time.perf_counter()
    model, _ = fit_influence_model(
        Z, xvq, influences, theta, eta,
        gptrend="quadratic", gpwidth="global", pool=None,
    )
    prediction = np.asarray(psi0(model, Z))
    return prediction, model, time.perf_counter() - t0


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, default=RUN_ROOT)
    parser.add_argument("--out-dir", type=Path,
                        default=Path("results/selective_cadj_draw1"))
    parser.add_argument("--K", default="0,25,50,100,200,400")
    parser.add_argument("--skip-gp", action="store_true")
    args = parser.parse_args(argv)
    K_values = sorted(set(int(v) for v in args.K.split(",")))
    args.out_dir.mkdir(parents=True, exist_ok=True)

    case = registry.case("cloudfil", "p2")
    T = case.make_T()
    X = case.draw(10000, 1)
    Z = np.asarray(X, dtype=float)
    xvq = fit_xvq(Z, 1094, 1, workers=1)
    M, N = xvq.M_used, len(Z)

    prototypes = pd.read_parquet(args.run_root / "qij/s00001.prototypes.parquet")
    assert np.array_equal(prototypes["j"].to_numpy(), np.arange(M))
    survey = np.column_stack([prototypes[f"I_{name}"].to_numpy() for name in OUTPUTS])
    psi_frame = pd.read_parquet(args.run_root / "ij/s00001.psi.parquet")
    truth = np.column_stack([psi_frame[f"psi_{name}"].to_numpy() for name in OUTPUTS])
    oracle = pd.read_parquet(args.run_root / "oracle/s00001.parquet")
    theta = np.array([float(oracle[f"theta_hat_{name}"].iloc[0]) for name in OUTPUTS])

    # Directed pair IDs are exactly ordered (bmu,bmu2), never symmetrized.
    pair_key = xvq.bmu.astype(np.int64) * M + xvq.bmu2.astype(np.int64)
    unique_key, point_child = np.unique(pair_key, return_inverse=True)
    child_parent = unique_key // M
    child_second = unique_key % M
    C = len(unique_key)
    child_mean_x, child_count = grouped_mean(Z, point_child, C)
    child_mean_y, count_check = grouped_mean(truth, point_child, C)
    assert np.array_equal(child_count, count_check)

    order, score, survey_scale = blind_peel_order(
        child_parent, child_second, child_count, survey, max(K_values), N,
    )
    np.savez_compressed(
        args.out_dir / "selection.npz", order=order, score=score,
        child_parent=child_parent, child_second=child_second,
        child_count=child_count, survey_scale=survey_scale,
    )

    summary = {
        "draw": 1, "N": N, "M_vor1": M, "directed_cells": C,
        "outputs": list(OUTPUTS), "K_values": K_values,
        "gp_eta": float(T.eta),
        "selection_formula": (
            "n_A*n_R/[N*(n_R-n_A)] * mean_c((I_k-I_R)^2/s_c^2); "
            "ranking residual updated using I_k only"
        ),
        "cases": [],
    }
    for K in K_values:
        labels, direct, centers, influences, counts = hybrid_prefix(
            K, order, point_child, child_parent, child_mean_x, child_mean_y,
            child_count, xvq.bmu, survey,
        )
        row = {
            "K": K,
            "n_rows": len(counts),
            "raw_partition_capture": variance_capture(truth, labels).tolist(),
            "direct_17bin_Vbtw_over_Vij": bins_share(direct, truth).tolist(),
        }
        arrays = {"labels": labels, "direct_field": direct, "centers": centers,
                  "influences": influences, "counts": counts}
        if not args.skip_gp:
            pred, model, wall = gp_prediction(
                Z, labels, centers, influences, counts, theta, T.eta,
            )
            row.update({
                "gp_17bin_Vbtw_over_Vij": bins_share(pred, truth).tolist(),
                "gp_fit_predict_seconds": wall,
                "gp_width": model.width.tolist(),
                "gp_lambda": model.lam.tolist(),
            })
            arrays["gp_prediction"] = pred
        np.savez_compressed(args.out_dir / f"K{K:03d}.npz", **arrays)
        summary["cases"].append(row)
        (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        print(json.dumps(row, indent=2), flush=True)


if __name__ == "__main__":
    main()
