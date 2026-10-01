#!/usr/bin/env python3.9
"""Standalone cloudfil draw-1 selective directed-CADJ oracle-GP study.

This is diagnostic code, not a production QIJ path.  A selected directed
Vor2 child (bmu=j, bmu2=k) becomes its own weighted centroid.  All unselected
points of Vor1 field j remain together as one residual weighted centroid.
The selected children are ranked using only the stored Vor1 survey responses.
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

RUN_ROOT = Path("/Users/jtaylor/Dropbox/Research/QIJ_joint/runs/"
                "cloudfil_clean/cloudfil_p2_N10000")
OUTPUTS = ("p2_x", "p2_y", "p2_log_reff", "p2_log_axis_ratio",
           "p2_pa", "p2_logit_w", "p2_log_contrast")


def grouped(values, labels):
    m = int(labels.max()) + 1
    count = np.bincount(labels, minlength=m).astype(float)
    means = np.column_stack([
        np.bincount(labels, weights=values[:, c], minlength=m) / count
        for c in range(values.shape[1])
    ])
    return means, count


def hybrid_labels(bmu, pair_key, selected):
    """Contiguous labels: selected pair cells plus one residual per parent."""
    selected = np.asarray(selected, dtype=np.int64)
    order = np.argsort(selected)
    sorted_key = selected[order]
    pos = np.searchsorted(sorted_key, pair_key)
    hit = (pos < len(sorted_key))
    hit[hit] &= sorted_key[pos[hit]] == pair_key[hit]
    raw = bmu.astype(np.int64).copy()
    raw[hit] = int(bmu.max()) + 1 + order[pos[hit]]
    _, labels = np.unique(raw, return_inverse=True)
    return labels


def rank_pairs(xvq, survey_I):
    """Fixed proxy: largest normalized predicted binary-split energy."""
    m, n = xvq.M_used, len(xvq.bmu)
    pair = xvq.bmu.astype(np.int64) * m + xvq.bmu2.astype(np.int64)
    keys, count = np.unique(pair, return_counts=True)
    j, k = keys // m, keys % m
    parent_n = np.bincount(xvq.bmu, minlength=m).astype(float)
    residual_n = parent_n[j] - count
    mass_factor = count * residual_n / parent_n[j] / n
    center = survey_I - np.average(survey_I, axis=0, weights=xvq.p)
    denom = np.sum(xvq.p[:, None] * center ** 2, axis=0)
    diff = survey_I[k] - survey_I[j]
    energy = mass_factor[:, None] * diff ** 2
    normalized = np.divide(energy, denom, out=np.zeros_like(energy), where=denom > 0)
    score = normalized.max(axis=1)
    useful = residual_n > 0
    idx = np.flatnonzero(useful)
    idx = idx[np.lexsort((keys[idx], -score[idx]))]
    return pair, keys[idx], score[idx]


def score_prediction(pred, truth):
    pc = pred - pred.mean(axis=0)
    yc = truth - truth.mean(axis=0)
    cov = np.sum(pc * yc, axis=0)
    r2 = cov ** 2 / (np.sum(pc ** 2, axis=0) * np.sum(yc ** 2, axis=0))
    share = np.empty(truth.shape[1])
    for c in range(truth.shape[1]):
        lab, _ = kmeans_1d(pred[:, c], 17)
        means, count = grouped(yc[:, [c]], lab)
        share[c] = np.sum(count * means[:, 0] ** 2) / np.sum(yc[:, c] ** 2)
    return r2, share


def fit_case(Z, Y, labels, theta, eta):
    centers, count = grouped(Z, labels)
    cell_y, count_y = grouped(Y, labels)
    assert np.array_equal(count, count_y)
    cell_y -= np.average(cell_y, axis=0, weights=count)
    pseudo = SimpleNamespace(centers=centers, p=count / len(Z), bmu=labels,
                             conn=None, M_used=len(count))
    t0 = time.perf_counter()
    model, _ = fit_influence_model(
        Z, pseudo, cell_y, theta, eta,
        gptrend="quadratic", gpwidth="global", pool=None)
    pred = np.asarray(psi0(model, Z))
    wall = time.perf_counter() - t0
    yc = Y - Y.mean(axis=0)
    capture = np.sum(count[:, None] * cell_y ** 2, axis=0) / np.sum(yc ** 2, axis=0)
    r2, share = score_prediction(pred, Y)
    return {"support_rows": len(count), "fit_predict_wall_seconds": wall,
            "raw_cell_capture": capture.tolist(), "gp_r2": r2.tolist(),
            "oracle_17bin_Vbtw_over_Vij": share.tolist(),
            "gp_width": model.width.tolist(), "gp_lambda": model.lam.tolist(),
            "gp_width_at_bound": model.at_bound[:, 0].tolist(),
            "gp_lambda_at_bound": model.at_bound[:, 1].tolist()}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("/tmp/hybrid_cadj_gp_draw1"))
    ap.add_argument("--budgets", default="0,25,50,100,200")
    args = ap.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)

    case = registry.case("cloudfil", "p2")
    T = case.make_T()
    X = case.draw(10000, 1)
    Z = np.asarray(X, dtype=float)
    xvq = fit_xvq(Z, 1094, 1, workers=1)
    psi_df = pd.read_parquet(RUN_ROOT / "ij/s00001.psi.parquet")
    Y = np.column_stack([psi_df[f"psi_{o}"].to_numpy() for o in OUTPUTS])
    oracle = pd.read_parquet(RUN_ROOT / "oracle/s00001.parquet")
    theta = np.array([oracle[f"theta_hat_{o}"].iloc[0] for o in OUTPUTS])
    qij = pd.read_parquet(RUN_ROOT / "qij/s00001.parquet")
    eta_full = float(qij["eta_full"].iloc[0])
    proto = pd.read_parquet(RUN_ROOT / "qij/s00001.prototypes.parquet").sort_values("j")
    assert np.array_equal(proto["j"].to_numpy(), np.arange(xvq.M_used))
    survey_I = np.column_stack([proto[f"I_{o}"].to_numpy() for o in OUTPUTS])
    pair, ranked, proxy_score = rank_pairs(xvq, survey_I)

    payload = {"outputs": list(OUTPUTS), "ranking": "max over outputs of "
               "mass-balanced split energy normalized by Vor1 survey variance",
               "eta_full": eta_full, "eta_production": float(T.eta), "cases": {}}
    for budget in [int(v) for v in args.budgets.split(",")]:
        labels = hybrid_labels(xvq.bmu, pair, ranked[:budget])
        result = fit_case(Z, Y, labels, theta, eta_full)
        result["selected_children"] = budget
        result["top_proxy_score"] = float(proxy_score[0]) if budget else None
        result["last_proxy_score"] = float(proxy_score[budget - 1]) if budget else None
        payload["cases"][f"K{budget}_eta_full"] = result
        (args.out / "summary.partial.json").write_text(json.dumps(payload, indent=2) + "\n")
        print(json.dumps({f"K{budget}_eta_full": result}, indent=2), flush=True)

    # Quantify the eta semantic at endpoints without repeating dense Vor2.
    for budget in (0, 200):
        labels = hybrid_labels(xvq.bmu, pair, ranked[:budget])
        result = fit_case(Z, Y, labels, theta, float(T.eta))
        result["selected_children"] = budget
        payload["cases"][f"K{budget}_production_eta"] = result
        print(json.dumps({f"K{budget}_production_eta": result}, indent=2), flush=True)
    (args.out / "summary.json").write_text(json.dumps(payload, indent=2) + "\n")


if __name__ == "__main__":
    main()
