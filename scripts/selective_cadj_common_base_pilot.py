#!/usr/bin/env python3.9
"""Common-base fidelity pilot for selective directed-CADJ refinement.

Cloudfil draw 1 only.  The estimator is fit once to the weighted centroids
of the nonempty ordered (BMU1, BMU2) cells.  Vor1-parent and directed-child
finite differences are then measured on that SAME row set.  Analytic
full-data influence is used only as the audit target.

No production module is modified.  The default panel contains the first 25
children chosen by the deployable selector in selective_cadj_draw1 plus 25
deterministic controls spread over the remaining ranking.
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

from qij_joint import registry
from qij_joint.core.differences import forward_step, step_parameter
from qij_joint.core.xvq import fit_xvq


RUN_ROOT = Path(
    "/Users/jtaylor/Dropbox/Research/QIJ_joint/runs/"
    "cloudfil_clean/cloudfil_p2_N10000"
)
SELECTION = Path("results/selective_cadj_draw1/selection.npz")
OUTPUTS = (
    "p2_x", "p2_y", "p2_log_reff", "p2_log_axis_ratio", "p2_pa",
    "p2_logit_w", "p2_log_contrast",
)


def grouped_mean(values: np.ndarray, labels: np.ndarray, m: int):
    count = np.bincount(labels, minlength=m).astype(float)
    sums = np.column_stack([
        np.bincount(labels, weights=values[:, c], minlength=m)
        for c in range(values.shape[1])
    ])
    return count, sums / count[:, None]


def relative_difference(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    scale = np.maximum(np.abs(a), np.abs(b))
    return np.abs(a - b) / np.where(scale > 0.0, scale, 1.0)


def perturb(omega0: np.ndarray, member: np.ndarray, p: float, t: float):
    """Production field-weight direction for weights whose sum is N."""
    return (1.0 - t) * omega0 + t * omega0 * member.astype(float) / p


def regression_metrics(measured: np.ndarray, truth: np.ndarray):
    q = truth.shape[1]
    out = {"n": int(len(truth)), "r2": [], "slope": [], "nmse": []}
    for c in range(q):
        x = measured[:, c] - measured[:, c].mean()
        y = truth[:, c] - truth[:, c].mean()
        xx = float(x @ x)
        yy = float(y @ y)
        xy = float(x @ y)
        out["r2"].append(float(xy * xy / (xx * yy)) if xx > 0 and yy > 0 else None)
        out["slope"].append(float(xy / xx) if xx > 0 else None)
        out["nmse"].append(float(np.sum((x - y) ** 2) / yy) if yy > 0 else None)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path,
                    default=Path("results/selective_cadj_common_base_pilot"))
    ap.add_argument("--top", type=int, default=25)
    ap.add_argument("--controls", type=int, default=25)
    args = ap.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)

    case = registry.case("cloudfil", "p2")
    T = case.make_T()
    X = case.draw(10000, 1)
    N = len(X)
    xvq = fit_xvq(X, 1094, 1, workers=1)

    # Directed cell identities and centroid measure.
    pair_key_point = (xvq.bmu.astype(np.int64) * xvq.M_used
                      + xvq.bmu2.astype(np.int64))
    pair_keys, pair_label = np.unique(pair_key_point, return_inverse=True)
    P = len(pair_keys)
    pair_parent = (pair_keys // xvq.M_used).astype(int)
    pair_second = (pair_keys % xvq.M_used).astype(int)
    pair_count, pair_center = grouped_mean(X, pair_label, P)
    omega0 = pair_count.copy()  # sum N: preserve the estimator's penalty scale
    parent_count = np.bincount(xvq.bmu, minlength=xvq.M_used).astype(float)

    # Stored full-data base and its production survey continuation tolerance.
    oracle = pd.read_parquet(RUN_ROOT / "oracle/s00001.parquet")
    theta_hat = np.array([float(oracle[f"theta_hat_{o}"].iloc[0])
                          for o in T.outputs])
    qij = pd.read_parquet(RUN_ROOT / "qij/s00001.parquet")
    eta_rows = float(qij["eta_full"].iloc[0])
    measured = np.asarray(T.measured, dtype=int)
    prep = T.prepare(pair_center)

    calls = 0
    failures = 0
    call_wall = []

    def evaluate(weights, start):
        nonlocal calls, failures
        t0 = time.perf_counter()
        value = np.asarray(T(pair_center, weights, prep=prep, start=start,
                             eta=eta_rows), dtype=float)
        call_wall.append(time.perf_counter() - t0)
        calls += 1
        if not np.all(np.isfinite(value)):
            failures += 1
        return value

    theta_Q = evaluate(omega0, theta_hat)
    if not np.all(np.isfinite(theta_Q)):
        raise RuntimeError("CADJ-centroid base fit failed")

    # Same reproducibility measurement that sets the production forward step.
    f_a = evaluate(omega0, theta_Q)
    f_b = evaluate(omega0, theta_Q)
    j_heavy = int(np.argmax(parent_count))
    heavy_rows = pair_parent == j_heavy
    p_heavy = parent_count[j_heavy] / N
    t_heavy = step_parameter(1e-6, p_heavy)
    f_c = evaluate(perturb(omega0, heavy_rows, p_heavy, t_heavy), theta_Q)
    eta_Q = max(eta_rows, float(np.max(np.concatenate([
        relative_difference(f_a, f_b), relative_difference(f_a, f_c)
    ]))))
    delta_f = forward_step(eta_Q)

    # Use the existing deployable ranking.  Controls are spaced through ranks
    # 100..min(1000,last), so they are deterministic and not cherry-picked by
    # analytic influence.
    sel = np.load(SELECTION)
    ranking = np.asarray(sel["order"], dtype=int)
    top = ranking[:args.top]
    if args.controls:
        lo = min(100, len(ranking) - 1)
        hi = min(1000, len(ranking) - 1)
        control_pos = np.linspace(lo, hi, args.controls, dtype=int)
        controls = ranking[control_pos]
    else:
        controls = np.empty(0, dtype=int)
    panel = np.unique(np.concatenate([top, controls]))
    parents = np.unique(pair_parent[panel])

    def derivative(member, mass):
        t = step_parameter(delta_f, float(mass))
        value = evaluate(perturb(omega0, member, float(mass), t), theta_Q)
        return (value[measured] - theta_Q[measured]) / t, value

    parent_I = {}
    parent_raw = {}
    for j in parents:
        value, raw = derivative(pair_parent == j, parent_count[j] / N)
        parent_I[int(j)] = value
        parent_raw[int(j)] = raw

    child_I = {}
    child_raw = {}
    for g in panel:
        value, raw = derivative(np.arange(P) == g, pair_count[g] / N)
        child_I[int(g)] = value
        child_raw[int(g)] = raw

    # Step-doubling on the five strongest purchased directions.
    step_ratio = {}
    for g in top[:5]:
        p = pair_count[g] / N
        t2 = step_parameter(2.0 * delta_f, p)
        raw2 = evaluate(perturb(omega0, np.arange(P) == g, p, t2), theta_Q)
        d1 = child_raw[int(g)][measured] - theta_Q[measured]
        d2 = raw2[measured] - theta_Q[measured]
        step_ratio[str(int(g))] = (d2 / d1).tolist()

    # Full-data analytic audit targets.
    psi_df = pd.read_parquet(RUN_ROOT / "ij/s00001.psi.parquet")
    psi = np.column_stack([psi_df[f"psi_{o}"].to_numpy() for o in OUTPUTS])
    _, true_pair = grouped_mean(psi, pair_label, P)
    _, true_parent = grouped_mean(psi, xvq.bmu, xvq.M_used)

    measured_parent = np.stack([parent_I[int(j)] for j in parents])
    analytic_parent = true_parent[parents]
    measured_child = np.stack([child_I[int(g)] for g in panel])
    analytic_child = true_pair[panel]

    residual_measured = []
    residual_truth = []
    residual_meta = []
    for g in panel:
        j = int(pair_parent[g])
        n_parent = parent_count[j]
        n_child = pair_count[g]
        n_resid = n_parent - n_child
        if n_resid <= 0:
            continue
        u_resid = (n_parent * parent_I[j] - n_child * child_I[int(g)]) / n_resid
        mask = (xvq.bmu == j) & (pair_label != g)
        residual_measured.append(u_resid)
        residual_truth.append(psi[mask].mean(axis=0))
        residual_meta.append((int(g), j, int(n_resid)))
    residual_measured = np.asarray(residual_measured)
    residual_truth = np.asarray(residual_truth)

    measured_contrast = measured_child - residual_measured
    analytic_contrast = analytic_child - residual_truth
    contrast_sign = np.mean(
        np.sign(measured_contrast) == np.sign(analytic_contrast), axis=0)

    # A split's exact between-cell gain is proportional to
    # n_A*n_R/n_parent * (U_A-U_R)^2.  Aggregate after normalizing each
    # output by its full-data analytic sum of squares.  This is audit-only:
    # the analytic influence still plays no role in selecting the panel.
    panel_n = pair_count[panel]
    residual_n = np.asarray([m[2] for m in residual_meta], dtype=float)
    gain_factor = panel_n * residual_n / (panel_n + residual_n)
    total_ss = np.sum(psi ** 2, axis=0)
    measured_gain = gain_factor[:, None] * measured_contrast ** 2 / total_ss
    analytic_gain = gain_factor[:, None] * analytic_contrast ** 2 / total_ss
    measured_gain_score = measured_gain.sum(axis=1)
    analytic_gain_score = analytic_gain.sum(axis=1)
    gain_corr = float(np.corrcoef(measured_gain_score, analytic_gain_score)[0, 1])
    measured_rank = np.argsort(np.argsort(measured_gain_score, kind="stable"), kind="stable")
    analytic_rank = np.argsort(np.argsort(analytic_gain_score, kind="stable"), kind="stable")
    gain_rank_corr = float(np.corrcoef(measured_rank, analytic_rank)[0, 1])

    Vij = np.mean(psi ** 2, axis=0) / N
    base_z = ((theta_Q[measured] - theta_hat[measured]) / np.sqrt(Vij)).tolist()
    result = {
        "draw": 1, "N": N, "M_vor1": int(xvq.M_used),
        "M_directed": P, "centroid_rows": P,
        "panel_top": int(args.top), "panel_controls": int(args.controls),
        "panel_children": int(len(panel)), "panel_unique_parents": int(len(parents)),
        "outputs": list(OUTPUTS), "eta_rows": eta_rows, "eta_Q": eta_Q,
        "delta_f": delta_f, "calls": calls, "failures": failures,
        "call_wall_seconds_sum": float(np.sum(call_wall)),
        "call_wall_seconds_median": float(np.median(call_wall)),
        "theta_Q_minus_theta_hat_in_IJ_se": base_z,
        "parent": regression_metrics(measured_parent, analytic_parent),
        "child": regression_metrics(measured_child, analytic_child),
        "residual": regression_metrics(residual_measured, residual_truth),
        "contrast": {
            **regression_metrics(measured_contrast, analytic_contrast),
            "sign_agreement": contrast_sign.tolist(),
            "aggregate_gain_correlation": gain_corr,
            "aggregate_gain_rank_correlation": gain_rank_corr,
        },
        "step_ratio": step_ratio,
        "information_rule": (
            "Top and control panels are fixed from stored Vor1 survey/CADJ ranking; "
            "analytic influence is loaded only after every finite difference."
        ),
    }
    args.out.joinpath("summary.json").write_text(json.dumps(result, indent=2) + "\n")
    np.savez_compressed(
        args.out / "panel.npz", panel=panel, top=top, controls=controls,
        parents=parents, pair_keys=pair_keys, pair_parent=pair_parent,
        pair_second=pair_second, pair_count=pair_count,
        measured_parent=measured_parent, analytic_parent=analytic_parent,
        measured_child=measured_child, analytic_child=analytic_child,
        residual_measured=residual_measured, residual_truth=residual_truth,
        measured_contrast=measured_contrast, analytic_contrast=analytic_contrast,
        measured_gain=measured_gain, analytic_gain=analytic_gain,
        residual_meta=np.asarray(residual_meta, dtype=int),
        theta_Q=theta_Q, theta_hat=theta_hat,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
