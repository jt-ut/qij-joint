#!/usr/bin/env python3.9
"""Metric-only audit of selective directed-CADJ refinement, cloudfil draw 1.

Selection rules labelled ``deployable`` use only the stored Vor1 survey,
Vor1 masses, and the directed (BMU1, BMU2) memberships.  Analytic point
influences are used only after selection to score the resulting partition.
Oracle greedy rankings are explicitly leakage-only upper bounds.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from qij_joint import registry
from qij_joint.core.xvq import fit_xvq


RUN_ROOT = Path("/Users/jtaylor/Dropbox/Research/QIJ_joint/runs/cloudfil_clean/cloudfil_p2_N10000")
NAMES = ("p2_x", "p2_y", "p2_log_reff", "p2_log_axis_ratio", "p2_pa",
         "p2_logit_w", "p2_log_contrast")
KS = (0, 25, 50, 100, 200)


def grouped(values, labels, m):
    n = np.bincount(labels, minlength=m).astype(float)
    sums = np.column_stack([
        np.bincount(labels, weights=values[:, c], minlength=m)
        for c in range(values.shape[1])])
    return n, sums, sums / n[:, None]


def capture_for_selection(selected, parent, count, sums, parent_count,
                          parent_sums, total_ss):
    """Between/total after selected pair groups split from parent residuals."""
    selected = np.asarray(selected, dtype=int)
    residual_n = parent_count.copy()
    residual_s = parent_sums.copy()
    between = np.zeros(parent_sums.shape[1])
    if len(selected):
        residual_n -= np.bincount(parent[selected], weights=count[selected],
                                  minlength=len(parent_count))
        for c in range(parent_sums.shape[1]):
            residual_s[:, c] -= np.bincount(
                parent[selected], weights=sums[selected, c],
                minlength=len(parent_count))
        between += np.sum(count[selected, None] * (sums[selected] / count[selected, None]) ** 2,
                          axis=0)
    live = residual_n > 0
    between += np.sum(residual_s[live] ** 2 / residual_n[live, None], axis=0)
    return between / total_ss


def oracle_greedy(parent, count, sums, parent_count, parent_sums, total_ss,
                  limit, coordinate=None):
    """True marginal-gain greedy: benchmark only, never a deployable rule."""
    residual_n, residual_s = parent_count.copy(), parent_sums.copy()
    available = np.ones(len(parent), dtype=bool)
    order = []
    for _ in range(limit):
        rn = residual_n[parent]
        rs = residual_s[parent]
        left = rn - count
        valid = available & (left > 0)
        mu_r = rs / rn[:, None]
        mu_e = sums / count[:, None]
        gain = count[:, None] * rn[:, None] / np.maximum(left[:, None], 1.0) * (mu_e - mu_r) ** 2
        if coordinate is None:
            score = np.sum(gain / total_ss, axis=1)
        else:
            score = gain[:, coordinate] / total_ss[coordinate]
        score[~valid] = -np.inf
        e = int(np.argmax(score))
        if not np.isfinite(score[e]):
            break
        order.append(e)
        available[e] = False
        j = parent[e]
        residual_n[j] -= count[e]
        residual_s[j] -= sums[e]
    return np.asarray(order, dtype=int)


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path,
                   default=Path("results/selective_cadj_oracle_metric_audit.json"))
    p.add_argument("--random-reps", type=int, default=200)
    args = p.parse_args(argv)

    case = registry.case("cloudfil", "p2")
    X = case.draw(10000, 1)
    xvq = fit_xvq(X, 1094, 1, workers=1)
    proto = pd.read_parquet(RUN_ROOT / "qij" / "s00001.prototypes.parquet")
    if not np.array_equal(proto["j"].to_numpy(), np.arange(xvq.M_used)):
        raise RuntimeError("stored prototype rows are not in live-XVQ order")
    I = np.column_stack([proto[f"I_{name}"].to_numpy() for name in NAMES])
    p_mass = proto["p"].to_numpy()
    if not np.all(np.isfinite(I)) or not np.allclose(p_mass @ I, 0.0, atol=2e-12):
        raise RuntimeError("stored survey is nonfinite or not mass-centred")

    truth_df = pd.read_parquet(RUN_ROOT / "ij" / "s00001.psi.parquet")
    truth = np.column_stack([truth_df[f"psi_{name}"].to_numpy() for name in NAMES])
    total_ss = np.sum(truth ** 2, axis=0)  # exact pipeline V_ij convention; /N^2 cancels

    key = xvq.bmu.astype(np.int64) * xvq.M_used + xvq.bmu2.astype(np.int64)
    keys, pair_label = np.unique(key, return_inverse=True)
    parent = (keys // xvq.M_used).astype(int)
    neighbor = (keys % xvq.M_used).astype(int)
    pair_count, pair_sums, pair_mean = grouped(truth, pair_label, len(keys))
    parent_count, parent_sums, _ = grouped(truth, xvq.bmu, xvq.M_used)

    # Only edges that leave a nonempty residual can buy a genuine split.
    valid = pair_count < parent_count[parent]
    valid_edges = np.flatnonzero(valid)
    survey_scale2 = np.sum(p_mass[:, None] * I ** 2, axis=0)
    delta2 = (I[parent] - I[neighbor]) ** 2 / survey_scale2
    dispersion_score = np.sum(delta2, axis=1)
    haar_factor = pair_count * parent_count[parent] / np.maximum(
        parent_count[parent] - pair_count, 1.0)
    haar_score = haar_factor * dispersion_score
    dispersion_score[~valid] = -np.inf
    haar_score[~valid] = -np.inf
    mass_score = pair_count.copy()
    mass_score[~valid] = -np.inf

    rankings = {
        "neighbor_I_dispersion": np.argsort(-dispersion_score, kind="stable"),
        "mass_weighted_predicted_Haar": np.argsort(-haar_score, kind="stable"),
        "pair_mass": np.argsort(-mass_score, kind="stable"),
        "oracle_shared_greedy": oracle_greedy(
            parent, pair_count, pair_sums, parent_count, parent_sums,
            total_ss, max(KS)),
    }
    oracle_coordinate = [oracle_greedy(
        parent, pair_count, pair_sums, parent_count, parent_sums,
        total_ss, max(KS), c) for c in range(len(NAMES))]

    base = capture_for_selection([], parent, pair_count, pair_sums,
                                 parent_count, parent_sums, total_ss)
    full = capture_for_selection(valid_edges, parent, pair_count, pair_sums,
                                 parent_count, parent_sums, total_ss)
    recoverable = full - base

    def rows_for_order(order):
        rows = {}
        for K in KS:
            cap = capture_for_selection(order[:K], parent, pair_count, pair_sums,
                                        parent_count, parent_sums, total_ss)
            frac = np.divide(cap - base, recoverable, out=np.zeros_like(cap),
                             where=recoverable > 0)
            rows[str(K)] = {"capture": cap.tolist(),
                            "recoverable_fraction": frac.tolist(),
                            "recoverable_fraction_mean": float(frac.mean())}
        return rows

    methods = {name: rows_for_order(order) for name, order in rankings.items()}
    methods["oracle_per_output_greedy"] = {}
    for K in KS:
        cap_diag = np.array([
            capture_for_selection(oracle_coordinate[c][:K], parent, pair_count,
                                  pair_sums, parent_count, parent_sums,
                                  total_ss)[c] for c in range(len(NAMES))])
        frac = (cap_diag - base) / recoverable
        methods["oracle_per_output_greedy"][str(K)] = {
            "capture": cap_diag.tolist(), "recoverable_fraction": frac.tolist(),
            "recoverable_fraction_mean": float(frac.mean())}

    rng = np.random.default_rng(20260929)
    random_frac = {str(K): [] for K in KS}
    for _ in range(args.random_reps):
        order = rng.permutation(valid_edges)
        for K in KS:
            cap = capture_for_selection(order[:K], parent, pair_count, pair_sums,
                                        parent_count, parent_sums, total_ss)
            random_frac[str(K)].append((cap - base) / recoverable)
    methods["random"] = {}
    for K in KS:
        a = np.asarray(random_frac[str(K)])
        methods["random"][str(K)] = {
            "recoverable_fraction_mean_by_output": a.mean(axis=0).tolist(),
            "recoverable_fraction_mean": float(a.mean()),
            "recoverable_fraction_mean_p05_p95":
                np.quantile(a.mean(axis=1), [.05, .95]).tolist()}

    result = {
        "outputs": list(NAMES), "N": len(X), "M_vor1": xvq.M_used,
        "n_directed_pairs": len(keys), "n_splittable_pairs": int(valid.sum()),
        "survey_source": str(RUN_ROOT / "qij" / "s00001.prototypes.parquet"),
        "truth_source": str(RUN_ROOT / "ij" / "s00001.psi.parquet"),
        "survey_mass_center_residual": (p_mass @ I).tolist(),
        "base_vor1_capture": base.tolist(), "full_vor2_capture": full.tolist(),
        "recoverable_increment": recoverable.tolist(), "methods": methods,
        "information_rule": (
            "neighbor_I_dispersion and mass_weighted_predicted_Haar use only "
            "stored Vor1 I, masses, and directed CADJ membership; oracle rankings "
            "and all reported realized captures use analytic psi only for audit."),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
