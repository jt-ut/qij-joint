#!/usr/bin/env python3.9
"""End-to-end cloudfil draw-1 ablation for the aggregate-moment GP.

The production QIJ algorithm is left unchanged except for the field handed
from stage 1 to IVQ: the ordinary point-observation GP is still fitted (and
still supplies posterior uncertainty), while its posterior mean is replaced
by the aggregate-observation posterior mean from
``oracle_aggregate_moment_gp``.  Thus this test isolates whether the improved
moment-aware mean gives IVQ and the existing refinement a better start.
"""
from __future__ import annotations

import importlib
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.spatial.distance import cdist

from oracle_aggregate_moment_gp import fit_aggregate_gp
from qij_joint import QIJ, registry
from qij_joint.core.influence_model import (
    _basis, _length_scale_bounds, _whitening_from,
)
from qij_joint.core.xvq import _moments_survey_rows
from qij_joint.parallel import Pool


RUN_ROOT = Path(
    "/Users/jtaylor/Dropbox/Research/QIJ_joint/runs/"
    "cloudfil_clean/cloudfil_p2_N10000"
)
OUT = Path("results/end_to_end_moment_gp_draw1")
OUTPUTS = (
    "p2_x", "p2_y", "p2_log_reff", "p2_log_axis_ratio", "p2_pa",
    "p2_logit_w", "p2_log_contrast",
)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    qij_module = importlib.import_module("qij_joint.qij")
    original_fit = qij_module.fit_influence_model
    original_psi0 = qij_module._psi0
    captured = {}

    def aggregate_fit(Z, xvq, I_proto, theta_Q, eta, gptrend="affine",
                      gpwidth="global", pool=None):
        if gptrend != "quadratic" or gpwidth != "global":
            raise ValueError("this ablation is pinned to quadratic/global")

        # Preserve the production model in full.  In particular, this model
        # supplies uncertainty() and bin_posterior_variance() downstream.
        model, busy = original_fit(
            Z, xvq, I_proto, theta_Q, eta, gptrend=gptrend,
            gpwidth=gpwidth, pool=pool,
        )

        rows, row_field, omega0 = _moments_survey_rows(
            Z, xvq.bmu, xvq.p, xvq.M_used,
        )
        mean, transform = _whitening_from(Z)
        Zw = (Z - mean) @ transform.T
        roww = (rows - mean) @ transform.T
        qweight = omega0 / (len(Z) * xvq.p[row_field])
        P = csr_matrix(
            (qweight, (np.arange(len(rows)), row_field)),
            shape=(len(rows), xvq.M_used),
        )
        Hobs = np.asarray(P.T @ _basis(roww, 6))
        centerw = (xvq.centers - mean) @ transform.T
        bounds = _length_scale_bounds(cdist(centerw, centerw))
        pred, chosen, trace = fit_aggregate_gp(
            Zw, roww, P, Hobs, np.asarray(I_proto), xvq.p,
            np.asarray(theta_Q), float(eta), bounds,
        )
        model._aggregate_moment_prediction = pred
        model.offset[:] = pred.mean(axis=0)
        captured.update(
            prediction=pred,
            ell=float(chosen["ell"]),
            lam=np.array([v["lam"] for v in chosen["per"]]),
            n_width_evals=len(trace),
            moment_rows=len(rows),
        )
        return model, busy

    def aggregate_psi0(model, Z):
        pred = getattr(model, "_aggregate_moment_prediction", None)
        return pred if pred is not None else original_psi0(model, Z)

    qij_module.fit_influence_model = aggregate_fit
    qij_module._psi0 = aggregate_psi0

    case = registry.case("cloudfil", "p2")
    T = case.make_T()
    X = case.draw(10000, 1)
    pool = Pool(args.workers, T=T) if args.workers > 1 else None
    t0 = time.perf_counter()
    try:
        result = QIJ(
            eps=0.01, seed=1, M_X=1094,
            gptrend="quadratic", gpwidth="global",
            ivqbins="marginal", survey="moments",
            quantized_start="full-data", refine_schedule="rounds",
        ).fit(X, T, pool=pool)
    finally:
        if pool is not None:
            pool.close()
        qij_module.fit_influence_model = original_fit
        qij_module._psi0 = original_psi0
    wall = time.perf_counter() - t0

    old_df = pd.read_parquet(RUN_ROOT / "qij/s00001.parquet").iloc[0]
    ij_df = pd.read_parquet(RUN_ROOT / "ij/s00001.parquet").iloc[0]
    Vij = np.array([float(ij_df[f"V_ij_{o}"]) for o in OUTPUTS])
    old_vbtw = np.array([float(old_df[f"V_btw_{o}"]) for o in OUTPUTS])
    old_vtot = np.array([float(old_df[f"V_tot_hat_{o}"]) for o in OUTPUTS])
    old_evals = np.array([int(old_df[f"n_refine_evals_{o}"]) for o in OUTPUTS])
    old_L = np.array([int(old_df[f"L_{o}"]) for o in OUTPUTS])

    summary = {
        "design": (
            "moment-aware posterior mean; production point-GP posterior "
            "uncertainty and all IVQ/refinement mechanics retained"
        ),
        "outputs": list(OUTPUTS),
        "N": len(X),
        "M_X": 1094,
        "moment_rows": int(captured["moment_rows"]),
        "aggregate_ell": captured["ell"],
        "aggregate_lambda": captured["lam"].tolist(),
        "aggregate_width_evaluations": int(captured["n_width_evals"]),
        "control_stored": {
            "V_btw_over_Vij": (old_vbtw / Vij).tolist(),
            "V_tot_hat_over_Vij": (old_vtot / Vij).tolist(),
            "n_refine_evals": old_evals.tolist(),
            "total_refine_evals": int(old_evals.sum()),
            "L": old_L.tolist(),
        },
        "moment_mean_experiment": {
            "V_btw_over_Vij": (result.V_btw / Vij).tolist(),
            "V_win_hat_over_Vij": (result.V_win_hat / Vij).tolist(),
            "V_tot_hat_over_Vij": (result.V_tot_hat / Vij).tolist(),
            "n_refine_evals": result.n_refine_evals.tolist(),
            "total_refine_evals": int(result.n_refine_evals.sum()),
            "L": result.L.tolist(),
            "evals_by_stage": {k: int(v) for k, v in result.evals_by_stage.items()},
            "wall_time_by_stage": {
                k: float(v) for k, v in result.wall_time_by_stage.items()
            },
            "wall_seconds_harness": wall,
        },
    }
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    np.savez_compressed(
        args.out / "fields.npz", psi0=captured["prediction"],
        psi_hat=result.psi_hat, labels=result.bin_label,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
