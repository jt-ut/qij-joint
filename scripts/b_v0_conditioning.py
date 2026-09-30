#!/usr/bin/env python3.9
"""V0 (change B maths, spec/QIJ_joint_check_measured_spec.md section 7):
offline check of the GP-conditioning functions in
`core/influence_model.py` -- `design_index`, `condition_alpha`,
`prototype_mean`, `psi_at` -- with NO estimator evaluation: a synthetic
`InfluenceModel` fit directly by `fit_influence_model` from hand-built
Z, a minimal xvq-like stub, and synthetic I_proto.

    PYTHONPATH=src python3.9 scripts/b_v0_conditioning.py

Three coordinates: c=0 a smooth function on the full 50-prototype
design, c=1 a smooth function with 5 prototypes withheld (its own,
smaller design -- a distinct group from c=0), c=2 constant (the
constant path). Per non-constant coordinate: (a) interpolation --
observing coordinate c's own design exactly, at tiny noise, must
reproduce the observed values there; (b) no-information -- observing
at enormous noise must leave alpha unchanged; both via
`condition_alpha` + `prototype_mean`. (c) `psi_at` with every alpha
left at None must reproduce `psi0` exactly. The constant-path
coordinate is checked separately: `design_index` all -1,
`condition_alpha` raises, `prototype_mean`/`psi_at` return
`const_value` everywhere.
"""
import types

import numpy as np

from qij_joint.core.influence_model import (
    condition_alpha,
    design_index,
    fit_influence_model,
    prototype_mean,
    psi0,
    psi_at,
)

RNG = np.random.default_rng(20260930)
N = 2000
D_Z = 2
M = 50


def build_model():
    Z = RNG.normal(size=(N, D_Z))
    centers = RNG.uniform(-2.5, 2.5, size=(M, D_Z))
    p = RNG.uniform(0.5, 1.5, size=M)
    p /= p.sum()
    xvq = types.SimpleNamespace(p=p, centers=centers)

    I_proto = np.empty((M, 3), dtype=float)
    I_proto[:, 0] = np.sin(centers[:, 0]) + 0.5 * centers[:, 1]
    I_proto[:, 1] = centers[:, 0] ** 2 - centers[:, 1]
    I_proto[:, 2] = 7.3  # constant path

    drop = RNG.choice(M, size=5, replace=False)
    I_proto[drop, 1] = np.nan

    theta_Q = np.array([1.0, 1.0, 7.3])
    eta = 0.01

    model, busy = fit_influence_model(
        Z, xvq, I_proto, theta_Q, eta, gptrend='affine', gpwidth='global', pool=None,
    )
    assert busy == 0.0
    return model, Z, I_proto


def recover_design_order(model, c):
    """idx_g (M_c,): the live-prototype index at each row of
    `model.centers[c]`, recovered purely from `design_index` (the
    inverse mapping) -- exercises item 1 of the interface."""
    di = design_index(model, c)
    mask = di >= 0
    M_c = model.centers[c].shape[0]
    idx_g = np.empty(M_c, dtype=np.intp)
    idx_g[di[mask]] = np.where(mask)[0]
    return idx_g


def check_coordinate(model, I_proto, c):
    print(f"\n--- coordinate {c} (M_c = {model.centers[c].shape[0]}) ---")
    idx_g = recover_design_order(model, c)
    y_c = I_proto[idx_g, c]
    M_c = idx_g.size
    s2_c = float(model.s2[c])

    # (a) interpolation -- vector-norm relative error (a component-wise
    # ratio is dominated by whichever y_c entry sits nearest 0)
    W_id = np.eye(M_c)
    noise_tiny = np.full(M_c, 1e-14 * s2_c)
    alpha_interp = condition_alpha(model, c, W_id, y_c, noise_tiny)
    pm = prototype_mean(model, c, alpha_interp)
    rel_interp = np.linalg.norm(pm[idx_g] - y_c) / np.linalg.norm(y_c)
    print(f"(a) interpolation rel err (norm): {rel_interp:.3e}  (tol 1e-8)")
    assert rel_interp < 1e-8

    # (b) no-information -- same norm convention
    noise_huge = np.full(M_c, 1e12 * s2_c)
    alpha_noinfo = condition_alpha(model, c, W_id, y_c, noise_huge)
    rel_noinfo = np.linalg.norm(alpha_noinfo - model.alpha[c]) / np.linalg.norm(model.alpha[c])
    print(f"(b) no-information rel err (norm): {rel_noinfo:.3e}  (tol 1e-10)")
    assert rel_noinfo < 1e-10


def check_constant(model):
    c = 2
    print(f"\n--- coordinate {c} (constant path) ---")
    di = design_index(model, c)
    assert np.all(di == -1)
    print("design_index: all -1, as required")

    try:
        condition_alpha(model, c, np.eye(1), np.zeros(1), np.ones(1))
        raised = False
    except ValueError:
        raised = True
    assert raised
    print("condition_alpha: raised ValueError, as required")

    pm = prototype_mean(model, c, model.alpha[c])
    assert np.all(pm == model.const_value[c])
    print(f"prototype_mean: const_value ({model.const_value[c]:.6f}) at every row")


def check_psi_at_matches_psi0(model, Z):
    print("\n--- (c) psi_at([None]*q) vs psi0 ---")
    q = len(model.centers)
    p0 = psi0(model, Z)
    pa = psi_at(model, Z, [None] * q)
    max_abs = float(np.max(np.abs(pa - p0)))
    denom = np.maximum(np.abs(p0), 1e-12)
    max_rel = float(np.max(np.abs(pa - p0) / denom))
    print(f"max abs diff: {max_abs:.3e}, max rel diff: {max_rel:.3e} (state: exact per design)")
    assert max_abs == 0.0


def main():
    model, Z, I_proto = build_model()
    print(f"q = {len(model.centers)}, constant_path = {model.constant_path.tolist()}")
    print(f"coordinate 0 design size: {model.centers[0].shape[0]}")
    print(f"coordinate 1 design size: {model.centers[1].shape[0]}")
    print(f"distinct design groups (0 vs 1): {model.centers[0] is not model.centers[1]}")

    check_coordinate(model, I_proto, 0)
    check_coordinate(model, I_proto, 1)
    check_constant(model)
    check_psi_at_matches_psi0(model, Z)

    print("\nALL V0 CHECKS PASSED")


if __name__ == '__main__':
    main()
