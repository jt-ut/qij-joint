"""
`gp_condition`: the joint check's `gp_update` conditioning pass
(spec/QIJ_joint_check_measured_spec.md section 7). Split out of
`core.joint` to keep that module within its line budget (R1); the only
caller is `core.joint.run_joint`'s own `_apply_gp_update` closure, once
after the initial measurement and the seeds, and again after every
check round that measured a split.
"""
from __future__ import annotations

from typing import Dict, Tuple

import numpy as np

from .influence_model import condition_alpha, design_index, prototype_mean, psi_at

__all__ = ["gp_condition", "gp_design_weights"]


def gp_design_weights(model, xvq, label_arr: np.ndarray, L: int, c: int):
    """Bin k's composition over its points' bmu, restricted to output
    c's design columns (spec 7.1): w_kj = n_kj/n_k over design column
    j, a point whose bmu maps to no design column (`design_index`'s
    -1) dropped and the rest renormalised -- one `bincount` over (bin
    label, design column) (E2), never a per-bin loop over points.
    Returns (keep, W_c): keep (L,) bool, the bins left with >= 1 such
    point; W_c (keep.sum(), M_c), rows summing to 1."""
    col = design_index(model, c)
    M_c = int(col.max()) + 1 if col.size and col.max() >= 0 else 0
    if M_c == 0:
        return np.zeros(L, dtype=bool), np.zeros((0, 0))
    dc = col[xvq.bmu]
    valid = dc >= 0
    flat = label_arr[valid] * M_c + dc[valid]
    counts = np.bincount(flat, minlength=L * M_c).reshape(L, M_c).astype(float)
    n_valid = counts.sum(axis=1)
    keep = n_valid > 0
    return keep, counts[keep] / n_valid[keep, None]


def gp_condition(
    model, Z: np.ndarray, xvq, I_proto: np.ndarray, leaves: Dict[int, dict],
    a: np.ndarray, N: int, q: int,
) -> Tuple[np.ndarray, np.ndarray, int]:
    """One `gp_update` conditioning pass (spec section 7.1-7.2): observe
    every CURRENT bin once through its prototype composition
    (`gp_design_weights`), condition coordinate c's posterior
    (`condition_alpha`) on those observations, and push the result to
    the N points (`psi_at`) and to the M_X_used prototypes
    (`prototype_mean`). A skipped coordinate -- constant path, a_c 0 or
    non-finite, or a bin with no remaining design-mapped point -- keeps
    the model's own alpha (psi_at's alphas[c]=None) and I_proto's own
    column; `n_skipped` counts skipped (bin, output) pairs. Returns
    (psi0_cond (N, q), proto_cond (M_X_used, q), n_skipped). Every
    leaf in `leaves` must carry its own 'U' and 'delta_U' (q,)."""
    ids = sorted(leaves.keys())
    L = len(ids)
    label_arr = np.empty(N, dtype=int)
    for pos, lid in enumerate(ids):
        label_arr[leaves[lid]['indices']] = pos
    U = np.array([leaves[lid]['U'] for lid in ids])
    delta_U = np.array([leaves[lid]['delta_U'] for lid in ids])

    alphas = [None] * q
    proto_cols = [None] * q
    n_skipped = 0
    for c in range(q):
        a_c = float(a[c])
        if bool(model.constant_path[c]) or not np.isfinite(a_c) or a_c == 0.0:
            n_skipped += L
            continue
        keep, W_c = gp_design_weights(model, xvq, label_arr, L, c)
        n_skipped += int(L - keep.sum())
        if not keep.any():
            continue
        y_c = U[keep, c] / a_c
        noise_var_c = (delta_U[keep, c] / a_c) ** 2
        alpha_c = condition_alpha(model, c, W_c, y_c, noise_var_c)
        alphas[c] = alpha_c
        proto_cols[c] = prototype_mean(model, c, alpha_c)

    psi0_cond = psi_at(model, Z, alphas)
    proto_cond = np.column_stack(
        [proto_cols[c] if proto_cols[c] is not None else I_proto[:, c] for c in range(q)])
    return psi0_cond, proto_cond, n_skipped
