"""
Agent C self-check (spec/QIJ_pilot_feedback_interface.md section 3):
`influence_model.refit_frozen`.

Draw 45 of cloudfil_u, study settings (eps=0.01, survey='moments',
quantized_start='full-data', gptrend='quadratic', gpwidth='local',
fit_weights='mass'). Stage 1 is run through `QIJ.fit` itself (not
reimplemented): `qij.fit_influence_model` is monkeypatched to capture
its own call's arguments and return value, then raise a sentinel
exception, aborting the draw right after stage 1's model is built
(before the costly stage-2 loop runs) -- the captured `(Z, xvq,
I_proto[:, measured], theta_Q[measured], eta, model)` are then exactly
what `qij.py` itself used, no code path duplicated.

Part 1 (KEY self-check, mandatory): `refit_frozen(model, Z, xvq,
I_proto_m, theta_Q_m, eta)` with the ORIGINAL inputs must reproduce the
original model's `psi0`/`uncertainty` over Z to <= 1e-10 relative.

Part 2: refit on a design with 5 prototypes dropped (a fresh XVQ-like
shim reusing xvq's own centers/p/bmu/bmu2/conn minus 5 prototypes, bmu
of their points reassigned to each point's recorded bmu2) -- shows the
function running on a genuinely different design and reports timing.
"""
from __future__ import annotations

import copy
import time
from dataclasses import dataclass

import numpy as np

import qij_joint.qij as qij_mod
from qij_joint import cloudfil_u, datasets
from qij_joint.core.influence_model import refit_frozen
from qij_joint.core.influence_model import psi0 as _psi0
from qij_joint.core.influence_model import uncertainty as _uncertainty
from qij_joint.qij import QIJ


class _Stop(Exception):
    pass


def _run_stage1(seed: int = 45):
    captured = {}
    orig_fit_influence_model = qij_mod.fit_influence_model

    def _capture(Z, xvq, I_proto_m, theta_Q_m, eta, **kwargs):
        model, busy = orig_fit_influence_model(Z, xvq, I_proto_m, theta_Q_m, eta, **kwargs)
        captured['Z'] = Z
        captured['xvq'] = xvq
        captured['I_proto_m'] = I_proto_m
        captured['theta_Q_m'] = theta_Q_m
        captured['eta'] = eta
        captured['model'] = model
        raise _Stop()

    qij_mod.fit_influence_model = _capture
    try:
        X = datasets.cloudfil_G_U_P3_v1(10000, seed)
        T = cloudfil_u.P2ShearMixture()
        q_obj = QIJ(eps=0.01, survey='moments', quantized_start='full-data',
                    gptrend='quadratic', gpwidth='local', fit_weights='mass')
        try:
            q_obj.fit(X, T)
            raise RuntimeError('stage 1 did not abort as expected')
        except _Stop:
            pass
    finally:
        qij_mod.fit_influence_model = orig_fit_influence_model

    return captured


def _rel_err(a: np.ndarray, b: np.ndarray) -> float:
    """max_i |a_i - b_i| / max(|a_i|, |b_i|, 1e-300)."""
    diff = np.abs(a - b)
    denom = np.maximum(np.abs(a), np.abs(b))
    denom = np.maximum(denom, 1e-300)
    return float(np.max(diff / denom))


@dataclass
class _XVQView:
    """A minimal stand-in exposing exactly the attributes `refit_frozen`
    (and `_conn_spacing`) read from an XVQ: `centers`, `p`, `bmu`,
    `conn`. Built from the real `xvq` with `drop` prototype ids removed
    and their points' `bmu` repointed to their recorded `bmu2`
    (clamped to stay inside the surviving set; any point whose bmu2 was
    itself dropped falls back to its nearest surviving prototype by
    whitened distance -- this view is for the timing/shape check only,
    not a claim about what the real `xvq_refine.refine_cells` would
    produce)."""

    centers: np.ndarray
    p: np.ndarray
    bmu: np.ndarray
    conn: object


def _drop_prototypes(xvq, drop_ids, Z):
    from scipy.spatial.distance import cdist

    M_old = xvq.centers.shape[0]
    keep_mask = np.ones(M_old, dtype=bool)
    keep_mask[list(drop_ids)] = False
    keep_ids = np.where(keep_mask)[0]
    old2new = np.full(M_old, -1, dtype=np.intp)
    old2new[keep_ids] = np.arange(keep_ids.size)

    bmu = np.asarray(xvq.bmu, dtype=np.intp).copy()
    bmu2 = np.asarray(xvq.bmu2, dtype=np.intp).copy()
    hit = ~keep_mask[bmu]
    # Points whose bmu was dropped: try bmu2 first, if it also survives.
    candidate = bmu2[hit]
    candidate_ok = keep_mask[candidate]
    bmu[np.where(hit)[0][candidate_ok]] = candidate[candidate_ok]
    # Remaining points (bmu2 also dropped, or unresolved): nearest
    # surviving prototype by whitened distance -- rare for a 5-id drop.
    still_bad = hit.copy()
    still_bad[np.where(hit)[0][candidate_ok]] = False
    if np.any(still_bad):
        centers_keep = xvq.centers[keep_mask]
        d = cdist(Z[still_bad], centers_keep)
        bmu[still_bad] = keep_ids[np.argmin(d, axis=1)]

    bmu_new = old2new[bmu]
    assert np.all(bmu_new >= 0)

    centers_new = xvq.centers[keep_mask]
    conn_new = xvq.conn.tocsr()[np.ix_(keep_ids, keep_ids)].tocsr()
    # Prototype mass redistributed from each dropped id onto whichever
    # surviving id its own points now carry (keeps p summing to 1; the
    # exact rule does not matter for this timing/shape check).
    p_new = np.zeros(keep_ids.size, dtype=float)
    counts = np.bincount(bmu_new, minlength=keep_ids.size)
    p_new = counts / counts.sum()

    return _XVQView(centers=centers_new, p=p_new, bmu=bmu_new, conn=conn_new)


def main():
    print("=== Part 1: KEY self-check (original inputs) ===")
    cap = _run_stage1(seed=45)
    model, Z, xvq = cap['model'], cap['Z'], cap['xvq']
    I_proto_m, theta_Q_m, eta = cap['I_proto_m'], cap['theta_Q_m'], cap['eta']
    print(f"M_X_used={xvq.centers.shape[0]}, q={I_proto_m.shape[1]}, N={Z.shape[0]}")

    psi0_orig = _psi0(model, Z).copy()
    sigma_orig = _uncertainty(model, Z).copy()

    t0 = time.perf_counter()
    model2 = refit_frozen(model, Z, xvq, I_proto_m, theta_Q_m, eta)
    t1 = time.perf_counter()

    psi0_new = _psi0(model2, Z)
    sigma_new = _uncertainty(model2, Z)

    rel_psi0 = _rel_err(psi0_orig, psi0_new)
    rel_sigma = _rel_err(sigma_orig, sigma_new)
    print(f"refit_frozen wall time (original inputs): {t1 - t0:.4f} s")
    print(f"psi0 max relative diff over Z:    {rel_psi0:.3e}")
    print(f"sigma max relative diff over Z:   {rel_sigma:.3e}")
    print(f"PASS (<=1e-10): {rel_psi0 <= 1e-10 and rel_sigma <= 1e-10}")

    print()
    print("=== Part 2: a design with 5 prototypes dropped ===")
    M_old = xvq.centers.shape[0]
    rng = np.random.default_rng(0)
    drop_ids = rng.choice(M_old, size=5, replace=False)
    print(f"dropping prototype ids: {sorted(drop_ids.tolist())}")

    xvq_small = _drop_prototypes(xvq, drop_ids, Z)
    keep_mask = np.ones(M_old, dtype=bool)
    keep_mask[drop_ids] = False
    I_proto_small = I_proto_m[keep_mask]
    print(f"new M_X_used={xvq_small.centers.shape[0]} (was {M_old})")

    t0 = time.perf_counter()
    model3 = refit_frozen(model, Z, xvq_small, I_proto_small, theta_Q_m, eta)
    t1 = time.perf_counter()
    psi0_3 = _psi0(model3, Z)
    sigma_3 = _uncertainty(model3, Z)
    print(f"refit_frozen wall time (5 dropped prototypes): {t1 - t0:.4f} s")
    print(f"psi0 shape {psi0_3.shape}, finite everywhere: {np.all(np.isfinite(psi0_3))}")
    print(f"sigma shape {sigma_3.shape}, finite everywhere: {np.all(np.isfinite(sigma_3))}")
    print(f"max |psi0_new - psi0_orig| (dropped design): {np.max(np.abs(psi0_3 - psi0_orig)):.6e}")


if __name__ == '__main__':
    main()
