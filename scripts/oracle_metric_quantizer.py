#!/usr/bin/env python3.9
"""Oracle influence-pullback quantizer for the cloudfil feasibility test.

This is deliberately a standalone research harness, not production QIJ code.
It asks a narrow question: if the exact point influences were known while the
codebook was learned, could ``M`` ordinary weighted centroids retain their
variance substantially better than Euclidean VQ?

The construction is dimensionless.  Data are whitened to ``z`` and each
influence output is divided by its pointwise RMS, giving ``phi``.  A local
linear regression on deterministic k-nearest-neighbour sets estimates

    phi(x') - phi(x) ~= (z(x') - z(x)) B(x),

and the pullback metric is ``G(x) = B(x) B(x)' / q``.  The generalized Lloyd
objective is

    sum_i (z_i - c[label_i])' (G_i + tau I) (z_i - c[label_i]).

The centroid update solves the exact normal equation for that objective.  A
small isotropic floor keeps regions with locally constant influence covered;
the influence part retains its magnitude, rather than being normalized point
by point, so the fixed prototype budget can migrate toward high-gradient
regions.  No estimator-specific score or mixture responsibility is used.

Example
-------
PYTHONPATH=src python3.9 scripts/oracle_metric_quantizer.py \
  --out /tmp/oracle_metric_draw1.npz
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from qij_joint import datasets
from qij_joint.core.xvq import fit_xvq
from qij_joint.registry import case as registry_case


DEFAULT_RUN_ROOT = Path(
    "/Users/jtaylor/Dropbox/Research/QIJ_joint/runs/"
    "cloudfil_clean/cloudfil_p2_N10000"
)


@dataclass(frozen=True)
class Standardization:
    x_mean: np.ndarray
    x_factor: np.ndarray
    psi_scale: np.ndarray


@dataclass(frozen=True)
class Quantization:
    centers_z: np.ndarray
    centers_x: np.ndarray
    labels: np.ndarray
    weights: np.ndarray
    objective: np.ndarray
    standardization: Standardization


def load_stored_influence(run_root: Path, draw: int, outputs) -> np.ndarray:
    path = run_root / "ij" / f"s{draw:05d}.psi.parquet"
    frame = pd.read_parquet(path)
    return np.column_stack([frame[f"psi_{name}"].to_numpy() for name in outputs])


def dimensionless_coordinates(X: np.ndarray, psi: np.ndarray):
    """Whiten X and RMS-standardize the centred influence columns."""
    X = np.asarray(X, dtype=float)
    psi = np.asarray(psi, dtype=float)
    x_mean = X.mean(axis=0)
    xc = X - x_mean
    cov = xc.T @ xc / len(X)
    eigval, eigvec = np.linalg.eigh(cov)
    floor = max(float(eigval.max()) * 1e-12, np.finfo(float).tiny)
    x_factor = eigvec @ np.diag(1.0 / np.sqrt(np.maximum(eigval, floor))) @ eigvec.T
    z = xc @ x_factor

    psi_c = psi - psi.mean(axis=0, keepdims=True)
    psi_scale = np.sqrt(np.mean(psi_c * psi_c, axis=0))
    psi_scale = np.where(psi_scale > 0.0, psi_scale, 1.0)
    phi = psi_c / psi_scale
    return z, phi, Standardization(x_mean, x_factor, psi_scale)


def local_pullback_metric(
    z: np.ndarray,
    phi: np.ndarray,
    *,
    neighbors: int = 32,
    ridge_fraction: float = 1e-3,
    trace_clip_quantile: float = 0.995,
    isotropic_fraction: float = 0.05,
) -> np.ndarray:
    """Estimate one regularized pullback metric per observation.

    ``ridge_fraction`` is relative to the neighbourhood design's mean
    eigenvalue.  The upper trace clip limits isolated regression explosions,
    while preserving spatial variation below that deterministic empirical
    quantile.  ``isotropic_fraction`` is relative to the median (unclipped)
    influence-metric trace per data dimension.
    """
    z = np.asarray(z, dtype=float)
    phi = np.asarray(phi, dtype=float)
    n, d = z.shape
    q = phi.shape[1]
    if not 2 <= neighbors < n:
        raise ValueError("neighbors must be in [2, N-1]")
    # Query includes the point itself; retain it as the regression origin.
    _, idx = cKDTree(z).query(z, k=neighbors + 1, workers=1)
    G = np.empty((n, d, d), dtype=float)
    eye = np.eye(d)
    for i in range(n):
        nb = idx[i, 1:]
        dx = z[nb] - z[i]
        dy = phi[nb] - phi[i]
        gram = dx.T @ dx
        ridge = ridge_fraction * max(float(np.trace(gram)) / d, np.finfo(float).eps)
        B = np.linalg.solve(gram + ridge * eye, dx.T @ dy)
        G[i] = (B @ B.T) / q

    trace = np.trace(G, axis1=1, axis2=2)
    cap = float(np.quantile(trace, trace_clip_quantile))
    multiplier = np.minimum(1.0, cap / np.maximum(trace, np.finfo(float).tiny))
    G *= multiplier[:, None, None]
    base = isotropic_fraction * float(np.median(trace)) / d
    if not np.isfinite(base) or base <= 0.0:
        base = isotropic_fraction
    G += base * eye[None, :, :]
    return G


def _metric_distances(z: np.ndarray, centers: np.ndarray, G: np.ndarray) -> np.ndarray:
    delta = z[:, None, :] - centers[None, :, :]
    return np.einsum("nmd,ndk,nmk->nm", delta, G, delta, optimize=True)


def _assigned_loss(z: np.ndarray, assigned_centers: np.ndarray, G: np.ndarray) -> np.ndarray:
    delta = z - assigned_centers
    return np.einsum("nd,ndk,nk->n", delta, G, delta, optimize=True)


def _assign_batched(
    z: np.ndarray, centers: np.ndarray, G: np.ndarray, batch: int
):
    n = len(z)
    labels = np.empty(n, dtype=np.int64)
    loss = np.empty(n, dtype=float)
    for lo in range(0, n, batch):
        hi = min(n, lo + batch)
        dist = _metric_distances(z[lo:hi], centers, G[lo:hi])
        labels[lo:hi] = np.argmin(dist, axis=1)
        loss[lo:hi] = dist[np.arange(hi - lo), labels[lo:hi]]
    return labels, loss


def _repair_empty(labels: np.ndarray, loss: np.ndarray, m: int) -> np.ndarray:
    """Give each empty cluster a distinct high-loss point, deterministically."""
    labels = labels.copy()
    counts = np.bincount(labels, minlength=m)
    empty = np.flatnonzero(counts == 0)
    if not len(empty):
        return labels
    # Stable tie break by point index.
    candidates = np.lexsort((np.arange(len(loss)), -loss))
    used = np.zeros(len(labels), dtype=bool)
    for cluster in empty:
        chosen = -1
        for point in candidates:
            donor = labels[point]
            if not used[point] and counts[donor] > 1:
                chosen = int(point)
                break
        if chosen < 0:
            raise RuntimeError("cannot repair empty generalized-Lloyd cluster")
        donor = labels[chosen]
        counts[donor] -= 1
        labels[chosen] = cluster
        counts[cluster] = 1
        used[chosen] = True
    return labels


def generalized_lloyd(
    X: np.ndarray,
    z: np.ndarray,
    G: np.ndarray,
    initial_centers_z: np.ndarray,
    standardization: Standardization,
    *,
    max_iter: int = 30,
    rtol: float = 1e-7,
    batch: int = 256,
) -> Quantization:
    """Minimize the point-metric quadratic objective at fixed M."""
    centers = np.asarray(initial_centers_z, dtype=float).copy()
    m, d = centers.shape
    eye = np.eye(d)
    history = []
    prior_labels: Optional[np.ndarray] = None
    for _ in range(max_iter):
        labels, loss = _assign_batched(z, centers, G, batch)
        labels = _repair_empty(labels, loss, m)
        objective = float(np.sum(_assigned_loss(z, centers[labels], G)))
        history.append(objective)

        updated = np.empty_like(centers)
        for j in range(m):
            member = labels == j
            lhs = G[member].sum(axis=0)
            rhs = np.einsum("nij,nj->i", G[member], z[member])
            # The isotropic floor should make lhs PD; pinv is a deterministic
            # guard for extreme floating-point conditioning.
            try:
                updated[j] = np.linalg.solve(lhs, rhs)
            except np.linalg.LinAlgError:
                updated[j] = np.linalg.pinv(lhs + 1e-14 * eye) @ rhs
        same = prior_labels is not None and np.array_equal(labels, prior_labels)
        improvement = np.inf if len(history) < 2 else (history[-2] - history[-1]) / max(history[-2], 1e-300)
        centers = updated
        prior_labels = labels
        if same or (0.0 <= improvement < rtol):
            break

    # Report assignments to the final updated centers, not the penultimate set.
    labels, loss = _assign_batched(z, centers, G, batch)
    labels = _repair_empty(labels, loss, m)
    weights = np.bincount(labels, minlength=m).astype(float)
    # x = mean + z @ inv(x_factor); solve instead of materializing inverse.
    centers_x = standardization.x_mean + np.linalg.solve(
        standardization.x_factor.T, centers.T
    ).T
    return Quantization(
        centers_z=centers,
        centers_x=centers_x,
        labels=labels,
        weights=weights,
        objective=np.asarray(history),
        standardization=standardization,
    )


def variance_share(psi: np.ndarray, labels: np.ndarray, m: int) -> np.ndarray:
    """Exact oracle between-cell / total influence variance by output."""
    psi = psi - psi.mean(axis=0, keepdims=True)
    count = np.bincount(labels, minlength=m).astype(float)
    cell_mean = np.column_stack([
        np.bincount(labels, weights=psi[:, c], minlength=m) / count
        for c in range(psi.shape[1])
    ])
    between = np.sum(count[:, None] * cell_mean * cell_mean, axis=0)
    total = np.sum(psi * psi, axis=0)
    return between / total


def _initial_centers_in_z(xvq, standardization: Standardization) -> np.ndarray:
    # fit_xvq was run on z, so its centers are already dimensionless z rows.
    return np.asarray(xvq.centers, dtype=float)


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--draw", type=int, default=1)
    parser.add_argument("--M", type=int, default=1094)
    parser.add_argument("--neighbors", type=int, default=32)
    parser.add_argument("--ridge-fraction", type=float, default=1e-3)
    parser.add_argument("--trace-clip-quantile", type=float, default=0.995)
    parser.add_argument("--isotropic-fraction", type=float, default=0.05)
    parser.add_argument("--max-iter", type=int, default=30)
    parser.add_argument("--batch", type=int, default=256)
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)

    X = datasets.cloudfil_G_B6_P3_v1(10000, args.draw)
    T = registry_case("cloudfil", "p2").make_T()
    all_psi = load_stored_influence(args.run_root, args.draw, T.outputs)
    measured = np.asarray(T.measured, dtype=int)
    psi = all_psi[:, measured]
    output_names = [T.outputs[c] for c in measured]
    z, phi, standardization = dimensionless_coordinates(X, psi)
    euclidean = fit_xvq(z, args.M, args.draw, workers=1)
    G = local_pullback_metric(
        z, phi, neighbors=args.neighbors,
        ridge_fraction=args.ridge_fraction,
        trace_clip_quantile=args.trace_clip_quantile,
        isotropic_fraction=args.isotropic_fraction,
    )
    result = generalized_lloyd(
        X, z, G, _initial_centers_in_z(euclidean, standardization),
        standardization, max_iter=args.max_iter, batch=args.batch,
    )
    share_euclidean = variance_share(psi, euclidean.bmu, euclidean.M_used)
    share_oracle = variance_share(psi, result.labels, len(result.centers_x))
    summary = {
        "draw": args.draw,
        "N": len(X),
        "M_requested": args.M,
        "M_euclidean_live": int(euclidean.M_used),
        "M_oracle_live": int(np.count_nonzero(result.weights)),
        "neighbors": args.neighbors,
        "ridge_fraction": args.ridge_fraction,
        "trace_clip_quantile": args.trace_clip_quantile,
        "isotropic_fraction": args.isotropic_fraction,
        "iterations": len(result.objective),
        "outputs": output_names,
        "share_euclidean": share_euclidean.tolist(),
        "share_oracle_metric": share_oracle.tolist(),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.out,
        X=X,
        psi=psi,
        metric=G,
        centers_x=result.centers_x,
        centers_z=result.centers_z,
        labels=result.labels,
        weights=result.weights,
        objective=result.objective,
        euclidean_centers_z=euclidean.centers,
        euclidean_labels=euclidean.bmu,
        summary_json=json.dumps(summary, sort_keys=True),
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
