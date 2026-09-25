"""
The one weight constructor and the one step rule (spec/method_notes.md
section 1), used by stage 1's prototype survey (`xvq.prototype_influences`)
and stage 2's bin stencil (`ivq.bin_differences`).
"""

from __future__ import annotations

from typing import Callable

import numpy as np


def forward_step(eta: float) -> float:
    """delta_f = 2*sqrt(eta): the single forward step used by the
    prototype survey and by a refinement split's one-shot measurement."""
    return float(2.0 * np.sqrt(eta))


def central_step(eta: float) -> float:
    """delta = (3*eta)^(1/3): the central-stencil step used to measure
    the initial bins on the full data."""
    return float(np.cbrt(3.0 * eta))


def step_parameter(delta: float, p: float) -> float:
    """t = delta*p/(1-p): the weight parameter of a relative step delta
    on a member set of mass p (spec/method_notes.md section 1)."""
    return float(delta * p / (1.0 - p))


def perturbed_weights(
    omega0: np.ndarray,
    member_mask: np.ndarray,
    t: float,
) -> np.ndarray:
    """
    The one weight constructor: weights at weight parameter t along
    e_K - p, for member set K = {i : member_mask[i]}, base weights
    omega0 summing to R = omega0.size (spec/method_notes.md section 1):

        p = sum_{i in K} omega0_i / R
        omega_i(t) = (1 - t) * omega0_i + t * omega0_i * 1{i in K} / p

    Every omega_i(t) sums to R for every t.
    """
    omega0 = np.asarray(omega0, dtype=float)
    member_mask = np.asarray(member_mask, dtype=bool)
    R = omega0.size
    p = float(omega0[member_mask].sum()) / R
    return (1.0 - t) * omega0 + t * omega0 * member_mask.astype(float) / p


def difference(
    p: float,
    delta: float,
    evaluate: Callable[[float], np.ndarray],
) -> np.ndarray:
    """
    The central three-point stencil U = [T(+t) - T(-t)] / (2*t), t =
    step_parameter(delta, p) (spec/method_notes.md section 1). Always
    central: every registered estimator's eta keeps delta =
    central_step(eta) <= 1, so the downward step never drives a member
    weight negative. Exactly two calls to `evaluate`, in order +t then
    -t; exceptions from `evaluate` propagate uncaught.
    """
    t = step_parameter(delta, p)
    T_plus = np.asarray(evaluate(+t), dtype=float)
    T_minus = np.asarray(evaluate(-t), dtype=float)
    return (T_plus - T_minus) / (2.0 * t)
