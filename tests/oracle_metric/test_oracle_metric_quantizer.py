"""Focused invariants for the standalone oracle-metric experiment."""

import numpy as np

from scripts.oracle_metric_quantizer import (
    dimensionless_coordinates,
    generalized_lloyd,
    local_pullback_metric,
)


def test_pullback_metric_and_fixed_budget_are_deterministic():
    rng = np.random.default_rng(17)
    X = rng.normal(size=(120, 2)) @ np.array([[2.0, 0.4], [0.0, 0.5]])
    psi = np.column_stack((X[:, 0] ** 2, np.sin(2.0 * X[:, 1])))
    z, phi, standardization = dimensionless_coordinates(X, psi)
    G = local_pullback_metric(z, phi, neighbors=12)

    assert np.allclose(G, np.swapaxes(G, 1, 2))
    assert np.linalg.eigvalsh(G).min() > 0.0

    initial = z[np.linspace(0, len(z) - 1, 12, dtype=int)]
    first = generalized_lloyd(
        X, z, G, initial, standardization, max_iter=8, batch=31
    )
    second = generalized_lloyd(
        X, z, G, initial, standardization, max_iter=8, batch=31
    )
    assert first.centers_x.shape == (12, 2)
    assert np.count_nonzero(first.weights) == 12
    assert first.weights.sum() == len(X)
    assert np.array_equal(first.labels, second.labels)
    assert np.allclose(first.centers_x, second.centers_x)
    assert np.all(np.diff(first.objective) <= 1e-10)
