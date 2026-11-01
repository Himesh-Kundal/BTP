"""Smoke test to ensure testing infrastructure and imports work."""


def test_smoke():
    import numpy as np

    a = np.array([1, 2, 3], dtype=np.int64)
    assert a.sum() == 6
