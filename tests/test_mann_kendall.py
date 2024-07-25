from __future__ import annotations

from src.rhis_timeseries.hypothesis_tests.mann_kendall import mann_kendall_test


def test_mann_kendall():
    ts = [20, 20, 20, 20, 15, 20, 20, 30, 27, 26, 23, 35, 25, 28, 70, 26, 24, 34, 32, 23, 50, 30]
    correct_z = 3.1
    greater = mann_kendall_test(ts, 'greater')
    error = 0.1

    assert greater.z - correct_z < error
    assert not greater.decision
