from __future__ import annotations

import scipy.stats as sts

from src.rhis_ts.hypothesis_tests.mann_kendall import mann_kendall


def test_mann_kendall():
    """
    Test the Mann-Kendall test.

    It uses an example from GILBERT (1987).
    It is in page 212 in chapter 16 - Detecting and Estimating Trends.

    References
    ----------
        GILBERT, R. O. (1987). Statistical Methods for Environmental Pollution Monitoring.
    """
    ts = [20, 20, 20, 20, 15, 20, 20, 30, 27, 26, 23, 35, 25, 28, 70, 26, 24, 34, 32, 23, 50, 30]

    expected_z = 3.1
    accepted_error = 0.02
    result = mann_kendall(ts, 'greater')
    z = abs(sts.norm.ppf(result.p_value))
    error = abs(z - expected_z) / expected_z

    assert error <= accepted_error
    assert result.reject
