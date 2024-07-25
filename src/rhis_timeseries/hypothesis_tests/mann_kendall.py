from __future__ import annotations

from collections import namedtuple
from typing import TYPE_CHECKING

import numpy as np

from rhis_timeseries.hypothesis_tests.methods.p_value import p_value_normal, test_decision_normal
from rhis_timeseries.hypothesis_tests.methods.ties import ties_correction

if TYPE_CHECKING:
    from rhis_timeseries.types.hypothesis_types import TestResults


def mann_kendall_test(
        ts: list[int|float] | np.ndarray[int|float],
        alternative: str = 'two-sided',
        alpha: float=0.05,
    ) -> TestResults:
    """
    Apply the Mann-Kendall test using the normal approximation,
    which is valid for series with 10 or more elements (GILBERT, 1987).

    Parameters
    ----------
        alternative
            One of the alternative hypotheses: 'two-sided', 'greater', or 'less'.
        alpha
            The significance level for the test. Default is 0.05.

    Return
    ------
        namedtuple('Mann_Kendall', ['z', 'p_value', 'alternative', 'decision'])
            z: test statistic
            p_value: p-value of the test
            alternative: 'two-sided', 'greater', or 'less'
            decision: boolean. True if null hypothesis could not be rejected, False otherwise.

    References
    ----------
        GILBERT, R. O. (1987). Statistical Methods for Environmental Pollution Monitoring.

        HELSEL & HIRSCH (2002). Techniques of Water Resources investigations fo the United States Geological Survey.
        Chapter 3 - Statistical Methods in Water Resources.
    """
    n = len(ts)
    ts = np.array(ts)
    signs = []

    for i in range(n - 1):
        s = ts[i + 1] - ts[:i + 1]
        signs.extend(np.sign(s))

    signs_array = np.array(signs)
    test_s = float(len(signs_array[signs_array > 0]) - len(signs_array[signs_array < 0]))

    ties_data = ties_correction(ts, ties_data=True)['ties_groups_count']

    ties_factor = 0
    for value in ties_data:
        ties_factor += (value * (value - 1) * (2 * value + 5))

    sigma = ((1 / 18) * ((n * (n - 1.) * (2. * n + 5.)) - ties_factor)) ** 0.5

    condition_value = 0.

    if test_s > condition_value:
        z = abs((test_s - 1.)/sigma)
    if test_s == condition_value:
        z = condition_value
    if test_s < condition_value:
        z = abs((test_s + 1.)/sigma)

    p_value = p_value_normal(z)
    decision = test_decision_normal(z, alpha, alternative)['decision']
    Results = namedtuple('Mann_Kendall', ['z', 'p_value', 'alternative', 'decision'])  # noqa: PYI024

    return Results(round(z, 2), round(p_value, 4), alternative, decision)

