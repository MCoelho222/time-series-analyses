from __future__ import annotations

from collections import namedtuple
from typing import TYPE_CHECKING

import numpy as np
import scipy.stats as sts

from rhis_ts.stats.decorators.hyps import check_test_args
from rhis_ts.stats.utils.ranks import ranks_ties_corrected

if TYPE_CHECKING:
    from rhis_ts.types.stats import TestResults


@check_test_args('mann-kendall')
def mann_kendall(
        ts: list[int|float] | np.ndarray[int|float],
        alternative: str = 'two-sided',
        alpha: float=0.05,
    ) -> TestResults:
    """
    ---------------------------------------------------------------------------
    Apply the Mann-Kendall test using the normal approximation,
    which is valid for series with 10 or more elements (GILBERT, 1987).
    ---------------------------------------------------------------------------
    References
    ----------
        GILBERT, R. O. (1987). Statistical Methods for Environmental Pollution
        Monitoring.

        HELSEL & HIRSCH (2002). Techniques of Water Resources investigations of
        the United States Geological Survey. Chapter 3 - Statistical Methods in
        Water Resources.
    ---------------------------------------------------------------------------
    Parameters
    ----------
        ts
            A time series to be tested.

        alternative
            One of the alternative hypotheses: 'two-sided', 'greater',
            or 'less'.

        alpha
            The significance level for the test. Default is 0.05.
    ---------------------------------------------------------------------------
    Return
    ------
        namedtuple
            ('Mann_Kendall', ['statistic', 'p_value', 'reject'])

            The parameter 'reject' is of type bool. 'True' means the null
            hypothesis was reject.
    ---------------------------------------------------------------------------
    """
    Results = namedtuple('Mann_Kendall', ['statistic', 'p_value', 'reject', 'alternative'])  # noqa: PYI024
    n = len(ts)
    ts = np.array(ts)
    signs = []

    for i in range(n - 1):
        s = ts[i + 1] - ts[:i + 1]
        signs.extend(np.sign(s))

    signs_array = np.array(signs)
    test_s = float(len(signs_array[signs_array > 0]) - len(signs_array[signs_array < 0]))

    ties_data = ranks_ties_corrected(ts, ties_data=True)['ties_groups_count']

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

    p = (1 - sts.norm.cdf(z))

    if alternative == 'two-sided':
        p = p * 2
        reject = p < alpha

    if alternative == 'less':
        reject = test_s < condition_value and p < alpha

    if alternative == 'greater':
        reject = test_s > condition_value and p < alpha


    return Results(test_s, round(p, 4), reject, alternative)

