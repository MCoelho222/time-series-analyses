from __future__ import annotations

from collections import namedtuple
from typing import TYPE_CHECKING

import numpy as np
import scipy.stats as sts

from rhis.stats.hypothesis import mann_kendall, mann_whitney, wald_wolfowitz, wallismoore

if TYPE_CHECKING:
    from rhis.custom_types import TimeSeriesFlex


def p_value_normal(z: float) -> float:
    """
    Calculate the p_value from the normal distribution.

    Parameters
    ----------
        z
            The z value from a test that uses normal approximation.

    Returns
    -------
        The p_value.
    """
    z_abs = abs(z)
    p_value = 1 - sts.norm.cdf(z_abs)

    return p_value

def test_decision_normal(
        stat: float,
        stat_mean: float,
        z: float,
        alternative: str,
        alpha: float
        ) -> dict[str, float]:
    """
    Decide about rejection of the null hypothesis using normal
    approximation.

    Parameters
    ----------
        stat
            The value of the test statistic.
        stat_mean
            The expected value of the test statistic.
        z
            The value of the normalized test statistic.
        alpha
            The significance level of the test.
        alternative
            The alternative hypothesis: 'two-sided', 'greater',
            or 'less'.

    Return
    ------
        A namedtuple
            ('TestDecisionNormal', ['p_value', 'alpha', 'reject'
            , 'alternative'])
            The parameter 'reject' is of type bool. 'True' means
            the null hypothesis was reject.
    """
    p = p_value_normal(z)

    if alternative == 'two-sided':
        p = p * 2
        reject = p < alpha
    if alternative == 'less':
        reject = stat < stat_mean and p < alpha
    if alternative == 'greater':
        reject = stat > stat_mean and p < alpha

    Result = namedtuple('TestDecisionNormal', ['p_value', 'alpha', 'reject', 'alternative'])  # noqa: PYI024

    return Result(p, alpha, reject, alternative)


def calculate_rhis(ts: TimeSeriesFlex, alpha: float, *, min: bool = True) -> int | dict[float]:
    hypos = ['R', 'H', 'I', 'S']
    rhis_tests = [wallismoore, mann_whitney, wald_wolfowitz, mann_kendall]
    test_dict = dict(zip(hypos, rhis_tests))

    ps = []
    for hyp in hypos:
        ps.append(test_dict[hyp](ts, alpha).p_value)

    result = round(np.min(ps), 4) if min else ps

    return result

