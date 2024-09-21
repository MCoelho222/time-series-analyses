from __future__ import annotations

from collections import namedtuple

import scipy.stats as sts


def p_value_normal(statistic: float) -> float:
    """
    Calculate the p_value for a given test statistic,
    using the normal distribution.

    Parameters
    ----------
        statistic
            The value of the statistic of the test.

    Returns
    -------
        The p_value.
    """
    z = abs(statistic)
    p_value = 1 - sts.norm.cdf(z)

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

