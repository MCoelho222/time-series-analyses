from __future__ import annotations

import scipy.stats as sts


def p_value_normal(statistic: float, alternative: str) -> float:
    """
    Calculate the p_value for a given test statistic, using the normal distribution.

    Parameters
    ----------
        statistic
            The value of the statistic of the test.
        alternative
            The alternative hypothesis. Options: less, greater or two-sided.

    Returns
    -------
        The p_value.
    """
    z = abs(statistic)
    if alternative in ('less', 'greater'):
        p_value = 1 - sts.norm.cdf(z)
    if alternative == 'two-sided':
        p_value = 2 * (1 - sts.norm.cdf(z))

    return p_value
