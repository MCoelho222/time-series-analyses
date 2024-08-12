from __future__ import annotations

from collections import namedtuple

import scipy.stats as sts


def p_value_normal(statistic: float) -> float:
    """
    Calculate the p_value for a given test statistic, using the normal distribution.

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


def test_decision_normal(z: float, alternative: str, alpha: float, ) -> dict[str, float]:
    """
    Decision about null hypothesis using normal approximation.

    Parameters
    ----------
        z
            The value of the test statistic.
        alpha
            The significance level of the test.
        alternative
            The alternative hypothesis: 'two-sided', 'greater', or 'less'.

    Return
    ------
        A dictionary with z, alpha, and decision. The parameter 'decision' is a boolean,
        It is True when the null hypothesis could not be rejected, and False otherwise.
    """
    z_abs = abs(z)
    p = (1 - sts.norm.cdf(z_abs))

    if alternative == 'two-sided':
        p = p * 2
        reject = p < alpha
    if alternative == 'less':
        reject = p < alpha
    if alternative == 'greater':
        reject = p < alpha

    Result = namedtuple('TestDecisionNormal', ['p', 'alpha', 'reject'])  # noqa: PYI024

    return Result(p, alpha, reject)

if __name__ == "__main__":
    reject = test_decision_normal(-1.97, 'less', 0.05)
    print(reject)

