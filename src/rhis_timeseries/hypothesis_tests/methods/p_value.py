from __future__ import annotations

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


def test_decision_normal(z: float, alpha: float, alternative: str) -> dict[str, float]:
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
    if alternative == 'two-sided':
        alpha = 1 - alpha / 2
        z = abs(z)
    if alternative == 'greater':
        alpha = 1 - alpha

    z_alpha = sts.norm.ppf(alpha)
    cond = z > z_alpha if alternative != 'less' else z < z_alpha

    decision = False if cond else True

    return {
        'alpha': alpha,
        'z': z_alpha,
        'decision': decision
    }

