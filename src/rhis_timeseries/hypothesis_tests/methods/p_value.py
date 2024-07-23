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

def test_decision_norm(z, alpha, alternative):
    if alternative == 'two-sided':
        limit = 1 - alpha / 2
        z_alpha = sts.norm.ppf(limit)
        print(z_alpha)
        return False if z > z_alpha else True
    if alternative == 'greater':
        z_alpha = 1 - sts.norm.ppf(alpha)
        print(z_alpha)
        return False if z > z_alpha and z > 0 else True
    if alternative == 'less':
        z_alpha = sts.norm.ppf(alpha)
        print(z_alpha)
        return False if z > z_alpha and z < 0 else True

if __name__ == "__main__":

    p_value_normal(1.65)

    test_decision_norm(1.2, 0.05, 'two-sided')
