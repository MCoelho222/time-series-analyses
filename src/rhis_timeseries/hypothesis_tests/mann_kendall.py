from __future__ import annotations

from collections import namedtuple
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as sts

from rhis_timeseries.hypothesis_tests.methods.p_value import p_value_normal, test_decision_norm
from rhis_timeseries.hypothesis_tests.methods.ties import ties_correction

if TYPE_CHECKING:
    from rhis_timeseries.types.hypothesis_types import TestResults


def mann_kendall_test(
        ts: list[int|float] | np.ndarray[int|float],
        alternative: str = 'two-sided',
        alpha: float=0.05,*,
        continuity: bool=True
        ) -> TestResults:
    """
    Apply the Mann-Kendall test to a time series.

    Reference: HELSEL & HIRSCH (2002). Techniques of Water Resources investigations fo the United States Geological Survey.
    Chapter 3 - Statistical Methods in Water Resources. p. 212-216 (Kendall's Tau).

    Parameters
    ----------
        alternative
            One of the alternative hypotheses:
                two-sided
                greater
                less
        alpha
            The significance level for the test.
        continuity
            If True, applies the correction for continuity for the normal approximation.

    Return
    ------
        namedtuple('Mann_Kandall_Test', ['statistic', 'p_value', 'alternative', 'decision'])
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
    print('ties_factor', ties_factor)

    sigma = ((1 / 18) * ((n * (n - 1.) * (2. * n + 5.)) - ties_factor)) ** 0.5
    print('sigma', sigma)

    condition_value = 0.

    if test_s > condition_value:
        z = abs((test_s - 1.)/sigma)
    if test_s == condition_value:
        z = condition_value
    if test_s < condition_value:
        z = abs((test_s + 1.)/sigma)
    print(z)
    # p = 2 * (1 - sts.norm.cdf(z))

    p_value = p_value_normal(z)
    # print(p_value)
    decision = test_decision_norm(z, alpha, alternative)
    Results = namedtuple('Mann_Kendall', ['z', 'p_value', 'alternative', 'decision'])  # noqa: PYI024

    return Results(z, p_value, alternative, decision)


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import pandas as pd

    # df = pd.read_csv('./data/MarchMilwaukeeChloride.csv')
    # print(df.head())
    # ts = df['Conc']
    ts = [18, 20, 23, 23, 23, 35]
    # df.plot()
    # plt.show()

    # rng = np.random.default_rng()
    # ts = rng.integers(0, 100, 100)
    # print(ts)

    print(mann_kendall_test(ts, alpha=0.1, alternative='greater'))

    # plt.figure()
    # plt.scatter(np.arange(len(ts)), ts)
    # plt.show()
