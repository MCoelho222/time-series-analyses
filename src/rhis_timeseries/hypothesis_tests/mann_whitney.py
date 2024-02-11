from __future__ import annotations

from collections import namedtuple
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as sts

from rhis_timeseries.hypothesis_tests.methods.ties import ties_correction

if TYPE_CHECKING:
    from rhis_timeseries.types.hypothesis_types import TestResults


def mann_whitney_test(a: list[int | float], b: list[int | float], alpha: float=0.05) -> dict[str, float | TestResults]:  # noqa: C901
    """
    Compare if two groups of data has equal medians.

    Assumes homoscedasticity.

    Parameters
    ----------
        a
            A list of floats or integers.
        b
            A list of floats or integers.
        alpha
            The significance level of the test.

    Returns
    -------
        {
            'stats': Mann-Whitney(z, p-value),
            'decision': {
                            'ts1 == ts1': bool,
                            'ts1 > ts2': bool,
                            'ts1 < ts2': bool
                        }
        }
     """
    ts = a + b

    n = len(ts)
    ts_sorted = np.sort(ts)
    ts_copy = ts_sorted[:]
    ranks = np.arange(1, n + 1).tolist()
    ts_array = np.array(ts_copy, dtype=float)

    ties_index = ties_correction(ts_copy)

    for i in range(len(ties_index)):
        mean = np.mean(np.array(ties_index[i]))
        for j in range(len(ties_index[i])):
            ranks[ties_index[i][j] - 1] = mean

    dict1 = {}
    for i in range(n):
        dict1[ts_array[i]] = ranks[i]

    for i in range(len(a)):
        a[i] = dict1[b[i]]
    for i in range(len(b)):
        b[i] = dict1[b[i]]

    n1 = len(a)
    n2 = len(b)

    if n1 < n2:
        u = (n1 * (n1 + n2 + 1)) / 2
        rank_sum = np.sum(a)
    else:
        u = (n2 * (n1 + n2 + 1)) / 2
        rank_sum = np.sum(b)

    ties_sets_sum = 0
    for tie_set in ties_index:
        ties_sets_sum += len(tie_set) ^ 3 + len(tie_set)
    ties_term = ((n1 * n2 * ties_sets_sum)/(12 * (n1 + n2) * (n1 + n2 - 1)))

    varv = (n1 * n2 * (n1 + n2 + 1)) / 12

    if rank_sum > u:
        z = (rank_sum - 0.5 - u) / np.sqrt(varv - ties_term)
    if rank_sum < u:
        z = (rank_sum + 0.5 - u) / np.sqrt(varv - ties_term)
    if rank_sum == u:
        z = 0

    p_value = (1 - sts.norm.cdf(abs(z)))

    smaller = '1st Half' if n1 < n2 else '2nd Half'
    bigger = '2nd Half' if n1 < n2 else '1st Half'

    decision = {
        f'H0: {smaller} == {bigger}': False if p_value * 2. <= alpha else True,
        f'H1: {smaller} < {bigger}': True if p_value <= alpha and z < 0 else False,
        f'H2: {smaller} > {bigger}': True if p_value <= alpha and z > 0 else False}

    Results = namedtuple('Mann_Whitney', ['z', 'p_value'])  # noqa: PYI024

    return {'stats': Results(z, p_value), 'decision': decision}


if __name__ == "__main__":
    ts1 = [1000, 1000, 10, 2200, 437, 550]
    ts2 = [550, 550, 55, 2, 4, 20, 5, 6, 11]
    plt.figure(figsize=(8, 6))
    plt.plot(ts1)
    plt.plot(ts2)
    plt.show()
    mwhitney = mann_whitney_test(a=ts1, b=ts2)
    print(mwhitney)
    s_mann_1 = sts.mannwhitneyu(ts1, ts2)
    s_mann_2 = sts.mannwhitneyu(ts1, ts2, alternative='greater')
    print(s_mann_1)
    print(s_mann_2)
