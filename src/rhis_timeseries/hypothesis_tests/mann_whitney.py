from __future__ import annotations

from collections import namedtuple
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as sts

from rhis_timeseries.errors.exception import raise_timeseries_type_error
from rhis_timeseries.hypothesis_tests.methods.p_value import p_value_normal
from rhis_timeseries.hypothesis_tests.methods.ties import ties_correction

if TYPE_CHECKING:
    from rhis_timeseries.types.hypothesis_types import TestResults


def mann_whitney_u(a: list[int | float], b: list[int | float], alternative: str = 'two-sided') \
    -> TestResults:
    """
    Compare two independent groups of data.

    The Mann-Whitney test is also known as The Rank-Sum test or Wilcoxon Rank-Sum.
    This test assumes homoscedasticity and independent groups.

    Parameters
    ----------
        a
            A list of floats or integers.
        b
            A list of floats or integers.
        alternative
            two-sided: a!=b
            greater: a > b
            less: a < b
    Returns
    -------
        Mann-Whitney(statistic, p-value). This p-value refers to a two-sided test.
     """
    raise_timeseries_type_error(a)
    raise_timeseries_type_error(b)

    ts = a + b
    ts_sorted = np.sort(ts)

    n = len(ts)
    n1 = len(a)
    n2 = len(b)

    ranks = np.arange(1, n + 1).tolist()

    updated_ranks = ties_correction(ts_sorted)

    dict1 = {}
    for i in range(n):
        dict1[ts_sorted[i]] = updated_ranks[i]
    for i in range(len(a)):
        a[i] = dict1[a[i]]
    for i in range(len(b)):
        b[i] = dict1[b[i]]

    if n1 < n2:
        mean = (n1 * (n1 + n2 + 1)) / 2
        rank_sum = np.sum(a)
    else:
        mean = (n2 * (n1 + n2 + 1)) / 2
        rank_sum = np.sum(b)

    square_sum_ranks = np.sum(np.array(updated_ranks) ** 2)

    var_no_ties = (n1 * n2 * (n1 + n2 + 1)) / 12

    var_ties = ((n1 * n2) / ((n1 + n2) * (n1 + n2 - 1))) * square_sum_ranks \
        - ((n1 * n2 * (n1 + n2 + 1) ** 2) / (4 * (n1 + n2 - 1)))

    var = var_no_ties if updated_ranks == ranks else var_ties

    if rank_sum > mean:
        stat = (rank_sum - 0.5 - mean) / np.sqrt(var)
    if rank_sum < mean:
        stat = (rank_sum + 0.5 - mean) / np.sqrt(var)
    if rank_sum == mean:
        stat = 0

    p_value = p_value_normal(stat, alternative)

    Results = namedtuple('Mann_Whitney', ['statistic', 'p_value', 'alternative'])  # noqa: PYI024

    return Results(stat, p_value, alternative)


if __name__ == "__main__":
    """
    Example from the book Statistical Methods in Water Resources

    Auhtor: Helsel & Hirsch
    Year: 2002
    Source: https://pubs.usgs.gov/twri/twri4a3/twri4a3.pdf

    Chapter 5 - Differences between two independent groups
    """
    ts1 = [0.59, 0.87, 1.1, 1.1, 1.2, 1.3, 1.6, 1.7, 3.2, 4.0]
    ts2 = [0.3, 0.36, 0.5, 0.7, 0.7, 0.9, 0.92, 1., 1.3, 9.7]

    plt.figure(figsize=(8, 6))
    plt.title('Nitrogen concentration in precipitation (mg/L)')
    plt.plot(ts1, label='industrial site')
    plt.plot(ts2, label='residential site')
    plt.legend()
    plt.show()

    mwhitney_less = mann_whitney_u(a=ts1, b=ts2, alternative='less')
    mwhitney_greater = mann_whitney_u(a=ts1, b=ts2, alternative='greater')
    mwhitney_2sided = mann_whitney_u(a=ts1, b=ts2, alternative='two-sided')
    scipy_mwhitney_less = sts.mannwhitneyu(ts1, ts2, alternative='less')
    scipy_mwhitney_greater = sts.mannwhitneyu(ts1, ts2, alternative='greater')
    scipy_mwhitney_2sided = sts.mannwhitneyu(ts1, ts2, alternative='two-sided')
    print('LESS')
    print()
    print(mwhitney_less)
    print(scipy_mwhitney_less)
    print()
    print('GREATER')
    print()
    print(mwhitney_greater)
    print(scipy_mwhitney_greater)
    print()
    print('TWO-SIDED')
    print()
    print(mwhitney_2sided)
    print(scipy_mwhitney_2sided)
