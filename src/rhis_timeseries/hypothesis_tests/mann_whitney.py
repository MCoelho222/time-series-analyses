from __future__ import annotations

from collections import namedtuple
from typing import TYPE_CHECKING

import numpy as np
import scipy.stats as sts

from rhis_timeseries.hypothesis_tests.decorators.hypothesis_test import check_test_args
from rhis_timeseries.hypothesis_tests.methods.ranks import ranks_ties_corrected
from rhis_timeseries.utils.timeseries import break_list_in_equal_parts

if TYPE_CHECKING:
    from rhis_timeseries.types.hypothesis_types import TestResults


@check_test_args('mann-whitney')
def mann_whitney(  # noqa: PLR0913
        x: list[int | float],
        y: list[int | float] | None = None,
        alternative: str='two-sided',
        alpha: float=0.05,
        *,
        continuity: bool=True,
        ties: bool=True,
        ) -> TestResults:
    """
    ----------------------------------------------------------------------
    Compare two independent groups of data using the Mann-Whitney U test.

    This implementation applies the large sample approximation (applicable
    if x, y > 10 elements). The Mann-Whitney test is also known as The
    Rank-Sum test or Wilcoxon Rank-Sum.

    Assumptions

        - Each sample has been randomnly selected from the population it
          represents.
        - The two samples are independent of one another.
        - The original variable observed (which is subsequently ranked) is
          a continuous random variable.
        - The underlying distributions from which the samples are derived
          are identical in shape.

    Null and Alternative Hypotheses

        H0: prob[x > y] = 0.5

        H1: prob[ x > y] != 0.5 (two-sided)
        H2: prob[ x > y] > 0.5 (greater)
        H3: prob[ x > y] < 0.5 (less)
    ----------------------------------------------------------------------
    References
    ----------
        HELSEL & HIRSCH (2002). Techniques of Water Resources
        investigations fo the United States Geological Survey.Chapter 5 -
        Statistical Methods in Water Resources.
        Source: https://pubs.usgs.gov/twri/twri4a3/twri4a3.pdf
    ----------------------------------------------------------------------
    Parameters
    ----------
        x
            A list of floats or integers.

        y
            A list of floats or integers.

        alternative
            two-sided: x != y
            greater: x > y
            less: x < y

        alpha
            The significance level (0.05 by default).

        continuity
            If True, applies correction for continuity.

        ties
            If True, applies correction for ties.
    ----------------------------------------------------------------------
    Returns
    -------
        A namedtuple
            ('MannWhitney', ['statistic', 'p_value', 'reject'])

            The parameter 'reject' is of type bool. 'True' means the null
            hypothesis was reject.
    ----------------------------------------------------------------------
    """
    if y is None:
        data = break_list_in_equal_parts(x, 2)
        x = data[0]
        y = data[1]

    g1 = x[:] if isinstance(x, list) else x[:].tolist()
    g2 = y[:] if isinstance(y, list) else y[:].tolist()

    gs_concat = g1 + g2
    gs_sorted = np.sort(gs_concat)

    n = len(gs_concat)
    ranks = np.sort(ranks_ties_corrected(gs_concat)) if ties else [ i + 1 for i in range(n) ]

    ranks_dict = dict(zip(gs_sorted, ranks))
    g1_ranks = [ ranks_dict[value] for value in g1 ]
    g2_ranks = [ ranks_dict[value] for value in g2 ]

    rank_sum1 = sum(g1_ranks)
    rank_sum2 = sum(g2_ranks)

    n1 = len(g1)
    n2 = len(g2)
    u1 = n1 * n2 + (n1 * (n1 + 1)) / 2 - rank_sum1
    u2 = n1 * n2 + (n2 * (n2 + 1)) / 2 - rank_sum2

    stat = min(u1, u2)

    mean_stat = (n1 * n2) / 2
    var = (n1 * n2 * (n1 + n2 + 1)) / 12
    if ties:
        var = ((n1 * n2) / ((n) * (n - 1))) * np.sum(np.array(ranks) ** 2) \
            - ((n1 * n2 * (n + 1) ** 2) / (4 * (n - 1)))

    z = abs(stat - mean_stat) / np.sqrt(var)

    if continuity:
        z = (abs(stat - mean_stat) - 0.5) / np.sqrt(var)

    p = (1 - sts.norm.cdf(z))

    if alternative == 'two-sided':
        p = p * 2
        reject = p < alpha

    if alternative == 'less':
        reject = rank_sum1 < rank_sum2 and p < alpha

    if alternative == 'greater':
        reject = rank_sum1 > rank_sum2 and p < alpha

    Results = namedtuple('MannWhitney', ['statistic', 'p_value', 'reject', 'alternative'])  # noqa: PYI024

    return Results(stat, round(p, 4), reject, alternative)
