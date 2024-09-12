"""Methods for ties correction."""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from rhis_ts.types.data import TimeSeriesFlex


def get_ties_index(ts: TimeSeriesFlex, start: int=0) -> list[int]:
    """
    --------------------------------------------------------------------
    Check if there equal numbers in sequence and get their ranks.
    --------------------------------------------------------------------
    Parameters
    ----------
        ts
            A list of integers or floats.

        start
            The index from where the verification should start checking.
    --------------------------------------------------------------------
    Returns
    -------
        A list with ranks where ties are present.
    --------------------------------------------------------------------
    """
    n = len(ts)
    ts_sorted = np.sort(ts)
    ranks = list(range(1, n + 1))

    tie_ranks = []
    i = start
    while ts_sorted[i] == ts_sorted[i + 1]:
        tie_ranks.append(ranks[i])
        i += 1
        if i == n - 1:
            break

    if len(tie_ranks) > 0:
        tie_ranks.append(tie_ranks[-1] + 1)

    return tie_ranks


def ranks_ties_corrected(ts: TimeSeriesFlex,*, ties_data: bool=False) \
      -> list[int | float] | dict[str, str | int]:  # noqa: C901
    """
    -----------------------------------------------------------------
    Apply correction for ties.
    -----------------------------------------------------------------
    Parameters
    ----------
        ts
            A list of integers or floats.

        ties_data
            If True, information about ties will be returned
    -----------------------------------------------------------------
    Returns
    -------
        A list with ranks corrected for ties, or a dictionary with
        information about ties, including the list with ranks. The
        ranks will be in the original time series order.
    -----------------------------------------------------------------
    """
    ranks = np.array(to_ranks(ts), dtype=float)

    ts_sorted = np.sort(ts)
    ties_index = []
    m = 0
    while m < len(ts) - 1:
        if m == 0:
            tie_ind = get_ties_index(ts, m)
            if len(tie_ind) > 0:
                ties_index.append(tie_ind)
        if m > 0:
            if ts_sorted[m] == ts_sorted[m + 1]:
                if ts_sorted[m] != ts_sorted[m - 1]:
                    new_index = m
                    new_tie_ind = get_ties_index(ts, new_index)
                    ties_index.append(new_tie_ind)
        m += 1

    if len(ties_index) > 0:

        for index in ties_index:
            mean = np.mean(index)
            for ind in index:
                ranks[ranks == ind] = mean

    if ties_data:
        ties_group_counts = []
        unique_ranks = np.unique(ranks)
        for rank in unique_ranks:
            count = np.count_nonzero(np.array(ranks) == rank)
            ties_group_counts.append(count if count > 0 else 1)

        ties_data = {
            'ranks': ranks,
            'ties_indexes': ties_index, # The indexes where ties are present.
            'ties_count': len(ties_group_counts), # How many groups of ties.
            'ties_groups_count': ties_group_counts, # How many elements in each tie group.
        }

        return ties_data

    return ranks


def to_ranks(ts: TimeSeriesFlex) -> TimeSeriesFlex:
    """
    ----------------------------------------------------------
    Transform the original series in a series of ranks.
    ----------------------------------------------------------
    Parameters
    ----------
        ts
            A list or array with numbers.
    ----------------------------------------------------------
    Return
    ------
        A list with the original data replaced by their ranks.
    ----------------------------------------------------------
    """
    ts_unique = np.unique(ts)
    ts_sorted = np.sort(ts)
    ranks_dict = {}
    for el in ts_unique:
        index = np.where(ts_sorted == el)[0]
        ranks_dict[el] = list(index)

    ranks = []
    for el in np.array(ts):
        rank = ranks_dict[el].pop(0)
        ranks.append(rank + 1)

    return ranks

