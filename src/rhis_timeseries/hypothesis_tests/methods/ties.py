"""Methods for ties correction."""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from loguru import logger

from rhis_timeseries.errors.exception import raise_timeseries_type_error

if TYPE_CHECKING:
    from rhis_timeseries.types.timeseries_types import TimeSeriesFlex


def get_ties_index(ts: TimeSeriesFlex, start: int=0) -> list[int | np.int32]:
    """
    Check if there equal numbers in sequence and get their ranks.

    Parameters
    ----------
        ts
            A list of integers or floats.
        start
            The index from where the verification should start checking.

    Returns
    -------
        A list with ranks where ties are present.
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


def ties_correction(ts: TimeSeriesFlex, ties_data: bool=False) -> list[list[int | float]]:
    """
    Apply correction for ties.

    Parameters
    ----------
        ts
            A list of integers or floats.

    Returns
    -------
        A 2D list with ranks where ties are present.
    """
    raise_timeseries_type_error(ts)

    logger.info('Checking for ties...')

    ts_sorted = np.sort(ts)
    ranks = np.arange(1, len(ts_sorted) + 1).tolist()

    ties_index = []

    m = 0
    while m < len(ts) - 1:
        if m == 0: # Here the index n - 1 is not checked, as it doesn't exist
            tie_ind = get_ties_index(ts, m)
            if len(tie_ind) > 0:
                ties_index.append(tie_ind)
        if m > 0:
            if ts_sorted[m] == ts_sorted[m + 1]:
                if ts_sorted[m] != ts_sorted[m - 1]: # indicates a new tie
                    new_index = m
                    new_tie_ind = get_ties_index(ts, new_index)
                    ties_index.append(new_tie_ind)
        m += 1

    if len(ties_index) > 0:
        logger.info('Applying correction for ties...')
        for i in range(len(ties_index)):
            mean = np.mean(np.array(ties_index[i]))
            for j in range(len(ties_index[i])):
                ranks[ties_index[i][j] - 1] = mean
        logger.info('Correction for ties complete.')
    else:
        logger.info('No ties present.')

    if ties_data:
        logger.info('Gathering ties data...')
        ties_group_counts = []
        unique_ranks = np.unique(ranks)
        for rank in unique_ranks:
            count = np.count_nonzero(np.array(ranks) == rank)
            ties_group_counts.append(count if count > 0 else 1)

        ties_data = {
            'ranks': ranks,
            'ties_indexes': ties_index,
            'ties_count': len(ties_group_counts),
            'ties_groups_count': ties_group_counts,
        }
        logger.info('Ties data complete.')

        return ties_data

    return ranks


if __name__ == "__main__":
    ties = [5, 1, 2.2, 2.2, 8, 8]
    ties = np.array(ties)
    ties_correction = ties_correction(ties, ties_data=True)

    print(ties_correction)
