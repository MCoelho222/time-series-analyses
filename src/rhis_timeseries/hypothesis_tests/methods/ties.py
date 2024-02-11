"""Methods for ties correction."""
from __future__ import annotations

import numpy as np
from loguru import logger


def get_ties_index(ts: list[int | float], start: int) -> list[int]:
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


def ties_correction(ts: list[int | float] | np.ndarray[int | float]) -> list[list[int | float]]:
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
    logger.debug('Correcting ties...')
    ts_sorted = np.sort(ts)
    ties_index = []
    m = 0
    while m < len(ts) - 1:
        if m == 0: # Here the index n - 1 is not checked, as it doesn't exist
            tie_ind = get_ties_index(ts, m)
            ties_index.append(tie_ind)
        if m > 0:
            if ts_sorted[m] == ts_sorted[m + 1]:
                if ts_sorted[m] != ts_sorted[m - 1]: # indicates a new tie
                    new_index = m
                    new_tie_ind = get_ties_index(ts, new_index)
                    ties_index.append(new_tie_ind)
        m += 1
    return ties_index

if __name__ == "__main__":
    ties = [1, 1, 2, 3, 8, 23, 87, 55, 55, 1, 2, 4, 4]
    ties = np.array(ties)
    ties_correction = ties_correction(ties)

    print(ties_correction)
