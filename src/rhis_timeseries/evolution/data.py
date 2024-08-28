"""Methods for evolution insights."""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from rhis_timeseries.error.exception import handle_exception_msg

if TYPE_CHECKING:
    from rhis_timeseries.types.timeseries_types import TimeSeriesFlex


def slice_init(n: int):
    limit = 100
    return 10 if n > limit else 5


def slices_to_evol(ts: TimeSeriesFlex) -> list[list[int | float]]:
    """
    --------------------------------------------------------------------------------
    Break a flat list into a list of lists (2D).

    Each list will have an increasing number of elements.
    The element with index n will have one more element than element with index n-1.
    --------------------------------------------------------------------------------
    Parameters
    ----------
        ts
            A list with integers or floats.
    --------------------------------------------------------------------------------
    Returns
    -------
        A 2D list of lists. The lists are slices with increasing number of elements,
        so that the last is the complete timeseries.
    --------------------------------------------------------------------------------
    """
    n = len(ts)
    start = slice_init(n)
    try:
        slices = []
        for i in range(len(ts) - (start - 1)):
            slices.append(ts[: i + start])

        return slices
    except TypeError as exc:
        return handle_exception_msg(exc, 'ts must be list | numpy.ndarray, start must be integer')


def is_evol_x_shaped(bw: TimeSeriesFlex, fw: TimeSeriesFlex) -> bool:
    cond1 = bw[0] < fw[0]
    cond2 = bw[-1] > fw[-1]
    # cond3 = bw[0] > fw[0]
    # cond4 = bw[-1] < fw[-1]

    # is_x_shaped1 = cond1 and cond4
    is_x_shaped2 = cond1 and cond2

    return is_x_shaped2


def find_intersections(bw: TimeSeriesFlex, fw: TimeSeriesFlex):
    n = len(bw)
    idxs =[]
    for i in range(n):
        if bw[i] > fw[i]:
            idxs.append(i)
            break
    for i in range(1, n + 1):
        if bw[-i] < fw[-i]:
            idxs.append(-i)
            break

    return idxs


def drop_nans_from_evol(bw: TimeSeriesFlex, fw: TimeSeriesFlex) -> TimeSeriesFlex:
    bw_isnan = np.isnan(bw)
    fw_isnan = np.isnan(fw)
    bw_floats = bw[~bw_isnan]
    fw_floats = fw[~fw_isnan]

    return {'bw': bw_floats, 'fw': fw_floats}


def ends_rejected_both_directions(bw: TimeSeriesFlex, fw: TimeSeriesFlex, alpha) -> bool:
    return bw[0] > alpha or fw[-1] > alpha


def starts_rejected(alpha: float, ps: TimeSeriesFlex, direction: str) -> bool:
    if direction == 'bw':
        return ps[-1] <= alpha
    else:
        return ps[0] <= alpha


def idx_of_last_not_rejected(alpha, ps: TimeSeriesFlex, direction: str):
    data = ps[:]
    if direction == 'fw':
        data = ps[::-1]
    alpha_arr = np.full(len(data), alpha)
    idx = 0
    is_rejection = data <= alpha_arr
    while is_rejection[idx]:
        if idx == len(data) - 1:
            break
        idx += 1

    if direction == 'fw':
        idx = len(ps) - idx

    return idx
