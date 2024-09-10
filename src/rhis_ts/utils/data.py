from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from rhis_ts.errors.exceptions import handle_exc_msg

if TYPE_CHECKING:
    from pandas import DatetimeIndex

    from rhis_ts.types.data import TimeSeriesFlex


def break_list_in_equal_parts(ts: TimeSeriesFlex, parts: int) -> list[TimeSeriesFlex]:
    """
    Divide a series in multiple equal parts, as much as possible.

    Parameters
    ----------
        ts
            A list with items of any type.

    Return
    ------
        A list of lists with each part of the series respecting the original order.
    """
    size = len(ts)
    cut_index = size / parts

    if size % parts != 0:
        cut_index = int(np.ceil(cut_index))
    else:
        cut_index = int(cut_index)

    return [ts[cut_index * i: cut_index * (i + 1)] for i in range(parts)]


def slice_init(n: int):
    limit = 100
    return 10 if n > limit else 5


def slices_to_evol(ts: TimeSeriesFlex, init: int) -> list[list[int | float]]:
    """
    Break a flat list into a list of lists (2D).

    Each list will have an increasing number of elements.
    The element with index n will have one more element than element with index n-1.
    Parameters
    ----------
        ts
            A list with integers or floats.
    Returns
    -------
        A 2D list of lists. The lists are slices with increasing number of elements,
        so that the last is the complete timeseries.
    """
    start = init
    try:
        slices = []
        for i in range(len(ts) - (start - 1)):
            slices.append(ts[: i + start])

        return slices
    except TypeError as exc:
        return handle_exc_msg(exc, 'ts must be list | numpy.ndarray, start must be integer')


def calc_time_rng_of_slices(df_index: DatetimeIndex, slice_init: int):
    rng_start = df_index[0]
    rng_end = df_index[-1]
    total_delta = rng_end - rng_start
    t_rngs_ba = []
    t_rngs_fo = []

    for i in range(slice_init - 1, len(df_index)):
        t_delta = df_index[i] - rng_start
        t_delta_perc = t_delta.days / total_delta.days
        t_rngs_fo.append(round(t_delta_perc, 4))

        t_delta = rng_end - df_index[- (i + 1)]
        t_delta_perc = t_delta.days / total_delta.days
        t_rngs_ba.append(round(t_delta_perc, 4))

    return {
        'ba_t_rngs': t_rngs_ba[::-1],
        'fo_t_rngs': t_rngs_fo,
    }
