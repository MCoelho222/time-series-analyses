from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from rhis_timeseries.types.timeseries_types import TimeSeriesFlex


def break_list_equal_parts(ts: TimeSeriesFlex, parts: int) -> list[TimeSeriesFlex]:
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

