"""Methods for evolution insights."""
from __future__ import annotations

import numpy as np

from rhis_timeseries.errors.exception import handle_exception_msg


def slices2evol(ts: list[int | float] | np.ndarray, start: int) -> list[list[int | float]]:
    """Break a flat list into a list of lists (2D).

    Each list will have an increasing number of elements.
    The element with index n will have one more element than element with index n-1.

    Parameters
    ----------
        ts
            A list with integers or floats.
        start
            The number of elements in the first slice.

    Returns
    -------
        A 2D list of lists. The lists are slices with increasing number of elements,
        so that the last is the complete timeseries.
    """
    try:
        slices = []
        for i in range(len(ts) - (start - 1)):
            slices.append(ts[: i + start])

        return slices
    except TypeError as exc:
        return handle_exception_msg(exc, 'ts must be list | numpy.ndarray, start must be integer')
