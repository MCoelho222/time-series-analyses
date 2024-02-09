"""Methods for evolution insights."""
from __future__ import annotations


def slices2evol(ts: list[int | float], start: int) -> list[list[int | float]]:
    """Break a flat list into a list of lists (2D).

    Each list will have an increasing number of elements.
    The element with index n will have one more element than element with index n-1.

    Parameters
    ----------
        ts
            A list with integers or floats.
    """
    slices = []
    for i in range(len(ts) - (start - 1)):
        slices.append(ts[:i + start])

    return slices
