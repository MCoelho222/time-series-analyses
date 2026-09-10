"""Methods for array manipulation."""
from __future__ import annotations

from math import isfinite
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Iterable

    from numpy.typing import NDArray


def clean_numeric_array(values: Iterable[object]) -> NDArray[np.float64]:
    """Return finite numeric values from an arbitrary iterable."""
    numeric_values = []
    for value in values:
        try:
            numeric_value = float(value)
        except (TypeError, ValueError):
            continue
        if isfinite(numeric_value):
            numeric_values.append(numeric_value)

    if not numeric_values:
        msg = 'The time series contains no finite numeric values.'
        raise ValueError(msg)

    return np.asarray(numeric_values, dtype=np.float64)


def nans_nums_from_array(
    ps: NDArray[np.int64 | np.float64], *, only_nums: bool = True
) -> NDArray[np.int64 | np.float64] | tuple[
    NDArray[np.int64 | np.float64], NDArray[np.int64 | np.float64]
]:
    ps_mask = np.isnan(ps)
    ps_nums = ps[~ps_mask]

    if only_nums:
        return ps_nums

    ps_nan = ps[ps_mask]

    return ps_nums, ps_nan
