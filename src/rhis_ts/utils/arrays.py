"""Methods for array manipulation."""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from rhis_ts.types.data import TimeSeriesFlex

def nans_nums_from_array(ps: TimeSeriesFlex,*, only_nums: bool=True) -> TimeSeriesFlex:
    ps_mask = np.isnan(ps)
    ps_nums = ps[~ps_mask]

    if only_nums:
        return ps_nums

    ps_nan = ps[ps_mask]

    return ps_nums, ps_nan
