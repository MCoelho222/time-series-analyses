from __future__ import annotations

import numpy as np
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from rhis_timeseries.types.timeseries_types import TimeSeriesFlex


def raise_ts_diff_lengths(bw: TimeSeriesFlex, fw: TimeSeriesFlex):
    if len(bw) != len(fw):
        err_msg = "The parameters 'ts', 'bw', and 'fw' must be of the same length."
        raise ValueError(err_msg)

def raise_starts_rejected(is_reject_arr: np.ndarray[bool], direction: str):
    start = -1 if direction == 'bw' else 0
    if is_reject_arr[start]:
        err_msg = "This function should be called only when the evolution starts not rejected."
        raise ValueError(err_msg)