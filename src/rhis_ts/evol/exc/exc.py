from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np
    from pandas import DataFrame

    from rhis_ts.types.data import TimeSeriesFlex


def raise_ts_diff_lengths(ba: TimeSeriesFlex, fo: TimeSeriesFlex):
    if len(ba) != len(fo):
        err_msg = "The parameters 'ts', 'ba', and 'fo' must be of the same length."
        raise ValueError(err_msg)


def raise_starts_rejected(is_reject_arr: np.ndarray[bool], direction: str):
    start = -1 if direction == 'ba' else 0
    if is_reject_arr[start]:
        err_msg = "This function should be called only when the evolution starts not rejected."
        raise ValueError(err_msg)


def raise_no_evol_process_ran(evol_df: DataFrame | None):
    if evol_df is None:
        error_msg = "No evolution process has been performed yet."
        raise RuntimeError(error_msg)


def raise_incorrect_mode_raw(*, raw: bool):
    if raw:
        error_msg = "This function works only for 'median', 'mean' and 'min' modes only."
        raise ValueError(error_msg)
