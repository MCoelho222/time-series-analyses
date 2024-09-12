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


def raise_raw_evol_not_performed(df: DataFrame,*, use_raw: bool):
    if use_raw and df is None:
        err_msg = "No evolution process ran in raw mode, use_raw should be False."
        raise ValueError(err_msg)


def raise_starts_rejected(is_reject_arr: np.ndarray[bool], direction: str):
    start = -1 if direction == 'ba' else 0
    if is_reject_arr[start]:
        err_msg = "This function should be called only when the evolution starts not rejected."
        raise ValueError(err_msg)


def raise_start_end_evol_not_performed(*, start_end_evol: bool):
    if not start_end_evol:
        error_msg = "You need to run the evolution process with start_end_evol=True."
        raise ValueError(error_msg)