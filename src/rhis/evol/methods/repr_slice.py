
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from rhis.utils.arrays import nans_nums_from_array

if TYPE_CHECKING:
    from rhis.custom_types.data import TimeSeriesFlex


def idx_of_last_not_rejected(alpha: float, ps: TimeSeriesFlex, direction: str, sli_init: int) -> int:
    data = ps[:]
    if direction == 'fo':
        data = ps[::-1]
    alpha_arr = np.full(len(data), alpha)
    idx = 0
    is_rejection = data <= alpha_arr

    while is_rejection[idx]:
        if idx == len(data) - 1:
            break
        idx += 1

    if direction == 'fo' and idx > 0:
        idx = len(ps) - idx + sli_init - 1

    return idx


def representative_slice_idxs(ps: np.ndarray[float], alpha: float, sli_init: int, backwards: bool = True) -> tuple[int]:  # noqa: FBT001, FBT002
    ps_nums = nans_nums_from_array(ps)

    ps_last = len(ps_nums) + sli_init - 1
    if ps[0] >= alpha:
        return (0, ps_last)

    return idx_of_last_not_rejected(alpha, ps_nums, backwards, sli_init), ps_last

