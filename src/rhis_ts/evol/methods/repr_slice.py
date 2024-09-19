
from __future__ import annotations

from typing import TYPE_CHECKING

from rhis_ts.evol.utils.ba_fo import idx_of_last_not_rejected
from rhis_ts.utils.arrays import nans_nums_from_array

if TYPE_CHECKING:
    import numpy as np

def repr_slice_idxs(ps: np.ndarray[float], alpha: float, sli_init: int, direction: str) -> tuple[int]:
    ps_nums_and_nans = nans_nums_from_array(ps)
    ps_nums = ps_nums_and_nans

    ps_last = len(ps_nums) + sli_init - 1
    if ps[0] >= alpha:
        return (0, ps_last)

    return idx_of_last_not_rejected(alpha, ps_nums, direction, sli_init), ps_last

