"""Methods for evolution insights."""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from rhis_ts.types.data import TimeSeriesFlex

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

    if direction == 'fo':
        idx = len(ps) - idx + sli_init

    return idx
