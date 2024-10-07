from __future__ import annotations

from typing import TYPE_CHECKING

from rhis_ts.stats.utils.rhis import calculate_rhis
from rhis_ts.utils.data import slices_to_evol

if TYPE_CHECKING:
    import numpy as np


def rhis_evol_raw(ts: np.ndarray, alpha: float, sli_init: int) -> dict[list[float]]:
    slices = slices_to_evol(ts, sli_init)
    evol = {'R': [], 'H': [], 'I': [], 'S': []}

    for sli in slices:
        r, h, i, s = calculate_rhis(sli, alpha, min=False)
        evol['R'].append(r)
        evol['H'].append(h)
        evol['I'].append(i)
        evol['S'].append(s)

    return evol
