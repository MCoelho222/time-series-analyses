from __future__ import annotations

import numpy as np

from rhis.stats.utils.rhis import calculate_rhis
from rhis.utils.data import slices_to_evol


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


def rhis_standard_evol(ts: np.ndarray, alpha: float, sli_init: int, stat: str|None,*, backwards: bool=False) \
    -> list[float] | dict[list[float]]:
    evol = rhis_evol_raw(ts, alpha, sli_init)
    fill = np.full(sli_init - 1, np.nan)

    for hyp, ps in evol.items():
        evol[hyp] = np.append(ps[::-1], fill) if backwards else np.append(fill, ps)

    if stat is None:
        return evol

    stat_funcs = {'min': np.min, 'mean': np.mean, 'med': np.median, 'max': np.max}
    evol = stat_funcs[stat](list(evol.values()), axis=0, keepdims=True).ravel()

    return evol
