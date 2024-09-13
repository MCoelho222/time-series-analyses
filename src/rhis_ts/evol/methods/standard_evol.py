from __future__ import annotations

import numpy as np

from rhis_ts.evol.methods.raw_evol import rhis_evol_raw


def rhis_standard_evol(ts: np.ndarray, alpha: float, sli_init: int, stat: str,*, rhis: bool=False, ba: bool=False) \
    -> list[float] | dict[list[float]]:  # noqa: PLR0913
    evol = rhis_evol_raw(ts, alpha, sli_init)
    fill = np.full(sli_init - 1, np.nan)

    for hyp, ps in evol.items():
        evol[hyp] = np.append(ps[::-1], fill) if ba else np.append(fill, ps)

    if rhis:
        return evol

    stat_funcs = {'min': np.min, 'mean': np.mean, 'med': np.median, 'max': np.max}
    evol = stat_funcs[stat]([ps for _, ps in evol.items()], axis=0, keepdims=True).ravel()

    return evol
