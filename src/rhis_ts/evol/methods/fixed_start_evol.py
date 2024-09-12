from __future__ import annotations

import numpy as np

from rhis_ts.evol.methods.raw_evol import rhis_evol_raw


def rhis_evol_fixed_start(ts: np.ndarray, alpha: float, sli_init: int,*, raw: bool, ba: bool=False) \
    -> list[float] | dict[list[float]]:
    evol = rhis_evol_raw(ts, alpha, sli_init)
    fill = np.full(sli_init - 1, np.nan)

    for hyp, ps in evol.items():
        evol[hyp] = np.append(ps[::-1], fill) if ba else np.append(fill, ps)

    if raw:
        return evol

    evol = np.min([ps for _, ps in evol.items()], axis=0, keepdims=True).ravel()

    return evol
