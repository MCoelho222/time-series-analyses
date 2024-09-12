from __future__ import annotations

import numpy as np

from rhis_ts.evol.methods.raw_evol import rhis_evol_raw


def rhis_evol_flex_start(ts: np.ndarray, alpha: float, sli_init: int,*, raw: bool, ba: bool=False) -> list[float]:
    evol = rhis_evol_raw(ts, alpha, sli_init)
    check = np.min([evol['R'][0], evol['H'][0], evol['I'][0], evol['S'][0]])
    idx = 0
    if check <= alpha:
        data = ts[:]
        idx = 1

        while True:
            new_data = data[idx:]
            evol = rhis_evol_raw(new_data, alpha, sli_init)
            check_2 = np.min([evol['R'][0], evol['H'][0], evol['I'][0], evol['S'][0]])
            if check_2 > alpha:
                break
            idx +=1

    fill = np.full(idx + sli_init - 1, np.nan)

    for hyp, ps in evol.items():
        evol[hyp] = np.append(ps[::-1], fill) if ba else np.append(fill, ps)

    if raw:
        return evol

    evol = np.min([ps for _, ps in evol.items()], axis=0, keepdims=True).ravel()

    return evol

if __name__ == "__main__":
    ts = [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 4, 2, 5, 3, 10, 9, 9.5, 3.4, 5.7, 2.5, 7, 4.3, 11]
    # print(rhis_evol_flex_start(ts, 0.05, raw=True))
    print(rhis_evol_flex_start(ts[::-1], 0.05, 5, raw=True, ba=True))
