from __future__ import annotations

import numpy as np

from rhis_ts.stats.mann_kendall import mann_kendall
from rhis_ts.stats.mann_whitney import mann_whitney
from rhis_ts.stats.runs import wallismoore
from rhis_ts.stats.wald_wolfowitz import wald_wolfowitz
from rhis_ts.utils.data import slices_to_evol


def bafo_init(ts: np.ndarray, alpha: float, sli_init: int):
    slices = slices_to_evol(ts, sli_init)
    hyps = ['R', 'H', 'I', 'S']
    rhis_tests = [wallismoore, mann_whitney, wald_wolfowitz, mann_kendall]
    test_dict = dict(zip(hyps, rhis_tests))
    evol = {}

    for hyp in hyps:
        ps = []
        for sli in slices:
            ps.append(test_dict[hyp](sli, alpha=alpha).p_value)
        evol[hyp] = ps

    return evol


def bafo_weak_memo(ts: np.ndarray, alpha: float, sli_init: int,*, raw: bool, ba: bool=False):
    evol_and_init = bafo_init(ts, alpha, sli_init)
    evol = evol_and_init[0]
    check = np.min([evol['R'][0], evol['H'][0], evol['I'][0], evol['S'][0]])
    idx = 0
    if check <= alpha:
        data = ts[:]
        idx = 1

        while True:
            new_data = data[idx:]
            evol = bafo_init(new_data, alpha)[0]
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
    # print(bafo_weak_memo(ts, 0.05, raw=True))
    print(bafo_weak_memo(ts[::-1], 0.05, raw=False, ba=True))
