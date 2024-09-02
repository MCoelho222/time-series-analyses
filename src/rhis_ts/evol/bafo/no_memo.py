from __future__ import annotations

import numpy as np

from rhis_ts.stats.utils.rhis_calc import min_from_rhis


def rhis_evol_no_memo(ts: np.ndarray, alpha, sli_init: int):
    fill = np.full(sli_init - 1, np.nan)
    p_evol = []
    p_evol.extend(fill)
    ts_slis = []

    start = 0
    end = start + sli_init
    while end <= len(ts) and start <= len(ts) - sli_init:
        sli = ts[start:end]
        pval = min_from_rhis(sli, alpha)
        p_evol.append(pval)
        if end == len(ts):
            ts_slis.append((start, end))
            break

        if pval <= alpha:
            ts_slis.append((start, end))
            remain = len(ts) - end
            if remain < sli_init:
                ts_slis.append((end, end + remain))
                p_evol.extend(fill[:remain])
                break

            p_evol.extend(fill)
            start =+ end
            end = start + sli_init
        else:
            end += 1
    return ts_slis, p_evol


if __name__ == "__main__":
    ts = [2, 2, 2, 2, 2, 1, 2.5, 6, 2.5, 3.3, 5, 10]

    print(rhis_evol_no_memo(ts[::-1], 0.05))
