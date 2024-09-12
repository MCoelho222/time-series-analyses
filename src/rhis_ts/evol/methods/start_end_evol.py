from __future__ import annotations

import numpy as np

from rhis_ts.stats.utils.rhis import calculate_rhis


def restart_evol_on_rhis_reject(ts: np.ndarray, alpha, sli_init: int):
    fill = np.full(sli_init - 1, np.nan)
    p_evol = []
    p_evol.extend(fill)
    ts_repr_idxs = []

    start = 0
    end = start + sli_init
    while end <= len(ts) and start <= len(ts) - sli_init:
        sli = ts[start:end]
        pval = calculate_rhis(sli, alpha)
        p_evol.append(pval)
        if end == len(ts):
            ts_repr_idxs.append((start, end))
            break

        if pval <= alpha:
            ts_repr_idxs.append((start, end))
            remain = len(ts) - end
            if remain < sli_init:
                ts_repr_idxs.append((end, end + remain))
                p_evol.extend(fill[:remain])
                break

            p_evol.extend(fill)
            start =+ end
            end = start + sli_init
        else:
            end += 1
    return ts_repr_idxs, p_evol


if __name__ == "__main__":
    ts = [2, 2, 2, 2, 2, 1, 2.5, 6, 2.5, 3.3, 5, 10]

    print(restart_evol_on_rhis_reject(ts[::-1], 0.05))
