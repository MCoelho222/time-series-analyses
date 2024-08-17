from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from rhis_timeseries.evolution.rhis import rhis_evolution

if TYPE_CHECKING:
    from rhis_timeseries.types.timeseries_types import TimeSeriesFlex


def representative_slice(ts: TimeSeriesFlex, mode: str = 'median', start_slice: int = 10) -> TimeSeriesFlex:
    data = np.array(ts)

    rhis_evol = rhis_evolution(ts, mode, start_slice, bidirectional=True)

    key = f"rhis_{mode}"
    bw_pvalues = rhis_evol['backward'][key]
    fw_pvalues = rhis_evol['forward'][key]

    n_fill = start_slice - 1

    if isinstance(bw_pvalues, list):
        bw_pvalues = np.array(bw_pvalues)
    if isinstance(fw_pvalues, list):
        fw_pvalues = np.array(fw_pvalues)

    mask = bw_pvalues[:-n_fill][::-1] > fw_pvalues[:-n_fill]

    if mask[-1]:
        bw_fill = np.ones(n_fill)
    else:
        bw_fill = np.zeros(n_fill)

    if mask[0]:
        fw_fill = np.ones(n_fill)
    else:
        fw_fill = np.zeros(n_fill)

    mask = np.append(mask, bw_fill)
    mask = np.append(fw_fill, mask)

    last_index = len(data)

    cut_index = -1
    while mask[last_index + cut_index]:
        cut_index -= 1

    repr_data = data[cut_index:]
    repr_index = range(len(ts) - len(repr_data), len(ts))

    return repr_index, repr_data

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import pandas as pd

    ts = pd.read_csv('./data/BigSiouxAnnualQ.csv')['Y']

    repr_ts = representative_slice(np.array(ts))

    plt.figure()
    plt.scatter(range(len(ts)), ts, color='red')
    plt.scatter(repr_ts[0], repr_ts[1], color='green')
    plt.show()


