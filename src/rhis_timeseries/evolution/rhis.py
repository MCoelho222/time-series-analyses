from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np

from rhis_timeseries.evolution.data import slices_incr_len
from rhis_timeseries.hypothesis_tests.mann_kendall import mann_kendall
from rhis_timeseries.hypothesis_tests.mann_whitney import mann_whitney
from rhis_timeseries.hypothesis_tests.runs import runs_test
from rhis_timeseries.hypothesis_tests.wald_wolfowitz import wald_wolfowitz

if TYPE_CHECKING:
    from rhis_timeseries.types.timeseries_types import TimeSeriesFlex


def rhis_evolution(
        ts: TimeSeriesFlex,
        mode: str = 'raw',
        slice_start: int = 10,*,
        forward: bool = False,
        bidirectional: bool = False,
        ) -> dict[str, list[float | int]]:
    """
    ---------------------------------------------------------------------------
    Calculate randomness, homogeneity, independence and stationarity (rhis)
    p-values for time series slices.

    The main purpose is to calculate rhis p-values for slices with increasing
    or decreasing length. The p-values of the different lengths will indicate
    where or with how much elements (from beginning to end or end to beginning)
    the series is no longer representative, due to the presence of some
    variability pattern, e.g., trends or seasonality.

    The first or last slice must have at least 5 elements to the tests to be
    performed.

    Example
    -------
        Slices with increasing length

            ts = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
            slices = [ts[:5], ts[:6], ts[:7], ts[:8], ts[:9], ts]

        Slices with decreasing length

            ts = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
            slices = [ts, ts[:9], ts[:8], ts[:7], ts[:6], ts[:5]]
    ---------------------------------------------------------------------------
    Parameters
    ----------
        slices
            A list with slices from another list with float or integers.
    ---------------------------------------------------------------------------
    Return
    ------
        A dictionary with rhis p-values

        Example:

            rhis_evol = {
                'r': [0.556, 0.265, 0.945, 0.159],
                'h': [0.112, 0.232, 0.284, 0.492],
                'i': [0.253, 0.022, 0.248, 0.995],
                's': [0.534, 0.003, 0.354, 0.009],
            }
    ---------------------------------------------------------------------------
    """
    direction = 'backward'
    data = ts[::-1]
    if not bidirectional and forward:
        data = ts[:]
        direction = 'forward'

    slices = slices_incr_len(data, slice_start)
    all_slices = [slices]
    directions = [direction,]

    result = {}
    if bidirectional:
        fw_slices = slices_incr_len(ts, slice_start)
        all_slices.append(fw_slices)
        directions.append('forward')

    hyps = ['R', 'H', 'I', 'S']
    for i in range(len(all_slices)):
        ps = [[], [], [], []]
        for ts_slice in all_slices[i]:
            ps[0].append(runs_test(ts_slice).p_value)
            ps[1].append(mann_whitney(ts_slice).p_value)
            ps[2].append(wald_wolfowitz(ts_slice).p_value)
            ps[3].append(mann_kendall(ts_slice).p_value)

        if mode == 'raw':
            rhis_evol = dict(zip(hyps, ps))
        if mode == 'median':
            rhis_evol = dict(zip(['rhis_median',], np.median(np.array(ps), axis=0, keepdims=True)))
        if mode == 'mean':
            rhis_evol = dict(zip(['rhis_mean',], np.mean(np.array(ps), axis=0, keepdims=True)))

        result[directions[i]] = rhis_evol

    return result # TODO (Marcelo): make it also return the start and end index of the series  # noqa: TD003


if __name__ == '__main__':
    import pandas as pd

    ts = pd.read_csv('./data/BigSiouxAnnualQ.csv')['Y']
    ts1 = np.array(ts)
    rng = np.random.default_rng(seed=30)

    # ts = [list(rng.uniform(-10.0, 100.0, 80)), list(rng.uniform(30.0, 200.0, 30))]
    # ts1 = np.concatenate((ts[0], ts[1]))

    slices = slices_incr_len(ts1, 5)
    plt.figure(figsize=(8, 6))
    plt.plot(ts1)
    plt.title('Original timeseries')
    plt.tight_layout()
    plt.show()

    rhis_evol = rhis_evolution(ts1, 'median', bidirectional=True)
    rhis_evol_bw = rhis_evol['backward']
    rhis_evol_fw = rhis_evol['forward']

    plt.figure(figsize=(12, 6))
    hyps = ['R', 'H', 'I', 'S']
    hyps = ['rhis_median',]
    for hyp in hyps:
        plt.plot(rhis_evol_bw[hyp][::-1], label='backward')
        plt.plot(rhis_evol_fw[hyp], label='forward')

    plt.legend()
    plt.title('RHIS Evolution')
    plt.tight_layout()
    plt.show()
