from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from rhis_timeseries.evolution.data import slices2evol
from rhis_timeseries.hypothesis_tests.non_parametric import NonParametric


def rhis_evol(slices: list[list[float | int]]) -> dict[str, list[float | int]]:
    """
    Calculate randomness, homogeneity, independence and stationarity (rhis) p-values for time series slices.

    The main purpose is to calculate rhis p-values for slices with increasing or decreasing length. The p-values
    of the different lengths will indicate where or with how much elements (from beginning to end or end to beginning)
    the series is no longer representative, due to the presence of some variability pattern, e.g., trends or seasonality.

    The first or last slice must have at least 5 elements to the tests to be performed.

    Example
    -------
        Slices with increasing length

            ts = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
            slices = [ts[:5], ts[:6], ts[:7], ts[:8], ts[:9], ts]

        Slices with decreasing length

            ts = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
            slices = [ts, ts[:9], ts[:8], ts[:7], ts[:6], ts[:5]]

    Parameters
    ----------
        slices
            A list with slices from another list with float or integers.

    Return
    ------
        A dictionary with rhis p-values

        Example:

            evol_rhis = {
                'randomness': [0.556, 0.265, 0.945, 0.159],
                'homogeneity': [0.112, 0.232, 0.284, 0.492],
                'independence': [0.253, 0.022, 0.248, 0.995],
                'stationarity': [0.534, 0.003, 0.354, 0.009],
            }
    """
    evol_rhis = {
        'randomness': [],
        'homogeneity': [],
        'independence': [],
        'stationarity': [],
    }

    for i in range(len(slices)):
        ts = slices[i]

        runs = NonParametric.runstest(ts)
        whit = NonParametric.mwhitney_test(ts)
        wald = NonParametric.waldwolf_test(ts)
        mann = NonParametric.mann_kendall_test(ts)

        rhis = {
            'randomness': runs,
            'homogeneity': whit,
            'independence': wald,
            'stationarity': mann
        }

        for key in evol_rhis.keys():
            p_value = rhis[key]['stats'][1]
            evol_rhis[key].append(p_value)

    return evol_rhis


if __name__ == '__main__':
    ts = [list(np.random.uniform(-10.0, 100.0, 80)), list(np.random.uniform(30.0, 200.0, 30))]  # noqa: NPY002
    ts1 = np.concatenate((ts[0], ts[1]))

    slices = slices2evol(ts1, 5)

    plt.figure(figsize=(8, 6))
    plt.plot(ts1)
    plt.title('Original timeseries')
    plt.tight_layout()
    plt.show()

    evol_rhis = rhis_evol(slices)

    plt.figure(figsize=(20, 6))
    hyps = ['randomness', 'homogeneity', 'independence', 'stationarity']
    for hyp in hyps:
        plt.plot(evol_rhis[hyp], label=hyp)

    plt.legend()
    plt.title('RHIS Evolution')
    plt.tight_layout()
    plt.show()
