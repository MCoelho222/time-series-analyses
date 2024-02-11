from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from rhis_timeseries.evolution.data import slices2evol
from rhis_timeseries.hypothesis_tests.non_parametric import NonParametric


def rhis_evol(slices):
    """"""
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
    plt.title('Stationarity Evolution')
    plt.tight_layout()
    plt.show()
