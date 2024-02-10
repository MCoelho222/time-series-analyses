"""Methods to provide statistical data for evolution plots."""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from rhis_timeseries.evolution.data import slices2evol


def outliers_summary_evol(slices: list[list[int | float]]):
    evol_data = {
        'non_outlier_inf_lim': [],
        'non_outlier_sup_lim': [],
        'min_non_outlier': [],
        'max_non_outlier': [],
        'number_inf_outliers': [],
        'number_sup_outliers': [],
    }
    for i in range(len(slices)):
        q1 = np.percentile(slices[i], 25)
        q3 = np.percentile(slices[i], 75)
        iqr = q3 - q1

        evol_data['non_outlier_inf_lim'].append(q1 - 1.5 * iqr)
        evol_data['non_outlier_sup_lim'].append(q3 + 1.5 * iqr)

        asc = np.sort(slices[i])
        evol_data['min_non_outlier'].append(asc[asc >= (q1 - 1.5 * iqr)][0])
        evol_data['max_non_outlier'].append(asc[asc <= (q3 + 1.5 * iqr)][-1])

        evol_data['number_inf_outliers'].append(len(slices[i][slices[i] < (q1 - 1.5 * iqr)]))
        evol_data['number_sup_outliers'].append(len(slices[i][slices[i] > (q3 + 1.5 * iqr)]))

    return evol_data


if __name__ == '__main__':
    ts = [list(np.random.uniform(-10.0, 100.0, 80)), list(np.random.uniform(30.0, 200.0, 30))]  # noqa: NPY002
    ts1 = np.concatenate((ts[0], ts[1]))

    slices = slices2evol(ts1, 5)

    plt.figure(figsize=(8, 6))
    plt.plot(ts1)
    plt.title('Original timeseries')
    plt.tight_layout()
    plt.show()


    plt.figure(figsize=(20, 6))
    # for hyp in hyps:
    #     plt.plot(evol_rhis[hyp], label=hyp)

    plt.legend()
    plt.title('Stationarity Evolution')
    plt.tight_layout()
    plt.show()


