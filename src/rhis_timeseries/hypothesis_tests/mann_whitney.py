from __future__ import annotations

from collections import namedtuple

import numpy as np
import scipy.stats as sts


def mwhitney_test(ts, alpha=0.05):
    """-------------------------------------------------------------
    OBJECTIVE => It compares two groups of data
    ----------------------------------------------------------------
    PARAMS
    ----------------------------------------------------------------
    ts      => dict or list;
              {ts1: [float1, float2,...], ts2: [float1, float2,...]}
              or [float1, float2,...]

    alpha   => float; the significance level of the test
    ----------------------------------------------------------------
    RETURN => dict;
    ----------------------------------------------------------------
    {
      'stats': Mann-Whitney(z, p-value),
      'decision': {
                    'ts1 == ts1': bool,
                    'ts1 > ts2': bool,
                    'ts1 < ts2': bool
                   }
     }
    ------------------------------------------------------------ """
    data = ts
    if isinstance(ts, dict):
        ts_keys = list(ts.keys())
        ts1 = ts[ts_keys[0]]
        ts2 = ts[ts_keys[1]]
        data = ts1 + ts2

    n = len(data)
    ts_ = np.sort(data)
    ts_copy = [element for element in ts_]
    index = [i for i in np.arange(1, n + 1)]
    ts_array = np.array(ts_copy, dtype=float)
    ties_index = []
    m = 0
    while m < n - 1:
        # For repeated values starting at index 0
        if m == 0:
            tie_ind = []
            k = 0
            try:
                while ts_[k] == ts_[k + 1]:
                    tie_ind.append(index[k])
                    k += 1
                    if k == n - 1:
                        break
                # will raise an error if len(tie_ind) = 0
                tie_ind.append(tie_ind[-1] + 1)
                ties_index.append(tie_ind)
            except IndexError:
                pass
        if m > 0:
            if ts_[m] == ts_[m + 1]:
                if ts_[m] != ts_[m - 1]:
                    tie_indb = []
                    l = int(m/1)
                    try:
                        while ts_[l] == ts_[l + 1]:
                            tie_indb.append(index[l])
                            l += 1
                            if l == n - 1:
                                break
                        tie_indb.append(tie_indb[-1] + 1)
                        ties_index.append(tie_indb)
                    except IndexError:
                        pass

        m += 1
    for i in range(len(ties_index)):
        mean = np.mean(np.array(ties_index[i]))
        for j in range(len(ties_index[i])):
            index[ties_index[i][j] - 1] = mean
    dict1 = {}
    for i in range(n):
        dict1[ts_array[i]] = index[i]
    if isinstance(ts, dict):
        ts_1_list = []
        ts_2_list = []
        for i in range(len(ts1)):
            ts_1_list.append(dict1[ts1[i]])
        for i in range(len(ts2)):
            ts_2_list.append(dict1[ts2[i]])
        ts_1 = np.array(ts_1_list)
        ts_2 = np.array(ts_2_list)

    if isinstance(ts, list) or isinstance(ts, np.ndarray):
        for i in range(n):
            ts_array[i] = dict1[ts[i]]

        cut = int(n / 2)
        ts_1 = ts_array[:cut]
        ts_2 = ts_array[cut:]

    n1 = len(ts_1)
    n2 = len(ts_2)

    if n1 < n2:
        u = (n1 * (n1 + n2 + 1)) / 2
        rank_sum = np.sum(ts_1)
    else:
        u = (n2 * (n1 + n2 + 1)) / 2
        rank_sum = np.sum(ts_2)

    ties_sets_sum = 0
    for tie_set in ties_index:
        ties_sets_sum += len(tie_set) ^ 3 + len(tie_set)
    ties_term = ((n1 * n2 * ties_sets_sum)/(12 * (n1 + n2) * (n1 + n2 - 1)))

    varv = (n1 * n2 * (n1 + n2 + 1)) / 12

    if rank_sum > u:
        z = (rank_sum - 0.5 - u) / np.sqrt(varv - ties_term)
    if rank_sum < u:
        z = (rank_sum + 0.5 - u) / np.sqrt(varv - ties_term)
    if rank_sum == u:
        z = 0

    p = (1 - sts.norm.cdf(abs(z)))

    if isinstance(ts, dict):
        smaller = ts_keys[0] if n1 < n2 else ts_keys[1]
        bigger = ts_keys[1] if n1 < n2 else ts_keys[0]
    else:
        smaller = '1st Half' if n1 < n2 else '2nd Half'
        bigger = '2nd Half' if n1 < n2 else '1st Half'

    decision = {
        f'H0: {smaller} == {bigger}': False if p * 2. <= alpha else True,
        f'H1: {smaller} < {bigger}': True if p <= alpha and z < 0 else False,
        f'H2: {smaller} > {bigger}': True if p <= alpha and z > 0 else False}

    Results = namedtuple('Mann-Whitney', ['z', 'p_value'])

    return {'stats': Results(z, p), 'decision': decision}


if __name__ == "__main__":

    print(mwhitney_test([1000, 1000, 10, 2200, 437, 550, 550, 550, 55, 2, 4, 20, 5, 6, 11]))

    print(mwhitney_test({'A': [1000, 1000, 10, 2200, 437, 550, 550, 550],
                         'B': [55, 2, 4, 20, 5, 6, 11]}))
