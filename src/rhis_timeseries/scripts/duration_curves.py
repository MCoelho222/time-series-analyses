from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np


def asc(ts, interval, q_ref):
    q_ref1 = float(q_ref)
    ts1 = np.array(ts)
    ts2 = ts1[ts1 > 0]

    try:
        ts3 = ts2[interval[0] - 1:interval[1] + 1]
    except TypeError:
        ts3 = ts2

    ts3_asc = np.sort(ts3)
    freq = np.ones(len(ts3_asc))
    cumul_freq = np.cumsum(freq)*(100./len(ts3_asc))

    try:
        perm = list(cumul_freq)
        refindex = perm.index(q_ref1)
        q = ts3_asc[refindex]

    except ValueError:
        dif = float(q_ref1) - cumul_freq
        positives = dif[dif > 0]
#        negatives = dif[dif < 0]
        index = len(positives) - 1
        q = (((q_ref1 - perm[index])*(ts3_asc[index + 1] - ts3_asc[index]))/
             (perm[index + 1] - perm[index])) + ts3_asc[index]

    return [cumul_freq, ts3_asc, q]

def dsc(ts, interval, q_ref):
    q_ref1 = float(q_ref)
    ts1 = np.array(ts)
    ts2 = ts1[ts1 == ts1**1]

    try:
        ts3 = ts2[interval[0] - 1:interval[1] + 1]
    except TypeError:
        ts3 = ts2

    ts3_asc = np.sort(ts3)
    ts3_dsc = ts3_asc[::-1]
    freq = np.ones(len(ts3_dsc))
    cumul_freq = np.cumsum(freq)*(100./len(ts3_dsc))

    try:
        perm = list(cumul_freq)
        refindex = perm.index(q_ref1)
        q = ts3_dsc[refindex]

    except ValueError:
        dif = float(q_ref) - cumul_freq
        positives = dif[dif > 0]
#        negatives = dif[dif < 0]
        index = len(positives) - 1
        q = (((q_ref1 - perm[index])*(ts3_dsc[index + 1] - ts3_dsc[index]))/
             (perm[index + 1] - perm[index])) + ts3_dsc[index]

    return [cumul_freq, ts3_dsc, q]

if __name__ == "__main__":

    ts = [10., 20., 30., 11., 5., 6., 33., 24., 15., 17.]
    result1 = asc(ts, ('all'), 95)
    result2 = dsc(ts, ('all'), 95)
    print(result1)
    print(result2)
    plt.figure()
    plt.plot(result1[0], result1[1])
    plt.plot(result2[0], result2[1])
    plt.show()