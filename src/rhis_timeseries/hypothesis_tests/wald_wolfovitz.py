from __future__ import annotations

from collections import namedtuple

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as sts


def waldwolf_test(ts):
    ts1 = np.array(ts)
    ts2 = ts1[ts1 == ts1**1]
    if len(ts2) == 0:
        return np.nan, np.nan, np.nan
    if len(ts2) > 0:
        n = len(ts2)
        ts_incr = np.sort(ts2)
        ts_ind = [] # copy of original time series, list
        index = [] # 1, 2, 3, ..., n
        for i in range(n):
            ts_ind.append(ts2[i])
            index.append(i + 1.)
        ts_index = np.array(ts_ind) # copy of original time series, array
        ties_index = []
        m = 0

        while m < n - 1:
            if m == 0:
                tie_ind = []
                k = 0
                try:
                    while ts_incr[k] == ts_incr[k + 1]:
                        tie_ind.append(index[k])
                        k = k + 1
                        if k == n - 1:
                            break
                    tie_ind.append(tie_ind[-1] + 1.)
                    ties_index.append(tie_ind)
                except:
                    pass
            if m > 0:
                if ts_incr[m] == ts_incr[m + 1]:
                    if ts_incr[m] != ts_incr[m - 1]:
                        tie_indb = []
                        l = m
                        try:
                            while ts_incr[l] == ts_incr[l + 1]:
                                tie_indb.append(index[l])
                                l = l + 1
                            tie_indb.append(tie_indb[-1] + 1.)
                            ties_index.append(tie_indb)
                        except:
                            tie_indb.append(tie_indb[-1] + 1.)
                            ties_index.append(tie_indb)
            m = m + 1

        for i in range(len(ties_index)):
            mean = np.mean(np.array(ties_index[i]))
            for j in range(len(ties_index[i])):
                index[int(ties_index[i][j]) - 1] = mean
        dict1 = {}
        for i in range(n):
            dict1[ts_incr[i]] = index[i]
        for i in range(n):
            ts_index[i] = dict1[ts_index[i]]
        ts4 = ts_index - np.mean(ts_index)
        n1 = float(len(ts4))
        r = np.sum(ts4[:-1]*ts4[1:]) + ts4[0]*ts4[-1]
        m1 = float(np.sum(ts4)/n1)
        m2 = float(np.sum(ts4**2)/n1)
        m3 = float(np.sum(ts4**3)/n1)
        m4 = float(np.sum(ts4**4)/n1)
        s1 = n1*m1
        s2 = n1*m2
        s3 = n1*m3
        s4 = n1*m4
        e_r = (s1**2 - s2)/(n1 - 1)
        a = (s2**2 - s4)/(n1 - 1)
        try:
            b = (s1**4 - 4*s1**2*s2 + 4*s1*s3 + s2**2 - 2*s4)/((n1 - 1)*(n1 - 2))
        except ZeroDivisionError:
            b = (s1**4 - 4*s1**2*s2 + 4*s1*s3 + s2**2 - 2*s4)/0.00001
        c = (s1**2 - s2)**2/(n1 - 1)**2
        var_r = a + b - c

        Results = namedtuple('Wald_Wolfovitz', ['z', 'p_value'])

        if var_r > 0:
            z = abs((r - e_r)/np.sqrt(var_r))
            p = 2*(1 - sts.norm.cdf(z))

            if z < 1.96:
                decision = -1
            if z >= 1.96:
                decision = 1

            return {'stats': Results(z, p), 'decision': decision}

        if var_r == 0 or var_r < 0:
            decision = 1
            return {'stats': Results(np.nan, 0.0), 'decision': decision}

if __name__ == "__main__":

    ts = np.random.randint(0, 100, 100)
    print(ts)
    print(waldwolf_test(ts))

    plt.figure()
    plt.scatter(np.arange(len(ts)), ts)
    plt.show()
