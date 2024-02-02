from __future__ import annotations

from collections import namedtuple

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as sts


def mann_kendall_test(tseries):
    y1 = np.array(tseries)

    y2 = y1[y1 == y1**1]
    n = len(y2)

    signs = []

    for i in range(n - 1):
        s = y2 - y2[i]
        signs.extend(np.sign(s[i + 1:]))
    signs1 = np.array(signs)
    S = float(len(signs1[signs1 > 0]) - len(signs1[signs1 < 0]))
    sigma = ((n/18.)*(n - 1.)*(2.*n + 5.))**0.5

    if S > 0.:
        z = abs((S - 1.)/sigma)
    if S == 0.:
        z = 0.
    if S < 0.:
        z = abs((S + 1.)/sigma)

    p = 2*(1 - sts.norm.cdf(z))

    if z < 1.96:
        decision = -1
    if z >= 1.96:
        decision = 1

    Results = namedtuple('Mann_Kendall', ['z', 'p_value'])
    return {'stats': Results(z, p), 'decision': decision}


if __name__ == "__main__":

    ts = np.random.randint(0, 100, 100)
    print(ts)
    print(mann_kendall_test(ts))

    plt.figure()
    plt.scatter(np.arange(len(ts)), ts)
    plt.show()
