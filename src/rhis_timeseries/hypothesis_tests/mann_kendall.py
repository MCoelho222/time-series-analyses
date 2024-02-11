from __future__ import annotations

from collections import namedtuple
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as sts

if TYPE_CHECKING:
    from rhis_timeseries.types.hypothesis_types import TestResults


def mann_kendall_test(ts: list[int|float] | np.ndarray[int|float]) -> TestResults:

    n = len(ts)

    signs = []
    for i in range(n - 1):
        s = ts - ts[i]
        signs.extend(np.sign(s[i + 1:]))

    signs_array = np.array(signs)

    test_s = float(len(signs_array[signs_array > 0]) - len(signs_array[signs_array < 0]))

    sigma = ((n/18.)*(n - 1.)*(2.*n + 5.))**0.5

    condition_value = 0.

    if test_s > condition_value:
        z = abs((test_s - 1.)/sigma)
    if test_s == condition_value:
        z = condition_value
    if test_s < condition_value:
        z = abs((test_s + 1.)/sigma)

    p = 2*(1 - sts.norm.cdf(z))

    Results = namedtuple('Mann_Kendall', ['z', 'p_value'])  # noqa: PYI024

    return Results(z, p)


if __name__ == "__main__":

    ts = np.random.randint(0, 100, 100)
    print(ts)
    print(mann_kendall_test(ts))

    plt.figure()
    plt.scatter(np.arange(len(ts)), ts)
    plt.show()
