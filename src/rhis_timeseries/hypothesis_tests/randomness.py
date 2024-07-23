from __future__ import annotations

from collections import namedtuple
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np

from rhis_timeseries.errors.exception import raise_timeseries_type_error
from rhis_timeseries.hypothesis_tests.methods.p_value import p_value_normal, test_decision_norm

if TYPE_CHECKING:
    from rhis_timeseries.types.hypothesis_types import TestResults
    from rhis_timeseries.types.timeseries_types import TimeSeriesFlex


def runs_test(
        ts: TimeSeriesFlex,
        alternative: str = 'two-sided',
        alpha: float=0.05,*,
        continuity: bool=True
        ) -> TestResults:
    """
    Apply the Single-Sample Runs Test in on a time series.
    Uses the median as a criteria for defining runs (up or down).

    Hypotheses
    ----------

        Null hypothesis
            H0: The events in the underlying population represented by the sample
                series are distributed randomly.

        Alternative hypothesis
            H1: (two-sided): The events in the underlying population represented by
                the sample series are distributed nonrandomly.
            H1: (less): The events in the underlying population represented by the
                sample series are distributed non-randomly due to few runs.
            H1: (greater): The events in the underlying population represented by the
                sample series are distributed non-randomly due to too many runs.

        Reference
        ---------
            SHESKIN, 2004. Handbook of Parametric and Nonparametric Statistical Procedures
            - Test 10. 3rd edition.

    Parameters
    ----------
        ts
            The time series (1D list or numpy ndarray).
        alternative
            One of the alternative hypotheses:
                two-sided
                greater
                less
        alpha
            The significance level for the test.
        continuity
            If True, applies the correction for continuity for the normal approximation.

    Return
    ------
        namedtuple('Runs_Test', ['statistic', 'p_value', 'alternative'])
    """
    raise_timeseries_type_error(ts)

    ts = np.array(ts) if isinstance(ts, list) else ts

    median = np.median(np.array(ts))

    runs1 = [] # upward runs
    runs2 = [] # downward runs
    signs = [] # +1 (higher than median); -1 (lower than median)
    n = [] # number of runs in each group (above and under the median)
    for element in ts:
        # Values equal to the median do not count
        if element > median:
            signs.append(1)
        if element < median:
            signs.append(-1)

    for i in range(1, len(signs)):
        el = signs[i]
        next_el = signs[i - 1]
        if el < next_el:
            runs1.append(1)
        if el > next_el:
            runs2.append(1)

    if signs[0] > 0:
        n.append(np.sum(runs1))
        n.append(np.sum(runs2) + 1)
    if signs[0] < 0:
        n.append(np.sum(runs1) + 1)
        n.append(np.sum(runs2))

    signs1 = np.array(signs)

    positives = signs1[signs1 > 0]
    negatives = signs1[signs1 < 0]

    n1 = float(len(positives))
    n2 = float(len(negatives))

    stat = float(np.sum(n))

    stat_med = (((2. * n1 * n2) / (n1 + n2)) + 1.)
    variance_num = (2. * n1 * n2 * (2. * n1 * n2 - n1 - n2))
    variance_den = ((n1 + n2) ** 2 * (n1 + n2 - 1.))

    # Test statistic Z
    # With or without correction for continuity
    num_z = (abs(stat - stat_med) - 0.5) if continuity else stat - stat_med
    z = num_z / ((variance_num / variance_den) ** 0.5)

    p_value = p_value_normal(z)
    decision = test_decision_norm(z, alpha, alternative)
    Results = namedtuple('Runs_Test', ['statistic', 'p_value', 'alternative', 'decision'])  # noqa: PYI024

    return Results(z, p_value, alternative, decision)


def wallismoore(
        ts: TimeSeriesFlex,
        alternative: str = 'two-sided',
        alpha: float=0.05,
    ) -> TestResults:
    """
    Applies the Wallis and Moore (1941) runtest for randomness.

    Reference
    ---------
        SHESKIN, 2004. Handbook of Parametric and Nonparametric Statistical Procedures - Test 10. 3rd edition.

    Parameters
    ----------
        ts       => 1D list or numpy array.
        interval => 1D list or tuple with length 2. The first object is the
                    index referent to the sample number to start the time
                    series. The second number is last sample number.
        alpha
            The significance level for the test.

    Return
    -------
        WallisMooreResult(statistic, p_value, alternative).
    """
    #Group 1 (pluses for zeros)
    signs1 = [] # positives and negatives
    plus1 = [] # positives
    minus1 = [] # negatives
    runs1 = [] # ones for each run

    #Group 2 (minuses for zeros)
    signs2 = [] # positives and negatives
    plus2 = [] # positives
    minus2 = [] # negatives
    runs2 = [] # ones for each run

    for i in range(1, len(ts)):

        if ts[i] < ts[i - 1]:
            signs1.append(-1)
            signs2.append(-1)
            minus1.append(1)
            minus2.append(1)

        if ts[i] > ts[i - 1]:
            signs1.append(1)
            signs2.append(1)
            plus1.append(1)
            plus2.append(1)

        if ts[i] == ts[i - 1]:
            signs1.append(1)
            signs2.append(-1)
            plus1.append(1)
            minus2.append(1)

    for i in range(1, len(ts) - 1):
        if signs1[i] != signs1[i - 1]:
            runs1.append(1)
        if signs2[i] != signs2[i - 1]:
            runs2.append(1)

    #Group 1
    runs11 = np.array(runs1)
    nruns1 = np.sum(runs11) + 1 # total number of runs

    #Group 2
    runs22 = np.array(runs2)
    nruns2 = np.sum(runs22) + 1 # total number of runs

    runs = (nruns1 + nruns2) / 2.

    n = len(ts)
    u = (2. * n - 1.) / 3.
    sigma = ((16. * n - 29.) / 90.) ** 0.5

    z = (runs - u) / sigma

    p_value = p_value_normal(z)
    decision = test_decision_norm(z, alpha, alternative)
    Results = namedtuple('WallisMooreResult', ['statistic', 'p_value', 'alternative', 'decision'])  # noqa: PYI024

    return Results(z, p_value, alternative, decision)


if __name__ == "__main__":
    """
    SHESKIN (2004) Example 10.2

    A quality control study is conducted on a machine that pours milk into containers.
    The amount of milk (in liters) dispensed by the machine into 21 consecutive containers follows:
    1.90, 1.99, 2.00, 1.78, 1.77, 1.76, 1.98, 1.90, 1.65, 1.76, 2.01, 1.78, 1.99, 1.76, 1.94, 1.78,
    1.67, 1.87, 1.91, 1.91, 1.89. Are the successive increments and decrements in the amount of milk
    dispensed random?
    """
    rng = np.random.default_rng(seed=42)
    ts_discrete = rng.integers(10, 50, size=10)
    ts_continuous = [1.90, 1.99, 2., 1.78, 1.77, 1.76, 1.98, 1.9, 1.65, \
          1.76, 2.01, 1.78, 1.99, 1.76, 1.94, 1.78, 1.67, 1.87, 1.91, 1.91, 1.89]

    ts_median_dis = np.median(ts_discrete)
    median_ts_dis = np.ones(len(ts_discrete)) * ts_median_dis

    ts_median_con = np.median(ts_continuous)
    median_ts_con = np.ones(len(ts_continuous)) * ts_median_con

    plt.figure(figsize=(8, 6))
    plt.plot(ts_discrete)
    plt.plot(median_ts_dis)
    plt.show()

    plt.figure(figsize=(8, 6))
    plt.plot(ts_continuous)
    plt.plot(median_ts_con)
    plt.show()

    print(runs_test(ts_continuous, alternative='two-sided'))
    print(runs_test(ts_discrete, alternative='two-sided'))
    # print(runs_test(ts_discrete, alternative='less'))
    # print(runs_test(ts_discrete, alternative='greater'), end='\n\n')
    print(wallismoore(ts_continuous, alternative='two-sided'))
    print(wallismoore(ts_discrete, alternative='two-sided'))
    # print(wallismoore(ts_continuous, alternative='less'))
    # print(wallismoore(ts_continuous, alternative='greater'))
