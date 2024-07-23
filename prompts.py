from __future__ import annotations

# from collections import namedtuple
import matplotlib.pyplot as plt
import numpy as np

# import scipy.stats as sts


# def mann_kendall_test(timeseries):
#     ts = np.array(timeseries)
#     n = len(ts)

#     signs = []
#     for i in range(n - 1):
#         s = ts - ts[i]
#         signs.extend(np.sign(s[i + 1:]))

#     signs_array = np.array(signs)

#     test_s = float(len(signs_array[signs_array > 0]) - len(signs_array[signs_array < 0]))

#     sigma = ((n/18.)*(n - 1.)*(2.*n + 5.))**0.5

#     condition_value = 0.

#     if test_s > condition_value:
#         z = abs((test_s - 1.)/sigma)
#     if test_s == condition_value:
#         z = condition_value
#     if test_s < condition_value:
#         z = abs((test_s + 1.)/sigma)

#     p = 2*(1 - sts.norm.cdf(z))

#     Results = namedtuple('Mann_Kendall', ['z', 'p_value'])  # noqa: PYI024

#     return Results(z, p)


# def stat_evol(slices):
#     stats = {
#         'stationarity': [],
#     }

#     for i in range(len(slices)):
#         ts = slices[i]
#         result = mann_kendall_test(ts)

#         rhis = {
#             'stationarity': result,
#         }

#         for key in stats.keys():
#             p_value = rhis[key].p_value
#             stats[key].append(p_value)

#     return stats


# def slices2evol(ts, start):
#     try:
#         slices = []
#         for i in range(len(ts) - (start - 1)):
#             slices.append(ts[: i + start])

#         return slices
#     except TypeError as exc:
#         return {'error': str(exc)}


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

    # ts = [10., 20., 30., 11., 5., 6., 33., 24., 15., 17., 9., 8.5, 8.8, 5.5, 3., 3.3, 2., 1.]
    ts = np.random.randint(0, 101, size=100)
    ts.sort()
    plt.figure()
    plt.plot(ts)
    plt.show()
    # evol = stat_evol(slices2evol(ts, 5))['stationarity']
    # evol_reverted = evol[::-1]
    # plt.figure()
    # plt.plot(evol)
    # plt.show()

    # a_level = 0.05
    # mean = np.mean(evol_reverted[:2])
    # print('mean', mean)
    # i = 2

    # while mean <= a_level and i < len(evol):
    #     mean = np.mean(evol_reverted[:i])
    #     i = i + 1
    # print('i', i)

    # x = len(evol) if i == 2 else len(evol) - i
    # plt.figure()
    # plt.plot(evol)
    # plt.axvline(x=x, color='r', linestyle='--')
    # plt.show()
    # randts = ts
    # if i > 2:
    #     randts = ts[-i:]
    #     print('oi', randts)
    #     if len(randts) < 5:
    #         randts = ts[:5]
    # print('slice', ts)
    # print('randts', randts)
    # if len(randts) > 5:
    #     check = mann_kendall_test(randts)

    #     i = 0
    #     while len(randts) >= 5 and check.p_value < a_level:
    #         randts = randts[i:]
    #         check = mann_kendall_test(ts)
    #         i += 1

    # print(randts)
    # print('check', check)
    # if check.p_value > a_level:
    #     result1 = asc(randts, ('all'), 95)
    #     result2 = dsc(randts, ('all'), 95)
    #     print(result1)
    #     print(result2)
    #     plt.figure()
    #     plt.plot(result1[0], result1[1])
    #     plt.plot(result2[0], result2[1])
    #     plt.show()

    result1 = asc(ts, ('all'), 95)
    result2 = dsc(ts, ('all'), 95)
    print(result1)
    print(result2)
    plt.figure()
    plt.plot(result1[0], result1[1])
    plt.plot(result2[0], result2[1])
    plt.axhline(y=result1[2], color='r', linestyle='--')
    plt.axhline(y=result2[2], color='r', linestyle='--')
    plt.show()
