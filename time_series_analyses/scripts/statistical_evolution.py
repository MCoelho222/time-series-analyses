import matplotlib.pyplot as plt
import numpy as np
import scipy.stats.mstats as mst
from scripts.duration_curves import dsc #, asc
from time_series_analyses.hypothesis_tests.non_parametric import NonParametric


def ts_for_bxp(ts, interval):    
    ts1 = []
    for i in range(len(ts)):
        ts2 = np.array(ts[i])
        ts3 = ts2[ts2 > 0]
        try:
            ts1.append(ts3[interval[0]:-interval[1]])
        except TypeError:
            ts1.append(ts3)
    return ts1


def bxp_evol(ts, interval):
    ts1 = np.array(ts)
    ts2 = ts1[ts1 > 0]
    try:
        ts3 = ts2[interval[0]:-interval[1]]
    except TypeError:
        ts3 = ts2
    slices = []
    for k in range(len(ts3) - 4):
        slices.append(ts3[:k + 5])    
    bp = plt.boxplot(slices, patch_artist=True, showfliers=True, showmeans=False)
    plt.setp(bp['boxes'], color='0.8')
    plt.setp(bp['whiskers'], color='k', linestyle='-')
    plt.setp(bp['fliers'], color='k', marker='+')
    plt.setp(bp['medians'], color='k')


def medians_evol(ts, interval):
    ts1 = np.array(ts)
    ts2 = ts1[ts1 == ts1**1]
    try:
        ts3 = ts2[interval[0]:-interval[1]]
    except TypeError:
        ts3 = ts2
    slices = []
    for k in range(len(ts3) - 4):
        slices.append(round(np.median(ts3[:k + 5]), 2)) 
    return slices


def evol(stat, element, ts, interval, clean):
    ts1 = np.array(ts)
    ts2 = ts1[ts1 == ts1**1]

    try:
        if clean == 'yes':
            ts3 = ts2[interval[0]:-interval[1]]
        if clean == 'no':
            ts3 = ts1[interval[0]:-interval[1]]
    except TypeError:
        if clean == 'yes':
            ts3 = ts2
        if clean == 'no':
            ts3 = ts1

    slices = []
    start = 10
    for i in range(len(ts3) - (start - 1)):
        slices.append(ts3[:i + start])
    slices2 = np.array(slices)

    if stat == 'mean':
        if element == 'regular':
            u = []
            for i in range(len(slices)):
                u.append(np.mean(slices2[i]))
            return u

        if element == 'geometric':
            u = []
            for i in range(len(slices)):
                u.append(mst.gmean(slices2[i]))
            return u

    if stat == 'std':
        if element == '':
            s = []
            for i in range(len(slices)):
                s.append(np.std(slices2[i]))
            return s

    if stat == 'boxplot':
        if clean == 'yes':
            if element == 'all':
                bp = plt.boxplot(slices, patch_artist=False, showfliers=True, showmeans=False)
                plt.setp(bp['boxes'], color='k', linewidth=0.5)
                plt.setp(bp['whiskers'], color='k', linestyle='-', linewidth=0.5)
                plt.setp(bp['fliers'], color='k', marker='+')
                plt.setp(bp['medians'], color='k')
                locs = np.arange(1, len(slices) + 1, 5)
                labels = np.arange(start, len(slices) + start, 5)
                plt.xticks(locs, labels)

            if element == '25th percentile':
                q1 = []
                for i in range(len(slices)):
                    q1.append(np.percentile(slices[i], 25))
                return q1

            if element == '50th percentile':
                q2 = []
                for i in range(len(slices)):
                    q2.append(np.percentile(slices[i], 50))
                return q2

            if element == '75th percentile':
                q3 = []
                for i in range(len(slices)):
                    q3.append(np.percentile(slices[i], 75))
                return q3

            if element == '5th percentile':
                q4 = []
                for i in range(len(slices)):
                    q4.append(np.percentile(slices[i], 5))
                return q4

            if element == '95th percentile':
                q5 = []
                for i in range(len(slices)):
                    q5.append(np.percentile(slices[i], 95))
                return q5

            if element == 'inferior limit':
                inflim = []
                for i in range(len(slices)):
                    q1 = np.percentile(slices[i], 25)
                    q3 = np.percentile(slices[i], 75)
                    iqr = q3 - q1
                    inflim.append(q1 - 1.5*iqr)
                return inflim

            if element == 'superior limit':
                suplim = []
                for i in range(len(slices)):
                    q1 = np.percentile(slices[i], 25)
                    q3 = np.percentile(slices[i], 75)
                    iqr = q3 - q1
                    suplim.append(q3 + 1.5*iqr)
                return suplim

            if element == 'minimum non outlier':
                infnout = []
                for i in range(len(slices)):
                    q1 = np.percentile(slices[i], 25)
                    q3 = np.percentile(slices[i], 75)
                    iqr = q3 - q1
                    x_asc = np.sort(slices2[i])
                    infnout.append(x_asc[x_asc >= (q1 - 1.5*iqr)][0])
                return infnout

            if element == 'maximum non outlier':
                supnout = []
                for i in range(len(slices)):
                    q1 = np.percentile(slices[i], 25)
                    q3 = np.percentile(slices[i], 75)
                    iqr = q3 - q1
                    x_asc = np.sort(slices2[i])
                    supnout.append(x_asc[x_asc <= (q3 + 1.5*iqr)][-1])
                return supnout

            if element == 'inferior outliers quantity':
                infouts = []
                for i in range(len(slices)):
                    q1 = np.percentile(slices[i], 25)
                    q3 = np.percentile(slices[i], 75)
                    iqr = q3 - q1
                    infouts.append(len(slices2[i][slices2[i] < (q1 - 1.5*iqr)]))
                return infouts

            if element == 'superior outliers quantity':
                supouts = []
                for i in range(len(slices)):
                    q1 = np.percentile(slices[i], 25)
                    q3 = np.percentile(slices[i], 75)
                    iqr = q3 - q1
                    supouts.append(len(slices2[i][slices2[i] > (q3 + 1.5*iqr)]))
                return supouts

        if clean == 'no':
            if element == '25th percentile':
                q1 = []
                for i in range(len(slices)):
                    slice1 = slices2[i]
                    cond = slice1[slice1 >= 0]
                    if len(cond) >= 10:
                        q1.append(np.percentile(cond, 25))
                    if len(cond) < 10:
                        q1.append(np.nan)
                return q1

            if element == '50th percentile':
                q1 = []
                for i in range(len(slices)):
                    slice1 = slices2[i]
                    cond = slice1[slice1 >= 0]
                    if len(cond) >= 10:
                        q1.append(np.percentile(cond, 50))
                    if len(cond) < 10:
                        q1.append(np.nan)
                return q1

            if element == '75th percentile':
                q1 = []
                for i in range(len(slices)):
                    slice1 = slices2[i]
                    cond = slice1[slice1 >= 0]
                    if len(cond) >= 10:
                        q1.append(np.percentile(cond, 75))
                    if len(cond) < 10:
                        q1.append(np.nan)
                return q1

            if element == '5th percentile':
                q1 = []
                for i in range(len(slices)):
                    slice1 = slices2[i]
                    cond = slice1[slice1 >= 0]
                    if len(cond) >= 10:
                        q1.append(np.percentile(cond, 5))
                    if len(cond) < 10:
                        q1.append(np.nan)
                return q1

            if element == '95th percentile':
                q1 = []
                for i in range(len(slices)):
                    slice1 = slices2[i]
                    cond = slice1[slice1 >= 0]
                    if len(cond) >= 10:
                        q1.append(np.percentile(cond, 95))
                    if len(cond) < 10:
                        q1.append(np.nan)
                return q1

            if element == 'minimum non outlier':
                infnout = []
                for i in range(len(slices)):
                    slice1 = slices2[i]
                    cond = slice1[slice1 >= 0]
                    if len(cond) >= 10:
                        q1 = np.percentile(cond, 25)
                        q3 = np.percentile(cond, 75)
                        iqr = q3 - q1
                        x_asc = np.sort(slice1)
                        infnout.append(x_asc[x_asc >= (q1 - 1.5*iqr)][0])
                    if len(cond) < 10:
                        infnout.append(np.nan)
                return infnout

            if element == 'maximum non outlier':
                supnout = []
                for i in range(len(slices)):
                    slice1 = slices2[i]
                    cond = slice1[slice1 >= 0]
                    if len(cond) >= 10:
                        q1 = np.percentile(cond, 25)
                        q3 = np.percentile(cond, 75)
                        iqr = q3 - q1
                        x_asc = np.sort(slice1)
            #            x_dsc = x_asc[::-1]
                        supnout.append(x_asc[x_asc <= (q3 + 1.5*iqr)][-1])
                    if len(cond) < 10:
                        supnout.append(np.nan)
                return supnout

    if stat == 'randomness':
        d = []
        for i in range(len(slices)):
            if element == 'z':
                x = NonParametric.runstest(slices2[i])
                d.append(x[1])
            if element == 'decision':
                x = NonParametric.runstest(slices2[i])
                d.append(x[2])
        return d

    if stat == 'homogeneity':
        d = []
        for i in range(len(slices)):
            if element == 'z':
                x = NonParametric.mwhitney_test(slices2[i])
                d.append(x[1])
            if element == 'decision':
                x = NonParametric.mwhitney_test(slices2[i])
                d.append(x[2])
        return d

    if stat == 'independence':
        d = []
        for i in range(len(slices)):
            if element == 'z':
                x = NonParametric.waldwolf_test(slices2[i])
                d.append(x[1])
            if element == 'decision':
                x = NonParametric.waldwolf_test(slices2[i])
                d.append(x[2])
        return d

    if stat == 'stationarity':
        d = []
        for i in range(len(slices)):
            if element == 'z':
                x = NonParametric.mann_kendall_test(slices2[i])
                d.append(x[1])
            if element == 'decision':
                x = NonParametric.mann_kendall_test(slices2[i])
                d.append(x[2])
            if element == 'slope':
                x = NonParametric.mann_kendall_test(slices2[i])
                d.append(x[3])
        return d

    if stat == 'duration curve':
        dsc_perms = []
#        asc_perms = []
        for i in range(len(slices)):
            x = dsc(slices2[i], ('all'), element)[2]
#            y = asc(slices2[i], ('all'), element)[2]
            dsc_perms.append(x)
#            asc_perms.append(y)
#        return (dsc_perms, asc_perms)
        return dsc_perms


def slices2evol(ts, start):
    ts1 = np.array(ts)
    slices = []
    for i in range(len(ts1) - (start - 1)):
        slices.append(ts1[:i + start])
    return slices


def evolrhis(slices, stat):  
    if stat == 'randomness':
        d = []
        for i in range(len(slices)):
            x = NonParametric.runstest(slices[i])
            pvalue = x['stats'][1]
            d.append(pvalue)
        return d
    if stat == 'homogeneity':
        d = []
        for i in range(len(slices)):
            x = NonParametric.mwhitney_test(slices[i])
            pvalue = x['stats'][1]
            d.append(pvalue)
        return d
    if stat == 'independence':
        d = []
        for i in range(len(slices)):
            x = NonParametric.waldwolf_test(slices[i])
            pvalue = x['stats'][1]
            d.append(pvalue)
        return d
    if stat == 'stationarity':
        d = []
        for i in range(len(slices)):
            x = NonParametric.mann_kendall_test(slices[i])
            pvalue = x['stats'][1]
            d.append(pvalue)
        return d


def evolpercentile(slices, percentile):
    if percentile == 25:
        q1 = []
        for i in range(len(slices)):
            q1.append(np.percentile(slices[i], 25))
        return q1
    if percentile == 50:
        q2 = []
        for i in range(len(slices)):
            q2.append(np.percentile(slices[i], 50))
        return q2
    if percentile == 75:
        q3 = []
        for i in range(len(slices)):
            q3.append(np.percentile(slices[i], 75))
        return q3
    if percentile == 5:
        q4 = []
        for i in range(len(slices)):
            q4.append(np.percentile(slices[i], 5))
        return q4
    if percentile == 95:
        q5 = []
        for i in range(len(slices)):
            q5.append(np.percentile(slices[i], 95))
        return q5
    

if __name__ == "__main__":
    
    x = [list(np.random.uniform(-10., 100., 80)), list(np.random.uniform(30., 200., 30))]
    ts = ts_for_bxp(x, 'all')
    ts1 = np.concatenate((ts[0], ts[1]))

    plt.figure(figsize=(8, 6))
    plt.plot(ts1)
    plt.title("Original timeseries")
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(20, 6))
    bxp_evol(ts1, 'all')
    plt.title("Boxplot Evolution")
    plt.tight_layout()
    plt.show()
    
    slices = slices2evol(ts1, 10)
    trend_evol = evolrhis(slices, 'stationarity')
    homo_evol = evolrhis(slices, 'homogeneity')
    ind_evol = evolrhis(slices, 'independence')
    rand_evol = evolrhis(slices, 'randomness')

    plt.figure(figsize=(20, 6))
    plt.plot(trend_evol, label='Stationarity')
    plt.plot(homo_evol, label='Homogeneity')
    plt.plot(ind_evol, label='Independence')
    plt.plot(rand_evol, label='Randomness')
    plt.legend()
    plt.title("Stationarity Evolution")
    plt.tight_layout()
    plt.show()