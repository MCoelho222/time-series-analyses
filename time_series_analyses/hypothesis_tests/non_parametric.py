import numpy as np
import scipy.stats as sts
from collections import namedtuple

class NonParametric():

    @staticmethod
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
    

    @staticmethod
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
        ------------------------------------------------------------"""
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

        Results = namedtuple('Mann_Whitney', ['z', 'p_value'])
        
        return {'stats': Results(z, p), 'decision': decision}


    @staticmethod
    def runstest(ts):
        """-----------------------------------------------------------------------
        FUNCTION         A test for randomness
        PARAMETERS       ts => 1D list or numpy array.
        RETURNS          float
        REFERENCE        Sheskin (2004)
        -----------------------------------------------------------------------""" 
        ts1 = np.array(ts)
        ts2 = ts1[ts1 == ts1**1] #drop non-numbers
        
        if len(ts2) == 0:
            return np.nan, np.nan, np.nan
        
        if len(ts2) > 0:
            n = []
            runs1 = []
            runs2 = []
            signs = [] # +1 (higher than median); -1 (lower than median)
            median = np.median(np.array(ts2))
            for i in range(len(ts2)):
                if ts2[i] > median:
                    signs.append(1)
                if ts2[i] < median:
                    signs.append(-1)
                    
            for i in np.arange(1, len(signs)):
                if signs[i] < signs[i - 1]:
                    runs1.append(1)
                if signs[i] > signs[i - 1]:
                    runs2.append(1)
            try:
                if signs[0] > 0:
                    n.append(np.sum(runs1))
                    n.append(np.sum(runs2) + 1)
                if signs[0] < 0:
                    n.append(np.sum(runs1) + 1)
                    n.append(np.sum(runs2))
                R = float(np.sum(n))
                signs1 = np.array(signs)
                positives = signs1[signs1 > 0]
                negatives = signs1[signs1 < 0]
                n1 = float(np.sum(positives))
                n2 = float(np.sum(negatives))*(-1.)
                Rmed = ((2.*n1*n2)/(n1 + n2)) + 1.
                num_den = 2.*n1*n2*(2.*n1*n2 - n1 - n2)
                den_den = ((n1 + n2)**2.)*(n1 + n2 - 1.)
                z = abs((R - Rmed)/(num_den/den_den)**0.5)
                p = 2*(1 - sts.norm.cdf(z))

                if z < 1.96:
                    decision = -1
                if z >= 1.96:
                    decision = 1 

                Results = namedtuple('Runs_Test', ['z', 'p_value'])
                return {'stats': Results(z, p), 'decision': decision}
            except:
                decision = 1
                return {'stats': Results(np.nan, 0.0), 'decision': decision}


    @staticmethod
    def wallismoore(ts, interval):
        """------------------------------------------------------------------------
        FUNCTION 
        ---------------------------------------------------------------------------
            Applies the Wallis and Moore (1941) runtest for randomness, presented
            in SHESKIN (2004).
        ---------------------------------------------------------------------------
        PARAMETERS
        ---------------------------------------------------------------------------
            ts       => 1D list or numpy array.
            interval => 1D list or tuple with length 2. The first object is the 
                        index referent to the sample number to start the time 
                        series. The second number is last sample number.
        ---------------------------------------------------------------------------
        RETURNS 
        ---------------------------------------------------------------------------
        a 2D list with quantities Runs, n1, n2
        
        [0] = (runs, n1, n2), if zeros are +
        [1] = (runs, n1, n2), if zeros are -
        --------------------------------------------------------------------"""
        
        ts1 = np.array(ts)
        ts2 = ts1[ts1 == ts1**1]
        try:
            ts3 = ts2[interval[0] - 1:interval[1] + 1]
        except TypeError:
            ts3 = ts2
        #Group 1 (+ for zeros)
        signs1 = [] #positives and negatives
        plus1 = [] #positives
        minus1 = [] #negatives
        runs1 = [] #ones for each run
        #Group 2 (- for zeros)
        signs2 = [] #positives and negatives
        plus2 = [] #positives
        minus2 = [] #negatives
        runs2 = [] #ones for each run
        for i in np.arange(1, len(ts3)):
            if ts3[i] < ts3[i - 1]:
                signs1.append(-1)
                signs2.append(-1)
                minus1.append(1)
                minus2.append(1)
            if ts3[i] > ts3[i - 1]:
                signs1.append(1)
                signs2.append(1)
                plus1.append(1)
                plus2.append(1)
            if ts3[i] == ts3[i - 1]:
                signs1.append(1)
                signs2.append(-1)
                plus1.append(1)
                minus2.append(1)
        for i in np.arange(1, len(ts3) - 1):
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
        try:
            runs = (nruns1 + nruns2)/2.
            n = len(ts)
            u = (2.*n - 1.)/3.
            sigma = ((16.*n - 29.)/90.)**0.5
            z = abs((runs - u)/sigma)
            
            p = 2*(1 - sts.norm.cdf(z))

            if z < 1.96:
                decision = -1
            if z >= 1.96:
                decision = 1 

            Results = namedtuple('Wallis_Moore', ['z', 'p_value'])
            return {'stats': Results(z, p), 'decision': decision}
        except:
            decision = 1
            return {'stats': Results(np.nan, 0.0), 'decision': decision}
    

    @staticmethod
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