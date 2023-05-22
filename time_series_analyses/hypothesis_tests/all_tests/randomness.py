import numpy as np
import scipy.stats as sts
from collections import namedtuple


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


if __name__ == "__main__":
    
    ts = np.random.randint(0, 100, 100)
    print(ts)
    print(runstest(ts))