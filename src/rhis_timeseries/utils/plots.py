"""Methods to generate features for plots."""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np


def boxplot_evol(ts: list[int | float] | np.ndarray) -> None:
    """Prepare data and build boxplot.

    Parameters
    ----------
        ts
            A list or numpy array with integers or floats

    Returns
    -------
        A boxplot
    """
    ts1 = np.array(ts) if isinstance(ts, list) else ts

    slices = []
    for k in range(len(ts1) - 4):
        slices.append(ts1[:k + 5])

    bp = plt.boxplot(slices, patch_artist=True, showfliers=True, showmeans=False)
    plt.setp(bp['boxes'], color='0.8')
    plt.setp(bp['whiskers'], color='k', linestyle='-')
    plt.setp(bp['fliers'], color='k', marker='+')
    plt.setp(bp['medians'], color='k')