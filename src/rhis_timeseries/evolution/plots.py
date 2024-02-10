"""Methods to generate features for plots."""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from rhis_timeseries.evolution.data import slices2evol


def boxplot_evolution_plot(ts: list[int | float] | np.ndarray, length_1st_slice: int) -> None:
    """Prepare data and build boxplot evolution.

    Parameters
    ----------
        ts
            A list or numpy array with integers or floats.
        length_1st_slice
            The length of the first slice of the the evolution data.

    """
    slices = slices2evol(ts, length_1st_slice)

    bp = plt.boxplot(slices, patch_artist=True, showfliers=True, showmeans=False)
    plt.setp(bp['boxes'], color='0.8')
    plt.setp(bp['whiskers'], color='k', linestyle='-')
    plt.setp(bp['fliers'], color='k', marker='+')
    plt.setp(bp['medians'], color='k')

    locs = np.arange(1, len(slices) + 1, length_1st_slice)
    labels = np.arange(length_1st_slice, len(slices) + length_1st_slice, length_1st_slice)
    plt.xticks(locs, labels)
