from __future__ import annotations

from os.path import abspath, dirname, join

import matplotlib.pyplot as plt
import numpy as np

from rhis_timeseries.scripts.statistical_evolution import evolrhis
from rhis_timeseries.utils.evolution import slices2evol
from rhis_timeseries.utils.plots import bxp_evol


def main():
    ROOTPATH = dirname(abspath(__file__))
    EXAMPLE_PLOTS = join(ROOTPATH, "example_plots/")

    dataset = [list(np.random.uniform(-10., 100., 80)), list(np.random.uniform(50., 200., 30))]  # noqa: NPY002
    ts = np.concatenate((dataset[0], dataset[1]))

    plt.figure(figsize=(8, 6))
    plt.plot(ts)
    plt.title("Original timeseries")
    plt.tight_layout()
    plt.savefig(f"{EXAMPLE_PLOTS}original_ts.png")
    plt.show()

    plt.figure(figsize=(20, 6))
    bxp_evol(ts)
    plt.title("Boxplot Evolution")
    plt.tight_layout()
    plt.savefig(f"{EXAMPLE_PLOTS}boxplot_evolution.png")
    plt.show()

    slices = slices2evol(ts, 10)
    trend_evol = evolrhis(slices, 'stationarity')
    homo_evol = evolrhis(slices, 'homogeneity')
    ind_evol = evolrhis(slices, 'independence')
    rand_evol = evolrhis(slices, 'randomness')

    plt.figure(figsize=(20, 6))
    plt.plot(trend_evol, label='Stationarity', color='m')
    plt.plot(homo_evol, label='Homogeneity', color='k')
    plt.plot(ind_evol, label='Independence', color='blue')
    plt.plot(rand_evol, label='Randomness', color='0.8')
    plt.plot(0.05 * np.ones(len(trend_evol)), color='red')
    plt.legend()
    plt.ylabel('p-value')
    plt.xlim(0, len(trend_evol))
    plt.ylim(0, 1)
    plt.title("Representativeness Evolution")
    plt.tight_layout()
    plt.savefig(f"{EXAMPLE_PLOTS}representativeness_evolution.png")
    plt.show()


if __name__ == "__main__":
    main()
