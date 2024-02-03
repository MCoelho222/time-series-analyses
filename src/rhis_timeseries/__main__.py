from __future__ import annotations

from os.path import abspath, dirname, join

import matplotlib.pyplot as plt
import numpy as np

from rhis_timeseries.scripts.statistical_evolution import bxp_evol, evolrhis, slices2evol, ts_for_bxp


def main():
    ROOTPATH = dirname(abspath(__file__))
    EXAMPLE_PLOTS = join(ROOTPATH, "example_plots/")

    x = [list(np.random.uniform(-10., 100., 80)), list(np.random.uniform(30., 200., 30))]
    ts = ts_for_bxp(x, 'all')
    ts1 = np.concatenate((ts[0], ts[1]))

    plt.figure(figsize=(8, 6))
    plt.plot(ts1)
    plt.title("Original timeseries")
    plt.tight_layout()
    plt.savefig(f"{EXAMPLE_PLOTS}original_ts.png")
    plt.show()

    plt.figure(figsize=(20, 6))
    bxp_evol(ts1, 'all')
    plt.title("Boxplot Evolution")
    plt.tight_layout()
    plt.savefig(f"{EXAMPLE_PLOTS}boxplot_evolution.png")
    plt.show()

    slices = slices2evol(ts1, 10)
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
