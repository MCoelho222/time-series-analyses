from __future__ import annotations

from os.path import abspath, dirname, join

import matplotlib.pyplot as plt
import numpy as np

from rhis_timeseries.evolution.data import slices2evol
from rhis_timeseries.evolution.plots import boxplot_evolution_plot
from rhis_timeseries.evolution.rhis import rhis_evol


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
    boxplot_evolution_plot(ts, 5)
    plt.title("Boxplot Evolution")
    plt.tight_layout()
    plt.savefig(f"{EXAMPLE_PLOTS}boxplot_evolution.png")
    plt.show()

    slices = slices2evol(ts, 10)

    evol_rhis = rhis_evol(slices)

    plt.figure(figsize=(20, 6))

    hyps = ['randomness', 'homogeneity', 'independence', 'stationarity']

    for hyp in hyps:
        ts = evol_rhis[hyp]
        plt.plot(ts, label=hyp)

    plt.plot(0.05 * np.ones(len(slices)), color='red')
    plt.legend()
    plt.ylabel('p-value')
    plt.xlim(0, len(slices))
    plt.ylim(0, 1)
    plt.title("Representativeness Evolution")
    plt.tight_layout()
    plt.savefig(f"{EXAMPLE_PLOTS}representativeness_evolution.png")
    plt.show()


if __name__ == "__main__":
    main()
