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

    rng = np.random.default_rng(seed=555)
    dataset = [list(rng.uniform(-10., 100., 80)), list(rng.uniform(50., 200., 30))]
    ts = np.concatenate((dataset[0], dataset[1]))

    slices = slices2evol(ts, 10)

    evol_rhis = rhis_evol(slices)

    x1 = range(len(ts))
    x2 = range(9, len(ts))

    fig, axes = plt.subplots(3, 1, sharex=True, figsize=(16,12))
    plt.savefig(f"{EXAMPLE_PLOTS}representativeness_evolution.png")
    axes[0].scatter(x1, ts, color='k')
    axes[0].set_title("Original timeseries")

    axes[1].boxplot(slices,
                    positions=range(9, len(ts)),
                    manage_ticks=False,
                    showmeans=True,
                    meanline=True,
                    showcaps=False,
                    medianprops={'color': 'k'},
                    meanprops={'color': 'b'},
                    flierprops={'marker': '+', 'color': 'k'}
                    )
    axes[1].set_title("Boxplot Evolution")

    hyps = [('randomness', 'tab:blue'), ('homogeneity', 'tab:green'),
            ('independence', 'c'), ('stationarity', 'tab:purple')]

    for hyp, c in hyps:
        ts = evol_rhis[hyp]
        axes[2].plot(x2, ts, label=hyp, color=c)
        axes[2].plot(x2, 0.05 * np.ones(len(x2)), '--', color='r', linewidth=1.0)
        axes[2].set_ylim(0, 1)
        axes[2].legend()
        axes[2].set_title("Representativeness Evolution")
        axes[2].set_xlabel('Number of elements used')
    # plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
