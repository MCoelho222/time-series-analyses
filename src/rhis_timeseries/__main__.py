from __future__ import annotations

from os.path import abspath, dirname, join

import matplotlib.pyplot as plt
import numpy as np

from rhis_timeseries.evolution.data import slices_incr_len
from rhis_timeseries.evolution.rhis import rhis_evolution


def main():
    ROOTPATH = dirname(abspath(__file__))
    print(ROOTPATH)
    EXAMPLE_PLOTS = join(ROOTPATH, "example_plots/")

    rng = np.random.default_rng(seed=555)
    dataset = [list(rng.uniform(-10., 100., 80)), list(rng.uniform(50., 200., 30))]
    ts = np.concatenate((dataset[0], dataset[1]))

    slices = slices_incr_len(ts, 10)

    x_ts = range(len(ts))
    x_rhis_fw = range(9, len(ts))
    x_rhis_bw = range(len(ts) - 9)
    x_pvalue_lim = range(len(ts))

    fig, axes = plt.subplots(5, 1, sharex=True, figsize=(16, 12))
    axes[0].scatter(x_ts, ts, color='k')
    axes[0].set_title("Original timeseries")

    axes[1].boxplot(slices,
                    positions=range(9, len(ts)),
                    manage_ticks=False,
                    showmeans=False,
                    meanline=False,
                    showcaps=False,
                    medianprops={'color': 'k'},
                    meanprops={'color': 'b'},
                    flierprops={'marker': '+', 'color': '0.6'},
                    boxprops={'color': '0.6'},
                    whiskerprops={'color': '0.6'},
                    )
    axes[1].set_title("Boxplot Evolution - forward")

    modes = ['raw', 'mean', 'median']
    hyps = [('randomness', 'tab:blue'), ('homogeneity', 'tab:green'),
            ('independence', 'c'), ('stationarity', 'tab:purple')]

    counter = 2
    for mode in modes:

        evol_rhis = rhis_evolution(ts, mode, bidirectional=True)
        evol_rhis_bw = evol_rhis['backward']
        evol_rhis_fw = evol_rhis['forward']

        if mode == 'raw':
            for hyp, c in hyps:
                plot_ts = evol_rhis_fw[hyp]
                plot_ts_inv = evol_rhis_bw[hyp]
                axes[counter].scatter(x_rhis_fw, plot_ts, color=c, marker='+')
                axes[counter].plot(x_rhis_bw, plot_ts_inv[::-1], color=c, label=hyp)
                axes[counter].plot(x_pvalue_lim, 0.05 * np.ones(len(x_pvalue_lim)), '--', color='r', linewidth=1.0)
                axes[counter].set_ylim(0, 1)
                axes[counter].set_xlim(0, len(x_rhis_fw))
                axes[counter].legend()
                axes[counter].set_title("RHIS Evolution")

            counter += 1

        else:
            plot_ts = evol_rhis_fw['rhis_' + mode]
            plot_ts_inv = evol_rhis_bw['rhis_' + mode]
            axes[counter].plot(x_rhis_fw, plot_ts, label='forward', color='0.6')
            axes[counter].plot(x_rhis_bw, plot_ts_inv[::-1], label='backward', color='k')
            axes[counter].plot(x_pvalue_lim, 0.05 * np.ones(len(x_pvalue_lim)), '--', color='r', linewidth=1.0)
            axes[counter].set_ylim(0, 1)
            axes[counter].set_xlim(0, len(x_rhis_fw))
            axes[counter].legend()
            axes[counter].set_title(F"RHIS Evolution - {mode}")

            ax = 4
            if counter == ax:
                axes[counter].set_xlabel('Index')

            counter += 1

    plt.savefig(f"{EXAMPLE_PLOTS}representativeness_evolution.png")
    plt.show()


if __name__ == "__main__":
    main()
