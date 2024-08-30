from __future__ import annotations

from os.path import abspath, dirname, join

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from rhis_ts.utils.data import slices_to_evol
from rhis_ts.evol.rhis import rhis
from rhis_ts.repr.most_recent import representative_slice


def main():  # noqa: PLR0915
    ROOTPATH = dirname(abspath(__file__))
    EXAMPLE_PLOTS = join(ROOTPATH, "example_plots/")
    test_start = 10
    figtitle_fontsize = 9
    legend_fontsize = 7
    alpha_repr_ts = 0.1
    alpha_repr_stat = 0.3

    rng = np.random.default_rng(seed=555)
    dataset = [list(rng.uniform(-10., 100., 80)), list(rng.uniform(50., 200., 30))]
    ts = np.concatenate((dataset[0], dataset[1]))

    # ts = np.array(pd.read_csv('./data/BigSiouxAnnualQ.csv')['Y'])

    reprsentative_slice = representative_slice(ts)

    slices = slices_to_evol(ts, test_start)

    x_ts = range(len(ts))
    x_rhis_fw = range(9, len(ts))
    x_rhis_bw = range(len(ts) - 9)
    x_pvalue_lim = range(len(ts))

    fig, axes = plt.subplots(5, 1, sharex=True, figsize=(15, 10), dpi=100)
    axes[0].axvspan(0, reprsentative_slice[0][0], color='r', alpha=alpha_repr_ts)
    axes[0].scatter(x_ts, ts, color='r', label='Not representative')
    axes[0].axvspan(reprsentative_slice[0][0], axes[0].get_xlim()[1], color='green', alpha=alpha_repr_ts)
    axes[0].scatter(reprsentative_slice[0], reprsentative_slice[1], color='tab:green', label='Representative')
    axes[0].set_title("Original timeseries", fontsize=figtitle_fontsize)
    axes[0].legend(fontsize=legend_fontsize)

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
    axes[1].axvspan(reprsentative_slice[0][0], axes[0].get_xlim()[1], color='0.9', alpha=alpha_repr_stat)
    axes[1].set_title(f"Boxplot Forward Evolution - (starts with the fisrt {test_start} elements)",
                      fontsize=figtitle_fontsize)

    modes = ['raw', 'mean', 'median']
    hyps = [('R', 'tab:blue'), ('H', 'tab:green'), ('I', 'c'), ('S', 'tab:purple')]

    counter = 2
    for mode in modes:

        evol_rhis = rhis(ts, mode, bidirectional=True)
        evol_rhis_bw = evol_rhis['backward']
        evol_rhis_fw = evol_rhis['forward']

        if mode == 'raw':
            for hyp, c in hyps:
                plot_ts = evol_rhis_fw[hyp]
                plot_ts_inv = evol_rhis_bw[hyp]
                axes[counter].plot(x_pvalue_lim, 0.05 * np.ones(len(x_pvalue_lim)), '--', color='r', linewidth=1.0)
                axes[counter].axvspan(reprsentative_slice[0][0], axes[0].get_xlim()[1], color='0.9', alpha=alpha_repr_stat)
                axes[counter].scatter(x_rhis_fw, plot_ts, color=c, marker='+', label=hyp + ' forward')
                axes[counter].plot(x_rhis_bw, plot_ts_inv[::-1], color=c, label=hyp + ' backward')
                axes[counter].set_ylim(0, 1)
                axes[counter].set_xlim(0, len(ts))
                axes[counter].legend(ncols=1, fontsize=legend_fontsize)
                axes[counter].set_title(
                    f"Evolution of RHIS p-values - moving forward "
                    f"(starts with the fisrt {test_start} elements) "
                    f"and backward (starts with the last {test_start} elements)",
                    fontsize=figtitle_fontsize)

            counter += 1

        else:
            plot_ts = evol_rhis_fw['rhis_' + mode]
            plot_ts_inv = evol_rhis_bw['rhis_' + mode]
            axes[counter].plot(x_pvalue_lim, 0.05 * np.ones(len(x_pvalue_lim)), '--', color='r', linewidth=1.0)
            axes[counter].axvspan(reprsentative_slice[0][0], axes[0].get_xlim()[1], color='0.9', alpha=alpha_repr_stat)
            axes[counter].scatter(x_rhis_fw, plot_ts, marker='+', label='forward', color='k')
            axes[counter].plot(x_rhis_bw, plot_ts_inv[::-1], label='backward', color='k')
            axes[counter].set_ylim(0, 1)
            axes[counter].set_xlim(0, len(ts))
            axes[counter].legend(fontsize=legend_fontsize)
            axes[counter].set_title(
                f"Evolution of the {mode} RHIS p-values - moving forward "
                f"(starts with the fisrt {test_start} elements) and backward "
                f"(starts with the last {test_start} elements)",
                fontsize=figtitle_fontsize)

            ax = 4
            if counter == ax:
                axes[counter].set_xlabel('Index')

            counter += 1
    plt.subplots_adjust(hspace=0.2)
    plt.savefig(f"{EXAMPLE_PLOTS}representativeness_evolution.png", dpi=120)
    plt.show()

if __name__ == "__main__":
    main()
