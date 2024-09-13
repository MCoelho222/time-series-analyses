from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt

from rhis_ts.evol.errors.exceptions import raise_rhis_evol_not_performed

if TYPE_CHECKING:
    from pandas import DataFrame

def plot_standard_evol(  # noqa: C901, PLR0913
        orig_df: DataFrame,
        evol_df: DataFrame,
        evol_df_rhis: DataFrame,*,
        ba: bool=True,
        fo: bool=True,
        rhis: bool=False,
        show_repr: bool=True):
    data_marker_s = 120
    repr_marker_s = 120
    cols = set()
    for col, _ in evol_df.columns:
        cols.add(col)

    for col in cols:
        raise_rhis_evol_not_performed(evol_df_rhis, rhis=rhis)
        if rhis and evol_df_rhis is not None:
            hypos = ['R', 'H', 'I', 'S']
            colors = ['m', 'c', 'r', 'b']
            for i in range(len(hypos)):
                if ba:
                    ax1 = evol_df_rhis[(col, 'ba', hypos[i])].plot(color=colors[i], alpha=0.7, linestyle='-', linewidth=2)
                if fo:
                    ax1 = evol_df_rhis[(col, 'fo', hypos[i])].plot(color=colors[i], alpha=0.2, linestyle='-', linewidth=2)

        else:
            if ba:
                ax1 = evol_df[(col, 'ba')].plot(
                        color='k',
                        alpha=0.2,
                        linestyle='-',
                        linewidth=2)

            if fo:
                ax1 = evol_df[(col, 'fo')].plot(
                        color='k',
                        alpha=0.1,
                        linestyle='--',
                        linewidth=2)

        ax1.axhline(y=0.05, color='r', linestyle='--', alpha=0.4)
        ax2 = ax1.twinx()

        ax2.scatter(
            x=orig_df.index,
            y=orig_df[col],
            label=col,
            color='k',
            alpha=0.7,
            edgecolors='none',
            s=data_marker_s)

        if show_repr:
            ax2.scatter(
                x=evol_df.index,
                y=orig_df[col + '_repr'],
                marker='*',
                label=col + '_repr',
                color='tab:green',
                edgecolors='none',
                s=repr_marker_s)
            try:
                ax2.scatter(
                    x=orig_df.index,
                    y=orig_df[col + '_repr_ext'],
                    marker='*',
                    label=col + '_repr_ext',
                    color='orange',
                    edgecolors='none',
                    s=repr_marker_s)

            except KeyError:
                pass

        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax2.legend(lines1 + lines2, labels1 + labels2)
        plt.show()
