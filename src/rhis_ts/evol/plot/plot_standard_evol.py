from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from pandas import DataFrame

def plot_rhis_evol(
        col: str,
        evol_df_rhis: DataFrame,
        direction: str,
        ) -> Axes:
    if evol_df_rhis is not None:
        hypos = ['R', 'H', 'I', 'S']
        colors = ['m', 'c', 'r', 'b']
        for i in range(len(hypos)):
            if direction == 'ba':
                ax = evol_df_rhis[(col, 'ba', hypos[i])].plot(color=colors[i], alpha=0.3, linestyle='-', linewidth=2)
            if direction == 'fo':
                ax = evol_df_rhis[(col, 'fo', hypos[i])].plot(color=colors[i], alpha=0.3, linestyle='-', linewidth=2)
    else:
        err_msg = "No evol process ran in 'rhis' mode, rhis should be False."
        raise ValueError(err_msg)

    return ax


def plot_rhis_stats_evol(
        col: str,
        evol_df: DataFrame,
        direction: str,
        ) -> Axes:

    if direction == 'ba':
        ax = evol_df[(col, 'ba')].plot(
                color='k',
                alpha=0.2,
                linestyle='-',
                linewidth=2)

    if direction == 'fo':
        ax = evol_df[(col, 'fo')].plot(
                color='k',
                alpha=0.1,
                linestyle='--',
                linewidth=2)

    return ax


def plot_data(
        ax: Axes,
        col: str,
        orig_df: DataFrame,*,
        show_repr: bool=True) -> Axes:

    data_marker_s = 150
    repr_marker_s = 150
    ax1 = ax.twinx()

    ax1.scatter(
        x=orig_df.index,
        y=orig_df[col],
        label=col,
        color='k',
        alpha=0.7,
        edgecolors='none',
        s=data_marker_s)

    if show_repr:
        ax1.scatter(
            x=orig_df.index,
            y=orig_df[col + '_repr'],
            marker='*',
            label=col + '_repr',
            color='tab:green',
            edgecolors='none',
            s=repr_marker_s)
        try:
            ax1.scatter(
                x=orig_df.index,
                y=orig_df[col + '_repr_ext'],
                marker='*',
                label=col + '_repr_ext',
                color='orange',
                edgecolors='none',
                s=repr_marker_s)

        except KeyError:
            pass

    return ax1


def finalize_plot(evol_ax: Axes, data_ax: Axes, col_name: str, direction: str, savefig_path: str | None=None):
    evol_ax.axhline(y=0.05, color='r', linestyle='--', alpha=0.4, label='alpha=0.05')
    evol_ax.set_ylabel('p_value')
    evol_ax.set_xlabel('Time')
    data_ax.set_ylabel(col_name)

    lines1, labels1 = evol_ax.get_legend_handles_labels()
    lines2, labels2 = data_ax.get_legend_handles_labels()

    data_ax.legend(lines1 + lines2, labels1 + labels2)

    direction_name = 'Backwards' if direction == 'ba' else 'Forwards'

    title = f"Representative Selection by RHIS evol ({direction_name})"

    plt.title(title)
    plt.tight_layout()
    if savefig_path is not None:
        plt.savefig(savefig_path)
    plt.show()
