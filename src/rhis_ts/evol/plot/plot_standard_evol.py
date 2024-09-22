from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from pandas import DataFrame

def plot_rhis_evol(  # noqa: PLR0913
        col_name: str,
        evol_df: DataFrame,
        evol_df_rhis: DataFrame,
        direction: str,
        xlabel: str|None,
        rhis_params: dict[str|int]|None=None,
        rhis_stat_params: dict[str|int]|None=None,*,
        rhis: bool=False,
        ) -> Axes:
    if rhis_params is None:
        rhis_params = {}
    if rhis_stat_params is None:
        rhis_stat_params = {}
    if rhis:
        if evol_df_rhis is not None:
            hypos = ['R', 'H', 'I', 'S']
            colors_default = {'R': 'm', 'H': 'c', 'I': 'r', 'S': 'b'}
            for i in range(len(hypos)):
                ax = evol_df_rhis[(col_name, direction, hypos[i])].plot(
                    color=rhis_params.get('colors', colors_default)[hypos[i]],
                    alpha=rhis_params.get('alpha', 0.4),
                    linestyle=rhis_params.get('linestyle', '-'),
                    linewidth=rhis_params.get('linewidth', 2))
        else:
            err_msg = "To plot the rhis evolution, please first call the method 'evol' with parameter 'rhis'=True."
            raise ValueError(err_msg)
    else:
        ax = evol_df[(col_name, direction)].plot(
            color=rhis_stat_params.get('color', 'k'),
            alpha=rhis_stat_params.get('alpha', 0.7),
            linestyle=rhis_stat_params.get('linestyle', '-'),
            linewidth=rhis_stat_params.get('linewidth', 2))

    ax.set_xlabel(xlabel if xlabel is not None else 'Time')
    return ax


def plot_data(  # noqa: PLR0913
        evol_ax: Axes,
        col_name: str,
        orig_df: DataFrame,
        ylabel: str|None=None,
        data_params: dict[str|int]|None=None,
        repr_params: dict[str|int]|None=None,*,
        show_repr: bool=True
        ) -> Axes:
    if data_params is None:
        data_params = {}
    if repr_params is None:
        repr_params = {}
    data_ax = evol_ax.twinx()
    data_ax.scatter(
        x=orig_df.index,
        y=orig_df[col_name],
        label=col_name,
        marker=data_params.get('marker', 'o'),
        color=data_params.get('color', 'none'),
        edgecolors=data_params.get('edgecolors', 'k'),
        facecolors=data_params.get('facecolors', 'none'),
        alpha=data_params.get('alpha', 0.7),
        s=data_params.get('s', 100))
    if show_repr:
        data_ax.scatter(
            x=orig_df.index,
            y=orig_df[col_name + '_repr'],
            label=col_name + '_repr',
            marker=repr_params.get('marker', 'o'),
            color=repr_params.get('color', 'none'),
            edgecolors=repr_params.get('edgecolors', 'none'),
            facecolors=repr_params.get('facecolors', 'k'),
            alpha=repr_params.get('alpha', 0.7),
            s=repr_params.get('s', 100))

    data_ax.set_ylabel(ylabel if ylabel is not None else col_name)
    return data_ax

def finalize_plot(  # noqa: PLR0913
        evol_ax: Axes,
        data_ax: Axes,
        alpha: float,
        figtitle: str|None=None,
        alpha_line_params: dict[str|int]|None=None,
        savefig_path: str|None=None
        ):
    if alpha_line_params is None:
        alpha_line_params = {}

    label_alpha = f"alpha={alpha}"
    evol_ax.axhline(
        y=alpha,
        color=alpha_line_params.get('color', 'k'),
        linestyle=alpha_line_params.get('linestyle', '--'),
        alpha=alpha_line_params.get('alpha', 0.4),
        label=label_alpha)

    evol_ax.set_ylabel('p_value')
    evol_ax.set_ylim(0, 1)

    lines1, labels1 = evol_ax.get_legend_handles_labels()
    lines2, labels2 = data_ax.get_legend_handles_labels()

    data_ax.legend(lines1 + lines2, labels1 + labels2)

    if figtitle:
        plt.title(figtitle, loc='left')
    plt.tight_layout()
    if savefig_path is not None:
        plt.savefig(savefig_path)
    plt.show()
