from __future__ import annotations

import matplotlib.pyplot as plt


def plot_standard_evol(self,*, ba: bool=True, fo: bool=True, use_raw: bool):  # noqa: C901
    data_marker_s = 100
    repr_marker_s = 100
    cols = set()
    for col, _ in self.evol_df.columns:
        cols.add(col)

    for col in cols:
        if use_raw and self.evol_df_raw is not None:
            hypos = ['R', 'H', 'I', 'S']
            colors = ['m', 'c', 'r', 'b']
            for i in range(len(hypos)):
                if ba:
                    ax1 = self.evol_df_raw[(col, 'ba', hypos[i])].plot(color=colors[i], alpha=0.7, linestyle='-', linewidth=2)
                if fo:
                    ax1 = self.evol_df_raw[(col, 'fo', hypos[i])].plot(color=colors[i], alpha=0.2, linestyle='-', linewidth=2)

        else:
            if ba:
                ax1 = self.evol_df[(col, 'ba')].plot(
                        color='k',
                        alpha=0.1,
                        linestyle='-',
                        linewidth=2)

            if fo:
                ax1 = self.evol_df[(col, 'fo')].plot(
                        color='k',
                        alpha=0.1,
                        linestyle='--',
                        linewidth=2)

        ax1.axhline(y=0.05, color='r', linestyle='--', alpha=0.5)
        ax2 = ax1.twinx()

        ax2.scatter(
            x=self.orig_df.index,
            y=self.orig_df[col],
            label=col,
            color='k',
            alpha=0.5,
            edgecolors='none',
            s=data_marker_s)

        if self.repr_df is not None:
            ax2.scatter(
                x=self.evol_df.index,
                y=self.orig_df[col + '_repr'],
                marker='*',
                label=col + '_repr',
                color='tab:green',
                edgecolors='none',
                s=repr_marker_s)
            try:
                ax2.scatter(
                    x=self.orig_df.index,
                    y=self.orig_df[col + '_repr_ext'],
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
