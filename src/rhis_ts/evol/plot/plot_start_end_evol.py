from __future__ import annotations

import matplotlib.pyplot as plt


def plot_start_end_evol(self,):
    data_marker_s = 100
    cols = set()
    idxs = {}
    for col, idx in self.repr_df.columns:
        cols.add(col)
        if col in idxs.keys():
            idxs[col] += [idx]
        else:
            idxs[col] = [idx]

    for col in cols:
        ax1 = self.evol_df[(col, 'ba')].plot(
                color='k',
                alpha=0.1,
                linestyle='-',
                linewidth=2)
        ax1 = self.evol_df[(col, 'fo')].plot(
                color='k',
                alpha=0.1,
                linestyle='--',
                linewidth=2)

        ax1.axhline(y=0.05, color='r', linestyle='--', alpha=0.5)

        ax2 = ax1.twinx()

        for idx in idxs[col]:
            ax2.scatter(
                x=self.repr_df.index,
                y=self.repr_df[(col, idx)],
                label=col,
                alpha=0.3,
                edgecolors='none',
                s=data_marker_s)

        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax2.legend(lines1 + lines2, labels1 + labels2)
        plt.show()
