from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from rhis.core import Rhis


def main():
    rng = np.random.default_rng()

    df = pd.DataFrame({
        "series_A": np.clip(rng.normal(50, 15, 100), 0, 100),
        "series_B": np.clip(rng.normal(50, 15, 100), 0, 100),
        "series_C": np.clip(rng.normal(50, 15, 100), 0, 100),
        "series_D": np.clip(rng.normal(50, 15, 100), 0, 100),
    })

    # Forcing a trend on series_A and series_B
    df["series_A"] = np.sort(df["series_A"].to_numpy())
    df["series_B"] = np.sort(df["series_B"].to_numpy())

    cols = df.columns
    rhis = Rhis(df)
    rhis.evol()
    rhis_stat = 'min'
    rhis.add_rhis_compliant_to_df(rhis_stat)

    orig_df = rhis.orig_df
    rhis_df = rhis.rhis_df
    alpha = rhis.alpha

    def plot(evol_ax, col):
        data_ax = evol_ax.twinx()
        data_ax.scatter(
            x=orig_df.index,
            y=orig_df[col],
            label=col,
            marker='o',
            color='none',
            edgecolors='k',
            facecolors='none',
            alpha=1,
            s=50,
        )

        data_ax.scatter(
            x=orig_df.index,
            y=orig_df[col + '_repr'],
            label=col + '_repr',
            marker='o',
            color='none',
            edgecolors='k',
            facecolors='k',
            alpha=1,
            s=50,
        )

        data_ax.set_ylabel(col)

        label_alpha = f"alpha={alpha}"
        evol_ax.axhline(y=alpha, color='k', linestyle='--', linewidth=0.5, alpha=0.4, label=label_alpha)

        evol_ax.set_ylabel('p_value')
        evol_ax.set_ylim(0, 1)

        lines1, labels1 = evol_ax.get_legend_handles_labels()
        lines2, labels2 = data_ax.get_legend_handles_labels()

        data_ax.legend(lines1 + lines2, labels1 + labels2)

        plt.title('RHIS Example', loc='left', fontsize=11)
        plt.tight_layout()
        plt.show()

    for col in cols:
        evol_ax = rhis_df[(col, rhis_stat)].plot(figsize=(16, 8), color='b', alpha=1, linestyle='-', linewidth=1)
        evol_ax.set_xlabel('Date')
        plot(evol_ax, col)
        hypotheses = ['R', 'H', 'I', 'S']
        colors_default = {'R': 'm', 'H': 'c', 'I': 'r', 'S': 'b'}
        for i in range(len(hypotheses)):
            ax = rhis_df[(col, hypotheses[i])].plot(
                figsize=(16, 8),
                color=colors_default[hypotheses[i]],
                alpha=0.4,
                linestyle='-',
                linewidth=1)

        ax.set_xlabel('Date')
        plot(ax, col)

if __name__ == "__main__":
    main()
