

from __future__ import annotations

from typing import TYPE_CHECKING, Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from rhis_timeseries.evolution.data import slices_incr_len
from rhis_timeseries.hypothesis_tests.mann_kendall import mann_kendall
from rhis_timeseries.hypothesis_tests.mann_whitney import mann_whitney
from rhis_timeseries.hypothesis_tests.runs import runs_test
from rhis_timeseries.hypothesis_tests.wald_wolfowitz import wald_wolfowitz

if TYPE_CHECKING:
    from pandas import DataFrame, Index, Series


class RHIS:
    """
    """
    def __init__(self, df: DataFrame, index_col: str='Dates'):
        self.orig_df = df
        self.df_index = index_col
        self.orig_df.set_index(self.df_index, inplace=True)
        self.evol_df = None
        self.slice_init = None
        self.mode = None
        self.directions = ()

    def evol(self,
            directions: tuple[str]=('backward',),
            mode: str='raw',
            target_cols: Iterable[str] | None=None,
            slice_init: int=10,
            ) -> dict[str, list[float | int]]:
        """
        Calculate randomness, homogeneity, independence and stationarity (rhis)
        p-values for time series slices.

        The main purpose is to calculate rhis p-values for slices with increasing
        or decreasing length. The p-values of the different lengths will indicate
        where or with how much elements (from beginning to end or end to beginning)
        the series is no longer representative, due to the presence of some
        variability pattern, e.g., trends or seasonality.

        The first or last slice must have at least 5 elements to the tests to be
        performed.

        Example
        -------
            Slices with increasing length

                ts = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
                slices = [ts[:5], ts[:6], ts[:7], ts[:8], ts[:9], ts]

            Slices with decreasing length

                ts = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
                slices = [ts, ts[:9], ts[:8], ts[:7], ts[:6], ts[:5]]

        Parameters
        ----------
            slices
                A list with slices from another list with float or integers.

        Return
        ------
            A dictionary with rhis p-values

            Example:

                rhis_evol = {
                    'r': [0.556, 0.265, 0.945, 0.159],
                    'h': [0.112, 0.232, 0.284, 0.492],
                    'i': [0.253, 0.022, 0.248, 0.995],
                    's': [0.534, 0.003, 0.354, 0.009],
                }

        """
        self.slice_init = slice_init
        self.directions = directions
        self.mode = mode

        self.orig_df = self.orig_df if target_cols is None else self.orig_df[target_cols]
        orig_df = self.orig_df
        orig_cols = orig_df.columns

        self.evol_df = self.__build_col_names(orig_cols, orig_df.index, mode, directions)

        for ts_name in orig_cols:
            ts = orig_df[ts_name]
            rhis_evol = self.evol_ts(ts, directions, mode, slice_init)

            for direct in directions:
                if mode == 'raw':
                    for col in self.evol_df.columns:
                        self.evol_df[(ts_name, direct, col[2])] = rhis_evol[(ts_name, direct, col[2])]
                else:
                    self.evol_df[(ts.name, direct)] = rhis_evol[(ts.name, direct)]

        return self.evol_df


    def evol_ts(  # noqa: C901, PLR0912
            self,
            ts: Series,
            directions: tuple[str],
            mode: str = 'raw',
            slice_init: int = 10,
            ) -> dict[str, list[float | int]]:

        ts_arr = ts.to_numpy()
        dir_slices = []
        for direction in directions:
            data = ts_arr[:] if direction == 'forward' else ts_arr[::-1]
            slices = slices_incr_len(data, slice_init)
            dir_slices.append(slices)

        nan_arr = np.full(9, np.nan)
        hyps = ['R', 'H', 'I', 'S']
        result = {}
        for i in range(len(dir_slices)):
            ps = [[], [], [], []]
            for ts_slice in dir_slices[i]:
                ps[0].append(runs_test(ts_slice).p_value)
                ps[1].append(mann_whitney(ts_slice).p_value)
                ps[2].append(wald_wolfowitz(ts_slice).p_value)
                ps[3].append(mann_kendall(ts_slice).p_value)

            if mode == 'raw':
                nan_arr = np.full((4, slice_init - 1), np.nan)
                ps_filled = np.concatenate([nan_arr, ps], axis=1)

                if directions[i] == 'backward':
                    ps_filled = [arr[::-1] for arr in ps_filled]

                rhis_evol = dict(zip(hyps, ps_filled))
                result[directions[i]] = rhis_evol

            else:
                if mode == 'median':
                    rhis_evol = np.median(ps, axis=0, keepdims=True).ravel()
                if mode == 'mean':
                    rhis_evol = np.mean(ps, axis=0, keepdims=True).ravel()

                ps_filled = np.concatenate([nan_arr, rhis_evol])

                if directions[i] == 'backward':
                    ps_filled = ps_filled[::-1]

                result[directions[i]] = ps_filled

        df = self.__build_col_names([ts.name], ts.index, mode, directions)

        for direct in directions:
            if mode == 'raw':
                for col in df.columns:
                    df[(ts.name, direct, col[2])] = result[direct][col[2]]
            else:
                df[(ts.name, direct)] = result[direct]

        return df


    def __build_col_names(self, orig_colnames: list[str], index: Index,  mode: str, directions: list[str]) -> DataFrame:
        cols_tuples = []
        for col in orig_colnames:
            for direct in directions:
                if mode == 'raw':
                    hyps = ['R', 'H', 'I', 'S']
                    for hyp in hyps:
                        cols_tuples.append((col, direct, hyp))
                else:
                    cols_tuples.append((col, direct))

        cols = pd.MultiIndex.from_tuples(cols_tuples)
        result_df = pd.DataFrame(columns=cols, index=index)

        return result_df


    def repr_slices(self,) -> DataFrame:
        if self.evol_df is None:
            error_msg = "No evolution process has been performed yet."
            raise ValueError(error_msg)

        if 'backward' not in self.directions or 'forward' not in self.directions:
            error_msg = "This function should be called only after a \
                bidirectional (backward and forward) evolution is performed."
            raise RuntimeError(error_msg)

        if self.mode == 'raw' or self.mode is None:
            error_msg = "This function works only for 'median' and 'mean' modes only."
            raise ValueError(error_msg)

        orig_cols = self.orig_df.columns
        print(orig_cols)
        for orig_col in orig_cols:
            filtered_evol_df = self.evol_df[[col for col in self.evol_df.columns if col[0] == orig_col]]

            bw_ps = filtered_evol_df[(orig_col, 'backward')].to_numpy()
            fw_ps = filtered_evol_df[(orig_col, 'forward')].to_numpy()

            n_fill = self.slice_init - 1
            mask = bw_ps[n_fill:-n_fill] > fw_ps[n_fill:-n_fill]

            if mask[-1]:
                bw_fill = np.ones(n_fill)
            else:
                bw_fill = np.zeros(n_fill)

            if mask[0]:
                fw_fill = np.ones(n_fill)
            else:
                fw_fill = np.zeros(n_fill)

            mask = np.append(mask, bw_fill)
            mask = np.append(fw_fill, mask)

            last_index = len(filtered_evol_df)

            cut_index = -1
            while mask[last_index + cut_index]:
                if abs(cut_index) == last_index:
                    break
                cut_index -= 1

            repr_data = self.orig_df[orig_col].to_numpy()[cut_index:]
            n_full = last_index - len(repr_data)
            repr_data = np.append(np.full(n_full, np.nan), repr_data)
            self.orig_df[orig_col + '_repr'] = repr_data

        return self.orig_df


if __name__ == '__main__':

    # df = pd.read_csv('./data/BigSiouxAnnualQ.csv')
    df = pd.read_csv('./data/MarchMilwaukeeChloride.csv')
    # rhis = RHIS(df, index_col='T')
    rhis = RHIS(df, index_col='Time')

    rhis_evol_df = rhis.evol(('backward', 'forward'), 'median', ['Q', 'Conc'])
    print(rhis_evol_df.head())
    rhis_repr_df = rhis.repr_slices()
    print(rhis_repr_df.head())
    ax1 = rhis_evol_df.plot()
    ax2 = ax1.twinx()
    for col in rhis_repr_df.columns:
        ax2.scatter(x=rhis_repr_df.index, y=rhis_repr_df[col], label=col)
    ax1.get_legend().remove()
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2)
    plt.show()
