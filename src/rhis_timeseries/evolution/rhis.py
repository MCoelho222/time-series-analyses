from __future__ import annotations

from typing import TYPE_CHECKING, Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from loguru import logger

from rhis_timeseries.evolution.data import slices_incr_len
from rhis_timeseries.evolution.methods.select_representative import get_repr_index
from rhis_timeseries.hypothesis_tests.mann_kendall import mann_kendall
from rhis_timeseries.hypothesis_tests.mann_whitney import mann_whitney
from rhis_timeseries.hypothesis_tests.runs import wallismoore
from rhis_timeseries.hypothesis_tests.wald_wolfowitz import wald_wolfowitz

if TYPE_CHECKING:
    from pandas import DataFrame, Index, Series


class RHIS:
    """
    """
    def __init__(self,):
        self.orig_df = None
        self.evol_df = None
        self.mode = None
        self.alpha = 0.05
        self.repr_df = False
        self.directions = ('backward', 'forward')

    def evol(self,
            df: DataFrame,
            index_col: str|None=None,
            mode: str='raw',
            target_cols: Iterable[str]|None=None,
            alpha: float=0.05,
            ) -> dict[str, list[float|int]]:
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
        logger.info("Processing RHIS evolution on dataframe...")
        self.orig_df = df
        if index_col is not None:
            self.orig_df.set_index(index_col, inplace=True)
        self.mode = mode
        self.alpha = alpha

        self.orig_df = self.orig_df if target_cols is None else self.orig_df[target_cols]
        orig_df = self.orig_df
        orig_cols = orig_df.columns

        self.evol_df = self.__build_col_names(orig_cols, orig_df.index, mode)

        for ts_name in orig_cols:
            ts = orig_df[ts_name]
            rhis_evol = self.evol_ts(ts, mode, alpha)
            for direct in self.directions:
                if mode == 'raw':
                    for col in self.evol_df.columns:
                        self.evol_df[(ts_name, direct, col[2])] = rhis_evol[(ts_name, direct, col[2])]
                else:
                    self.evol_df[(ts.name, direct)] = rhis_evol[(ts.name, direct)]
        logger.info("RHIS evolution on dataframe complete.")

        return self.evol_df


    def evol_ts(  # noqa: C901, PLR0912
            self,
            ts: Series,
            mode: str = 'raw',
            alpha: float=0.05,
            ) -> dict[str, list[float | int]]:

        ts_arr = ts.to_numpy()
        dir_slices = []
        slice_init = 0
        for direction in self.directions:
            data = ts_arr[:] if direction == 'forward' else ts_arr[::-1]
            slices = slices_incr_len(data)
            slice_init = len(slices[0])
            dir_slices.append(slices)
        nan_arr = np.full(slice_init - 1, np.nan)
        hyps = ['R', 'H', 'I', 'S']
        result = {}
        for i in range(2):
            ps = [[], [], [], []]
            for ts_slice in dir_slices[i]:
                ps[0].append(wallismoore(ts_slice, alpha=alpha).p_value)
                ps[1].append(mann_whitney(ts_slice, alpha=alpha).p_value)
                ps[2].append(wald_wolfowitz(ts_slice, alpha=alpha).p_value)
                ps[3].append(mann_kendall(ts_slice, alpha=alpha).p_value)

            if mode == 'raw':
                nan_arr = np.full((4, slice_init - 1), np.nan)
                ps_filled = np.concatenate([nan_arr, ps], axis=1)

                if self.directions[i] == 'backward':
                    ps_filled = [arr[::-1] for arr in ps_filled]

                rhis_evol = dict(zip(hyps, ps_filled))
                result[self.directions[i]] = rhis_evol

            else:
                if mode == 'median':
                    rhis_evol = np.median(ps, axis=0, keepdims=True).ravel()
                if mode == 'mean':
                    rhis_evol = np.mean(ps, axis=0, keepdims=True).ravel()
                if mode == 'min':
                    rhis_evol = np.min(ps, axis=0, keepdims=True).ravel()

                ps_filled = np.concatenate([nan_arr, rhis_evol])

                if self.directions[i] == 'backward':
                    ps_filled = ps_filled[::-1]

                result[self.directions[i]] = ps_filled

        df = self.__build_col_names([ts.name], ts.index, mode)

        for direct in self.directions:
            if mode == 'raw':
                for col in df.columns:
                    df[(ts.name, direct, col[2])] = result[direct][col[2]]
            else:
                df[(ts.name, direct)] = result[direct]
        return df


    def __build_col_names(self, orig_colnames: list[str], index: Index,  mode: str) -> DataFrame:
        cols_tuples = []
        for col in orig_colnames:
            for direct in self.directions:
                if mode == 'raw':
                    hyps = ['R', 'H', 'I', 'S']
                    for hyp in hyps:
                        cols_tuples.append((col, direct, hyp))
                else:
                    cols_tuples.append((col, direct))

        cols = pd.MultiIndex.from_tuples(cols_tuples)
        result_df = pd.DataFrame(columns=cols, index=index)

        return result_df


    def __create_repr_series_from_rejections(self, cut_idx: tuple[int], df_col: str, mode: str):
        start = cut_idx[0]
        end = cut_idx[1]
        orig_ts = self.orig_df[df_col].to_numpy()
        repr_data = orig_ts[start:end]
        fill_arr = np.full(len(orig_ts) - len(repr_data), np.nan)
        repr_data = np.append(fill_arr, repr_data) if end == len(orig_ts) else np.append(repr_data, fill_arr)

        self.orig_df.loc[:, df_col + '_repr'] = repr_data


    def repr_slices(self,*, most_recent: bool=True) -> DataFrame:
        logger.info("Selecting representative data...")
        if self.evol_df is None:
            error_msg = "No evolution process has been performed yet."
            raise RuntimeError(error_msg)

        if 'backward' not in self.directions and most_recent:
            error_msg = "To use 'most_recent'=True, the 'directions' parameter " + \
                "of the 'RHIS.evol' method must include the option 'backward'."
            raise ValueError(error_msg)

        if 'forward' not in self.directions and not most_recent:
            error_msg = "To use 'most_recent'=False, the 'directions' parameter " + \
                "of the 'RHIS.evol' method must include the option 'forward'."
            raise ValueError(error_msg)

        if self.mode == 'raw' or self.mode is None:
            error_msg = "This function works only for 'median', 'mean' and 'min' modes only."
            raise ValueError(error_msg)

        orig_cols = self.orig_df.columns
        directions = self.directions
        direction = 'backward' if most_recent else 'forward'

        for orig_col in orig_cols:
            evol_df_cols = self.evol_df.columns
            filtered_evol_df = self.evol_df[[col for col in evol_df_cols if col[0] == orig_col]]

            evol_tss = {}

            for direct in directions:
                evol_tss[direct] = filtered_evol_df[(orig_col, direct)].to_numpy()

            cut_idx = get_repr_index(bw=evol_tss['backward'], fw=evol_tss['forward'], most_recent=True, alpha=self.alpha)
            self.__create_repr_series_from_rejections(cut_idx, orig_col, direction)
            self.repr_df = True

        logger.info("Representative data complete.")
        return self.orig_df


    def plot(self,):
        column = None
        for col, _ in self.evol_df.columns:
            if col != column:
                column = col
                ax1 = self.evol_df[[(col, 'backward'), (col, 'forward')]].plot()
                ax1.axhline(y=0.05, color='r', linestyle='--')
                ax2 = ax1.twinx()
                ax2.scatter(x=self.orig_df.index, y=self.orig_df[col], label=col)
                if self.repr_df:
                    ax2.scatter(x=self.orig_df.index, y=self.orig_df[col + '_repr'], label=col + '_repr')
                ax1.get_legend().remove()
                lines1, labels1 = ax1.get_legend_handles_labels()
                lines2, labels2 = ax2.get_legend_handles_labels()
                ax2.legend(lines1 + lines2, labels1 + labels2)
                plt.show()


if __name__ == '__main__':

    rhis = RHIS()
    # df = pd.read_csv('./data/BigSiouxAnnualQ.csv')
    # df = pd.read_csv('./data/MarchMilwaukeeChloride.csv')
    # rhis = RHIS(df, index_col='Time')

    # UFPR DATASET
    df = pd.read_excel('./data/dataset.xlsx')
    df[df.select_dtypes(include=['object']).columns.drop('PONTO')] = \
        df.select_dtypes(include=['object']).drop(columns='PONTO').apply(pd.to_numeric, errors='coerce')
    print(df.info())
    df = df.loc[df['PONTO'] == 'IG6', ['DATA', 'BOD (mg/L)', 'NH4 (mg/L)', 'DO (mg/L)']]
    df.dropna(inplace=True)
    # ts = [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 4, 2, 5, 3, 10, 9, 9.5, 3.4, 5.7, 2.5, 7, 4.3, 11]
    # df = pd.DataFrame({'data': ts[::-1]})
    # print(df.info())
    # print(df.head())
    rhis_evol_df = rhis.evol(df=df, index_col='DATA', mode='median')
    rhis_repr_df = rhis.repr_slices(most_recent=True)
    # print(rhis_evol_df)
    rhis.plot()
