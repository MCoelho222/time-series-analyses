from __future__ import annotations

from typing import TYPE_CHECKING, Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from loguru import logger

from rhis_timeseries.evolution.data import slices_to_evol
from rhis_timeseries.evolution.errors import raise_incorrect_mode_raw, raise_no_evol_process_ran
from rhis_timeseries.evolution.methods.select_representative import calculate_representative_range
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
        self.directions = ('bw', 'fw')

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
        self.orig_df = self.orig_df if target_cols is None else self.orig_df[target_cols]
        if index_col is not None:
            self.orig_df.set_index(index_col, inplace=True)

        self.mode = mode
        self.alpha = alpha

        orig_cols = self.orig_df.columns

        self.evol_df = self.__build_init_df(orig_cols, self.orig_df.index)

        for ts_name in orig_cols:
            ts = self.orig_df[ts_name]
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

        slice_init = None

        for direction in self.directions:
            data = ts_arr[:] if direction == 'fw' else ts_arr[::-1]
            slices = slices_to_evol(data)
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

                if self.directions[i] == 'bw':
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

                if self.directions[i] == 'bw':
                    ps_filled = ps_filled[::-1]

                result[self.directions[i]] = ps_filled

        df = self.__build_init_df([ts.name], ts.index)

        for direct in self.directions:
            if mode == 'raw':
                for col in df.columns:
                    df[(ts.name, direct, col[2])] = result[direct][col[2]]
            else:
                df[(ts.name, direct)] = result[direct]
        return df


    def repr_slices(self,) -> DataFrame:
        logger.info("Selecting representative data...")
        raise_incorrect_mode_raw(self.mode)
        raise_no_evol_process_ran(self.evol_df)

        orig_cols = self.orig_df.columns

        for orig_col in orig_cols:
            evol_df_cols = self.evol_df.columns
            filtered_evol_df = self.evol_df[[col for col in evol_df_cols if col[0] == orig_col]]

            evol_tss = {}

            for direct in self.directions:
                evol_tss[direct] = filtered_evol_df[(orig_col, direct)].to_numpy()

            ranges = calculate_representative_range(bw=evol_tss['bw'], fw=evol_tss['fw'], most_recent=True, alpha=self.alpha)
            self.__create_repr_series_from_rejections(ranges, orig_col)
            self.repr_df = True

        logger.info("Representative data complete.")
        return self.orig_df


    def __create_repr_series_from_rejections(self, ranges: dict[tuple], df_col: str):
        st_r = ranges['start_range']
        ext_r = ranges['extension_range']

        orig_ts = self.orig_df[df_col].to_numpy()

        nums_st_r_ts = orig_ts[st_r[0]:st_r[1]]

        st_r_nan_range = len(orig_ts) - len(nums_st_r_ts)

        if st_r_nan_range > 0:
            st_r_ts_nans = np.full(st_r_nan_range, np.nan)
            full_st_r_ts = np.append(st_r_ts_nans, nums_st_r_ts)
            self.orig_df.loc[:, df_col + '_repr'] = full_st_r_ts
        else:
            self.orig_df.loc[:, df_col + '_repr'] = orig_ts

        if ext_r is not None:
            nums_ext_r_ts = orig_ts[ext_r[0]:ext_r[1]]
            ext_r_end_nans = np.full(len(nums_st_r_ts), np.nan)
            ext_r_start_nans = np.full(ext_r[0], np.nan)
            full_ext_r_ts = np.append(ext_r_start_nans, nums_ext_r_ts)
            full_ext_r_ts = np.append(full_ext_r_ts, ext_r_end_nans)
            self.orig_df.loc[:, df_col + '_repr_ext'] = full_ext_r_ts


    def __build_init_df(self, orig_colnames: list[str], index: Index) -> DataFrame:
        cols_tuples = []
        for col in orig_colnames:
            for direct in self.directions:
                if self.mode == 'raw':
                    hyps = ['R', 'H', 'I', 'S']
                    for hyp in hyps:
                        cols_tuples.append((col, direct, hyp))
                else:
                    cols_tuples.append((col, direct))

        cols = pd.MultiIndex.from_tuples(cols_tuples)
        result_df = pd.DataFrame(columns=cols, index=index)

        return result_df


    def plot(self,):
        markersize = 80
        column = None
        for col, _ in self.evol_df.columns:
            if col != column:
                column = col
                ax1 = self.evol_df[[(col, 'bw'), (col, 'fw')]].plot(color='k', alpha=0.1)
                ax1.axhline(y=0.05, color='r', linestyle='--')
                ax2 = ax1.twinx()
                ax2.scatter(
                    x=self.orig_df.index,
                    y=self.orig_df[col],
                    label=col,
                    color='k',
                    alpha=0.3,
                    edgecolors='none',
                    s=markersize)
                if self.repr_df:
                    ax2.scatter(
                        x=self.orig_df.index,
                        y=self.orig_df[col + '_repr'],
                        label=col + '_repr',
                        color='tab:green',
                        edgecolors='none',
                        s=markersize)
                    try:
                        ax2.scatter(
                            x=self.orig_df.index,
                            y=self.orig_df[col + '_repr_ext'],
                            label=col + '_repr_ext',
                            color='orange',
                            edgecolors='none',
                            s=markersize)
                    except KeyError:
                        pass
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
    df = df.loc[df['PONTO'] == 'IG6', ['DATA', 'BOD (mg/L)', 'NH4 (mg/L)', 'DO (mg/L)', 'COND', 'T (°C)']]
    df.dropna(inplace=True)
    # ts = [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 4, 2, 5, 3, 10, 9, 9.5, 3.4, 5.7, 2.5, 7, 4.3, 11]
    # df = pd.DataFrame({'data': ts[::-1]})
    # print(df.info())
    # print(df.head())
    rhis_evol_df = rhis.evol(df=df, index_col='DATA', mode='min')
    rhis_repr_df = rhis.repr_slices()
    # print(rhis_evol_df)
    rhis.plot()
