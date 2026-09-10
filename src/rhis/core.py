from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from loguru import logger
from pandas import DataFrame, Index

from rhis.exceptions import raise_if_no_rhis_run
from rhis.utils import calculate_rhis, nans_nums_from_array, slice_init, slices_to_evol

if TYPE_CHECKING:
    from pandas import Series

    from rhis.custom_types import RhisCode, RhisStat


class Rhis:
    def __init__(self, df):
        if (not isinstance(df, pd.DataFrame) or isinstance(df.index, pd.MultiIndex)):
            msg = "The parameter 'df' must be a non-MultiIndex pandas.DataFrame."
            logger.debug(msg)
            raise ValueError(msg)

        self.orig_df = df
        self.rhis_df = None
        self.rhis_stats_included = False
        self.is_rhis_complete = False
        self.alpha = 0.05

        self.length_init_ts = slice_init(len(self.orig_df))


    def _build_rhis_initial_df(self, df_cols: list[str], df_index: Index | None, *, include_rhis_stats: bool) -> DataFrame:
        rhis = ['R', 'H', 'I', 'S']
        if include_rhis_stats:
            rhis.extend(['min', 'avg', 'med', 'max'])
        cols = [(col, hyp) for col in df_cols for hyp in rhis]
        multi_index_cols = pd.MultiIndex.from_tuples(cols)
        result_df = pd.DataFrame(columns=multi_index_cols, index=df_index)

        return result_df


    def _include_rhis_compliant_ts_in_df(self, df: DataFrame, idx: tuple, df_col: str):
        orig_ts = df[df_col].to_numpy()
        nums_ts = orig_ts[idx[0]:idx[1]]
        nan_init = np.full(idx[0], np.nan)
        nan_fin= np.full(len(orig_ts) - idx[1], np.nan)
        full_ts = np.append(nan_init, nums_ts)
        full_ts = np.append(full_ts, nan_fin)

        df.loc[:, df_col + '_repr'] = full_ts


    def _retrieve_rhis_ts_idxs(self, ps: np.ndarray[float], alpha: float, length_init_ts: int) -> tuple[int]:
        ps_nums = nans_nums_from_array(ps)

        data = ps[:]
        alpha_arr = np.full(len(data), alpha)
        idx = 0
        is_rejection = data <= alpha_arr

        while is_rejection[idx]:
            if idx == len(data) - 1:
                break
            idx += 1

        ps_last = len(ps_nums) + length_init_ts - 1
        if ps[0] >= alpha:
            return (0, ps_last)

        return idx, ps_last


    def _include_rhis_stats_in_df(self, df: DataFrame):
        col_groups = [df.columns[i:i + 4] for i in range(0, len(df.columns), 4)]
        for group in col_groups:
            df[(group[0][0], "min")] = df[group].min(axis=1)
            df[(group[0][0], "mean")] = df[group].mean(axis=1)
            df[(group[0][0], "median")] = df[group].median(axis=1)
            df[(group[0][0], "max")] = df[group].max(axis=1)


    def _rhis_evol_raw(self, ts: Series, alpha: float, length_init_ts: int) -> dict[list[float]]:
        ts = ts.to_numpy()[::-1]
        slices = slices_to_evol(ts, length_init_ts)
        evol = {'R': [], 'H': [], 'I': [], 'S': []}

        for sli in slices:
            r, h, i, s = calculate_rhis(sli, alpha, min=False)
            evol['R'].append(r)
            evol['H'].append(h)
            evol['I'].append(i)
            evol['S'].append(s)

        fill = np.full(length_init_ts - 1, np.nan)

        for hyp, ps in evol.items():
            evol[hyp] = np.append(ps[::-1], fill)

        return evol


    def _add_rhis_stats_to_evol(self, evol_dict: dict[list[float]]) -> dict[list[float]]:
        stats_dict = {'min': np.min, 'med': np.median, 'avg': np.mean, 'max': np.max}
        for name, method in stats_dict.items():
            evol_dict[name] = method(list(evol_dict.values()), axis=0, keepdims=True).ravel()

        return evol_dict


    def _ts_evol(self, ts: Series,*, include_rhis_stats: bool):
        evol = self._rhis_evol_raw(ts, self.alpha, self.length_init_ts)
        if include_rhis_stats:
            evol = self._add_rhis_stats_to_evol(evol)
        for hyp, ps in evol.items():
            self.rhis_df[(ts.name, hyp)] = ps


    def evol(self, cols: tuple[str] | None=None, length_init_ts: int | None=None,*, include_rhis_stats: bool=True) -> DataFrame:
        """
        Generate a dataframe (self.rhis_statistic_df or self.rhis_full_df) with the series from
        the evolutional application of the randomness, homogeneity, independence and
        stationarity (rhis) tests to the time series in the original dataframe
        (self.orig_df).

        Parameters
        ----------
            cols
                An Iterable with string representing the columns' names to be analyzed.
            stat
                One of ['min', 'med', 'max', None]. The statistic to be applied to the rhis
                evolution. For example, if 'min', the minimum p-value among the rhis p-values
                is used, and self.rhis_statistic_df is created.
            alpha
                The significance level.

        Return
        ------
            DataFrame with p-values evolution
        """
        if length_init_ts is not None:
            self.length_init_ts = length_init_ts

        msg = "Generating RHIS series..."
        logger.info(msg)

        evol_cols = cols if cols is not None else self.orig_df.columns
        self.rhis_df = self._build_rhis_initial_df(evol_cols, self.orig_df.index, include_rhis_stats=include_rhis_stats)
        for col in evol_cols:
            ts = self.orig_df[col]
            self._ts_evol(ts, include_rhis_stats=include_rhis_stats)
        if include_rhis_stats:
            self.rhis_stats_included = True

        logger.info("RHIS completed successfully.")
        self.is_rhis_complete = True

        return self.rhis_df


    def add_rhis_compliant_to_df(self, rhis_stat: RhisStat | RhisCode='min') -> DataFrame:
        raise_if_no_rhis_run(is_rhis_complete=self.is_rhis_complete)
        cols_orig_df = self.orig_df.columns
        if rhis_stat in ['min', 'max', 'mean', 'median'] and not self.rhis_stats_included:
            print("passes")
            self._include_rhis_stats_in_df(self.rhis_df)

        for col in cols_orig_df:
            target_col = (col, rhis_stat)
            rhis_series = self.rhis_df[target_col].to_numpy()
            cut_idxs = self._retrieve_rhis_ts_idxs(rhis_series, self.alpha, self.length_init_ts)
            self._include_rhis_compliant_ts_in_df(self.orig_df, cut_idxs, col)

        logger.info("RHIS compliant data successfully included in the dataframe.")
        return self.orig_df

if __name__ == '__main__':

    df = pd.read_csv('./data/MarchMilwaukeeChloride.csv')
    df['Time'] = pd.to_datetime(df['Time'].astype(int), format='%Y')
    df.set_index('Time', inplace=True)

    rhis = Rhis(df)
    rhis.evol(include_rhis_stats=False)
    rhis.add_rhis_compliant_to_df('min')
    print(rhis.orig_df.info())
    print(rhis.rhis_df.info())

