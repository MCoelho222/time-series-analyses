from __future__ import annotations

from typing import TYPE_CHECKING, Iterable

import numpy as np
import pandas as pd
from loguru import logger
from pandas import DataFrame

from rhis_ts.controllers.rhis_controller import build_init_evol_df, insert_repr_in_df_from_idx
from rhis_ts.evol.errors.exceptions import raise_start_end_evol_not_performed
from rhis_ts.evol.methods.fixed_start_evol import rhis_evol_fixed_start
from rhis_ts.evol.methods.repr_slice import cut_idx_for_representative
from rhis_ts.evol.methods.start_end_evol import restart_evol_on_rhis_reject
from rhis_ts.evol.plot.standard_evol_plot import plot_standard_evol
from rhis_ts.evol.plot.start_end_evol_plot import plot_start_end_evol
from rhis_ts.utils.data import slice_init

if TYPE_CHECKING:
    from pandas import Series


class RHIS:

    def __init__(self, df):
        self.raw = False
        self.directions = ('ba', 'fo')
        self.alpha = 0.05

        self.orig_df = df
        self.evol_df = None
        self.evol_df_raw = None
        self.len_df = len(self.orig_df)

        self.slice_init = slice_init(self.len_df)

        self.repr_idxs = {}
        self.repr_df = None

        self.start_end_evol = False


    def evol(self,
            cols: Iterable[str]|None=None,
            alpha: float=0.05,*,
            raw: bool=False,
            start_end_evol: bool=False
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

                evol = {
                    'r': [0.556, 0.265, 0.945, 0.159],
                    'h': [0.112, 0.232, 0.284, 0.492],
                    'i': [0.253, 0.022, 0.248, 0.995],
                    's': [0.534, 0.003, 0.354, 0.009],
                }

        """
        mode = 'RAW' if raw else 'STANDARD'
        msg = f"Processing RHIS evolution in {mode} mode..."
        logger.info(msg)
        self.raw = raw
        self.alpha = alpha
        self.start_end_evol = start_end_evol

        try:
            if raw:
                self.evol_df_raw = build_init_evol_df(self, self.orig_df.columns, self.orig_df.index, raw=raw)
            else:
                self.evol_df = build_init_evol_df(self, self.orig_df.columns, self.orig_df.index, raw=raw)
        except AttributeError:
            msg = "The parameter df should be an instance of pandas.DataFrame."
            logger.exception(msg)

        try:
            evol_cols = cols if cols is not None else self.orig_df.columns
            for col in evol_cols:
                ts = self.orig_df[col]
                if start_end_evol:
                    self._evol_ts_start_end(ts, alpha)
                else:
                    self._evol_ts_fixed_start(ts, alpha)

            evol_df = self.evol_df[evol_cols] if self.evol_df is not None else self.evol_df_raw[evol_cols]

        except (ValueError, KeyError):
            msg = "The evolution process could not be complete."
            logger.exception(msg)
            return

        logger.info("RHIS evolution successfully complete.")

        return evol_df


    def _evol_ts_fixed_start(self, ts: Series, alpha: float=0.05) -> dict[str, list[float | int]]:
        ts_arr = ts.to_numpy()

        fo_evol = rhis_evol_fixed_start(ts_arr, alpha, self.slice_init, raw=self.raw)
        ba_evol = rhis_evol_fixed_start(ts_arr[::-1], alpha, self.slice_init, raw=self.raw, ba=True)

        if self.raw:
            for hyp, ps in fo_evol.items():
                self.evol_df_raw[(ts.name, 'fo', hyp)] = ps
            for hyp, ps in ba_evol.items():
                self.evol_df_raw[(ts.name, 'ba', hyp)] = ps
        else:
            self.evol_df[(ts.name, 'fo')] = fo_evol
            self.evol_df[(ts.name, 'ba')] = ba_evol


    def _evol_ts_start_end(
            self,
            ts: Series,
            alpha: float=0.05,
            ) -> dict[str, list[float | int]]:

        ts_arr = ts.to_numpy()

        evol = restart_evol_on_rhis_reject(ts_arr, alpha, self.slice_init)
        # TODO (Marcelo Coelho): ba  # noqa: TD003
        self.repr_idxs[ts.name] = evol[0]
        p_evol = evol[1]

        self.evol_df[(ts.name, 'fo')] = p_evol
        self.evol_df[(ts.name, 'ba')] = np.nan

        return


    def build_start_end_evol_repr_df(self,):
        logger.info("Building representative dataframe from start_end_evol RHIS evolution...")
        raise_start_end_evol_not_performed(start_end_evol=self.start_end_evol)

        orig_cols = self.orig_df.columns
        repr_cols = []
        for orig_col in orig_cols:
            repr_cols.extend([(orig_col, f"{idx[0]}_{idx[1]}") for idx in self.repr_idxs[orig_col]])

        repr_df_cols = pd.MultiIndex.from_tuples(repr_cols)
        self.repr_df = pd.DataFrame(columns=repr_df_cols, index=self.orig_df.index)

        for col, idx in repr_cols:
            idxs = idx.split('_')
            start = int(idxs[0])
            end = int(idxs[1])
            self.repr_df[(col, idx)] = self.orig_df.loc[start: end, col].reindex(self.repr_df.index)

        logger.info("Representative dataframe complete.")

        return self.repr_df


    def add_repr_cols_to_df(self,) -> DataFrame:
        logger.info("Adding new columns to the original dataframe with representative slices - FLEX START mode...")
        if self.evol_df is None:
            self.evol()

        orig_cols = self.orig_df.columns

        for orig_col in orig_cols:
            evol_df_cols = self.evol_df.columns
            filtered_evol_df = self.evol_df[[col for col in evol_df_cols if col[0] == orig_col]]

            evol_bafo = {}

            for direct in self.directions:
                evol_bafo[direct] = filtered_evol_df[(orig_col, direct)].to_numpy()

            cut_idxs = cut_idx_for_representative(evol_bafo['ba'], self.alpha, self.slice_init)
            insert_repr_in_df_from_idx(self, cut_idxs, orig_col)

            self.repr_df = True

        logger.info("Representative slices successfully included in original dataframe.")

        return self.orig_df


    def plot(self,*, ba: bool=True, fo: bool=True, use_raw: bool=False):
        if self.start_end_evol:
            plot_start_end_evol(self)
        else:
            plot_standard_evol(self, ba=ba, fo=fo, use_raw=use_raw)

if __name__ == '__main__':

    # df = pd.read_csv('./data/BigSiouxAnnualQ.csv')
    # df = pd.read_csv('./data/MarchMilwaukeeChloride.csv')
    # df['Time'] = pd.to_datetime(df['Time'].astype(int), format='%Y')
    # df.set_index('Time', inplace=True)

    # rhis = RHIS(df, index_col=)

    # UFPR DATASET
    df = pd.read_excel('./data/dataset.xlsx')
    df[df.select_dtypes(include=['object']).columns.drop('PONTO')] = \
        df.select_dtypes(include=['object']).drop(columns='PONTO').apply(pd.to_numeric, errors='coerce')
    df = df.loc[df['PONTO'] == 'IG5',
                ['DATA', 'TURB (NTU)', 'TP (mg/L)']] # 'TP (mg/L)', 'COLIF_T (NMP/100mL)',
    df['DATA'] = pd.to_datetime(df['DATA'])
    df.set_index('DATA', inplace=True)
    df.dropna(inplace=True)

    # yearly_range = pd.date_range(start='2000-01-01', end='2013-01-01', freq='YE')
    # monthly_range = pd.date_range(start='2013-02-20', end='2013-12-20', freq='ME')
    # index = yearly_range.union(monthly_range)
    # index = pd.date_range(start='2000-12-31', end='2022-12-31', freq='YE')

    # ts = [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 4, 2, 5, 3, 10, 9, 9.5, 3.4, 5.7, 2.5, 7, 4.3, \
    #       11, 5, 5, 5, 5, 5, 5, 5, 5, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100]

    # df = pd.DataFrame({'data': ts})

    rhis = RHIS(df)
    evol_df = rhis.evol(raw=False)
    # print(evol_df)
    # print(rhis.orig_df)
    # rhis_repr_df = rhis.select_repr()
    # rhis.build_start_end_evol_repr_df()
    rhis.add_repr_cols_to_df()
    rhis.plot(ba=True, fo=False, use_raw=False)
