from __future__ import annotations

from typing import TYPE_CHECKING, Iterable

import numpy as np
import pandas as pd
from loguru import logger

from rhis_ts.controllers.rhis_controller import build_bafo_init_df, insert_repr_in_df_from_idx
from rhis_ts.evol.bafo.no_memo import rhis_evol_no_memo
from rhis_ts.evol.bafo.part_memo import bafo_weak_memo
from rhis_ts.evol.exc.exc import raise_incorrect_mode_raw, raise_no_evol_process_ran, raise_no_memo_not_performed
from rhis_ts.evol.plot.no_memo_plot import no_memo_plot
from rhis_ts.evol.plot.weak_memo_plot import weak_memo_plot
from rhis_ts.evol.repr.get_idxs import repr_idxs_from_ba
from rhis_ts.utils.data import slice_init

if TYPE_CHECKING:
    from pandas import DataFrame, Series


class RHIS:

    def __init__(self, df):
        self.raw = False
        self.alpha = 0.05
        self.directions = ('ba', 'fo')
        self.slice_init = None

        self.orig_df = df
        self.evol_df = build_bafo_init_df(self, self.orig_df.columns, self.orig_df.index)
        self.repr_idxs = {}
        self.repr_df = None

        self.no_memo = False

    def evol(self,
            cols: Iterable[str]|None=None,
            alpha: float=0.05,*,
            raw: bool=False,
            no_memo: bool=True
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

        self.alpha = alpha
        evol_cols = cols if cols is not None else self.orig_df.columns

        self.raw = raw
        self.no_memo = no_memo

        for col in evol_cols:
            ts = self.orig_df[col]
            if no_memo:
                self._evol_ts_no_memo(ts, alpha)
            else:
                self._evol_ts_weak_memo(ts, alpha, raw=raw)

        logger.info("RHIS evolution on dataframe complete.")

        return self.evol_df[cols] if cols is not None else self.evol_df


    def _evol_ts_weak_memo(
            self,
            ts: Series,
            alpha: float=0.05,*,
            raw: bool=False
            ) -> dict[str, list[float | int]]:

        ts_arr = ts.to_numpy()

        fo_data = bafo_weak_memo(ts_arr, alpha, raw=raw)
        ba_data = bafo_weak_memo(ts_arr[::-1], alpha, raw=raw, ba=True)
        fo_evol = fo_data[0]
        ba_evol = ba_data[0]

        self.slice_init = fo_data[1]

        if raw:
            for hyp, ps in fo_evol.items():
                self.evol_df[(ts.name, 'fo', hyp)] = ps
            for hyp, ps in ba_evol.items():
                self.evol_df[(ts.name, 'ba', hyp)] = ps
        else:
            self.evol_df[(ts.name, 'fo')] = fo_evol
            self.evol_df[(ts.name, 'ba')] = ba_evol

        return


    def _evol_ts_no_memo(
            self,
            ts: Series,
            alpha: float=0.05,
            ) -> dict[str, list[float | int]]:

        self.slice_init = slice_init(len(ts))

        ts_arr = ts.to_numpy()

        evol = rhis_evol_no_memo(ts_arr, alpha, self.slice_init)
        # TODO (Marcelo Coelho): ba  # noqa: TD003
        self.repr_idxs[ts.name] = evol[0]
        p_evol = evol[1]

        self.evol_df[(ts.name, 'fo')] = p_evol
        self.evol_df[(ts.name, 'ba')] = np.nan

        return


    def build_no_memo_repr_df(self,):
        logger.info("Building representative dataframe from NO_memo RHIS evolution...")
        raise_no_evol_process_ran(self.evol_df)
        raise_no_memo_not_performed(no_memo=self.no_memo)

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


    def build_weak_memo_repr_df(self,) -> DataFrame:
        logger.info("Building representative dataframe from WEAK_memo RHIS evolution...")

        raise_incorrect_mode_raw(raw=self.raw)
        raise_no_evol_process_ran(self.evol_df)

        orig_cols = self.orig_df.columns

        for orig_col in orig_cols:
            evol_df_cols = self.evol_df.columns
            filtered_evol_df = self.evol_df[[col for col in evol_df_cols if col[0] == orig_col]]

            evol_bafo = {}

            for direct in self.directions:
                evol_bafo[direct] = filtered_evol_df[(orig_col, direct)].to_numpy()

            idx = repr_idxs_from_ba(evol_bafo['ba'], evol_bafo['fo'], self.alpha, self.slice_init)
            insert_repr_in_df_from_idx(self, idx, orig_col)

            self.repr_df = True

        logger.info("Representative dataframe complete.")

        return self.orig_df


    def plot(self,):
        if self.no_memo:
            no_memo_plot(self)
        else:
            weak_memo_plot(self)

if __name__ == '__main__':

    # df = pd.read_csv('./data/BigSiouxAnnualQ.csv')
    # df = pd.read_csv('./data/MarchMilwaukeeChloride.csv')
    # df['Time'] = pd.to_datetime(df['Time'].astype(int), format='%Y')
    # df.set_index('Time', inplace=True)

    # rhis = RHIS(df, index_col=)

    # UFPR DATASET
    # df = pd.read_excel('./data/dataset.xlsx')
    # df[df.select_dtypes(include=['object']).columns.drop('PONTO')] = \
    #     df.select_dtypes(include=['object']).drop(columns='PONTO').apply(pd.to_numeric, errors='coerce')
    # df = df.loc[df['PONTO'] == 'IG5',
    #             ['DATA', 'TURB (NTU)', 'TP (mg/L)']] # 'TP (mg/L)', 'COLIF_T (NMP/100mL)',
    # df['DATA'] = pd.to_datetime(df['DATA'])
    # df.set_index('DATA', inplace=True)
    # df.dropna(inplace=True)

    # yearly_range = pd.date_range(start='2000-01-01', end='2013-01-01', freq='YE')
    # monthly_range = pd.date_range(start='2013-02-20', end='2013-12-20', freq='ME')
    # index = yearly_range.union(monthly_range)
    # index = pd.date_range(start='2000-12-31', end='2022-12-31', freq='YE')
    ts = [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 4, 2, 5, 3, 10, 9, 9.5, 3.4, 5.7, 2.5, 7, 4.3, \
          11, 5, 5, 5, 5, 5, 5, 5, 5, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100]
    df = pd.DataFrame({'data': ts})

    rhis = RHIS(df)
    evol_df = rhis.evol()
    # print(rhis.orig_df)
    # rhis_repr_df = rhis.select_repr()
    rhis.build_no_memo_repr_df()
    rhis.plot()
