from __future__ import annotations

from typing import TYPE_CHECKING, Iterable

import pandas as pd
from loguru import logger
from pandas import DataFrame

from rhis_ts.controllers.rhis_controller import RhisController
from rhis_ts.evol.methods.repr_slice import repr_slice_idxs
from rhis_ts.evol.methods.standard_evol import rhis_standard_evol
from rhis_ts.evol.plot.plot_standard_evol import finalize_plot, plot_data, plot_rhis_evol, plot_rhis_stats_evol
from rhis_ts.utils.data import slice_init

if TYPE_CHECKING:
    from pandas import Series


class Rhis:

    def __init__(self, df):
        self.alpha = 0.05
        self.rhis = None
        self.stat = None
        self.direction = None
        self.orig_df = df
        self.evol_df = None
        self.evol_df_rhis = None
        self.slice_init = slice_init(len(self.orig_df))


    def evol(self,
            cols: Iterable[str]|None=None,
            stat: str='min',
            alpha: float=0.05,*,
            rhis: bool=False
            ) -> DataFrame:
        """
        Generate a dataframe (self.evol_df or self.evol_df_rhis) with the series from
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
                is used, and self.evol_df is created.
            alpha
                The significance level.
            rhis
                If True, the pure rhis evolution is performed and self.evol_df_rhis is created.

        Return
        ------
            DataFrame with p-values evolution
        """
        mode = 'RHIS' if rhis else 'STAT'
        msg = f"Processing RHIS evol in {mode.upper()} mode..."
        logger.info(msg)
        self.alpha = alpha
        self.rhis = rhis
        self.stat = stat
        try:
            init_df = RhisController.build_init_evol_df(self.orig_df.columns, self.orig_df.index, rhis=self.rhis)
            if rhis:
                self.evol_df_rhis = init_df
            else:
                self.evol_df = init_df
        except AttributeError:
            msg = "The parameter df should be an instance of pandas.DataFrame."
            logger.exception(msg)
        try:
            evol_cols = cols if cols is not None else self.orig_df.columns
            for col in evol_cols:
                ts = self.orig_df[col]
                self._ts_evol(ts, alpha)
            evol_df = self.evol_df[evol_cols] if self.evol_df is not None else self.evol_df_rhis[evol_cols]
        except (ValueError, KeyError):
            msg = "Evol process could not be complete."
            logger.exception(msg)

            return
        logger.info("RHIS evol successfully complete.")
        return evol_df


    def _ts_evol(self, ts: Series, alpha: float=0.05):
        ts_arr = ts.to_numpy()
        fo_evol = rhis_standard_evol(ts_arr, alpha, self.slice_init, self.stat, rhis=self.rhis)
        ba_evol = rhis_standard_evol(ts_arr[::-1], alpha, self.slice_init, self.stat, rhis=self.rhis, ba=True)
        if self.rhis:
            for hyp, ps in fo_evol.items():
                self.evol_df_rhis[(ts.name, 'fo', hyp)] = ps
            for hyp, ps in ba_evol.items():
                self.evol_df_rhis[(ts.name, 'ba', hyp)] = ps
        else:
            self.evol_df[(ts.name, 'fo')] = fo_evol
            self.evol_df[(ts.name, 'ba')] = ba_evol


    def add_repr_cols_to_df(self, direction: str='ba') -> DataFrame:
        logger.info("Adding representative data to the original dataframe...")

        self.direction = direction
        if self.evol_df is None:
            self.evol(stat=self.stat)

        orig_cols = self.orig_df.columns
        for orig_col in orig_cols:
            evol_df_cols = self.evol_df.columns
            filtered_evol_df = self.evol_df[[col for col in evol_df_cols if col[0] == orig_col]]
            evol_bafo = {}
            for direct in ('ba', 'fo'):
                evol_bafo[direct] = filtered_evol_df[(orig_col, direct)].to_numpy()
            cut_idxs = repr_slice_idxs(evol_bafo[direction], self.alpha, self.slice_init, self.direction)
            RhisController.insert_repr_in_df_from_idx(self.orig_df, cut_idxs, orig_col)

        logger.info("Representative data successfully added.")
        return self.orig_df


    def plot(self, savefig_path: str | None=None,*, rhis: bool=False, show_repr: bool=True):
        cols = set()
        for col, _ in self.evol_df.columns:
            cols.add(col)

        for col in cols:
            if rhis:
                rhis_ax = plot_rhis_evol(
                    col,
                    self.evol_df_rhis,
                    self.direction,
                    )
                data_ax = plot_data(rhis_ax, col, self.orig_df, show_repr=show_repr)
                finalize_plot(rhis_ax, data_ax, col, self.direction)

            stat_ax = plot_rhis_stats_evol(col, self.evol_df, self.direction)
            data_ax1 = plot_data(stat_ax, col, self.orig_df, show_repr=show_repr)
            finalize_plot(stat_ax, data_ax1, col, self.direction, savefig_path)


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

    rhis = Rhis(df)
    evol_df = rhis.evol(stat='min', rhis=False)
    # print(evol_df)
    # print(rhis.orig_df)
    # rhis_repr_df = rhis.select_repr()
    # rhis.build_start_end_evol_repr_df()
    rhis.add_repr_cols_to_df(direction='fo')
    rhis.plot(rhis=False, show_repr=True)
