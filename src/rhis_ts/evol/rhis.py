from __future__ import annotations

from typing import TYPE_CHECKING, Iterable

import matplotlib.pyplot as plt
import pandas as pd
from loguru import logger

from rhis_ts.controllers.rhis_controller import build_init_df, insert_repr_in_df_from_idx
from rhis_ts.evol.exc.exc import raise_incorrect_mode_raw, raise_no_evol_process_ran
from rhis_ts.evol.utils.part_memo import rhis_evol_with_part_memo
from rhis_ts.evol.utils.repr_select import repr_rng_memo_init_loss

if TYPE_CHECKING:
    from pandas import DataFrame, Series


class RHIS:

    def __init__(self,):
        self.orig_df = None
        self.evol_df = None
        self.raw = False
        self.alpha = 0.05
        self.repr_df = False
        self.directions = ('ba', 'fo')
        self.slice_init = None

    def evol(self,
            df: DataFrame,
            target_cols: Iterable[str]|None=None,
            alpha: float=0.05,*,
            raw: bool=False
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
        self.orig_df = df
        self.orig_df = self.orig_df if target_cols is None else self.orig_df[target_cols]

        if raw:
            self.raw = raw

        orig_cols = self.orig_df.columns

        self.evol_df = build_init_df(self, orig_cols, self.orig_df.index)

        for ts_name in orig_cols:
            ts = self.orig_df[ts_name]
            self.evol_ts(ts, alpha, raw=raw)

        logger.info("RHIS evolution on dataframe complete.")

        return self.evol_df


    def evol_ts(
            self,
            ts: Series,
            alpha: float=0.05,*,
            raw: bool=False
            ) -> dict[str, list[float | int]]:

        ts_arr = ts.to_numpy()

        fo_data = rhis_evol_with_part_memo(ts_arr, alpha, raw=raw)
        ba_data = rhis_evol_with_part_memo(ts_arr[::-1], alpha, raw=raw, ba=True)
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


    def select_repr(self,) -> DataFrame:
        logger.info("Selecting representative data...")

        raise_incorrect_mode_raw(raw=self.raw)
        raise_no_evol_process_ran(self.evol_df)

        orig_cols = self.orig_df.columns

        for orig_col in orig_cols:
            evol_df_cols = self.evol_df.columns
            filtered_evol_df = self.evol_df[[col for col in evol_df_cols if col[0] == orig_col]]

            evol_bafo = {}

            for direct in self.directions:
                evol_bafo[direct] = filtered_evol_df[(orig_col, direct)].to_numpy()

            idx = repr_rng_memo_init_loss(evol_bafo['ba'], evol_bafo['fo'], self.alpha, self.slice_init)
            insert_repr_in_df_from_idx(self, idx, orig_col)

            self.repr_df = True

        logger.info("Representative data complete.")

        return self.orig_df


    def plot(self,):
        data_marker_s = 100
        repr_marker_s = 100
        cols = set()

        for col, _ in self.evol_df.columns:
            cols.add(col)

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

            ax2.scatter(
                x=self.evol_df.index,
                y=self.orig_df[col],
                label=col,
                color='k',
                alpha=0.3,
                edgecolors='none',
                s=data_marker_s)

            if self.repr_df:
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


if __name__ == '__main__':

    rhis = RHIS()
    # df = pd.read_csv('./data/BigSiouxAnnualQ.csv')
    # df = pd.read_csv('./data/MarchMilwaukeeChloride.csv')
    # df['Time'] = pd.to_datetime(df['Time'].astype(int), format='%Y')
    # df.set_index('Time', inplace=True)

    # rhis = RHIS(df, index_col=)

    # UFPR DATASET
    df = pd.read_excel('./data/dataset.xlsx')
    df[df.select_dtypes(include=['object']).columns.drop('PONTO')] = \
        df.select_dtypes(include=['object']).drop(columns='PONTO').apply(pd.to_numeric, errors='coerce')
    df = df.loc[df['PONTO'] == 'IG6', ['DATA', 'BOD (mg/L)', 'NH4 (mg/L)', 'DO (mg/L)', 'COND', 'T (°C)']]
    df['DATA'] = pd.to_datetime(df['DATA'])
    df.set_index('DATA', inplace=True)
    df.dropna(inplace=True)

    # yearly_range = pd.date_range(start='2000-01-01', end='2013-01-01', freq='YE')
    # monthly_range = pd.date_range(start='2013-02-20', end='2013-12-20', freq='ME')
    # index = yearly_range.union(monthly_range)
    # index = pd.date_range(start='2000-12-31', end='2022-12-31', freq='YE')
    # ts = [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 4, 2, 5, 3, 10, 9, 9.5, 3.4, 5.7, 2.5, 7, 4.3, 11]
    # df = pd.DataFrame({'data': ts}, index=index)

    evol_df = rhis.evol(df, raw=False)
    rhis_repr_df = rhis.select_repr()
    rhis.plot()
