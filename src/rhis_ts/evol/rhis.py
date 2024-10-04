from __future__ import annotations

from typing import TYPE_CHECKING, Iterable

import pandas as pd
from loguru import logger
from pandas import DataFrame

from rhis_ts.controllers.rhis_controller import RhisController
from rhis_ts.evol.exc import PlotEvolError
from rhis_ts.evol.methods.repr_slice import repr_slice_idxs
from rhis_ts.evol.methods.standard_evol import rhis_standard_evol
from rhis_ts.evol.plot.plot_standard_evol import finalize_plot, plot_data, plot_rhis_evol
from rhis_ts.evol.validators import validate_plot_params
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

    @validate_plot_params
    def plot(
            self,
            col_name: str|None=None,
            save_dir_path: str | None=None,
            save_format: str | None='png',*,
            rhis: bool=False,
            show_repr: bool=True,
            **kwargs
            ):
        try:
            if self.evol_df is None:
                msg = "Please, before trying to plot, run the evolution process by calling the 'evol' method."
                raise PlotEvolError(msg)

            cols = [col_name,]

            if col_name is None:
                cols = {col for col, _ in self.evol_df.columns}
            elif col_name not in self.orig_df.columns.values:
                msg = f"The name '{col_name}' is not in the columns of the dataframe."
                raise ValueError(msg)

            for col in cols:
                evol_ax = plot_rhis_evol(
                    col,
                    self.evol_df,
                    self.evol_df_rhis,
                    self.direction,
                    kwargs.get('figsize'),
                    kwargs.get('xlabel'),
                    kwargs.get('rhis_params'),
                    kwargs.get('rhis_stat_params'),
                    rhis=rhis
                    )
                data_ax = plot_data(
                    evol_ax,
                    col,
                    self.orig_df,
                    kwargs.get('ylabel'),
                    kwargs.get('data_params'),
                    kwargs.get('repr_params'),
                    show_repr=show_repr
                    )
                filename = 'rhis_evol_' + col.lower().strip() + '.' + save_format
                filename_clean = filename.replace(' ', '_').replace('(', '').replace(')', '').replace('/', '_')
                col_save_path = filename if save_dir_path is None else f'{save_dir_path}{filename_clean}'
                finalize_plot(
                    evol_ax,
                    data_ax,
                    self.alpha,
                    kwargs.get('figtitle'),
                    kwargs.get('alpha_line_params'),
                    col_save_path
                    )
        except (PlotEvolError, ValueError) as exc:
            logger.exception(exc)

if __name__ == '__main__':

    df = pd.read_csv('./data/MarchMilwaukeeChloride.csv')
    df['Time'] = pd.to_datetime(df['Time'].astype(int), format='%Y')
    df.set_index('Time', inplace=True)

    rhis = Rhis(df)
    evol_df = rhis.evol(stat='min', rhis=False)
    rhis.add_repr_cols_to_df(direction='fo')
    rhis.plot(rhis=False, show_repr=True)
