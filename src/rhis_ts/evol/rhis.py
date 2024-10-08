from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd
from loguru import logger
from pandas import DataFrame

from rhis_ts.evol.exc import EvolDirectionError, EvolNotRunInDirectionError, EvolRunMissingError, PlotEvolError
from rhis_ts.evol.methods import repr_slice_idxs, rhis_standard_evol
from rhis_ts.evol.plot.plot_standard_evol import finalize_plot, plot_data, plot_rhis_evol
from rhis_ts.evol.utils.dataframe import build_init_evol_df, insert_repr_in_df_from_idx
from rhis_ts.evol.validators import validate_evol_params, validate_plot_params
from rhis_ts.utils.data import slice_init

if TYPE_CHECKING:
    from pandas import Series


class Rhis:
    def __init__(self, df):
        self.alpha = 0.05
        self.rhis = None
        self.stat = None
        self.backwards = True

        if (not isinstance(df, pd.DataFrame)
            or isinstance(df.index, pd.MultiIndex)
            or isinstance(df.index, pd.MultiIndex)):
            msg = "The parameter 'df' must be a non-MultiIndex pandas.DataFrame."
            logger.debug(msg)
            raise ValueError(msg)

        self.orig_df = df

        self.evol_df = None
        self.evol_df_rhis = None
        self.slice_init = slice_init(len(self.orig_df))


    @validate_evol_params
    def evol(self,
            cols: tuple[str]|None=None,
            stat: str|None=None,
            alpha: float=0.05,*,
            backwards: bool=True
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

        Return
        ------
            DataFrame with p-values evolution
        """
        mode = 'RHIS' if stat is None else f'RHIS-{stat}'
        msg = f"Processing {mode} evolution..."
        logger.info(msg)

        self.stat = stat
        self.alpha = alpha
        self.backwards = backwards

        evol_cols = cols if cols is not None else self.orig_df.columns
        if self.evol_df_rhis is None and stat is None:
            init_df = build_init_evol_df(evol_cols, self.orig_df.index, stat, backwards=backwards)
            self.evol_df_rhis = init_df
        if self.evol_df is None and stat is not None:
            init_df = build_init_evol_df(evol_cols, self.orig_df.index, stat, backwards=backwards)
            self.evol_df = init_df

        for col in evol_cols:
            ts = self.orig_df[col]
            self._ts_evol(ts, alpha)
        evol_df = self.evol_df[evol_cols] if self.evol_df is not None else self.evol_df_rhis[evol_cols]

        logger.info("RHIS evolution successfully complete.")
        return evol_df


    def _ts_evol(self, ts: Series, alpha: float=0.05):
        ts_arr = ts.to_numpy()
        if self.backwards:
            ts_arr = ts_arr[::-1]

        evol = rhis_standard_evol(ts_arr, alpha, self.slice_init, self.stat, backwards=self.backwards)

        direction = 'ba' if self.backwards else 'fo'
        if self.stat is None:
            for hyp, ps in evol.items():
                self.evol_df_rhis[(ts.name, direction, hyp)] = ps
        else:
            self.evol_df[(ts.name, direction)] = evol


    def add_repr_cols_to_df(self,*, backwards: bool=True) -> DataFrame:
        logger.info("Adding representative data...")
        try:
            if self.evol_df is None:
                msg = 'Please, run the evolution process before adding representative data.'
                raise EvolRunMissingError(msg)
            if not isinstance(backwards, bool):
                msg = f"The value '{backwards}' is invalid. The 'backwards' parameter should be a boolean."
                raise EvolDirectionError(msg)

            direction = 'ba' if backwards else 'fo'

            orig_cols = self.orig_df.columns
            for orig_col in orig_cols:
                evol_df_cols = self.evol_df.columns
                filtered_evol_df = self.evol_df[[col for col in evol_df_cols if col[0] == orig_col]]

                if (orig_col, direction) not in filtered_evol_df.columns:
                    direction_name = 'backwards' if backwards else 'forwards'
                    msg = f"Please, run the evolution process in the {direction_name} direction."
                    raise EvolNotRunInDirectionError(msg)
                evol_bafo = filtered_evol_df[(orig_col, direction)].to_numpy()
                cut_idxs = repr_slice_idxs(evol_bafo, self.alpha, self.slice_init, direction)
                insert_repr_in_df_from_idx(self.orig_df, cut_idxs, orig_col)
        except (EvolDirectionError, EvolNotRunInDirectionError, EvolRunMissingError, ValueError) as exc:
            logger.exception(exc)
            return exc

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

            direction = 'ba' if self.backwards else 'fo'
            for col in cols:
                evol_ax = plot_rhis_evol(
                    col,
                    self.evol_df,
                    self.evol_df_rhis,
                    direction,
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
    evol_df = rhis.evol(stat='min')
    # evol_df = rhis.evol(stat='min', backwards=False)
    # print(rhis.evol_df.head())
    rhis.add_repr_cols_to_df(backwards='True')
    # print(rhis.evol_df.head())
    # rhis.plot(rhis=False, show_repr=True)
