from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd
from loguru import logger
from pandas import DataFrame

from rhis.evol.exc import EvolDirectionError, EvolNotRunInDirectionError, EvolRunMissingError, PlotEvolError
from rhis.evol.methods import representative_slice_idxs, rhis_standard_evol
from rhis.evol.plot.plot_standard_evol import finalize_plot, plot_data, plot_rhis_evol
from rhis.evol.utils.dataframe import build_init_evol_df, insert_repr_in_df_from_idx
from rhis.evol.validators import validate_evol_params, validate_plot_params
from rhis.utils.data import slice_init

if TYPE_CHECKING:
    from pandas import Series

    from rhis.custom_types.stats import RhisCode


class Rhis:
    def __init__(self, df):
        self.alpha = 0.05
        self.rhis = None
        self.stat = None
        self.backwards = True
        self.direction = 'backwards'

        if (not isinstance(df, pd.DataFrame) or isinstance(df.index, pd.MultiIndex)):
            msg = "The parameter 'df' must be a non-MultiIndex pandas.DataFrame."
            logger.debug(msg)
            raise ValueError(msg)

        self.orig_df = df

        self.rhis_statistic_df = None
        self.rhis_full_df = None
        self.slice_init = slice_init(len(self.orig_df))


    @validate_evol_params
    def evol(self,
            cols: tuple[str]|None=None,
            stat: str|None=None,
            alpha: float=0.05,*,
            backwards: bool=True
            ) -> DataFrame:
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
        mode = 'RHIS' if stat is None else f'RHIS-{stat}'
        msg = f"Processing {mode} evolution..."
        logger.info(msg)

        self.stat = stat
        self.alpha = alpha
        self.backwards = backwards

        self.direction = 'backwards' if self.backwards else 'forward'

        evol_cols = cols if cols is not None else self.orig_df.columns
        init_df = build_init_evol_df(evol_cols, self.orig_df.index, stat, backwards=backwards)
        if self.rhis_full_df is None and stat is None:
            self.rhis_full_df = init_df
        else:
            self.rhis_statistic_df = init_df

        for col in evol_cols:
            ts = self.orig_df[col]
            self._ts_evol(ts)

        logger.info("RHIS evolution successfully complete.")
        return self.rhis_statistic_df[evol_cols] if self.rhis_statistic_df is not None else self.rhis_full_df[evol_cols]


    def _ts_evol(self, ts: Series):
        ts_arr = ts.to_numpy()
        if self.backwards:
            ts_arr = ts_arr[::-1]

        evol = rhis_standard_evol(ts_arr, self.alpha, self.slice_init, self.stat, backwards=self.backwards)

        if self.stat is None:
            for hyp, ps in evol.items():
                self.rhis_full_df[(ts.name, self.direction, hyp)] = ps
        else:
            self.rhis_statistic_df[(ts.name, self.direction)] = evol


    def mark_representative_data_with_bool(self, target_hyp: RhisCode | None=None) -> DataFrame:
        if target_hyp is not None and target_hyp not in {'r', 'h', 'i', 's'}:
            msg = "target_hyp must be one of 'r', 'h', 'i', or 's'."
            logger.debug(msg)
            raise ValueError(msg)

        if self.stat is None and target_hyp is None:
            msg = 'You must define the target_hyp (target hypothesis) to mark the representative data.'
            logger.debug(msg)
            raise ValueError(msg)

        if self.stat is not None and target_hyp is not None:
            msg = f"target_hyp was passed but won't be used since self.stat is defined as '{self.stat}'."
            logger.info(msg)

        if self.stat is not None and self.rhis_statistic_df is None:
            msg = 'You should run `Rhis.evol()` first.'
            logger.debug(msg)
            raise EvolRunMissingError(msg)

        original_cols = self.orig_df.columns
        for original_col in original_cols:
            target_df = self.rhis_full_df if self.stat is None else self.rhis_statistic_df
            target_col = (
                (original_col, self.direction, target_hyp.upper())
                if self.stat is None
                else (original_col, self.direction)
            )

            rhis_series = target_df[target_col].to_numpy()
            cut_idxs = representative_slice_idxs(rhis_series, self.alpha, self.slice_init, self.backwards)
            insert_repr_in_df_from_idx(self.orig_df, cut_idxs, original_col)

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
            if self.rhis_statistic_df is None:
                msg = "Please, before trying to plot, run the evolution process by calling the 'evol' method."
                raise PlotEvolError(msg)

            cols = [col_name,]

            if col_name is None:
                cols = {col for col, _ in self.rhis_statistic_df.columns}
            elif col_name not in self.orig_df.columns.values:
                msg = f"The name '{col_name}' is not in the columns of the dataframe."
                raise ValueError(msg)

            direction = 'ba' if self.backwards else 'fo'
            for col in cols:
                evol_ax = plot_rhis_evol(
                    col,
                    self.rhis_statistic_df,
                    self.rhis_full_df,
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
    rhis.evol()
    rhis.mark_representative_data_with_bool(target_hyp='s')
    print(rhis.orig_df.head())
    # rhis.plot()
