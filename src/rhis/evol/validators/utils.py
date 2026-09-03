from __future__ import annotations

import os
from typing import TYPE_CHECKING

from loguru import logger

from rhis.evol.exc import PlotRhisFullWithStatDefinedError, RhisEvolNotCalledError

if TYPE_CHECKING:
    from pandas import DataFrame

    from rhis.custom_types.stats import RhisCode


def  is_valid_path(path: str) -> bool:
    if os.path.exists(path):
        return True
    else:
        return False


def validate_or_raise_rhis_statistic_evol_not_called(rhis_statistic_df: DataFrame, stat: str | None) -> bool:
    if stat is not None and rhis_statistic_df is None:
        msg = 'You should run `Rhis.evol()` first.'
        logger.debug(msg)
        raise RhisEvolNotCalledError(msg)


def validate_or_raise_rhis_full_evol_not_called(rhis_full_df: DataFrame, stat: str | None) -> bool:
    if stat is None and rhis_full_df is None:
        msg = 'You should run `Rhis.evol()` first.'
        logger.debug(msg)
        raise RhisEvolNotCalledError(msg)


def validate_or_raise_plot_rhis_full_with_stat_defined(rhis_full_df: DataFrame, stat: str | None) -> bool:
    if stat is not None and rhis_full_df is None:
        msg = '`rhis.plot(rhis=True)` only works if `rhis.evol()` was called without `stat` parameter defined.'
        logger.debug(msg)
        raise PlotRhisFullWithStatDefinedError(msg)


def validate_or_raise_target_hyp_param_incorrect(target_hyp: RhisCode | None, stat: str | None):
    if target_hyp is not None and target_hyp not in {'r', 'h', 'i', 's'}:
        msg = "target_hyp must be one of 'r', 'h', 'i', or 's'."
        logger.debug(msg)
        raise ValueError(msg)

    if stat is None and target_hyp is None:
        msg = 'You must define the target_hyp (target hypothesis) to mark the representative data.'
        logger.debug(msg)
        raise ValueError(msg)

    if stat is not None and target_hyp is not None:
        msg = f"target_hyp was passed but won't be used since self.stat is defined as '{stat}'."
        logger.info(msg)

