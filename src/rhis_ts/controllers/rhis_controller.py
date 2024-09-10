from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from pandas import DataFrame, Index


def insert_repr_and_repr_ext_in_df_from_idx(self, idx: dict[tuple], df_col: str):
        st_r = idx['init_rng']
        ext_r = idx['ext_rng']

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


def insert_repr_in_df_from_idx(self, idx: tuple, df_col: str):
        orig_ts = self.orig_df[df_col].to_numpy()
        nums_ts = orig_ts[idx[0]:idx[1]]
        nan_init = np.full(idx[0], np.nan)
        nan_fin= np.full(len(orig_ts) - idx[1], np.nan)

        full_ts = np.append(nan_init, nums_ts)
        full_ts = np.append(full_ts, nan_fin)

        self.orig_df.loc[:, df_col + '_repr'] = full_ts


def build_init_evol_df(self, orig_colnames: list[str], index: Index) -> DataFrame:
        cols_tuples = []
        for col in orig_colnames:
            for direct in self.directions:
                if self.raw:
                    hyps = ['R', 'H', 'I', 'S']
                    for hyp in hyps:
                        cols_tuples.append((col, direct, hyp))
                else:
                    cols_tuples.append((col, direct))

        cols = pd.MultiIndex.from_tuples(cols_tuples)
        result_df = pd.DataFrame(columns=cols, index=index)

        return result_df
