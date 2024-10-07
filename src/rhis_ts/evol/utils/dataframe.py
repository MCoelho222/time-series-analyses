from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from pandas import DataFrame, Index


def insert_repr_in_df_from_idx(df: DataFrame, idx: tuple, df_col: str):
    orig_ts = df[df_col].to_numpy()
    nums_ts = orig_ts[idx[0]:idx[1]]
    nan_init = np.full(idx[0], np.nan)
    nan_fin= np.full(len(orig_ts) - idx[1], np.nan)

    full_ts = np.append(nan_init, nums_ts)
    full_ts = np.append(full_ts, nan_fin)

    df.loc[:, df_col + '_repr'] = full_ts


def build_init_evol_df(orig_colnames: list[str], index: Index, stat: str|None) -> DataFrame:
    cols_tuples = []
    for col in orig_colnames:
        for direct in ('ba', 'fo'):
            if stat is None:
                hyps = ['R', 'H', 'I', 'S']
                for hyp in hyps:
                    cols_tuples.append((col, direct, hyp))
            else:
                cols_tuples.append((col, direct))

    cols = pd.MultiIndex.from_tuples(cols_tuples)
    result_df = pd.DataFrame(columns=cols, index=index)

    return result_df
