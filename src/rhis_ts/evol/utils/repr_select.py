from __future__ import annotations

from typing import TYPE_CHECKING, Iterable

import numpy as np

from rhis_ts.evol.utils.ba_fo import (
    ends_rejected_both_directions,
    idx_of_last_not_rejected,
    separate_nans_from_evol,
    starts_rejected,
)
from rhis_ts.evol.exc.exc import raise_ts_diff_lengths


def repr_rng_no_memo_loss(
        ba: Iterable[float],
        fo: Iterable[float],
        alpha: float,*,
        most_recent: bool,
        ) -> Iterable[float]:

    raise_ts_diff_lengths(ba, fo)

    ba_floats_and_nans = separate_nans_from_evol(ba)
    fo_floats_and_nans = separate_nans_from_evol(fo)

    ba_floats = ba_floats_and_nans[0]
    fo_floats = fo_floats_and_nans[0]

    slice_init = len(ba) - len(ba_floats)

    cut_idx_ba = 0
    cut_idx_fo = 0

    if most_recent:
        if ends_rejected_both_directions(ba, fo, alpha):
            return {
            'init_rng': (cut_idx_ba, len(ba)),
            'ext_rng': None
            }

        if not starts_rejected(alpha, ba_floats, 'ba'):
            cut_idx_ba = idx_of_last_not_rejected(alpha, ba_floats, 'ba')

            if not starts_rejected(alpha, fo_floats, 'fo'):

                cut_idx_fo = slice_init + idx_of_last_not_rejected(alpha, fo_floats, 'fo')

    start = (cut_idx_ba, len(ba))
    ext = None
    if cut_idx_fo > cut_idx_ba:
        start = (cut_idx_fo, len(ba))
        ext = (cut_idx_ba, cut_idx_fo)

    return {
        'init_rng': start,
        'ext_rng': ext
        }


def repr_rng_memo_init_loss(ba: np.ndarray[float], fo: np.ndarray[float], alpha: float, sli_init: int):
    ba_floats_and_nans = separate_nans_from_evol(ba)
    # fo_floats_and_nans = separate_nans_from_evol(fo)
    ba_floats = ba_floats_and_nans[0]
    # fo_floats = fo_floats_and_nans[0]

    ba_last = len(ba_floats) + sli_init - 1
    if ba[0] >= alpha:
        return (0, ba_last)

    return idx_of_last_not_rejected(alpha, ba_floats, 'ba'), ba_last
















# is_reject = {}
# take_all = []
# is_x_shaped = []
# is_reject[direct] = ts_arrs[direct] < alpha_arr
# take_all.append(np.all(~is_reject[direct]))
# if not np.all(take_all) and n_dir == 2:
#     ts_drop_nan = ts_arrs[direct][~np.isnan(ts_arrs[direct])]
#     is_upward = ts_drop_nan[0] < self.alpha and ts_drop_nan[-1] > self.alpha
#     is_downward = ts_drop_nan[0] > self.alpha and ts_drop_nan[-1] < self.alpha
#     is_x_shaped.append(is_upward or is_downward)

# if np.all(take_all):
#     self.orig_df[orig_col + '_repr'] = self.orig_df[orig_col]
#     self.repr_df = True
# else:
#     if n_dir == 1:
#         bool_arr = is_reject[direction]
#         self.__create_repr_series_from_rejections(bool_arr, orig_col, direction)
#         self.repr_df = True

#     if n_dir == 2:
#         if np.all(is_x_shaped):
#             ba_mask = ts_arrs['backward'][n_fill:-n_fill]
#             fo_mask = ts_arrs['forward'][n_fill:-n_fill]
#             alpha_arr_slice = alpha_arr[n_fill:-n_fill]
#             mask = []
#             for i in range(len(ba_mask)):
#                 if most_recent:
#                     mask.append(~(ba_mask[i] > fo_mask[i] and ba_mask[i] > alpha_arr_slice[i]))
#                 else:
#                     mask.append(~(fo_mask[i] > ba_mask[i] and fo_mask[i] > alpha_arr_slice[i]))

#             if ba_mask[-1] > self.alpha:
#                 ba_fill = np.zeros(n_fill)
#             else:
#                 ba_fill = np.ones(n_fill)

#             if fo_mask[0] > self.alpha:
#                 fo_fill = np.zeros(n_fill)
#             else:
#                 fo_fill = np.ones(n_fill)

#             mask = np.append(mask, ba_fill)
#             bool_arr = np.append(fo_fill, mask)

#             self.__create_repr_series_from_rejections(bool_arr, orig_col, direction)
#             self.repr_df = True
#         else:
#             bool_arr = is_reject[direction]
#             self.__create_repr_series_from_rejections(bool_arr, orig_col, direction)
#             self.repr_df = True


