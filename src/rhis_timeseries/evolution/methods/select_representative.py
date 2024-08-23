from __future__ import annotations

from typing import Iterable

from rhis_timeseries.evolution.data import (
    drop_nans_from_evol,
    ends_rejected_both_directions,
    idx_of_last_not_rejected,
    starts_rejected,
)
from rhis_timeseries.evolution.errors import raise_ts_diff_lengths


def get_repr_index(
        bw: Iterable[float],
        fw: Iterable[float],
        alpha: float,*,
        most_recent: bool,
        ) -> Iterable[float]:
    """
    Get the start and final indexes of the representative time series.
    """
    raise_ts_diff_lengths(bw, fw)

    only_floats = drop_nans_from_evol(bw, fw)
    bw_floats = only_floats['bw']
    fw_floats = only_floats['fw']
    slice_init = len(bw) - len(bw_floats)

    cut_idx = [0, len(bw)]

    if most_recent:
        if ends_rejected_both_directions(bw, fw, alpha):
            return cut_idx

        if not starts_rejected(alpha, bw_floats, 'bw'):
            cut_idx[0] = idx_of_last_not_rejected(alpha, bw_floats, 'bw', slice_init)

        if starts_rejected(alpha, bw_floats, 'bw') and not starts_rejected(alpha, fw_floats, 'fw'):
            cut_idx[1] = idx_of_last_not_rejected(alpha, fw_floats, 'fw', slice_init)

        return cut_idx












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
#             bw_mask = ts_arrs['backward'][n_fill:-n_fill]
#             fw_mask = ts_arrs['forward'][n_fill:-n_fill]
#             alpha_arr_slice = alpha_arr[n_fill:-n_fill]
#             mask = []
#             for i in range(len(bw_mask)):
#                 if most_recent:
#                     mask.append(~(bw_mask[i] > fw_mask[i] and bw_mask[i] > alpha_arr_slice[i]))
#                 else:
#                     mask.append(~(fw_mask[i] > bw_mask[i] and fw_mask[i] > alpha_arr_slice[i]))

#             if bw_mask[-1] > self.alpha:
#                 bw_fill = np.zeros(n_fill)
#             else:
#                 bw_fill = np.ones(n_fill)

#             if fw_mask[0] > self.alpha:
#                 fw_fill = np.zeros(n_fill)
#             else:
#                 fw_fill = np.ones(n_fill)

#             mask = np.append(mask, bw_fill)
#             bool_arr = np.append(fw_fill, mask)

#             self.__create_repr_series_from_rejections(bool_arr, orig_col, direction)
#             self.repr_df = True
#         else:
#             bool_arr = is_reject[direction]
#             self.__create_repr_series_from_rejections(bool_arr, orig_col, direction)
#             self.repr_df = True


