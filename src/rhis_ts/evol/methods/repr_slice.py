
from __future__ import annotations

from typing import TYPE_CHECKING, Iterable

from rhis_ts.evol.errors.exceptions import raise_ts_diff_lengths
from rhis_ts.evol.utils.bafo import (
    ends_rejected_both_directions,
    idx_of_last_not_rejected,
    starts_rejected,
)
from rhis_ts.utils.arrays import nans_nums_from_array

if TYPE_CHECKING:
    import numpy as np


def repr_slice_idxs(ps: np.ndarray[float], alpha: float, sli_init: int, direction: str):
    ps_nums_and_nans = nans_nums_from_array(ps)
    ps_nums = ps_nums_and_nans

    ps_last = len(ps_nums) + sli_init - 1
    if ps[0] >= alpha:
        return (0, ps_last)

    return idx_of_last_not_rejected(alpha, ps_nums, direction, sli_init), ps_last


def repr_and_extension_slice(ba: Iterable[float], fo: Iterable[float], alpha: float,*, most_recent: bool) -> Iterable[float]:

    raise_ts_diff_lengths(ba, fo)

    ba_nums_and_nans = nans_nums_from_array(ba)
    fo_nums_and_nans = nans_nums_from_array(fo)

    ba_nums = ba_nums_and_nans[0]
    fo_floats = fo_nums_and_nans[0]

    slice_init = len(ba) - len(ba_nums)

    cut_idx_ba = 0
    cut_idx_fo = 0

    if most_recent:
        if ends_rejected_both_directions(ba, fo, alpha):
            return {
            'init_rng': (cut_idx_ba, len(ba)),
            'ext_rng': None
            }

        if not starts_rejected(alpha, ba_nums, 'ba'):
            cut_idx_ba = idx_of_last_not_rejected(alpha, ba_nums, 'ba')

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
