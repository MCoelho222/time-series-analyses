
from __future__ import annotations

from typing import TYPE_CHECKING, Iterable

from rhis_ts.evol.errors.exceptions import raise_ts_diff_lengths
from rhis_ts.evol.utils.bafo import (
    ends_rejected_both_directions,
    idx_of_last_not_rejected,
    starts_rejected,
)
from rhis_ts.utils.arrays import nans_floats_from_array

if TYPE_CHECKING:
    import numpy as np


def cut_idx_for_representative(ba: np.ndarray[float], alpha: float, sli_init: int):
    ba_floats_and_nans = nans_floats_from_array(ba)
    ba_floats = ba_floats_and_nans[0]

    ba_last = len(ba_floats) + sli_init - 1
    if ba[0] >= alpha:
        return (0, ba_last)

    return idx_of_last_not_rejected(alpha, ba_floats, 'ba'), ba_last


def repr_and_extension_slice(ba: Iterable[float], fo: Iterable[float], alpha: float,*, most_recent: bool) -> Iterable[float]:

    raise_ts_diff_lengths(ba, fo)

    ba_floats_and_nans = nans_floats_from_array(ba)
    fo_floats_and_nans = nans_floats_from_array(fo)

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
