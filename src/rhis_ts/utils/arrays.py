"""Methods for array manipulation."""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from rhis_ts.types.data import TimeSeriesFlex

def nans_floats_from_array(ps: TimeSeriesFlex) -> TimeSeriesFlex:
    ps_mask = np.isnan(ps)
    ps_nan = ps[ps_mask]
    ps_floats = ps[~ps_mask]

    return ps_floats, ps_nan
