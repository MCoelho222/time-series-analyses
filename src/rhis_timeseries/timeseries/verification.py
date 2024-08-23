from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from typing import Any


def is_numeric_series(series: list[Any] | np.ndarray[Any] ) -> bool:
    for serie in series:
        is_list = isinstance(serie, list)
        is_arr = isinstance(serie, np.ndarray)
        is_one_dim = np.array(serie).shape == (1,)
        ts_arr = np.array(serie)
        is_float = np.issubdtype(ts_arr, np.floating)
        is_int = np.issubdtype(ts_arr, np.integer)

        is_numeric = (is_list or is_arr) and is_one_dim and (is_float or is_int)
        if not is_numeric:
            err_msg = "The parameters 'bw' and 'fw' must be 1D-dimensional, " + \
                        "an instance of the classes 'list' or 'numpy.ndarray', and have float or integers."
            raise ValueError(err_msg)
