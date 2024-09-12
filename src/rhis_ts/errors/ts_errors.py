from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

# from loguru import logger

if TYPE_CHECKING:
    from typing import Any


def is_all_series_in_list(series: list[Any] | np.ndarray[Any] ):
    for serie in series:
        is_list = isinstance(serie, list)
        is_arr = isinstance(serie, np.ndarray)
        is_one_dim = np.array(serie).shape == (1,)
        ts_arr = np.array(serie)
        is_float = np.issubdtype(ts_arr, np.floating)
        is_int = np.issubdtype(ts_arr, np.integer)

        is_numeric = (is_list or is_arr) and is_one_dim and (is_float or is_int)
        return is_numeric
        # if not is_numeric:
        #     err_msg = "The parameters 'ba' and 'fo' must be 1D-dimensional, " + \
        #                 "an instance of the classes 'list' or 'numpy.ndarray', and have float or integers."

        #     raise ValueError(err_msg)


def raise_timeseries_type_error(ts: Any):
    """Raise TypeError if at least one element is not of type integer or float.

    Parameters
    ----------
        ts
            A list of integers or floats.
    """
    return (isinstance(ts, (list, np.ndarray))
            and all(isinstance(element, int | float | np.integer | np.floating) for element in ts))
    # if not isinstance(ts, (list, np.ndarray)):
    #     logger.error('Parameter ts must be a list or numpy.ndarray.')
    #     exc = 'Parameter ts must be a list or numpy.ndarray.'
    #     raise TypeError(exc)

    # if not all(isinstance(element, int | float | np.integer | np.floating) for element in ts):
    #     logger.error('The elements must be integers and/or floats.')
    #     exc = 'The elements must be integers and/or floats.'

    #     raise TypeError(exc)
