from __future__ import annotations

from functools import wraps

import numpy as np


def check_test_args(test: str):  # noqa: C901, PLR0915
    def check_args(func):  # noqa: C901, PLR0915
        @wraps(func)
        def _check_args(*args, **kwargs):  # noqa: C901, PLR0912, PLR0915
            try:
                x = args[0]
                if not isinstance(x, (list, np.ndarray)):
                    error = f"{x} is invalid. The x parameter must be a list or np.ndarray."
                    raise ValueError(error)
                for num in x:
                    if not isinstance(num,(float, int)):
                        error = f"{x} is invalid. The x parameter must contain only floats or integers."
                        raise ValueError(error)
            except IndexError:
                pass

            if test == 'mann-whitney':
                try:
                    y = args[1]
                    if not isinstance(y, (list, np.ndarray)):
                        error = f"{y} is invalid. The y parameter must be a list or np.ndarray."
                        raise ValueError(error)
                    for num in y:
                        if not isinstance(num,(float, int)):
                            error = f"{y} is invalid. The y parameter must contain only floats or integers."
                            raise ValueError(error)
                except IndexError:
                    pass

            try:
                alternative = args[2]
                alternative_options = ['two-sided', 'greater', 'less']
                if not isinstance(alternative, str):
                    error = f"{alternative} is invalid. The alternative parameter must be of type string."
                    raise ValueError(error)
                if alternative not in alternative_options:
                    error = f"{alternative} is invalid. The alternative parameter must be one of these: {alternative_options}."
                    raise ValueError(error)
            except IndexError:
                pass

            try:
                alpha = args[3]
                if not isinstance(alpha, float):
                    error = f"{alpha} is invalid. The alpha parameter must be of type float."
                    raise ValueError(error)
                if not (alpha > 0 and alpha < 1):
                    error = f"'{alpha}' is not a valid value for 'alpha'. It should be a float number between 0 and 1."
                    raise ValueError(error)
            except IndexError:
                pass

            try:
                continuity = kwargs['continuity']
                if not isinstance(continuity, bool):
                    error = f"{continuity} is invalid. The continuity parameter must be of type bool."
                    raise ValueError(error)
            except KeyError:
                pass

            try:
                ties = kwargs['ties']
                if not isinstance(ties, bool):
                    error = f"{ties} is invalid. The ties parameter must be of type bool."
                    raise ValueError(error)
            except KeyError:
                pass

            return func(*args, **kwargs)

        return _check_args
    return check_args


