from __future__ import annotations

from functools import wraps

from rhis_ts.stats.hypothesis_testing.errors.raises import (
    raise_if_alpha_invalid,
    raise_if_alternative_invalid,
    raise_if_not_bool,
    raise_if_not_list_or_array_of_nums,
)


def check_hypothesis_test_args(test: str):
    def check_args(func):
        @wraps(func)
        def _check_args(*args, **kwargs):
            arg_raises = {
                0: raise_if_not_list_or_array_of_nums,
                1: raise_if_alpha_invalid,
                2: raise_if_alternative_invalid,
            }

            arg_raises_by_test = {
                'mann-kendall': [arg_raises[0], arg_raises[1], arg_raises[2]],
                'wallis-moore': [arg_raises[0], arg_raises[1], arg_raises[2]],
                'mann-whitney': [arg_raises[0], arg_raises[1], arg_raises[2], arg_raises[0]],
                'wald-wolfowitz': [arg_raises[0], arg_raises[1]]
            }

            for i in range(len(args)):
                arg_raises_by_test[test][i](args[i])

            if kwargs:
                for kw, arg in kwargs.items():
                    raise_if_not_bool(arg, kw)

            return func(*args, **kwargs)
        return _check_args
    return check_args


