from __future__ import annotations

from functools import wraps

from loguru import logger


def validate_plot_params(func):  # noqa: C901
    @wraps(func)
    def _validate_plot_params(*args, **kwargs):  # noqa: C901
        try:
            if args[1:]:
                arg_names = ['col_name', 'save_dir_path', 'save_format']
                for i in range(len(args[1:])):
                    kwargs[arg_names[i]] = args[1:][i]

            types = {
                'alpha_line_params': {
                    'alpha': float,
                    'color': str,
                    'linestyle': str,
                    'linewidth': float,
                },
                'col_name': str,
                'data_params': {
                    'alpha': float,
                    'color': str,
                    'edgecolors': str,
                    'facecolors': str,
                    'marker': str,
                    's': int,
                },
                'figsize': (int, int),
                'figtitle': str,
                'save_dir_path': str,
                'save_format': str,
                'repr_params': {
                    'alpha': float,
                    'color': str,
                    'edgecolors': str,
                    'facecolors': str,
                    'marker': str,
                    's': int,
                },
                'rhis': bool,
                'rhis_params': {
                    'alpha': float,
                    'colors': (str, str, str, str),
                    'linestyle': str,
                    'linewidth': float,
                },
                'rhis_stat_params': {
                    'alpha': float,
                    'color': (str, str, str, str),
                    'linestyle': str,
                    'linewidth': float,
                },
                'show_repr': bool,
                'xlabel': str,
                'ylabel': str,
            }

            for kwarg, val in kwargs.items():
                if isinstance(types[kwarg], type) and not isinstance(val, types[kwarg]):
                    bool_type = types[kwarg].__name__
                    msg = f"The argument {kwarg} must be a {bool_type}"
                    raise ValueError(msg)
                elif not isinstance(val, type) and isinstance(val, tuple):
                    max_length = 2
                    if len(val) > max_length or not all(isinstance(el, types[kwarg][0]) for el in val):
                        bool_type = types[kwarg].__name__
                        msg = f"The argument '{kwarg}' must be a {bool_type}."
                        raise ValueError(msg)
                elif not isinstance(types[kwarg], type) and isinstance(val, dict):
                    for inner_kwarg, inner_val in val.items():
                        if isinstance(types[kwarg][inner_kwarg], type) and not isinstance(inner_val, types[kwarg][inner_kwarg]):
                            bool_type = types[kwarg][inner_kwarg].__name__
                            msg = f"The argument '{inner_kwarg}' from '{kwarg}' must be a {bool_type}."
                            raise ValueError(msg)
                        elif not isinstance(types[kwarg][inner_kwarg], type) and isinstance(inner_val, tuple):
                            max_length = 4
                            if len(inner_val) > max_length or not all(
                                isinstance(el, types[kwarg][inner_kwarg][0]) for el in inner_val):
                                bool_type = types[kwarg][inner_kwarg].__name__
                                msg = f"The argument '{inner_kwarg}' from '{kwarg}' must be a {bool_type}."
                                raise ValueError(msg)
        except ValueError as exc:
            logger.exception(exc)
            return exc

        return func(*args, **kwargs)

    return _validate_plot_params


def validate_evol_params(func):
    @wraps(func)
    def _validate_evol_params(*args, **kwargs):
        try:
            noself_args = args[1:]
            if noself_args:
                if not isinstance(noself_args[0], tuple) or not all(isinstance(arg, str) for arg in noself_args[0]):
                    msg = f"The value {noself_args[0]} is invalid. The parameter 'cols' should be a tuple of strings."
                    raise ValueError(msg)

            if kwargs:
                param_types = {
                    'cols': (str),
                    'stat': str,
                    'alpha': float,
                }
                for k, v in kwargs.items():
                    print(k, v)
                    if isinstance(param_types[k], tuple):
                        if not isinstance(v, tuple) or not all(isinstance(col, str) for col in v):
                            msg = f"The value {v} is invalid. The parameter 'cols' should be a tuple of strings."
                            raise ValueError(msg)
                    elif not isinstance(v, param_types[k]):
                        msg = f"The value {v} is invalid. The parameter {k} should be a {param_types[k].__name__}."
                        raise ValueError(msg)
        except ValueError as exc:
            logger.exception(exc)
            return exc

        return func(*args, **kwargs)

    return _validate_evol_params
