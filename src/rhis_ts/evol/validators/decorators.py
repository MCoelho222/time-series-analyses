from __future__ import annotations

from functools import wraps

from loguru import logger

from rhis_ts.evol.validators.utils import is_valid_path


def validate_plot_params(func):  # noqa: C901
    @wraps(func)
    def _validate_plot_params(*args, **kwargs):  # noqa: C901, PLR0912
        try:
            noself_args = args[1:]
            if noself_args:
                arg_names = ['col_name', 'save_dir_path', 'save_format']
                for i in range(len(noself_args)):
                    kwargs[arg_names[i]] = noself_args[i]

            arg_types = {
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

            accepted_save_formats = (
                        'png', 'jpg', 'jpeg', 'pdf', 'svg', 'eps', 'tiff',
                        'tif', 'bmp', 'ps', 'raw',)

            for kw, val in kwargs.items():
                if isinstance(arg_types[kw], type):
                    if not isinstance(val, arg_types[kw]):
                        bool_type = arg_types[kw].__name__
                        msg = f"The argument '{kw}' must be a {bool_type}"
                        raise ValueError(msg)

                    elif kw == 'alpha' and (val < 0 or val > 1):
                        msg = (f"The value '{val}' is invalid. The parameter '{kw}' should be "
                               f"a {arg_types[kw].__name__} between 0 and 1.")
                        raise ValueError(msg)

                    elif kw == 'save_path' and not is_valid_path(val):
                        msg = (f"The value '{val}' is invalid. The parameter '{kw}' should be "
                               f"a {arg_types[kw].__name__} between 0 and 1.")
                        raise ValueError(msg)

                    elif kw == 'save_format' and val not in accepted_save_formats:
                        formats = ', '.join(accepted_save_formats).rstrip()
                        msg = (f"The format '{val}' is invalid. The parameter '{kw}' should be "
                               f"a {arg_types[kw].__name__} of one of these options: {formats}.")
                        raise ValueError(msg)

                elif not isinstance(val, type) and isinstance(val, tuple):
                    max_length = 2
                    if len(val) > max_length or not all(isinstance(el, arg_types[kw][0]) for el in val):
                        bool_type = arg_types[kw].__name__
                        msg = f"The argument '{kw}' must be a {bool_type}."
                        raise ValueError(msg)

                elif not isinstance(arg_types[kw], type) and isinstance(val, dict):
                    for inner_kw, inner_val in val.items():
                        if isinstance(arg_types[kw][inner_kw], type) and not isinstance(inner_val, arg_types[kw][inner_kw]):
                            bool_type = arg_types[kw][inner_kw].__name__
                            msg = f"The argument '{inner_kw}' from '{kw}' must be a {bool_type}."
                            raise ValueError(msg)

                        elif not isinstance(arg_types[kw][inner_kw], type) and isinstance(inner_val, tuple):
                            max_length = 4
                            if len(inner_val) > max_length or not all(
                                isinstance(el, arg_types[kw][inner_kw][0]) for el in inner_val):
                                bool_type = arg_types[kw][inner_kw].__name__
                                msg = f"The argument '{inner_kw}' from '{kw}' must be a {bool_type}."
                                raise ValueError(msg)

        except ValueError as exc:
            logger.exception(exc)
            return exc

        return func(*args, **kwargs)

    return _validate_plot_params


def validate_evol_params(func):  # noqa: C901
    @wraps(func)
    def _validate_evol_params(*args, **kwargs):
        try:
            noself_args = args[1:]
            if noself_args:
                arg_names = ['cols', 'stat', 'alpha']
                for i in range(len(noself_args)):
                    kwargs[arg_names[i]] = noself_args[i]

            arg_types = {
                'cols': (str,),
                'stat': ('min', 'max', 'mean', 'med',),
                'alpha': float,
                'backwards': bool,
            }

            for kw, val in kwargs.items():
                if isinstance(arg_types[kw], tuple):
                    if kw == 'stat' and val not in arg_types[kw]:
                        msg = (
                            f"The value '{val}' is invalid. The parameter 'stat' "
                            f"should be one of these: 'min', 'max', 'mean', or 'med'.")
                        raise ValueError(msg)

                    elif kw != 'stat' and not isinstance(val, tuple) or not all(isinstance(col, str) for col in val):
                        msg = f"The value '{val}' is invalid. The parameter '{kw}' should be a tuple of strings."
                        raise ValueError(msg)


                elif not isinstance(val, arg_types[kw]):
                    msg = f"The value '{val}' is invalid. The parameter {kw} should be a {arg_types[kw].__name__}."
                    raise ValueError(msg)

                elif kw == 'alpha' and (val < 0 or val > 1):
                    msg = (f"The value '{val}' is invalid. The parameter '{kw}' should be "
                            f"a {arg_types[kw].__name__} between 0 and 1.")
                    raise ValueError(msg)

        except (Exception, ValueError) as exc:
            logger.exception(exc)
            return

        return func(*args, **kwargs)

    return _validate_evol_params
