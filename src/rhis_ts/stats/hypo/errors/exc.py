from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from typing import Any

def raise_if_not_list_or_array_of_nums(arg: Any):
    if not isinstance(arg, (list, np.ndarray)):
        error = f"{arg} is an invalid value for arg. It should be a list or np.ndarray."
        raise ValueError(error)

    for num in arg:
        if not isinstance(num,(float, int)):
            error = f"{arg} is an invalid value inside arg. It should contain only floats or integers."
            raise ValueError(error)


def raise_if_not_valid_alternative(alt: str):
    alt_options = ['two-sided', 'greater', 'less']
    if not isinstance(alt, str):
        error = f"{alt} is an invalid value for alternative. The alternative should be a string."
        raise ValueError(error)

    if alt not in alt_options:
        error = f"{alt} is an invalid value for alternative. The alternative should be one of these: {alt_options}."
        raise ValueError(error)


def raise_if_alpha_not_valid(alpha: Any):
    if not isinstance(alpha, float):
        error = f"{alpha} is an invalid value for alpha. It should be a float."
        raise ValueError(error)

    if not (alpha > 0 and alpha < 1):
        error = f"'{alpha}' is an invalid value for alpha. It should be between 0 and 1."
        raise ValueError(error)



def raise_if_not_bool(arg: Any, argname: str) -> bool:
    if not isinstance(arg, bool):
        error = f"{arg} is an invalid value for {argname}. It should be a boolean."
        raise ValueError(error)
