from __future__ import annotations

from rhis.evol.validators.decorators import validate_evol_params, validate_plot_params
from rhis.evol.validators.utils import (
    validate_or_raise_plot_rhis_full_with_stat_defined,
    validate_or_raise_rhis_full_evol_not_called,
    validate_or_raise_rhis_statistic_evol_not_called,
    validate_or_raise_target_hyp_param_incorrect,
)
