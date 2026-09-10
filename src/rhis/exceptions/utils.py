from __future__ import annotations

from loguru import logger

from rhis.exceptions import RhisEvolNotCalledError


def raise_if_no_rhis_run(*, is_rhis_complete: bool) -> None:
    if not is_rhis_complete:
        msg = 'Run `Rhis.evol()` before plot.'
        logger.debug(msg)
        raise RhisEvolNotCalledError(msg)
