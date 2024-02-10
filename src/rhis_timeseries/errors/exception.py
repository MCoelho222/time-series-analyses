"""Functions for handling try except errors."""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class Error:
    """A detailed Error handle class."""

    exception: Exception
    type: str
    details: str
    message: str

    @staticmethod
    def from_exception(exception: Exception, message: str = '') -> Error:
        """Construct the detailed error class from an exception."""
        return Error(exception, str(exception.__class__.__name__), str(exception.__cause__), message)

    def to_dict(self) -> dict[str, str]:
        """Generate string dictionary from the detailed error."""
        return {k: str(v) for k, v in asdict(self).items()}


def handle_exception_msg(exception: Exception, msg: str) -> Error:
    """
    Handle an error message by formatting it into a dictionary.

    Parameters
    ----------
        exception
            The exception or its type.
        msg
            The error message.

    Returns
    -------
        Error
            Includes the formatted error message, exception, type and details.
    """
    return Error.from_exception(exception, msg)
