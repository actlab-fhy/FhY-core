"""Core compiler errors and error registration."""

__all__ = ["get_registered_errors", "register_error"]

from collections.abc import Mapping
from types import MappingProxyType
from typing import TypeVar

_COMPILER_ERRORS: dict[type[Exception], str] = {}

_ErrorClassT = TypeVar("_ErrorClassT", bound=type[Exception])


def register_error(error: _ErrorClassT) -> _ErrorClassT:
    """Decorator to register custom compiler exceptions.

    Decorated exception classes are added to a read-only catalog that
    tooling (CLI ``--list-errors`` flags, documentation generators) can
    consult via :func:`get_registered_errors`. The class is returned
    unchanged and typed as the class it is, so a type checker that
    applies class decorators still sees the attributes it adds.

    Args:
        error: Custom exception to be registered.

    Returns:
        The registered exception class itself.

    """
    _COMPILER_ERRORS[error] = error.__doc__ or error.__name__

    return error


def get_registered_errors() -> Mapping[type[Exception], str]:
    """Return a read-only view of all ``@register_error``-decorated classes.

    Maps each registered exception class to its docstring (or class name
    when the docstring is empty). Intended for tooling -- CLI help,
    documentation generators, error catalogs -- not for runtime
    discrimination of exceptions.
    """
    return MappingProxyType(_COMPILER_ERRORS)
