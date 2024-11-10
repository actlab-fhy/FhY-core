"""Core compiler errors and error registration."""

COMPILER_ERRORS: dict[type[Exception], str] = {}


def register_error(error: type[Exception]) -> type[Exception]:
    """Decorator to register custom compiler exceptions.

    Args:
        error: Custom exception to be registered.

    Returns:
        Custom exception registered

    """
    COMPILER_ERRORS[error] = error.__doc__ or error.__name__

    return error


@register_error
class FhYCoreTypeError(TypeError):
    """Core type error."""


@register_error
class SymbolTableError(Exception):
    """Symbol table error."""
