"""Core compiler errors and error registration."""

__all__ = ["register_error"]

_COMPILER_ERRORS: dict[type[Exception], str] = {}


def register_error(error: type[Exception]) -> type[Exception]:
    """Decorator to register custom compiler exceptions.

    Args:
        error: Custom exception to be registered.

    Returns:
        Custom exception registered

    """
    _COMPILER_ERRORS[error] = error.__doc__ or error.__name__

    return error
