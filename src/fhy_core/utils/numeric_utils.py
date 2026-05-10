"""Numeric type predicates."""

__all__ = ["is_strict_int"]

from typing import TypeGuard


def is_strict_int(value: object) -> TypeGuard[int]:
    """Return whether ``value`` is an ``int`` and not a ``bool``.

    Python's ``bool`` is a subclass of ``int``, so ``isinstance(True, int)``
    is ``True``. In contexts that require a strict integer (numeric
    parameter bounds, span offsets, position coordinates), use this
    predicate to reject booleans without rejecting other valid integer
    inputs.

    Args:
        value: The value to check.

    Returns:
        ``True`` if ``value`` is an ``int`` whose runtime type is not
        ``bool``; ``False`` otherwise.

    """
    return isinstance(value, int) and not isinstance(value, bool)
