"""Function-level sorts.

A `FunctionSort` is the coarse mathematical classification of a
registered function's parameter or result. Sorts describe a whole family
of concrete values at once, so a single declared signature covers a
function over many runtime types.

This module owns the sort vocabulary and the check against Python
runtime values: registration of native functions and constants uses
:func:`is_python_value_compatible_with_sort` to validate the values it
is handed. Translating a sort to an IR core data type is the job of
the type-checking layer, which sits above this package.
"""

__all__ = [
    "FunctionSort",
    "is_python_value_compatible_with_sort",
]

from fhy_core.utils import StrEnum


class FunctionSort(StrEnum):
    """Coarse mathematical sort of a function parameter or result.

    Sorts form a containment chain ``NAT < INT < REAL``: an argument
    whose core data type satisfies a narrower sort also satisfies every
    wider sort. ``BOOL`` is a side branch and is never compatible with
    the numeric sorts.

    Members:
        BOOL: A boolean value. Compatible only with ``CoreDataType.BOOL``.
        NAT: A non-negative integer. Compatible with the unsigned-
            integer family (``UINT``, ``UINT8``, ``UINT16``, ``UINT32``).
        INT: A (possibly negative) integer. Compatible with the signed
            and unsigned integer families.
        REAL: A real number. Compatible with the integer and real-
            float families.
    """

    BOOL = "bool"
    NAT = "nat"
    INT = "int"
    REAL = "real"


def is_python_value_compatible_with_sort(
    value: bool | int | float, sort: FunctionSort
) -> bool:
    """Return whether a Python runtime ``value`` is compatible with ``sort``.

    The check is strict on ``bool``: ``True`` / ``False`` satisfy only
    ``BOOL``, not ``NAT`` / ``INT`` / ``REAL``, even though Python's
    ``bool`` is a subclass of ``int``. This prevents silently
    registering a boolean as a numeric constant.

    Args:
        value: Python runtime value to classify.
        sort: Declared sort the value must satisfy.

    Returns:
        ``True`` when ``value`` is compatible with ``sort``; ``False``
        otherwise.

    """
    value_is_bool = isinstance(value, bool)
    if sort is FunctionSort.BOOL:
        return value_is_bool
    elif value_is_bool:
        return False
    elif sort is FunctionSort.NAT:
        return isinstance(value, int) and value >= 0
    elif sort is FunctionSort.INT:
        return isinstance(value, int)
    else:
        return isinstance(value, (int, float))
