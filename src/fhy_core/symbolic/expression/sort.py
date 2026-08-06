"""Function-level sorts and core-data-type compatibility.

A `FunctionSort` is the type-checker's view of a registered function's
parameter or result. Sorts are intentionally coarser than `CoreDataType`
so that one signature can describe a function over many concrete IR
types (e.g. ``REAL`` accepts every integer and real-float core data
type).

The helpers in this module bridge sorts to concrete IR types: the type
checker uses :func:`is_core_data_type_compatible_with_sort` to validate
arguments at call sites and :func:`get_result_core_data_type_for_sort`
to synthesize result types. Registration of native functions and
constants uses :func:`is_python_value_compatible_with_sort` to validate
Python runtime values at registration time.
"""

__all__ = [
    "FunctionSort",
    "get_result_core_data_type_for_sort",
    "is_core_data_type_compatible_with_sort",
    "is_python_value_compatible_with_sort",
]

from fhy_core.types.core import CoreDataType
from fhy_core.utils import StrEnum


class FunctionSort(StrEnum):
    """Coarse mathematical sort of a function parameter or result.

    Sorts form a containment chain ``NAT < INT < REAL``: an argument
    whose core data type satisfies a narrower sort also satisfies every
    wider sort. ``BOOL`` is a side branch and is never compatible with
    the numeric sorts.

    Contrast :class:`~fhy_core.symbolic.symbol_type.SymbolType`, which
    selects a Z3 lowering sort for an expression rather than describing a
    registered function's declared signature.

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


_UINT_CORE_DATA_TYPES: frozenset[CoreDataType] = frozenset(
    {
        CoreDataType.UINT,
        CoreDataType.UINT8,
        CoreDataType.UINT16,
        CoreDataType.UINT32,
    }
)

_SIGNED_INT_CORE_DATA_TYPES: frozenset[CoreDataType] = frozenset(
    {
        CoreDataType.INT,
        CoreDataType.INT8,
        CoreDataType.INT16,
        CoreDataType.INT32,
        CoreDataType.INT64,
    }
)

_INTEGER_CORE_DATA_TYPES: frozenset[CoreDataType] = (
    _UINT_CORE_DATA_TYPES | _SIGNED_INT_CORE_DATA_TYPES
)

_REAL_FLOAT_CORE_DATA_TYPES: frozenset[CoreDataType] = frozenset(
    {
        CoreDataType.FLOAT,
        CoreDataType.FLOAT16,
        CoreDataType.FLOAT32,
        CoreDataType.FLOAT64,
    }
)

_REAL_CORE_DATA_TYPES: frozenset[CoreDataType] = (
    _INTEGER_CORE_DATA_TYPES | _REAL_FLOAT_CORE_DATA_TYPES
)


_SORT_TO_COMPATIBLE_CORE_DATA_TYPES: dict[FunctionSort, frozenset[CoreDataType]] = {
    FunctionSort.BOOL: frozenset({CoreDataType.BOOL}),
    FunctionSort.NAT: _UINT_CORE_DATA_TYPES,
    FunctionSort.INT: _INTEGER_CORE_DATA_TYPES,
    FunctionSort.REAL: _REAL_CORE_DATA_TYPES,
}


# Concrete (non-weak) result types so downstream arithmetic on the
# returned value triggers the type-checker's weak-literal rescue against
# this operand. Widths match Python's native runtime types (int / float
# map to INT64 / FLOAT64).
_SORT_TO_RESULT_CORE_DATA_TYPE: dict[FunctionSort, CoreDataType] = {
    FunctionSort.BOOL: CoreDataType.BOOL,
    FunctionSort.NAT: CoreDataType.UINT32,
    FunctionSort.INT: CoreDataType.INT64,
    FunctionSort.REAL: CoreDataType.FLOAT64,
}


def is_core_data_type_compatible_with_sort(
    core_data_type: CoreDataType, sort: FunctionSort
) -> bool:
    """Return whether ``core_data_type`` satisfies ``sort``.

    Args:
        core_data_type: Synthesized core data type of a value at the
            call site.
        sort: Declared sort of the corresponding parameter or result.

    Returns:
        ``True`` when ``core_data_type`` is in the family admitted by
        ``sort``; ``False`` otherwise.

    """
    return core_data_type in _SORT_TO_COMPATIBLE_CORE_DATA_TYPES[sort]


def get_result_core_data_type_for_sort(sort: FunctionSort) -> CoreDataType:
    """Return the core data type assigned to a value of ``sort``.

    Used for both function results and constant references; the caller
    wraps the returned type in ``NumericalType(PrimitiveDataType(...))``.

    Args:
        sort: Declared result or constant sort.

    Returns:
        The concrete core data type to assign to a value of ``sort``.

    """
    return _SORT_TO_RESULT_CORE_DATA_TYPE[sort]


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
