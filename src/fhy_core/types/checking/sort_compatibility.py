"""Bridge between function sorts and concrete IR core data types.

A :class:`~fhy_core.symbolic.expression.FunctionSort` is deliberately coarser
than a :class:`~fhy_core.types.CoreDataType`, so one declared signature
describes a function over a whole family of concrete IR types (``REAL``
admits every integer and real-float core data type, for example).

The two helpers here are the only place that mapping is written down.
The type checker uses :func:`is_core_data_type_compatible_with_sort` to
validate arguments at call sites and
:func:`get_result_core_data_type_for_sort` to synthesize the type of a
function result or constant reference.
"""

__all__ = [
    "get_result_core_data_type_for_sort",
    "is_core_data_type_compatible_with_sort",
]

from fhy_core.symbolic.expression.sort import FunctionSort

from ..core import CoreDataType

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
