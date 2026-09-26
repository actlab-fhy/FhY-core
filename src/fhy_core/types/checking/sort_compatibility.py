"""Bridge between function sorts and concrete IR core data types.

A :class:`~fhy_core.symbolic.expression.FunctionSort` is deliberately coarser
than a :class:`~fhy_core.types.CoreDataType`, so one declared signature
describes a function over a whole family of concrete IR types (``REAL``
admits every integer and real-float core data type, for example).

The two helpers here are the only place that mapping is exposed; the
Rust core holds it (D-S11-22 of ``docs/design/python-switch.md``): ``BOOL``
satisfies the Boolean sort, the unsigned integers ``NAT``, every integer
``INT``, and every integer and real float ``REAL``, and a value of a sort
takes ``BOOL``, ``UINT32``, ``INT64`` or ``FLOAT64``. The type checker
uses :func:`is_core_data_type_compatible_with_sort` to
validate arguments at call sites and
:func:`get_result_core_data_type_for_sort` to synthesize the type of a
function result or constant reference.
"""

__all__ = [
    "get_result_core_data_type_for_sort",
    "is_core_data_type_compatible_with_sort",
]

from fhy_core import _rs
from fhy_core.symbolic.expression.sort import FunctionSort

from ..core import CoreDataType


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
    return _rs.is_core_data_type_compatible_with_sort(core_data_type, sort)


def get_result_core_data_type_for_sort(sort: FunctionSort) -> CoreDataType:
    """Return the core data type assigned to a value of ``sort``.

    Used for both function results and constant references; the caller
    wraps the returned type in ``NumericalType(PrimitiveDataType(...))``.

    Args:
        sort: Declared result or constant sort.

    Returns:
        The concrete core data type to assign to a value of ``sort``.

    """
    return _rs.get_result_core_data_type_for_sort(sort)
