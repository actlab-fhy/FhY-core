"""Symbol type tag for compiler symbols (real, integer, boolean)."""

__all__ = ["SymbolType"]

from fhy_core.utils import StrEnum


class SymbolType(StrEnum):
    """Symbol type.

    Contrast :class:`~fhy_core.symbolic.expression.sort.FunctionSort`, which
    tags a registered function's declared parameter or result sort rather
    than an expression's Z3 lowering sort.
    """

    REAL = "real"
    INT = "int"
    BOOL = "bool"
