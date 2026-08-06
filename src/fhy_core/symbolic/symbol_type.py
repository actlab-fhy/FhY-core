"""Symbol type tag for compiler symbols (real, integer, boolean)."""

__all__ = ["SymbolType"]

from fhy_core.utils import StrEnum


class SymbolType(StrEnum):
    """Symbol type."""

    REAL = "real"
    INT = "int"
    BOOL = "bool"
