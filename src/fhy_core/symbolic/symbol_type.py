"""Symbol type tag for compiler symbols (real, integer, boolean)."""

__all__ = ["SymbolType"]

from enum import Enum, auto


class SymbolType(Enum):
    """Symbol type."""

    REAL = auto()
    INT = auto()
    BOOL = auto()
