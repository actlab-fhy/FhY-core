"""String and integer enumeration fix for Python 3.10.

Note:
    The following code is adapted from the CPython source code. The original
    code can be found at: https://github.com/python/cpython/blob/main/Lib/enum.py

"""

__all__ = ["IntEnum", "StrEnum"]

from typing import Any

try:
    from enum import IntEnum, StrEnum

except ImportError:
    import enum

    class IntEnum(int, enum.Enum):  # type: ignore[no-redef]
        """Integer enumeration."""

    class _StrEnum(str, enum.Enum):
        """String enumeration."""

        def __new__(cls, *values: str) -> "_StrEnum":
            value = str(*values)
            member = str.__new__(cls, value)
            member._value_ = value

            return member

        @staticmethod
        def _generate_next_value_(
            name: str, start: int, count: int, last_values: list[Any]
        ) -> str:
            """Return the lower-cased version of the member name."""
            return name.lower()

    class StrEnum(_StrEnum):  # type: ignore[no-redef]
        """String enumeration."""
