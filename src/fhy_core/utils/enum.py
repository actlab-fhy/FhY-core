"""String and integer enumeration fix for Python 3.10.

``enum.StrEnum`` was added in Python 3.11; this module backports it (and a
matching ``IntEnum``) for Python 3.10.

Note:
    Remove this module when Python 3.10 support is dropped (i.e. when the
    minimum supported version becomes 3.11). At that point, import ``IntEnum``
    and ``StrEnum`` directly from :mod:`enum` instead.

Note:
    The following code is adapted from the CPython source code. The original
    code can be found at: https://github.com/python/cpython/blob/main/Lib/enum.py

"""

from fhy_core.utils.override import override

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

        __str__ = str.__str__

        @staticmethod
        @override
        def _generate_next_value_(
            name: str, start: int, count: int, last_values: list[Any]
        ) -> str:
            """Return the lower-cased version of the member name."""
            return name.lower()

    class StrEnum(_StrEnum):  # type: ignore[no-redef]
        """String enumeration."""
