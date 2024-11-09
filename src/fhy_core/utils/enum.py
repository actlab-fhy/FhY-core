"""String and integer enumeration fix for Python 3.10.

Note:
    The following code is adapted from the CPython source code. The original
    code can be found at: https://github.com/python/cpython/blob/main/Lib/enum.py

"""

try:
    from enum import IntEnum, StrEnum

except ImportError:
    import enum

    class IntEnum(int, enum.Enum):  # type: ignore[no-redef]
        """Integer enumeration."""

    class StrEnum(str, enum.Enum):  # type: ignore[no-redef]
        """String enumeration."""

        def __new__(cls, *values):
            """Values must already be of type `str`."""
            value = str(*values)
            member = str.__new__(cls, value)
            member._value_ = value

            return member

        @staticmethod
        def _generate_next_value_(name, start, count, last_values):
            """Return the lower-cased version of the member name."""
            return name.lower()
