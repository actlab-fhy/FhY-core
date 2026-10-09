"""Tests for `FunctionSort` and its Python runtime-value check.

`FunctionSort` is a `StrEnum` with four members forming a containment
chain `NAT < INT < REAL` plus the side-branch member `BOOL`. Registration
of native functions and constants uses
`is_python_value_compatible_with_sort` to validate the runtime type of
the value it is handed.

The tests in this file pin the helper down on every sort, including the
rejection cases that distinguish the chain members from each other (e.g.
`NAT` accepts `0` but not `-1`; `INT` accepts both).
"""

import math

import pytest

from fhy_core.symbolic.expression.sort import (
    FunctionSort,
    is_python_value_compatible_with_sort,
)

# =============================================================================
# `FunctionSort` enum
# =============================================================================


class TestFunctionSort:
    """Tests for the `FunctionSort` enum surface."""

    def test_members_are_bool_nat_int_real(self) -> None:
        """Test the enum exposes exactly the four expected members."""
        assert {member.name for member in FunctionSort} == {
            "BOOL",
            "NAT",
            "INT",
            "REAL",
        }

    def test_str_enum_values_match_lowercase_names(self) -> None:
        """Test each member's string value is the lowercase member name."""
        assert FunctionSort.BOOL.value == "bool"
        assert FunctionSort.NAT.value == "nat"
        assert FunctionSort.INT.value == "int"
        assert FunctionSort.REAL.value == "real"

    def test_members_are_string_instances(self) -> None:
        """Test members participate as strings (StrEnum semantics)."""
        assert isinstance(FunctionSort.REAL, str)
        assert str(FunctionSort.REAL) == "real"


# =============================================================================
# `is_python_value_compatible_with_sort`
# =============================================================================


class TestIsPythonValueCompatibleWithSort:
    """Tests for `is_python_value_compatible_with_sort`."""

    # ---- BOOL sort ----

    @pytest.mark.parametrize("value", [True, False])
    def test_bool_sort_accepts_bool_values(self, value: bool) -> None:
        """Test the `BOOL` sort accepts `True` and `False`."""
        assert is_python_value_compatible_with_sort(value, FunctionSort.BOOL)

    @pytest.mark.parametrize("value", [0, 1, -1, 0.0, 1.5, -2.5])
    def test_bool_sort_rejects_non_bool_values(self, value: int | float) -> None:
        """Test the `BOOL` sort rejects every non-`bool` numerical value."""
        assert not is_python_value_compatible_with_sort(value, FunctionSort.BOOL)

    # ---- NAT sort ----

    @pytest.mark.parametrize("value", [0, 1, 2, 100])
    def test_nat_sort_accepts_non_negative_int_values(self, value: int) -> None:
        """Test the `NAT` sort accepts non-negative `int` values."""
        assert is_python_value_compatible_with_sort(value, FunctionSort.NAT)

    @pytest.mark.parametrize("value", [-1, -100])
    def test_nat_sort_rejects_negative_int_values(self, value: int) -> None:
        """Test the `NAT` sort rejects negative `int` values."""
        assert not is_python_value_compatible_with_sort(value, FunctionSort.NAT)

    @pytest.mark.parametrize("value", [True, False])
    def test_nat_sort_rejects_bool_values(self, value: bool) -> None:
        """Test the `NAT` sort rejects `bool` (despite `bool` subclassing `int`).

        Pins down the strict-`bool` rule against a permissive
        ``isinstance(value, int)`` implementation.
        """
        assert not is_python_value_compatible_with_sort(value, FunctionSort.NAT)

    @pytest.mark.parametrize("value", [0.0, 1.5, -2.5])
    def test_nat_sort_rejects_float_values(self, value: float) -> None:
        """Test the `NAT` sort rejects `float` values."""
        assert not is_python_value_compatible_with_sort(value, FunctionSort.NAT)

    # ---- INT sort ----

    @pytest.mark.parametrize("value", [0, 1, -1, 100, -100])
    def test_int_sort_accepts_int_values(self, value: int) -> None:
        """Test the `INT` sort accepts any non-`bool` `int`."""
        assert is_python_value_compatible_with_sort(value, FunctionSort.INT)

    @pytest.mark.parametrize("value", [True, False])
    def test_int_sort_rejects_bool_values(self, value: bool) -> None:
        """Test the `INT` sort rejects `bool` values (strict-`bool` rule)."""
        assert not is_python_value_compatible_with_sort(value, FunctionSort.INT)

    @pytest.mark.parametrize("value", [0.0, 1.5, -2.5])
    def test_int_sort_rejects_float_values(self, value: float) -> None:
        """Test the `INT` sort rejects `float` values."""
        assert not is_python_value_compatible_with_sort(value, FunctionSort.INT)

    # ---- REAL sort ----

    @pytest.mark.parametrize("value", [0, 1, -1, 0.0, 1.5, -2.5])
    def test_real_sort_accepts_int_and_float_values(self, value: int | float) -> None:
        """Test the `REAL` sort accepts any non-`bool` `int` or any `float`."""
        assert is_python_value_compatible_with_sort(value, FunctionSort.REAL)

    @pytest.mark.parametrize("value", [True, False])
    def test_real_sort_rejects_bool_values(self, value: bool) -> None:
        """Test the `REAL` sort rejects `bool` values (strict-`bool` rule)."""
        assert not is_python_value_compatible_with_sort(value, FunctionSort.REAL)

    def test_real_sort_accepts_special_float_values(self) -> None:
        """Test the `REAL` sort accepts `math.inf` and `math.nan`.

        These are the float representations of seeded constants and
        must pass registration validation.
        """
        assert is_python_value_compatible_with_sort(math.inf, FunctionSort.REAL)
        assert is_python_value_compatible_with_sort(math.nan, FunctionSort.REAL)
        assert is_python_value_compatible_with_sort(-math.inf, FunctionSort.REAL)
