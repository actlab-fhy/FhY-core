"""Tests for `FunctionSort` and the sort-compatibility helpers.

`FunctionSort` is a `StrEnum` with four members forming a containment
chain `NAT < INT < REAL` plus the side-branch member `BOOL`. The
helpers translate between sorts and IR core data types: the call-site
type checker uses `is_core_data_type_compatible_with_sort` and
`get_result_core_data_type_for_sort`; registration of native constants
uses `is_python_value_compatible_with_sort` to validate the runtime
type of the value.

The tests in this file pin each helper down on every sort, including
the rejection cases that distinguish the chain members from each other
(e.g. `NAT` accepts `UINT8` but not `INT8`; `INT` accepts both).
"""

import math

import pytest

from fhy_core.expression.sort import (
    FunctionSort,
    get_result_core_data_type_for_sort,
    is_core_data_type_compatible_with_sort,
    is_python_value_compatible_with_sort,
)
from fhy_core.types.core import CoreDataType

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
# `is_core_data_type_compatible_with_sort`
# =============================================================================


_UINT_CORE_TYPES = (
    CoreDataType.UINT,
    CoreDataType.UINT8,
    CoreDataType.UINT16,
    CoreDataType.UINT32,
)
_SIGNED_INT_CORE_TYPES = (
    CoreDataType.INT,
    CoreDataType.INT8,
    CoreDataType.INT16,
    CoreDataType.INT32,
    CoreDataType.INT64,
)
_REAL_FLOAT_CORE_TYPES = (
    CoreDataType.FLOAT,
    CoreDataType.FLOAT16,
    CoreDataType.FLOAT32,
    CoreDataType.FLOAT64,
)
_COMPLEX_CORE_TYPES = (
    CoreDataType.COMPLEX32,
    CoreDataType.COMPLEX64,
    CoreDataType.COMPLEX128,
)


class TestIsCoreDataTypeCompatibleWithSort:
    """Tests for `is_core_data_type_compatible_with_sort`."""

    # ---- BOOL sort ----

    def test_bool_sort_accepts_bool_core_type(self) -> None:
        """Test `BOOL` sort accepts the `BOOL` core data type."""
        assert is_core_data_type_compatible_with_sort(
            CoreDataType.BOOL, FunctionSort.BOOL
        )

    @pytest.mark.parametrize(
        "core_data_type",
        [
            *_UINT_CORE_TYPES,
            *_SIGNED_INT_CORE_TYPES,
            *_REAL_FLOAT_CORE_TYPES,
            *_COMPLEX_CORE_TYPES,
        ],
    )
    def test_bool_sort_rejects_every_non_bool_core_type(
        self, core_data_type: CoreDataType
    ) -> None:
        """Test `BOOL` sort rejects every non-`BOOL` core data type."""
        assert not is_core_data_type_compatible_with_sort(
            core_data_type, FunctionSort.BOOL
        )

    # ---- NAT sort ----

    @pytest.mark.parametrize("core_data_type", _UINT_CORE_TYPES)
    def test_nat_sort_accepts_every_unsigned_integer_core_type(
        self, core_data_type: CoreDataType
    ) -> None:
        """Test `NAT` accepts every unsigned-integer core data type."""
        assert is_core_data_type_compatible_with_sort(core_data_type, FunctionSort.NAT)

    @pytest.mark.parametrize(
        "core_data_type",
        [
            CoreDataType.BOOL,
            *_SIGNED_INT_CORE_TYPES,
            *_REAL_FLOAT_CORE_TYPES,
            *_COMPLEX_CORE_TYPES,
        ],
    )
    def test_nat_sort_rejects_non_unsigned_integer_core_types(
        self, core_data_type: CoreDataType
    ) -> None:
        """Test `NAT` rejects every non-unsigned-integer core data type."""
        assert not is_core_data_type_compatible_with_sort(
            core_data_type, FunctionSort.NAT
        )

    # ---- INT sort ----

    @pytest.mark.parametrize(
        "core_data_type", [*_UINT_CORE_TYPES, *_SIGNED_INT_CORE_TYPES]
    )
    def test_int_sort_accepts_every_integer_core_type(
        self, core_data_type: CoreDataType
    ) -> None:
        """Test `INT` accepts every signed and unsigned integer core data type."""
        assert is_core_data_type_compatible_with_sort(core_data_type, FunctionSort.INT)

    @pytest.mark.parametrize(
        "core_data_type",
        [CoreDataType.BOOL, *_REAL_FLOAT_CORE_TYPES, *_COMPLEX_CORE_TYPES],
    )
    def test_int_sort_rejects_non_integer_core_types(
        self, core_data_type: CoreDataType
    ) -> None:
        """Test `INT` rejects `BOOL`, real-float, and complex core data types."""
        assert not is_core_data_type_compatible_with_sort(
            core_data_type, FunctionSort.INT
        )

    # ---- REAL sort ----

    @pytest.mark.parametrize(
        "core_data_type",
        [*_UINT_CORE_TYPES, *_SIGNED_INT_CORE_TYPES, *_REAL_FLOAT_CORE_TYPES],
    )
    def test_real_sort_accepts_integer_and_real_float_core_types(
        self, core_data_type: CoreDataType
    ) -> None:
        """Test `REAL` accepts every integer and real-float core data type."""
        assert is_core_data_type_compatible_with_sort(core_data_type, FunctionSort.REAL)

    @pytest.mark.parametrize(
        "core_data_type", [CoreDataType.BOOL, *_COMPLEX_CORE_TYPES]
    )
    def test_real_sort_rejects_bool_and_complex_core_types(
        self, core_data_type: CoreDataType
    ) -> None:
        """Test `REAL` rejects `BOOL` and every complex core data type."""
        assert not is_core_data_type_compatible_with_sort(
            core_data_type, FunctionSort.REAL
        )


# =============================================================================
# `get_result_core_data_type_for_sort`
# =============================================================================


class TestGetResultCoreDataTypeForSort:
    """Tests for `get_result_core_data_type_for_sort`."""

    def test_bool_sort_maps_to_bool_core_type(self) -> None:
        """Test the `BOOL` sort maps to `CoreDataType.BOOL`."""
        assert (
            get_result_core_data_type_for_sort(FunctionSort.BOOL) == CoreDataType.BOOL
        )

    def test_nat_sort_maps_to_concrete_uint32_core_type(self) -> None:
        """Test the `NAT` sort maps to the concrete `CoreDataType.UINT32`.

        Concrete (rather than weak ``UINT``) so downstream arithmetic
        with literal operands triggers the type-checker's weak-literal
        rescue.
        """
        assert (
            get_result_core_data_type_for_sort(FunctionSort.NAT) == CoreDataType.UINT32
        )

    def test_int_sort_maps_to_concrete_int64_core_type(self) -> None:
        """Test the `INT` sort maps to the concrete `CoreDataType.INT64`."""
        assert (
            get_result_core_data_type_for_sort(FunctionSort.INT) == CoreDataType.INT64
        )

    def test_real_sort_maps_to_concrete_float64_core_type(self) -> None:
        """Test the `REAL` sort maps to the concrete `CoreDataType.FLOAT64`.

        Matches the precision of Python's native ``float`` so a native
        result composes cleanly with downstream literal arithmetic.
        """
        assert (
            get_result_core_data_type_for_sort(FunctionSort.REAL)
            == CoreDataType.FLOAT64
        )


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
