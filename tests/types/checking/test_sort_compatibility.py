"""Tests for the sort-to-core-data-type bridge helpers.

`FunctionSort` is deliberately coarser than `CoreDataType`, so one
declared signature covers a function over a family of concrete IR types.
The call-site type checker uses `is_core_data_type_compatible_with_sort`
to validate arguments and `get_result_core_data_type_for_sort` to
synthesize result types.

The tests in this file pin each helper down on every sort, including the
rejection cases that distinguish the chain members from each other (e.g.
`NAT` accepts `UINT8` but not `INT8`; `INT` accepts both).
"""

import pytest

from fhy_core.symbolic.expression.sort import FunctionSort
from fhy_core.types.checking.sort_compatibility import (
    get_result_core_data_type_for_sort,
    is_core_data_type_compatible_with_sort,
)
from fhy_core.types.core import CoreDataType

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
