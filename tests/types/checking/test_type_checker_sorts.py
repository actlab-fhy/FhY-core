"""Type-checker tests for the sort-driven call-site path and constant references.

The call-site type checker validates arguments against the registered
function's declared sorts and synthesizes the result type from them.
For each argument, the synthesized core data type is checked against
the corresponding parameter sort via
``is_core_data_type_compatible_with_sort``; the result type is the
sort-derived weak core data type wrapped in
``NumericalType(PrimitiveDataType(...))``.

Constant references in the expression tree are resolved against the
registry when the local lookup does not bind the identifier; the
synthesized type is the constant's sort-derived weak type, qualifier
``PARAM``.

Arithmetic and piecewise cases are covered in ``test_type_checker.py``
and ``test_type_checker_booleans.py``.
"""

import math
from collections.abc import Callable, Sequence

import pytest

from fhy_core.identifier import Identifier
from fhy_core.symbolic.expression import (
    CallExpression,
    FunctionSort,
    IdentifierExpression,
    LiteralExpression,
    NativeFunction,
    call,
    register_function,
    register_native_constant,
    register_native_function,
)
from fhy_core.types import (
    CoreDataType,
    FhYCoreTypeError,
    NumericalType,
    PrimitiveDataType,
    Type,
    TypeQualifier,
)
from fhy_core.types.checking.type_checker import synthesize_expression_type

from ..conftest import mock_identifier


def _scalar(core_data_type: CoreDataType) -> NumericalType:
    return NumericalType(PrimitiveDataType(core_data_type))


def _unexpected_lookup(identifier: Identifier) -> tuple[Type, TypeQualifier]:
    raise KeyError(identifier.name_hint)


def _single_lookup(
    expected_identifier: Identifier, value_type: Type, qualifier: TypeQualifier
) -> Callable[[Identifier], tuple[Type, TypeQualifier]]:
    def lookup(identifier: Identifier) -> tuple[Type, TypeQualifier]:
        assert identifier == expected_identifier, (
            f"unexpected identifier lookup: {identifier}"
        )
        return value_type, qualifier

    return lookup


def _placeholder_native_impl(*_: object) -> float:
    """Placeholder native implementation; tests in this file never evaluate."""
    return 0.0


def register_native(
    name: str,
    parameter_sorts: Sequence[FunctionSort],
    result_sort: FunctionSort,
) -> NativeFunction:
    """Register a native function with a placeholder implementation."""
    return register_native_function(
        name=name,
        parameter_sorts=parameter_sorts,
        result_sort=result_sort,
        implementation=_placeholder_native_impl,
    )


def register_real_unary_native(
    name: str, result_sort: FunctionSort = FunctionSort.REAL
) -> NativeFunction:
    """Register a unary native function with a ``REAL`` parameter."""
    return register_native(name, [FunctionSort.REAL], result_sort)


# =============================================================================
# Sort-driven call-site type inference
# =============================================================================


class TestNativeCallSiteSortInference:
    """Tests for the sort-driven path on `CallExpression` whose target is native."""

    def test_native_call_with_compatible_real_argument_returns_concrete_float64(
        self, function_registry_snapshot: None
    ) -> None:
        """Test a `REAL`-sort native call with a literal arg yields `FLOAT64`."""
        register_real_unary_native("test_tc_exp")
        expression = CallExpression("test_tc_exp", (LiteralExpression(0.0),))

        result_type, _ = synthesize_expression_type(expression, _unexpected_lookup)

        assert result_type.is_structurally_equivalent(_scalar(CoreDataType.FLOAT64))

    def test_native_call_with_int_result_sort_returns_concrete_int64(
        self, function_registry_snapshot: None
    ) -> None:
        """Test a native with declared `INT` result sort yields concrete `INT64`."""
        register_real_unary_native("test_tc_ceil", result_sort=FunctionSort.INT)
        expression = CallExpression("test_tc_ceil", (LiteralExpression(2.5),))

        result_type, _ = synthesize_expression_type(expression, _unexpected_lookup)

        assert result_type.is_structurally_equivalent(_scalar(CoreDataType.INT64))

    def test_native_call_with_bool_result_sort_returns_bool_type(
        self, function_registry_snapshot: None
    ) -> None:
        """Test a native function with declared `BOOL` result sort yields `BOOL`."""
        register_real_unary_native("test_tc_bool_native", result_sort=FunctionSort.BOOL)
        expression = CallExpression("test_tc_bool_native", (LiteralExpression(1.0),))

        result_type, _ = synthesize_expression_type(expression, _unexpected_lookup)

        assert result_type.is_structurally_equivalent(_scalar(CoreDataType.BOOL))

    def test_native_call_propagates_param_qualifier_when_all_args_are_param(
        self, function_registry_snapshot: None
    ) -> None:
        """Test all-`PARAM` argument qualifiers promote to `PARAM` result."""
        register_real_unary_native("test_tc_param_qualifier")

        expression = CallExpression(
            "test_tc_param_qualifier", (LiteralExpression(4.0),)
        )
        _, qualifier = synthesize_expression_type(expression, _unexpected_lookup)

        assert qualifier == TypeQualifier.PARAM

    def test_native_call_promotes_qualifier_to_temp_when_any_argument_is_temp(
        self, function_registry_snapshot: None
    ) -> None:
        """Test a `TEMP` argument promotes the call qualifier to `TEMP`."""
        register_real_unary_native("test_tc_temp_qualifier")
        x = mock_identifier("x", 0)
        expression = call("test_tc_temp_qualifier", x)

        lookup = _single_lookup(x, _scalar(CoreDataType.FLOAT64), TypeQualifier.TEMP)
        _, qualifier = synthesize_expression_type(expression, lookup)

        assert qualifier == TypeQualifier.TEMP


# =============================================================================
# Sort violations
# =============================================================================


class TestNativeCallSortViolations:
    """Tests for argument / parameter sort mismatches at the call site."""

    def test_real_sort_native_rejects_bool_argument(
        self, function_registry_snapshot: None
    ) -> None:
        """Test passing a `BOOL` literal to a `REAL` parameter is rejected."""
        register_real_unary_native("test_tc_real_rejects_bool")
        expression = CallExpression(
            "test_tc_real_rejects_bool", (LiteralExpression(True),)
        )

        with pytest.raises(FhYCoreTypeError, match="test_tc_real_rejects_bool"):
            synthesize_expression_type(expression, _unexpected_lookup)

    def test_nat_sort_native_rejects_signed_integer_argument(
        self, function_registry_snapshot: None
    ) -> None:
        """Test passing a signed-integer literal to a `NAT` parameter is rejected."""
        register_native(
            "test_tc_nat_rejects_signed",
            parameter_sorts=[FunctionSort.NAT],
            result_sort=FunctionSort.NAT,
        )
        # ``LiteralExpression(-1)`` synthesizes to weak ``INT``, which is
        # not compatible with the unsigned-only ``NAT`` sort.
        expression = CallExpression(
            "test_tc_nat_rejects_signed", (LiteralExpression(-1),)
        )

        with pytest.raises(FhYCoreTypeError, match="test_tc_nat_rejects_signed"):
            synthesize_expression_type(expression, _unexpected_lookup)

    def test_int_sort_native_rejects_float_argument(
        self, function_registry_snapshot: None
    ) -> None:
        """Test passing a `FLOAT` literal to an `INT` parameter is rejected."""
        register_native(
            "test_tc_int_rejects_float",
            parameter_sorts=[FunctionSort.INT],
            result_sort=FunctionSort.INT,
        )
        expression = CallExpression(
            "test_tc_int_rejects_float", (LiteralExpression(1.5),)
        )

        with pytest.raises(FhYCoreTypeError, match="test_tc_int_rejects_float"):
            synthesize_expression_type(expression, _unexpected_lookup)

    def test_bool_sort_native_rejects_int_argument(
        self, function_registry_snapshot: None
    ) -> None:
        """Test passing an integer literal to a `BOOL` parameter is rejected."""
        register_native(
            "test_tc_bool_rejects_int",
            parameter_sorts=[FunctionSort.BOOL],
            result_sort=FunctionSort.BOOL,
        )
        expression = CallExpression("test_tc_bool_rejects_int", (LiteralExpression(1),))

        with pytest.raises(FhYCoreTypeError, match="test_tc_bool_rejects_int"):
            synthesize_expression_type(expression, _unexpected_lookup)

    def test_native_call_with_wrong_arity_raises(
        self, function_registry_snapshot: None
    ) -> None:
        """Test a native call with the wrong argument count is rejected."""
        register_native(
            "test_tc_arity",
            parameter_sorts=[FunctionSort.REAL, FunctionSort.REAL],
            result_sort=FunctionSort.REAL,
        )
        expression = CallExpression("test_tc_arity", (LiteralExpression(1.0),))

        with pytest.raises(FhYCoreTypeError, match="test_tc_arity"):
            synthesize_expression_type(expression, _unexpected_lookup)


# =============================================================================
# Expression-bodied call sites use the declared sort, not the body
# =============================================================================


class TestExpressionBodiedCallSiteSortInference:
    """Tests for `RegisteredFunction` call sites using the declared sort."""

    def test_expression_bodied_call_uses_declared_result_sort(
        self, function_registry_snapshot: None
    ) -> None:
        """Test the call-site type is built from the declared `result_sort`."""
        parameter = mock_identifier("x", 0)
        register_function(
            "test_tc_expr_bodied_real",
            parameters=[parameter],
            parameter_sorts=[FunctionSort.REAL],
            result_sort=FunctionSort.REAL,
            body=IdentifierExpression(parameter),
        )
        expression = CallExpression("test_tc_expr_bodied_real", (LiteralExpression(0),))

        result_type, _ = synthesize_expression_type(expression, _unexpected_lookup)

        assert result_type.is_structurally_equivalent(_scalar(CoreDataType.FLOAT64))

    def test_expression_bodied_call_rejects_sort_incompatible_argument(
        self, function_registry_snapshot: None
    ) -> None:
        """Test the call-site validates arguments against declared parameter sorts."""
        parameter = mock_identifier("x", 0)
        register_function(
            "test_tc_expr_bodied_arg_sort",
            parameters=[parameter],
            parameter_sorts=[FunctionSort.REAL],
            result_sort=FunctionSort.REAL,
            body=IdentifierExpression(parameter),
        )
        expression = CallExpression(
            "test_tc_expr_bodied_arg_sort", (LiteralExpression(True),)
        )

        with pytest.raises(FhYCoreTypeError, match="test_tc_expr_bodied_arg_sort"):
            synthesize_expression_type(expression, _unexpected_lookup)


# =============================================================================
# Native-constant identifier-reference resolution
# =============================================================================


class TestNativeConstantTypeInference:
    """Tests for `IdentifierExpression` references that resolve to a constant."""

    def test_constant_reference_synthesizes_float64_for_real_sort(
        self, function_registry_snapshot: None
    ) -> None:
        """Test a `REAL`-sort constant reference yields `FLOAT64` and `PARAM`."""
        register_native_constant(
            "test_tc_const_real", sort=FunctionSort.REAL, value=math.pi
        )
        expression = IdentifierExpression(mock_identifier("test_tc_const_real", 0))

        # The local lookup is not consulted because the identifier
        # resolves through the registry.
        result_type, qualifier = synthesize_expression_type(
            expression, _unexpected_lookup
        )

        assert result_type.is_structurally_equivalent(_scalar(CoreDataType.FLOAT64))
        assert qualifier == TypeQualifier.PARAM

    def test_constant_reference_synthesizes_int64_for_int_sort(
        self, function_registry_snapshot: None
    ) -> None:
        """Test an `INT`-sort constant reference yields concrete `INT64`."""
        register_native_constant("test_tc_const_int", sort=FunctionSort.INT, value=42)
        expression = IdentifierExpression(mock_identifier("test_tc_const_int", 0))

        result_type, _ = synthesize_expression_type(expression, _unexpected_lookup)

        assert result_type.is_structurally_equivalent(_scalar(CoreDataType.INT64))

    def test_local_identifier_binding_shadows_registered_constant(
        self, function_registry_snapshot: None
    ) -> None:
        """Test the local `get_identifier_type` lookup shadows a same-named constant.

        When the identifier resolves to a value in the local lookup, the
        local type is used; the registry is consulted only as a fallback.
        """
        register_native_constant(
            "test_tc_const_shadow", sort=FunctionSort.REAL, value=math.pi
        )
        identifier = mock_identifier("test_tc_const_shadow", 0)
        expression = IdentifierExpression(identifier)

        # Local lookup returns INT32 — different from the constant's REAL sort.
        lookup = _single_lookup(
            identifier, _scalar(CoreDataType.INT32), TypeQualifier.INPUT
        )
        result_type, qualifier = synthesize_expression_type(expression, lookup)

        assert result_type.is_structurally_equivalent(_scalar(CoreDataType.INT32))
        assert qualifier == TypeQualifier.INPUT


# =============================================================================
# Calling a constant is a dedicated diagnostic
# =============================================================================


class TestCallingAConstantIsRejected:
    """Tests for the dedicated diagnostic raised when a constant is called."""

    def test_calling_a_registered_constant_raises_dedicated_error(
        self, function_registry_snapshot: None
    ) -> None:
        """Test a `CallExpression` whose target is a constant raises a typed error."""
        register_native_constant(
            "test_tc_const_called", sort=FunctionSort.REAL, value=math.pi
        )
        expression = CallExpression("test_tc_const_called", ())

        with pytest.raises(FhYCoreTypeError) as exc_info:
            synthesize_expression_type(expression, _unexpected_lookup)

        message = str(exc_info.value)
        assert "test_tc_const_called" in message
        assert "constant" in message.lower()
