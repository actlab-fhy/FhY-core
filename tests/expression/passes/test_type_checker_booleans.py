"""Type-checker tests for booleans, ``TernaryExpression``, and ``CallExpression``.

Lives alongside ``test_type_checker.py``; this file holds new cases for
the boolean type, the ternary conditional node, and the function-call
node introduced by the expression-functions feature.
"""

import pytest

from fhy_core.expression import (
    BinaryExpression,
    BinaryOperation,
    CallExpression,
    FunctionSort,
    IdentifierExpression,
    LiteralExpression,
    TernaryExpression,
    UnaryExpression,
    UnaryOperation,
    register_function,
)
from fhy_core.expression.passes.type_checker import (
    get_core_data_type_from_literal_type,
    synthesize_expression_type,
)
from fhy_core.identifier import Identifier
from fhy_core.types import (
    CoreDataType,
    FhYCoreTypeError,
    NumericalType,
    PrimitiveDataType,
    Type,
    TypeQualifier,
)

from ..conftest import make_identifier_checker, make_single_type_checker


def _make_scalar(core_data_type: CoreDataType) -> NumericalType:
    return NumericalType(PrimitiveDataType(core_data_type))


_BOOL_TYPE = _make_scalar(CoreDataType.BOOL)
_INT32 = _make_scalar(CoreDataType.INT32)


def _lookup_failure(identifier: Identifier) -> tuple[Type, TypeQualifier]:
    raise KeyError(identifier.name_hint)


# =============================================================================
# get_core_data_type_from_literal_type: bool now returns BOOL
# =============================================================================


@pytest.mark.parametrize("value", [True, False])
def test_get_core_data_type_from_literal_type_returns_bool_for_bool_value(
    value: bool,
) -> None:
    """Test ``get_core_data_type_from_literal_type`` returns ``BOOL`` for bool input."""
    assert get_core_data_type_from_literal_type(value) is CoreDataType.BOOL


# =============================================================================
# Synthesis: boolean literals and logical operations
# =============================================================================


@pytest.mark.parametrize("value", [True, False])
def test_synthesize_boolean_literal_produces_bool_param(value: bool) -> None:
    """Test ``LiteralExpression(True | False)`` synthesizes ``(BOOL, PARAM)``."""
    result_type, result_qualifier = synthesize_expression_type(
        LiteralExpression(value), _lookup_failure
    )

    assert result_type.is_structurally_equivalent(_BOOL_TYPE)
    assert result_qualifier is TypeQualifier.PARAM


def test_synthesize_logical_not_on_bool_returns_bool() -> None:
    """Test ``!True`` synthesizes ``BOOL``."""
    expression = UnaryExpression(UnaryOperation.LOGICAL_NOT, LiteralExpression(True))

    result_type, _ = synthesize_expression_type(expression, _lookup_failure)

    assert result_type.is_structurally_equivalent(_BOOL_TYPE)


def test_synthesize_logical_not_on_int_raises() -> None:
    """Test ``LOGICAL_NOT`` rejects a non-boolean operand."""
    expression = UnaryExpression(UnaryOperation.LOGICAL_NOT, LiteralExpression(5))

    with pytest.raises(FhYCoreTypeError):
        synthesize_expression_type(expression, _lookup_failure)


@pytest.mark.parametrize(
    "operation", [BinaryOperation.LOGICAL_AND, BinaryOperation.LOGICAL_OR]
)
def test_synthesize_logical_binary_on_bools_returns_bool(
    operation: BinaryOperation,
) -> None:
    """Test ``&&`` / ``||`` on two booleans synthesizes ``BOOL``."""
    expression = BinaryExpression(
        operation, LiteralExpression(True), LiteralExpression(False)
    )

    result_type, result_qualifier = synthesize_expression_type(
        expression, _lookup_failure
    )

    assert result_type.is_structurally_equivalent(_BOOL_TYPE)
    assert result_qualifier is TypeQualifier.PARAM


@pytest.mark.parametrize(
    "operation", [BinaryOperation.LOGICAL_AND, BinaryOperation.LOGICAL_OR]
)
def test_synthesize_logical_binary_rejects_non_bool_operand(
    operation: BinaryOperation,
) -> None:
    """Test ``&&`` / ``||`` reject a non-boolean operand."""
    expression = BinaryExpression(
        operation, LiteralExpression(True), LiteralExpression(1)
    )

    with pytest.raises(FhYCoreTypeError):
        synthesize_expression_type(expression, _lookup_failure)


# =============================================================================
# Synthesis: comparison operations now return BOOL
# =============================================================================


@pytest.mark.parametrize(
    "operation",
    [
        BinaryOperation.EQUAL,
        BinaryOperation.NOT_EQUAL,
        BinaryOperation.LESS,
        BinaryOperation.LESS_EQUAL,
        BinaryOperation.GREATER,
        BinaryOperation.GREATER_EQUAL,
    ],
)
def test_synthesize_numeric_comparison_returns_bool(
    operation: BinaryOperation,
) -> None:
    """Test all six numeric comparison operations synthesize ``BOOL``."""
    expression = BinaryExpression(operation, LiteralExpression(1), LiteralExpression(2))

    result_type, _ = synthesize_expression_type(expression, _lookup_failure)

    assert result_type.is_structurally_equivalent(_BOOL_TYPE)


@pytest.mark.parametrize(
    "operation", [BinaryOperation.EQUAL, BinaryOperation.NOT_EQUAL]
)
def test_synthesize_equality_on_two_bools_returns_bool(
    operation: BinaryOperation,
) -> None:
    """Test ``==`` / ``!=`` on two booleans synthesizes ``BOOL``."""
    expression = BinaryExpression(
        operation, LiteralExpression(True), LiteralExpression(False)
    )

    result_type, _ = synthesize_expression_type(expression, _lookup_failure)

    assert result_type.is_structurally_equivalent(_BOOL_TYPE)


@pytest.mark.parametrize(
    "operation",
    [
        BinaryOperation.LESS,
        BinaryOperation.LESS_EQUAL,
        BinaryOperation.GREATER,
        BinaryOperation.GREATER_EQUAL,
    ],
)
def test_synthesize_ordering_on_two_bools_raises(
    operation: BinaryOperation,
) -> None:
    """Test ordering operators reject boolean operands (booleans are not ordered)."""
    expression = BinaryExpression(
        operation, LiteralExpression(True), LiteralExpression(False)
    )

    with pytest.raises(FhYCoreTypeError):
        synthesize_expression_type(expression, _lookup_failure)


@pytest.mark.parametrize(
    "operation", [BinaryOperation.EQUAL, BinaryOperation.NOT_EQUAL]
)
def test_synthesize_equality_rejects_mixed_bool_and_numeric(
    operation: BinaryOperation,
) -> None:
    """Test equality between a boolean and a numeric operand raises."""
    expression = BinaryExpression(
        operation, LiteralExpression(True), LiteralExpression(1)
    )

    with pytest.raises(FhYCoreTypeError):
        synthesize_expression_type(expression, _lookup_failure)


# =============================================================================
# Synthesis: arithmetic on booleans is rejected
# =============================================================================


@pytest.mark.parametrize(
    "operation",
    [
        BinaryOperation.ADD,
        BinaryOperation.SUBTRACT,
        BinaryOperation.MULTIPLY,
        BinaryOperation.DIVIDE,
        BinaryOperation.MODULO,
    ],
)
def test_synthesize_arithmetic_rejects_boolean_operand(
    operation: BinaryOperation,
) -> None:
    """Test arithmetic operations reject a boolean operand."""
    expression = BinaryExpression(
        operation, LiteralExpression(True), LiteralExpression(1)
    )

    with pytest.raises(FhYCoreTypeError):
        synthesize_expression_type(expression, _lookup_failure)


def test_synthesize_unary_negation_rejects_boolean_operand() -> None:
    """Test unary negation of a boolean literal is rejected."""
    expression = UnaryExpression(UnaryOperation.NEGATE, LiteralExpression(True))

    with pytest.raises(FhYCoreTypeError):
        synthesize_expression_type(expression, _lookup_failure)


# =============================================================================
# Synthesis: TernaryExpression
# =============================================================================


def test_synthesize_ternary_with_matching_branch_types_yields_branch_type() -> None:
    """Test ``cond ? x : y`` synthesizes the branches' common type.

    Two weak unsigned-int literal branches stay weak ``UINT`` without an
    outer context, matching the existing arithmetic-literal behavior.
    """
    expression = TernaryExpression(
        LiteralExpression(True), LiteralExpression(10), LiteralExpression(20)
    )

    result_type, _ = synthesize_expression_type(expression, _lookup_failure)

    assert result_type.is_structurally_equivalent(_make_scalar(CoreDataType.UINT))


def test_synthesize_ternary_promotes_branch_types_to_common() -> None:
    """Test ``cond ? int8 : int32`` synthesizes the promoted (wider) type."""
    a = Identifier("a")
    b = Identifier("b")
    checker = make_identifier_checker(
        {
            a: (_make_scalar(CoreDataType.INT8), TypeQualifier.PARAM),
            b: (_make_scalar(CoreDataType.INT32), TypeQualifier.PARAM),
        }
    )
    expression = TernaryExpression(
        LiteralExpression(True),
        IdentifierExpression(a),
        IdentifierExpression(b),
    )

    result_type, _ = checker.synthesize(expression)

    assert result_type.is_structurally_equivalent(_make_scalar(CoreDataType.INT32))


def test_synthesize_ternary_with_non_boolean_condition_raises() -> None:
    """Test the ternary form rejects a non-boolean condition."""
    expression = TernaryExpression(
        LiteralExpression(1), LiteralExpression(10), LiteralExpression(20)
    )

    with pytest.raises(FhYCoreTypeError):
        synthesize_expression_type(expression, _lookup_failure)


def test_synthesize_ternary_with_incompatible_branches_raises() -> None:
    """Test the ternary form rejects branches whose types do not promote."""
    a = Identifier("a")
    b = Identifier("b")
    checker = make_identifier_checker(
        {
            a: (_make_scalar(CoreDataType.INT32), TypeQualifier.PARAM),
            b: (_make_scalar(CoreDataType.FLOAT32), TypeQualifier.PARAM),
        }
    )
    expression = TernaryExpression(
        LiteralExpression(True),
        IdentifierExpression(a),
        IdentifierExpression(b),
    )

    with pytest.raises(FhYCoreTypeError):
        checker.synthesize(expression)


def test_check_ternary_propagates_expected_type_into_literal_branches() -> None:
    """Test the expected type propagates into both literal branches."""
    checker = make_single_type_checker(_INT32)
    expression = TernaryExpression(
        LiteralExpression(True), LiteralExpression(5), LiteralExpression(10)
    )

    result_type, _ = checker.check(expression, _INT32)

    assert result_type.is_structurally_equivalent(_INT32)


def test_synthesize_ternary_qualifier_promotes_across_three_operands() -> None:
    """Test the ternary qualifier is the promotion across all three operands."""
    a = Identifier("a")
    b = Identifier("b")
    cond = Identifier("c")
    checker = make_identifier_checker(
        {
            cond: (_BOOL_TYPE, TypeQualifier.PARAM),
            a: (_make_scalar(CoreDataType.INT32), TypeQualifier.INPUT),
            b: (_make_scalar(CoreDataType.INT32), TypeQualifier.PARAM),
        }
    )
    expression = TernaryExpression(
        IdentifierExpression(cond),
        IdentifierExpression(a),
        IdentifierExpression(b),
    )

    _, qualifier = checker.synthesize(expression)

    # PARAM only when all three operands are PARAM; INPUT mixed in -> TEMP.
    assert qualifier is TypeQualifier.TEMP


# =============================================================================
# Synthesis: CallExpression
# =============================================================================


def test_synthesize_call_to_unregistered_function_raises() -> None:
    """Test a call to an unregistered function raises during synthesis."""
    expression = CallExpression("never_registered", (LiteralExpression(1),))

    with pytest.raises(FhYCoreTypeError, match="never_registered"):
        synthesize_expression_type(expression, _lookup_failure)


def test_synthesize_call_with_wrong_arity_raises(
    function_registry_snapshot: None,
) -> None:
    """Test a call with an argument count not matching the parameters raises."""
    parameter = Identifier("x")
    register_function(
        "test_synthesize_arity",
        parameters=[parameter],
        parameter_sorts=[FunctionSort.REAL],
        result_sort=FunctionSort.REAL,
        body=IdentifierExpression(parameter),
    )

    expression = CallExpression(
        "test_synthesize_arity",
        (LiteralExpression(1), LiteralExpression(2)),
    )

    with pytest.raises(FhYCoreTypeError, match="test_synthesize_arity"):
        synthesize_expression_type(expression, _lookup_failure)


def test_synthesize_call_returns_sort_derived_type(
    function_registry_snapshot: None,
) -> None:
    """Test ``call(name, args)`` synthesizes the declared result sort's core type.

    The call-site result type is driven by the function's declared
    ``result_sort`` (concrete ``FLOAT64`` for ``REAL``), not by walking
    the body and propagating the argument types.
    """
    parameter = Identifier("x")
    register_function(
        "test_synthesize_identity",
        parameters=[parameter],
        parameter_sorts=[FunctionSort.REAL],
        result_sort=FunctionSort.REAL,
        body=IdentifierExpression(parameter),
    )

    expression = CallExpression(
        "test_synthesize_identity",
        (LiteralExpression(5),),
    )

    result_type, _ = synthesize_expression_type(expression, _lookup_failure)

    assert result_type.is_structurally_equivalent(_make_scalar(CoreDataType.FLOAT64))


def test_synthesize_call_to_max_returns_sort_derived_float64(
    function_registry_snapshot: None,
) -> None:
    """Test ``max(a, b)`` returns the declared ``REAL`` sort's concrete ``FLOAT64``.

    ``max`` is registered as ``REAL`` -> ``REAL``; the call-site result is
    the sort-derived concrete type, not a promotion of the argument
    types.
    """
    a = Identifier("a")
    b = Identifier("b")
    checker = make_identifier_checker(
        {
            a: (_make_scalar(CoreDataType.INT8), TypeQualifier.PARAM),
            b: (_make_scalar(CoreDataType.INT32), TypeQualifier.PARAM),
        }
    )
    expression = CallExpression(
        "max", (IdentifierExpression(a), IdentifierExpression(b))
    )

    result_type, _ = checker.synthesize(expression)

    assert result_type.is_structurally_equivalent(_make_scalar(CoreDataType.FLOAT64))
