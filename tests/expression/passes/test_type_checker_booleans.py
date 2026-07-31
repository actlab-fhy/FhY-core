"""Type-checker tests for booleans, ``PiecewiseExpression``, and ``CallExpression``.

Lives alongside ``test_type_checker.py``; this file holds new cases for
the boolean type, the piecewise conditional node, and the function-call
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
    PiecewiseExpression,
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
    IndexType,
    NumericalType,
    PrimitiveDataType,
    Type,
    TypeQualifier,
)

from ..conftest import (
    make_identifier_checker,
    make_single_type_checker,
    mock_identifier,
)


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
# Synthesis: PiecewiseExpression
# =============================================================================


def test_synthesize_single_case_piecewise_yields_value_and_otherwise_common_type() -> (
    None
):
    """Test a single-case piecewise synthesizes the value/otherwise common type.

    Two weak unsigned-int literal operands stay weak ``UINT`` without an
    outer context, matching the existing arithmetic-literal behavior.
    """
    expression = PiecewiseExpression(
        (LiteralExpression(True),), (LiteralExpression(10),), LiteralExpression(20)
    )

    result_type, _ = synthesize_expression_type(expression, _lookup_failure)

    assert result_type.is_structurally_equivalent(_make_scalar(CoreDataType.UINT))


def test_synthesize_single_case_piecewise_promotes_value_and_otherwise_to_common() -> (
    None
):
    """Test ``{int8 if cond; int32 otherwise}`` synthesizes the wider promoted type."""
    a = mock_identifier("a", 0)
    b = mock_identifier("b", 1)
    checker = make_identifier_checker(
        {
            a: (_make_scalar(CoreDataType.INT8), TypeQualifier.PARAM),
            b: (_make_scalar(CoreDataType.INT32), TypeQualifier.PARAM),
        }
    )
    expression = PiecewiseExpression(
        (LiteralExpression(True),), (IdentifierExpression(a),), IdentifierExpression(b)
    )

    result_type, _ = checker.synthesize(expression)

    assert result_type.is_structurally_equivalent(_make_scalar(CoreDataType.INT32))


def test_synthesize_piecewise_promotes_across_multiple_case_values() -> None:
    """Test the result type folds ``promote_primitive_data_types`` across every case.

    Three case values (``int8``, ``int16``, ``int32``) plus an ``int64``
    otherwise all fold left-to-right into the widest promoted type.
    """
    a = mock_identifier("a", 0)
    b = mock_identifier("b", 1)
    c = mock_identifier("c", 2)
    d = mock_identifier("d", 3)
    checker = make_identifier_checker(
        {
            a: (_make_scalar(CoreDataType.INT8), TypeQualifier.PARAM),
            b: (_make_scalar(CoreDataType.INT16), TypeQualifier.PARAM),
            c: (_make_scalar(CoreDataType.INT32), TypeQualifier.PARAM),
            d: (_make_scalar(CoreDataType.INT64), TypeQualifier.PARAM),
        }
    )
    expression = PiecewiseExpression(
        (LiteralExpression(True), LiteralExpression(False), LiteralExpression(True)),
        (IdentifierExpression(a), IdentifierExpression(b), IdentifierExpression(c)),
        IdentifierExpression(d),
    )

    result_type, _ = checker.synthesize(expression)

    assert result_type.is_structurally_equivalent(_make_scalar(CoreDataType.INT64))


def test_synthesize_piecewise_with_non_boolean_condition_raises() -> None:
    """Test a non-boolean condition is rejected with an indexed error message.

    The condition is an identifier bound to a non-boolean numerical type
    (rather than a bare non-boolean literal) so the boolean-condition
    check itself is what raises, with the case index in the message.
    """
    a = mock_identifier("a", 0)
    checker = make_identifier_checker(
        {a: (_make_scalar(CoreDataType.INT32), TypeQualifier.PARAM)}
    )
    expression = PiecewiseExpression(
        (IdentifierExpression(a),), (LiteralExpression(10),), LiteralExpression(20)
    )

    with pytest.raises(FhYCoreTypeError, match=r"case 0"):
        checker.synthesize(expression)


def test_synthesize_piecewise_with_non_boolean_literal_condition_raises() -> None:
    """Test a bare non-boolean literal condition is rejected.

    Unlike the identifier-bound case above, this condition is a plain
    non-boolean ``LiteralExpression``, so the failure comes from the
    literal-vs-expected-type check (a non-boolean literal against the
    boolean context) rather than the piecewise-specific indexed
    boolean-condition check.
    """
    expression = PiecewiseExpression(
        (LiteralExpression(1),), (LiteralExpression(10),), LiteralExpression(20)
    )

    with pytest.raises(FhYCoreTypeError):
        synthesize_expression_type(expression, _lookup_failure)


def test_synthesize_piecewise_reports_index_of_failing_condition() -> None:
    """Test the second case's non-boolean condition is named by its zero-based index."""
    cond0 = mock_identifier("cond0", 0)
    cond1 = mock_identifier("cond1", 1)
    checker = make_identifier_checker(
        {
            cond0: (_BOOL_TYPE, TypeQualifier.PARAM),
            cond1: (_make_scalar(CoreDataType.INT32), TypeQualifier.PARAM),
        }
    )
    expression = PiecewiseExpression(
        (IdentifierExpression(cond0), IdentifierExpression(cond1)),
        (LiteralExpression(10), LiteralExpression(20)),
        LiteralExpression(30),
    )

    with pytest.raises(FhYCoreTypeError, match=r"case 1"):
        checker.synthesize(expression)


def test_synthesize_piecewise_with_incompatible_value_types_raises() -> None:
    """Test the piecewise form rejects branches whose types do not promote."""
    a = mock_identifier("a", 0)
    b = mock_identifier("b", 1)
    checker = make_identifier_checker(
        {
            a: (_make_scalar(CoreDataType.INT32), TypeQualifier.PARAM),
            b: (_make_scalar(CoreDataType.FLOAT32), TypeQualifier.PARAM),
        }
    )
    expression = PiecewiseExpression(
        (LiteralExpression(True),), (IdentifierExpression(a),), IdentifierExpression(b)
    )

    with pytest.raises(FhYCoreTypeError):
        checker.synthesize(expression)


def test_synthesize_piecewise_rejects_index_typed_case_value() -> None:
    """Test an ``IndexType`` case value is rejected (only scalar numerics allowed)."""
    index_id = mock_identifier("idx", 0)
    otherwise_id = mock_identifier("o", 1)
    index_type = IndexType(
        LiteralExpression(0), LiteralExpression(10), LiteralExpression(1)
    )
    checker = make_identifier_checker(
        {
            index_id: (index_type, TypeQualifier.PARAM),
            otherwise_id: (_make_scalar(CoreDataType.INT32), TypeQualifier.PARAM),
        }
    )
    expression = PiecewiseExpression(
        (LiteralExpression(True),),
        (IdentifierExpression(index_id),),
        IdentifierExpression(otherwise_id),
    )

    with pytest.raises(FhYCoreTypeError, match=r"case 0"):
        checker.synthesize(expression)


def test_synthesize_piecewise_rejects_index_typed_otherwise() -> None:
    """Test an ``IndexType`` otherwise is rejected, named ``"otherwise"``."""
    value_id = mock_identifier("v", 0)
    index_id = mock_identifier("idx", 1)
    index_type = IndexType(
        LiteralExpression(0), LiteralExpression(10), LiteralExpression(1)
    )
    checker = make_identifier_checker(
        {
            value_id: (_make_scalar(CoreDataType.INT32), TypeQualifier.PARAM),
            index_id: (index_type, TypeQualifier.PARAM),
        }
    )
    expression = PiecewiseExpression(
        (LiteralExpression(True),),
        (IdentifierExpression(value_id),),
        IdentifierExpression(index_id),
    )

    with pytest.raises(FhYCoreTypeError, match=r"otherwise"):
        checker.synthesize(expression)


def test_check_piecewise_propagates_expected_type_into_all_branches() -> None:
    """Test the expected type propagates into every case value and ``otherwise``."""
    checker = make_single_type_checker(_INT32)
    expression = PiecewiseExpression(
        (LiteralExpression(True), LiteralExpression(False)),
        (LiteralExpression(5), LiteralExpression(6)),
        LiteralExpression(10),
    )

    result_type, _ = checker.check(expression, _INT32)

    assert result_type.is_structurally_equivalent(_INT32)


def test_synthesize_piecewise_qualifier_promotes_across_all_operands() -> None:
    """Test the qualifier promotes across every condition, value, and otherwise."""
    a = mock_identifier("a", 0)
    b = mock_identifier("b", 1)
    cond = mock_identifier("c", 2)
    checker = make_identifier_checker(
        {
            cond: (_BOOL_TYPE, TypeQualifier.PARAM),
            a: (_make_scalar(CoreDataType.INT32), TypeQualifier.INPUT),
            b: (_make_scalar(CoreDataType.INT32), TypeQualifier.PARAM),
        }
    )
    expression = PiecewiseExpression(
        (IdentifierExpression(cond),),
        (IdentifierExpression(a),),
        IdentifierExpression(b),
    )

    _, qualifier = checker.synthesize(expression)

    # PARAM only when every operand is PARAM; INPUT mixed in -> TEMP.
    assert qualifier is TypeQualifier.TEMP


def test_synthesize_piecewise_qualifier_promotes_across_multiple_cases() -> None:
    """Test the qualifier fold considers every case, not just the first one."""
    cond1 = mock_identifier("c1", 0)
    cond2 = mock_identifier("c2", 1)
    value1 = mock_identifier("v1", 2)
    value2 = mock_identifier("v2", 3)
    otherwise_id = mock_identifier("o", 4)
    checker = make_identifier_checker(
        {
            cond1: (_BOOL_TYPE, TypeQualifier.PARAM),
            cond2: (_BOOL_TYPE, TypeQualifier.PARAM),
            value1: (_make_scalar(CoreDataType.INT32), TypeQualifier.PARAM),
            value2: (_make_scalar(CoreDataType.INT32), TypeQualifier.INPUT),
            otherwise_id: (_make_scalar(CoreDataType.INT32), TypeQualifier.PARAM),
        }
    )
    expression = PiecewiseExpression(
        (IdentifierExpression(cond1), IdentifierExpression(cond2)),
        (IdentifierExpression(value1), IdentifierExpression(value2)),
        IdentifierExpression(otherwise_id),
    )

    _, qualifier = checker.synthesize(expression)

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
