"""Tests for `fhy_core.expression.core`."""

import dataclasses
import operator
from collections.abc import Callable
from typing import Any

import pytest

from fhy_core.expression import (
    BinaryExpression,
    BinaryOperation,
    Expression,
    IdentifierExpression,
    LiteralExpression,
    UnaryExpression,
    UnaryOperation,
    collect_identifiers,
    logical_and,
    logical_or,
    make_binary_expression,
    make_unary_expression,
)
from fhy_core.identifier import Identifier
from fhy_core.serialization import (
    DeserializationDictStructureError,
    DeserializationValueError,
    SerializedDict,
)
from fhy_core.trait import FrozenMutationError, HasOperands, StructuralEquivalence

from .conftest import mock_identifier

# =============================================================================
# Construction & accessors
# =============================================================================


def test_unary_expression_stores_operation_and_operand() -> None:
    """Test `UnaryExpression` exposes the fields it was built with."""
    operand = LiteralExpression(5)
    expression = UnaryExpression(UnaryOperation.NEGATE, operand)
    assert expression.operation == UnaryOperation.NEGATE
    assert expression.operand is operand


def test_binary_expression_stores_operation_left_and_right() -> None:
    """Test `BinaryExpression` exposes its three constructor fields."""
    left = LiteralExpression(5)
    right = LiteralExpression(10)
    expression = BinaryExpression(BinaryOperation.ADD, left, right)
    assert expression.operation == BinaryOperation.ADD
    assert expression.left is left
    assert expression.right is right


def test_identifier_expression_stores_identifier() -> None:
    """Test `IdentifierExpression` exposes the `Identifier` it was built with."""
    identifier = Identifier("x")
    expression = IdentifierExpression(identifier)
    assert expression.identifier is identifier


@pytest.mark.parametrize("value", [0, 5, -3, 1_000_000])
def test_literal_expression_stores_native_int_as_int(value: int) -> None:
    """Test a native ``int`` is stored unchanged with type ``int``."""
    literal = LiteralExpression(value)

    assert literal.value == value
    assert type(literal.value) is int


@pytest.mark.parametrize("value", [0.0, 3.14, -2.5, 1e-10])
def test_literal_expression_stores_native_float_as_float(value: float) -> None:
    """Test a native ``float`` is stored unchanged with type ``float``."""
    literal = LiteralExpression(value)

    assert literal.value == value
    assert type(literal.value) is float


@pytest.mark.parametrize("value", [True, False])
def test_literal_expression_stores_native_bool_as_bool(value: bool) -> None:
    """Test a native ``bool`` is stored unchanged with type ``bool`` (not ``int``)."""
    literal = LiteralExpression(value)

    assert literal.value is value
    assert type(literal.value) is bool


@pytest.mark.parametrize(
    "string_value, expected_int",
    [("0", 0), ("5", 5), ("42", 42), ("00", 0), ("01", 1)],
)
def test_literal_expression_canonicalizes_integer_shaped_string_to_int(
    string_value: str, expected_int: int
) -> None:
    """Test an integer-shaped ``str`` value is canonicalized to ``int``.

    The constructor accepts ``"5"`` and the resulting expression is
    indistinguishable from ``LiteralExpression(5)``.
    """
    literal = LiteralExpression(string_value)

    assert literal.value == expected_int
    assert type(literal.value) is int


@pytest.mark.parametrize("string_value", ["3.14", "0.0", "1.", ".5", "0.1", "100.001"])
def test_literal_expression_keeps_float_shaped_string_as_str(string_value: str) -> None:
    """Test a float-shaped ``str`` value is kept as ``str`` to preserve exact decimal.

    Native ``float`` would be lossy; the design preserves the textual form so
    e.g. ``LiteralExpression("0.1")`` does not become the IEEE-754 approximation.
    """
    literal = LiteralExpression(string_value)

    assert literal.value == string_value
    assert type(literal.value) is str


@pytest.mark.parametrize(
    "string_value",
    [
        "not_a_number",
        "inf",
        "-inf",
        "Infinity",
        "NaN",
        "1e10",
        "0x1f",
        "-5",
        "+5",
        "5.5e2",
        "",
        "  ",
        "5 ",
    ],
)
def test_literal_expression_rejects_string_outside_parser_grammar(
    string_value: str,
) -> None:
    """Test ``str`` values not matching the parser's int or float grammar raise.

    The validator pins to the parser's numeric grammar so any string that the
    parser couldn't have produced is rejected at construction time.
    """
    with pytest.raises(ValueError, match="(?i)literal"):
        LiteralExpression(string_value)


@pytest.mark.parametrize(
    "value",
    [
        pytest.param(5 + 6j, id="complex"),
        pytest.param(b"bytes", id="bytes"),
        pytest.param(None, id="none"),
        pytest.param([1, 2, 3], id="list"),
        pytest.param((1, 2), id="tuple"),
        pytest.param({"k": 1}, id="dict"),
        pytest.param(object(), id="arbitrary_object"),
    ],
)
def test_literal_expression_rejects_unsupported_python_types(value: object) -> None:
    """Test values whose type is not in ``LiteralType`` raise ``TypeError``.

    The runtime contract matches ``LiteralType``; values outside that union
    are rejected at construction time rather than slipping through to a
    downstream pass.
    """
    with pytest.raises(TypeError, match="(?i)literal"):
        LiteralExpression(value)  # type: ignore[arg-type]


# =============================================================================
# Operand & equivalence protocols
# =============================================================================


def test_unary_expression_satisfies_has_operands_protocol() -> None:
    """Test `UnaryExpression` satisfies the `HasOperands` runtime protocol."""
    expression = UnaryExpression(UnaryOperation.NEGATE, LiteralExpression(1))
    assert isinstance(expression, HasOperands)


def test_unary_expression_operands_tuple_matches_operand_field() -> None:
    """Test `UnaryExpression.get_operands()` returns `(operand,)` in order."""
    operand = LiteralExpression(1)
    expression = UnaryExpression(UnaryOperation.NEGATE, operand)
    assert expression.get_operands() == (operand,)


def test_binary_expression_satisfies_has_operands_protocol() -> None:
    """Test `BinaryExpression` satisfies the `HasOperands` runtime protocol."""
    expression = BinaryExpression(
        BinaryOperation.ADD, LiteralExpression(1), LiteralExpression(2)
    )
    assert isinstance(expression, HasOperands)


def test_binary_expression_operands_tuple_matches_left_then_right() -> None:
    """Test `BinaryExpression.get_operands()` returns `(left, right)` in order."""
    left = LiteralExpression(1)
    right = LiteralExpression(2)
    expression = BinaryExpression(BinaryOperation.ADD, left, right)
    assert expression.get_operands() == (left, right)


def test_expression_satisfies_structural_equivalence_protocol() -> None:
    """Test an `Expression` satisfies the `StructuralEquivalence` runtime protocol."""
    assert isinstance(LiteralExpression(7), StructuralEquivalence)


# =============================================================================
# Structural equivalence
# =============================================================================


@pytest.mark.parametrize(
    "expression",
    [
        pytest.param(LiteralExpression(7), id="literal"),
        pytest.param(IdentifierExpression(mock_identifier("x", 0)), id="identifier"),
        pytest.param(
            UnaryExpression(UnaryOperation.NEGATE, LiteralExpression(1)),
            id="unary",
        ),
        pytest.param(
            BinaryExpression(
                BinaryOperation.ADD, LiteralExpression(1), LiteralExpression(2)
            ),
            id="binary",
        ),
    ],
)
def test_structural_equivalence_is_reflexive(expression: Expression) -> None:
    """Test every expression is structurally equivalent to itself."""
    assert expression.is_structurally_equivalent(expression)


def test_structurally_equivalent_trees_compare_equivalent() -> None:
    """Test two independently-built identical trees are structurally equivalent."""
    left = BinaryExpression(
        BinaryOperation.ADD, LiteralExpression(1), LiteralExpression(2)
    )
    right = BinaryExpression(
        BinaryOperation.ADD, LiteralExpression(1), LiteralExpression(2)
    )
    assert left.is_structurally_equivalent(right)
    assert right.is_structurally_equivalent(left)


def test_structurally_different_trees_compare_non_equivalent() -> None:
    """Test trees differing only by operation are not structurally equivalent."""
    left = BinaryExpression(
        BinaryOperation.ADD, LiteralExpression(1), LiteralExpression(2)
    )
    right = BinaryExpression(
        BinaryOperation.SUBTRACT, LiteralExpression(1), LiteralExpression(2)
    )
    assert not left.is_structurally_equivalent(right)


def test_literal_equivalence_is_false_when_values_differ_in_either_direction() -> None:
    """Test `LiteralExpression` equivalence is value-equal and symmetrically false."""
    smaller = LiteralExpression(5)
    larger = LiteralExpression(10)
    assert not smaller.is_structurally_equivalent(larger)
    assert not larger.is_structurally_equivalent(smaller)


def test_unary_equivalence_requires_matching_operation() -> None:
    """Test `UnaryExpression` equivalence is false when operations differ."""
    operand = LiteralExpression(1)
    negate = UnaryExpression(UnaryOperation.NEGATE, operand)
    positive = UnaryExpression(UnaryOperation.POSITIVE, operand)
    assert not negate.is_structurally_equivalent(positive)


def test_unary_equivalence_requires_matching_operand() -> None:
    """Test `UnaryExpression` equivalence is false when operands differ."""
    left = UnaryExpression(UnaryOperation.NEGATE, LiteralExpression(1))
    right = UnaryExpression(UnaryOperation.NEGATE, LiteralExpression(2))
    assert not left.is_structurally_equivalent(right)


def test_binary_equivalence_requires_matching_operation() -> None:
    """Test `BinaryExpression` equivalence is false when operations differ."""
    left = BinaryExpression(
        BinaryOperation.ADD, LiteralExpression(1), LiteralExpression(2)
    )
    right = BinaryExpression(
        BinaryOperation.MULTIPLY, LiteralExpression(1), LiteralExpression(2)
    )
    assert not left.is_structurally_equivalent(right)


def test_binary_equivalence_requires_matching_left() -> None:
    """Test `BinaryExpression` equivalence is false when only left operands differ."""
    left = BinaryExpression(
        BinaryOperation.ADD, LiteralExpression(1), LiteralExpression(3)
    )
    right = BinaryExpression(
        BinaryOperation.ADD, LiteralExpression(2), LiteralExpression(3)
    )
    assert not left.is_structurally_equivalent(right)


def test_binary_equivalence_requires_matching_right() -> None:
    """Test `BinaryExpression` equivalence is false when only right operands differ."""
    left = BinaryExpression(
        BinaryOperation.ADD, LiteralExpression(1), LiteralExpression(2)
    )
    right = BinaryExpression(
        BinaryOperation.ADD, LiteralExpression(1), LiteralExpression(3)
    )
    assert not left.is_structurally_equivalent(right)


def test_identifier_equivalence_requires_matching_identifier() -> None:
    """Test `IdentifierExpression` equivalence compares underlying identifiers."""
    left = IdentifierExpression(Identifier("x"))
    right = IdentifierExpression(Identifier("x"))
    assert not left.is_structurally_equivalent(right)


def test_binary_equivalence_is_not_commutative() -> None:
    """Test ``a + b`` and ``b + a`` are not structurally equivalent.

    Structural equivalence is positional, not algebraic; commuting operands
    must produce a non-equivalent tree so future "smart" equivalence changes
    are caught.
    """
    a = LiteralExpression(1)
    b = LiteralExpression(2)
    assert not BinaryExpression(BinaryOperation.ADD, a, b).is_structurally_equivalent(
        BinaryExpression(BinaryOperation.ADD, b, a)
    )


@pytest.mark.parametrize(
    "left_value, right_value",
    [
        pytest.param(1.0, "1.0", id="float_vs_str_form_float"),
        pytest.param(1.5, "1.5", id="non_integer_float_vs_str_form_float"),
        pytest.param(0.5, "0.5", id="half_native_vs_str"),
    ],
)
def test_literal_equivalence_is_false_for_native_float_vs_str_form_float(
    left_value: float, right_value: str
) -> None:
    """Test native ``float`` literal and equivalent str-form-float are not equivalent.

    The design treats native ``float`` as a (possibly imprecise) IEEE-754 value
    and str-form floats as exact decimals; they are intentionally distinct.
    """
    assert not LiteralExpression(left_value).is_structurally_equivalent(
        LiteralExpression(right_value)
    )


@pytest.mark.parametrize(
    "left_value, right_value",
    [
        pytest.param(0, False, id="int_zero_vs_bool_false"),
        pytest.param(1, True, id="int_one_vs_bool_true"),
        pytest.param(0, 0.0, id="int_zero_vs_float_zero"),
        pytest.param(1, 1.0, id="int_one_vs_float_one"),
        pytest.param(False, 0.0, id="bool_false_vs_float_zero"),
        pytest.param(True, 1.0, id="bool_true_vs_float_one"),
    ],
)
def test_literal_equivalence_is_false_when_value_types_differ(
    left_value: bool | int | float, right_value: bool | int | float
) -> None:
    """Test `LiteralExpression` equivalence is false across distinct numeric types.

    Python evaluates ``0 == False``, ``1 == True`` and ``0 == 0.0`` as
    ``True``, but the codebase distinguishes ``bool``, ``int`` and ``float``
    everywhere else (``is_strict_int``, ``_has_bool_numeric_mismatch``,
    the ``type(value) is bool`` ordering in ``__post_init__``, the
    ``bool``/``int``/``float`` discrimination in
    ``get_core_data_type_from_literal_type``). Structural equivalence
    must agree.
    """
    left = LiteralExpression(left_value)
    right = LiteralExpression(right_value)
    assert not left.is_structurally_equivalent(right)
    assert not right.is_structurally_equivalent(left)


@pytest.mark.parametrize(
    "int_value, equivalent_string",
    [(0, "0"), (1, "1"), (42, "42"), (0, "00")],
)
def test_int_literal_and_integer_shaped_string_compare_structurally_equivalent(
    int_value: int, equivalent_string: str
) -> None:
    """Test int and integer-shaped-string literals compare structurally equivalent.

    Integer-shaped strings canonicalize to ``int`` at construction time, so
    the two forms produce structurally-equivalent trees.
    """
    left = LiteralExpression(int_value)
    right = LiteralExpression(equivalent_string)

    assert left.is_structurally_equivalent(right)
    assert right.is_structurally_equivalent(left)


def _make_pair_by_subclass(
    subclass: type[Expression],
) -> Expression:
    if subclass is LiteralExpression:
        return LiteralExpression(1)
    elif subclass is IdentifierExpression:
        return IdentifierExpression(mock_identifier("x", 0))
    elif subclass is UnaryExpression:
        return UnaryExpression(UnaryOperation.NEGATE, LiteralExpression(1))
    elif subclass is BinaryExpression:
        return BinaryExpression(
            BinaryOperation.ADD, LiteralExpression(1), LiteralExpression(2)
        )
    else:
        raise AssertionError(f"Unknown subclass: {subclass}")


@pytest.mark.parametrize(
    "left_subclass, right_subclass",
    [
        (LiteralExpression, IdentifierExpression),
        (LiteralExpression, UnaryExpression),
        (LiteralExpression, BinaryExpression),
        (IdentifierExpression, UnaryExpression),
        (IdentifierExpression, BinaryExpression),
        (UnaryExpression, BinaryExpression),
    ],
)
def test_equivalence_is_false_across_distinct_expression_subclasses(
    left_subclass: type[Expression], right_subclass: type[Expression]
) -> None:
    """Test structural equivalence is false across distinct `Expression` subclasses."""
    left = _make_pair_by_subclass(left_subclass)
    right = _make_pair_by_subclass(right_subclass)
    assert not left.is_structurally_equivalent(right)
    assert not right.is_structurally_equivalent(left)


# =============================================================================
# Unary operator dunders
# =============================================================================


@pytest.mark.parametrize(
    "unary_operator, expected_operation",
    [
        (operator.neg, UnaryOperation.NEGATE),
        (operator.pos, UnaryOperation.POSITIVE),
        (lambda x: x.logical_not(), UnaryOperation.LOGICAL_NOT),
    ],
)
def test_unary_operator_dunders_produce_matching_expression(
    unary_operator: Callable[[Expression], Expression],
    expected_operation: UnaryOperation,
) -> None:
    """Test each unary dunder lowers to a `UnaryExpression` with the matching op."""
    operand = LiteralExpression(5)
    expected = UnaryExpression(expected_operation, operand)
    assert unary_operator(operand).is_structurally_equivalent(expected)


# =============================================================================
# Binary operator dunders
# =============================================================================

_BINARY_OPERATOR_PAIRS = [
    (operator.add, BinaryOperation.ADD),
    (operator.sub, BinaryOperation.SUBTRACT),
    (operator.mul, BinaryOperation.MULTIPLY),
    (operator.truediv, BinaryOperation.DIVIDE),
    (operator.floordiv, BinaryOperation.FLOOR_DIVIDE),
    (operator.mod, BinaryOperation.MODULO),
    (operator.pow, BinaryOperation.POWER),
    (lambda x, y: x.equals(y), BinaryOperation.EQUAL),
    (lambda x, y: x.not_equals(y), BinaryOperation.NOT_EQUAL),
    (operator.lt, BinaryOperation.LESS),
    (operator.le, BinaryOperation.LESS_EQUAL),
    (operator.gt, BinaryOperation.GREATER),
    (operator.ge, BinaryOperation.GREATER_EQUAL),
]


@pytest.mark.parametrize("binary_operator, expected_operation", _BINARY_OPERATOR_PAIRS)
def test_binary_operator_dunders_produce_matching_expression(
    binary_operator: Callable[[Expression, Expression], Expression],
    expected_operation: BinaryOperation,
) -> None:
    """Test each binary dunder lowers to a `BinaryExpression` with the matching op."""
    left = LiteralExpression(5)
    right = LiteralExpression(10)
    expected = BinaryExpression(expected_operation, left, right)
    assert binary_operator(left, right).is_structurally_equivalent(expected)


_NON_EXPRESSION_RIGHT_OPERANDS: tuple[tuple[Any, type[Expression]], ...] = (
    (10, LiteralExpression),
    (10.5, LiteralExpression),
    (False, LiteralExpression),
    (mock_identifier("y", 42), IdentifierExpression),
)


@pytest.mark.parametrize("binary_operator, expected_operation", _BINARY_OPERATOR_PAIRS)
@pytest.mark.parametrize("right, expected_right_type", _NON_EXPRESSION_RIGHT_OPERANDS)
def test_binary_dunder_promotes_right_python_operand_to_expression(
    binary_operator: Callable[[Any, Any], Expression],
    expected_operation: BinaryOperation,
    right: Any,
    expected_right_type: type[Expression],
) -> None:
    """Test binary dunders wrap a non-`Expression` right operand."""
    left = LiteralExpression(5)
    expected = BinaryExpression(
        expected_operation,
        left,
        expected_right_type(right),  # type: ignore[call-arg]
    )
    assert binary_operator(left, right).is_structurally_equivalent(expected)


_NON_EXPRESSION_LEFT_OPERANDS: tuple[tuple[Any, type[Expression]], ...] = (
    (6, LiteralExpression),
    (10.3, LiteralExpression),
    (True, LiteralExpression),
    (mock_identifier("x", 1), IdentifierExpression),
)


@pytest.mark.parametrize(
    "binary_operator, expected_operation",
    [
        (operator.add, BinaryOperation.ADD),
        (operator.sub, BinaryOperation.SUBTRACT),
        (operator.mul, BinaryOperation.MULTIPLY),
        (operator.truediv, BinaryOperation.DIVIDE),
        (operator.floordiv, BinaryOperation.FLOOR_DIVIDE),
        (operator.mod, BinaryOperation.MODULO),
        (operator.pow, BinaryOperation.POWER),
    ],
)
@pytest.mark.parametrize("left, expected_left_type", _NON_EXPRESSION_LEFT_OPERANDS)
def test_binary_dunder_promotes_left_python_operand_to_expression(
    binary_operator: Callable[[Any, Any], Expression],
    expected_operation: BinaryOperation,
    left: Any,
    expected_left_type: type[Expression],
) -> None:
    """Test reflected binary dunders wrap a non-`Expression` left operand."""
    right = LiteralExpression(5)
    expected = BinaryExpression(
        expected_operation,
        expected_left_type(left),  # type: ignore[call-arg]
        right,
    )
    assert binary_operator(left, right).is_structurally_equivalent(expected)


@pytest.mark.parametrize("value", [True, False])
def test_binary_dunder_preserves_bool_type_when_wrapping(value: bool) -> None:
    """Test wrapping a Python ``bool`` keeps it as a ``bool``, not coerced to ``int``.

    ``bool`` is a subtype of ``int`` and ``LiteralType`` admits both, so a naive
    isinstance check could misclassify. This test pins down that the wrapped
    value is the ``bool`` singleton.
    """
    expression = LiteralExpression(0) + value
    assert isinstance(expression, BinaryExpression)
    assert isinstance(expression.right, LiteralExpression)
    assert type(expression.right.value) is bool
    assert expression.right.value is value


def test_binary_dunder_rejects_unsupported_type_on_right() -> None:
    """Test a binary dunder rejects a right operand of unsupported type."""
    with pytest.raises(
        ValueError,
        match=r"Unable to cast \[\] with type <class 'list'> to an expression.",
    ):
        LiteralExpression(5) + []


def test_binary_dunder_rejects_str_operand_on_right() -> None:
    """Test a binary dunder rejects a ``str`` right operand.

    Implicit ``str`` coercion was removed because ``LiteralExpression``
    happily round-trips ``"5"`` into the integer literal ``5``,
    producing a surprising result when a caller meant to pass a
    string. The explicit form ``LiteralExpression("5")`` is still
    accepted.
    """
    with pytest.raises(
        ValueError,
        match=r"Unable to cast '5' with type <class 'str'> to an expression.",
    ):
        LiteralExpression(5) + "5"


def test_binary_dunder_rejects_unsupported_type_on_left() -> None:
    """Test a reflected binary dunder rejects a left operand of unsupported type."""
    with pytest.raises(
        ValueError,
        match=r"Unable to cast \[\] with type <class 'list'> to an expression.",
    ):
        [] + LiteralExpression(5)


# =============================================================================
# Commutative / associative tree builders
# =============================================================================


_MODULE_LOGICAL_BUILDERS = (
    pytest.param(logical_and, BinaryOperation.LOGICAL_AND, id="and"),
    pytest.param(logical_or, BinaryOperation.LOGICAL_OR, id="or"),
)


@pytest.mark.parametrize("builder, expected_operation", _MODULE_LOGICAL_BUILDERS)
def test_module_level_logical_builder_folds_three_args_right_associatively(
    builder: Callable[..., BinaryExpression],
    expected_operation: BinaryOperation,
) -> None:
    """Test module-level `logical_and`/`logical_or` fold three args right-assoc."""
    first = LiteralExpression(True)
    second = LiteralExpression(False)
    third = True  # coerced to LiteralExpression

    result = builder(first, second, third)

    expected = BinaryExpression(
        expected_operation,
        first,
        BinaryExpression(expected_operation, second, LiteralExpression(third)),
    )
    assert result.is_structurally_equivalent(expected)


@pytest.mark.parametrize("builder, expected_operation", _MODULE_LOGICAL_BUILDERS)
def test_module_level_logical_builder_accepts_a_two_argument_call(
    builder: Callable[..., BinaryExpression],
    expected_operation: BinaryOperation,
) -> None:
    """Test module-level `logical_and`/`logical_or` accept exactly two args."""
    first = LiteralExpression(True)
    second = LiteralExpression(False)

    result = builder(first, second)

    expected = BinaryExpression(expected_operation, first, second)
    assert result.is_structurally_equivalent(expected)


@pytest.mark.parametrize("builder, _expected_operation", _MODULE_LOGICAL_BUILDERS)
@pytest.mark.parametrize(
    "args",
    [
        pytest.param((), id="zero_args"),
        pytest.param((LiteralExpression(True),), id="one_arg"),
    ],
)
def test_module_level_logical_builder_requires_at_least_two_expressions(
    builder: Callable[..., BinaryExpression],
    _expected_operation: BinaryOperation,
    args: tuple[Expression, ...],
) -> None:
    """Test module-level `logical_and`/`logical_or` raise on fewer than two args."""
    with pytest.raises(ValueError, match="(?i)at least two"):
        builder(*args)


_INSTANCE_LOGICAL_METHODS = (
    pytest.param("logical_and", BinaryOperation.LOGICAL_AND, id="and"),
    pytest.param("logical_or", BinaryOperation.LOGICAL_OR, id="or"),
)


@pytest.mark.parametrize("method_name, expected_operation", _INSTANCE_LOGICAL_METHODS)
def test_instance_logical_builder_includes_self_with_one_other(
    method_name: str, expected_operation: BinaryOperation
) -> None:
    """Test `expr.logical_*(other)` builds ``expr OP other`` (self is included)."""
    first = LiteralExpression(True)
    second = LiteralExpression(False)

    result = getattr(first, method_name)(second)

    expected = BinaryExpression(expected_operation, first, second)
    assert result.is_structurally_equivalent(expected)


@pytest.mark.parametrize("method_name, expected_operation", _INSTANCE_LOGICAL_METHODS)
def test_instance_logical_builder_includes_self_with_two_others(
    method_name: str, expected_operation: BinaryOperation
) -> None:
    """Test `expr.logical_*(a, b)` folds ``(self, a, b)`` right-associatively."""
    first = LiteralExpression(True)
    second = LiteralExpression(False)
    third = LiteralExpression(True)

    result = getattr(first, method_name)(second, third)

    expected = BinaryExpression(
        expected_operation,
        first,
        BinaryExpression(expected_operation, second, third),
    )
    assert result.is_structurally_equivalent(expected)


@pytest.mark.parametrize("method_name, _expected_operation", _INSTANCE_LOGICAL_METHODS)
def test_instance_logical_builder_requires_at_least_one_other(
    method_name: str, _expected_operation: BinaryOperation
) -> None:
    """Test `expr.logical_*()` with no others raises ``ValueError``.

    The instance method counts ``self`` toward the minimum-two contract, so a
    no-other call still fails the >=-2 check on the underlying module-level
    function.
    """
    expression = LiteralExpression(True)

    with pytest.raises(ValueError, match="(?i)at least two"):
        getattr(expression, method_name)()


# =============================================================================
# Frozen dataclass & identity equality
# =============================================================================


def _build_instance_pair(
    subclass: type[Expression],
) -> tuple[Expression, Expression]:
    """Return two field-equal but distinct instances of `subclass`."""
    if subclass is LiteralExpression:
        return LiteralExpression(42), LiteralExpression(42)
    elif subclass is IdentifierExpression:
        identifier = Identifier("shared")
        return IdentifierExpression(identifier), IdentifierExpression(identifier)
    elif subclass is UnaryExpression:
        operand = LiteralExpression(1)
        return (
            UnaryExpression(UnaryOperation.NEGATE, operand),
            UnaryExpression(UnaryOperation.NEGATE, operand),
        )
    elif subclass is BinaryExpression:
        left = LiteralExpression(1)
        right = LiteralExpression(2)
        return (
            BinaryExpression(BinaryOperation.ADD, left, right),
            BinaryExpression(BinaryOperation.ADD, left, right),
        )
    else:
        raise AssertionError(f"Unknown subclass: {subclass}")


@pytest.mark.parametrize(
    "subclass, field",
    [
        (UnaryExpression, "operand"),
        (UnaryExpression, "operation"),
        (BinaryExpression, "left"),
        (BinaryExpression, "right"),
        (BinaryExpression, "operation"),
        (IdentifierExpression, "identifier"),
        (LiteralExpression, "value"),
    ],
)
def test_expression_instances_are_frozen(
    subclass: type[Expression], field: str
) -> None:
    """Test assigning to any field of an `Expression` instance is rejected."""
    instance, _ = _build_instance_pair(subclass)
    with pytest.raises(FrozenMutationError):
        setattr(instance, field, None)


@pytest.mark.parametrize(
    "subclass",
    [LiteralExpression, IdentifierExpression, UnaryExpression, BinaryExpression],
)
def test_distinct_expression_instances_are_unequal_under_eq(
    subclass: type[Expression],
) -> None:
    """Test two field-equal but distinct `Expression` instances compare `!=`."""
    first, second = _build_instance_pair(subclass)
    assert first is not second
    assert first != second
    assert second != first


@pytest.mark.parametrize(
    "subclass",
    [LiteralExpression, IdentifierExpression, UnaryExpression, BinaryExpression],
)
def test_distinct_expression_instances_have_distinct_object_ids(
    subclass: type[Expression],
) -> None:
    """Test two field-equal but distinct `Expression` instances have distinct ids.

    The ``set``-based test below (which depends on ``hash`` behavior) covers
    the user-visible consequence; this test pins that the two instances are
    genuinely distinct objects in memory, not interned by the dataclass
    machinery. ``id`` is the right comparison because it is the underlying
    invariant, not ``hash`` (which could in principle collide).
    """
    first, second = _build_instance_pair(subclass)
    assert id(first) != id(second)


@pytest.mark.parametrize(
    "subclass",
    [LiteralExpression, IdentifierExpression, UnaryExpression, BinaryExpression],
)
def test_set_of_distinct_field_equal_expressions_keeps_both_members(
    subclass: type[Expression],
) -> None:
    """Test a `set` of two field-equal but distinct instances retains both."""
    first, second = _build_instance_pair(subclass)
    assert len({first, second}) == 2


# =============================================================================
# Serialization: happy-path round trips
# =============================================================================


@pytest.mark.parametrize(
    "expression, expected_dict",
    [
        (
            LiteralExpression(True),
            {
                "__type__": "literal_expression",
                "__data__": {"value": True},
            },
        ),
        (
            IdentifierExpression(mock_identifier("x", 1)),
            {
                "__type__": "identifier_expression",
                "__data__": {"identifier": {"id": 1, "name_hint": "x"}},
            },
        ),
        (
            UnaryExpression(
                UnaryOperation.NEGATE,
                IdentifierExpression(mock_identifier("y", 2)),
            ),
            {
                "__type__": "unary_expression",
                "__data__": {
                    "operation": "negate",
                    "operand": {
                        "__type__": "identifier_expression",
                        "__data__": {
                            "identifier": {"id": 2, "name_hint": "y"},
                        },
                    },
                },
            },
        ),
        (
            BinaryExpression(
                BinaryOperation.ADD,
                IdentifierExpression(mock_identifier("x", 0)),
                LiteralExpression(5),
            ),
            {
                "__type__": "binary_expression",
                "__data__": {
                    "operation": "add",
                    "left": {
                        "__type__": "identifier_expression",
                        "__data__": {
                            "identifier": {"id": 0, "name_hint": "x"},
                        },
                    },
                    "right": {
                        "__type__": "literal_expression",
                        "__data__": {"value": 5},
                    },
                },
            },
        ),
    ],
)
def test_expression_round_trips_through_serialize_to_dict(
    expression: Expression, expected_dict: SerializedDict
) -> None:
    """Test `serialize_to_dict` yields the expected payload and round-trips."""
    assert expression.serialize_to_dict() == expected_dict
    restored = Expression.deserialize_from_dict(expected_dict)
    assert restored.is_structurally_equivalent(expression)


@pytest.mark.parametrize("operation", list(UnaryOperation))
def test_unary_expression_round_trips_for_every_operation(
    operation: UnaryOperation,
) -> None:
    """Test serialize/deserialize round-trips every `UnaryOperation` enum value."""
    expression = UnaryExpression(operation, LiteralExpression(1))
    restored = Expression.deserialize_from_dict(expression.serialize_to_dict())
    assert restored.is_structurally_equivalent(expression)


@pytest.mark.parametrize("operation", list(BinaryOperation))
def test_binary_expression_round_trips_for_every_operation(
    operation: BinaryOperation,
) -> None:
    """Test serialize/deserialize round-trips every `BinaryOperation` enum value."""
    expression = BinaryExpression(operation, LiteralExpression(1), LiteralExpression(2))
    restored = Expression.deserialize_from_dict(expression.serialize_to_dict())
    assert restored.is_structurally_equivalent(expression)


def test_expression_round_trips_through_a_deeply_nested_tree() -> None:
    """Test serialize/deserialize round-trips a 50-level left-leaning binary chain."""
    depth = 50
    expression: Expression = LiteralExpression(0)
    for term in range(1, depth + 1):
        expression = BinaryExpression(
            BinaryOperation.ADD, expression, LiteralExpression(term)
        )
    restored = Expression.deserialize_from_dict(expression.serialize_to_dict())
    assert restored.is_structurally_equivalent(expression)


def test_deserialize_unary_rejects_invalid_operation_name() -> None:
    """Test unary deserialization raises `DeserializationValueError` on bad op name."""
    data: SerializedDict = {
        "__type__": "unary_expression",
        "__data__": {
            "operation": "not_an_operation",
            "operand": {
                "__type__": "literal_expression",
                "__data__": {"value": 1},
            },
        },
    }
    with pytest.raises(DeserializationValueError):
        Expression.deserialize_from_dict(data)


def test_deserialize_binary_rejects_invalid_operation_name() -> None:
    """Test binary deserialization raises `DeserializationValueError` on bad op name."""
    data: SerializedDict = {
        "__type__": "binary_expression",
        "__data__": {
            "operation": "not_an_operation",
            "left": {
                "__type__": "literal_expression",
                "__data__": {"value": 1},
            },
            "right": {
                "__type__": "literal_expression",
                "__data__": {"value": 2},
            },
        },
    }
    with pytest.raises(DeserializationValueError):
        Expression.deserialize_from_dict(data)


# =============================================================================
# Serialization: structural validation errors
# =============================================================================

_VALID_LITERAL_DICT: SerializedDict = {
    "__type__": "literal_expression",
    "__data__": {"value": 1},
}


@pytest.mark.parametrize(
    "data",
    [
        pytest.param(
            {
                "__type__": "unary_expression",
                "__data__": {"operand": _VALID_LITERAL_DICT},
            },
            id="missing_operation",
        ),
        pytest.param(
            {
                "__type__": "unary_expression",
                "__data__": {"operation": 42, "operand": _VALID_LITERAL_DICT},
            },
            id="operation_not_str",
        ),
        pytest.param(
            {"__type__": "unary_expression", "__data__": {"operation": "negate"}},
            id="missing_operand",
        ),
        pytest.param(
            {
                "__type__": "unary_expression",
                "__data__": {"operation": "negate", "operand": "not-a-dict"},
            },
            id="operand_not_serialized_dict",
        ),
    ],
)
def test_deserialize_unary_rejects_invalid_data_shape(data: SerializedDict) -> None:
    """Test unary deserialization raises on missing or wrong-typed fields."""
    with pytest.raises(DeserializationDictStructureError):
        Expression.deserialize_from_dict(data)


@pytest.mark.parametrize(
    "data",
    [
        pytest.param(
            {
                "__type__": "binary_expression",
                "__data__": {
                    "left": _VALID_LITERAL_DICT,
                    "right": _VALID_LITERAL_DICT,
                },
            },
            id="missing_operation",
        ),
        pytest.param(
            {
                "__type__": "binary_expression",
                "__data__": {
                    "operation": 7,
                    "left": _VALID_LITERAL_DICT,
                    "right": _VALID_LITERAL_DICT,
                },
            },
            id="operation_not_str",
        ),
        pytest.param(
            {
                "__type__": "binary_expression",
                "__data__": {
                    "operation": "add",
                    "right": _VALID_LITERAL_DICT,
                },
            },
            id="missing_left",
        ),
        pytest.param(
            {
                "__type__": "binary_expression",
                "__data__": {
                    "operation": "add",
                    "left": "not-a-dict",
                    "right": _VALID_LITERAL_DICT,
                },
            },
            id="left_not_serialized_dict",
        ),
        pytest.param(
            {
                "__type__": "binary_expression",
                "__data__": {
                    "operation": "add",
                    "left": _VALID_LITERAL_DICT,
                },
            },
            id="missing_right",
        ),
        pytest.param(
            {
                "__type__": "binary_expression",
                "__data__": {
                    "operation": "add",
                    "left": _VALID_LITERAL_DICT,
                    "right": "not-a-dict",
                },
            },
            id="right_not_serialized_dict",
        ),
    ],
)
def test_deserialize_binary_rejects_invalid_data_shape(data: SerializedDict) -> None:
    """Test binary deserialization raises on missing or wrong-typed fields."""
    with pytest.raises(DeserializationDictStructureError):
        Expression.deserialize_from_dict(data)


@pytest.mark.parametrize(
    "data",
    [
        pytest.param(
            {"__type__": "identifier_expression", "__data__": {}},
            id="missing_identifier",
        ),
        pytest.param(
            {
                "__type__": "identifier_expression",
                "__data__": {"identifier": "not-a-dict"},
            },
            id="identifier_not_serialized_dict",
        ),
    ],
)
def test_deserialize_identifier_rejects_invalid_data_shape(
    data: SerializedDict,
) -> None:
    """Test identifier deserialization raises on missing or wrong-typed fields."""
    with pytest.raises(DeserializationDictStructureError):
        Expression.deserialize_from_dict(data)


@pytest.mark.parametrize(
    "data",
    [
        pytest.param(
            {"__type__": "literal_expression", "__data__": {}},
            id="missing_value",
        ),
        pytest.param(
            {"__type__": "literal_expression", "__data__": {"value": [1, 2, 3]}},
            id="value_of_unsupported_type_list",
        ),
        pytest.param(
            {"__type__": "literal_expression", "__data__": {"value": None}},
            id="value_of_unsupported_type_none",
        ),
    ],
)
def test_deserialize_literal_rejects_invalid_data_shape(
    data: SerializedDict,
) -> None:
    """Test literal deserialization raises on missing or unsupported-value fields."""
    with pytest.raises(DeserializationDictStructureError):
        Expression.deserialize_from_dict(data)


# =============================================================================
# Walker-driven traversal via `get_visit_children`
# =============================================================================


def test_collect_identifiers_walks_into_unary_expression_operand() -> None:
    """Test the walker visits a `UnaryExpression` operand via `get_visit_children`."""
    x = Identifier("x")
    expression = UnaryExpression(UnaryOperation.NEGATE, IdentifierExpression(x))

    assert collect_identifiers(expression) == {x}


def test_collect_identifiers_walks_into_binary_expression_children() -> None:
    """Test the walker visits both `BinaryExpression` children via child dispatch."""
    left = Identifier("a")
    right = Identifier("b")
    expression = BinaryExpression(
        BinaryOperation.ADD, IdentifierExpression(left), IdentifierExpression(right)
    )

    assert collect_identifiers(expression) == {left, right}


# =============================================================================
# Structural equivalence fallback for unregistered `Expression` subclasses
# =============================================================================


def test_unregistered_expression_subclass_raises_not_implemented_on_equivalence() -> (
    None
):
    """Test the singledispatch default raises ``NotImplementedError``.

    A concrete subclass that does not register a custom dispatcher must
    surface as a clear error rather than silently returning ``False``,
    so adding a new node type without wiring up equivalence fails loudly.
    """

    @dataclasses.dataclass(frozen=True, eq=False)
    class _UnregisteredExpression(Expression):  # test-local subclass
        value: int

        def serialize_data_to_dict(self) -> SerializedDict:  # pragma: no cover
            return {"value": self.value}

        @classmethod
        def deserialize_data_from_dict(  # pragma: no cover
            cls, data: SerializedDict
        ) -> "_UnregisteredExpression":
            return cls(value=int(data["value"]))  # type: ignore[arg-type]

    instance = _UnregisteredExpression(value=1)

    with pytest.raises(NotImplementedError, match="_UnregisteredExpression"):
        instance.is_structurally_equivalent(LiteralExpression(1))


# =============================================================================
# logical_and / logical_or: right-fold and arity
# =============================================================================


def test_logical_and_right_folds_three_operands() -> None:
    """Test `logical_and(a, b, c)` produces ``a && (b && c)``."""
    a = LiteralExpression(True)
    b = LiteralExpression(False)
    c = LiteralExpression(True)

    result = logical_and(a, b, c)

    expected = BinaryExpression(
        BinaryOperation.LOGICAL_AND,
        a,
        BinaryExpression(BinaryOperation.LOGICAL_AND, b, c),
    )
    assert result.is_structurally_equivalent(expected)


def test_logical_or_right_folds_three_operands() -> None:
    """Test `logical_or(a, b, c)` produces ``a || (b || c)``."""
    a = LiteralExpression(True)
    b = LiteralExpression(False)
    c = LiteralExpression(True)

    result = logical_or(a, b, c)

    expected = BinaryExpression(
        BinaryOperation.LOGICAL_OR,
        a,
        BinaryExpression(BinaryOperation.LOGICAL_OR, b, c),
    )
    assert result.is_structurally_equivalent(expected)


def test_logical_and_right_folds_four_operands() -> None:
    """Test `logical_and(a, b, c, d)` produces ``a && (b && (c && d))``."""
    a, b, c, d = (
        LiteralExpression(True),
        LiteralExpression(False),
        LiteralExpression(True),
        LiteralExpression(False),
    )

    result = logical_and(a, b, c, d)

    expected = BinaryExpression(
        BinaryOperation.LOGICAL_AND,
        a,
        BinaryExpression(
            BinaryOperation.LOGICAL_AND,
            b,
            BinaryExpression(BinaryOperation.LOGICAL_AND, c, d),
        ),
    )
    assert result.is_structurally_equivalent(expected)


def test_logical_and_rejects_fewer_than_two_operands() -> None:
    """Test `logical_and` with one operand raises ``ValueError``."""
    with pytest.raises(ValueError, match="requires at least two"):
        logical_and(LiteralExpression(True))


def test_logical_and_rejects_zero_operands() -> None:
    """Test `logical_and` with no operands raises ``ValueError``."""
    with pytest.raises(ValueError, match="requires at least two"):
        logical_and()


def test_logical_or_rejects_fewer_than_two_operands() -> None:
    """Test `logical_or` with one operand raises ``ValueError``."""
    with pytest.raises(ValueError, match="requires at least two"):
        logical_or(LiteralExpression(False))


# =============================================================================
# make_binary_expression / make_unary_expression: direct construction
# =============================================================================


def test_make_binary_expression_constructs_with_literal_coercion() -> None:
    """Test `make_binary_expression` coerces ``int``/``float`` operands."""
    result = make_binary_expression(BinaryOperation.ADD, 1, 2.5)

    expected = BinaryExpression(
        BinaryOperation.ADD, LiteralExpression(1), LiteralExpression(2.5)
    )
    assert result.is_structurally_equivalent(expected)


def test_make_binary_expression_rejects_unsupported_left_operand() -> None:
    """Test `make_binary_expression` rejects an unsupported left operand type."""
    with pytest.raises(ValueError, match="Unable to cast"):
        make_binary_expression(BinaryOperation.ADD, [], LiteralExpression(1))  # type: ignore[arg-type]


def test_make_binary_expression_rejects_unsupported_right_operand() -> None:
    """Test `make_binary_expression` rejects an unsupported right operand type."""
    with pytest.raises(ValueError, match="Unable to cast"):
        make_binary_expression(BinaryOperation.ADD, LiteralExpression(1), {})  # type: ignore[arg-type]


def test_make_unary_expression_constructs_with_literal_coercion() -> None:
    """Test `make_unary_expression` coerces a numeric operand."""
    result = make_unary_expression(UnaryOperation.NEGATE, 5)

    expected = UnaryExpression(UnaryOperation.NEGATE, LiteralExpression(5))
    assert result.is_structurally_equivalent(expected)


def test_make_unary_expression_rejects_unsupported_operand() -> None:
    """Test `make_unary_expression` raises ``ValueError`` for an unsupported type."""
    with pytest.raises(ValueError, match="Unable to cast"):
        make_unary_expression(UnaryOperation.NEGATE, object())  # type: ignore[arg-type]
