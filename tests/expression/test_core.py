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
    logical_not,
    logical_or,
    make_binary_expression,
    make_unary_expression,
)
from fhy_core.identifier import Identifier
from fhy_core.serialization import (
    DeserializationDictStructureError,
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


@pytest.mark.parametrize("string_value", ["0", "5", "42", "00", "01"])
def test_literal_expression_keeps_integer_shaped_string_as_str(
    string_value: str,
) -> None:
    """Test an integer-shaped ``str`` is preserved as the caller's exact text.

    String-form literals retain the caller's textual representation so
    downstream passes can do exact-decimal arithmetic before any
    conversion to ``int`` or ``float`` at a native boundary; equivalence
    against the matching ``int`` form is handled by the structural- and
    alpha-equivalence dispatches, not by canonicalization at
    construction time.
    """
    literal = LiteralExpression(string_value)

    assert literal.value == string_value
    assert type(literal.value) is str


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
def test_literal_expression_rejects_string_outside_numeric_grammar(
    string_value: str,
) -> None:
    """Test ``str`` values not matching the integer or float grammar raise.

    ``LiteralExpression`` accepts string-form numeric literals to preserve
    exact decimal text without IEEE-754 rounding; any string outside the
    integer or float grammar is rejected at construction time.
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


def test_literal_equivalence_is_false_when_values_differ_in_either_direction() -> None:
    """Test `LiteralExpression` equivalence is value-equal and symmetrically false."""
    smaller = LiteralExpression(5)
    larger = LiteralExpression(10)
    assert not smaller.is_structurally_equivalent(larger)
    assert not larger.is_structurally_equivalent(smaller)


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
    [(0, "0"), (1, "1"), (42, "42"), (0, "00"), (1, "01")],
)
def test_int_literal_and_integer_shaped_string_compare_structurally_equivalent(
    int_value: int, equivalent_string: str
) -> None:
    """Test int and integer-shaped-string literals compare structurally equivalent.

    String-form integer literals are preserved as ``str`` at
    construction time; structural equivalence canonicalizes to the
    underlying ``int`` so the two forms are interchangeable for
    comparison.
    """
    left = LiteralExpression(int_value)
    right = LiteralExpression(equivalent_string)

    assert left.is_structurally_equivalent(right)
    assert right.is_structurally_equivalent(left)


@pytest.mark.parametrize(
    "left_text, right_text",
    [
        pytest.param("5", "05", id="leading_zero"),
        pytest.param("00", "0", id="multiple_zero_forms"),
        pytest.param("42", "042", id="larger_value_with_leading_zero"),
    ],
)
def test_integer_shaped_strings_with_distinct_text_compare_structurally_equivalent(
    left_text: str, right_text: str
) -> None:
    """Test integer-shaped strings with the same int value are equivalent."""
    left = LiteralExpression(left_text)
    right = LiteralExpression(right_text)

    assert left.is_structurally_equivalent(right)
    assert right.is_structurally_equivalent(left)


@pytest.mark.parametrize(
    "left_text, right_text",
    [
        pytest.param("1.5", "1.50", id="trailing_zero"),
        pytest.param("0.1", "0.10", id="trailing_zero_after_leading_zero"),
        pytest.param(".5", "0.5", id="missing_leading_zero"),
        pytest.param("1.", "1.0", id="trailing_dot_vs_dot_zero"),
        pytest.param("1.5", "01.5", id="leading_zero_before_decimal"),
    ],
)
def test_float_shaped_strings_with_distinct_text_compare_structurally_equivalent(
    left_text: str, right_text: str
) -> None:
    """Test float-shaped strings with the same decimal value are equivalent.

    Decimal canonicalization handles trailing zeros, leading zeros, and
    elided integer or fractional parts so ``"1.5"``, ``"1.50"``,
    ``"01.5"``, and ``"1."`` all compare equal to their normalized
    siblings.
    """
    left = LiteralExpression(left_text)
    right = LiteralExpression(right_text)

    assert left.is_structurally_equivalent(right)
    assert right.is_structurally_equivalent(left)


@pytest.mark.parametrize(
    "left_value, right_value",
    [
        pytest.param("1.5", 1.5, id="float_decimal_vs_float_binary"),
        pytest.param("0.1", 0.1, id="exact_decimal_vs_ieee_approximation"),
        pytest.param("5", "5.0", id="int_form_vs_float_decimal"),
        pytest.param(5, "5.0", id="int_vs_float_decimal"),
        pytest.param("5", 5.0, id="int_form_vs_float_binary"),
    ],
)
def test_literal_equivalence_distinguishes_buckets(
    left_value: bool | int | float | str,
    right_value: bool | int | float | str,
) -> None:
    """Test cross-bucket literals are not structurally equivalent.

    ``LiteralExpression`` distinguishes four equivalence buckets:
    ``bool``, integer (``int`` and integer-grammar ``str``),
    float-binary (Python ``float``), and float-decimal (float-grammar
    ``str``). Values that fall in different buckets are not equivalent
    even when their numeric values agree, because the buckets carry
    different precision contracts (exact-decimal text vs IEEE-754
    binary, integer vs float).
    """
    left = LiteralExpression(left_value)
    right = LiteralExpression(right_value)

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


@pytest.mark.parametrize(
    "right_value, expected_value",
    [
        pytest.param("5", "5", id="integer_form_string"),
        pytest.param("1.5", "1.5", id="float_form_string"),
    ],
)
def test_binary_dunder_lifts_str_operand_on_right(
    right_value: str, expected_value: str
) -> None:
    """Test a binary dunder lifts a ``str`` right operand into a ``LiteralExpression``.

    ``LiteralExpression`` preserves string-form numeric literals so they
    can flow through implicit coercion without losing their textual
    representation.
    """
    expression = LiteralExpression(1) + right_value

    assert isinstance(expression.right, LiteralExpression)
    assert expression.right.value == expected_value
    assert type(expression.right.value) is str


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


def test_module_level_logical_not_wraps_operand_in_unary_expression() -> None:
    """Test `logical_not(expr)` builds a ``LOGICAL_NOT`` unary node."""
    operand = LiteralExpression(True)

    result = logical_not(operand)

    expected = UnaryExpression(UnaryOperation.LOGICAL_NOT, operand)
    assert result.is_structurally_equivalent(expected)


def test_module_level_logical_not_coerces_bare_identifier() -> None:
    """Test `logical_not` lifts a bare `Identifier` via `IdentifierExpression`."""
    identifier = Identifier("a")

    result = logical_not(identifier)

    expected = UnaryExpression(
        UnaryOperation.LOGICAL_NOT, IdentifierExpression(identifier)
    )
    assert result.is_structurally_equivalent(expected)


def test_module_level_logical_not_coerces_bare_python_literal() -> None:
    """Test `logical_not` lifts a bare Python ``bool`` via `LiteralExpression`."""
    result = logical_not(True)

    expected = UnaryExpression(UnaryOperation.LOGICAL_NOT, LiteralExpression(True))
    assert result.is_structurally_equivalent(expected)


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


def test_literal_expression_round_trips_through_serialize_to_dict() -> None:
    """Test `LiteralExpression` (hand-written codec) round-trips with its dict shape.

    ``LiteralExpression`` keeps a bespoke ``serialize_data_to_dict`` /
    ``deserialize_data_from_dict`` because its ``value`` is a scalar union the
    derivation engine cannot infer, so it retains a dedicated serialization
    test. The derived expression classes are covered centrally by the
    serialization-engine tests instead.
    """
    expression = LiteralExpression(True)
    expected_dict: SerializedDict = {
        "__type__": "literal_expression",
        "__data__": {"value": True},
    }
    assert expression.serialize_to_dict() == expected_dict
    restored = Expression.deserialize_from_dict(expected_dict)
    assert restored.is_structurally_equivalent(expression)


# =============================================================================
# Serialization: structural validation errors
# =============================================================================


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


def test_new_expression_subclass_derives_equivalence_without_registration() -> None:
    """Test a new `Expression` subclass derives equivalence from its fields.

    Derivation replaces the per-type registry: a concrete subclass needs no
    wiring to gain structural and alpha equivalence. Same-type instances
    compare field by field; a different concrete type returns ``False``
    rather than raising.
    """

    @dataclasses.dataclass(frozen=True, eq=False)
    class _NewExpression(Expression):  # test-local subclass
        value: int

        def serialize_data_to_dict(self) -> SerializedDict:  # pragma: no cover
            return {"value": self.value}

        @classmethod
        def deserialize_data_from_dict(  # pragma: no cover
            cls, data: SerializedDict
        ) -> "_NewExpression":
            return cls(value=int(data["value"]))  # type: ignore[arg-type]

    assert _NewExpression(1).is_structurally_equivalent(_NewExpression(1))
    assert not _NewExpression(1).is_structurally_equivalent(_NewExpression(2))
    assert _NewExpression(1).is_alpha_equivalent(_NewExpression(1))
    assert not _NewExpression(1).is_structurally_equivalent(LiteralExpression(1))


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
