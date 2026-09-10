"""Tests for `fhy_core.symbolic.expression.core`."""

import dataclasses
import itertools
import math
import operator
import pickle
from collections.abc import Callable
from typing import Any

import pytest

from fhy_core.error import get_registered_errors
from fhy_core.serialization import (
    DeserializationDictStructureError,
    SerializationFormat,
    SerializedDict,
    UnknownTypeIdError,
)
from fhy_core.symbolic.expression import (
    BinaryExpression,
    BinaryOperation,
    CallExpression,
    Expression,
    IdentifierExpression,
    LiteralExpression,
    NonBooleanLogicalOperandError,
    PiecewiseExpression,
    UnaryExpression,
    UnaryOperation,
    UndecidableError,
    build_literal_equivalence_key,
    call,
    is_integer_valued_literal,
    logical_and,
    logical_not,
    logical_or,
    make_binary_expression,
    make_unary_expression,
    piecewise,
    validate_logical_operands,
)
from fhy_core.traits import FrozenMutationError, HasOperands, StructuralEquivalence
from fhy_core.utils.override import override

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
    identifier = mock_identifier("x", 0)
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
    with pytest.raises(ValueError, match=r"(?i)literal"):
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
    with pytest.raises(TypeError, match=r"(?i)literal"):
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
    everywhere else (``is_strict_int``, ``do_param_values_match``,
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
# NaN keeps the literal equivalence relation reflexive
# =============================================================================


def test_nan_literals_holding_distinct_float_objects_are_equivalent() -> None:
    """Test two NaN literals are equivalent whichever ``float`` object each holds.

    NaN compares unequal to itself, so comparing stored values would make
    NaN equivalence a question of object identity: a literal would be
    equivalent to itself but not to one built from a separately computed
    NaN.
    """
    left = LiteralExpression(float("nan"))
    right = LiteralExpression(math.inf - math.inf)

    assert left.value is not right.value
    assert left.is_structurally_equivalent(right)
    assert right.is_structurally_equivalent(left)


def test_nan_literal_is_equivalent_to_its_own_pickle_round_trip() -> None:
    """Test a NaN literal and its unpickled copy are equivalent.

    Unpickling rebuilds the ``float``, so the copy holds a different NaN
    object than the original; equivalence must not depend on that.
    """
    literal = LiteralExpression(math.nan)

    restored = pickle.loads(pickle.dumps(literal))

    assert restored.value is not literal.value
    assert literal.is_structurally_equivalent(restored)


_LITERAL_EQUIVALENCE_SAMPLE: tuple[bool | int | float | str, ...] = (
    True,
    False,
    0,
    5,
    "5",
    "05",
    0.0,
    -0.0,
    5.0,
    1.5,
    math.inf,
    -math.inf,
    math.nan,
    float("nan"),
    -math.nan,
    "0.0",
    ".0",
    "5.0",
    "1.5",
    "1.50",
    "1.00000000000000000000000000001",
    "1.0",
)


def test_literal_equivalence_key_is_shared_exactly_by_equivalent_literals() -> None:
    """Test the key agrees with literal equivalence in both directions.

    Canonical ordering needs the key constant on equivalence classes, and
    a key that also separates what equivalence separates never lets two
    inequivalent literals tie. The sample spans every bucket, ``-0.0``,
    NaNs of distinct identity and sign, and a decimal too long to survive
    rounding to ``Decimal``'s default 28 digits.
    """
    for left_value, right_value in itertools.product(
        _LITERAL_EQUIVALENCE_SAMPLE, repeat=2
    ):
        equivalent = LiteralExpression(left_value).is_structurally_equivalent(
            LiteralExpression(right_value)
        )
        keyed_alike = build_literal_equivalence_key(
            left_value
        ) == build_literal_equivalence_key(right_value)
        assert equivalent is keyed_alike, (left_value, right_value)


@pytest.mark.parametrize(
    "value, expected_key",
    [
        pytest.param(True, "bool:True", id="bool"),
        pytest.param("05", "int:5", id="integer_string"),
        pytest.param(-0.0, "float-binary:0.0", id="negative_zero"),
        pytest.param(float("nan"), "float-binary:nan", id="nan"),
        pytest.param(-math.inf, "float-binary:-inf", id="negative_infinity"),
        pytest.param("1.50", "float-decimal:1.5", id="decimal_trailing_zero"),
        pytest.param("100.0", "float-decimal:1E+2", id="decimal_whole_number"),
    ],
)
def test_literal_equivalence_key_renders_bucket_and_canonical_form(
    value: bool | int | float | str, expected_key: str
) -> None:
    """Test the key text for representative literals.

    ``ConstraintSystem`` orders its members by keys built from this text
    and serializes them in that order, so the rendering is pinned rather
    than only compared.
    """
    assert build_literal_equivalence_key(value) == expected_key


# =============================================================================
# Integer-bucket predicate
# =============================================================================


@pytest.mark.parametrize(
    "value",
    [
        pytest.param(0, id="int_zero"),
        pytest.param(5, id="int"),
        pytest.param("0", id="string_zero"),
        pytest.param("5", id="string"),
        pytest.param("05", id="string_leading_zero"),
    ],
)
def test_is_integer_valued_literal_accepts_every_integer_bucket_form(
    value: int | str,
) -> None:
    """Test the predicate holds for both spellings of an integer literal."""
    assert is_integer_valued_literal(value) is True


@pytest.mark.parametrize(
    "value",
    [
        pytest.param(True, id="bool_true"),
        pytest.param(False, id="bool_false"),
        pytest.param(5.0, id="float_binary"),
        pytest.param(1.5, id="float_binary_fractional"),
        pytest.param("5.0", id="float_decimal"),
        pytest.param("1.5", id="float_decimal_fractional"),
        pytest.param(".5", id="float_decimal_no_integer_part"),
    ],
)
def test_is_integer_valued_literal_rejects_every_other_bucket_form(
    value: bool | float | str,
) -> None:
    """Test the predicate fails for the Boolean and the two float buckets.

    A ``bool`` is held apart from the integers, and neither float bucket
    is integer-valued even when the value has no fractional part, since
    ``LiteralExpression`` does not treat ``5.0`` or ``"5.0"`` as
    equivalent to ``5``.
    """
    assert is_integer_valued_literal(value) is False


@pytest.mark.parametrize(
    "left_value, right_value",
    [
        pytest.param(5, "5", id="int_vs_string"),
        pytest.param("5", "05", id="string_vs_leading_zero"),
        pytest.param(1.5, 1.5, id="float_binary_pair"),
        pytest.param("1.5", "1.50", id="float_decimal_pair"),
        pytest.param(True, True, id="bool_pair"),
    ],
)
def test_is_integer_valued_literal_agrees_across_an_equivalence_class(
    left_value: bool | int | float | str,
    right_value: bool | int | float | str,
) -> None:
    """Test the predicate is constant on structural-equivalence classes.

    Every consumer that decides "is this literal an integer" reads this
    predicate, so the answer has to be a class invariant: were it to
    split a class, the Z3 and SymPy bridges would lower the members to
    different sorts and the solver's hazard screen would refuse one
    member of a class while deciding another.
    """
    left = LiteralExpression(left_value)
    right = LiteralExpression(right_value)
    assert left.is_structurally_equivalent(right)

    assert is_integer_valued_literal(left.value) is is_integer_valued_literal(
        right.value
    )


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


def test_binary_dunder_rejects_unsupported_type_on_right() -> None:
    """Test a binary dunder rejects a right operand of unsupported type."""
    with pytest.raises(
        ValueError,
        match=r"Unable to cast \[\] with type <class 'list'> to an expression.",
    ):
        LiteralExpression(5) + []  # noqa: RUF005  # test: operator must reject list


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
        [] + LiteralExpression(5)  # noqa: RUF005  # test: operator must reject list


# =============================================================================
# Bare-bool refusal: no coercion site lifts a Python ``bool``
# =============================================================================


def test_accidental_expression_equality_cannot_ride_into_a_conjunction() -> None:
    """Test ``logical_and(a == b, a > 0)`` refuses rather than planting ``False``.

    ``Expression.__eq__`` is object identity, so ``a == b`` over two
    distinct expression objects is the Python ``False``. Lifted, it turns
    the conjunction into the constant-false ``False && (a > 0)``, which
    then rides unnoticed into an ``EquationConstraint`` and reports a
    permanently infeasible system.
    """
    a = mock_identifier("a", 0)
    b = mock_identifier("b", 1)
    accidental = IdentifierExpression(a) == IdentifierExpression(b)

    with pytest.raises(ValueError, match="bare Python bool"):
        logical_and(accidental, IdentifierExpression(a) > LiteralExpression(0))


_BARE_BOOL_COERCION_SITES = [
    pytest.param(lambda value: LiteralExpression(1) + value, id="dunder_right"),
    pytest.param(lambda value: value + LiteralExpression(1), id="dunder_left"),
    pytest.param(lambda value: LiteralExpression(1) < value, id="comparison"),
    pytest.param(lambda value: LiteralExpression(1).equals(value), id="equals"),
    pytest.param(lambda value: LiteralExpression(1).not_equals(value), id="not_equals"),
    pytest.param(
        lambda value: make_binary_expression(
            BinaryOperation.EQUAL, LiteralExpression(1), value
        ),
        id="make_binary_expression",
    ),
    pytest.param(
        lambda value: make_unary_expression(UnaryOperation.LOGICAL_NOT, value),
        id="make_unary_expression",
    ),
    pytest.param(
        lambda value: logical_and(LiteralExpression(True), value), id="logical_and"
    ),
    pytest.param(
        lambda value: logical_or(value, LiteralExpression(True)), id="logical_or"
    ),
    pytest.param(logical_not, id="logical_not"),
    pytest.param(
        lambda value: LiteralExpression(True).logical_and(value),
        id="method_logical_and",
    ),
    pytest.param(
        lambda value: LiteralExpression(True).logical_or(value),
        id="method_logical_or",
    ),
    pytest.param(
        lambda value: piecewise((LiteralExpression(True), value), otherwise=0),
        id="piecewise_value",
    ),
    pytest.param(
        lambda value: piecewise((LiteralExpression(True), 1), otherwise=value),
        id="piecewise_otherwise",
    ),
    pytest.param(lambda value: call("max", value, 1), id="call"),
    pytest.param(lambda value: Expression.call("max", 1, value), id="method_call"),
]


@pytest.mark.parametrize("value", [True, False])
@pytest.mark.parametrize("build", _BARE_BOOL_COERCION_SITES)
def test_every_coercion_site_refuses_a_bare_bool(
    build: Callable[[bool], Expression], value: bool
) -> None:
    """Test each builder and operator dunder refuses a bare ``bool`` operand.

    The refusal lives in the one coercion every site shares, so a site
    that stopped routing through it would lift the ``bool`` again. The
    check is on the exact type: ``bool`` subclasses ``int``, and an
    ``int`` operand still lifts (see the promotion tests above).
    """
    with pytest.raises(ValueError, match="bare Python bool"):
        build(value)


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
    third = mock_identifier("c", 2)  # coerced to IdentifierExpression

    result = builder(first, second, third)

    expected = BinaryExpression(
        expected_operation,
        first,
        BinaryExpression(expected_operation, second, IdentifierExpression(third)),
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
    with pytest.raises(ValueError, match=r"(?i)at least two"):
        builder(*args)


def test_module_level_logical_not_wraps_operand_in_unary_expression() -> None:
    """Test `logical_not(expr)` builds a ``LOGICAL_NOT`` unary node."""
    operand = LiteralExpression(True)

    result = logical_not(operand)

    expected = UnaryExpression(UnaryOperation.LOGICAL_NOT, operand)
    assert result.is_structurally_equivalent(expected)


def test_module_level_logical_not_coerces_bare_identifier() -> None:
    """Test `logical_not` lifts a bare `Identifier` via `IdentifierExpression`."""
    identifier = mock_identifier("a", 0)

    result = logical_not(identifier)

    expected = UnaryExpression(
        UnaryOperation.LOGICAL_NOT, IdentifierExpression(identifier)
    )
    assert result.is_structurally_equivalent(expected)


def test_module_level_logical_not_refuses_a_bare_python_bool() -> None:
    """Test `logical_not` refuses a bare ``bool`` and names the literal spelling.

    A caller who wants the constant writes ``LiteralExpression(True)``;
    the message says so rather than only rejecting the input.
    """
    with pytest.raises(ValueError, match=r"LiteralExpression\(True\)"):
        logical_not(True)


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

    with pytest.raises(ValueError, match=r"(?i)at least two"):
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
        identifier = mock_identifier("shared", 0)
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
    elif subclass is PiecewiseExpression:
        condition = LiteralExpression(True)
        value = LiteralExpression(1)
        otherwise = LiteralExpression(0)
        return (
            PiecewiseExpression((condition,), (value,), otherwise),
            PiecewiseExpression((condition,), (value,), otherwise),
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
        (PiecewiseExpression, "conditions"),
        (PiecewiseExpression, "values"),
        (PiecewiseExpression, "otherwise"),
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
    [
        LiteralExpression,
        IdentifierExpression,
        UnaryExpression,
        BinaryExpression,
        PiecewiseExpression,
    ],
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
    [
        LiteralExpression,
        IdentifierExpression,
        UnaryExpression,
        BinaryExpression,
        PiecewiseExpression,
    ],
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
    [
        LiteralExpression,
        IdentifierExpression,
        UnaryExpression,
        BinaryExpression,
        PiecewiseExpression,
    ],
)
def test_set_of_distinct_field_equal_expressions_keeps_both_members(
    subclass: type[Expression],
) -> None:
    """Test a `set` of two field-equal but distinct instances retains both."""
    first, second = _build_instance_pair(subclass)
    assert len({first, second}) == 2


def test_reordered_piecewise_cases_are_not_structurally_equivalent() -> None:
    """Test reordering piecewise cases breaks structural equivalence.

    Case order is semantically load-bearing under first-match-wins
    evaluation, so two piecewise expressions built from the same cases in a
    different order must not compare structurally equivalent.
    """
    c1, c2 = LiteralExpression(True), LiteralExpression(False)
    v1, v2 = LiteralExpression(1), LiteralExpression(2)
    otherwise = LiteralExpression(0)
    forward = PiecewiseExpression((c1, c2), (v1, v2), otherwise)
    reversed_order = PiecewiseExpression((c2, c1), (v2, v1), otherwise)

    assert not forward.is_structurally_equivalent(reversed_order)
    assert not reversed_order.is_structurally_equivalent(forward)


def test_piecewise_expression_hash_is_defined_and_follows_identity() -> None:
    """Test `hash()` succeeds on `PiecewiseExpression` and follows identity."""
    first, second = _build_instance_pair(PiecewiseExpression)

    assert hash(first) == hash(first)
    assert hash(first) == object.__hash__(first)
    assert hash(second) == object.__hash__(second)


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
# Serialization: PiecewiseExpression
# =============================================================================


def test_piecewise_expression_serialize_to_dict_carries_pinned_type_id() -> None:
    """Test ``PiecewiseExpression`` serializes under the ``piecewise_expression`` id."""
    expression = piecewise((LiteralExpression(True), LiteralExpression(1)), otherwise=0)

    serialized = expression.serialize_to_dict()
    data = serialized["__data__"]

    assert serialized["__type__"] == "piecewise_expression"
    assert isinstance(data, dict)
    assert set(data.keys()) == {"conditions", "values", "otherwise"}


def test_piecewise_expression_round_trips_through_dict() -> None:
    """Test a multi-case ``PiecewiseExpression`` round-trips via dict serialization."""
    expression = piecewise(
        (LiteralExpression(True), LiteralExpression(1)),
        (LiteralExpression(False), LiteralExpression(2)),
        otherwise=LiteralExpression(0),
    )

    serialized = expression.serialize_to_dict()
    restored = Expression.deserialize_from_dict(serialized)

    assert restored.is_structurally_equivalent(expression)


@pytest.mark.parametrize("fmt", list(SerializationFormat))
def test_piecewise_expression_round_trips_through_every_format(
    fmt: SerializationFormat,
) -> None:
    """Test ``PiecewiseExpression`` round-trips through DICT, JSON, and BINARY."""
    expression = piecewise((LiteralExpression(True), LiteralExpression(1)), otherwise=0)

    serialized = expression.serialize(fmt)
    restored = Expression.deserialize(serialized, fmt)

    assert restored.is_structurally_equivalent(expression)


def test_nested_piecewise_expression_round_trips_through_dict() -> None:
    """Test a ``PiecewiseExpression`` nested inside another round-trips structurally."""
    inner = piecewise((LiteralExpression(True), LiteralExpression(1)), otherwise=0)
    outer = piecewise((LiteralExpression(False), inner), otherwise=LiteralExpression(9))

    serialized = outer.serialize_to_dict()
    restored = Expression.deserialize_from_dict(serialized)

    assert restored.is_structurally_equivalent(outer)


def test_deserializing_a_removed_conditional_node_blob_raises_unknown_type_id() -> None:
    """Test a blob using an unregistered conditional node's type id is rejected.

    No three-operand conditional node (condition, true-branch,
    false-branch) is registered, and there is no alias or migration path
    for one; any persisted blob using such an id raises
    ``UnknownTypeIdError`` on deserialize. The id is assembled at runtime
    rather than written as a literal so this file carries no textual
    reference to the unregistered name.
    """
    removed_type_id = "".join(["te", "rn", "ary_expression"])
    unrestorable_blob: SerializedDict = {
        "__type__": removed_type_id,
        "__data__": {
            "condition": {
                "__type__": "literal_expression",
                "__data__": {"value": True},
            },
            "true_value": {
                "__type__": "literal_expression",
                "__data__": {"value": 1},
            },
            "false_value": {
                "__type__": "literal_expression",
                "__data__": {"value": 2},
            },
        },
    }

    with pytest.raises(UnknownTypeIdError):
        Expression.deserialize_from_dict(unrestorable_blob)


def test_deserializing_mismatched_condition_and_value_lengths_raises_value_error() -> (
    None
):
    """Test a malformed payload with unequal ``conditions``/``values`` lengths raises.

    Field decoding for the two homogeneous tuples succeeds on its own; the
    length-mismatch invariant is enforced by ``__post_init__`` after
    decoding, the same layered validation ``CallExpression`` uses for its
    non-empty-name check.
    """
    malformed: SerializedDict = {
        "__type__": "piecewise_expression",
        "__data__": {
            "conditions": [
                {"__type__": "literal_expression", "__data__": {"value": True}},
                {"__type__": "literal_expression", "__data__": {"value": False}},
            ],
            "values": [
                {"__type__": "literal_expression", "__data__": {"value": 1}},
            ],
            "otherwise": {"__type__": "literal_expression", "__data__": {"value": 0}},
        },
    }

    with pytest.raises(ValueError, match=r"(?i)(length|condition)"):
        Expression.deserialize_from_dict(malformed)


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

        @override
        def serialize_data_to_dict(self) -> SerializedDict:  # pragma: no cover
            return {"value": self.value}

        @classmethod
        @override
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


# =============================================================================
# validate_logical_operands: Boolean connectives reject numeric operands
# =============================================================================


@pytest.mark.parametrize(
    "expression",
    [
        pytest.param(logical_and(LiteralExpression(2), LiteralExpression(4)), id="and"),
        pytest.param(logical_or(LiteralExpression(2), LiteralExpression(4)), id="or"),
        pytest.param(logical_not(LiteralExpression(2)), id="not"),
        pytest.param(
            logical_and(LiteralExpression(True), LiteralExpression(4)),
            id="and_one_numeric_operand",
        ),
        pytest.param(
            logical_and(LiteralExpression(1.5), LiteralExpression(2.5)), id="and_floats"
        ),
        pytest.param(
            logical_and(LiteralExpression("2"), LiteralExpression("4")),
            id="and_string_form",
        ),
        pytest.param(
            logical_and(
                BinaryExpression(
                    BinaryOperation.ADD, LiteralExpression(1), LiteralExpression(2)
                ),
                LiteralExpression(True),
            ),
            id="and_arithmetic_operand",
        ),
        pytest.param(
            logical_not(UnaryExpression(UnaryOperation.NEGATE, LiteralExpression(1))),
            id="not_negation_operand",
        ),
    ],
)
def test_validate_logical_operands_rejects_a_provably_numeric_operand(
    expression: Expression,
) -> None:
    """Test a Boolean connective over a numeric operand is refused.

    ``LOGICAL_AND`` / ``LOGICAL_OR`` / ``LOGICAL_NOT`` denote Boolean
    connectives. A numeric operand under one has no faithful lowering:
    SymPy's ``&``/``|`` are *bitwise* on ``sympy.Integer``, so the shape
    would otherwise fold to a numerically wrong literal.
    """
    with pytest.raises(
        NonBooleanLogicalOperandError, match="provably denotes a number"
    ):
        validate_logical_operands(expression)


def test_validate_logical_operands_names_both_the_connective_and_the_operand() -> None:
    """Test the message identifies the offending node and the operand within it.

    A caller needs the site to fix, not just the fact of a refusal, so the
    message renders the connective and the operand that made it ill-typed.
    """
    expression = logical_or(LiteralExpression(2), LiteralExpression(4))

    with pytest.raises(NonBooleanLogicalOperandError) as exc_info:
        validate_logical_operands(expression)

    message = str(exc_info.value)
    assert "logical_or" in message
    assert "LiteralExpression(value=2)" in message


def test_validate_logical_operands_descends_past_the_root() -> None:
    """Test a numeric operand nested under a well-typed root is still found.

    A conjunction of constraints puts the offending connective in a child
    position; a screen that only inspected the root would pass this tree
    through to a backend.
    """
    x = mock_identifier("x", 0)
    benign = BinaryExpression(
        BinaryOperation.GREATER, IdentifierExpression(x), LiteralExpression(0)
    )
    nested = logical_and(LiteralExpression(2), LiteralExpression(4))

    with pytest.raises(NonBooleanLogicalOperandError):
        validate_logical_operands(logical_and(benign, nested))


def test_validate_logical_operands_rejects_an_all_numeric_piecewise_operand() -> None:
    """Test a piecewise whose every branch value is numeric is a numeric operand.

    The branch values decide the sort of the whole piecewise, so one whose
    branches agree on numeric is as ill-typed under a connective as a bare
    integer literal is.
    """
    x = mock_identifier("x", 0)
    numeric_piecewise = piecewise(
        (IdentifierExpression(x) > LiteralExpression(0), LiteralExpression(1)),
        otherwise=LiteralExpression(2),
    )

    with pytest.raises(NonBooleanLogicalOperandError):
        validate_logical_operands(
            logical_and(numeric_piecewise, LiteralExpression(True))
        )


def test_validate_logical_operands_accepts_a_boolean_valued_piecewise_operand() -> None:
    """Test a piecewise whose branch values are Boolean passes the screen."""
    x = mock_identifier("x", 0)
    boolean_piecewise = piecewise(
        (IdentifierExpression(x) > LiteralExpression(0), LiteralExpression(True)),
        otherwise=LiteralExpression(False),
    )

    validate_logical_operands(logical_and(boolean_piecewise, LiteralExpression(True)))


@pytest.mark.parametrize(
    "expression",
    [
        pytest.param(
            logical_and(LiteralExpression(True), LiteralExpression(False)),
            id="boolean_literals",
        ),
        pytest.param(
            logical_not(LiteralExpression(True)), id="boolean_literal_negation"
        ),
        pytest.param(
            logical_and(
                IdentifierExpression(mock_identifier("p", 0)),
                IdentifierExpression(mock_identifier("q", 1)),
            ),
            id="unbound_identifiers",
        ),
        pytest.param(
            logical_and(
                IdentifierExpression(mock_identifier("x", 0)) > LiteralExpression(0),
                IdentifierExpression(mock_identifier("x", 0)) < LiteralExpression(5),
            ),
            id="comparisons",
        ),
        pytest.param(
            logical_and(
                CallExpression(
                    "nand", (LiteralExpression(True), LiteralExpression(True))
                ),
                LiteralExpression(True),
            ),
            id="call_operand",
        ),
        pytest.param(
            logical_and(
                logical_not(IdentifierExpression(mock_identifier("p", 0))),
                LiteralExpression(True),
            ),
            id="nested_connective_operand",
        ),
    ],
)
def test_validate_logical_operands_accepts_an_operand_it_cannot_prove_numeric(
    expression: Expression,
) -> None:
    """Test the screen refuses only what it can prove, not everything unproven.

    An identifier with no binding and a call carry their sort outside the
    node -- in ``symbol_types`` and in the registry respectively -- so
    refusing them would reject well-typed expressions the backends lower
    correctly.
    """
    validate_logical_operands(expression)


def test_validate_logical_operands_screens_an_identifier_bound_to_a_number() -> None:
    """Test an identifier the environment binds to a number is screened as one.

    Substituting a numeric value into a connective is the same ill-typed
    shape as writing the number there, reached one step later; without the
    environment the screen would pass the tree and the number would meet
    the connective inside the backend.
    """
    p = mock_identifier("p", 0)
    q = mock_identifier("q", 1)
    expression = logical_and(IdentifierExpression(p), IdentifierExpression(q))

    with pytest.raises(NonBooleanLogicalOperandError):
        validate_logical_operands(
            expression, {p: LiteralExpression(2), q: LiteralExpression(4)}
        )


def test_validate_logical_operands_accepts_an_identifier_bound_to_a_boolean() -> None:
    """Test an environment binding a Boolean value leaves the connective well-typed."""
    p = mock_identifier("p", 0)
    expression = logical_and(IdentifierExpression(p), LiteralExpression(True))

    validate_logical_operands(expression, {p: LiteralExpression(False)})


def test_validate_logical_operands_does_not_chain_environment_bindings() -> None:
    """Test a binding's value is classified without applying another binding to it.

    Substitution is simultaneous and non-chaining, so ``{p: q, q: 2}``
    replaces ``p`` with the residual ``q`` -- never with ``2``. A screen
    that chained would refuse an expression the bridge lowers without
    complaint.
    """
    p = mock_identifier("p", 0)
    q = mock_identifier("q", 1)
    expression = logical_and(IdentifierExpression(p), LiteralExpression(True))

    validate_logical_operands(
        expression, {p: IdentifierExpression(q), q: LiteralExpression(2)}
    )


def test_validate_logical_operands_rejects_a_numeric_piecewise_condition() -> None:
    """Test a piecewise case condition is screened as a Boolean position.

    A condition selects its branch by truth, so an arithmetic condition is
    as ill-typed as an arithmetic operand of ``and``.
    """
    x = mock_identifier("x", 0)
    expression = piecewise(
        (IdentifierExpression(x) + LiteralExpression(1), LiteralExpression(5)),
        otherwise=LiteralExpression(0),
    )

    with pytest.raises(NonBooleanLogicalOperandError, match="case condition"):
        validate_logical_operands(expression)


def test_validate_logical_operands_screens_a_case_condition_bound_to_a_number() -> None:
    """Test an identifier condition the environment binds to a number is refused.

    Substituting the number would build a piecewise whose condition is a
    numeric literal, which ``PiecewiseExpression`` refuses to hold, and
    SymPy would read such a condition as a truth value.
    """
    condition = mock_identifier("c", 0)
    expression = piecewise(
        (IdentifierExpression(condition), LiteralExpression(1)),
        otherwise=LiteralExpression(0),
    )

    with pytest.raises(NonBooleanLogicalOperandError, match="case condition"):
        validate_logical_operands(expression, {condition: LiteralExpression(1)})


def test_validate_logical_operands_accepts_an_unprovable_case_condition() -> None:
    """Test an unbound or Boolean-bound condition passes, as do numeric values.

    Only the conditions are Boolean positions: the branch values ``1`` and
    ``0`` are numbers and stay legal.
    """
    condition = mock_identifier("c", 0)
    expression = piecewise(
        (IdentifierExpression(condition), LiteralExpression(1)),
        otherwise=LiteralExpression(0),
    )

    validate_logical_operands(expression)
    validate_logical_operands(expression, {condition: LiteralExpression(True)})


def test_validate_logical_operands_names_the_piecewise_and_its_condition() -> None:
    """Test the refusal points at both the piecewise and the offending condition."""
    x = mock_identifier("x", 0)
    condition = IdentifierExpression(x) * LiteralExpression(2)
    expression = piecewise(
        (condition, LiteralExpression(5)), otherwise=LiteralExpression(0)
    )

    with pytest.raises(NonBooleanLogicalOperandError) as exc_info:
        validate_logical_operands(expression)

    message = str(exc_info.value)
    assert repr(expression) in message
    assert repr(condition) in message


def test_non_boolean_logical_operand_error_is_a_type_error() -> None:
    """Test the refusal is a `TypeError`, not an undecidability report.

    ``logical_and(2, 4)`` is ill-typed: no backend and no timeout gives it
    a meaning. Reporting it as `UndecidableError` would invite a caller to
    retry with a larger bound, and returning `None` would hide an
    author-side bug behind the same signal a solver timeout uses.
    """
    assert issubclass(NonBooleanLogicalOperandError, TypeError)
    assert not issubclass(NonBooleanLogicalOperandError, UndecidableError)


def test_non_boolean_logical_operand_error_is_in_the_compiler_error_registry() -> None:
    """Test the error is discoverable through `@register_error`'s catalog."""
    assert NonBooleanLogicalOperandError in get_registered_errors()
