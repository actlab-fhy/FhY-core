"""Tests for `fhy_core.symbolic.expression.pprint`.

The printed text is the Rust core's (decision D-S4-1): a literal as the
core writes it (``true``, ``1``, ``NaN``), a connective as one n-ary node,
and each operation's functional name its Rust name.
"""

import math
from decimal import Decimal

import pytest

from fhy_core.identifier import Identifier
from fhy_core.pass_infrastructure import PassExecutionError
from fhy_core.symbolic.expression import (
    BinaryExpression,
    BinaryOperation,
    CallExpression,
    Expression,
    IdentifierExpression,
    LiteralExpression,
    LogicalExpression,
    LogicalOperation,
    PiecewiseExpression,
    UnaryExpression,
    UnaryOperation,
    logical_and,
    logical_or,
    pformat_expression,
)
from fhy_core.symbolic.expression.pprint import ExpressionPrettyFormatter

# =============================================================================
# Symbolic format (default)
# =============================================================================


@pytest.mark.parametrize(
    "expression, expected_str",
    [
        (LiteralExpression(4.5), "4.5"),
        (LiteralExpression("0.1"), "0.1"),
        (IdentifierExpression(Identifier("baz")), "baz"),
        (
            UnaryExpression(UnaryOperation.LOGICAL_NOT, LiteralExpression(True)),
            "(!true)",
        ),
        (
            BinaryExpression(
                BinaryOperation.MULTIPLY,
                LiteralExpression("3.14"),
                LiteralExpression(10.5),
            ),
            "(3.14 * 10.5)",
        ),
    ],
)
def test_pformat_expression_renders_symbolic_form(
    expression: Expression, expected_str: str
) -> None:
    """Test `pformat_expression` emits the expected symbolic-notation string."""
    assert pformat_expression(expression) == expected_str


@pytest.mark.parametrize(
    ("value", "expected_text"),
    [
        (1e300, "1e300"),
        (5e-324, "5e-324"),
        (1.7976931348623157e308, "1.7976931348623157e308"),
        (1e16, "1e16"),
        (1e-5, "0.00001"),
        (9.9e-6, "9.9e-6"),
    ],
)
def test_str_of_an_extreme_float_literal_is_its_canonical_text(
    value: float, expected_text: str
) -> None:
    """Test ``str`` writes a float outside ``[1e-5, 1e16)`` with an exponent."""
    assert str(LiteralExpression(value)) == expected_text


@pytest.mark.parametrize(
    "value, expected_text",
    [
        pytest.param(True, "true", id="true"),
        pytest.param(False, "false", id="false"),
        pytest.param(5, "5", id="int"),
        pytest.param(-3, "-3", id="negative_int"),
        pytest.param(10**30, "1" + "0" * 30, id="big_int"),
        pytest.param(1.0, "1", id="integral_float"),
        pytest.param(1.5, "1.5", id="float"),
        pytest.param(-0.0, "-0", id="negative_zero"),
        pytest.param(1e16, "1e16", id="large_float"),
        pytest.param(
            9_999_999_999_999_998.0, "9999999999999998", id="positional_float"
        ),
        pytest.param(1e-5, "0.00001", id="smallest_positional_float"),
        pytest.param(1e-7, "1e-7", id="small_float"),
        pytest.param(1e300, "1e300", id="huge_float"),
        pytest.param(5e-324, "5e-324", id="subnormal_float"),
        pytest.param(math.nan, "NaN", id="nan"),
        pytest.param(math.inf, "inf", id="infinity"),
        pytest.param(-math.inf, "-inf", id="negative_infinity"),
        pytest.param("05", "5", id="integer_text"),
        pytest.param("1.50", "1.5", id="decimal_text"),
        pytest.param("100.0", "100", id="integral_decimal_text"),
        pytest.param(".5", "0.5", id="fraction_text"),
        pytest.param(Decimal("2.50"), "2.5", id="decimal"),
    ],
)
def test_pformat_literal_renders_the_core_text(
    value: bool | int | float | str | Decimal, expected_text: str
) -> None:
    """Test a literal renders as the Rust core writes its normalized value.

    Unequal literals may render alike: ``1``, ``1.0`` and ``Decimal("1")``
    all render as ``1``.
    """
    literal = LiteralExpression(value)

    assert pformat_expression(literal) == expected_text
    assert pformat_expression(literal, functional=True) == expected_text
    assert str(literal) == expected_text


@pytest.mark.parametrize(
    "build, expected_symbolic, expected_functional",
    [
        pytest.param(logical_and, "(p && q && r)", "(and p q r)", id="conjunction"),
        pytest.param(logical_or, "(p || q || r)", "(or p q r)", id="disjunction"),
    ],
)
def test_pformat_logical_expression_renders_one_n_ary_node(
    build: object, expected_symbolic: str, expected_functional: str
) -> None:
    """Test a connective renders its operands joined by one symbol, or its name."""
    p, q, r = (IdentifierExpression(Identifier(name)) for name in "pqr")
    expression = build(p, q, r)  # type: ignore[operator]

    assert pformat_expression(expression) == expected_symbolic
    assert pformat_expression(expression, functional=True) == expected_functional


def test_pformat_nested_logical_expression_keeps_the_nesting() -> None:
    """Test a nested connective renders as its own parenthesized node."""
    p, q, r = (IdentifierExpression(Identifier(name)) for name in "pqr")
    expression = LogicalExpression(
        LogicalOperation.AND, (LogicalExpression(LogicalOperation.OR, (p, q)), r)
    )

    assert pformat_expression(expression) == "((p || q) && r)"


def test_pformat_modulo_renders_its_rust_name_functionally() -> None:
    """Test ``MODULO`` renders as ``%`` and, functionally, as ``floor_mod``."""
    x = IdentifierExpression(Identifier("x"))

    assert pformat_expression(x % 3) == "(x % 3)"
    assert pformat_expression(x % 3, functional=True) == "(floor_mod x 3)"


def test_str_of_an_expression_is_its_symbolic_text() -> None:
    """Test ``str`` is the core's display, ``pformat_expression``'s default."""
    x = IdentifierExpression(Identifier("x"))
    expression = piecewise_of(x)

    assert str(expression) == pformat_expression(expression)
    assert str(expression) == "{1 if (x > 0); -1 otherwise}"


def test_repr_of_an_expression_is_its_class_and_functional_text_with_ids() -> None:
    """Test ``repr`` is the class name around the core's bounded diagnostic text.

    The diagnostic text is the functional notation with identifier ids.
    """
    identifier = Identifier("x")
    x = IdentifierExpression(identifier)

    assert repr(x + 1) == f"BinaryExpression((add x::{identifier.id} 1))"
    assert repr(LiteralExpression(1.5)) == "LiteralExpression(1.5)"
    assert repr(x) == f"IdentifierExpression(x::{identifier.id})"
    assert repr(logical_and(x > 0, x < 1)) == (
        f"LogicalExpression((and (greater x::{identifier.id} 0) "
        f"(less x::{identifier.id} 1)))"
    )


def test_repr_of_a_very_large_expression_is_bounded() -> None:
    """Test ``repr`` of a DAG with an exponential tree elides past 1,000 nodes."""
    node: Expression = IdentifierExpression(Identifier("x"))
    for _ in range(40):
        node = node + node

    text = repr(node)

    assert text.startswith("BinaryExpression((add ")
    assert ".." in text
    assert len(text) < 20_000


def piecewise_of(x: Expression) -> Expression:
    """Return ``{1 if x > 0; -1 otherwise}``."""
    return PiecewiseExpression(
        (x > LiteralExpression(0),), (LiteralExpression(1),), LiteralExpression(-1)
    )


# =============================================================================
# Functional format
# =============================================================================


@pytest.mark.parametrize(
    "expression, expected_str",
    [
        (LiteralExpression(5), "5"),
        (
            IdentifierExpression(Identifier("test_identifier")),
            "test_identifier",
        ),
        (
            UnaryExpression(UnaryOperation.NEGATE, LiteralExpression(5)),
            "(negate 5)",
        ),
        (
            BinaryExpression(
                BinaryOperation.ADD,
                LiteralExpression(5),
                LiteralExpression(10),
            ),
            "(add 5 10)",
        ),
        (
            BinaryExpression(
                BinaryOperation.DIVIDE,
                UnaryExpression(UnaryOperation.NEGATE, LiteralExpression(5)),
                LiteralExpression(10),
            ),
            "(divide (negate 5) 10)",
        ),
    ],
)
def test_pformat_expression_renders_functional_form(
    expression: Expression, expected_str: str
) -> None:
    """Test `pformat_expression(..., functional=True)` emits functional-notation."""
    assert pformat_expression(expression, functional=True) == expected_str


# =============================================================================
# Identifier-id visibility
# =============================================================================


def test_pformat_expression_with_show_id_includes_name_hint_and_id() -> None:
    """Test `pformat_expression(..., show_id=True)` renders both name hint and id."""
    identifier = Identifier("foo")
    result = pformat_expression(IdentifierExpression(identifier), show_id=True)
    assert identifier.name_hint in result
    assert str(identifier.id) in result


def test_pformat_expression_with_show_id_propagates_into_piecewise() -> None:
    """Test ``show_id=True`` reaches identifiers nested in a ``PiecewiseExpression``."""
    condition_identifier = Identifier("cond")
    expression = PiecewiseExpression(
        (IdentifierExpression(condition_identifier),),
        (LiteralExpression(1),),
        LiteralExpression(0),
    )

    result = pformat_expression(expression, show_id=True)

    assert repr(condition_identifier) in result


def test_pformat_expression_with_show_id_propagates_into_call_arguments() -> None:
    """Test ``show_id=True`` reaches identifiers nested in call arguments."""
    argument_identifier = Identifier("arg")
    expression = CallExpression(
        "nested_show_id", (IdentifierExpression(argument_identifier),)
    )

    result = pformat_expression(expression, show_id=True)

    assert argument_identifier.name_hint in result
    assert str(argument_identifier.id) in result


# =============================================================================
# ExpressionPrettyFormatter defaults
# =============================================================================


def test_pretty_formatter_default_does_not_show_identifier_id() -> None:
    """Test `ExpressionPrettyFormatter()` defaults render identifiers without an id."""
    identifier = Identifier("name_only")
    result = ExpressionPrettyFormatter()(IdentifierExpression(identifier))
    assert result == identifier.name_hint


def test_pretty_formatter_default_uses_symbolic_notation() -> None:
    """Test `ExpressionPrettyFormatter()` defaults render binary ops symbolically."""
    expression = BinaryExpression(
        BinaryOperation.ADD, LiteralExpression(1), LiteralExpression(2)
    )
    assert ExpressionPrettyFormatter()(expression) == "(1 + 2)"


# =============================================================================
# Defensive guards
# =============================================================================


def test_pretty_formatter_refuses_a_subclass_defining_a_visitor() -> None:
    """Test a formatter subclass defining a ``visit_*`` method is refused.

    The core renders the text, with no per-node hook, so an override that
    would be ignored is refused when the class is created.
    """
    with pytest.raises(
        TypeError,
        match=r"^_NonStringFormatter defines visit_literal_expression, but "
        r"ExpressionPrettyFormatter renders through the Rust core and has no "
        r"per-node hooks\.$",
    ):

        class _NonStringFormatter(ExpressionPrettyFormatter):
            def visit_literal_expression(
                self, literal_expression: LiteralExpression
            ) -> str:
                return "42"


def test_pretty_formatter_get_noop_output_raises() -> None:
    """Test `ExpressionPrettyFormatter.get_noop_output` raises `PassExecutionError`."""
    formatter = ExpressionPrettyFormatter()

    with pytest.raises(PassExecutionError, match=r"does not define noop output"):
        formatter.get_noop_output(LiteralExpression(0))


# =============================================================================
# PiecewiseExpression rendering
# =============================================================================


def test_pformat_single_case_piecewise_renders_symbolic_case_form() -> None:
    """Test a one-case ``PiecewiseExpression`` renders as ``{v if c; o otherwise}``."""
    expression = PiecewiseExpression(
        (LiteralExpression(True),), (LiteralExpression(1),), LiteralExpression(2)
    )

    assert pformat_expression(expression) == "{1 if true; 2 otherwise}"


def test_pformat_multi_case_piecewise_renders_all_cases_then_otherwise() -> None:
    """Test a multi-case piecewise renders every case, then ``otherwise``."""
    expression = PiecewiseExpression(
        (LiteralExpression(True), LiteralExpression(False)),
        (LiteralExpression(1), LiteralExpression(2)),
        LiteralExpression(3),
    )

    assert pformat_expression(expression) == "{1 if true; 2 if false; 3 otherwise}"


def test_pformat_piecewise_expression_renders_in_functional_form() -> None:
    """Test piecewise renders as ``(piecewise c1 v1 ... o)`` in functional form."""
    expression = PiecewiseExpression(
        (LiteralExpression(True),), (LiteralExpression(1),), LiteralExpression(2)
    )

    assert pformat_expression(expression, functional=True) == "(piecewise true 1 2)"


def test_pformat_multi_case_piecewise_renders_functional_form_with_odd_arity() -> None:
    """Test the functional form lists every case pair, then ``otherwise``."""
    expression = PiecewiseExpression(
        (LiteralExpression(True), LiteralExpression(False)),
        (LiteralExpression(1), LiteralExpression(2)),
        LiteralExpression(3),
    )

    assert (
        pformat_expression(expression, functional=True)
        == "(piecewise true 1 false 2 3)"
    )


def test_pformat_piecewise_with_over_one_hundred_cases_renders_all_in_order() -> None:
    """Test a 120-case piecewise renders every case, in order, in both the
    symbolic and functional forms.
    """
    NUM_CASES = 120
    x = Identifier("x")
    x_expression = IdentifierExpression(x)
    cases = tuple(
        (x_expression.equals(i), LiteralExpression(i)) for i in range(NUM_CASES)
    )
    expression = PiecewiseExpression(
        tuple(condition for condition, _ in cases),
        tuple(value for _, value in cases),
        LiteralExpression(-1),
    )

    symbolic = pformat_expression(expression)
    expected_symbolic = (
        "{"
        + "; ".join(f"{i} if (x == {i})" for i in range(NUM_CASES))
        + "; -1 otherwise}"
    )
    assert symbolic == expected_symbolic

    functional = pformat_expression(expression, functional=True)
    expected_functional_parts: list[str] = []
    for i in range(NUM_CASES):
        expected_functional_parts.append(f"(equal x {i})")
        expected_functional_parts.append(str(i))
    expected_functional_parts.append("-1")
    expected_functional = f"(piecewise {' '.join(expected_functional_parts)})"
    assert functional == expected_functional


def test_pformat_nested_piecewise_renders_inner_form_inside_branch() -> None:
    """Test a nested piecewise renders with the inner form inside a branch."""
    expression = PiecewiseExpression(
        (LiteralExpression(True),),
        (
            PiecewiseExpression(
                (LiteralExpression(False),),
                (LiteralExpression(1),),
                LiteralExpression(2),
            ),
        ),
        LiteralExpression(3),
    )

    assert (
        pformat_expression(expression)
        == "{{1 if false; 2 otherwise} if true; 3 otherwise}"
    )


# =============================================================================
# CallExpression rendering
# =============================================================================


def test_pformat_call_expression_renders_function_name_with_arguments() -> None:
    """Test ``call(name, args)`` renders as ``name(arg1, arg2, ...)``."""
    expression = CallExpression("max", (LiteralExpression(1), LiteralExpression(2)))

    assert pformat_expression(expression) == "max(1, 2)"


def test_pformat_call_expression_with_zero_arguments_renders_with_empty_parens() -> (
    None
):
    """Test a zero-argument ``CallExpression`` renders as ``name()``."""
    expression = CallExpression("noargs", ())

    assert pformat_expression(expression) == "noargs()"


def test_pformat_call_expression_renders_in_functional_form() -> None:
    """Test ``CallExpression`` renders as ``(name arg1 arg2 ...)`` functionally."""
    expression = CallExpression("max", (LiteralExpression(1), LiteralExpression(2)))

    assert pformat_expression(expression, functional=True) == "(max 1 2)"


def test_pformat_call_expression_renders_nested_call_in_argument_position() -> None:
    """Test nested ``CallExpression`` renders with the inner call inside an argument."""
    expression = CallExpression(
        "max",
        (
            CallExpression("min", (LiteralExpression(1), LiteralExpression(2))),
            LiteralExpression(3),
        ),
    )

    assert pformat_expression(expression) == "max(min(1, 2), 3)"


# =============================================================================
# ExpressionPrettyFormatter renders the core's text
# =============================================================================


@pytest.mark.parametrize("show_id", [False, True])
@pytest.mark.parametrize("functional", [False, True])
def test_pretty_formatter_renders_what_pformat_expression_renders(
    show_id: bool, functional: bool
) -> None:
    """Test the Python formatter and the core render every node kind alike."""
    x = IdentifierExpression(Identifier("x"))
    expression = PiecewiseExpression(
        (logical_and(x > 0, -x < LiteralExpression(Decimal("1.5"))),),
        (CallExpression("max", (x % 3, LiteralExpression(True))),),
        logical_or(
            LiteralExpression(False), UnaryExpression(UnaryOperation.POSITIVE, x)
        ),
    )

    formatted = ExpressionPrettyFormatter(
        is_id_shown=show_id, is_printed_functional=functional
    )(expression)

    assert formatted == pformat_expression(
        expression, show_id=show_id, functional=functional
    )


def test_pretty_formatter_subclass_without_visitors_formats_as_the_core() -> None:
    """Test a formatter subclass that defines no visitor formats the core's text."""

    class _Formatter(ExpressionPrettyFormatter):
        pass

    x = IdentifierExpression(Identifier("x"))

    assert _Formatter()(x + LiteralExpression(True)) == "(x + true)"


def test_pformat_expression_refuses_a_value_that_is_no_expression() -> None:
    """Test ``pformat_expression`` raises ``TypeError`` for a non-expression."""
    with pytest.raises(TypeError, match="takes an Expression, got int"):
        pformat_expression(5)  # type: ignore[arg-type]
