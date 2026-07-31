"""Tests for `fhy_core.symbolic.expression.pprint`."""

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
    PiecewiseExpression,
    UnaryExpression,
    UnaryOperation,
    pformat_expression,
)
from fhy_core.symbolic.expression.pprint import ExpressionPrettyFormatter
from fhy_core.utils.override import override

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
            "(!True)",
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


# =============================================================================
# Functional format
# =============================================================================


@pytest.mark.parametrize(
    "expression, expected_str",
    [
        (LiteralExpression(5), "5"),
        (IdentifierExpression(Identifier("test_identifier")), "test_identifier"),
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

    assert condition_identifier.name_hint in result
    assert str(condition_identifier.id) in result


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


def test_pretty_formatter_call_rejects_non_string_formatted_result() -> None:
    """Test `__call__` raises `TypeError` when the visit result is not a `str`."""

    class _NonStringFormatter(ExpressionPrettyFormatter):
        @override
        def visit_literal_expression(
            self, literal_expression: LiteralExpression
        ) -> str:
            return 42  # type: ignore[return-value]  # intentional contract violation

    with pytest.raises(TypeError, match=r"Invalid formatted expression type"):
        _NonStringFormatter()(LiteralExpression(5))


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

    assert pformat_expression(expression) == "{1 if True; 2 otherwise}"


def test_pformat_multi_case_piecewise_renders_all_cases_then_otherwise() -> None:
    """Test a multi-case piecewise renders every case, then ``otherwise``."""
    expression = PiecewiseExpression(
        (LiteralExpression(True), LiteralExpression(False)),
        (LiteralExpression(1), LiteralExpression(2)),
        LiteralExpression(3),
    )

    assert pformat_expression(expression) == "{1 if True; 2 if False; 3 otherwise}"


def test_pformat_piecewise_expression_renders_in_functional_form() -> None:
    """Test piecewise renders as ``(piecewise c1 v1 ... o)`` in functional form."""
    expression = PiecewiseExpression(
        (LiteralExpression(True),), (LiteralExpression(1),), LiteralExpression(2)
    )

    assert pformat_expression(expression, functional=True) == "(piecewise True 1 2)"


def test_pformat_multi_case_piecewise_renders_functional_form_with_odd_arity() -> None:
    """Test the functional form lists every case pair, then ``otherwise``."""
    expression = PiecewiseExpression(
        (LiteralExpression(True), LiteralExpression(False)),
        (LiteralExpression(1), LiteralExpression(2)),
        LiteralExpression(3),
    )

    assert (
        pformat_expression(expression, functional=True)
        == "(piecewise True 1 False 2 3)"
    )


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
        == "{{1 if False; 2 otherwise} if True; 3 otherwise}"
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
