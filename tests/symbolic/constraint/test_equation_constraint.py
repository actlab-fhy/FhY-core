"""Behavioral tests for `EquationConstraint`."""

import logging
from typing import Any

import pytest

from fhy_core.symbolic.constraint import (
    ConstraintError,
    ConstraintOutcome,
    EquationConstraint,
)
from fhy_core.symbolic.expression import (
    BinaryExpression,
    BinaryOperation,
    IdentifierExpression,
    LiteralExpression,
    UnaryExpression,
    UnaryOperation,
    call,
    logical_and,
    make_binary_expression,
    pformat_expression,
)

from .conftest import mock_identifier

_CONSTRAINT_LOGGER = "fhy_core.symbolic.constraint"


def _find_records(
    caplog: pytest.LogCaptureFixture, level: int
) -> list[logging.LogRecord]:
    """Return the constraint module's records emitted at exactly ``level``."""
    return [
        record
        for record in caplog.records
        if record.levelno == level and record.name == _CONSTRAINT_LOGGER
    ]


# =============================================================================
# Constructor: `expression` attribute and rejection of non-`Expression` input
# =============================================================================


def test_constructor_stores_the_expression() -> None:
    """Test the constructor argument is reflected on the `expression` attribute."""
    expression = LiteralExpression(True)

    constraint = EquationConstraint(expression)

    assert constraint.expression is expression


@pytest.mark.parametrize(
    "non_expression",
    [
        pytest.param(True, id="bool"),
        pytest.param(1, id="int"),
        pytest.param(1.5, id="float"),
        pytest.param("x == 1", id="str"),
        pytest.param(None, id="none"),
        pytest.param([LiteralExpression(True)], id="list_of_expression"),
    ],
)
def test_constructor_rejects_non_expression_input(non_expression: Any) -> None:
    """Test a non-`Expression` constructor argument raises `ConstraintError`."""
    with pytest.raises(ConstraintError):
        EquationConstraint(non_expression)


def test_constructor_rejects_the_equality_operator_footgun() -> None:
    """Test `IdentifierExpression(x) == IdentifierExpression(y)` is rejected.

    `Expression` deliberately does not override `__eq__`: comparison
    dunders return `BinaryExpression` IR nodes for `<`/`<=`/`>`/`>=`, but
    `==` falls back to identity comparison (`eq=False` dataclasses) and
    evaluates to a plain `bool`. A caller who reaches for `==` expecting
    to build an equality constraint gets a `bool` that the constructor
    must catch here, at the source, rather than silently accepting.
    """
    x = mock_identifier("x", 0)
    y = mock_identifier("y", 1)
    footgun_result = IdentifierExpression(x) == IdentifierExpression(y)
    assert isinstance(footgun_result, bool)

    with pytest.raises(ConstraintError):
        EquationConstraint(footgun_result)  # type: ignore[arg-type]


# =============================================================================
# `get_free_identifiers`: the scope is exactly the expression's free identifiers
# =============================================================================


def test_get_free_identifiers_ground_expression_is_empty() -> None:
    """Test a ground (variable-free) expression has an empty scope."""
    constraint = EquationConstraint(LiteralExpression(True))

    assert constraint.get_free_identifiers() == frozenset()


def test_get_free_identifiers_single_identifier() -> None:
    """Test the scope is exactly the expression's one free identifier."""
    x = mock_identifier("x", 0)
    constraint = EquationConstraint(IdentifierExpression(x))

    assert constraint.get_free_identifiers() == frozenset({x})


def test_get_free_identifiers_multiple_identifiers() -> None:
    """Test the scope unions every free identifier of a multi-variable expression."""
    x = mock_identifier("x", 0)
    y = mock_identifier("y", 1)
    constraint = EquationConstraint(make_binary_expression(BinaryOperation.LESS, x, y))

    assert constraint.get_free_identifiers() == frozenset({x, y})


# =============================================================================
# `convert_to_expression`
# =============================================================================


def test_convert_to_expression_returns_the_wrapped_expression_unchanged() -> None:
    """Test `convert_to_expression` returns the constructor's expression unchanged."""
    expression = make_binary_expression(
        BinaryOperation.EQUAL, mock_identifier("x", 0), 1
    )
    constraint = EquationConstraint(expression)

    assert constraint.convert_to_expression() is expression


# =============================================================================
# Tri-state `evaluate_with_bindings`
# =============================================================================


def test_evaluate_with_bindings_ground_expression_decidable_under_empty_bindings() -> (
    None
):
    """Test a ground expression is decidable with no bindings at all."""
    constraint = EquationConstraint(LiteralExpression(True))

    assert constraint.evaluate_with_bindings({}) is ConstraintOutcome.SATISFIED


def test_evaluate_with_bindings_ground_false_expression_is_violated() -> None:
    """Test a ground `False`-valued expression is decidably VIOLATED."""
    constraint = EquationConstraint(LiteralExpression(False))

    assert constraint.evaluate_with_bindings({}) is ConstraintOutcome.VIOLATED


def test_evaluate_with_bindings_full_assignment_satisfied() -> None:
    """Test a full multi-variable assignment that holds reports SATISFIED."""
    x = mock_identifier("x", 0)
    y = mock_identifier("y", 1)
    expression = make_binary_expression(
        BinaryOperation.LESS, make_binary_expression(BinaryOperation.ADD, x, y), 10
    )
    constraint = EquationConstraint(expression)

    outcome = constraint.evaluate_with_bindings({x: 3, y: 5})

    assert outcome is ConstraintOutcome.SATISFIED


def test_evaluate_with_bindings_full_assignment_violated() -> None:
    """Test a full multi-variable assignment that fails reports VIOLATED."""
    x = mock_identifier("x", 0)
    y = mock_identifier("y", 1)
    expression = make_binary_expression(
        BinaryOperation.LESS, make_binary_expression(BinaryOperation.ADD, x, y), 10
    )
    constraint = EquationConstraint(expression)

    outcome = constraint.evaluate_with_bindings({x: 20, y: 1})

    assert outcome is ConstraintOutcome.VIOLATED


def test_evaluate_with_bindings_partial_assignment_is_undecided() -> None:
    """Test binding only one of two free identifiers is UNDECIDED."""
    x = mock_identifier("x", 0)
    y = mock_identifier("y", 1)
    expression = make_binary_expression(
        BinaryOperation.LESS, make_binary_expression(BinaryOperation.ADD, x, y), 10
    )
    constraint = EquationConstraint(expression)

    outcome = constraint.evaluate_with_bindings({x: 3})

    assert outcome is ConstraintOutcome.UNDECIDED


def test_evaluate_with_bindings_empty_bindings_undecided_for_open_expression() -> None:
    """Test empty bindings is UNDECIDED for an expression with free identifiers."""
    x = mock_identifier("x", 0)
    constraint = EquationConstraint(IdentifierExpression(x))

    outcome = constraint.evaluate_with_bindings({})

    assert outcome is ConstraintOutcome.UNDECIDED


def test_evaluate_with_bindings_non_bool_literal_reduction_is_violated() -> None:
    """Test a substituted expression reducing to a non-bool literal is VIOLATED."""
    constraint = EquationConstraint(LiteralExpression(1))

    assert constraint.evaluate_with_bindings({}) is ConstraintOutcome.VIOLATED


def test_evaluate_with_bindings_symbolic_binding_can_decide() -> None:
    """Test a symbolic (non-literal) binding can still decide the outcome."""
    x = mock_identifier("x", 0)
    y = mock_identifier("y", 1)
    expression = make_binary_expression(BinaryOperation.GREATER, x, y)
    constraint = EquationConstraint(expression)
    successor_of_y = make_binary_expression(
        BinaryOperation.ADD, IdentifierExpression(y), 1
    )

    outcome = constraint.evaluate_with_bindings({x: successor_of_y})

    assert outcome is ConstraintOutcome.SATISFIED


def test_evaluate_with_bindings_chained_assignment_is_undecided() -> None:
    """Test a chained binding leaves a residual instead of folding through it.

    ``{x: y, y: 5}`` must not be applied sequentially (``x -> y -> 5``,
    folding ``x < 5`` to the literal ``False``/VIOLATED); simultaneous
    substitution leaves the residual ``y < 5``, which is UNDECIDED because
    ``y`` remains free.
    """
    x = mock_identifier("x", 0)
    y = mock_identifier("y", 1)
    constraint = EquationConstraint(make_binary_expression(BinaryOperation.LESS, x, 5))

    outcome = constraint.evaluate_with_bindings({x: IdentifierExpression(y), y: 5})

    assert outcome is ConstraintOutcome.UNDECIDED


def test_evaluate_with_bindings_swap_assignment_is_undecided_not_violated() -> None:
    """Test a swap binding on `x < y` is UNDECIDED, not VIOLATED.

    Sequential substitution would resolve `x < y` to `y < y` (VIOLATED);
    simultaneous substitution swaps the identifiers instead, leaving an
    undecided residual comparing two distinct identifiers.
    """
    x = mock_identifier("x", 0)
    y = mock_identifier("y", 1)
    constraint = EquationConstraint(make_binary_expression(BinaryOperation.LESS, x, y))

    outcome = constraint.evaluate_with_bindings(
        {x: IdentifierExpression(y), y: IdentifierExpression(x)}
    )

    assert outcome is ConstraintOutcome.UNDECIDED


def test_evaluate_with_bindings_ignores_extraneous_keys() -> None:
    """Test bindings for identifiers outside the expression do not affect the result."""
    z = mock_identifier("z", 2)
    constraint = EquationConstraint(LiteralExpression(True))

    outcome = constraint.evaluate_with_bindings({z: 999})

    assert outcome is ConstraintOutcome.SATISFIED


# =============================================================================
# DEBUG-vs-WARNING logging split
# =============================================================================


def test_evaluate_with_bindings_logs_debug_when_free_identifiers_unbound(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Test a partial (expected) UNDECIDED case logs at DEBUG, not WARNING."""
    x = mock_identifier("x", 0)
    y = mock_identifier("y", 1)
    expression = make_binary_expression(
        BinaryOperation.LESS, make_binary_expression(BinaryOperation.ADD, x, y), 10
    )
    constraint = EquationConstraint(expression)

    with caplog.at_level(logging.DEBUG, logger=_CONSTRAINT_LOGGER):
        outcome = constraint.evaluate_with_bindings({x: 3})

    assert outcome is ConstraintOutcome.UNDECIDED
    assert _find_records(caplog, logging.DEBUG)
    assert not _find_records(caplog, logging.WARNING)


def test_evaluate_with_bindings_logs_debug_for_symbolic_residual(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Test a symbolic binding that leaves the residual open logs at DEBUG.

    Every free identifier of the *original* expression is bound here, but
    the bound value is itself a symbolic `Expression` referencing a new
    identifier, so the substituted-and-simplified residual still has a
    free identifier. This is ordinary partial evaluation (DEBUG), not the
    fully-grounded anomaly case (WARNING).
    """
    x = mock_identifier("x", 0)
    w = mock_identifier("w", 1)
    constraint = EquationConstraint(make_binary_expression(BinaryOperation.LESS, x, 10))

    with caplog.at_level(logging.DEBUG, logger=_CONSTRAINT_LOGGER):
        outcome = constraint.evaluate_with_bindings({x: IdentifierExpression(w)})

    assert outcome is ConstraintOutcome.UNDECIDED
    assert _find_records(caplog, logging.DEBUG)
    assert not _find_records(caplog, logging.WARNING)


def test_evaluate_with_bindings_logs_warning_when_fully_bound_but_irreducible(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Test a fully bound, fully closed, yet irreducible residual logs at WARNING.

    ``arcsin`` is not registered as a native-constant-foldable function for
    an out-of-domain argument, so the simplifier returns an unevaluated
    `CallExpression` with no free identifiers at all: every free identifier
    was bound, but the simplifier still failed to reduce the residual to a
    literal -- the genuine anomaly case.
    """
    y = mock_identifier("y", 1)
    constraint = EquationConstraint(call("arcsin", IdentifierExpression(y)))

    with caplog.at_level(logging.DEBUG, logger=_CONSTRAINT_LOGGER):
        outcome = constraint.evaluate_with_bindings({y: 2})

    assert outcome is ConstraintOutcome.UNDECIDED
    assert _find_records(caplog, logging.WARNING)
    assert not _find_records(caplog, logging.DEBUG)


def test_evaluate_with_bindings_logs_nothing_when_decided(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Test a decided outcome emits no record at any level."""
    constraint = EquationConstraint(LiteralExpression(True))

    with caplog.at_level(logging.DEBUG, logger=_CONSTRAINT_LOGGER):
        outcome = constraint.evaluate_with_bindings({})

    assert outcome is ConstraintOutcome.SATISFIED
    assert not _find_records(caplog, logging.DEBUG)
    assert not _find_records(caplog, logging.WARNING)


# =============================================================================
# `is_satisfied_with_bindings`
# =============================================================================


def test_is_satisfied_with_bindings_folds_undecided_to_false() -> None:
    """Test an UNDECIDED bindings outcome maps to `False`."""
    y = mock_identifier("y", 1)
    constraint = EquationConstraint(IdentifierExpression(y))

    assert constraint.evaluate_with_bindings({}) is ConstraintOutcome.UNDECIDED
    assert constraint.is_satisfied_with_bindings({}) is False


def test_is_satisfied_with_bindings_true_when_satisfied() -> None:
    """Test a SATISFIED bindings outcome maps to `True`."""
    constraint = EquationConstraint(LiteralExpression(True))

    assert constraint.is_satisfied_with_bindings({}) is True


def test_is_satisfied_with_bindings_false_when_violated() -> None:
    """Test a VIOLATED bindings outcome maps to `False`."""
    constraint = EquationConstraint(LiteralExpression(False))

    assert constraint.is_satisfied_with_bindings({}) is False


# =============================================================================
# `repr` / `str`
# =============================================================================


def test_repr_includes_class_name_and_expression() -> None:
    """Test `repr` includes the class name and the wrapped expression."""
    expression = LiteralExpression(True)
    constraint = EquationConstraint(expression)

    rendered = repr(constraint)

    assert "EquationConstraint" in rendered
    assert repr(expression) in rendered


def test_str_matches_expression_pformat() -> None:
    """Test `str(constraint)` matches `pformat_expression` of the expression."""
    expression = LiteralExpression(True)
    constraint = EquationConstraint(expression)

    assert str(constraint) == pformat_expression(expression)


# =============================================================================
# Extra shapes carried over from the retired unary contract
# =============================================================================


@pytest.mark.parametrize(
    "expression, bindings, expected_outcome",
    [
        pytest.param(
            UnaryExpression(UnaryOperation.LOGICAL_NOT, LiteralExpression(True)),
            {},
            ConstraintOutcome.VIOLATED,
            id="not_true",
        ),
        pytest.param(
            BinaryExpression(
                BinaryOperation.LOGICAL_AND,
                LiteralExpression(True),
                LiteralExpression(False),
            ),
            {},
            ConstraintOutcome.VIOLATED,
            id="and_true_false",
        ),
        pytest.param(
            BinaryExpression(
                BinaryOperation.LOGICAL_OR,
                LiteralExpression(True),
                LiteralExpression(False),
            ),
            {},
            ConstraintOutcome.SATISFIED,
            id="or_true_false",
        ),
        pytest.param(
            BinaryExpression(
                BinaryOperation.NOT_EQUAL,
                LiteralExpression(True),
                LiteralExpression(False),
            ),
            {},
            ConstraintOutcome.SATISFIED,
            id="not_equal_true",
        ),
    ],
)
def test_evaluate_with_bindings_decides_a_variety_of_ground_shapes(
    expression: Any, bindings: Any, expected_outcome: ConstraintOutcome
) -> None:
    """Test a variety of ground Boolean-combinator shapes decide correctly."""
    constraint = EquationConstraint(expression)

    assert constraint.evaluate_with_bindings(bindings) is expected_outcome


def test_evaluate_with_bindings_two_free_identifiers_undecided_when_unbound(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Test an expression over two free identifiers stays UNDECIDED when unbound."""
    y = mock_identifier("y", 1)
    z = mock_identifier("z", 2)
    expression = logical_and(y, z)
    constraint = EquationConstraint(expression)

    with caplog.at_level(logging.DEBUG, logger=_CONSTRAINT_LOGGER):
        outcome = constraint.evaluate_with_bindings({})

    assert outcome is ConstraintOutcome.UNDECIDED
    assert _find_records(caplog, logging.DEBUG)
