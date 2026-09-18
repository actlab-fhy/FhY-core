"""Behavioral tests for `EquationConstraint`."""

import logging
import math
from collections.abc import Callable, Mapping
from typing import Any

import pytest

from fhy_core.identifier import Identifier
from fhy_core.pass_infrastructure import PassExecutionError
from fhy_core.symbolic.constraint import (
    ConstraintError,
    ConstraintOutcome,
    EquationConstraint,
)
from fhy_core.symbolic.expression import (
    BinaryExpression,
    BinaryOperation,
    Expression,
    IdentifierExpression,
    LiteralExpression,
    NonBooleanLogicalOperandError,
    UnaryExpression,
    UnaryOperation,
    call,
    get_native_constant_identifier,
    logical_and,
    logical_not,
    make_binary_expression,
    pformat_expression,
    piecewise,
)

from .conftest import mock_identifier

_CONSTRAINT_LOGGER = "fhy_core.symbolic.constraint.core"


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


def test_evaluate_with_bindings_non_bool_literal_reduction_raises() -> None:
    """Test a substituted expression reducing to a non-bool literal raises.

    ``LiteralExpression(1)`` is a numeric root: the constraint's
    expression is itself a Boolean position, so a residual that
    simplifies to a non-bool literal is ill-typed rather than a decided
    ``VIOLATED``.
    """
    constraint = EquationConstraint(LiteralExpression(1))

    with pytest.raises(NonBooleanLogicalOperandError):
        constraint.evaluate_with_bindings({})


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
# A numeric root is refused, not decided VIOLATED
# =============================================================================


def test_evaluate_with_bindings_refuses_a_numeric_result_call_under_not() -> None:
    """Test a connective over a numeric-result call raises, not VIOLATED.

    ``floor`` is registered with an INT result sort, so
    ``logical_not(floor(x))`` is ill-typed once ``x`` is bound: the
    constraint's expression is itself a Boolean position, and the
    expression is ill-typed regardless of what value ``x`` takes.
    """
    x = mock_identifier("x", 0)
    constraint = EquationConstraint(logical_not(call("floor", IdentifierExpression(x))))

    with pytest.raises(NonBooleanLogicalOperandError):
        constraint.evaluate_with_bindings({x: 0.5})


def test_evaluate_with_bindings_refuses_an_arithmetic_root() -> None:
    """Test an arithmetic root raises rather than being decided VIOLATED.

    ``x + 1`` denotes a number, not a predicate, so it is refused before
    it is ever substituted or simplified.
    """
    x = mock_identifier("x", 0)
    constraint = EquationConstraint(
        make_binary_expression(BinaryOperation.ADD, x, LiteralExpression(1))
    )

    with pytest.raises(NonBooleanLogicalOperandError):
        constraint.evaluate_with_bindings({x: 0})


def test_evaluate_with_bindings_refuses_a_root_bound_to_a_mixed_piecewise() -> None:
    """Test a root identifier bound to a piecewise with a numeric branch raises.

    The expression's root is the bound identifier, itself a Boolean
    position: the numeric ``value`` branch beside the Boolean
    ``otherwise`` makes the substituted root ill-typed regardless of
    which case the condition would pick, so this is refused ahead of
    simplification rather than decided.
    """
    x = mock_identifier("x", 0)
    mixed = piecewise(
        (LiteralExpression(False), LiteralExpression(1)),
        otherwise=LiteralExpression(True),
    )
    constraint = EquationConstraint(IdentifierExpression(x))

    with pytest.raises(NonBooleanLogicalOperandError):
        constraint.evaluate_with_bindings({x: mixed})


def test_evaluate_with_bindings_refuses_a_mixed_piecewise_with_unbound_condition() -> (
    None
):
    """Test the mixed-branch refusal holds even when the condition is unbound.

    ``b`` carries no declared sort and no binding, so it cannot itself
    make the piecewise ill-typed; the numeric ``value`` branch beside the
    Boolean ``otherwise`` does that on its own.
    """
    x = mock_identifier("x", 0)
    b = mock_identifier("b", 1)
    mixed = piecewise(
        (IdentifierExpression(b), LiteralExpression(1)),
        otherwise=LiteralExpression(True),
    )
    constraint = EquationConstraint(IdentifierExpression(x))

    with pytest.raises(NonBooleanLogicalOperandError):
        constraint.evaluate_with_bindings({x: mixed})


def test_is_satisfied_with_bindings_refuses_a_numeric_result_call_under_not() -> None:
    """Test `is_satisfied_with_bindings` raises for the same ill-typed constraint."""
    x = mock_identifier("x", 0)
    constraint = EquationConstraint(logical_not(call("floor", IdentifierExpression(x))))

    with pytest.raises(NonBooleanLogicalOperandError):
        constraint.is_satisfied_with_bindings({x: 0.5})


def test_is_satisfied_with_bindings_refuses_an_arithmetic_root() -> None:
    """Test `is_satisfied_with_bindings` raises for an arithmetic root."""
    x = mock_identifier("x", 0)
    constraint = EquationConstraint(
        make_binary_expression(BinaryOperation.ADD, x, LiteralExpression(1))
    )

    with pytest.raises(NonBooleanLogicalOperandError):
        constraint.is_satisfied_with_bindings({x: 0})


def test_evaluate_with_bindings_raises_when_a_binding_divides_by_zero() -> None:
    """Test a zero-divisor binding under a comparison raises `PassExecutionError`.

    Substituting a zero divisor rebuilds ``(x / y) > 1`` against SymPy's
    complex infinity, which the SymPy bridge cannot compare and raises a
    raw `TypeError`; that must surface as `PassExecutionError`, like
    every other bridge failure, rather than escaping unwrapped.
    """
    x = mock_identifier("x", 0)
    y = mock_identifier("y", 1)
    expression = BinaryExpression(
        BinaryOperation.GREATER,
        BinaryExpression(
            BinaryOperation.DIVIDE, IdentifierExpression(x), IdentifierExpression(y)
        ),
        LiteralExpression(1),
    )
    constraint = EquationConstraint(expression)

    with pytest.raises(PassExecutionError) as exc_info:
        constraint.evaluate_with_bindings({x: 1, y: 0})

    assert isinstance(exc_info.value.__cause__, TypeError)


def test_evaluate_with_bindings_raises_when_a_nan_binding_reaches_a_comparison() -> (
    None
):
    """Test a NaN binding under a strict comparison raises `PassExecutionError`."""
    x = mock_identifier("x", 0)
    constraint = EquationConstraint(make_binary_expression(BinaryOperation.LESS, x, 5))

    with pytest.raises(PassExecutionError) as exc_info:
        constraint.evaluate_with_bindings({x: math.nan})

    assert isinstance(exc_info.value.__cause__, TypeError)


def test_evaluate_with_bindings_still_decides_a_well_defined_divided_comparison() -> (
    None
):
    """Test a nonzero-divisor binding still decides the comparison as SATISFIED.

    Confirms the `PassExecutionError` wrapping above is specific to the
    unrepresentable shapes, not a regression over ordinary division.
    """
    x = mock_identifier("x", 0)
    y = mock_identifier("y", 1)
    expression = BinaryExpression(
        BinaryOperation.GREATER,
        BinaryExpression(
            BinaryOperation.DIVIDE, IdentifierExpression(x), IdentifierExpression(y)
        ),
        LiteralExpression(1),
    )
    constraint = EquationConstraint(expression)

    outcome = constraint.evaluate_with_bindings({x: 4, y: 2})

    assert outcome is ConstraintOutcome.SATISFIED


def test_evaluate_with_bindings_equality_of_two_nan_bindings_is_violated() -> None:
    """Test an equality comparison between two NaN bindings decides VIOLATED.

    Unlike a strict inequality, SymPy's ``Eq`` does not raise for a NaN
    operand; it decides ``False``, matching IEEE-754 NaN-is-never-equal
    semantics, so this must stay a decided outcome rather than degrade.
    """
    x = mock_identifier("x", 0)
    y = mock_identifier("y", 1)
    constraint = EquationConstraint(make_binary_expression(BinaryOperation.EQUAL, x, y))

    outcome = constraint.evaluate_with_bindings({x: math.nan, y: math.nan})

    assert outcome is ConstraintOutcome.VIOLATED


def test_construction_still_accepts_a_numeric_literal_expression() -> None:
    """Test construction stays permissive for a numeric root.

    The refusal happens when the constraint is used
    (``evaluate_with_bindings``), not at construction: an
    ``EquationConstraint`` is a plain wrapper over any ``Expression``.
    """
    expression = LiteralExpression(5)

    constraint = EquationConstraint(expression)

    assert constraint.expression is expression


# =============================================================================
# A piecewise over Boolean identifiers decides by its first-match meaning
# =============================================================================


@pytest.mark.parametrize(
    ("b_value", "c_value", "expected"),
    [
        (True, True, ConstraintOutcome.VIOLATED),
        (True, False, ConstraintOutcome.VIOLATED),
        (False, True, ConstraintOutcome.VIOLATED),
        (False, False, ConstraintOutcome.SATISFIED),
    ],
)
def test_evaluate_with_bindings_negates_a_piecewise_with_an_identifier_condition(
    b_value: bool, c_value: bool, expected: ConstraintOutcome
) -> None:
    """Test ``!(True if b, otherwise c)`` holds only when ``b`` and ``c`` both fail."""
    b = mock_identifier("b", 0)
    c = mock_identifier("c", 1)
    constraint = EquationConstraint(
        logical_not(
            piecewise(
                (IdentifierExpression(b), LiteralExpression(True)),
                otherwise=IdentifierExpression(c),
            )
        )
    )

    outcome = constraint.evaluate_with_bindings({b: b_value, c: c_value})

    assert outcome is expected


@pytest.mark.parametrize(
    ("b_value", "c_value", "x_value", "expected"),
    [
        (True, True, 2, ConstraintOutcome.SATISFIED),
        (False, True, 2, ConstraintOutcome.SATISFIED),
        (False, False, 2, ConstraintOutcome.VIOLATED),
        (False, True, 0, ConstraintOutcome.VIOLATED),
    ],
)
def test_evaluate_with_bindings_reads_a_case_condition_with_an_identifier_condition(
    b_value: bool, c_value: bool, x_value: int, expected: ConstraintOutcome
) -> None:
    """Test ``x > 1 if (True if b, otherwise c), otherwise x < 1`` follows bindings."""
    b = mock_identifier("b", 0)
    c = mock_identifier("c", 1)
    x = mock_identifier("x", 2)
    condition = piecewise(
        (IdentifierExpression(b), LiteralExpression(True)),
        otherwise=IdentifierExpression(c),
    )
    constraint = EquationConstraint(
        piecewise(
            (condition, IdentifierExpression(x) > 1),
            otherwise=IdentifierExpression(x) < 1,
        )
    )

    outcome = constraint.evaluate_with_bindings({b: b_value, c: c_value, x: x_value})

    assert outcome is expected


@pytest.mark.parametrize(
    "operation", [BinaryOperation.EQUAL, BinaryOperation.NOT_EQUAL]
)
@pytest.mark.parametrize(
    "build_bindings",
    [
        pytest.param(lambda b, _, value: {b: value}, id="b_bound"),
        pytest.param(lambda b, x, value: {b: value, x: 5}, id="b_and_x_bound"),
    ],
)
def test_evaluate_with_bindings_leaves_a_bound_open_boolean_piecewise_undecided(
    operation: BinaryOperation,
    build_bindings: Callable[
        [Identifier, Identifier, Expression], Mapping[Identifier, Expression | int]
    ],
) -> None:
    """Test ``b == True`` under ``b = (True if x > 0, otherwise False)`` is undecided.

    ``x`` is free in the value bound to ``b``, and a binding for ``x``
    does not reach into that value, so the outcome is neither satisfied
    nor violated.
    """
    b = mock_identifier("b", 0)
    x = mock_identifier("x", 1)
    constraint = EquationConstraint(
        BinaryExpression(operation, IdentifierExpression(b), LiteralExpression(True))
    )
    boolean_piecewise = piecewise(
        (IdentifierExpression(x) > 0, LiteralExpression(True)),
        otherwise=LiteralExpression(False),
    )

    outcome = constraint.evaluate_with_bindings(build_bindings(b, x, boolean_piecewise))

    assert outcome is ConstraintOutcome.UNDECIDED


@pytest.mark.parametrize(
    ("bound_values", "expected"),
    [
        pytest.param({}, ConstraintOutcome.UNDECIDED, id="unbound"),
        pytest.param({"x": 0, "c": True}, ConstraintOutcome.SATISFIED, id="first_case"),
        pytest.param(
            {"x": 0, "c": False}, ConstraintOutcome.VIOLATED, id="first_case_false"
        ),
        pytest.param(
            {"x": 1, "c": True}, ConstraintOutcome.UNDECIDED, id="otherwise_unbound"
        ),
        pytest.param(
            {"x": 1, "b": False, "c": False},
            ConstraintOutcome.SATISFIED,
            id="otherwise_bound",
        ),
        pytest.param(
            {"x": 1, "b": True, "c": False},
            ConstraintOutcome.VIOLATED,
            id="otherwise_bound_false",
        ),
    ],
)
def test_evaluate_with_bindings_compares_a_piecewise_mixing_identifier_branches(
    bound_values: dict[str, bool | int], expected: ConstraintOutcome
) -> None:
    """Test ``(True if x == 0, otherwise b) == c`` follows its bindings.

    The unbound rows leave the comparison open, which must report
    ``UNDECIDED`` rather than raise.
    """
    identifiers = {
        "x": mock_identifier("x", 0),
        "b": mock_identifier("b", 1),
        "c": mock_identifier("c", 2),
    }
    constraint = EquationConstraint(
        BinaryExpression(
            BinaryOperation.EQUAL,
            piecewise(
                (
                    IdentifierExpression(identifiers["x"]).equals(0),
                    LiteralExpression(True),
                ),
                otherwise=IdentifierExpression(identifiers["b"]),
            ),
            IdentifierExpression(identifiers["c"]),
        )
    )

    outcome = constraint.evaluate_with_bindings(
        {identifiers[name]: value for name, value in bound_values.items()}
    )

    assert outcome is expected


@pytest.mark.parametrize(
    ("b_value", "d_value", "c_value", "expected"),
    [
        (True, True, False, ConstraintOutcome.VIOLATED),
        (False, False, True, ConstraintOutcome.SATISFIED),
        (True, False, False, ConstraintOutcome.SATISFIED),
        (True, False, True, ConstraintOutcome.VIOLATED),
    ],
)
def test_evaluate_with_bindings_negates_a_piecewise_comparing_a_branch_identifier(
    b_value: bool, d_value: bool, c_value: bool, expected: ConstraintOutcome
) -> None:
    """Test ``!(b if b == d, otherwise c)`` follows its first case when ``b == d``."""
    b = mock_identifier("b", 0)
    d = mock_identifier("d", 1)
    c = mock_identifier("c", 2)
    constraint = EquationConstraint(
        logical_not(
            piecewise(
                (
                    IdentifierExpression(b).equals(IdentifierExpression(d)),
                    IdentifierExpression(b),
                ),
                otherwise=IdentifierExpression(c),
            )
        )
    )

    outcome = constraint.evaluate_with_bindings({b: b_value, d: d_value, c: c_value})

    assert outcome is expected


@pytest.mark.parametrize(
    ("x_value", "expected"),
    [
        (3, ConstraintOutcome.SATISFIED),
        (1, ConstraintOutcome.VIOLATED),
        (2, ConstraintOutcome.UNDECIDED),
    ],
)
def test_evaluate_with_bindings_compares_a_numeric_piecewise_over_an_unbound_condition(
    x_value: int, expected: ConstraintOutcome
) -> None:
    """Test ``(1 if b, otherwise 2) < x`` is decided only when ``b`` cannot matter."""
    b = mock_identifier("b", 0)
    x = mock_identifier("x", 1)
    numeric = piecewise(
        (IdentifierExpression(b), LiteralExpression(1)),
        otherwise=LiteralExpression(2),
    )
    constraint = EquationConstraint(numeric < IdentifierExpression(x))

    outcome = constraint.evaluate_with_bindings({x: x_value})

    assert outcome is expected


def test_evaluate_with_bindings_refutes_a_boolean_comparison_beside_relations() -> None:
    """Test ``(d == True) && x < 1 && x > 2`` is violated for every ``d`` and ``x``."""
    d = mock_identifier("d", 0)
    x = mock_identifier("x", 1)
    constraint = EquationConstraint(
        logical_and(
            logical_and(
                IdentifierExpression(d).equals(LiteralExpression(True)),
                IdentifierExpression(x) < 1,
            ),
            IdentifierExpression(x) > 2,
        )
    )

    outcome = constraint.evaluate_with_bindings({})

    assert outcome is ConstraintOutcome.VIOLATED


@pytest.mark.parametrize(
    ("c_value", "d_value", "expected"),
    [
        (True, True, ConstraintOutcome.SATISFIED),
        (False, False, ConstraintOutcome.VIOLATED),
        (True, False, ConstraintOutcome.UNDECIDED),
    ],
)
def test_evaluate_with_bindings_reads_a_case_condition_comparing_a_relation(
    c_value: bool, d_value: bool, expected: ConstraintOutcome
) -> None:
    """Test ``c if (x == 0) == b, otherwise d`` is decided only when ``c == d``."""
    x = mock_identifier("x", 0)
    b = mock_identifier("b", 1)
    c = mock_identifier("c", 2)
    d = mock_identifier("d", 3)
    condition = BinaryExpression(
        BinaryOperation.EQUAL,
        IdentifierExpression(x).equals(0),
        IdentifierExpression(b),
    )
    constraint = EquationConstraint(
        piecewise(
            (condition, IdentifierExpression(c)), otherwise=IdentifierExpression(d)
        )
    )

    outcome = constraint.evaluate_with_bindings({c: c_value, d: d_value})

    assert outcome is expected


@pytest.mark.parametrize(
    ("b_value", "expected"),
    [(True, ConstraintOutcome.SATISFIED), (False, ConstraintOutcome.VIOLATED)],
)
def test_evaluate_with_bindings_divides_an_even_valued_piecewise(
    b_value: bool, expected: ConstraintOutcome
) -> None:
    """Test ``(6 if b, otherwise 0) / 3 == 2`` holds exactly when ``b`` does."""
    b = mock_identifier("b", 0)
    even_valued = piecewise(
        (IdentifierExpression(b), LiteralExpression(6)),
        otherwise=LiteralExpression(0),
    )
    constraint = EquationConstraint(
        (even_valued / LiteralExpression(3)).equals(LiteralExpression(2))
    )

    outcome = constraint.evaluate_with_bindings({b: b_value})

    assert outcome is expected


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

    The root is an equality of two calls, which is a well-typed Boolean
    position (an ``EQUAL`` node, not a numeric one), so it survives the
    ill-typedness screen. Once ``y`` is bound to a value outside
    ``arcsin``/``arccos``'s real domain, SymPy leaves both sides
    unevaluated and cannot decide the equality between them, so the
    residual is a closed (no free identifiers) but irreducible
    `BinaryExpression`: every free identifier was bound, yet the
    simplifier still failed to reduce it to a literal -- the genuine
    anomaly case.
    """
    y = mock_identifier("y", 1)
    constraint = EquationConstraint(
        make_binary_expression(
            BinaryOperation.EQUAL,
            call("arcsin", IdentifierExpression(y)),
            call("arccos", IdentifierExpression(y)),
        )
    )

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


def test_evaluate_with_bindings_refuses_to_decide_a_bound_native_constant(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Test binding a constant's canonical identifier reports UNDECIDED.

    The bridge lowers that identifier to the constant's value rather than
    to a substitutable symbol, so the binding cannot take part in the
    decision. Reporting the outcome the constant happens to give would
    be a decision the caller's binding never reached.
    """
    pi = get_native_constant_identifier("pi")
    expression = BinaryExpression(
        BinaryOperation.EQUAL, IdentifierExpression(pi), LiteralExpression(1)
    )
    constraint = EquationConstraint(expression)

    with caplog.at_level(logging.WARNING, logger=_CONSTRAINT_LOGGER):
        outcome = constraint.evaluate_with_bindings({pi: 1})

    assert outcome is ConstraintOutcome.UNDECIDED
    records = _find_records(caplog, logging.WARNING)
    assert records
    assert repr(pi) in records[0].getMessage()


def test_evaluate_with_bindings_honors_a_binding_named_after_a_constant() -> None:
    """Test a binding for an identifier that merely shares ``pi``'s name is applied.

    Only the canonical identifier denotes the constant, so this is an
    ordinary variable: substituting ``1`` decides the equation.
    """
    pi_lookalike = mock_identifier("pi", 1088)
    expression = BinaryExpression(
        BinaryOperation.EQUAL,
        IdentifierExpression(pi_lookalike),
        LiteralExpression(1),
    )
    constraint = EquationConstraint(expression)

    outcome = constraint.evaluate_with_bindings({pi_lookalike: 1})

    assert outcome is ConstraintOutcome.SATISFIED
