"""Tests for `fhy_core.expression.passes.z3`.

Notes on known-equivalent mutants not targeted by this file:

- The ``value == "True"`` / ``value == "False"`` string branches in
  ``visit_literal_expression`` are unreachable: ``LiteralExpression``'s
  ``__post_init__`` rejects any non-numeric string at construction, so the
  comparison-operator mutants there cannot be distinguished from the public
  surface.
- ``identifier_type == SymbolType.{REAL, INT, BOOL}`` comparisons are
  ``Enum``-singleton equivalents under ``is``; those mutants are not
  distinguishable in CPython.
"""

import pytest
import z3  # type: ignore[import-untyped]

from fhy_core.expression import (
    BinaryExpression,
    BinaryOperation,
    Expression,
    IdentifierExpression,
    LiteralExpression,
    SymbolType,
    UnaryExpression,
    UnaryOperation,
    convert_expression_to_z3_expression,
    is_satisfiable,
)
from fhy_core.identifier import Identifier

from .conftest import mock_identifier

pytestmark = pytest.mark.z3


# =============================================================================
# Expression -> Z3
# =============================================================================


@pytest.mark.parametrize(
    "expression, symbol_types, expected_z3_expression",
    [
        (LiteralExpression(5), {}, z3.IntVal(5)),
        (LiteralExpression(5.5), {}, z3.RealVal(5.5)),
        (LiteralExpression(True), {}, z3.BoolVal(True)),
        (LiteralExpression(False), {}, z3.BoolVal(False)),
        (LiteralExpression("10.6"), {}, z3.RealVal(10.6)),
        (
            UnaryExpression(
                UnaryOperation.POSITIVE, IdentifierExpression(mock_identifier("x", 0))
            ),
            {mock_identifier("x", 0): SymbolType.REAL},
            z3.Real("x_0"),
        ),
        (
            UnaryExpression(
                UnaryOperation.NEGATE, IdentifierExpression(mock_identifier("x", 0))
            ),
            {mock_identifier("x", 0): SymbolType.REAL},
            -z3.Real("x_0"),
        ),
        (
            UnaryExpression(
                UnaryOperation.LOGICAL_NOT,
                IdentifierExpression(mock_identifier("x", 0)),
            ),
            {mock_identifier("x", 0): SymbolType.BOOL},
            z3.Not(z3.Bool("x_0")),
        ),
        (
            BinaryExpression(
                BinaryOperation.ADD,
                IdentifierExpression(mock_identifier("x", 0)),
                LiteralExpression(5),
            ),
            {mock_identifier("x", 0): SymbolType.INT},
            z3.Int("x_0") + z3.IntVal(5),
        ),
        (
            BinaryExpression(
                BinaryOperation.SUBTRACT,
                IdentifierExpression(mock_identifier("x", 0)),
                LiteralExpression(5),
            ),
            {mock_identifier("x", 0): SymbolType.INT},
            z3.Int("x_0") - z3.IntVal(5),
        ),
        (
            BinaryExpression(
                BinaryOperation.MULTIPLY,
                IdentifierExpression(mock_identifier("x", 0)),
                LiteralExpression(5.5),
            ),
            {mock_identifier("x", 0): SymbolType.REAL},
            z3.Real("x_0") * z3.RealVal(5.5),
        ),
        (
            BinaryExpression(
                BinaryOperation.DIVIDE,
                IdentifierExpression(mock_identifier("x", 0)),
                LiteralExpression(5.5),
            ),
            {mock_identifier("x", 0): SymbolType.REAL},
            z3.Real("x_0") / z3.RealVal(5.5),
        ),
        (
            BinaryExpression(
                BinaryOperation.FLOOR_DIVIDE,
                IdentifierExpression(mock_identifier("x", 0)),
                LiteralExpression(5),
            ),
            {mock_identifier("x", 0): SymbolType.INT},
            z3.Int("x_0") / z3.IntVal(5),
        ),
        (
            BinaryExpression(
                BinaryOperation.FLOOR_DIVIDE,
                IdentifierExpression(mock_identifier("x", 0)),
                LiteralExpression(5),
            ),
            {mock_identifier("x", 0): SymbolType.REAL},
            z3.ToInt(z3.Real("x_0") / z3.IntVal(5)),
        ),
        (
            BinaryExpression(
                BinaryOperation.MODULO,
                IdentifierExpression(mock_identifier("x", 0)),
                LiteralExpression(5),
            ),
            {mock_identifier("x", 0): SymbolType.INT},
            z3.Int("x_0") % z3.IntVal(5),
        ),
        (
            BinaryExpression(
                BinaryOperation.POWER,
                IdentifierExpression(mock_identifier("x", 0)),
                LiteralExpression(5),
            ),
            {mock_identifier("x", 0): SymbolType.INT},
            z3.Int("x_0") ** z3.IntVal(5),
        ),
        (
            BinaryExpression(
                BinaryOperation.LOGICAL_AND,
                IdentifierExpression(mock_identifier("x", 0)),
                IdentifierExpression(mock_identifier("y", 1)),
            ),
            {
                mock_identifier("x", 0): SymbolType.BOOL,
                mock_identifier("y", 1): SymbolType.BOOL,
            },
            z3.And(z3.Bool("x_0"), z3.Bool("y_1")),
        ),
        (
            BinaryExpression(
                BinaryOperation.LOGICAL_OR,
                IdentifierExpression(mock_identifier("x", 0)),
                IdentifierExpression(mock_identifier("y", 1)),
            ),
            {
                mock_identifier("x", 0): SymbolType.BOOL,
                mock_identifier("y", 1): SymbolType.BOOL,
            },
            z3.Or(z3.Bool("x_0"), z3.Bool("y_1")),
        ),
        (
            BinaryExpression(
                BinaryOperation.EQUAL,
                IdentifierExpression(mock_identifier("x", 0)),
                LiteralExpression(5),
            ),
            {mock_identifier("x", 0): SymbolType.REAL},
            z3.Real("x_0") == z3.IntVal(5),
        ),
        (
            BinaryExpression(
                BinaryOperation.NOT_EQUAL,
                IdentifierExpression(mock_identifier("x", 0)),
                LiteralExpression(5),
            ),
            {mock_identifier("x", 0): SymbolType.REAL},
            z3.Real("x_0") != z3.IntVal(5),
        ),
        (
            BinaryExpression(
                BinaryOperation.LESS,
                IdentifierExpression(mock_identifier("x", 0)),
                LiteralExpression(5),
            ),
            {mock_identifier("x", 0): SymbolType.INT},
            z3.Int("x_0") < z3.IntVal(5),
        ),
        (
            BinaryExpression(
                BinaryOperation.LESS_EQUAL,
                IdentifierExpression(mock_identifier("x", 0)),
                LiteralExpression(5),
            ),
            {mock_identifier("x", 0): SymbolType.INT},
            z3.Int("x_0") <= z3.IntVal(5),
        ),
        (
            BinaryExpression(
                BinaryOperation.GREATER,
                IdentifierExpression(mock_identifier("x", 0)),
                LiteralExpression(5),
            ),
            {mock_identifier("x", 0): SymbolType.INT},
            z3.Int("x_0") > z3.IntVal(5),
        ),
        (
            BinaryExpression(
                BinaryOperation.GREATER_EQUAL,
                IdentifierExpression(mock_identifier("x", 0)),
                LiteralExpression(5),
            ),
            {mock_identifier("x", 0): SymbolType.INT},
            z3.Int("x_0") >= z3.IntVal(5),
        ),
        (
            BinaryExpression(
                BinaryOperation.EQUAL,
                BinaryExpression(
                    BinaryOperation.MODULO,
                    IdentifierExpression(mock_identifier("x", 0)),
                    LiteralExpression(5),
                ),
                LiteralExpression(0),
            ),
            {mock_identifier("x", 0): SymbolType.INT},
            z3.Int("x_0") % z3.IntVal(5) == z3.IntVal(0),
        ),
    ],
)
def test_convert_expression_to_z3_expression(
    expression: Expression,
    symbol_types: dict[Identifier, SymbolType],
    expected_z3_expression: z3.ExprRef,
) -> None:
    """Test `convert_expression_to_z3_expression` maps each expression to Z3."""
    result, _ = convert_expression_to_z3_expression(expression, symbol_types)
    assert result.eq(expected_z3_expression)


@pytest.mark.parametrize(
    "symbol_type, expected_sort_class",
    [
        (SymbolType.REAL, z3.ArithSortRef),
        (SymbolType.INT, z3.ArithSortRef),
        (SymbolType.BOOL, z3.BoolSortRef),
    ],
)
def test_symbol_type_maps_to_correct_z3_sort(
    symbol_type: SymbolType, expected_sort_class: type
) -> None:
    """Test each `SymbolType` maps to the expected Z3 sort."""
    identifier = mock_identifier("x", 0)
    result, _ = convert_expression_to_z3_expression(
        IdentifierExpression(identifier), {identifier: symbol_type}
    )
    assert isinstance(result.sort(), expected_sort_class)
    if symbol_type is SymbolType.INT:
        assert result.sort().is_int()
    elif symbol_type is SymbolType.REAL:
        assert result.sort().is_real()


# =============================================================================
# is_satisfiable
# =============================================================================


@pytest.mark.parametrize(
    "expression, considered_identifiers, symbol_types, expected_output",
    [
        (
            BinaryExpression(
                BinaryOperation.EQUAL,
                IdentifierExpression(mock_identifier("x", 0)),
                LiteralExpression(5),
            ),
            {mock_identifier("x", 0)},
            {mock_identifier("x", 0): SymbolType.INT},
            True,
        ),
        (
            BinaryExpression(
                BinaryOperation.LOGICAL_AND,
                BinaryExpression(
                    BinaryOperation.LESS,
                    IdentifierExpression(mock_identifier("x", 0)),
                    IdentifierExpression(mock_identifier("N", 3)),
                ),
                BinaryExpression(
                    BinaryOperation.GREATER,
                    IdentifierExpression(mock_identifier("x", 0)),
                    IdentifierExpression(mock_identifier("N", 3)),
                ),
            ),
            {mock_identifier("x", 0)},
            {
                mock_identifier("x", 0): SymbolType.INT,
                mock_identifier("N", 3): SymbolType.INT,
            },
            False,
        ),
        (
            BinaryExpression(
                BinaryOperation.LOGICAL_AND,
                BinaryExpression(
                    BinaryOperation.LESS,
                    IdentifierExpression(mock_identifier("x", 0)),
                    IdentifierExpression(mock_identifier("N", 3)),
                ),
                BinaryExpression(
                    BinaryOperation.LESS,
                    IdentifierExpression(mock_identifier("x", 0)),
                    BinaryExpression(
                        BinaryOperation.SUBTRACT,
                        IdentifierExpression(mock_identifier("N", 3)),
                        LiteralExpression(1),
                    ),
                ),
            ),
            {mock_identifier("x", 0)},
            {
                mock_identifier("x", 0): SymbolType.INT,
                mock_identifier("N", 3): SymbolType.INT,
            },
            True,
        ),
    ],
)
def test_is_satisfiable_on_example_expressions(
    expression: Expression,
    considered_identifiers: set[Identifier],
    symbol_types: dict[Identifier, SymbolType],
    expected_output: bool | None,
) -> None:
    """Test `is_satisfiable` returns the expected tri-valued result on examples."""
    assert (
        is_satisfiable(considered_identifiers, expression, symbol_types)
        == expected_output
    )


def _make_trivial_satisfiability_inputs() -> tuple[
    set[Identifier], Expression, dict[Identifier, SymbolType]
]:
    identifier = mock_identifier("x", 0)
    expression = BinaryExpression(
        BinaryOperation.EQUAL,
        IdentifierExpression(identifier),
        LiteralExpression(0),
    )
    return {identifier}, expression, {identifier: SymbolType.INT}


def test_is_satisfiable_returns_none_when_solver_reports_unknown(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test `is_satisfiable` returns `None` when the solver reports `z3.unknown`."""
    monkeypatch.setattr(z3.Solver, "check", lambda self: z3.unknown)
    considered, expression, symbol_types = _make_trivial_satisfiability_inputs()
    assert is_satisfiable(considered, expression, symbol_types) is None


def test_is_satisfiable_returns_false_when_solver_reports_sat(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test `is_satisfiable` returns `False` when the solver reports `z3.sat`."""
    monkeypatch.setattr(z3.Solver, "check", lambda self: z3.sat)
    considered, expression, symbol_types = _make_trivial_satisfiability_inputs()
    assert is_satisfiable(considered, expression, symbol_types) is False


def test_is_satisfiable_returns_true_when_solver_reports_unsat(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test `is_satisfiable` returns `True` when the solver reports `z3.unsat`."""
    monkeypatch.setattr(z3.Solver, "check", lambda self: z3.unsat)
    considered, expression, symbol_types = _make_trivial_satisfiability_inputs()
    assert is_satisfiable(considered, expression, symbol_types) is True
