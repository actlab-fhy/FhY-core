"""Tests for `fhy_core.symbolic.expression.passes.z3`.

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

import logging
from unittest.mock import Mock

import pytest
import z3  # type: ignore[import-untyped]

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
    convert_expression_to_z3_expression,
)
from fhy_core.symbolic.expression.passes.z3 import (
    ExpressionToZ3Converter,
    _z3_floor_divide,
)
from fhy_core.symbolic.solver import (
    does_expression_imply,
    holds_for_all_free_assignments,
)
from fhy_core.symbolic.symbol_type import SymbolType

from ..conftest import mock_identifier

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
# holds_for_all_free_assignments
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
def test_holds_for_all_free_assignments_on_example_expressions(
    expression: Expression,
    considered_identifiers: set[Identifier],
    symbol_types: dict[Identifier, SymbolType],
    expected_output: bool | None,
) -> None:
    """Test the tri-valued result of ``holds_for_all_free_assignments`` on examples."""
    assert (
        holds_for_all_free_assignments(considered_identifiers, expression, symbol_types)
        == expected_output
    )


def test_holds_for_all_free_with_empty_considered_set_is_validity_check() -> None:
    """Test the empty-considered branch skips the ForAll wrapper.

    With no considered identifiers, the implementation degenerates to
    "is the expression universally valid over its free identifiers"
    -- here ``5 == 5`` is, so the result is True.
    """
    expression = BinaryExpression(
        BinaryOperation.EQUAL, LiteralExpression(5), LiteralExpression(5)
    )
    assert holds_for_all_free_assignments(set(), expression, {}) is True


def test_holds_for_all_free_ignores_considered_identifiers_absent_from_expression() -> (
    None
):
    """Test extra `considered_identifiers` not in the expression are skipped.

    A quantifier over an identifier that does not appear in the body is
    semantically vacuous, so the entry point silently filters such
    identifiers rather than raising a `KeyError` from the converter's
    identifier-to-Z3 mapping.
    """
    x = mock_identifier("x", 0)
    unused = mock_identifier("u", 1)
    expression = BinaryExpression(
        BinaryOperation.EQUAL, IdentifierExpression(x), LiteralExpression(5)
    )

    result = holds_for_all_free_assignments(
        {x, unused},
        expression,
        {x: SymbolType.INT, unused: SymbolType.INT},
    )

    assert result is True


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


@pytest.mark.parametrize(
    "solver_result, expected_satisfiability",
    [
        pytest.param(z3.unknown, None, id="unknown_to_none"),
        pytest.param(z3.sat, False, id="sat_to_false"),
        pytest.param(z3.unsat, True, id="unsat_to_true"),
    ],
)
def test_holds_for_all_free_assignments_maps_solver_result_to_satisfiability(
    monkeypatch: pytest.MonkeyPatch,
    solver_result: z3.CheckSatResult,
    expected_satisfiability: bool | None,
) -> None:
    """Test ``holds_for_all_free_assignments`` maps each z3 solver outcome correctly."""
    monkeypatch.setattr(z3.Solver, "check", lambda self: solver_result)
    considered, expression, symbol_types = _make_trivial_satisfiability_inputs()
    result = holds_for_all_free_assignments(considered, expression, symbol_types)
    assert result is expected_satisfiability


# =============================================================================
# Defensive dispatch branches
# =============================================================================


def test_z3_floor_divide_rejects_non_int_non_real_expression() -> None:
    """Test `_z3_floor_divide` raises `ValueError` on an expression that is neither."""
    fake = Mock()
    fake.is_real.return_value = False
    fake.is_int.return_value = False
    fake.__truediv__ = lambda self, other: fake

    with pytest.raises(ValueError, match=r"Unsupported floor divide expression type"):
        _z3_floor_divide(fake, fake)


def test_convert_expression_to_z3_returned_mapping_is_immutable() -> None:
    """Test the returned identifier-to-Z3 mapping rejects mutation."""
    x = mock_identifier("x", 0)
    converter = ExpressionToZ3Converter({x: SymbolType.INT})
    converter(IdentifierExpression(x))

    snapshot = converter.identifier_to_z3_expression

    with pytest.raises(TypeError):
        snapshot[x] = z3.Int("other")  # type: ignore[index]


def test_convert_expression_to_z3_raises_clear_error_on_missing_symbol_type() -> None:
    """Test the entry point reports identifiers absent from `symbol_types`."""
    x = mock_identifier("x", 0)
    y = mock_identifier("y", 1)
    expression = BinaryExpression(
        BinaryOperation.ADD, IdentifierExpression(x), IdentifierExpression(y)
    )

    with pytest.raises(KeyError, match=r"symbol_types is missing entries"):
        convert_expression_to_z3_expression(expression, {x: SymbolType.INT})


def test_z3_visit_identifier_rejects_invalid_symbol_type() -> None:
    """Test `visit_identifier_expression` rejects an unknown symbol-type entry."""
    identifier = mock_identifier("x", 0)
    converter = ExpressionToZ3Converter({identifier: "not-a-symbol-type"})  # type: ignore[dict-item]

    with pytest.raises(ValueError, match=r"Unsupported identifier type"):
        converter.visit_identifier_expression(IdentifierExpression(identifier))


@pytest.mark.parametrize(
    "value, expected_bool",
    [
        pytest.param("True", True, id="true_string"),
        pytest.param("False", False, id="false_string"),
    ],
)
def test_z3_visit_literal_bool_string_via_mock(value: str, expected_bool: bool) -> None:
    """Test the boolean-string branches map to `z3.BoolVal(True)` / `BoolVal(False)`."""
    converter = ExpressionToZ3Converter({})
    literal = Mock(spec=LiteralExpression)
    literal.value = value
    result = converter.visit_literal_expression(literal)
    assert result.eq(z3.BoolVal(expected_bool))


def test_z3_visit_literal_unsupported_value_raises() -> None:
    """Test `visit_literal_expression` raises on an unsupported literal value type."""
    converter = ExpressionToZ3Converter({})
    literal = Mock(spec=LiteralExpression)
    literal.value = object()

    with pytest.raises(TypeError, match=r"Unsupported literal type"):
        converter.visit_literal_expression(literal)


def test_z3_converter_get_noop_output_raises() -> None:
    """Test `ExpressionToZ3Converter.get_noop_output` raises `PassExecutionError`."""
    with pytest.raises(PassExecutionError, match=r"does not define noop output"):
        ExpressionToZ3Converter({}).get_noop_output(LiteralExpression(0))


def test_holds_for_all_free_assignments_raises_on_unexpected_solver_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test ``holds_for_all_free_assignments`` raises on an unexpected solver return."""
    monkeypatch.setattr(z3.Solver, "check", lambda self: object())
    considered, expression, symbol_types = _make_trivial_satisfiability_inputs()

    with pytest.raises(RuntimeError, match=r"Unexpected Z3 result"):
        holds_for_all_free_assignments(considered, expression, symbol_types)


def test_holds_for_all_free_assignments_logs_z3s_reason_for_unknown(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Test the `unknown`-result warning includes Z3's `reason_unknown()` text."""
    monkeypatch.setattr(z3.Solver, "check", lambda self: z3.unknown)
    monkeypatch.setattr(z3.Solver, "reason_unknown", lambda self: "timeout")
    considered, expression, symbol_types = _make_trivial_satisfiability_inputs()

    with caplog.at_level(
        logging.WARNING, logger="fhy_core.symbolic.expression.passes.z3"
    ):
        result = holds_for_all_free_assignments(considered, expression, symbol_types)

    assert result is None
    assert any(
        record.levelno == logging.WARNING and "timeout" in record.getMessage()
        for record in caplog.records
    ), "expected a WARNING log record mentioning z3's stated reason"


# =============================================================================
# does_expression_imply
# =============================================================================


def test_does_expression_imply_returns_true_when_antecedent_implies_consequent() -> (
    None
):
    """Test `does_expression_imply` reports True for `x >= 5 -> x > 3` over int x."""
    x = mock_identifier("x", 0)
    antecedent = BinaryExpression(
        BinaryOperation.GREATER_EQUAL,
        IdentifierExpression(x),
        LiteralExpression(5),
    )
    consequent = BinaryExpression(
        BinaryOperation.GREATER, IdentifierExpression(x), LiteralExpression(3)
    )
    assert does_expression_imply(antecedent, consequent, {x: SymbolType.INT}) is True


def test_does_expression_imply_returns_false_when_a_counterexample_exists() -> None:
    """Test `does_expression_imply` reports False for `x >= 5 -> x > 10` (x=6 fails)."""
    x = mock_identifier("x", 0)
    antecedent = BinaryExpression(
        BinaryOperation.GREATER_EQUAL,
        IdentifierExpression(x),
        LiteralExpression(5),
    )
    consequent = BinaryExpression(
        BinaryOperation.GREATER, IdentifierExpression(x), LiteralExpression(10)
    )
    assert does_expression_imply(antecedent, consequent, {x: SymbolType.INT}) is False


def test_does_expression_imply_holds_when_antecedent_is_false_everywhere() -> None:
    """Test ``False -> anything`` is True (vacuous truth from a false antecedent)."""
    x = mock_identifier("x", 0)
    contradictory_antecedent = BinaryExpression(
        BinaryOperation.LOGICAL_AND,
        BinaryExpression(
            BinaryOperation.GREATER,
            IdentifierExpression(x),
            LiteralExpression(10),
        ),
        BinaryExpression(
            BinaryOperation.LESS,
            IdentifierExpression(x),
            LiteralExpression(5),
        ),
    )
    consequent = BinaryExpression(
        BinaryOperation.EQUAL, IdentifierExpression(x), LiteralExpression(0)
    )
    assert (
        does_expression_imply(contradictory_antecedent, consequent, {x: SymbolType.INT})
        is True
    )


def test_does_expression_imply_returns_true_when_consequent_is_tautological() -> None:
    """Test ``anything -> True`` is True (tautological consequent)."""
    x = mock_identifier("x", 0)
    antecedent = BinaryExpression(
        BinaryOperation.GREATER, IdentifierExpression(x), LiteralExpression(0)
    )
    tautological_consequent = BinaryExpression(
        BinaryOperation.EQUAL, IdentifierExpression(x), IdentifierExpression(x)
    )
    assert (
        does_expression_imply(antecedent, tautological_consequent, {x: SymbolType.INT})
        is True
    )


def test_does_expression_imply_returns_none_when_z3_returns_unknown(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test `does_expression_imply` propagates Z3 ``unknown`` as ``None``."""
    monkeypatch.setattr(z3.Solver, "check", lambda self: z3.unknown)
    x = mock_identifier("x", 0)
    antecedent = BinaryExpression(
        BinaryOperation.GREATER, IdentifierExpression(x), LiteralExpression(0)
    )
    consequent = BinaryExpression(
        BinaryOperation.GREATER_EQUAL,
        IdentifierExpression(x),
        LiteralExpression(0),
    )
    assert does_expression_imply(antecedent, consequent, {x: SymbolType.INT}) is None


def test_does_expression_imply_handles_two_variable_implication() -> None:
    """Test `does_expression_imply` over two variables: ``x == y && x > 0 -> y > 0``."""
    x = mock_identifier("x", 0)
    y = mock_identifier("y", 1)
    antecedent = BinaryExpression(
        BinaryOperation.LOGICAL_AND,
        BinaryExpression(
            BinaryOperation.EQUAL,
            IdentifierExpression(x),
            IdentifierExpression(y),
        ),
        BinaryExpression(
            BinaryOperation.GREATER, IdentifierExpression(x), LiteralExpression(0)
        ),
    )
    consequent = BinaryExpression(
        BinaryOperation.GREATER, IdentifierExpression(y), LiteralExpression(0)
    )
    assert (
        does_expression_imply(
            antecedent, consequent, {x: SymbolType.INT, y: SymbolType.INT}
        )
        is True
    )


def test_does_expression_imply_raises_on_missing_symbol_type() -> None:
    """Test `does_expression_imply` raises `KeyError` when an identifier is unmapped."""
    x = mock_identifier("x", 0)
    y = mock_identifier("y", 1)
    antecedent = BinaryExpression(
        BinaryOperation.GREATER, IdentifierExpression(x), LiteralExpression(0)
    )
    consequent = BinaryExpression(
        BinaryOperation.GREATER, IdentifierExpression(y), LiteralExpression(0)
    )

    with pytest.raises(KeyError, match=r"symbol_types is missing entries"):
        does_expression_imply(antecedent, consequent, {x: SymbolType.INT})


# =============================================================================
# PiecewiseExpression -> z3.If
# =============================================================================


def test_convert_single_case_piecewise_expression_to_z3_if() -> None:
    """Test a one-case ``PiecewiseExpression`` lowers to the matching ``z3.If``."""
    x = mock_identifier("x", 0)
    expression = PiecewiseExpression(
        (
            BinaryExpression(
                BinaryOperation.GREATER,
                IdentifierExpression(x),
                LiteralExpression(0),
            ),
        ),
        (IdentifierExpression(x),),
        UnaryExpression(UnaryOperation.NEGATE, IdentifierExpression(x)),
    )

    z3_expression, _ = convert_expression_to_z3_expression(
        expression, {x: SymbolType.INT}
    )

    expected = z3.If(z3.Int("x_0") > z3.IntVal(0), z3.Int("x_0"), -z3.Int("x_0"))
    assert z3_expression.eq(expected)


def test_convert_piecewise_with_boolean_literal_branches_to_z3_if() -> None:
    """Test ``{True if cond; False otherwise}`` lowers to the matching ``z3.If``."""
    flag = mock_identifier("flag", 0)
    expression = PiecewiseExpression(
        (IdentifierExpression(flag),),
        (LiteralExpression(True),),
        LiteralExpression(False),
    )

    z3_expression, _ = convert_expression_to_z3_expression(
        expression, {flag: SymbolType.BOOL}
    )

    expected = z3.If(z3.Bool("flag_0"), z3.BoolVal(True), z3.BoolVal(False))
    assert z3_expression.eq(expected)


def test_multi_case_piecewise_z3_lowering_matches_hand_nested_encoding() -> None:
    """Test a flat multi-case piecewise's z3 lowering equals a hand-nested equivalent.

    First-match-wins semantics guarantee ``{1 if x > 0; -1 if x < 0;
    0 otherwise}`` denotes exactly the same value as the hand-nested
    ``{1 if x > 0; otherwise {-1 if x < 0; 0 otherwise}}``. Establishing
    ``does_expression_imply`` both ways over ``result == <expr>`` proves
    the flat right-folded ``z3.If`` chain the multi-case node lowers to
    is logically equivalent to the nested one.
    """
    x = mock_identifier("x", 0)
    result = mock_identifier("result", 1)
    x_expression = IdentifierExpression(x)
    result_expression = IdentifierExpression(result)

    flat = PiecewiseExpression(
        (x_expression > 0, x_expression < 0),
        (LiteralExpression(1), LiteralExpression(-1)),
        LiteralExpression(0),
    )
    nested = PiecewiseExpression(
        (x_expression > 0,),
        (LiteralExpression(1),),
        PiecewiseExpression(
            (x_expression < 0,), (LiteralExpression(-1),), LiteralExpression(0)
        ),
    )

    flat_holds = BinaryExpression(BinaryOperation.EQUAL, result_expression, flat)
    nested_holds = BinaryExpression(BinaryOperation.EQUAL, result_expression, nested)
    symbol_types = {x: SymbolType.INT, result: SymbolType.INT}

    assert does_expression_imply(flat_holds, nested_holds, symbol_types) is True
    assert does_expression_imply(nested_holds, flat_holds, symbol_types) is True


def test_piecewise_over_one_hundred_cases_z3_lowering_matches_hand_nested() -> None:
    """Test a 120-case piecewise's flat z3 lowering equals a hand-nested equivalent.

    Conditions are a monotonic, overlapping threshold ladder (``x < 1``,
    ``x < 2``, ..., ``x < NUM_CASES``) rather than the mutually exclusive
    equalities used elsewhere in this file, so more than one condition
    can hold at once and the comparison exercises which branch wins: the
    flat multi-case node's right-folded ``z3.If`` chain must pick the
    same first-matching branch as the hand-nested reference. The
    hand-nested reference is built from single-case
    ``PiecewiseExpression`` nodes, so its z3 lowering never itself
    exercises the multi-case fold, making it an independent ground truth
    for the comparison.
    """
    NUM_CASES = 120
    x = mock_identifier("x", 0)
    result = mock_identifier("result", 1)
    x_expression = IdentifierExpression(x)
    result_expression = IdentifierExpression(result)

    flat_cases = tuple(
        (x_expression < (i + 1), LiteralExpression(i)) for i in range(NUM_CASES)
    )
    flat = PiecewiseExpression(
        tuple(condition for condition, _ in flat_cases),
        tuple(value for _, value in flat_cases),
        LiteralExpression(-1),
    )
    nested: Expression = LiteralExpression(-1)
    for condition, value in reversed(flat_cases):
        nested = PiecewiseExpression((condition,), (value,), nested)

    flat_holds = BinaryExpression(BinaryOperation.EQUAL, result_expression, flat)
    nested_holds = BinaryExpression(BinaryOperation.EQUAL, result_expression, nested)
    symbol_types = {x: SymbolType.INT, result: SymbolType.INT}

    assert does_expression_imply(flat_holds, nested_holds, symbol_types) is True
    assert does_expression_imply(nested_holds, flat_holds, symbol_types) is True


# =============================================================================
# CallExpression interplay with the z3 converter
# =============================================================================


def test_convert_call_expression_to_z3_rejects_unresolved_call() -> None:
    """Test the z3 lowering rejects ``CallExpression`` (callers must inline first)."""
    expression = CallExpression("max", (LiteralExpression(1), LiteralExpression(2)))

    with pytest.raises(PassExecutionError, match="TypeError"):
        convert_expression_to_z3_expression(expression, {})


# =============================================================================
# Third-party characterization: Z3's Boolean-to-integer coercion
# =============================================================================
#
# The two tests below characterize the Z3 Python bindings themselves, not this
# package. `fhy_core.symbolic.constraint`'s bool sort-hazard screen exists only
# because Z3 silently coerces a Boolean operand in a numeric context instead of
# refusing the mixed-sort expression. Nothing else in the suite reaches that
# coercion: the screen short-circuits before the solver, so the tests around it
# never let Z3 see a hazardous expression. If either test fails, Z3's behavior
# changed and the screen's premise needs revisiting -- along with the
# docstrings on `ConstraintSystem.check_satisfiability`,
# `check_satisfiability_with_bindings`, and `_does_node_coerce_a_bool_operand`.
# That is the reading; it is not a broken test.


def test_z3_rewrites_a_bool_operand_compared_against_an_integer() -> None:
    """Test Z3 rewrites a lowered `BoolVal` in a numeric comparison to `If(b, 1, 0)`.

    A ``bool``-valued literal lowers to ``z3.BoolVal`` and an ``int``
    literal to ``z3.IntVal``. Comparing the two does not raise: the
    bindings insert the ``If(..., 1, 0)`` rewrite and report the
    comparison satisfiable, identifying ``True`` with ``1``. This
    package's literal semantics hold the two apart, so the lowering
    cannot be trusted to answer for it.
    """
    comparison = BinaryExpression(
        BinaryOperation.EQUAL, LiteralExpression(True), LiteralExpression(1)
    )

    lowered, _ = convert_expression_to_z3_expression(comparison, {})

    coerced_operand = lowered.children()[0]
    assert z3.is_app_of(coerced_operand, z3.Z3_OP_ITE)
    assert str(coerced_operand) == "If(True, 1, 0)"
    solver = z3.Solver()
    solver.add(lowered)
    assert solver.check() == z3.sat
    assert not LiteralExpression(True).is_structurally_equivalent(LiteralExpression(1))


def test_z3_bool_coercion_yields_a_model_this_package_rejects() -> None:
    """Test the coercion produces a satisfying assignment equating `True` with `1`.

    A ``SymbolType.BOOL`` identifier compared against an integer literal
    is satisfiable under Z3, and the witness assigns the identifier
    ``True`` -- Z3's answer to "can this Boolean equal 1?" is yes. Under
    this package's type-strict literal semantics ``True`` and ``1`` are
    distinct values, so a decided outcome read back from this lowering
    would contradict the semantics the constraint layer promises.
    """
    variable = mock_identifier("b", 0)
    comparison = BinaryExpression(
        BinaryOperation.EQUAL, IdentifierExpression(variable), LiteralExpression(1)
    )

    lowered, identifier_to_z3_expression = convert_expression_to_z3_expression(
        comparison, {variable: SymbolType.BOOL}
    )

    z3_variable = identifier_to_z3_expression[variable]
    assert str(lowered.children()[0]) == f"If({z3_variable}, 1, 0)"
    solver = z3.Solver()
    solver.add(lowered)
    assert solver.check() == z3.sat
    assert z3.is_true(solver.model()[z3_variable])
    assert not LiteralExpression(True).is_structurally_equivalent(LiteralExpression(1))
