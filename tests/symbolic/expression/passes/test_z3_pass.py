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
    NonBooleanLogicalOperandError,
    PiecewiseExpression,
    UnaryExpression,
    UnaryOperation,
    convert_expression_to_z3_expression,
    logical_and,
    logical_not,
    logical_or,
)
from fhy_core.symbolic.expression.errors import UndecidableError
from fhy_core.symbolic.expression.passes.z3 import (
    ExpressionToZ3Converter,
    _z3_floor_divide,
    assert_expression_implies,
    assert_holds_for_all_free_assignments,
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
        pytest.param(LiteralExpression(5), {}, z3.IntVal(5), id="literal_int"),
        pytest.param(LiteralExpression(5.5), {}, z3.RealVal(5.5), id="literal_float"),
        pytest.param(
            LiteralExpression(True), {}, z3.BoolVal(True), id="literal_bool_true"
        ),
        pytest.param(
            LiteralExpression(False), {}, z3.BoolVal(False), id="literal_bool_false"
        ),
        pytest.param(
            LiteralExpression("10.6"), {}, z3.RealVal(10.6), id="literal_numeric_string"
        ),
        pytest.param(
            LiteralExpression(0.1),
            {},
            z3.RatVal(*(0.1).as_integer_ratio()),
            id="literal_float_no_exact_binary_form",
        ),
        pytest.param(
            LiteralExpression("0.1"),
            {},
            z3.RatVal(1, 10),
            id="literal_numeric_string_exact_decimal",
        ),
        pytest.param(
            UnaryExpression(
                UnaryOperation.POSITIVE, IdentifierExpression(mock_identifier("x", 0))
            ),
            {mock_identifier("x", 0): SymbolType.REAL},
            z3.Real("x_0"),
            id="unary_positive",
        ),
        pytest.param(
            UnaryExpression(
                UnaryOperation.NEGATE, IdentifierExpression(mock_identifier("x", 0))
            ),
            {mock_identifier("x", 0): SymbolType.REAL},
            -z3.Real("x_0"),
            id="unary_negate",
        ),
        pytest.param(
            UnaryExpression(
                UnaryOperation.LOGICAL_NOT,
                IdentifierExpression(mock_identifier("x", 0)),
            ),
            {mock_identifier("x", 0): SymbolType.BOOL},
            z3.Not(z3.Bool("x_0")),
            id="unary_logical_not",
        ),
        pytest.param(
            BinaryExpression(
                BinaryOperation.ADD,
                IdentifierExpression(mock_identifier("x", 0)),
                LiteralExpression(5),
            ),
            {mock_identifier("x", 0): SymbolType.INT},
            z3.Int("x_0") + z3.IntVal(5),
            id="binary_add",
        ),
        pytest.param(
            BinaryExpression(
                BinaryOperation.SUBTRACT,
                IdentifierExpression(mock_identifier("x", 0)),
                LiteralExpression(5),
            ),
            {mock_identifier("x", 0): SymbolType.INT},
            z3.Int("x_0") - z3.IntVal(5),
            id="binary_subtract",
        ),
        pytest.param(
            BinaryExpression(
                BinaryOperation.MULTIPLY,
                IdentifierExpression(mock_identifier("x", 0)),
                LiteralExpression(5.5),
            ),
            {mock_identifier("x", 0): SymbolType.REAL},
            z3.Real("x_0") * z3.RealVal(5.5),
            id="binary_multiply",
        ),
        pytest.param(
            BinaryExpression(
                BinaryOperation.DIVIDE,
                IdentifierExpression(mock_identifier("x", 0)),
                LiteralExpression(5.5),
            ),
            {mock_identifier("x", 0): SymbolType.REAL},
            z3.Real("x_0") / z3.RealVal(5.5),
            id="binary_divide",
        ),
        pytest.param(
            BinaryExpression(
                BinaryOperation.FLOOR_DIVIDE,
                IdentifierExpression(mock_identifier("x", 0)),
                LiteralExpression(5),
            ),
            {mock_identifier("x", 0): SymbolType.INT},
            z3.Int("x_0") / z3.IntVal(5),
            id="binary_floor_divide_int_sort",
        ),
        pytest.param(
            BinaryExpression(
                BinaryOperation.FLOOR_DIVIDE,
                IdentifierExpression(mock_identifier("x", 0)),
                LiteralExpression(5),
            ),
            {mock_identifier("x", 0): SymbolType.REAL},
            z3.ToInt(z3.Real("x_0") / z3.IntVal(5)),
            id="binary_floor_divide_real_sort",
        ),
        pytest.param(
            BinaryExpression(
                BinaryOperation.MODULO,
                IdentifierExpression(mock_identifier("x", 0)),
                LiteralExpression(5),
            ),
            {mock_identifier("x", 0): SymbolType.INT},
            z3.Int("x_0") % z3.IntVal(5),
            id="binary_modulo",
        ),
        pytest.param(
            BinaryExpression(
                BinaryOperation.POWER,
                IdentifierExpression(mock_identifier("x", 0)),
                LiteralExpression(5),
            ),
            {mock_identifier("x", 0): SymbolType.INT},
            z3.Int("x_0") ** z3.IntVal(5),
            id="binary_power",
        ),
        pytest.param(
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
            id="binary_logical_and",
        ),
        pytest.param(
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
            id="binary_logical_or",
        ),
        pytest.param(
            BinaryExpression(
                BinaryOperation.EQUAL,
                IdentifierExpression(mock_identifier("x", 0)),
                LiteralExpression(5),
            ),
            {mock_identifier("x", 0): SymbolType.REAL},
            z3.Real("x_0") == z3.IntVal(5),
            id="binary_equal",
        ),
        pytest.param(
            BinaryExpression(
                BinaryOperation.NOT_EQUAL,
                IdentifierExpression(mock_identifier("x", 0)),
                LiteralExpression(5),
            ),
            {mock_identifier("x", 0): SymbolType.REAL},
            z3.Real("x_0") != z3.IntVal(5),
            id="binary_not_equal",
        ),
        pytest.param(
            BinaryExpression(
                BinaryOperation.LESS,
                IdentifierExpression(mock_identifier("x", 0)),
                LiteralExpression(5),
            ),
            {mock_identifier("x", 0): SymbolType.INT},
            z3.Int("x_0") < z3.IntVal(5),
            id="binary_less",
        ),
        pytest.param(
            BinaryExpression(
                BinaryOperation.LESS_EQUAL,
                IdentifierExpression(mock_identifier("x", 0)),
                LiteralExpression(5),
            ),
            {mock_identifier("x", 0): SymbolType.INT},
            z3.Int("x_0") <= z3.IntVal(5),
            id="binary_less_equal",
        ),
        pytest.param(
            BinaryExpression(
                BinaryOperation.GREATER,
                IdentifierExpression(mock_identifier("x", 0)),
                LiteralExpression(5),
            ),
            {mock_identifier("x", 0): SymbolType.INT},
            z3.Int("x_0") > z3.IntVal(5),
            id="binary_greater",
        ),
        pytest.param(
            BinaryExpression(
                BinaryOperation.GREATER_EQUAL,
                IdentifierExpression(mock_identifier("x", 0)),
                LiteralExpression(5),
            ),
            {mock_identifier("x", 0): SymbolType.INT},
            z3.Int("x_0") >= z3.IntVal(5),
            id="binary_greater_equal",
        ),
        pytest.param(
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
            id="nested_modulo_equals_zero",
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
        pytest.param(
            BinaryExpression(
                BinaryOperation.EQUAL,
                IdentifierExpression(mock_identifier("x", 0)),
                LiteralExpression(5),
            ),
            {mock_identifier("x", 0)},
            {mock_identifier("x", 0): SymbolType.INT},
            True,
            id="equality_is_universally_valid",
        ),
        pytest.param(
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
            id="contradictory_bounds_is_never_valid",
        ),
        pytest.param(
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
            id="tighter_bound_implies_looser_bound",
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


@pytest.fixture
def trivial_satisfiability_inputs() -> tuple[
    set[Identifier], Expression, dict[Identifier, SymbolType]
]:
    """Provide a trivially satisfiable (considered, expression, symbol_types) triple."""
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
    trivial_satisfiability_inputs: tuple[
        set[Identifier], Expression, dict[Identifier, SymbolType]
    ],
    solver_result: z3.CheckSatResult,
    expected_satisfiability: bool | None,
) -> None:
    """Test ``holds_for_all_free_assignments`` maps each z3 solver outcome correctly."""
    monkeypatch.setattr(z3.Solver, "check", lambda self: solver_result)
    considered, expression, symbol_types = trivial_satisfiability_inputs
    result = holds_for_all_free_assignments(considered, expression, symbol_types)
    assert result is expected_satisfiability


# =============================================================================
# Defensive dispatch branches
# =============================================================================


def test_z3_floor_divide_rejects_non_int_non_real_expression() -> None:
    """Test `_z3_floor_divide` raises `ValueError` on an expression that is neither."""
    fake = Mock(spec=z3.ArithRef)
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

    with pytest.raises(TypeError, match="does not support item assignment"):
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
    trivial_satisfiability_inputs: tuple[
        set[Identifier], Expression, dict[Identifier, SymbolType]
    ],
) -> None:
    """Test ``holds_for_all_free_assignments`` raises on an unexpected solver return."""
    monkeypatch.setattr(z3.Solver, "check", lambda self: object())
    considered, expression, symbol_types = trivial_satisfiability_inputs

    with pytest.raises(RuntimeError, match=r"Unexpected Z3 result"):
        holds_for_all_free_assignments(considered, expression, symbol_types)


def test_holds_for_all_free_assignments_logs_z3s_reason_for_unknown(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    trivial_satisfiability_inputs: tuple[
        set[Identifier], Expression, dict[Identifier, SymbolType]
    ],
) -> None:
    """Test the `unknown`-result warning includes Z3's `reason_unknown()` text."""
    monkeypatch.setattr(z3.Solver, "check", lambda self: z3.unknown)
    monkeypatch.setattr(z3.Solver, "reason_unknown", lambda self: "timeout")
    considered, expression, symbol_types = trivial_satisfiability_inputs

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
# UndecidableError.reason
# =============================================================================


def test_assert_holds_for_all_free_assignments_undecidable_error_carries_z3s_reason(
    monkeypatch: pytest.MonkeyPatch,
    trivial_satisfiability_inputs: tuple[
        set[Identifier], Expression, dict[Identifier, SymbolType]
    ],
) -> None:
    """Test the raised error's `reason` attribute carries Z3's stated reason."""
    monkeypatch.setattr(z3.Solver, "check", lambda self: z3.unknown)
    monkeypatch.setattr(z3.Solver, "reason_unknown", lambda self: "timeout")
    considered, expression, symbol_types = trivial_satisfiability_inputs

    with pytest.raises(UndecidableError, match="timeout") as exc_info:
        assert_holds_for_all_free_assignments(considered, expression, symbol_types)

    assert exc_info.value.reason == "timeout"


def test_assert_expression_implies_undecidable_error_carries_z3s_reason(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test the raised error's `reason` attribute carries Z3's stated reason."""
    monkeypatch.setattr(z3.Solver, "check", lambda self: z3.unknown)
    monkeypatch.setattr(z3.Solver, "reason_unknown", lambda self: "timeout")
    x = mock_identifier("x", 0)
    antecedent = BinaryExpression(
        BinaryOperation.GREATER, IdentifierExpression(x), LiteralExpression(0)
    )
    consequent = BinaryExpression(
        BinaryOperation.GREATER_EQUAL, IdentifierExpression(x), LiteralExpression(0)
    )

    with pytest.raises(UndecidableError, match="timeout") as exc_info:
        assert_expression_implies(antecedent, consequent, {x: SymbolType.INT})

    assert exc_info.value.reason == "timeout"


def test_assert_holds_for_all_free_assignments_bounds_z3_and_reports_its_timeout(
    monkeypatch: pytest.MonkeyPatch,
    trivial_satisfiability_inputs: tuple[
        set[Identifier], Expression, dict[Identifier, SymbolType]
    ],
) -> None:
    """Test a requested bound reaches Z3 and its timeout comes back as `reason`.

    The solver is stubbed to report what it reports when its bound runs
    out, so this pins the seam's own part of a timeout -- handing the
    bound to the solver, and carrying Z3's ``"timeout"`` text onto the
    raised error -- whatever the speed of the machine running it.
    """
    recorded: dict[str, object] = {}
    original_set = z3.Solver.set

    def record_solver_set_kwargs(
        self: z3.Solver, *args: object, **kwargs: object
    ) -> None:
        recorded.update(kwargs)
        original_set(self, *args, **kwargs)

    monkeypatch.setattr(z3.Solver, "set", record_solver_set_kwargs)
    monkeypatch.setattr(z3.Solver, "check", lambda self: z3.unknown)
    monkeypatch.setattr(z3.Solver, "reason_unknown", lambda self: "timeout")
    considered, expression, symbol_types = trivial_satisfiability_inputs

    with pytest.raises(UndecidableError, match="timeout") as exc_info:
        assert_holds_for_all_free_assignments(
            considered, expression, symbol_types, timeout_milliseconds=1
        )

    assert recorded.get("timeout") == 1
    assert exc_info.value.reason == "timeout"


def test_assert_holds_for_all_free_assignments_reports_a_real_z3_timeout() -> None:
    """Test a real Z3 timeout surfaces through the seam as `reason == "timeout"`.

    The expression claims ``x**3 + y**3 != z**3`` for all positive
    integers. That is true (the ``n = 3`` case of Fermat's Last Theorem),
    so Z3 can never find a counterexample, and ruling one out takes a
    proof by infinite descent, which none of Z3's arithmetic procedures
    attempt, so Z3 keeps searching until something stops it. Under a
    bound, the only thing that stops it is the bound running out, so the
    outcome depends on what Z3 can prove rather than on how fast the
    machine runs. The query is quantifier-free (no
    ``considered_identifiers``) because Z3 gives up on some quantified
    queries by itself, with an incompleteness reason rather than a
    timeout.

    Should a future Z3 decide the query, the test skips rather than
    fails: it can no longer observe a real timeout, which says nothing
    about the seam. The stubbed test above covers the seam's side of a
    timeout deterministically.
    """
    x = mock_identifier("x", 0)
    y = mock_identifier("y", 1)
    z = mock_identifier("z", 2)
    x_expression = IdentifierExpression(x)
    y_expression = IdentifierExpression(y)
    z_expression = IdentifierExpression(z)
    cube_sum = (
        x_expression * x_expression * x_expression
        + y_expression * y_expression * y_expression
    )
    z_cubed = z_expression * z_expression * z_expression
    no_positive_cube_sum_is_a_cube = logical_or(
        x_expression < 1,
        y_expression < 1,
        z_expression < 1,
        BinaryExpression(BinaryOperation.NOT_EQUAL, cube_sum, z_cubed),
    )
    symbol_types = {x: SymbolType.INT, y: SymbolType.INT, z: SymbolType.INT}

    try:
        decided = assert_holds_for_all_free_assignments(
            frozenset(),
            no_positive_cube_sum_is_a_cube,
            symbol_types,
            timeout_milliseconds=1,
        )
    except UndecidableError as error:
        reason = error.reason
    else:
        pytest.skip(
            f"Z3 decided the query ({decided}) before its bound ran out, so "
            "no real timeout was exercised."
        )

    assert reason == "timeout"


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


# =============================================================================
# Boolean connectives refuse a numeric operand rather than a backend exception
# =============================================================================


@pytest.mark.parametrize(
    "expression",
    [
        pytest.param(logical_and(LiteralExpression(2), LiteralExpression(4)), id="and"),
        pytest.param(logical_or(LiteralExpression(2), LiteralExpression(4)), id="or"),
        pytest.param(logical_not(LiteralExpression(2)), id="not"),
    ],
)
def test_convert_expression_to_z3_refuses_a_numeric_logical_operand(
    expression: Expression,
) -> None:
    """Test a Boolean connective over integers is refused before Z3 is called.

    ``z3.And``/``z3.Or``/``z3.Not`` reject an ``IntVal`` operand with a
    ``Z3Exception`` about sort mismatch, which the pass infrastructure
    wraps into a `PassExecutionError` naming Z3's SMT-LIB declaration
    rather than the expression. The bridge screens the shape out first and
    raises the package's own error, so the same ill-typed expression is
    reported the same way here as through the SymPy bridge.
    """
    with pytest.raises(NonBooleanLogicalOperandError) as exc_info:
        convert_expression_to_z3_expression(expression, {})

    assert type(exc_info.value) is NonBooleanLogicalOperandError
    assert not isinstance(exc_info.value, PassExecutionError)
    assert "Sort mismatch" not in str(exc_info.value)


def test_convert_expression_to_z3_reports_a_missing_symbol_type_before_the_screen() -> (
    None
):
    """Test the `symbol_types` precondition still wins over the operand screen.

    A caller who forgot a sort entry has a different bug from one who
    wrote an ill-typed connective, and the missing entry is the one they
    can act on without reading the tree. Screening first would mask it.
    """
    x = mock_identifier("x", 0)
    expression = logical_and(
        IdentifierExpression(x), logical_and(LiteralExpression(2), LiteralExpression(4))
    )

    with pytest.raises(KeyError, match="missing entries for identifiers"):
        convert_expression_to_z3_expression(expression, {})


@pytest.mark.parametrize(
    "expression",
    [
        pytest.param(logical_and(LiteralExpression(2), LiteralExpression(4)), id="and"),
        pytest.param(logical_or(LiteralExpression(2), LiteralExpression(4)), id="or"),
    ],
)
def test_assert_holds_for_all_free_assignments_refuses_a_numeric_connective(
    expression: Expression,
) -> None:
    """Test the strict universal-validity companion reports an ill-typed shape as such.

    The refusal is not `UndecidableError`: that error invites a retry with
    a larger ``timeout_milliseconds``, and no bound makes
    ``logical_and(2, 4)`` mean anything.
    """
    with pytest.raises(NonBooleanLogicalOperandError) as exc_info:
        assert_holds_for_all_free_assignments(frozenset(), expression, {})

    assert not isinstance(exc_info.value, UndecidableError)


def test_assert_expression_implies_refuses_a_numeric_connective_in_the_antecedent() -> (
    None
):
    """Test the implication companion screens the conjunction it encodes.

    The check lowers ``antecedent && !consequent``, so a numeric operand
    on either side reaches the same screen; placing it in the antecedent
    covers the composed tree the encoding builds.
    """
    antecedent = logical_and(LiteralExpression(2), LiteralExpression(4))

    with pytest.raises(NonBooleanLogicalOperandError):
        assert_expression_implies(antecedent, LiteralExpression(True), {})


@pytest.mark.parametrize(
    "expression, expected_satisfiable",
    [
        pytest.param(
            logical_and(LiteralExpression(True), LiteralExpression(False)),
            False,
            id="and_true_false",
        ),
        pytest.param(
            logical_and(LiteralExpression(True), LiteralExpression(True)),
            True,
            id="and_true_true",
        ),
        pytest.param(
            logical_or(LiteralExpression(True), LiteralExpression(False)),
            True,
            id="or_true_false",
        ),
        pytest.param(logical_not(LiteralExpression(False)), True, id="not_false"),
    ],
)
def test_z3_still_decides_a_ground_boolean_connective(
    expression: Expression, expected_satisfiable: bool
) -> None:
    """Test Boolean operands still lower and decide end to end through Z3.

    Refusing a numeric operand must not cost the Boolean case its
    decision, so each row pins the decided answer rather than only the
    absence of an exception.
    """
    lowered, _ = convert_expression_to_z3_expression(expression, {})
    solver = z3.Solver()
    solver.add(lowered)

    assert (solver.check() == z3.sat) is expected_satisfiable


def test_z3_decides_a_connective_mixing_an_identifier_with_a_boolean_literal() -> None:
    """Test a BOOL-sorted identifier conjoined with a Boolean literal still lowers.

    The realistic caller shape is symbolic rather than ground: the sort
    comes from ``symbol_types`` rather than from the node, so this covers
    the operand position the screen has to leave undetermined.
    """
    b = mock_identifier("b", 0)
    expression = logical_or(
        IdentifierExpression(b),
        UnaryExpression(UnaryOperation.LOGICAL_NOT, IdentifierExpression(b)),
    )

    result = holds_for_all_free_assignments(
        frozenset(), expression, {b: SymbolType.BOOL}
    )

    assert result is True


def test_z3_finds_a_counterexample_to_a_symbolic_conjunction() -> None:
    """Test a conjunction of a BOOL identifier and `False` is decided unsatisfiable."""
    b = mock_identifier("b", 0)
    expression = logical_and(IdentifierExpression(b), LiteralExpression(False))

    result = holds_for_all_free_assignments(
        frozenset({b}), expression, {b: SymbolType.BOOL}
    )

    assert result is False


# =============================================================================
# Non-finite float literals have no rational value
# =============================================================================


@pytest.mark.parametrize(
    "value",
    [
        pytest.param(float("inf"), id="positive_infinity"),
        pytest.param(float("-inf"), id="negative_infinity"),
        pytest.param(float("nan"), id="nan"),
    ],
)
def test_non_finite_float_literal_is_refused_rather_than_lowered(value: float) -> None:
    """Test a non-finite float literal fails lowering instead of reaching Z3.

    Lowering a float means naming the rational its bits denote, and an
    infinity or a NaN denotes no rational at all. The refusal has to be an
    error rather than some stand-in numeral: a substituted finite value
    would let the solver decide a query about a quantity Z3 was never
    given.
    """
    with pytest.raises(PassExecutionError) as exception_info:
        convert_expression_to_z3_expression(LiteralExpression(value), {})

    assert isinstance(exception_info.value.__cause__, (OverflowError, ValueError))
