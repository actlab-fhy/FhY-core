"""Tests for the backend-agnostic solver seam `fhy_core.symbolic.solver`."""

import inspect
from collections.abc import Callable

import pytest
import z3  # type: ignore[import-untyped]

from fhy_core.symbolic.expression import (
    BinaryExpression,
    BinaryOperation,
    Expression,
    IdentifierExpression,
    LiteralExpression,
    UnaryExpression,
    UnaryOperation,
)
from fhy_core.symbolic.expression.errors import UndecidableError
from fhy_core.symbolic.expression.passes.sympy import (
    simplify_expression as _bridge_simplify_expression,
)
from fhy_core.symbolic.solver import (
    SolverBackend,
    SolverCapabilityError,
    SolverQueryKind,
    assert_expression_implies,
    assert_holds_for_all_free_assignments,
    check_expression_satisfiability,
    does_expression_imply,
    get_backend_capabilities,
    holds_for_all_free_assignments,
    simplify_expression,
)
from fhy_core.symbolic.symbol_type import SymbolType

from .conftest import mock_identifier

# =============================================================================
# Capability table
# =============================================================================


def test_get_backend_capabilities_sympy_supports_only_simplification() -> None:
    """Test SYMPY's capability set is exactly `{SIMPLIFICATION}`."""
    assert get_backend_capabilities(SolverBackend.SYMPY) == frozenset(
        {SolverQueryKind.SIMPLIFICATION}
    )


def test_get_backend_capabilities_z3_supports_satisfiability_and_logic() -> None:
    """Test Z3's capability set is exactly the three logic query kinds."""
    assert get_backend_capabilities(SolverBackend.Z3) == frozenset(
        {
            SolverQueryKind.SATISFIABILITY,
            SolverQueryKind.IMPLICATION,
            SolverQueryKind.UNIVERSAL_VALIDITY,
        }
    )


@pytest.mark.parametrize(
    "call",
    [
        pytest.param(
            lambda: simplify_expression(LiteralExpression(1), backend=SolverBackend.Z3),
            id="simplify_with_z3",
        ),
        pytest.param(
            lambda: check_expression_satisfiability(
                LiteralExpression(True), {}, backend=SolverBackend.SYMPY
            ),
            id="satisfiability_with_sympy",
        ),
        pytest.param(
            lambda: does_expression_imply(
                LiteralExpression(True),
                LiteralExpression(True),
                {},
                backend=SolverBackend.SYMPY,
            ),
            id="implication_with_sympy",
        ),
        pytest.param(
            lambda: holds_for_all_free_assignments(
                frozenset(), LiteralExpression(True), {}, backend=SolverBackend.SYMPY
            ),
            id="universal_validity_with_sympy",
        ),
    ],
)
def test_incapable_backend_query_pairing_raises_solver_capability_error(
    call: Callable[[], object],
) -> None:
    """Test an incapable (backend, query) pairing raises `SolverCapabilityError`."""
    with pytest.raises(SolverCapabilityError):
        call()


def test_solver_capability_error_names_backend_and_query_kind() -> None:
    """Test the error message names both the offending backend and query kind."""
    with pytest.raises(SolverCapabilityError) as exc_info:
        simplify_expression(LiteralExpression(1), backend=SolverBackend.Z3)
    message = str(exc_info.value)
    assert "z3" in message.lower()
    assert "simplification" in message.lower()


# =============================================================================
# simplify_expression
# =============================================================================


def test_simplify_expression_matches_direct_bridge_pipeline() -> None:
    """Test the seam's simplification is structurally identical to the bridge's."""
    expression: Expression = BinaryExpression(
        BinaryOperation.ADD, LiteralExpression(1), LiteralExpression(2)
    )

    seam_result = simplify_expression(expression)
    bridge_result = _bridge_simplify_expression(expression)

    assert seam_result.is_structurally_equivalent(bridge_result)


def test_simplify_expression_with_environment_folds_to_a_literal() -> None:
    """Test substituting every free identifier lets the seam fold to a literal."""
    x = mock_identifier("x", 0)
    expression = BinaryExpression(
        BinaryOperation.ADD, IdentifierExpression(x), LiteralExpression(2)
    )

    result = simplify_expression(expression, {x: LiteralExpression(3)})

    assert isinstance(result, LiteralExpression)
    assert result.value == 5


# =============================================================================
# check_expression_satisfiability
# =============================================================================


@pytest.mark.z3
def test_check_expression_satisfiability_true_for_satisfiable_expression() -> None:
    """Test a satisfiable expression reports `True`."""
    x = mock_identifier("x", 0)
    expression = BinaryExpression(
        BinaryOperation.GREATER, IdentifierExpression(x), LiteralExpression(0)
    )

    assert check_expression_satisfiability(expression, {x: SymbolType.INT}) is True


@pytest.mark.z3
def test_check_expression_satisfiability_false_for_unsatisfiable_expression() -> None:
    """Test a provably unsatisfiable expression reports `False`."""
    x = mock_identifier("x", 0)
    expression = BinaryExpression(
        BinaryOperation.LOGICAL_AND,
        BinaryExpression(
            BinaryOperation.GREATER, IdentifierExpression(x), LiteralExpression(10)
        ),
        BinaryExpression(
            BinaryOperation.LESS, IdentifierExpression(x), LiteralExpression(5)
        ),
    )

    assert check_expression_satisfiability(expression, {x: SymbolType.INT}) is False


def test_check_expression_satisfiability_true_for_closed_true_expression() -> None:
    """Test a closed (no free identifiers) `True` expression needs no symbol types."""
    assert check_expression_satisfiability(LiteralExpression(True), {}) is True


def test_check_expression_satisfiability_false_for_closed_false_expression() -> None:
    """Test a closed (no free identifiers) `False` expression needs no symbol types."""
    assert check_expression_satisfiability(LiteralExpression(False), {}) is False


@pytest.mark.z3
def test_check_expression_satisfiability_returns_none_on_unknown(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test a solver `unknown` result surfaces as `None`."""
    monkeypatch.setattr(z3.Solver, "check", lambda self: z3.unknown)
    x = mock_identifier("x", 0)
    expression = BinaryExpression(
        BinaryOperation.GREATER, IdentifierExpression(x), LiteralExpression(0)
    )

    assert check_expression_satisfiability(expression, {x: SymbolType.INT}) is None


def test_check_expression_satisfiability_raises_key_error_for_missing_symbol_type() -> (
    None
):
    """Test a missing `symbol_types` entry propagates `KeyError`."""
    x = mock_identifier("x", 0)
    expression = BinaryExpression(
        BinaryOperation.GREATER, IdentifierExpression(x), LiteralExpression(0)
    )

    with pytest.raises(KeyError):
        check_expression_satisfiability(expression, {})


# =============================================================================
# does_expression_imply / holds_for_all_free_assignments parity
# =============================================================================


@pytest.mark.z3
def test_does_expression_imply_parity_with_pre_seam_contract() -> None:
    """Test `does_expression_imply` reports True for `x >= 5 -> x > 3` over int x."""
    x = mock_identifier("x", 0)
    antecedent = BinaryExpression(
        BinaryOperation.GREATER_EQUAL, IdentifierExpression(x), LiteralExpression(5)
    )
    consequent = BinaryExpression(
        BinaryOperation.GREATER, IdentifierExpression(x), LiteralExpression(3)
    )

    assert does_expression_imply(antecedent, consequent, {x: SymbolType.INT}) is True


@pytest.mark.z3
def test_holds_for_all_free_assignments_parity_with_pre_seam_contract() -> None:
    """Test the universal-validity query degenerates with no considered ids."""
    x = mock_identifier("x", 0)
    expression = BinaryExpression(
        BinaryOperation.GREATER_EQUAL,
        BinaryExpression(
            BinaryOperation.MULTIPLY, IdentifierExpression(x), IdentifierExpression(x)
        ),
        LiteralExpression(0),
    )

    assert (
        holds_for_all_free_assignments(frozenset(), expression, {x: SymbolType.REAL})
        is True
    )


@pytest.mark.z3
def test_assert_expression_implies_raises_undecidable_error_on_unknown(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test the strict variant raises `UndecidableError` instead of returning `None`."""
    monkeypatch.setattr(z3.Solver, "check", lambda self: z3.unknown)
    x = mock_identifier("x", 0)
    antecedent = BinaryExpression(
        BinaryOperation.GREATER, IdentifierExpression(x), LiteralExpression(0)
    )
    consequent = BinaryExpression(
        BinaryOperation.GREATER_EQUAL, IdentifierExpression(x), LiteralExpression(0)
    )

    with pytest.raises(UndecidableError):
        assert_expression_implies(antecedent, consequent, {x: SymbolType.INT})


@pytest.mark.z3
def test_assert_holds_for_all_free_assignments_raises_undecidable_error_on_unknown(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test the strict variant raises `UndecidableError` instead of returning `None`."""
    monkeypatch.setattr(z3.Solver, "check", lambda self: z3.unknown)
    x = mock_identifier("x", 0)
    expression = UnaryExpression(
        UnaryOperation.LOGICAL_NOT,
        BinaryExpression(
            BinaryOperation.EQUAL, IdentifierExpression(x), IdentifierExpression(x)
        ),
    )

    with pytest.raises(UndecidableError):
        assert_holds_for_all_free_assignments(
            frozenset(), expression, {x: SymbolType.INT}
        )


# =============================================================================
# timeout_milliseconds
# =============================================================================


def test_simplify_expression_signature_has_no_timeout_parameter() -> None:
    """Test the SymPy-backed seam function has no `timeout_milliseconds` parameter."""
    parameters = inspect.signature(simplify_expression).parameters
    assert "timeout_milliseconds" not in parameters


@pytest.mark.parametrize("bad_timeout", [0, -1, -1000])
def test_check_expression_satisfiability_rejects_non_positive_timeout(
    bad_timeout: int,
) -> None:
    """Test a zero or negative `timeout_milliseconds` raises `ValueError`."""
    with pytest.raises(ValueError, match=r"positive"):
        check_expression_satisfiability(
            LiteralExpression(True), {}, timeout_milliseconds=bad_timeout
        )


@pytest.mark.z3
def test_does_expression_imply_accepts_a_positive_timeout_without_raising() -> None:
    """Test a positive `timeout_milliseconds` is accepted and does not error."""
    x = mock_identifier("x", 0)
    antecedent = BinaryExpression(
        BinaryOperation.GREATER, IdentifierExpression(x), LiteralExpression(0)
    )
    consequent = BinaryExpression(
        BinaryOperation.GREATER_EQUAL, IdentifierExpression(x), LiteralExpression(0)
    )

    result = does_expression_imply(
        antecedent, consequent, {x: SymbolType.INT}, timeout_milliseconds=5000
    )

    assert result is True


@pytest.mark.z3
def test_timeout_milliseconds_is_threaded_to_the_z3_solver(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test `timeout_milliseconds` is set on the underlying `z3.Solver`."""
    recorded: dict[str, object] = {}
    original_set = z3.Solver.set

    def _recording_set(self: z3.Solver, *args: object, **kwargs: object) -> None:
        recorded.update(kwargs)
        original_set(self, *args, **kwargs)

    monkeypatch.setattr(z3.Solver, "set", _recording_set)
    x = mock_identifier("x", 0)
    expression = BinaryExpression(
        BinaryOperation.GREATER, IdentifierExpression(x), LiteralExpression(0)
    )

    check_expression_satisfiability(
        expression, {x: SymbolType.INT}, timeout_milliseconds=2500
    )

    assert recorded.get("timeout") == 2500
