"""Tests for the backend-agnostic solver seam `fhy_core.symbolic.solver`."""

import inspect
import logging
from collections.abc import Callable

import pytest
import z3  # type: ignore[import-untyped]
from immutabledict import immutabledict

from fhy_core.identifier import Identifier
from fhy_core.symbolic.expression import (
    BinaryExpression,
    BinaryOperation,
    Expression,
    IdentifierExpression,
    LiteralExpression,
    NonBooleanLogicalOperandError,
    PiecewiseExpression,
    UnaryExpression,
    UnaryOperation,
    get_native_constant_identifier,
)
from fhy_core.symbolic.expression.errors import UndecidableError
from fhy_core.symbolic.expression.passes.sympy import (
    simplify_expression as _bridge_simplify_expression,
)

# White-box import: no public accessor enumerates every backend's capability
# entry at once (`get_backend_capabilities` only looks up one backend at a
# time), so this drift guard reads the table directly, matching the
# convention in test_numpy_evaluator.py's lowering-table coverage tests.
from fhy_core.symbolic.solver import (
    _BACKEND_CAPABILITIES,
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


def test_every_solver_backend_has_a_capability_table_entry() -> None:
    """Test every `SolverBackend` member has an entry in the capability table.

    `get_backend_capabilities` subscripts the table directly, so a member
    added without an entry would otherwise raise a raw `KeyError` only when
    that backend is first queried; this makes the drift a deterministic
    failure instead.
    """
    assert set(_BACKEND_CAPABILITIES) == set(SolverBackend)


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
        pytest.param(
            lambda: assert_holds_for_all_free_assignments(
                frozenset(), LiteralExpression(True), {}, backend=SolverBackend.SYMPY
            ),
            id="assert_universal_validity_with_sympy",
        ),
        pytest.param(
            lambda: assert_expression_implies(
                LiteralExpression(True),
                LiteralExpression(True),
                {},
                backend=SolverBackend.SYMPY,
            ),
            id="assert_implication_with_sympy",
        ),
    ],
)
def test_incapable_backend_query_pairing_raises_solver_capability_error(
    call: Callable[[], object],
) -> None:
    """Test an incapable (backend, query) pairing raises `SolverCapabilityError`."""
    with pytest.raises(SolverCapabilityError, match="cannot answer"):
        call()


def test_solver_capability_error_names_backend_and_query_kind() -> None:
    """Test the error message names both the offending backend and query kind."""
    with pytest.raises(SolverCapabilityError, match="cannot answer") as exc_info:
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


def test_simplify_expression_with_residual_variable_reduces_to_the_identifier() -> None:
    """Test an open ``x + 0`` simplifies to the bare identifier expression.

    With no environment binding ``x``, simplification cannot fold to a
    literal; it still algebraically reduces the additive identity away,
    leaving exactly the identifier operand rather than a residual
    ``x + 0`` tree.
    """
    x = mock_identifier("x", 0)
    expression = BinaryExpression(
        BinaryOperation.ADD, IdentifierExpression(x), LiteralExpression(0)
    )

    result = simplify_expression(expression)

    assert result.is_structurally_equivalent(IdentifierExpression(x))


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


@pytest.mark.z3
def test_check_expression_satisfiability_true_for_closed_true_expression() -> None:
    """Test a closed (no free identifiers) `True` expression needs no symbol types."""
    assert check_expression_satisfiability(LiteralExpression(True), {}) is True


@pytest.mark.z3
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


@pytest.mark.z3
def test_check_expression_satisfiability_raises_key_error_for_missing_symbol_type() -> (
    None
):
    """Test a missing `symbol_types` entry propagates `KeyError`."""
    x = mock_identifier("x", 0)
    expression = BinaryExpression(
        BinaryOperation.GREATER, IdentifierExpression(x), LiteralExpression(0)
    )

    with pytest.raises(KeyError, match="symbol_types is missing"):
        check_expression_satisfiability(expression, {})


@pytest.mark.z3
def test_check_expression_satisfiability_is_threaded_through_symbol_type() -> None:
    """Test `0 < x < 1` is unsatisfiable over INT but satisfiable over REAL.

    Proves ``symbol_types`` is actually threaded through to the Z3 bridge
    rather than ignored: the same expression is decided differently
    depending on the sort assigned to its one free identifier.
    """
    x = mock_identifier("x", 0)
    expression = BinaryExpression(
        BinaryOperation.LOGICAL_AND,
        BinaryExpression(
            BinaryOperation.LESS, LiteralExpression(0), IdentifierExpression(x)
        ),
        BinaryExpression(
            BinaryOperation.LESS, IdentifierExpression(x), LiteralExpression(1)
        ),
    )

    assert check_expression_satisfiability(expression, {x: SymbolType.INT}) is False
    assert check_expression_satisfiability(expression, {x: SymbolType.REAL}) is True


# =============================================================================
# does_expression_imply / holds_for_all_free_assignments parity
# =============================================================================


@pytest.mark.z3
def test_does_expression_imply_reports_true_for_a_valid_implication() -> None:
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
def test_holds_for_all_free_assignments_reports_true_with_no_considered_ids() -> None:
    """Test the universal-validity query holds with no considered identifiers."""
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
def test_assert_expression_implies_returns_true_for_a_valid_implication() -> None:
    """Test the strict variant returns a decided `True` without raising."""
    x = mock_identifier("x", 0)
    antecedent = BinaryExpression(
        BinaryOperation.GREATER_EQUAL, IdentifierExpression(x), LiteralExpression(5)
    )
    consequent = BinaryExpression(
        BinaryOperation.GREATER, IdentifierExpression(x), LiteralExpression(3)
    )

    assert (
        assert_expression_implies(antecedent, consequent, {x: SymbolType.INT}) is True
    )


@pytest.mark.z3
def test_assert_expression_implies_returns_false_for_a_counterexample() -> None:
    """Test the strict variant returns a decided `False` without raising."""
    x = mock_identifier("x", 0)
    antecedent = BinaryExpression(
        BinaryOperation.GREATER_EQUAL, IdentifierExpression(x), LiteralExpression(5)
    )
    consequent = BinaryExpression(
        BinaryOperation.GREATER, IdentifierExpression(x), LiteralExpression(10)
    )

    assert (
        assert_expression_implies(antecedent, consequent, {x: SymbolType.INT}) is False
    )


@pytest.mark.z3
def test_assert_holds_for_all_free_assignments_returns_true_without_raising() -> None:
    """Test the strict variant returns a decided `True` without raising."""
    x = mock_identifier("x", 0)
    expression = BinaryExpression(
        BinaryOperation.GREATER_EQUAL,
        BinaryExpression(
            BinaryOperation.MULTIPLY, IdentifierExpression(x), IdentifierExpression(x)
        ),
        LiteralExpression(0),
    )

    assert (
        assert_holds_for_all_free_assignments(
            frozenset(), expression, {x: SymbolType.REAL}
        )
        is True
    )


@pytest.mark.z3
def test_assert_holds_for_all_free_assignments_returns_false_without_raising() -> None:
    """Test the strict variant returns a decided `False` without raising.

    ``x < N and x > N`` has no witness ``x`` for any free ``N``, so the
    universal-validity query is decided ``False`` rather than unknown.
    """
    x = mock_identifier("x", 0)
    n = mock_identifier("N", 1)
    expression = BinaryExpression(
        BinaryOperation.LOGICAL_AND,
        BinaryExpression(
            BinaryOperation.LESS, IdentifierExpression(x), IdentifierExpression(n)
        ),
        BinaryExpression(
            BinaryOperation.GREATER, IdentifierExpression(x), IdentifierExpression(n)
        ),
    )

    assert (
        assert_holds_for_all_free_assignments(
            {x}, expression, {x: SymbolType.INT, n: SymbolType.INT}
        )
        is False
    )


@pytest.mark.z3
def test_assert_expression_implies_raises_undecidable_error_on_unknown(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test the strict variant raises `UndecidableError` naming Z3's stated reason."""
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


@pytest.mark.z3
def test_assert_holds_for_all_free_assignments_raises_undecidable_error_on_unknown(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test the strict variant raises `UndecidableError` naming Z3's stated reason."""
    monkeypatch.setattr(z3.Solver, "check", lambda self: z3.unknown)
    monkeypatch.setattr(z3.Solver, "reason_unknown", lambda self: "timeout")
    x = mock_identifier("x", 0)
    expression = UnaryExpression(
        UnaryOperation.LOGICAL_NOT,
        BinaryExpression(
            BinaryOperation.EQUAL, IdentifierExpression(x), IdentifierExpression(x)
        ),
    )

    with pytest.raises(UndecidableError, match="timeout") as exc_info:
        assert_holds_for_all_free_assignments(
            frozenset(), expression, {x: SymbolType.INT}
        )

    assert exc_info.value.reason == "timeout"


# =============================================================================
# `symbol_types` accepts any `Mapping`, not only `dict`
# =============================================================================


@pytest.mark.z3
def test_check_expression_satisfiability_accepts_an_immutabledict() -> None:
    """Test the satisfiability seam accepts an `immutabledict` `symbol_types`."""
    x = mock_identifier("x", 0)
    expression = BinaryExpression(
        BinaryOperation.GREATER, IdentifierExpression(x), LiteralExpression(0)
    )
    symbol_types = immutabledict({x: SymbolType.INT})

    assert check_expression_satisfiability(expression, symbol_types) is True


@pytest.mark.z3
def test_does_expression_imply_accepts_an_immutabledict_symbol_types() -> None:
    """Test the implication seam accepts an `immutabledict` `symbol_types`."""
    x = mock_identifier("x", 0)
    antecedent = BinaryExpression(
        BinaryOperation.GREATER_EQUAL, IdentifierExpression(x), LiteralExpression(5)
    )
    consequent = BinaryExpression(
        BinaryOperation.GREATER, IdentifierExpression(x), LiteralExpression(3)
    )
    symbol_types = immutabledict({x: SymbolType.INT})

    assert does_expression_imply(antecedent, consequent, symbol_types) is True


@pytest.mark.z3
def test_holds_for_all_free_assignments_accepts_an_immutabledict() -> None:
    """Test the universal-validity seam accepts an `immutabledict` `symbol_types`."""
    x = mock_identifier("x", 0)
    expression = BinaryExpression(
        BinaryOperation.GREATER_EQUAL,
        BinaryExpression(
            BinaryOperation.MULTIPLY, IdentifierExpression(x), IdentifierExpression(x)
        ),
        LiteralExpression(0),
    )
    symbol_types = immutabledict({x: SymbolType.REAL})

    assert holds_for_all_free_assignments(frozenset(), expression, symbol_types) is True


@pytest.mark.z3
def test_assert_holds_for_all_free_assignments_with_immutabledict_raises_on_unknown(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test the strict variant raises when undecided, given an `immutabledict`.

    Z3 is forced to answer ``unknown``, so the claim's truth does not
    matter; the test pins that a `Mapping` other than `dict` reaches the
    raising path.
    """
    monkeypatch.setattr(z3.Solver, "check", lambda self: z3.unknown)
    monkeypatch.setattr(z3.Solver, "reason_unknown", lambda self: "timeout")
    x = mock_identifier("x", 0)
    expression = BinaryExpression(
        BinaryOperation.GREATER, IdentifierExpression(x), LiteralExpression(0)
    )
    symbol_types = immutabledict({x: SymbolType.INT})

    with pytest.raises(UndecidableError, match="timeout"):
        assert_holds_for_all_free_assignments(frozenset(), expression, symbol_types)


@pytest.mark.z3
def test_assert_expression_implies_with_immutabledict_symbol_types_raises_on_unknown(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test the strict variant raises when undecided, given an `immutabledict`.

    Z3 is forced to answer ``unknown``, so the claim's truth does not
    matter; the test pins that a `Mapping` other than `dict` reaches the
    raising path.
    """
    monkeypatch.setattr(z3.Solver, "check", lambda self: z3.unknown)
    monkeypatch.setattr(z3.Solver, "reason_unknown", lambda self: "timeout")
    x = mock_identifier("x", 0)
    antecedent = BinaryExpression(
        BinaryOperation.GREATER, IdentifierExpression(x), LiteralExpression(5)
    )
    consequent = BinaryExpression(
        BinaryOperation.GREATER, IdentifierExpression(x), LiteralExpression(10)
    )
    symbol_types = immutabledict({x: SymbolType.INT})

    with pytest.raises(UndecidableError, match="timeout"):
        assert_expression_implies(antecedent, consequent, symbol_types)


# =============================================================================
# UndecidableError.reason
# =============================================================================


def test_undecidable_error_reason_defaults_to_empty_string() -> None:
    """Test `UndecidableError.reason` defaults to an empty string when omitted."""
    error = UndecidableError("boom")

    assert error.reason == ""


def test_undecidable_error_reason_reflects_the_constructor_argument() -> None:
    """Test `UndecidableError.reason` reflects the `reason` constructor argument."""
    error = UndecidableError("boom", reason="timeout")

    assert error.reason == "timeout"


# =============================================================================
# considered_identifiers validation
# =============================================================================


def test_holds_for_all_free_assignments_rejects_unmapped_considered_id() -> None:
    """Test a considered identifier absent from `symbol_types` raises `KeyError`.

    `ghost` does not appear in the expression at all, so only the
    considered-identifiers side of the contract requires it to have a
    `symbol_types` entry.
    """
    ghost = mock_identifier("ghost", 0)

    with pytest.raises(KeyError, match="symbol_types is missing"):
        holds_for_all_free_assignments(frozenset({ghost}), LiteralExpression(True), {})


def test_assert_holds_for_all_free_assignments_rejects_unmapped_considered_id() -> None:
    """Test the strict variant also raises `KeyError` for an unmapped considered id."""
    ghost = mock_identifier("ghost", 0)

    with pytest.raises(KeyError, match="symbol_types is missing"):
        assert_holds_for_all_free_assignments(
            frozenset({ghost}), LiteralExpression(True), {}
        )


@pytest.mark.z3
def test_holds_for_all_free_assignments_accepts_a_mapped_considered_id() -> None:
    """Test a considered identifier with a `symbol_types` entry does not raise.

    An identifier that is legitimately considered (present in
    `symbol_types`) even though it is absent from the expression must
    not trip the same `KeyError` reserved for a genuinely unmapped
    identifier.
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


@pytest.mark.z3
def test_assert_holds_for_all_free_assignments_accepts_a_mapped_considered_id() -> None:
    """Test the strict variant also accepts a properly-mapped considered identifier."""
    x = mock_identifier("x", 0)
    unused = mock_identifier("u", 1)
    expression = BinaryExpression(
        BinaryOperation.EQUAL, IdentifierExpression(x), LiteralExpression(5)
    )

    result = assert_holds_for_all_free_assignments(
        {x, unused},
        expression,
        {x: SymbolType.INT, unused: SymbolType.INT},
    )

    assert result is True


# =============================================================================
# Lowering hazard screens
# =============================================================================

_SOLVER_LOGGER_NAME = "fhy_core.symbolic.solver"


def _collect_solver_warning_messages(caplog: pytest.LogCaptureFixture) -> list[str]:
    """Return the solver module's WARNING-level messages captured by ``caplog``."""
    return [
        record.getMessage()
        for record in caplog.records
        if record.levelno == logging.WARNING and record.name == _SOLVER_LOGGER_NAME
    ]


def test_check_expression_satisfiability_bool_coercion_hazard_returns_none(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Test a bool literal compared against an INT-sorted identifier is screened.

    Mirrors the ``InSetConstraint``-style shape ``x in {True}``: the Z3
    bindings would coerce the ``True`` literal to ``1`` in a numeric
    context, so the seam refuses to lower the comparison and reports
    None instead of a decided-but-wrong outcome.
    """
    x = mock_identifier("x", 0)
    expression = BinaryExpression(
        BinaryOperation.EQUAL, IdentifierExpression(x), LiteralExpression(True)
    )

    with caplog.at_level(logging.WARNING):
        result = check_expression_satisfiability(expression, {x: SymbolType.INT})

    assert result is None
    messages = _collect_solver_warning_messages(caplog)
    assert messages, "expected a WARNING naming the hazardous node"
    assert "check_expression_satisfiability" in messages[0]
    assert repr(x) in messages[0]


@pytest.mark.z3
def test_check_expression_satisfiability_bool_literal_under_bool_sort_decided() -> None:
    """Test the same comparison under a BOOL-sorted identifier is not screened.

    Contrasts the hazard: a BOOL-sorted ``x`` lowers ``x == True``
    faithfully, so the comparison reaches the solver and decides
    normally.
    """
    x = mock_identifier("x", 0)
    expression = BinaryExpression(
        BinaryOperation.EQUAL, IdentifierExpression(x), LiteralExpression(True)
    )

    assert check_expression_satisfiability(expression, {x: SymbolType.BOOL}) is True


def test_check_expression_satisfiability_division_hazard_returns_none(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Test DIVIDE by a non-literal divisor is screened.

    ``x / x``'s divisor could be zero for some assignment, and the Z3
    bridge's satisfiability encoding for division is unsound around a
    zero divisor, so the seam refuses to lower ``x / x != 1`` and
    reports None instead.
    """
    x = mock_identifier("x", 0)
    expression = BinaryExpression(
        BinaryOperation.NOT_EQUAL,
        BinaryExpression(
            BinaryOperation.DIVIDE, IdentifierExpression(x), IdentifierExpression(x)
        ),
        LiteralExpression(1),
    )

    with caplog.at_level(logging.WARNING):
        result = check_expression_satisfiability(expression, {x: SymbolType.REAL})

    assert result is None
    messages = _collect_solver_warning_messages(caplog)
    assert messages, "expected a WARNING naming the hazardous division node"
    assert "check_expression_satisfiability" in messages[0]


@pytest.mark.z3
def test_check_expression_satisfiability_modulo_by_nonzero_literal_stays_decided() -> (
    None
):
    """Test MODULO by a nonzero literal divisor is not screened.

    Contrasts the hazard: ``x % 2 == 0``'s divisor is a provably nonzero
    literal, so the comparison reaches the solver and decides normally.
    """
    x = mock_identifier("x", 0)
    expression = BinaryExpression(
        BinaryOperation.EQUAL,
        BinaryExpression(
            BinaryOperation.MODULO, IdentifierExpression(x), LiteralExpression(2)
        ),
        LiteralExpression(0),
    )

    assert check_expression_satisfiability(expression, {x: SymbolType.INT}) is True


@pytest.mark.parametrize(
    "claimed_quotient",
    [-4, -3],
    ids=["floor_semantics_value", "euclidean_semantics_value"],
)
def test_check_expression_satisfiability_floor_divide_negative_divisor_is_screened(
    claimed_quotient: int,
) -> None:
    """Test `7 // -2` compared against either candidate quotient is screened.

    Z3 lowers `FLOOR_DIVIDE` on two integers to its `div`, which is
    Euclidean, not floor: `7 // -2` is `-4` under this package's floor
    semantics (and what `simplify_expression` reports), but Z3's
    Euclidean division reports `-3` for the same inputs. The two
    candidate quotients are mutually exclusive, so screening the
    negative divisor must report both as undecided rather than decide
    either one definitively.
    """
    floor_div = BinaryExpression(
        BinaryOperation.FLOOR_DIVIDE, LiteralExpression(7), LiteralExpression(-2)
    )
    expression = BinaryExpression(
        BinaryOperation.EQUAL, floor_div, LiteralExpression(claimed_quotient)
    )

    assert check_expression_satisfiability(expression, {}) is None


def test_check_expression_satisfiability_floor_divide_negative_divisor_warns(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Test a screened negative FLOOR_DIVIDE divisor logs a WARNING naming the node."""
    floor_div = BinaryExpression(
        BinaryOperation.FLOOR_DIVIDE, LiteralExpression(7), LiteralExpression(-2)
    )
    expression = BinaryExpression(
        BinaryOperation.EQUAL, floor_div, LiteralExpression(-4)
    )

    with caplog.at_level(logging.WARNING):
        result = check_expression_satisfiability(expression, {})

    assert result is None
    messages = _collect_solver_warning_messages(caplog)
    assert messages, "expected a WARNING naming the hazardous division node"
    assert "check_expression_satisfiability" in messages[0]
    assert repr(floor_div) in messages[0]


def test_check_expression_satisfiability_modulo_negative_divisor_warns(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Test a screened negative MODULO divisor logs a WARNING naming the node.

    Z3 lowers `MODULO` on two integers to Euclidean modulo, whose result
    always shares the divisor's sign rather than the dividend's: `7 % -2`
    is `-1` under this package's floor semantics but Z3 reports `1` for
    the same inputs.
    """
    modulo = BinaryExpression(
        BinaryOperation.MODULO, LiteralExpression(7), LiteralExpression(-2)
    )
    expression = BinaryExpression(BinaryOperation.EQUAL, modulo, LiteralExpression(-1))

    with caplog.at_level(logging.WARNING):
        result = check_expression_satisfiability(expression, {})

    assert result is None
    messages = _collect_solver_warning_messages(caplog)
    assert messages, "expected a WARNING naming the hazardous division node"
    assert "check_expression_satisfiability" in messages[0]
    assert repr(modulo) in messages[0]


@pytest.mark.z3
def test_check_expression_satisfiability_floor_divide_positive_literal_decided() -> (
    None
):
    """Test FLOOR_DIVIDE by a positive literal divisor is not screened.

    Contrasts the hazard: a positive literal divisor lowers Z3's
    Euclidean division faithfully to this package's floor semantics --
    the two conventions agree whenever the divisor is positive -- so the
    comparison reaches the solver and decides normally.
    """
    expression = BinaryExpression(
        BinaryOperation.EQUAL,
        BinaryExpression(
            BinaryOperation.FLOOR_DIVIDE, LiteralExpression(7), LiteralExpression(2)
        ),
        LiteralExpression(3),
    )

    assert check_expression_satisfiability(expression, {}) is True


@pytest.mark.parametrize(
    "operation",
    [BinaryOperation.FLOOR_DIVIDE, BinaryOperation.MODULO],
    ids=["floor_divide", "modulo"],
)
def test_check_expression_satisfiability_zero_divisor_screened_for_euclidean_ops(
    operation: BinaryOperation,
) -> None:
    """Test a literal-zero divisor is still screened for FLOOR_DIVIDE/MODULO.

    Zero fails the screen's positive-divisor requirement the same way a
    negative divisor does, so it is screened alongside the negative-
    divisor cases.
    """
    expression = BinaryExpression(
        BinaryOperation.EQUAL,
        BinaryExpression(operation, LiteralExpression(7), LiteralExpression(0)),
        LiteralExpression(0),
    )

    assert check_expression_satisfiability(expression, {}) is None


@pytest.mark.z3
def test_check_expression_satisfiability_divide_by_negative_literal_stays_decided() -> (
    None
):
    """Test DIVIDE by a negative literal divisor is not screened.

    Contrasts FLOOR_DIVIDE/MODULO: true division has no floor/Euclidean
    divergence, so a negative (but finite, nonzero) literal divisor
    remains safe for DIVIDE and the comparison decides normally.
    """
    expression = BinaryExpression(
        BinaryOperation.EQUAL,
        BinaryExpression(
            BinaryOperation.DIVIDE, LiteralExpression(7.0), LiteralExpression(-2.0)
        ),
        LiteralExpression(-3.5),
    )

    assert check_expression_satisfiability(expression, {}) is True


@pytest.mark.parametrize(
    "operation",
    [BinaryOperation.DIVIDE, BinaryOperation.FLOOR_DIVIDE, BinaryOperation.MODULO],
    ids=["divide", "floor_divide", "modulo"],
)
@pytest.mark.parametrize(
    "divisor_value",
    [float("nan"), float("inf")],
    ids=["nan", "inf"],
)
def test_check_expression_satisfiability_non_finite_literal_divisor_is_screened(
    operation: BinaryOperation, divisor_value: float
) -> None:
    """Test a nan/inf literal divisor is screened for every division-like operation.

    A `nan`/`inf` value compares unequal to 0, so a divisor screen that
    only checked for nonzero would admit it despite carrying none of the
    finite-value guarantee the screen requires.
    """
    x = mock_identifier("x", 0)
    division = BinaryExpression(
        operation, IdentifierExpression(x), LiteralExpression(divisor_value)
    )
    expression = BinaryExpression(
        BinaryOperation.EQUAL, division, LiteralExpression(0.0)
    )

    result = check_expression_satisfiability(expression, {x: SymbolType.REAL})

    assert result is None


@pytest.mark.z3
@pytest.mark.parametrize(
    "claimed_quotient",
    [3, 3.5],
    ids=["truncating_semantics_value", "true_division_value"],
)
def test_check_expression_satisfiability_int_true_division_is_screened(
    claimed_quotient: float,
) -> None:
    """Test `7 / 2` compared against either candidate quotient is screened.

    Z3 divides two INT-sorted operands with truncating integer division,
    so it reports `3` where this package's true division (and
    `simplify_expression`) reports `3.5`. The two candidate quotients are
    mutually exclusive, so screening must report both as undecided rather
    than decide either definitively -- without the screen the seam
    answers `True` for one and `False` for the other, exactly inverting
    `simplify_expression` on the same node.
    """
    division = BinaryExpression(
        BinaryOperation.DIVIDE, LiteralExpression(7), LiteralExpression(2)
    )
    expression = BinaryExpression(
        BinaryOperation.EQUAL, division, LiteralExpression(claimed_quotient)
    )

    assert check_expression_satisfiability(expression, {}) is None


@pytest.mark.z3
def test_check_expression_satisfiability_int_dividend_real_divisor_stays_decided() -> (
    None
):
    """Test a REAL-sorted operand keeps true division decidable.

    Contrasts the INT/INT case: one real operand makes the Z3 bridge
    lower the whole division to real arithmetic, which agrees with this
    package, so the screen must not refuse it.
    """
    division = BinaryExpression(
        BinaryOperation.DIVIDE, LiteralExpression(7), LiteralExpression(2.0)
    )
    expression = BinaryExpression(
        BinaryOperation.EQUAL, division, LiteralExpression(3.5)
    )

    assert check_expression_satisfiability(expression, {}) is True


@pytest.mark.z3
def test_check_expression_satisfiability_int_true_division_warns(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Test a screened INT/INT division logs a WARNING naming the node."""
    division = BinaryExpression(
        BinaryOperation.DIVIDE, LiteralExpression(7), LiteralExpression(2)
    )
    expression = BinaryExpression(
        BinaryOperation.EQUAL, division, LiteralExpression(3.5)
    )

    with caplog.at_level(logging.WARNING):
        result = check_expression_satisfiability(expression, {})

    assert result is None
    messages = _collect_solver_warning_messages(caplog)
    assert messages, "expected a WARNING naming the hazardous division node"
    assert "check_expression_satisfiability" in messages[0]
    assert repr(division) in messages[0]


@pytest.mark.z3
@pytest.mark.parametrize(
    "is_equality",
    [True, False],
    ids=["asserted", "negated"],
)
def test_check_expression_satisfiability_zero_to_the_zero_is_screened(
    is_equality: bool,
) -> None:
    """Test `0 ** 0` is screened whether the claim is asserted or negated.

    Z3 leaves exponentiation underspecified at `0 ** 0`, and the
    satisfiability encoding does not bind a partial function outside its
    domain, so without the screen both `0 ** 0 == 1` and `0 ** 0 != 1`
    come back provably unsatisfiable -- a pair no concrete assignment can
    justify. Screening must report both as undecided.
    """
    power = BinaryExpression(
        BinaryOperation.POWER, LiteralExpression(0), LiteralExpression(0)
    )
    operation = BinaryOperation.EQUAL if is_equality else BinaryOperation.NOT_EQUAL
    expression = BinaryExpression(operation, power, LiteralExpression(1))

    assert check_expression_satisfiability(expression, {}) is None


@pytest.mark.z3
def test_check_expression_satisfiability_negative_exponent_is_screened() -> None:
    """Test a negative exponent is screened.

    A negative exponent makes exponentiation a division, underspecified
    at a zero base, so `x ** -1 == 0` must not report the satisfiable
    verdict Z3's partial lowering otherwise supports: no real `x`
    satisfies `1 / x == 0`.
    """
    x = mock_identifier("x", 0)
    power = BinaryExpression(
        BinaryOperation.POWER, IdentifierExpression(x), LiteralExpression(-1)
    )
    expression = BinaryExpression(BinaryOperation.EQUAL, power, LiteralExpression(0))

    result = check_expression_satisfiability(expression, {x: SymbolType.REAL})

    assert result is None


@pytest.mark.z3
@pytest.mark.parametrize(
    "exponent",
    [0, 0.5, "2.0"],
    ids=["zero", "non_integer", "float_grammar_string"],
)
def test_check_expression_satisfiability_unsafe_exponent_is_screened(
    exponent: str | float | int | bool,
) -> None:
    """Test an exponent that is not an integer literal of at least one is screened.

    A zero exponent leaves `0 ** 0` reachable, and a non-integer
    exponent -- whether a Python `float` or a float-grammar string --
    lowers to a real power undefined for a negative base.
    """
    x = mock_identifier("x", 0)
    power = BinaryExpression(
        BinaryOperation.POWER, IdentifierExpression(x), LiteralExpression(exponent)
    )
    expression = BinaryExpression(BinaryOperation.EQUAL, power, LiteralExpression(1))

    result = check_expression_satisfiability(expression, {x: SymbolType.INT})

    assert result is None


@pytest.mark.z3
@pytest.mark.parametrize(
    "exponent", [2, "2", "02"], ids=["int", "string", "leading_zero"]
)
def test_check_expression_satisfiability_positive_integer_exponent_stays_decided(
    exponent: str | int,
) -> None:
    """Test a literal positive integer exponent is not screened, in every form.

    Contrasts the refused exponents: `x ** 2` is total on every integer,
    so the screen must leave it decidable. The integer-grammar strings
    `"2"` and `"02"` are structurally equivalent to the integer `2`, so
    the screen has to admit all three or the same question would be
    decided for one member of that class and refused for another.
    """
    x = mock_identifier("x", 0)
    power = BinaryExpression(
        BinaryOperation.POWER, IdentifierExpression(x), LiteralExpression(exponent)
    )
    expression = BinaryExpression(BinaryOperation.EQUAL, power, LiteralExpression(4))

    result = check_expression_satisfiability(expression, {x: SymbolType.INT})

    assert result is True


@pytest.mark.z3
@pytest.mark.parametrize(
    "operation",
    [BinaryOperation.EQUAL, BinaryOperation.NOT_EQUAL],
    ids=["equal", "not_equal"],
)
def test_check_expression_satisfiability_real_operand_int_literal_is_screened(
    operation: BinaryOperation,
) -> None:
    """Test comparing a REAL-sorted identifier to a strict-int literal is screened.

    The mirror of the float-literal-against-INT-sorted case. Z3
    rationalizes whichever side is INT-sorted and compares numerically,
    so it reads `1` and `1.0` as the same value in either arrangement,
    while this package holds them type-strictly distinct. A type-strict
    set constraint over an integer member lowers to exactly this shape
    when its parameter is real-valued.
    """
    x = mock_identifier("x", 0)
    expression = BinaryExpression(
        operation, IdentifierExpression(x), LiteralExpression(1)
    )

    result = check_expression_satisfiability(expression, {x: SymbolType.REAL})

    assert result is None


@pytest.mark.z3
def test_check_expression_satisfiability_real_operand_float_literal_stays_decided() -> (
    None
):
    """Test a REAL-sorted identifier against a float literal is not screened.

    Contrasts the int-literal case: both sides are already real, so no
    rationalization happens and no type-strict distinction is collapsed.
    """
    x = mock_identifier("x", 0)
    expression = BinaryExpression(
        BinaryOperation.EQUAL, IdentifierExpression(x), LiteralExpression(1.0)
    )

    result = check_expression_satisfiability(expression, {x: SymbolType.REAL})

    assert result is True


@pytest.mark.z3
@pytest.mark.parametrize(
    "operation",
    [BinaryOperation.LESS, BinaryOperation.GREATER_EQUAL],
    ids=["less", "greater_equal"],
)
def test_check_expression_satisfiability_real_operand_int_literal_ordering_decides(
    operation: BinaryOperation,
) -> None:
    """Test ordering a REAL-sorted identifier against an int literal is not screened.

    Only `EQUAL`/`NOT_EQUAL` collapse the int/float distinction; mixed-sort
    ordering stays mathematically meaningful and must stay decidable.
    """
    x = mock_identifier("x", 0)
    expression = BinaryExpression(
        operation, IdentifierExpression(x), LiteralExpression(1)
    )

    result = check_expression_satisfiability(expression, {x: SymbolType.REAL})

    assert result is True


def test_check_expression_satisfiability_int_float_equality_hazard_returns_none(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Test EQUAL mixing an INT-sorted identifier and a float literal is screened.

    Z3's ``ToReal`` rationalization of the INT-sorted operand collapses
    the type-strict int/float distinction, so the seam refuses to lower
    ``x == 1.5`` for an INT-sorted ``x``.
    """
    x = mock_identifier("x", 0)
    expression = BinaryExpression(
        BinaryOperation.EQUAL, IdentifierExpression(x), LiteralExpression(1.5)
    )

    with caplog.at_level(logging.WARNING):
        result = check_expression_satisfiability(expression, {x: SymbolType.INT})

    assert result is None
    messages = _collect_solver_warning_messages(caplog)
    assert messages, "expected a WARNING naming the hazardous node"
    assert "check_expression_satisfiability" in messages[0]
    assert repr(x) in messages[0]


@pytest.mark.z3
def test_check_expression_satisfiability_real_identifier_eq_float_stays_decided() -> (
    None
):
    """Test EQUAL between a REAL-sorted identifier and a float literal is not screened.

    Contrasts the hazard: the identifier is not INT-sorted, so no
    sort-mixing hazard exists.
    """
    x = mock_identifier("x", 0)
    expression = BinaryExpression(
        BinaryOperation.EQUAL, IdentifierExpression(x), LiteralExpression(1.5)
    )

    assert check_expression_satisfiability(expression, {x: SymbolType.REAL}) is True


@pytest.mark.z3
def test_check_expression_satisfiability_int_identifier_lt_float_not_screened() -> None:
    """Test `<` between an INT-sorted identifier and a float literal is not screened.

    Ordering comparisons stay mathematically meaningful across a mixed
    int/float sort, so only EQUAL/NOT_EQUAL is screened.
    """
    x = mock_identifier("x", 0)
    expression = BinaryExpression(
        BinaryOperation.LESS, IdentifierExpression(x), LiteralExpression(1.5)
    )

    assert check_expression_satisfiability(expression, {x: SymbolType.INT}) is True


# -----------------------------------------------------------------------
# The numeric-kind classifier follows the IR's evaluated int/float kind,
# not Z3's sort, through arithmetic, NEGATE, POWER, and piecewise
# -----------------------------------------------------------------------


@pytest.mark.z3
@pytest.mark.parametrize(
    "operation",
    [BinaryOperation.EQUAL, BinaryOperation.NOT_EQUAL],
    ids=["equal", "not_equal"],
)
def test_check_expression_satisfiability_int_addition_against_float_literal_is_screened(
    operation: BinaryOperation,
) -> None:
    """Test EQUAL/NOT_EQUAL between INT arithmetic and a float literal is screened.

    `y + 1` stays INT-valued whenever `y` is, so comparing it to the
    float literal `3.0` hits the same Z3 rationalization hazard as
    comparing a bare INT identifier to a float literal.
    """
    y = mock_identifier("y", 0)
    addition = BinaryExpression(
        BinaryOperation.ADD, IdentifierExpression(y), LiteralExpression(1)
    )
    expression = BinaryExpression(operation, addition, LiteralExpression(3.0))

    result = check_expression_satisfiability(expression, {y: SymbolType.INT})

    assert result is None


@pytest.mark.z3
def test_check_expression_satisfiability_literal_left_of_int_addition_screened() -> (
    None
):
    """Test the screen catches the hazard with the float literal on the left.

    The mixed-kind hazard is symmetric in operand order: `3.0 == y + 1`
    must be refused exactly as `y + 1 == 3.0` is.
    """
    y = mock_identifier("y", 0)
    addition = BinaryExpression(
        BinaryOperation.ADD, IdentifierExpression(y), LiteralExpression(1)
    )
    expression = BinaryExpression(
        BinaryOperation.EQUAL, LiteralExpression(3.0), addition
    )

    result = check_expression_satisfiability(expression, {y: SymbolType.INT})

    assert result is None


@pytest.mark.z3
def test_check_expression_satisfiability_int_multiply_vs_float_literal_screened() -> (
    None
):
    """Test INT multiplication compared to a float literal is screened.

    `y * 2` stays INT-valued for an INT `y`, so it hits the same
    rationalization hazard as a bare INT operand.
    """
    y = mock_identifier("y", 0)
    multiplication = BinaryExpression(
        BinaryOperation.MULTIPLY, IdentifierExpression(y), LiteralExpression(2)
    )
    expression = BinaryExpression(
        BinaryOperation.EQUAL, multiplication, LiteralExpression(3.0)
    )

    result = check_expression_satisfiability(expression, {y: SymbolType.INT})

    assert result is None


@pytest.mark.z3
def test_check_expression_satisfiability_negated_int_vs_float_literal_screened() -> (
    None
):
    """Test NEGATE of an INT identifier compared to a float literal is screened.

    NEGATE keeps its operand's kind, so `-y` is still INT-valued for an
    INT `y` and hits the same hazard as the bare identifier.
    """
    y = mock_identifier("y", 0)
    negated = UnaryExpression(UnaryOperation.NEGATE, IdentifierExpression(y))
    expression = BinaryExpression(
        BinaryOperation.EQUAL, negated, LiteralExpression(3.0)
    )

    result = check_expression_satisfiability(expression, {y: SymbolType.INT})

    assert result is None


@pytest.mark.z3
def test_check_expression_satisfiability_int_power_vs_float_literal_screened() -> None:
    """Test INT POWER by an integer exponent compared to a float literal is screened.

    Z3 sorts `Int ** Int` as Real, but the IR evaluates `y ** 2` to an
    int for an INT `y`, so the type-strict distinction from `4.0` is
    still live and the comparison must stay refused.
    """
    y = mock_identifier("y", 0)
    power = BinaryExpression(
        BinaryOperation.POWER, IdentifierExpression(y), LiteralExpression(2)
    )
    expression = BinaryExpression(BinaryOperation.EQUAL, power, LiteralExpression(4.0))

    result = check_expression_satisfiability(expression, {y: SymbolType.INT})

    assert result is None


@pytest.mark.z3
def test_check_expression_satisfiability_int_literal_vs_real_piecewise_screened() -> (
    None
):
    """Test an int literal compared to an all-REAL piecewise is screened.

    Every branch of the piecewise is REAL-valued, so the piecewise as a
    whole hits the same hazard as a bare REAL-sorted operand compared to
    the int literal `1`.
    """
    y = mock_identifier("y", 0)
    b = mock_identifier("b", 1)
    branch = PiecewiseExpression(
        (IdentifierExpression(b),), (IdentifierExpression(y),), IdentifierExpression(y)
    )
    expression = BinaryExpression(BinaryOperation.EQUAL, LiteralExpression(1), branch)

    result = check_expression_satisfiability(
        expression, {y: SymbolType.REAL, b: SymbolType.BOOL}
    )

    assert result is None


@pytest.mark.z3
def test_check_expression_satisfiability_float_vs_mixed_kind_piecewise_screened() -> (
    None
):
    """Test a float literal compared to a piecewise mixing INT and REAL branches.

    The piecewise's branches disagree in kind -- one INT, one REAL -- so
    its own kind is unknown, and an unknown-kind operand next to a
    numeric literal is refused exactly as a provable INT/REAL mismatch
    is.
    """
    y = mock_identifier("y", 0)
    r = mock_identifier("r", 1)
    b = mock_identifier("b", 2)
    branch = PiecewiseExpression(
        (IdentifierExpression(b),), (IdentifierExpression(y),), IdentifierExpression(r)
    )
    expression = BinaryExpression(BinaryOperation.EQUAL, LiteralExpression(3.0), branch)

    result = check_expression_satisfiability(
        expression, {y: SymbolType.INT, r: SymbolType.REAL, b: SymbolType.BOOL}
    )

    assert result is None


@pytest.mark.z3
def test_check_expression_satisfiability_int_addition_eq_int_literal_decided() -> None:
    """Test INT arithmetic compared to an integer literal stays decided.

    Contrasts the hazard: `3` is integer-valued, so no kind mismatch
    exists against the INT-valued `y + 1`.
    """
    y = mock_identifier("y", 0)
    addition = BinaryExpression(
        BinaryOperation.ADD, IdentifierExpression(y), LiteralExpression(1)
    )
    expression = BinaryExpression(BinaryOperation.EQUAL, addition, LiteralExpression(3))

    assert check_expression_satisfiability(expression, {y: SymbolType.INT}) is True


@pytest.mark.z3
def test_check_expression_satisfiability_int_times_float_eq_float_decided() -> None:
    """Test multiplying an INT identifier by a float literal makes the product REAL.

    `y * 1.0` is REAL-valued regardless of `y`'s own sort, matching the
    float literal `3.0` on the other side, so the comparison stays
    decided.
    """
    y = mock_identifier("y", 0)
    multiplication = BinaryExpression(
        BinaryOperation.MULTIPLY, IdentifierExpression(y), LiteralExpression(1.0)
    )
    expression = BinaryExpression(
        BinaryOperation.EQUAL, multiplication, LiteralExpression(3.0)
    )

    assert check_expression_satisfiability(expression, {y: SymbolType.INT}) is True


@pytest.mark.z3
def test_check_expression_satisfiability_real_addition_eq_float_decided() -> None:
    """Test REAL arithmetic compared to a float literal stays decided.

    Both sides are REAL-valued, so no kind mismatch exists.
    """
    r = mock_identifier("r", 0)
    addition = BinaryExpression(
        BinaryOperation.ADD, IdentifierExpression(r), LiteralExpression(1)
    )
    expression = BinaryExpression(
        BinaryOperation.EQUAL, addition, LiteralExpression(3.0)
    )

    assert check_expression_satisfiability(expression, {r: SymbolType.REAL}) is True


@pytest.mark.z3
def test_check_expression_satisfiability_int_floor_divide_eq_int_decided() -> None:
    """Test INT floor-division compared to an integer literal stays decided.

    `y // 2` is INT-valued for an INT `y`, matching the integer literal
    `3` on the other side.
    """
    y = mock_identifier("y", 0)
    floor_divide = BinaryExpression(
        BinaryOperation.FLOOR_DIVIDE, IdentifierExpression(y), LiteralExpression(2)
    )
    expression = BinaryExpression(
        BinaryOperation.EQUAL, floor_divide, LiteralExpression(3)
    )

    assert check_expression_satisfiability(expression, {y: SymbolType.INT}) is True


@pytest.mark.z3
def test_check_expression_satisfiability_int_addition_lt_float_not_screened() -> None:
    """Test `<` between INT arithmetic and a float literal is not screened.

    Only EQUAL/NOT_EQUAL collapse the type-strict int/float distinction;
    ordering stays mathematically meaningful even when one side is an
    INT-valued arithmetic expression rather than a bare identifier.
    """
    y = mock_identifier("y", 0)
    addition = BinaryExpression(
        BinaryOperation.ADD, IdentifierExpression(y), LiteralExpression(1)
    )
    expression = BinaryExpression(
        BinaryOperation.LESS, addition, LiteralExpression(3.0)
    )

    assert check_expression_satisfiability(expression, {y: SymbolType.INT}) is True


@pytest.mark.z3
def test_check_expression_satisfiability_int_id_eq_real_id_not_screened() -> None:
    """Test EQUAL between two non-literal operands is not screened.

    The hazard only applies when one side is a numeric literal; an
    equality between two bare identifiers is left to the solver even
    when they are declared different sorts.
    """
    y = mock_identifier("y", 0)
    r = mock_identifier("r", 1)
    expression = BinaryExpression(
        BinaryOperation.EQUAL, IdentifierExpression(y), IdentifierExpression(r)
    )

    result = check_expression_satisfiability(
        expression, {y: SymbolType.INT, r: SymbolType.REAL}
    )

    assert result is True


def test_does_expression_imply_hazardous_premise_returns_none(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Test a hazardous antecedent is screened before the solver is consulted.

    Mirrors ``ConstraintSystem.check_implication``'s antecedent-side
    hazard: the bool-coercion screen applies equally to
    ``does_expression_imply``.
    """
    x = mock_identifier("x", 0)
    antecedent = BinaryExpression(
        BinaryOperation.EQUAL, IdentifierExpression(x), LiteralExpression(True)
    )
    consequent = BinaryExpression(
        BinaryOperation.GREATER, IdentifierExpression(x), LiteralExpression(0)
    )

    with caplog.at_level(logging.WARNING):
        result = does_expression_imply(antecedent, consequent, {x: SymbolType.INT})

    assert result is None
    messages = _collect_solver_warning_messages(caplog)
    assert messages, "expected a WARNING naming the hazardous node"
    assert "does_expression_imply" in messages[0]


def test_assert_holds_for_all_free_assignments_reason_is_hazard_screen() -> None:
    """Test a hazard-screened expression raises with `reason` set to a hazard marker.

    No Z3 solver call happens for a screened expression, so the raised
    error's `reason` cannot be Z3's `reason_unknown()` text; it carries a
    fixed marker instead, letting a caller tell this apart from a
    retryable Z3 timeout.
    """
    x = mock_identifier("x", 0)
    expression = BinaryExpression(
        BinaryOperation.EQUAL, IdentifierExpression(x), LiteralExpression(True)
    )

    with pytest.raises(
        UndecidableError, match="refused by the solver seam's hazard screen"
    ) as exc_info:
        assert_holds_for_all_free_assignments(
            frozenset(), expression, {x: SymbolType.INT}
        )

    assert exc_info.value.reason == "hazard_screen"


def test_assert_expression_implies_undecidable_error_reason_is_hazard_screen() -> None:
    """Test a hazard-screened implication raises with a hazard-marker `reason`."""
    x = mock_identifier("x", 0)
    antecedent = BinaryExpression(
        BinaryOperation.EQUAL, IdentifierExpression(x), LiteralExpression(True)
    )
    consequent = BinaryExpression(
        BinaryOperation.GREATER, IdentifierExpression(x), LiteralExpression(0)
    )

    with pytest.raises(
        UndecidableError, match="refused by the solver seam's hazard screen"
    ) as exc_info:
        assert_expression_implies(antecedent, consequent, {x: SymbolType.INT})

    assert exc_info.value.reason == "hazard_screen"


# =============================================================================
# timeout_milliseconds
# =============================================================================


def test_simplify_expression_signature_has_no_timeout_parameter() -> None:
    """Test the SymPy-backed seam function has no `timeout_milliseconds` parameter."""
    parameters = inspect.signature(simplify_expression).parameters
    assert "timeout_milliseconds" not in parameters


# Every seam function that accepts `timeout_milliseconds`, as a callable
# taking the timeout to pass. Each is exercised for both halves of the
# contract: a non-positive value is rejected, and a positive value reaches
# the underlying `z3.Solver`. Validating or forwarding in only some of them
# is the failure these parametrizations exist to catch.
TIMEOUT_ACCEPTING_SEAM_CALLS = [
    (
        "check_expression_satisfiability",
        lambda identifier, expression, timeout: check_expression_satisfiability(
            expression, {identifier: SymbolType.INT}, timeout_milliseconds=timeout
        ),
    ),
    (
        "does_expression_imply",
        lambda identifier, expression, timeout: does_expression_imply(
            expression,
            expression,
            {identifier: SymbolType.INT},
            timeout_milliseconds=timeout,
        ),
    ),
    (
        "holds_for_all_free_assignments",
        lambda identifier, expression, timeout: holds_for_all_free_assignments(
            frozenset(),
            expression,
            {identifier: SymbolType.INT},
            timeout_milliseconds=timeout,
        ),
    ),
    (
        "assert_holds_for_all_free_assignments",
        lambda identifier, expression, timeout: assert_holds_for_all_free_assignments(
            frozenset(),
            expression,
            {identifier: SymbolType.INT},
            timeout_milliseconds=timeout,
        ),
    ),
    (
        "assert_expression_implies",
        lambda identifier, expression, timeout: assert_expression_implies(
            expression,
            expression,
            {identifier: SymbolType.INT},
            timeout_milliseconds=timeout,
        ),
    ),
]


@pytest.mark.parametrize("bad_timeout", [0, -1, -1000])
@pytest.mark.parametrize(
    "invoke",
    [invoke for _, invoke in TIMEOUT_ACCEPTING_SEAM_CALLS],
    ids=[name for name, _ in TIMEOUT_ACCEPTING_SEAM_CALLS],
)
def test_seam_functions_reject_a_non_positive_timeout(
    invoke: Callable[[Identifier, Expression, int], object], bad_timeout: int
) -> None:
    """Test every timeout-accepting seam function rejects a non-positive bound.

    Z3 reads ``timeout=0`` as *no* timeout, so a zero that slips through
    silently inverts the caller's intent.
    """
    x = mock_identifier("x", 0)
    expression = BinaryExpression(
        BinaryOperation.GREATER_EQUAL, IdentifierExpression(x), IdentifierExpression(x)
    )

    with pytest.raises(ValueError, match=r"positive"):
        invoke(x, expression, bad_timeout)


@pytest.mark.parametrize("bad_timeout", [True, 1000.0], ids=["bool", "float"])
@pytest.mark.parametrize(
    "invoke",
    [invoke for _, invoke in TIMEOUT_ACCEPTING_SEAM_CALLS],
    ids=[name for name, _ in TIMEOUT_ACCEPTING_SEAM_CALLS],
)
def test_seam_functions_reject_a_timeout_that_is_not_a_strict_integer(
    invoke: Callable[[Identifier, Expression, object], object], bad_timeout: object
) -> None:
    """Test every timeout-accepting seam function rejects a bool or float bound.

    ``True`` equals ``1`` and ``1000.0`` is positive, yet neither is the
    unsigned integer Z3's timeout parameter takes, so each would otherwise
    pass validation and fail inside the solver.
    """
    x = mock_identifier("x", 0)
    expression = BinaryExpression(
        BinaryOperation.GREATER_EQUAL, IdentifierExpression(x), IdentifierExpression(x)
    )

    with pytest.raises(ValueError, match=r"positive integer"):
        invoke(x, expression, bad_timeout)


@pytest.mark.z3
@pytest.mark.parametrize(
    "invoke",
    [invoke for _, invoke in TIMEOUT_ACCEPTING_SEAM_CALLS],
    ids=[name for name, _ in TIMEOUT_ACCEPTING_SEAM_CALLS],
)
def test_seam_functions_thread_the_timeout_to_the_z3_solver(
    invoke: Callable[[Identifier, Expression, int], object],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test every timeout-accepting seam function forwards the bound to Z3.

    A dropped ``timeout_milliseconds=`` keyword leaves the solver
    unbounded, so a query the caller expected to give up on becomes a
    hang and ``UndecidableError`` never fires.
    """
    recorded: dict[str, object] = {}
    original_set = z3.Solver.set

    def record_solver_set_kwargs(
        self: z3.Solver, *args: object, **kwargs: object
    ) -> None:
        recorded.update(kwargs)
        original_set(self, *args, **kwargs)

    monkeypatch.setattr(z3.Solver, "set", record_solver_set_kwargs)
    x = mock_identifier("x", 0)
    expression = BinaryExpression(
        BinaryOperation.GREATER_EQUAL, IdentifierExpression(x), IdentifierExpression(x)
    )

    invoke(x, expression, 2500)

    assert recorded.get("timeout") == 2500


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

    def record_solver_set_kwargs(
        self: z3.Solver, *args: object, **kwargs: object
    ) -> None:
        recorded.update(kwargs)
        original_set(self, *args, **kwargs)

    monkeypatch.setattr(z3.Solver, "set", record_solver_set_kwargs)
    x = mock_identifier("x", 0)
    expression = BinaryExpression(
        BinaryOperation.GREATER, IdentifierExpression(x), LiteralExpression(0)
    )

    check_expression_satisfiability(
        expression, {x: SymbolType.INT}, timeout_milliseconds=2500
    )

    assert recorded.get("timeout") == 2500


# =============================================================================
# Hazard screening on every seam entry point and in every operand position
# =============================================================================


@pytest.mark.z3
def test_does_expression_imply_screens_a_hazard_in_the_consequent() -> None:
    """Test a hazard in the consequent is screened, not just one in the antecedent.

    The screen tests the antecedent `or` the consequent, so a suite that
    only ever places the hazard on the left never evaluates the right
    operand. With the consequent unscreened, `x == 1 implies x in {True}`
    reports a decided answer that contradicts the type-strict truth.
    """
    x = mock_identifier("x", 0)
    antecedent = BinaryExpression(
        BinaryOperation.EQUAL, IdentifierExpression(x), LiteralExpression(1)
    )
    consequent = BinaryExpression(
        BinaryOperation.EQUAL, IdentifierExpression(x), LiteralExpression(True)
    )

    result = does_expression_imply(antecedent, consequent, {x: SymbolType.INT})

    assert result is None


@pytest.mark.z3
def test_holds_for_all_free_assignments_screens_a_hazard() -> None:
    """Test the lenient universal-validity entry point screens hazards too.

    The `assert_` twin re-implements the screen, so covering only that one
    leaves this entry point's own screen call unexecuted, despite the
    design requiring every Z3-question entry point to be guarded.
    """
    x = mock_identifier("x", 0)
    expression = BinaryExpression(
        BinaryOperation.EQUAL, IdentifierExpression(x), LiteralExpression(1.5)
    )

    result = holds_for_all_free_assignments(
        frozenset({x}), expression, {x: SymbolType.INT}
    )

    assert result is None


@pytest.mark.z3
def test_check_expression_satisfiability_screens_a_bool_in_arithmetic() -> None:
    """Test a Boolean operand inside arithmetic is screened, not only in a comparison.

    Every other bool-hazard case reaches the screen through a comparison
    or a piecewise; the arithmetic arm covers `x + b`, where the Z3
    bindings rewrite `b` to `If(b, 1, 0)`.
    """
    x = mock_identifier("x", 0)
    b = mock_identifier("b", 1)
    total = BinaryExpression(
        BinaryOperation.ADD, IdentifierExpression(x), IdentifierExpression(b)
    )
    expression = BinaryExpression(BinaryOperation.GREATER, total, LiteralExpression(0))

    result = check_expression_satisfiability(
        expression, {x: SymbolType.INT, b: SymbolType.BOOL}
    )

    assert result is None


@pytest.mark.z3
def test_check_expression_satisfiability_screens_a_nested_int_float_equality() -> None:
    """Test the int/float equality screen descends past the root node.

    A multi-member `ConstraintSystem` lowers to `logical_and(...)`, so the
    realistic position for this hazard is a child rather than the root. A
    screen that only inspected the root would hand the conjunction to Z3
    and decide it.
    """
    x = mock_identifier("x", 0)
    hazard = BinaryExpression(
        BinaryOperation.EQUAL, IdentifierExpression(x), LiteralExpression(1.5)
    )
    benign = BinaryExpression(
        BinaryOperation.GREATER, IdentifierExpression(x), LiteralExpression(0)
    )
    expression = Expression.logical_and(benign, hazard)

    result = check_expression_satisfiability(expression, {x: SymbolType.INT})

    assert result is None


# =============================================================================
# Decidability is constant on literal structural-equivalence classes
# =============================================================================


@pytest.mark.z3
@pytest.mark.parametrize(
    "integer_form", [1, "1", "01"], ids=["int", "string", "leading_zero"]
)
def test_int_float_equality_screen_refuses_every_integer_literal_form(
    integer_form: int | str,
) -> None:
    """Test the int/float screen refuses an equality in every integer spelling.

    `LiteralExpression(1)`, `LiteralExpression("1")`, and
    `LiteralExpression("01")` are one structural-equivalence class, so the
    screen has to classify all three as integer-valued and refuse each
    against `1.0`. Deciding the string forms while refusing the `int` form
    would answer a question for one member of a class that the seam
    declares undecidable for another.
    """
    left = LiteralExpression(integer_form)
    expression = BinaryExpression(BinaryOperation.EQUAL, left, LiteralExpression(1.0))

    assert left.is_structurally_equivalent(LiteralExpression(1))
    assert check_expression_satisfiability(expression, {}) is None


@pytest.mark.z3
@pytest.mark.parametrize(
    "right_value, expected",
    [
        pytest.param(1, True, id="equal_integers"),
        pytest.param("01", True, id="equal_integers_leading_zero"),
        pytest.param(2, False, id="distinct_integers"),
    ],
)
def test_equality_of_integer_literal_forms_is_decided_as_an_integer_comparison(
    right_value: int | str, expected: bool
) -> None:
    """Test an integer-grammar string compares as the integer it denotes.

    Both sides lower to the INT sort, so the seam decides the comparison
    outright rather than screening it: `"1" == 1` is satisfiable and
    `"1" == 2` is not.
    """
    expression = BinaryExpression(
        BinaryOperation.EQUAL, LiteralExpression("1"), LiteralExpression(right_value)
    )

    assert check_expression_satisfiability(expression, {}) is expected


@pytest.mark.z3
@pytest.mark.parametrize(
    "float_form",
    [1.0, "1.0", 1.5, "1.5"],
    ids=["float", "decimal", "float_fractional", "decimal_fractional"],
)
def test_int_float_equality_screen_refuses_every_float_literal_form(
    float_form: float | str,
) -> None:
    """Test a float-valued literal against an integer literal stays refused.

    The counterpart to the integer-form screen: neither float bucket is
    integer-valued, so the mixed-sort equality is refused whichever
    spelling the float side uses.
    """
    expression = BinaryExpression(
        BinaryOperation.EQUAL, LiteralExpression(float_form), LiteralExpression(1)
    )

    assert check_expression_satisfiability(expression, {}) is None


@pytest.mark.z3
@pytest.mark.parametrize(
    "operation, result_value",
    [
        pytest.param(BinaryOperation.POWER, 4, id="power"),
        pytest.param(BinaryOperation.FLOOR_DIVIDE, 2, id="floor_divide"),
        pytest.param(BinaryOperation.MODULO, 1, id="modulo"),
    ],
)
@pytest.mark.parametrize(
    "operand_form", [2, "2", "02"], ids=["int", "string", "leading_zero"]
)
def test_partial_operation_screen_admits_every_integer_operand_form(
    operation: BinaryOperation, result_value: int, operand_form: int | str
) -> None:
    """Test the partial-operation screen admits an integer operand in every spelling.

    The exponent screen wants an integer of at least one and the
    floor-division and modulo screens want a positive integer divisor.
    Each holds of `2`, `"2"`, and `"02"` alike, so all three stay
    decidable: a screen keying off the stored Python type instead would
    refuse the string spellings of a divisor it accepts as an `int`.
    """
    x = mock_identifier("x", 0)
    applied = BinaryExpression(
        operation, IdentifierExpression(x), LiteralExpression(operand_form)
    )
    expression = BinaryExpression(
        BinaryOperation.EQUAL, applied, LiteralExpression(result_value)
    )

    result = check_expression_satisfiability(expression, {x: SymbolType.INT})

    assert result is True


@pytest.mark.z3
@pytest.mark.parametrize(
    "operation",
    [
        pytest.param(BinaryOperation.POWER, id="power"),
        pytest.param(BinaryOperation.FLOOR_DIVIDE, id="floor_divide"),
        pytest.param(BinaryOperation.MODULO, id="modulo"),
    ],
)
def test_partial_operation_screen_refuses_a_float_grammar_string_operand(
    operation: BinaryOperation,
) -> None:
    """Test a float-grammar string operand is still refused by the screen.

    A float-grammar string is not integer-valued, so it satisfies neither
    the integer-exponent nor the positive-integer-divisor requirement and
    the node stays undecidable.
    """
    x = mock_identifier("x", 0)
    applied = BinaryExpression(
        operation, IdentifierExpression(x), LiteralExpression("2.0")
    )
    expression = BinaryExpression(BinaryOperation.EQUAL, applied, LiteralExpression(1))

    result = check_expression_satisfiability(expression, {x: SymbolType.INT})

    assert result is None


# =============================================================================
# Backend agreement on identifiers named after native constants
# =============================================================================


@pytest.mark.z3
@pytest.mark.parametrize("constant_name", ["pi", "e"])
def test_backends_agree_on_an_identifier_named_after_a_native_constant(
    constant_name: str,
) -> None:
    """Test Z3 and SymPy treat a variable named ``pi`` the same way.

    Only a constant's canonical identifier denotes the constant, so an
    identifier that merely shares its name is an ordinary variable to
    both backends: satisfiability decides `pi == 1` `True` (the variable
    can be 1) and simplification leaves a residual, where reading it as
    the constant would fold it to `False` on one side and refuse it on
    the other.
    """
    variable = mock_identifier(constant_name, 1024)
    expression = BinaryExpression(
        BinaryOperation.EQUAL, IdentifierExpression(variable), LiteralExpression(1)
    )

    satisfiable = check_expression_satisfiability(
        expression, {variable: SymbolType.INT}
    )
    simplified = simplify_expression(expression)

    assert satisfiable is True
    assert not isinstance(simplified, LiteralExpression)
    assert simplified.get_free_identifiers() == {variable}


# =============================================================================
# Ill-typed logical connectives are refused, not reported as undecidable
# =============================================================================

_NUMERIC_CONNECTIVES = [
    pytest.param(
        Expression.logical_and(LiteralExpression(2), LiteralExpression(4)), id="and"
    ),
    pytest.param(
        Expression.logical_or(LiteralExpression(2), LiteralExpression(4)), id="or"
    ),
    pytest.param(
        UnaryExpression(UnaryOperation.LOGICAL_NOT, LiteralExpression(2)), id="not"
    ),
]


@pytest.mark.z3
@pytest.mark.parametrize("expression", _NUMERIC_CONNECTIVES)
def test_check_expression_satisfiability_refuses_a_numeric_logical_operand(
    expression: Expression,
) -> None:
    """Test the seam refuses an ill-typed connective with its own typed error.

    Z3 rejects an integer operand of ``z3.And`` with a ``Z3Exception``,
    which the pass infrastructure wraps into a `PassExecutionError` naming
    an SMT-LIB declaration. Neither name tells the caller their expression
    is ill-typed, so the seam reports it as such.
    """
    with pytest.raises(NonBooleanLogicalOperandError) as exc_info:
        check_expression_satisfiability(expression, {})

    assert type(exc_info.value) is NonBooleanLogicalOperandError
    assert "Sort mismatch" not in str(exc_info.value)


@pytest.mark.z3
@pytest.mark.parametrize("expression", _NUMERIC_CONNECTIVES)
def test_both_backends_refuse_one_ill_typed_expression_with_the_same_error(
    expression: Expression,
) -> None:
    """Test simplification and satisfiability agree on how they refuse.

    The two queries route to different backends whose native complaints
    differ -- a SymPy ``TypeError`` on one side, a wrapped ``Z3Exception``
    on the other -- so without a shared screen a caller would have to
    catch two unrelated error types for the same authoring mistake.
    """
    with pytest.raises(NonBooleanLogicalOperandError) as simplify_error:
        simplify_expression(expression)
    with pytest.raises(NonBooleanLogicalOperandError) as satisfiability_error:
        check_expression_satisfiability(expression, {})

    assert type(simplify_error.value) is type(satisfiability_error.value)


@pytest.mark.z3
def test_does_expression_imply_raises_rather_than_reporting_none() -> None:
    """Test an ill-typed operand raises instead of taking the hazard screen's `None`.

    ``None`` is this seam's report for "the solver could not settle a
    well-typed question". An ill-typed expression has no answer to settle,
    so folding it into the same channel would let an authoring bug pass
    for a solver limitation.
    """
    antecedent = Expression.logical_and(LiteralExpression(2), LiteralExpression(4))

    with pytest.raises(NonBooleanLogicalOperandError):
        does_expression_imply(antecedent, LiteralExpression(True), {})


@pytest.mark.z3
def test_holds_for_all_free_assignments_raises_rather_than_reporting_none() -> None:
    """Test the lenient universal-validity entry point raises on a numeric operand."""
    expression = Expression.logical_or(LiteralExpression(2), LiteralExpression(4))

    with pytest.raises(NonBooleanLogicalOperandError):
        holds_for_all_free_assignments(frozenset(), expression, {})


@pytest.mark.z3
@pytest.mark.parametrize(
    "query",
    [
        pytest.param(
            lambda expression: assert_holds_for_all_free_assignments(
                frozenset(), expression, {}
            ),
            id="assert_holds_for_all_free_assignments",
        ),
        pytest.param(
            lambda expression: assert_expression_implies(
                expression, LiteralExpression(True), {}
            ),
            id="assert_expression_implies",
        ),
    ],
)
def test_strict_companion_refuses_an_ill_typed_operand_as_a_type_error(
    query: Callable[[Expression], bool],
) -> None:
    """Test the strict companions do not dress an ill-typed shape as undecidability.

    `UndecidableError` carries a ``reason`` a caller can act on -- a
    timeout invites a larger bound, a hazard-screen marker says to stop
    asking. An ill-typed expression fits neither: it needs the tree fixed,
    which is what a `TypeError` says.
    """
    expression = Expression.logical_and(LiteralExpression(2), LiteralExpression(4))

    with pytest.raises(NonBooleanLogicalOperandError) as exc_info:
        query(expression)

    assert not isinstance(exc_info.value, UndecidableError)


_Z3_QUESTIONS_OVER_ONE_EXPRESSION = [
    pytest.param(
        lambda expression: check_expression_satisfiability(expression, {}),
        id="check_expression_satisfiability",
    ),
    pytest.param(
        lambda expression: does_expression_imply(
            expression, LiteralExpression(True), {}
        ),
        id="does_expression_imply_antecedent",
    ),
    pytest.param(
        lambda expression: does_expression_imply(
            LiteralExpression(True), expression, {}
        ),
        id="does_expression_imply_consequent",
    ),
    pytest.param(
        lambda expression: holds_for_all_free_assignments(frozenset(), expression, {}),
        id="holds_for_all_free_assignments",
    ),
    pytest.param(
        lambda expression: assert_holds_for_all_free_assignments(
            frozenset(), expression, {}
        ),
        id="assert_holds_for_all_free_assignments",
    ),
    pytest.param(
        lambda expression: assert_expression_implies(
            expression, LiteralExpression(True), {}
        ),
        id="assert_expression_implies_antecedent",
    ),
    pytest.param(
        lambda expression: assert_expression_implies(
            LiteralExpression(True), expression, {}
        ),
        id="assert_expression_implies_consequent",
    ),
]


@pytest.mark.z3
@pytest.mark.parametrize("query", _Z3_QUESTIONS_OVER_ONE_EXPRESSION)
def test_z3_question_reports_ill_typedness_ahead_of_the_hazard_screen(
    query: Callable[[Expression], bool | None],
) -> None:
    """Test an ill-typed tree the hazard screen would also refuse raises its own error.

    ``logical_and(2, 4) == 1`` is ill-typed, with numbers under ``and``,
    and hazardous, comparing a Boolean with a number. The screen's refusal
    -- ``None`` or ``UndecidableError`` -- says a different configuration
    might decide the question; ill-typedness says none can, so it is the
    diagnosis the caller gets.
    """
    expression = Expression.logical_and(
        LiteralExpression(2), LiteralExpression(4)
    ).equals(LiteralExpression(1))

    with pytest.raises(NonBooleanLogicalOperandError):
        query(expression)


@pytest.mark.z3
@pytest.mark.parametrize(
    "implies",
    [
        pytest.param(does_expression_imply, id="does_expression_imply"),
        pytest.param(assert_expression_implies, id="assert_expression_implies"),
    ],
)
def test_implication_reports_an_ill_typed_consequent_behind_a_hazardous_antecedent(
    implies: Callable[[Expression, Expression, dict[Identifier, SymbolType]], object],
) -> None:
    """Test both sides are checked for ill-typedness before either hazard screen.

    The antecedent is screened for hazards first, so an ill-typed
    consequent behind a hazardous antecedent is only reported if both
    sides are checked before the screen runs.
    """
    hazardous = LiteralExpression(True).equals(LiteralExpression(1))
    ill_typed = Expression.logical_or(LiteralExpression(2), LiteralExpression(4))

    with pytest.raises(NonBooleanLogicalOperandError):
        implies(hazardous, ill_typed, {})


@pytest.mark.z3
@pytest.mark.parametrize("query", _Z3_QUESTIONS_OVER_ONE_EXPRESSION)
def test_z3_question_raises_a_missing_symbol_type_ahead_of_ill_typedness(
    query: Callable[[Expression], bool | None],
) -> None:
    """Test the ``symbol_types`` precondition still raises before ill-typedness."""
    x = mock_identifier("x", 0)
    expression = Expression.logical_and(
        LiteralExpression(2), IdentifierExpression(x)
    ).equals(LiteralExpression(1))

    with pytest.raises(KeyError):
        query(expression)


# =============================================================================
# A numeric root -- not just a numeric operand of a connective -- is refused
# =============================================================================


@pytest.mark.z3
@pytest.mark.parametrize("query", _Z3_QUESTIONS_OVER_ONE_EXPRESSION)
def test_z3_question_refuses_a_bare_numeric_root(
    query: Callable[[Expression], bool | None],
) -> None:
    """Test a numeric root, with no connective above it, is refused as ill-typed.

    A predicate is itself a Boolean position, so a bare numeric literal
    handed to any of these seams is as ill-typed as one nested under a
    connective -- even though no connective here would otherwise catch
    it.
    """
    with pytest.raises(NonBooleanLogicalOperandError):
        query(LiteralExpression(2))


@pytest.mark.z3
def test_check_expression_satisfiability_refuses_an_arithmetic_root_before_hazard() -> (
    None
):
    """Test an arithmetic root is refused before the division hazard screen runs.

    ``x / y`` denotes a number, not a predicate. The hazard screen would
    otherwise catch its unscreened divisor first and report ``None`` --
    the same signal a solver timeout uses -- hiding that the expression
    was never a valid satisfiability question to begin with.
    """
    x = mock_identifier("x", 0)
    y = mock_identifier("y", 1)
    expression = BinaryExpression(
        BinaryOperation.DIVIDE, IdentifierExpression(x), IdentifierExpression(y)
    )

    with pytest.raises(NonBooleanLogicalOperandError):
        check_expression_satisfiability(
            expression, {x: SymbolType.REAL, y: SymbolType.REAL}
        )


@pytest.mark.z3
def test_does_expression_imply_refuses_a_numeric_antecedent_before_the_hazard() -> None:
    """Test the antecedent's ill-typedness is reported before the consequent's hazard.

    The consequent references ``pi``, which the native-constant hazard
    screen would otherwise report as ``None``; the antecedent is screened
    first, so its own numeric root is what surfaces instead.
    """
    pi = get_native_constant_identifier("pi")
    consequent = BinaryExpression(
        BinaryOperation.GREATER, IdentifierExpression(pi), LiteralExpression(3)
    )

    with pytest.raises(NonBooleanLogicalOperandError):
        does_expression_imply(LiteralExpression(2), consequent, {})


@pytest.mark.z3
def test_simplify_expression_refuses_a_number_bound_into_a_connective() -> None:
    """Test an environment binding a number under a connective is refused too.

    Substitution happens after lowering, so the number would otherwise
    meet SymPy's Boolean constructor inside the bridge and leak SymPy's
    own ``TypeError`` out of this entry point.
    """
    p = mock_identifier("p", 0)
    expression = Expression.logical_and(
        IdentifierExpression(p), LiteralExpression(True)
    )

    with pytest.raises(NonBooleanLogicalOperandError):
        simplify_expression(expression, {p: LiteralExpression(2)})


@pytest.mark.z3
def test_simplify_expression_refuses_a_number_bound_into_a_case_condition() -> None:
    """Test a number bound into a piecewise condition is refused, not read as truth.

    SymPy's ``Piecewise`` takes a substituted ``1`` as a true condition, so
    the unscreened simplification would select the branch by the number's
    truthiness and fold to that branch's value.
    """
    condition = mock_identifier("c", 0)
    expression = Expression.piecewise(
        (IdentifierExpression(condition), LiteralExpression(1)),
        otherwise=LiteralExpression(0),
    )

    with pytest.raises(NonBooleanLogicalOperandError):
        simplify_expression(expression, {condition: LiteralExpression(1)})


@pytest.mark.z3
def test_both_backends_refuse_an_arithmetic_case_condition_with_one_error() -> None:
    """Test SymPy and Z3 refuse an arithmetic piecewise condition alike.

    Each backend rejects the shape natively -- SymPy's ``Piecewise`` with
    a ``TypeError``, Z3's ``If`` with a sort mismatch -- which the pass
    infrastructure wraps; the shared screen reports it as ill-typedness.
    """
    x = mock_identifier("x", 0)
    expression = Expression.piecewise(
        (IdentifierExpression(x) + LiteralExpression(1), LiteralExpression(1)),
        otherwise=LiteralExpression(0),
    ).equals(LiteralExpression(1))

    with pytest.raises(NonBooleanLogicalOperandError):
        simplify_expression(expression)
    with pytest.raises(NonBooleanLogicalOperandError):
        check_expression_satisfiability(expression, {x: SymbolType.INT})


@pytest.mark.z3
@pytest.mark.parametrize(
    "expression, expected",
    [
        pytest.param(
            Expression.logical_and(LiteralExpression(True), LiteralExpression(False)),
            False,
            id="and_true_false_is_unsatisfiable",
        ),
        pytest.param(
            Expression.logical_and(LiteralExpression(True), LiteralExpression(True)),
            True,
            id="and_true_true_is_satisfiable",
        ),
        pytest.param(
            Expression.logical_or(LiteralExpression(False), LiteralExpression(True)),
            True,
            id="or_false_true_is_satisfiable",
        ),
    ],
)
def test_check_expression_satisfiability_still_decides_a_ground_boolean_connective(
    expression: Expression, expected: bool
) -> None:
    """Test the seam still decides Boolean connectives after the screen is in place."""
    assert check_expression_satisfiability(expression, {}) is expected


@pytest.mark.z3
def test_seam_decides_a_connective_mixing_an_identifier_with_a_boolean_literal() -> (
    None
):
    """Test a BOOL-sorted identifier under a connective is decided, not screened.

    An identifier carries its sort in ``symbol_types`` rather than in the
    tree, so the screen has to leave it alone; this pins that a symbolic
    Boolean operand still reaches Z3 and gets an answer.
    """
    b = mock_identifier("b", 0)
    conjunction = Expression.logical_and(
        IdentifierExpression(b), LiteralExpression(True)
    )

    assert check_expression_satisfiability(conjunction, {b: SymbolType.BOOL}) is True
    assert (
        does_expression_imply(
            conjunction, IdentifierExpression(b), {b: SymbolType.BOOL}
        )
        is True
    )


# =============================================================================
# Both bridges honour each literal form's precision contract
# =============================================================================


def _make_three_tenths_comparison(
    one_tenth: float | str, three_tenths: float | str
) -> Expression:
    """Build ``one_tenth + one_tenth + one_tenth == three_tenths``."""
    total = BinaryExpression(
        BinaryOperation.ADD,
        BinaryExpression(
            BinaryOperation.ADD,
            LiteralExpression(one_tenth),
            LiteralExpression(one_tenth),
        ),
        LiteralExpression(one_tenth),
    )
    return BinaryExpression(
        BinaryOperation.EQUAL, total, LiteralExpression(three_tenths)
    )


@pytest.mark.z3
def test_binary_float_tenths_are_decided_the_same_way_by_both_backends() -> None:
    """Test `0.1 + 0.1 + 0.1 == 0.3` is false for simplification and the solver.

    A Python `float` is an IEEE-754 binary value, and summing three
    copies of the nearest binary value to one tenth does not give the
    nearest binary value to three tenths. Both backends have to say so:
    reading the literals' shortest reprs as exact decimal instead would
    let the solver report a witness for a comparison simplification
    refutes.
    """
    expression = _make_three_tenths_comparison(0.1, 0.3)

    simplified = simplify_expression(expression)
    satisfiable = check_expression_satisfiability(expression, {})

    assert simplified.is_structurally_equivalent(LiteralExpression(False))
    assert satisfiable is False


@pytest.mark.z3
def test_decimal_string_tenths_are_decided_the_same_way_by_both_backends() -> None:
    """Test `"0.1" + "0.1" + "0.1" == "0.3"` is true for simplification and the solver.

    A float-grammar string is exact decimal, and three exact tenths sum
    to exactly three tenths. Both backends have to say so: rounding the
    decimal text to binary instead would let simplification refute a
    comparison the solver satisfies.
    """
    expression = _make_three_tenths_comparison("0.1", "0.3")

    simplified = simplify_expression(expression)
    satisfiable = check_expression_satisfiability(expression, {})

    assert simplified.is_structurally_equivalent(LiteralExpression(True))
    assert satisfiable is True


# =============================================================================
# Native constants are refused rather than lowered as variables
# =============================================================================

_NATIVE_CONSTANT_NAMES = ["pi", "e", "inf", "nan"]

_LENIENT_Z3_QUESTIONS = [
    pytest.param(
        lambda expression: check_expression_satisfiability(expression, {}),
        id="check_expression_satisfiability",
    ),
    pytest.param(
        lambda expression: does_expression_imply(
            expression, LiteralExpression(True), {}
        ),
        id="does_expression_imply_antecedent",
    ),
    pytest.param(
        lambda expression: does_expression_imply(
            LiteralExpression(True), expression, {}
        ),
        id="does_expression_imply_consequent",
    ),
    pytest.param(
        lambda expression: holds_for_all_free_assignments(frozenset(), expression, {}),
        id="holds_for_all_free_assignments",
    ),
]

_STRICT_Z3_QUESTIONS = [
    pytest.param(
        lambda expression: assert_holds_for_all_free_assignments(
            frozenset(), expression, {}
        ),
        id="assert_holds_for_all_free_assignments",
    ),
    pytest.param(
        lambda expression: assert_expression_implies(
            expression, LiteralExpression(True), {}
        ),
        id="assert_expression_implies_antecedent",
    ),
    pytest.param(
        lambda expression: assert_expression_implies(
            LiteralExpression(True), expression, {}
        ),
        id="assert_expression_implies_consequent",
    ),
]


def _refer_to_constant(constant_name: str) -> IdentifierExpression:
    """Return a reference to the named native constant's canonical identifier."""
    return IdentifierExpression(get_native_constant_identifier(constant_name))


def _declare_real(constant_name: str) -> dict[Identifier, SymbolType]:
    """Return ``symbol_types`` giving the named constant's identifier a REAL sort."""
    return {get_native_constant_identifier(constant_name): SymbolType.REAL}


@pytest.mark.parametrize("query", _LENIENT_Z3_QUESTIONS)
@pytest.mark.parametrize("constant_name", _NATIVE_CONSTANT_NAMES)
def test_lenient_z3_question_refuses_a_native_constant(
    constant_name: str, query: Callable[[Expression], bool | None]
) -> None:
    """Test a question over a native constant is undecided, with no sort supplied.

    Z3 has no term for the constant, so the only lowering left would be a
    free variable the solver could set to anything. The canonical
    identifier names a value rather than a variable, so the question
    needs no ``symbol_types`` entry to be refused.
    """
    expression = _refer_to_constant(constant_name) > 4

    assert query(expression) is None


@pytest.mark.parametrize("query", _STRICT_Z3_QUESTIONS)
@pytest.mark.parametrize("constant_name", _NATIVE_CONSTANT_NAMES)
def test_strict_z3_question_refuses_a_native_constant_as_screened(
    constant_name: str, query: Callable[[Expression], bool]
) -> None:
    """Test the strict companions refuse a native constant with the screen's reason.

    The refusal is permanent -- no timeout lets Z3 represent ``pi`` -- so
    the reason is the hazard-screen marker rather than a solver's
    ``unknown`` text a caller might retry on.
    """
    expression = _refer_to_constant(constant_name) > 4

    with pytest.raises(UndecidableError) as exc_info:
        query(expression)

    assert exc_info.value.reason == "hazard_screen"


@pytest.mark.parametrize(
    "query",
    [
        pytest.param(
            lambda: check_expression_satisfiability(
                _refer_to_constant("pi") > 4, _declare_real("pi")
            ),
            id="pi_above_four_is_satisfiable",
        ),
        pytest.param(
            lambda: does_expression_imply(
                _refer_to_constant("pi") > 4,
                _refer_to_constant("pi") > 5,
                _declare_real("pi"),
            ),
            id="pi_above_four_implies_pi_above_five",
        ),
        pytest.param(
            lambda: holds_for_all_free_assignments(
                frozenset(), _refer_to_constant("pi") > 3, _declare_real("pi")
            ),
            id="pi_above_three_holds",
        ),
        pytest.param(
            lambda: check_expression_satisfiability(
                _refer_to_constant("inf").equals(_refer_to_constant("inf") + 1),
                _declare_real("inf"),
            ),
            id="inf_equals_inf_plus_one_is_satisfiable",
        ),
        pytest.param(
            lambda: holds_for_all_free_assignments(
                frozenset(),
                _refer_to_constant("nan").equals(_refer_to_constant("nan")),
                _declare_real("nan"),
            ),
            id="nan_equals_nan_holds",
        ),
    ],
)
def test_z3_question_over_a_native_constant_with_a_declared_sort_is_refused(
    query: Callable[[], bool | None],
) -> None:
    """Test a sort entry for a constant does not let Z3 decide over it as a variable.

    The constant's value settles each of these questions: ``pi > 4`` is
    unsatisfiable, it implies ``pi > 5`` vacuously, ``pi > 3`` holds, and
    in IEEE arithmetic ``inf == inf + 1`` is true and ``nan == nan`` is
    false. Deciding over a free REAL instead, Z3 answers every one of them
    the other way, so the sort entry is tolerated but the question is
    still refused.
    """
    assert query() is None


@pytest.mark.parametrize("query", _Z3_QUESTIONS_OVER_ONE_EXPRESSION)
def test_z3_question_still_requires_a_sort_for_a_variable_beside_a_constant(
    query: Callable[[Expression], bool | None],
) -> None:
    """Test a variable's missing sort raises ahead of the constant screen.

    The constant needs no entry, so the error names only the variable the
    caller forgot to declare.
    """
    x = mock_identifier("x", 0)
    pi = get_native_constant_identifier("pi")
    expression = BinaryExpression(
        BinaryOperation.LESS, IdentifierExpression(pi), IdentifierExpression(x)
    )

    with pytest.raises(KeyError) as exc_info:
        query(expression)

    assert repr(x) in str(exc_info.value)
    assert repr(pi) not in str(exc_info.value)


@pytest.mark.parametrize("query", _Z3_QUESTIONS_OVER_ONE_EXPRESSION)
def test_z3_question_reports_ill_typedness_ahead_of_a_native_constant(
    query: Callable[[Expression], bool | None],
) -> None:
    """Test an ill-typed tree that also references a constant raises its own error.

    The constant makes the question unanswerable by this backend; the
    numeric ``and`` operand makes it meaningless to every backend, which
    is the diagnosis a caller can act on.
    """
    expression = Expression.logical_and(
        LiteralExpression(2), _refer_to_constant("pi") > 3
    )

    with pytest.raises(NonBooleanLogicalOperandError):
        query(expression)


@pytest.mark.parametrize("query", _Z3_QUESTIONS_OVER_ONE_EXPRESSION)
@pytest.mark.parametrize("constant_name", _NATIVE_CONSTANT_NAMES)
def test_z3_question_reports_a_native_constant_under_a_connective_as_ill_typed(
    constant_name: str, query: Callable[[Expression], bool | None]
) -> None:
    """Test a constant under a connective raises the typed error, not a refusal.

    The constant screen would refuse the question as well, but a
    REAL-sorted constant under ``and`` is meaningless to every backend,
    not just unanswerable by this one, and that is the diagnosis a caller
    can act on.
    """
    expression = Expression.logical_and(
        _refer_to_constant(constant_name), LiteralExpression(True)
    )

    with pytest.raises(NonBooleanLogicalOperandError):
        query(expression)


_Z3_QUESTIONS_OVER_ONE_SORTED_EXPRESSION = [
    pytest.param(check_expression_satisfiability, id="check_expression_satisfiability"),
    pytest.param(
        lambda expression, symbol_types: does_expression_imply(
            expression, LiteralExpression(True), symbol_types
        ),
        id="does_expression_imply_antecedent",
    ),
    pytest.param(
        lambda expression, symbol_types: does_expression_imply(
            LiteralExpression(True), expression, symbol_types
        ),
        id="does_expression_imply_consequent",
    ),
    pytest.param(
        lambda expression, symbol_types: holds_for_all_free_assignments(
            frozenset(), expression, symbol_types
        ),
        id="holds_for_all_free_assignments",
    ),
    pytest.param(
        lambda expression, symbol_types: assert_holds_for_all_free_assignments(
            frozenset(), expression, symbol_types
        ),
        id="assert_holds_for_all_free_assignments",
    ),
    pytest.param(
        lambda expression, symbol_types: assert_expression_implies(
            expression, LiteralExpression(True), symbol_types
        ),
        id="assert_expression_implies_antecedent",
    ),
    pytest.param(
        lambda expression, symbol_types: assert_expression_implies(
            LiteralExpression(True), expression, symbol_types
        ),
        id="assert_expression_implies_consequent",
    ),
]

_SortedQuery = Callable[[Expression, dict[Identifier, SymbolType]], bool | None]


@pytest.mark.parametrize("query", _Z3_QUESTIONS_OVER_ONE_SORTED_EXPRESSION)
@pytest.mark.parametrize("sort", [SymbolType.INT, SymbolType.REAL])
@pytest.mark.parametrize(
    "build_expression",
    [
        pytest.param(
            lambda operand: Expression.logical_and(operand, LiteralExpression(True)),
            id="and",
        ),
        pytest.param(
            lambda operand: (
                PiecewiseExpression(
                    (operand,), (LiteralExpression(1),), LiteralExpression(2)
                )
                > 0
            ),
            id="case_condition",
        ),
    ],
)
def test_z3_question_reports_a_numeric_sort_in_a_boolean_position(
    build_expression: Callable[[Expression], Expression],
    sort: SymbolType,
    query: _SortedQuery,
) -> None:
    """Test an identifier declared INT or REAL in a Boolean position is ill-typed.

    Z3 rejected the sort mismatch with a raw exception, which surfaced
    wrapped as ``PassExecutionError``: a failure of the tool, when the
    question itself has no meaning under any backend.
    """
    x = mock_identifier("x", 0)

    with pytest.raises(NonBooleanLogicalOperandError):
        query(build_expression(IdentifierExpression(x)), {x: sort})


@pytest.mark.parametrize("query", _Z3_QUESTIONS_OVER_ONE_SORTED_EXPRESSION)
def test_symbol_typed_ill_typedness_is_reported_despite_a_hazard_elsewhere(
    query: _SortedQuery,
) -> None:
    """Test an INT-declared identifier under a connective raises despite a hazard.

    ``y / 0`` is a division hazard the screen alone would report as
    ``None``, but ``x`` sits directly under ``logical_and`` and is
    declared INT in ``symbol_types``, so the expression is ill-typed:
    every entry point that accepts ``symbol_types`` has to read it when
    classifying a Boolean-position operand, not only when screening a
    hazard, or a numeric operand there would be reported as merely
    undecided instead of ill-typed.
    """
    x = mock_identifier("x", 0)
    y = mock_identifier("y", 1)
    hazardous_division = BinaryExpression(
        BinaryOperation.DIVIDE, IdentifierExpression(y), LiteralExpression(0)
    ).equals(LiteralExpression(1))
    expression = Expression.logical_and(IdentifierExpression(x), hazardous_division)

    with pytest.raises(NonBooleanLogicalOperandError):
        query(expression, {x: SymbolType.INT, y: SymbolType.REAL})


@pytest.mark.parametrize("query", _Z3_QUESTIONS_OVER_ONE_SORTED_EXPRESSION)
def test_z3_question_decides_a_boolean_sort_in_a_boolean_position(
    query: _SortedQuery,
) -> None:
    """Test an identifier declared BOOL under a connective is still decided."""
    x = mock_identifier("x", 0)
    expression = Expression.logical_and(
        IdentifierExpression(x), LiteralExpression(True)
    )

    assert isinstance(query(expression, {x: SymbolType.BOOL}), bool)


def test_native_constant_screen_warns_naming_the_constant_and_the_entry_point(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Test the refusal is logged with the constant it refused and who asked."""
    pi = get_native_constant_identifier("pi")

    with caplog.at_level(logging.WARNING):
        result = check_expression_satisfiability(IdentifierExpression(pi) > 4, {})

    assert result is None
    messages = _collect_solver_warning_messages(caplog)
    assert messages, "expected a WARNING naming the native constant"
    assert "check_expression_satisfiability" in messages[0]
    assert repr(pi) in messages[0]


@pytest.mark.z3
def test_holds_for_all_free_assignments_needs_no_sort_for_a_considered_constant() -> (
    None
):
    """Test a considered constant the expression never references needs no entry.

    Considering a constant quantifies nothing, and with no reference to it
    in the tree there is nothing to refuse, so the question is decided as
    though the constant had not been named.
    """
    x = mock_identifier("x", 0)
    pi = get_native_constant_identifier("pi")
    expression = IdentifierExpression(x).equals(IdentifierExpression(x))

    result = holds_for_all_free_assignments(
        frozenset({pi}), expression, {x: SymbolType.INT}
    )

    assert result is True


# =============================================================================
# Non-finite literal hazard
# =============================================================================


_NON_FINITE_LITERAL_HAZARD_EXPRESSIONS = [
    pytest.param(
        lambda x: IdentifierExpression(x) < LiteralExpression(float("inf")),
        id="less-than-positive-infinity",
    ),
    pytest.param(
        lambda x: IdentifierExpression(x) > LiteralExpression(float("-inf")),
        id="greater-than-negative-infinity",
    ),
    pytest.param(
        lambda x: IdentifierExpression(x).equals(LiteralExpression(float("nan"))),
        id="equals-nan",
    ),
]

_LENIENT_Z3_QUESTIONS_WITH_SORTS = [
    pytest.param(check_expression_satisfiability, id="check_expression_satisfiability"),
    pytest.param(
        lambda expression, symbol_types: does_expression_imply(
            expression, LiteralExpression(True), symbol_types
        ),
        id="does_expression_imply_antecedent",
    ),
    pytest.param(
        lambda expression, symbol_types: does_expression_imply(
            LiteralExpression(True), expression, symbol_types
        ),
        id="does_expression_imply_consequent",
    ),
    pytest.param(
        lambda expression, symbol_types: holds_for_all_free_assignments(
            frozenset(), expression, symbol_types
        ),
        id="holds_for_all_free_assignments",
    ),
]

_STRICT_Z3_QUESTIONS_WITH_SORTS = [
    pytest.param(
        lambda expression, symbol_types: assert_holds_for_all_free_assignments(
            frozenset(), expression, symbol_types
        ),
        id="assert_holds_for_all_free_assignments",
    ),
    pytest.param(
        lambda expression, symbol_types: assert_expression_implies(
            expression, LiteralExpression(True), symbol_types
        ),
        id="assert_expression_implies_antecedent",
    ),
    pytest.param(
        lambda expression, symbol_types: assert_expression_implies(
            LiteralExpression(True), expression, symbol_types
        ),
        id="assert_expression_implies_consequent",
    ),
]


@pytest.mark.parametrize("query", _LENIENT_Z3_QUESTIONS_WITH_SORTS)
@pytest.mark.parametrize("build_expression", _NON_FINITE_LITERAL_HAZARD_EXPRESSIONS)
def test_lenient_z3_question_refuses_a_non_finite_literal(
    build_expression: Callable[[Identifier], Expression],
    query: _SortedQuery,
) -> None:
    """Test an inf, -inf, or NaN literal anywhere in the tree is refused, not lowered.

    ``as_integer_ratio`` has no rational value for an infinity or a NaN,
    so the Z3 bridge cannot lower the literal at all; the screen refuses
    the question before the bridge is tried, the same as it does for a
    native constant.
    """
    x = mock_identifier("x", 0)
    expression = build_expression(x)

    assert query(expression, {x: SymbolType.REAL}) is None


@pytest.mark.parametrize("query", _STRICT_Z3_QUESTIONS_WITH_SORTS)
@pytest.mark.parametrize("build_expression", _NON_FINITE_LITERAL_HAZARD_EXPRESSIONS)
def test_strict_z3_question_refuses_a_non_finite_literal_as_screened(
    build_expression: Callable[[Identifier], Expression],
    query: _SortedQuery,
) -> None:
    """Test the strict companions raise `UndecidableError` with the screen's reason."""
    x = mock_identifier("x", 0)
    expression = build_expression(x)

    with pytest.raises(UndecidableError) as exc_info:
        query(expression, {x: SymbolType.REAL})

    assert exc_info.value.reason == "hazard_screen"


def test_non_finite_literal_screen_warns_naming_the_entry_point(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Test the refusal is logged with the entry point that asked."""
    x = mock_identifier("x", 0)
    expression = IdentifierExpression(x) < LiteralExpression(float("inf"))

    with caplog.at_level(logging.WARNING):
        result = check_expression_satisfiability(expression, {x: SymbolType.REAL})

    assert result is None
    messages = _collect_solver_warning_messages(caplog)
    assert messages, "expected a WARNING naming the non-finite literal"
    assert "check_expression_satisfiability" in messages[0]


def test_z3_question_requires_a_sort_beside_a_non_finite_literal() -> None:
    """Test a variable's missing sort raises ahead of the non-finite-literal screen."""
    x = mock_identifier("x", 0)
    expression = IdentifierExpression(x) < LiteralExpression(float("inf"))

    with pytest.raises(KeyError) as exc_info:
        check_expression_satisfiability(expression, {})

    assert repr(x) in str(exc_info.value)


@pytest.mark.z3
def test_check_expression_satisfiability_decides_a_finite_float_literal() -> None:
    """Test a large but finite float literal is still lowered and decided."""
    x = mock_identifier("x", 0)
    expression = IdentifierExpression(x) < LiteralExpression(1e300)

    assert check_expression_satisfiability(expression, {x: SymbolType.REAL}) is True
