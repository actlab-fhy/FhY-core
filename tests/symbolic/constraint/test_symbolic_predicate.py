"""Tests for the `SymbolicPredicate` protocol conformance."""

from typing import Any

import pytest

from fhy_core.symbolic.constraint import (
    Constraint,
    ConstraintBindings,
    ConstraintOutcome,
    ConstraintSystem,
    EquationConstraint,
    InSetConstraint,
    NotInSetConstraint,
    SymbolicPredicate,
    create_constraint_system,
)
from fhy_core.symbolic.expression import (
    Expression,
    IdentifierExpression,
    LiteralExpression,
)

from .conftest import mock_identifier

# =============================================================================
# `Constraint` leaves satisfy `SymbolicPredicate`
# =============================================================================


def test_equation_constraint_is_a_symbolic_predicate() -> None:
    """Test `EquationConstraint` satisfies `SymbolicPredicate` by isinstance."""
    constraint = EquationConstraint(LiteralExpression(True))

    assert isinstance(constraint, SymbolicPredicate)


def test_in_set_constraint_is_a_symbolic_predicate() -> None:
    """Test `InSetConstraint` satisfies `SymbolicPredicate` by isinstance."""
    constraint = InSetConstraint(mock_identifier("x", 0), {1, 2})

    assert isinstance(constraint, SymbolicPredicate)


def test_not_in_set_constraint_is_a_symbolic_predicate() -> None:
    """Test `NotInSetConstraint` satisfies `SymbolicPredicate` by isinstance."""
    constraint = NotInSetConstraint(mock_identifier("x", 0), {1, 2})

    assert isinstance(constraint, SymbolicPredicate)


def test_constraint_base_is_declared_as_a_symbolic_predicate_subclass() -> None:
    """Test `Constraint` declares `SymbolicPredicate` as an explicit base."""
    assert issubclass(Constraint, SymbolicPredicate)


# =============================================================================
# `ConstraintSystem` satisfies `SymbolicPredicate`
# =============================================================================


def test_constraint_system_is_a_symbolic_predicate() -> None:
    """Test `ConstraintSystem` satisfies `SymbolicPredicate` by isinstance."""
    system = create_constraint_system(InSetConstraint(mock_identifier("x", 0), {1, 2}))

    assert isinstance(system, SymbolicPredicate)


def test_empty_constraint_system_is_a_symbolic_predicate() -> None:
    """Test the empty `ConstraintSystem` also satisfies `SymbolicPredicate`."""
    system = create_constraint_system()

    assert isinstance(system, SymbolicPredicate)


def test_constraint_system_is_declared_as_a_symbolic_predicate_subclass() -> None:
    """Test `ConstraintSystem` declares `SymbolicPredicate` as an explicit base."""
    assert issubclass(ConstraintSystem, SymbolicPredicate)


# =============================================================================
# One generic consumer works over both shapes
# =============================================================================


def _count_undecided(
    predicate: SymbolicPredicate, bindings: ConstraintBindings
) -> bool:
    return predicate.evaluate_with_bindings(bindings) is ConstraintOutcome.UNDECIDED


def test_generic_consumer_accepts_a_constraint_leaf() -> None:
    """Test a function typed over `SymbolicPredicate` accepts a `Constraint` leaf."""
    x = mock_identifier("x", 0)
    constraint = InSetConstraint(x, {1, 2})

    assert _count_undecided(constraint, {}) is True
    assert _count_undecided(constraint, {x: 1}) is False


def test_generic_consumer_accepts_a_constraint_system() -> None:
    """Test a function typed over `SymbolicPredicate` accepts a `ConstraintSystem`."""
    x = mock_identifier("x", 0)
    system = create_constraint_system(InSetConstraint(x, {1, 2}))

    assert _count_undecided(system, {}) is True
    assert _count_undecided(system, {x: 1}) is False


# =============================================================================
# Structural (non-inheriting) third-party conformer
# =============================================================================


class _StructuralConformer:
    """A predicate satisfying `SymbolicPredicate` purely structurally.

    Declares no inheritance relationship to `Constraint`, `ConstraintSystem`,
    or `SymbolicPredicate` itself; only implements the four methods the
    protocol demands. Used to prove `SymbolicPredicate` is genuinely
    `@runtime_checkable` and structural, not nominal.
    """

    def get_free_identifiers(self) -> frozenset[Any]:
        return frozenset()

    def evaluate_with_bindings(self, bindings: ConstraintBindings) -> ConstraintOutcome:
        return ConstraintOutcome.SATISFIED

    def is_satisfied_with_bindings(self, bindings: ConstraintBindings) -> bool:
        return True

    def convert_to_expression(self) -> Expression:
        return LiteralExpression(True)


def test_structural_third_party_conformer_satisfies_the_protocol() -> None:
    """Test a class with no nominal relationship to the protocol still conforms."""
    conformer = _StructuralConformer()

    assert isinstance(conformer, SymbolicPredicate)
    assert not isinstance(conformer, Constraint)
    assert not isinstance(conformer, ConstraintSystem)


@pytest.mark.parametrize(
    "missing_method",
    [
        "get_free_identifiers",
        "evaluate_with_bindings",
        "is_satisfied_with_bindings",
        "convert_to_expression",
    ],
)
def test_structural_conformer_missing_a_method_fails_isinstance(
    missing_method: str,
) -> None:
    """Test omitting any one of the four methods breaks structural conformance."""
    namespace = {
        name: getattr(_StructuralConformer, name)
        for name in (
            "get_free_identifiers",
            "evaluate_with_bindings",
            "is_satisfied_with_bindings",
            "convert_to_expression",
        )
        if name != missing_method
    }
    incomplete_cls = type("_IncompleteConformer", (), namespace)

    assert not isinstance(incomplete_cls(), SymbolicPredicate)


def test_identifier_expression_does_not_satisfy_symbolic_predicate() -> None:
    """Test an unrelated object with a disjoint interface does not conform."""
    assert not isinstance(
        IdentifierExpression(mock_identifier("x", 0)), SymbolicPredicate
    )
    assert not isinstance(object(), SymbolicPredicate)
    assert not isinstance(42, SymbolicPredicate)
