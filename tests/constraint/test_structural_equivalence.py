"""Tests for `Constraint.is_structurally_equivalent` across kinds.

Each singledispatch handler shares the same three-predicate shape
(``isinstance``, ``variable ==``, member-collection equivalence). The
tests are parametrized across kinds so each handler is exercised
through identical scenarios.
"""

from typing import Any, Callable

import pytest

from fhy_core.constraint import (
    Constraint,
    EquationConstraint,
    InSetConstraint,
    NotInSetConstraint,
)
from fhy_core.expression import Expression, LiteralExpression
from fhy_core.identifier import Identifier
from fhy_core.serialization import SerializedDict
from fhy_core.trait import StructuralEquivalence

from .conftest import (
    build_equation_constraint,
    build_in_set_constraint,
    build_not_in_set_constraint,
    mock_identifier,
)

ConstraintFactory = Callable[[Identifier], Constraint]
SetConstraintFactory = Callable[[Identifier, Any], Constraint]


_ALL_KINDS = [
    pytest.param(build_equation_constraint, id="equation"),
    pytest.param(build_in_set_constraint, id="in_set"),
    pytest.param(build_not_in_set_constraint, id="not_in_set"),
]

_SET_KINDS = [
    pytest.param(InSetConstraint, id="in_set"),
    pytest.param(NotInSetConstraint, id="not_in_set"),
]


# =============================================================================
# Same-kind equivalence
# =============================================================================


@pytest.mark.parametrize("factory", _ALL_KINDS)
def test_structural_equivalence_is_reflexive(
    factory: ConstraintFactory,
) -> None:
    """Test ``c.is_structurally_equivalent(c)`` holds for every kind."""
    constraint = factory(mock_identifier("x", 0))
    assert constraint.is_structurally_equivalent(constraint)


@pytest.mark.parametrize("factory", _ALL_KINDS)
def test_constraint_equivalent_when_constructed_with_equal_inputs(
    factory: ConstraintFactory,
) -> None:
    """Test distinct identifier instances with equal ids compare equivalent."""
    left = factory(mock_identifier("x", 0))
    right = factory(mock_identifier("x", 0))
    assert left.is_structurally_equivalent(right)
    assert right.is_structurally_equivalent(left)


@pytest.mark.parametrize("factory", _ALL_KINDS)
def test_constraint_inequivalent_for_different_variables(
    factory: ConstraintFactory,
) -> None:
    """Test different variables make constraints inequivalent.

    Also kills the ``< / <= / > / >=`` mutations on ``variable``: the
    mock identifier carries no ordering, so the mutant raises
    ``TypeError``.
    """
    left = factory(mock_identifier("x", 0))
    right = factory(mock_identifier("y", 1))
    assert not left.is_structurally_equivalent(right)


def test_equation_constraint_inequivalent_for_different_expressions() -> None:
    """Test equation constraints with different expressions are inequivalent."""
    x = mock_identifier("x", 0)
    left = EquationConstraint(x, LiteralExpression(True))
    right = EquationConstraint(x, LiteralExpression(False))
    assert not left.is_structurally_equivalent(right)


# =============================================================================
# Set-kind member-collection comparisons
# =============================================================================


@pytest.mark.parametrize("factory", _SET_KINDS)
def test_set_constraint_inequivalent_for_different_values(
    factory: SetConstraintFactory,
) -> None:
    """Test different value sets make set constraints inequivalent.

    Kills set-comparison mutations (``<``, ``<=``, ``>``, ``>=``, ``is``).
    """
    x = mock_identifier("x", 0)
    left = factory(x, {1, 2})
    right = factory(x, {1, 3})
    assert not left.is_structurally_equivalent(right)


@pytest.mark.parametrize("factory", _SET_KINDS)
def test_set_constraint_uses_value_equality_not_identity(
    factory: SetConstraintFactory,
) -> None:
    """Test independent collections with equal contents are equivalent.

    Kills the ``is`` mutation: distinct frozenset objects compare
    unequal by identity but equal by value.
    """
    x = mock_identifier("x", 0)
    left = factory(x, [1, 2])
    right = factory(x, [2, 1])
    assert left.is_structurally_equivalent(right)


@pytest.mark.parametrize("factory", _SET_KINDS)
def test_set_constraint_inequivalent_for_strict_subset_values(
    factory: SetConstraintFactory,
) -> None:
    """Test strict subset/superset value sets are inequivalent.

    Kills the ``<=`` / ``>=`` mutations specifically.
    """
    x = mock_identifier("x", 0)
    left = factory(x, {1, 2})
    right = factory(x, {1, 2, 3})
    assert not left.is_structurally_equivalent(right)
    assert not right.is_structurally_equivalent(left)


# =============================================================================
# Cross-kind / non-Constraint comparisons
# =============================================================================


@pytest.mark.parametrize(
    "left_factory, right_factory",
    [
        pytest.param(
            build_equation_constraint,
            build_in_set_constraint,
            id="equation_vs_in_set",
        ),
        pytest.param(
            build_equation_constraint,
            build_not_in_set_constraint,
            id="equation_vs_not_in_set",
        ),
        pytest.param(
            build_in_set_constraint,
            build_not_in_set_constraint,
            id="in_set_vs_not_in_set",
        ),
    ],
)
def test_constraint_inequivalent_across_kinds(
    left_factory: ConstraintFactory, right_factory: ConstraintFactory
) -> None:
    """Test cross-kind equivalence short-circuits on ``isinstance``.

    Drives the ``and -> or`` mutation in dispatch handlers: relaxing
    the conjunction would let the call evaluate the wrong-kind's
    members and either raise or wrongly return ``True``.
    """
    x = mock_identifier("x", 0)
    left = left_factory(x)
    right = right_factory(x)
    assert left.is_structurally_equivalent(right) is False
    assert right.is_structurally_equivalent(left) is False


@pytest.mark.parametrize(
    "other",
    [
        pytest.param("not-a-constraint", id="string"),
        pytest.param(None, id="none"),
        pytest.param(object(), id="object"),
    ],
)
def test_constraint_inequivalent_against_arbitrary_object(other: object) -> None:
    """Test equivalence against non-`Constraint` objects always returns ``False``."""
    constraint = EquationConstraint(mock_identifier("x", 0), LiteralExpression(True))
    assert not constraint.is_structurally_equivalent(other)


def test_constraint_satisfies_structural_equivalence_protocol() -> None:
    """Test `Constraint` instances satisfy the `StructuralEquivalence` protocol."""
    constraint = InSetConstraint(mock_identifier("x", 0), {1, 2})
    assert isinstance(constraint, StructuralEquivalence)


# =============================================================================
# Singledispatch registration / fallback
# =============================================================================


def test_dispatch_default_returns_false_for_unregistered_constraint_subclass() -> None:
    """Test the singledispatch default branch returns ``False``.

    A `Constraint` subclass that isn't registered with
    ``_is_constraint_structurally_equivalent`` must always compare
    inequivalent (the registry default is ``False``).
    """

    class _UnregisteredConstraint(Constraint):
        def is_satisfied(self, value: object) -> bool:  # pragma: no cover - stub
            return True

        def convert_to_expression(self) -> Expression:  # pragma: no cover - stub
            return LiteralExpression(True)

        def __repr__(self) -> str:  # pragma: no cover - stub
            return "_UnregisteredConstraint"

        def __str__(self) -> str:  # pragma: no cover - stub
            return "_UnregisteredConstraint"

        def serialize_data_to_dict(self) -> SerializedDict:  # pragma: no cover - stub
            return {}

        @classmethod
        def deserialize_data_from_dict(
            cls, data: SerializedDict
        ) -> "_UnregisteredConstraint":  # pragma: no cover - stub
            return cls(mock_identifier("stub", 0))

    x = mock_identifier("x", 0)
    a = _UnregisteredConstraint(x)
    b = _UnregisteredConstraint(x)
    assert a.is_structurally_equivalent(b) is False
    assert b.is_structurally_equivalent(a) is False
