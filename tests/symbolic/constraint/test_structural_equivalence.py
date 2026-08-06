"""Tests for `Constraint.is_structurally_equivalent` across kinds."""

from collections.abc import Callable
from typing import Any

import pytest

from fhy_core.identifier import Identifier
from fhy_core.serialization import SerializedDict
from fhy_core.symbolic.constraint import (
    Constraint,
    ConstraintOutcome,
    EquationConstraint,
    InSetConstraint,
)
from fhy_core.symbolic.expression import Expression, LiteralExpression
from fhy_core.traits import StructuralEquivalence
from fhy_core.traits.derived_equivalence import EquivalenceDerivationError
from fhy_core.utils.override import override

from .conftest import (
    ALL_KINDS,
    SET_KINDS,
    build_equation_constraint,
    build_in_set_constraint,
    build_not_in_set_constraint,
    mock_identifier,
)

ConstraintFactory = Callable[[Identifier], Constraint]
SetConstraintFactory = Callable[[Identifier, Any], Constraint]


# =============================================================================
# Same-kind equivalence
# =============================================================================


@pytest.mark.parametrize("factory", ALL_KINDS)
def test_structural_equivalence_is_reflexive(
    factory: ConstraintFactory,
) -> None:
    """Test ``c.is_structurally_equivalent(c)`` holds for every kind."""
    constraint = factory(mock_identifier("x", 0))

    assert constraint.is_structurally_equivalent(constraint)


@pytest.mark.parametrize("factory", ALL_KINDS)
def test_constraint_equivalent_when_constructed_with_equal_inputs(
    factory: ConstraintFactory,
) -> None:
    """Test distinct identifier instances with equal ids compare equivalent."""
    left = factory(mock_identifier("x", 0))
    right = factory(mock_identifier("x", 0))

    assert left.is_structurally_equivalent(right)
    assert right.is_structurally_equivalent(left)


@pytest.mark.parametrize("factory", ALL_KINDS)
def test_constraint_inequivalent_for_different_variables(
    factory: ConstraintFactory,
) -> None:
    """Test different variables make constraints inequivalent."""
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


@pytest.mark.parametrize("factory", SET_KINDS)
def test_set_constraint_inequivalent_for_different_values(
    factory: SetConstraintFactory,
) -> None:
    """Test different value sets make set constraints inequivalent."""
    x = mock_identifier("x", 0)
    left = factory(x, {1, 2})
    right = factory(x, {1, 3})

    assert not left.is_structurally_equivalent(right)


@pytest.mark.parametrize("factory", SET_KINDS)
def test_set_constraint_uses_value_equality_not_identity(
    factory: SetConstraintFactory,
) -> None:
    """Test independent collections with equal contents are equivalent."""
    x = mock_identifier("x", 0)
    left = factory(x, [1, 2])
    right = factory(x, [2, 1])

    assert left.is_structurally_equivalent(right)


@pytest.mark.parametrize("factory", SET_KINDS)
def test_set_constraint_inequivalent_for_strict_subset_values(
    factory: SetConstraintFactory,
) -> None:
    """Test strict subset/superset value sets are inequivalent."""
    x = mock_identifier("x", 0)
    left = factory(x, {1, 2})
    right = factory(x, {1, 2, 3})

    assert not left.is_structurally_equivalent(right)
    assert not right.is_structurally_equivalent(left)


# =============================================================================
# Alpha equivalence (set-kind member-collection comparisons)
# =============================================================================


@pytest.mark.parametrize("factory", SET_KINDS)
def test_set_constraint_alpha_equivalence_matches_structural_for_same_variable(
    factory: SetConstraintFactory,
) -> None:
    """Test alpha equivalence agrees with structural equivalence for one variable."""
    x = mock_identifier("x", 0)
    left = factory(x, [1, 2])
    right = factory(x, [2, 1])

    assert left.is_alpha_equivalent(right) == left.is_structurally_equivalent(right)
    assert left.is_alpha_equivalent(right) is True


@pytest.mark.parametrize("factory", SET_KINDS)
def test_set_constraint_alpha_inequivalent_for_different_values(
    factory: SetConstraintFactory,
) -> None:
    """Test alpha equivalence is `False` when the member sets differ."""
    x = mock_identifier("x", 0)
    left = factory(x, {1, 2})
    right = factory(x, {1, 3})

    assert not left.is_alpha_equivalent(right)


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
    """Test cross-kind equivalence returns ``False``."""
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
# Derived-equivalence requirements
# =============================================================================


def test_equivalence_requires_dataclass_constraint_subclass() -> None:
    """Test a non-dataclass `Constraint` subclass fails derived equivalence.

    Structural equivalence is derived from the dataclass field schema, so a
    concrete `Constraint` subclass that is not a dataclass raises
    ``EquivalenceDerivationError`` on first comparison instead of silently
    comparing.
    """

    class _NonDataclassConstraint(Constraint):
        def __init__(self, variable: Identifier) -> None:
            object.__setattr__(self, "_variable", variable)

        @override
        def evaluate(self, value: object) -> ConstraintOutcome:
            return ConstraintOutcome.SATISFIED

        @override
        def convert_to_expression(self) -> Expression:
            return LiteralExpression(True)

        @override
        def __repr__(self) -> str:
            return "_NonDataclassConstraint"

        @override
        def __str__(self) -> str:
            return "_NonDataclassConstraint"

        @override
        def serialize_data_to_dict(self) -> SerializedDict:
            return {}

        @classmethod
        @override
        def deserialize_data_from_dict(
            cls, data: SerializedDict
        ) -> "_NonDataclassConstraint":
            return cls(mock_identifier("stub", 0))

    x = mock_identifier("x", 0)
    a = _NonDataclassConstraint(x)
    b = _NonDataclassConstraint(x)

    with pytest.raises(EquivalenceDerivationError):
        a.is_structurally_equivalent(b)
