"""Tests for `Constraint.is_structurally_equivalent` across kinds."""

from collections.abc import Callable
from typing import Any

import pytest

from fhy_core.identifier import Identifier
from fhy_core.serialization import SerializedDict
from fhy_core.symbolic.constraint import (
    Constraint,
    ConstraintBindings,
    ConstraintOutcome,
    EquationConstraint,
    InSetConstraint,
    NotInSetConstraint,
)
from fhy_core.symbolic.expression import Expression, LiteralExpression
from fhy_core.term.derived_equivalence import EquivalenceDerivationError
from fhy_core.traits import StructuralEquivalence
from fhy_core.utils.override import override

from .conftest import (
    ALL_KINDS,
    SET_KINDS,
    HashCollidingMember,
    build_equation_constraint,
    build_in_set_constraint,
    build_not_in_set_constraint,
    mock_identifier,
)

ConstraintFactory = Callable[[Identifier], Constraint]
SetConstraintFactory = Callable[[Identifier, Any], Constraint]

_SET_KINDS_WITH_FIELD = [
    pytest.param(InSetConstraint, "valid_values", id="in_set"),
    pytest.param(NotInSetConstraint, "invalid_values", id="not_in_set"),
]
"""Parametrize list pairing each set-constraint kind with its member field."""


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
    left = EquationConstraint(LiteralExpression(True))
    right = EquationConstraint(LiteralExpression(False))

    assert not left.is_structurally_equivalent(right)


def test_equation_constraint_equivalent_for_ground_expressions_with_no_scope() -> None:
    """Test two ground equation constraints over the same literal are equivalent.

    Ground constraints have an empty scope; equivalence still has to
    agree, since they carry no variable to distinguish them by.
    """
    left = EquationConstraint(LiteralExpression(True))
    right = EquationConstraint(LiteralExpression(True))

    assert left.is_structurally_equivalent(right)


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


@pytest.mark.parametrize("factory, field_name", _SET_KINDS_WITH_FIELD)
def test_set_constraint_uses_value_equality_not_identity(
    factory: SetConstraintFactory, field_name: str
) -> None:
    """Test independent collections with equal contents are equivalent.

    The two constraints are built from the same members in opposite
    orders and store them in genuinely different orders, so equivalence
    has to normalize rather than lean on the stored tuples comparing
    equal by accident.
    """
    x = mock_identifier("x", 0)
    members = [HashCollidingMember(1), HashCollidingMember(2)]
    left = factory(x, list(members))
    right = factory(x, list(reversed(members)))

    assert getattr(left, field_name) != getattr(right, field_name), (
        "the two constraints must store their members in different orders "
        "for this test to say anything about order independence"
    )
    assert left.is_structurally_equivalent(right)
    assert right.is_structurally_equivalent(left)


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


@pytest.mark.parametrize("factory, field_name", _SET_KINDS_WITH_FIELD)
def test_set_constraint_alpha_equivalence_matches_structural_for_same_variable(
    factory: SetConstraintFactory, field_name: str
) -> None:
    """Test alpha equivalence agrees with structural equivalence for one variable.

    As for the structural case, the two constraints genuinely store their
    members in different orders, so the agreement is not an artifact of
    the stored tuples being identical.
    """
    x = mock_identifier("x", 0)
    members = [HashCollidingMember(1), HashCollidingMember(2)]
    left = factory(x, list(members))
    right = factory(x, list(reversed(members)))

    assert getattr(left, field_name) != getattr(right, field_name), (
        "the two constraints must store their members in different orders "
        "for this test to say anything about order independence"
    )
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
# Type-strict member comparison
# =============================================================================

_TYPE_STRICT_MEMBER_PAIRS = [
    pytest.param([1], [True], id="int_vs_bool"),
    pytest.param([1], [1.0], id="int_vs_float"),
]
"""Member pairs that plain ``==`` on the stored tuples cannot tell apart.

``(1,) == (True,)`` and ``(1,) == (1.0,)`` are both ``True`` in Python,
yet membership is type-strict:
``InSetConstraint(x, [1]).evaluate_with_bindings({x: True})`` reports
``VIOLATED``. Equivalence has to agree with evaluation, so the member
field is compared through the type-strict normalizer rather than by the
stored tuples' own equality.
"""


@pytest.mark.parametrize("factory", SET_KINDS)
@pytest.mark.parametrize("left_members, right_members", _TYPE_STRICT_MEMBER_PAIRS)
def test_set_constraint_inequivalent_for_members_differing_only_in_type(
    factory: SetConstraintFactory,
    left_members: list[Any],
    right_members: list[Any],
) -> None:
    """Test members that are ``==``-equal but differently typed are inequivalent."""
    x = mock_identifier("x", 0)
    left = factory(x, left_members)
    right = factory(x, right_members)

    assert not left.is_structurally_equivalent(right)
    assert not right.is_structurally_equivalent(left)


@pytest.mark.parametrize("factory", SET_KINDS)
@pytest.mark.parametrize("left_members, right_members", _TYPE_STRICT_MEMBER_PAIRS)
def test_set_constraint_alpha_inequivalent_for_members_differing_only_in_type(
    factory: SetConstraintFactory,
    left_members: list[Any],
    right_members: list[Any],
) -> None:
    """Test alpha equivalence is also type-strict about member types."""
    x = mock_identifier("x", 0)
    left = factory(x, left_members)
    right = factory(x, right_members)

    assert not left.is_alpha_equivalent(right)
    assert not right.is_alpha_equivalent(left)


@pytest.mark.parametrize("factory", SET_KINDS)
@pytest.mark.parametrize("left_members, right_members", _TYPE_STRICT_MEMBER_PAIRS)
def test_set_constraint_equivalence_agrees_with_evaluation_on_member_types(
    factory: SetConstraintFactory,
    left_members: list[Any],
    right_members: list[Any],
) -> None:
    """Test two constraints called equivalent never disagree on a member probe.

    This is the invariant the type-strict comparison exists to protect:
    were the two treated as equivalent, one would report the other's
    member as a member and the other would not.
    """
    x = mock_identifier("x", 0)
    left = factory(x, left_members)
    right = factory(x, right_members)

    assert left.evaluate_with_bindings(
        {x: right_members[0]}
    ) is not left.evaluate_with_bindings({x: left_members[0]})
    assert not left.is_structurally_equivalent(right)


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
    constraint = EquationConstraint(LiteralExpression(True))

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
        def get_free_identifiers(self) -> frozenset[Identifier]:
            return frozenset({self._variable})  # type: ignore[attr-defined]

        @override
        def evaluate_with_bindings(
            self, bindings: ConstraintBindings
        ) -> ConstraintOutcome:
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
