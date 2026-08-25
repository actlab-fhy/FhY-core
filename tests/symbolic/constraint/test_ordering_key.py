"""Tests for the public `build_constraint_ordering_key`.

`ConstraintSystem` and (per the design doc) the param layer both order
their constraints by this key, so its contract -- constant on
structural-equivalence classes, distinct across kinds and member sets,
and a documented `repr` fallback for third-party subclasses -- is pinned
directly here rather than only observed indirectly through
`ConstraintSystem`'s canonical order.
"""

from dataclasses import dataclass, field

from fhy_core.identifier import Identifier
from fhy_core.symbolic.constraint import (
    Constraint,
    ConstraintBindings,
    ConstraintOutcome,
    EquationConstraint,
    InSetConstraint,
    NotInSetConstraint,
    build_constraint_ordering_key,
    create_constraint_system,
)
from fhy_core.symbolic.expression import (
    BinaryOperation,
    Expression,
    LiteralExpression,
    make_binary_expression,
)
from fhy_core.term import compared_as_reference
from fhy_core.utils.override import override

from .conftest import HashCollidingMember, mock_identifier

# =============================================================================
# Constant on structural-equivalence classes
# =============================================================================


def test_equal_keys_for_in_set_constraints_built_in_different_member_orders() -> None:
    """Test two `InSetConstraint`s over the same members key alike regardless of order.

    The members collide on hash, so the two constraints provably store
    them in different internal orders; the key still has to agree.
    """
    x = mock_identifier("x", 0)
    members = [HashCollidingMember(1), HashCollidingMember(2)]
    left = InSetConstraint(x, list(members))
    right = InSetConstraint(x, list(reversed(members)))

    assert left.values != right.values, (
        "the two constraints must store their members in different orders "
        "for this test to say anything about order independence"
    )
    assert build_constraint_ordering_key(left) == build_constraint_ordering_key(right)


def test_equal_keys_for_equation_constraints_built_from_equivalent_expressions() -> (
    None
):
    """Test two structurally equivalent `EquationConstraint`s key alike."""
    x = mock_identifier("x", 0)
    left = EquationConstraint(make_binary_expression(BinaryOperation.LESS, x, 5))
    right = EquationConstraint(make_binary_expression(BinaryOperation.LESS, x, 5))

    assert left.is_structurally_equivalent(right)
    assert build_constraint_ordering_key(left) == build_constraint_ordering_key(right)


def test_equal_keys_for_constraints_over_independently_built_identifiers() -> None:
    """Test the key reads an identifier through its `id`, not object identity."""
    x1 = mock_identifier("x", 0)
    x2 = mock_identifier("x", 0)
    left = InSetConstraint(x1, {1, 2})
    right = InSetConstraint(x2, {1, 2})

    assert x1 is not x2
    assert build_constraint_ordering_key(left) == build_constraint_ordering_key(right)


# =============================================================================
# Distinctness across kinds and member sets
# =============================================================================


def test_distinct_keys_across_kinds_for_the_same_variable() -> None:
    """Test `InSetConstraint` and `NotInSetConstraint` over the same variable differ."""
    x = mock_identifier("x", 0)
    in_set = InSetConstraint(x, {1, 2})
    not_in_set = NotInSetConstraint(x, {1, 2})

    assert build_constraint_ordering_key(in_set) != build_constraint_ordering_key(
        not_in_set
    )


def test_distinct_keys_across_different_variables() -> None:
    """Test the same member set over two different variables keys apart."""
    x = mock_identifier("x", 0)
    y = mock_identifier("y", 1)
    left = InSetConstraint(x, {1, 2})
    right = InSetConstraint(y, {1, 2})

    assert build_constraint_ordering_key(left) != build_constraint_ordering_key(right)


def test_distinct_keys_across_different_member_sets() -> None:
    """Test a strict superset member set keys apart from the subset."""
    x = mock_identifier("x", 0)
    left = InSetConstraint(x, {1, 2})
    right = InSetConstraint(x, {1, 2, 3})

    assert build_constraint_ordering_key(left) != build_constraint_ordering_key(right)


def test_distinct_keys_for_type_strict_members() -> None:
    """Test members differing only by type (`1` vs `True`) key apart."""
    x = mock_identifier("x", 0)
    left = InSetConstraint(x, [1])
    right = InSetConstraint(x, [True])

    assert build_constraint_ordering_key(left) != build_constraint_ordering_key(right)


def test_distinct_keys_for_different_equation_expressions() -> None:
    """Test two equations over different expressions key apart."""
    x = mock_identifier("x", 0)
    left = EquationConstraint(LiteralExpression(True))
    right = EquationConstraint(make_binary_expression(BinaryOperation.LESS, x, 5))

    assert build_constraint_ordering_key(left) != build_constraint_ordering_key(right)


# =============================================================================
# `repr` fallback for a third-party `Constraint` subclass
# =============================================================================


@dataclass(frozen=True, eq=False)
class _ThirdPartyConstraint(Constraint):
    """A `Constraint` subclass declared outside `fhy_core.symbolic.constraint`.

    Exercises the documented fallback: a subclass this module does not
    recognize keys on its own `repr` rather than on structural fields the
    ordering key function has no way to introspect generically.
    """

    variable: Identifier = field(metadata=compared_as_reference())

    @override
    def get_free_identifiers(self) -> frozenset[Identifier]:
        return frozenset({self.variable})

    @override
    def evaluate_with_bindings(self, bindings: ConstraintBindings) -> ConstraintOutcome:
        return ConstraintOutcome.UNDECIDED

    @override
    def convert_to_expression(self) -> Expression:
        return LiteralExpression(True)

    @override
    def __repr__(self) -> str:
        return f"_ThirdPartyConstraint(marker={self.variable!r})"

    @override
    def __str__(self) -> str:
        return "_ThirdPartyConstraint"


def test_third_party_subclass_ordering_key_falls_back_to_repr() -> None:
    """Test the ordering key for an unrecognized subclass is derived from `repr`."""
    x = mock_identifier("x", 0)
    constraint = _ThirdPartyConstraint(x)

    key = build_constraint_ordering_key(constraint)

    assert repr(constraint) in key


def test_third_party_subclass_ordering_key_distinguishes_distinct_instances() -> None:
    """Test two third-party instances with different `repr`s key apart."""
    x = mock_identifier("x", 0)
    y = mock_identifier("y", 1)
    left = _ThirdPartyConstraint(x)
    right = _ThirdPartyConstraint(y)

    assert build_constraint_ordering_key(left) != build_constraint_ordering_key(right)


# =============================================================================
# `ConstraintSystem` orders its members by exactly this key
# =============================================================================


def test_constraint_system_member_order_matches_sorting_by_the_public_key() -> None:
    """Test `ConstraintSystem.constraints` equals members sorted by the public key."""
    x = mock_identifier("x", 0)
    y = mock_identifier("y", 1)
    z = mock_identifier("z", 2)
    members = (
        NotInSetConstraint(z, {5, 6}),
        EquationConstraint(make_binary_expression(BinaryOperation.LESS, x, y)),
        InSetConstraint(x, {1, 2}),
    )

    system = create_constraint_system(*members)

    assert list(system.constraints) == sorted(
        members, key=build_constraint_ordering_key
    )


def test_constraint_system_member_order_matches_the_key_for_a_third_party_kind() -> (
    None
):
    """Test a system with a third-party `Constraint` member still sorts by the key."""
    x = mock_identifier("x", 0)
    y = mock_identifier("y", 1)
    members = (_ThirdPartyConstraint(y), InSetConstraint(x, {1, 2}))

    system = create_constraint_system(*members)

    assert list(system.constraints) == sorted(
        members, key=build_constraint_ordering_key
    )
