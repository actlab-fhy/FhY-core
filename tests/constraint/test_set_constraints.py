"""Behavioral tests shared by `InSetConstraint` and `NotInSetConstraint`.

Both kinds share an identical surface (constructor signature, ``__call__``
delegation, ``variable`` property, repr/str rendering, member shapes),
so the tests are parametrized over the constraint factory and the
expected ``is_satisfied`` polarity for membership.
"""

from typing import Any, Callable

import pytest

from fhy_core.constraint import Constraint, InSetConstraint, NotInSetConstraint
from fhy_core.identifier import Identifier

from .conftest import SerializableEqualHashable, mock_identifier

SetConstraintFactory = Callable[[Identifier, Any], Constraint]

# (factory, in_set_outcome, not_in_set_outcome, str_marker)
_KINDS = [
    pytest.param(InSetConstraint, True, False, " in {", id="in_set"),
    pytest.param(NotInSetConstraint, False, True, "not in", id="not_in_set"),
]


@pytest.mark.parametrize(
    "factory, member_outcome, non_member_outcome, str_marker", _KINDS
)
@pytest.mark.parametrize(
    "values, member, non_member",
    [
        ({1, 2, 3}, 1, 4),
        ({"a", "b", "c"}, "a", "d"),
        ({True, False}, True, "missing"),
        ({1.5, 2.5}, 1.5, 3.5),
    ],
)
def test_set_constraint_is_satisfied(
    factory: SetConstraintFactory,
    member_outcome: bool,
    non_member_outcome: bool,
    str_marker: str,
    values: set[Any],
    member: Any,
    non_member: Any,
) -> None:
    """Test ``is_satisfied`` returns the kind-appropriate polarity for membership."""
    constraint = factory(mock_identifier("x", 0), values)
    assert constraint.is_satisfied(member) is member_outcome
    assert constraint.is_satisfied(non_member) is non_member_outcome


@pytest.mark.parametrize(
    "factory, member_outcome, non_member_outcome, str_marker", _KINDS
)
@pytest.mark.parametrize(
    "values, member",
    [
        pytest.param({1, "a", True, 2.5}, "a", id="mixed_primitives"),
        pytest.param(
            {SerializableEqualHashable(7)},
            SerializableEqualHashable(7),
            id="serializable_hashable",
        ),
        pytest.param([(1, "a", True)], (1, "a", True), id="tuple_member"),
        pytest.param(
            [frozenset({1, 2, 3})], frozenset({1, 2, 3}), id="frozenset_member"
        ),
    ],
)
def test_set_constraint_supports_member_shapes(
    factory: SetConstraintFactory,
    member_outcome: bool,
    non_member_outcome: bool,
    str_marker: str,
    values: Any,
    member: Any,
) -> None:
    """Test set constraints accept the full range of supported member shapes."""
    constraint = factory(mock_identifier("x", 0), values)
    assert constraint.is_satisfied(member) is member_outcome


@pytest.mark.parametrize(
    "factory, member_outcome, non_member_outcome, str_marker", _KINDS
)
def test_set_constraint_call_delegates_to_is_satisfied(
    factory: SetConstraintFactory,
    member_outcome: bool,
    non_member_outcome: bool,
    str_marker: str,
) -> None:
    """Test ``constraint(value)`` matches ``constraint.is_satisfied(value)``."""
    constraint = factory(mock_identifier("x", 0), {1, 2, 3})
    assert constraint(2) == constraint.is_satisfied(2)


@pytest.mark.parametrize(
    "factory, member_outcome, non_member_outcome, str_marker", _KINDS
)
def test_set_constraint_variable_property_returns_constructor_argument(
    factory: SetConstraintFactory,
    member_outcome: bool,
    non_member_outcome: bool,
    str_marker: str,
) -> None:
    """Test the ``variable`` property returns the identifier passed to ``__init__``."""
    x = mock_identifier("x", 0)
    constraint = factory(x, {1, 2})
    assert constraint.variable is x


@pytest.mark.parametrize(
    "factory, member_outcome, non_member_outcome, str_marker", _KINDS
)
def test_set_constraint_repr_lists_values(
    factory: SetConstraintFactory,
    member_outcome: bool,
    non_member_outcome: bool,
    str_marker: str,
) -> None:
    """Test ``repr`` includes each member's textual form."""
    constraint = factory(mock_identifier("x", 0), {1, 2})
    rendered = repr(constraint)
    assert "1" in rendered
    assert "2" in rendered


@pytest.mark.parametrize(
    "factory, member_outcome, non_member_outcome, str_marker", _KINDS
)
def test_set_constraint_str_renders_membership_marker(
    factory: SetConstraintFactory,
    member_outcome: bool,
    non_member_outcome: bool,
    str_marker: str,
) -> None:
    """Test ``str`` renders the kind-appropriate ``in`` / ``not in`` marker."""
    constraint = factory(mock_identifier("x", 0), {1, 2})
    rendered = str(constraint)
    assert str_marker in rendered
    assert "1" in rendered
    assert "2" in rendered


# =============================================================================
# Adversarial / edge cases
# =============================================================================


def test_in_set_constraint_collapses_bool_and_int_per_python_set_semantics() -> None:
    """Test ``InSetConstraint(x, {1})`` treats ``True`` and ``1.0`` as members.

    Python sets collapse equal-and-equal-hash values; ``1 == True`` and
    ``hash(1) == hash(True)``. The constraint inherits — not overrides —
    that behavior.
    """
    constraint: InSetConstraint[Any] = InSetConstraint(mock_identifier("x", 0), {1})
    assert constraint.is_satisfied(True)
    assert constraint.is_satisfied(1)
    assert constraint.is_satisfied(1.0)


def test_in_set_constraint_with_nan_member_does_not_satisfy_distinct_nan_instance() -> (
    None
):
    """Test a distinct NaN instance is not detected as a member.

    NaN is its own non-equal: ``float('nan') != float('nan')``. Python
    sets only short-circuit on identity (``x is y or x == y``); two
    separate NaN objects miss that fast path and fall through to the
    failing equality check.
    """
    constraint = InSetConstraint(mock_identifier("x", 0), {float("nan")})
    assert not constraint.is_satisfied(float("nan"))


@pytest.mark.parametrize(
    "factory", [InSetConstraint, NotInSetConstraint], ids=["in_set", "not_in_set"]
)
@pytest.mark.parametrize(
    "empty_member",
    [
        pytest.param((), id="empty_tuple"),
        pytest.param(frozenset(), id="empty_frozenset"),
    ],
)
def test_set_constraint_accepts_empty_collection_as_member(
    factory: SetConstraintFactory, empty_member: object
) -> None:
    """Test an empty tuple / frozenset is a valid (and hashable) member."""
    constraint = factory(mock_identifier("x", 0), [empty_member])
    assert constraint.is_satisfied(empty_member) is (factory is InSetConstraint)


def test_in_set_constraint_isolates_from_post_construction_mutation() -> None:
    """Test mutating the source collection after construction does not leak in."""
    src = {1, 2}
    constraint = InSetConstraint(mock_identifier("x", 0), src)
    src.add(99)
    assert not constraint.is_satisfied(99)


@pytest.mark.parametrize(
    "factory", [InSetConstraint, NotInSetConstraint], ids=["in_set", "not_in_set"]
)
def test_set_constraint_is_satisfied_with_unhashable_value_raises_type_error(
    factory: SetConstraintFactory,
) -> None:
    """Test ``is_satisfied`` propagates ``TypeError`` for unhashable values.

    ``value in frozenset`` raises when ``value`` isn't hashable; the
    constraint family does not catch it.
    """
    constraint = factory(mock_identifier("x", 0), {1, 2})
    with pytest.raises(TypeError):
        constraint.is_satisfied({"a": 1})


@pytest.mark.parametrize(
    "factory", [InSetConstraint, NotInSetConstraint], ids=["in_set", "not_in_set"]
)
def test_set_constraint_supports_negative_and_zero_numeric_members(
    factory: SetConstraintFactory,
) -> None:
    """Test set constraints accept negative and zero numeric members."""
    constraint = factory(mock_identifier("x", 0), {-1, 0, -2.5})
    for v in (-1, 0, -2.5):
        assert constraint.is_satisfied(v) is (factory is InSetConstraint)
