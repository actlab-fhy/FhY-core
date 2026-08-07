"""Behavioral tests shared by `InSetConstraint` and `NotInSetConstraint`.

Both kinds share an identical surface (constructor signature, ``__call__``
delegation, ``variable`` property, repr/str rendering, member shapes),
so the tests are parametrized over the constraint factory.
"""

from collections.abc import Callable
from typing import Any

import pytest

from fhy_core.identifier import Identifier
from fhy_core.symbolic.constraint import (
    Constraint,
    ConstraintError,
    ConstraintOutcome,
    InSetConstraint,
    NotInSetConstraint,
)

from .conftest import SET_KINDS, SerializableEqualHashable, mock_identifier

SetConstraintFactory = Callable[[Identifier, Any], Constraint]

_KINDS_WITH_OUTCOMES = [
    pytest.param(InSetConstraint, True, False, id="in_set"),
    pytest.param(NotInSetConstraint, False, True, id="not_in_set"),
]

_KINDS_WITH_EVALUATE_OUTCOMES = [
    pytest.param(
        InSetConstraint,
        ConstraintOutcome.SATISFIED,
        ConstraintOutcome.VIOLATED,
        id="in_set",
    ),
    pytest.param(
        NotInSetConstraint,
        ConstraintOutcome.VIOLATED,
        ConstraintOutcome.SATISFIED,
        id="not_in_set",
    ),
]

_KINDS_WITH_STR_MARKER = [
    pytest.param(InSetConstraint, " in {", id="in_set"),
    pytest.param(NotInSetConstraint, "not in", id="not_in_set"),
]


@pytest.mark.parametrize(
    "factory, member_outcome, non_member_outcome", _KINDS_WITH_OUTCOMES
)
@pytest.mark.parametrize(
    "values, member, non_member",
    [
        pytest.param({1, 2, 3}, 1, 4, id="ints"),
        pytest.param({"a", "b", "c"}, "a", "d", id="strings"),
        pytest.param({True, False}, True, "missing", id="bools"),
        pytest.param({1.5, 2.5}, 1.5, 3.5, id="floats"),
    ],
)
def test_set_constraint_is_satisfied(
    factory: SetConstraintFactory,
    member_outcome: bool,
    non_member_outcome: bool,
    values: set[Any],
    member: Any,
    non_member: Any,
) -> None:
    """Test ``is_satisfied`` returns the kind-appropriate polarity for membership."""
    # pylint: disable=too-many-positional-arguments
    constraint = factory(mock_identifier("x", 0), values)

    assert constraint.is_satisfied(member) is member_outcome
    assert constraint.is_satisfied(non_member) is non_member_outcome


@pytest.mark.parametrize(
    "factory, member_outcome, non_member_outcome", _KINDS_WITH_OUTCOMES
)
@pytest.mark.parametrize(
    "values, member",
    [
        pytest.param({1, "a", 2.5}, "a", id="mixed_primitives"),
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
    values: Any,
    member: Any,
) -> None:
    """Test set constraints accept the full range of supported member shapes."""
    constraint = factory(mock_identifier("x", 0), values)

    assert constraint.is_satisfied(member) is member_outcome


@pytest.mark.parametrize("factory", SET_KINDS)
def test_set_constraint_call_delegates_to_is_satisfied(
    factory: SetConstraintFactory,
) -> None:
    """Test ``constraint(value)`` matches ``constraint.is_satisfied(value)``."""
    constraint = factory(mock_identifier("x", 0), {1, 2, 3})

    assert constraint(2) == constraint.is_satisfied(2)


@pytest.mark.parametrize("factory", SET_KINDS)
def test_set_constraint_variable_property_returns_constructor_argument(
    factory: SetConstraintFactory,
) -> None:
    """Test the ``variable`` property returns the identifier passed to ``__init__``."""
    x = mock_identifier("x", 0)
    constraint = factory(x, {1, 2})

    assert constraint.variable is x


@pytest.mark.parametrize("factory", SET_KINDS)
def test_set_constraint_repr_lists_values(
    factory: SetConstraintFactory,
) -> None:
    """Test ``repr`` includes each member's textual form."""
    constraint = factory(mock_identifier("x", 0), {1, 2})

    rendered = repr(constraint)

    assert "1" in rendered
    assert "2" in rendered


@pytest.mark.parametrize("factory", SET_KINDS)
def test_set_constraint_repr_includes_class_name(
    factory: SetConstraintFactory,
) -> None:
    """Test ``repr`` includes the concrete constraint class name."""
    constraint = factory(mock_identifier("x", 0), {1, 2})

    rendered = repr(constraint)

    assert type(constraint).__name__ in rendered


@pytest.mark.parametrize("factory", SET_KINDS)
def test_set_constraint_repr_includes_variable(
    factory: SetConstraintFactory,
) -> None:
    """Test ``repr`` includes a representation of the constrained variable."""
    x = mock_identifier("x", 0)
    constraint = factory(x, {1, 2})

    rendered = repr(constraint)

    assert repr(x) in rendered


@pytest.mark.parametrize("factory", SET_KINDS)
def test_set_constraint_repr_distinguishes_string_from_numeric_members(
    factory: SetConstraintFactory,
) -> None:
    """Test ``repr`` renders a ``str`` member distinguishably from an ``int`` member.

    Membership is type-strict, so ``{"5"}`` and ``{5}`` are different
    constraints; rendering both members bare would make the two textual
    forms indistinguishable.
    """
    x = mock_identifier("x", 0)

    string_member = repr(factory(x, {"5"}))
    integer_member = repr(factory(x, {5}))

    assert string_member != integer_member


@pytest.mark.parametrize("factory, str_marker", _KINDS_WITH_STR_MARKER)
def test_set_constraint_str_renders_membership_marker(
    factory: SetConstraintFactory,
    str_marker: str,
) -> None:
    """Test ``str`` renders the kind-appropriate ``in`` / ``not in`` marker."""
    constraint = factory(mock_identifier("x", 0), {1, 2})

    rendered = str(constraint)

    assert str_marker in rendered
    assert "1" in rendered
    assert "2" in rendered


# =============================================================================
# Tri-state `evaluate` outcomes
# =============================================================================


@pytest.mark.parametrize(
    "factory, member_outcome, non_member_outcome", _KINDS_WITH_EVALUATE_OUTCOMES
)
def test_set_constraint_evaluate_only_decides_satisfied_or_violated(
    factory: SetConstraintFactory,
    member_outcome: ConstraintOutcome,
    non_member_outcome: ConstraintOutcome,
) -> None:
    """Test set constraints only ever report SATISFIED or VIOLATED.

    Membership is always decidable, so a set constraint never reports
    ``ConstraintOutcome.UNDECIDED``.
    """
    constraint = factory(mock_identifier("x", 0), {1, 2, 3})

    member_result = constraint.evaluate(1)
    non_member_result = constraint.evaluate(4)

    assert member_result is member_outcome
    assert non_member_result is non_member_outcome
    assert ConstraintOutcome.UNDECIDED not in (member_result, non_member_result)


# =============================================================================
# Adversarial / edge cases
# =============================================================================


@pytest.mark.parametrize("factory", SET_KINDS)
def test_set_constraint_distinguishes_true_from_one(
    factory: SetConstraintFactory,
) -> None:
    """Test ``True`` and ``1`` are stored and compared as distinct members."""
    in_set = factory is InSetConstraint
    one_constraint = factory(mock_identifier("x", 0), {1})

    assert one_constraint.is_satisfied(True) is not in_set
    assert one_constraint.is_satisfied(1) is in_set


@pytest.mark.parametrize("factory", SET_KINDS)
def test_set_constraint_distinguishes_one_from_one_float(
    factory: SetConstraintFactory,
) -> None:
    """Test ``1`` and ``1.0`` are stored and compared as distinct members."""
    in_set = factory is InSetConstraint
    int_constraint = factory(mock_identifier("x", 0), {1})

    assert int_constraint.is_satisfied(1.0) is not in_set
    assert int_constraint.is_satisfied(1) is in_set


@pytest.mark.parametrize("factory", SET_KINDS)
def test_set_constraint_with_mixed_bool_and_int_stores_both(
    factory: SetConstraintFactory,
) -> None:
    """Test ``[1, True]`` retains both members under type-strict equality."""
    # A list literal is used at the call site; ``{1, True}`` would
    # collapse to ``{1}`` before the constructor sees it.
    in_set = factory is InSetConstraint
    constraint = factory(mock_identifier("x", 0), [1, True])

    assert constraint.is_satisfied(True) is in_set
    assert constraint.is_satisfied(1) is in_set
    assert constraint.is_satisfied(False) is not in_set
    assert constraint.is_satisfied(0) is not in_set


@pytest.mark.parametrize("factory", SET_KINDS)
def test_set_constraint_with_nested_tuple_uses_strict_inner_equality(
    factory: SetConstraintFactory,
) -> None:
    """Test type strictness applies to elements inside tuple members."""
    in_set = factory is InSetConstraint
    constraint = factory(mock_identifier("x", 0), [(True, 1)])

    assert constraint.is_satisfied((True, 1)) is in_set
    assert constraint.is_satisfied((1, 1)) is not in_set
    assert constraint.is_satisfied((1, True)) is not in_set


@pytest.mark.parametrize("factory", SET_KINDS)
def test_set_constraint_with_nested_frozenset_uses_strict_inner_equality(
    factory: SetConstraintFactory,
) -> None:
    """Test type strictness applies to elements inside frozenset members."""
    in_set = factory is InSetConstraint
    constraint = factory(mock_identifier("x", 0), [frozenset({True})])

    assert constraint.is_satisfied(frozenset({True})) is in_set
    assert constraint.is_satisfied(frozenset({1})) is not in_set


def test_in_set_constraint_with_nan_member_does_not_satisfy_distinct_nan_instance() -> (
    None
):
    """Test a distinct NaN instance is not detected as a member."""
    constraint = InSetConstraint(mock_identifier("x", 0), {float("nan")})

    assert not constraint.is_satisfied(float("nan"))


@pytest.mark.parametrize("factory", SET_KINDS)
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
    in_set = factory is InSetConstraint
    constraint = factory(mock_identifier("x", 0), [empty_member])

    assert constraint.is_satisfied(empty_member) is in_set


def test_in_set_constraint_isolates_from_post_construction_mutation() -> None:
    """Test mutating the source collection after construction does not leak in."""
    src = {1, 2}
    constraint = InSetConstraint(mock_identifier("x", 0), src)

    src.add(99)

    assert not constraint.is_satisfied(99)


@pytest.mark.parametrize("factory", SET_KINDS)
def test_set_constraint_is_satisfied_with_unhashable_value_raises_type_error(
    factory: SetConstraintFactory,
) -> None:
    """Test ``is_satisfied`` propagates ``TypeError`` for unhashable values."""
    constraint = factory(mock_identifier("x", 0), {1, 2})

    with pytest.raises(TypeError):
        constraint.is_satisfied({"a": 1})


@pytest.mark.parametrize("factory", SET_KINDS)
def test_set_constraint_supports_negative_and_zero_numeric_members(
    factory: SetConstraintFactory,
) -> None:
    """Test set constraints accept negative and zero numeric members."""
    in_set = factory is InSetConstraint
    constraint = factory(mock_identifier("x", 0), {-1, 0, -2.5})

    for value in (-1, 0, -2.5):
        assert constraint.is_satisfied(value) is in_set


@pytest.mark.parametrize("factory", SET_KINDS)
@pytest.mark.parametrize("members", ["abc", b"abc", bytearray(b"abc")])
def test_set_constraint_rejects_bare_string_like_members(
    factory: SetConstraintFactory, members: Any
) -> None:
    """Test a bare str/bytes/bytearray is rejected, not split into elements."""
    with pytest.raises(ConstraintError):
        factory(mock_identifier("x", 0), members)


# =============================================================================
# Public field encapsulation (`valid_values` / `invalid_values`)
# =============================================================================

_SET_KINDS_WITH_FIELD = [
    pytest.param(InSetConstraint, "valid_values", id="in_set"),
    pytest.param(NotInSetConstraint, "invalid_values", id="not_in_set"),
]


@pytest.mark.parametrize("factory, field_name", _SET_KINDS_WITH_FIELD)
def test_set_constraint_public_field_holds_the_raw_members(
    factory: SetConstraintFactory, field_name: str
) -> None:
    """Test the constructor-keyword field holds the raw member values."""
    constraint = factory(mock_identifier("x", 0), {1, 2})

    assert set(getattr(constraint, field_name)) == {1, 2}


# Regression guard for a field that used to store the internal type-strict
# wrapper directly: reading it gave a silently wrong membership answer
# (`1 in constraint.valid_values` was `False` for an actual member `1`,
# because the wrapper's `__eq__`/`__hash__` never matched a raw `1`). Direct
# membership on the field reflects the constructed member set regardless of
# in-set/not-in-set polarity; `is_satisfied` (exercised elsewhere) is what
# differs by kind.
@pytest.mark.parametrize("factory, field_name", _SET_KINDS_WITH_FIELD)
def test_set_constraint_public_field_direct_membership_reflects_true_membership(
    factory: SetConstraintFactory, field_name: str
) -> None:
    """Test membership on the public field matches what was constructed."""
    constraint = factory(mock_identifier("x", 0), {1, 2})

    assert 1 in getattr(constraint, field_name)
    assert 2 in getattr(constraint, field_name)
    assert 99 not in getattr(constraint, field_name)


@pytest.mark.parametrize("factory, field_name", _SET_KINDS_WITH_FIELD)
def test_set_constraint_public_field_never_yields_internal_wrapper_type(
    factory: SetConstraintFactory, field_name: str
) -> None:
    """Test every element of the public field is a plain member type, not a wrapper."""
    constraint = factory(mock_identifier("x", 0), {1, 2})

    for member in getattr(constraint, field_name):
        assert type(member) in (int, float, str, bool)


@pytest.mark.parametrize("factory, field_name", _SET_KINDS_WITH_FIELD)
def test_set_constraint_public_field_matches_members_property(
    factory: SetConstraintFactory, field_name: str
) -> None:
    """Test the public field and the `members` property agree on content."""
    constraint = factory(mock_identifier("x", 0), {1, 2, 3})
    assert isinstance(constraint, (InSetConstraint, NotInSetConstraint))

    assert set(getattr(constraint, field_name)) == set(constraint.members)
