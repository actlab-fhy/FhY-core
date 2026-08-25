"""Behavioral tests shared by `InSetConstraint` and `NotInSetConstraint`.

Both kinds share an identical surface (constructor signature, ``variable``
property, repr/str rendering, member shapes), so the tests are
parametrized over the constraint factory.
"""

import copy
import dataclasses
import io
import pickle
from collections.abc import Callable
from typing import Any, cast

import pytest

from fhy_core.identifier import Identifier
from fhy_core.symbolic.constraint import (
    Constraint,
    ConstraintError,
    ConstraintOutcome,
    InSetConstraint,
    NotInSetConstraint,
)
from fhy_core.symbolic.constraint import core as constraint_core_module
from fhy_core.traits import FrozenMutationError
from fhy_core.utils.override import override

from .conftest import (
    SET_KINDS,
    HashCollidingMember,
    SerializableEqualHashable,
    mock_identifier,
)

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
def test_set_constraint_is_satisfied_with_bindings(
    factory: SetConstraintFactory,
    member_outcome: bool,
    non_member_outcome: bool,
    values: set[Any],
    member: Any,
    non_member: Any,
) -> None:
    """Test ``is_satisfied_with_bindings`` returns the kind-appropriate polarity."""
    # pylint: disable=too-many-positional-arguments
    x = mock_identifier("x", 0)
    constraint = factory(x, values)

    assert constraint.is_satisfied_with_bindings({x: member}) is member_outcome
    assert constraint.is_satisfied_with_bindings({x: non_member}) is non_member_outcome


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
    x = mock_identifier("x", 0)
    constraint = factory(x, values)

    assert constraint.is_satisfied_with_bindings({x: member}) is member_outcome


@pytest.mark.parametrize("factory", SET_KINDS)
def test_set_constraint_instance_is_not_callable(
    factory: SetConstraintFactory,
) -> None:
    """Test a constraint instance is not callable; the `__call__` sugar is removed."""
    constraint = factory(mock_identifier("x", 0), {1, 2, 3})

    with pytest.raises(TypeError, match="not callable"):
        constraint(2)  # type: ignore[operator]


@pytest.mark.parametrize("factory", SET_KINDS)
def test_set_constraint_variable_property_returns_constructor_argument(
    factory: SetConstraintFactory,
) -> None:
    """Test the ``variable`` property returns the identifier passed to ``__init__``."""
    x = mock_identifier("x", 0)
    constraint = factory(x, {1, 2})

    assert constraint.variable is x


@pytest.mark.parametrize("factory", SET_KINDS)
def test_set_constraint_get_free_identifiers_is_just_the_variable(
    factory: SetConstraintFactory,
) -> None:
    """Test a set constraint's scope is exactly its single variable."""
    x = mock_identifier("x", 0)
    constraint = factory(x, {1, 2})

    assert constraint.get_free_identifiers() == frozenset({x})


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
# Tri-state `evaluate_with_bindings` outcomes
# =============================================================================


@pytest.mark.parametrize(
    "factory, member_outcome, non_member_outcome", _KINDS_WITH_EVALUATE_OUTCOMES
)
def test_set_constraint_evaluate_only_decides_satisfied_or_violated(
    factory: SetConstraintFactory,
    member_outcome: ConstraintOutcome,
    non_member_outcome: ConstraintOutcome,
) -> None:
    """Test a bound set constraint only ever reports SATISFIED or VIOLATED.

    Membership against a concrete, bound value is always decidable, so a
    set constraint never reports ``ConstraintOutcome.UNDECIDED`` once its
    variable is bound to a literal.
    """
    x = mock_identifier("x", 0)
    constraint = factory(x, {1, 2, 3})

    member_result = constraint.evaluate_with_bindings({x: 1})
    non_member_result = constraint.evaluate_with_bindings({x: 4})

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
    x = mock_identifier("x", 0)
    in_set = factory is InSetConstraint
    one_constraint = factory(x, {1})

    assert one_constraint.is_satisfied_with_bindings({x: True}) is not in_set
    assert one_constraint.is_satisfied_with_bindings({x: 1}) is in_set


@pytest.mark.parametrize("factory", SET_KINDS)
def test_set_constraint_distinguishes_one_from_one_float(
    factory: SetConstraintFactory,
) -> None:
    """Test ``1`` and ``1.0`` are stored and compared as distinct members."""
    x = mock_identifier("x", 0)
    in_set = factory is InSetConstraint
    int_constraint = factory(x, {1})

    assert int_constraint.is_satisfied_with_bindings({x: 1.0}) is not in_set
    assert int_constraint.is_satisfied_with_bindings({x: 1}) is in_set


@pytest.mark.parametrize("factory", SET_KINDS)
def test_set_constraint_with_mixed_bool_and_int_stores_both(
    factory: SetConstraintFactory,
) -> None:
    """Test ``[1, True]`` retains both members under type-strict equality."""
    # A list literal is used at the call site; ``{1, True}`` would
    # collapse to ``{1}`` before the constructor sees it.
    x = mock_identifier("x", 0)
    in_set = factory is InSetConstraint
    constraint = factory(x, [1, True])

    assert constraint.is_satisfied_with_bindings({x: True}) is in_set
    assert constraint.is_satisfied_with_bindings({x: 1}) is in_set
    assert constraint.is_satisfied_with_bindings({x: False}) is not in_set
    assert constraint.is_satisfied_with_bindings({x: 0}) is not in_set


@pytest.mark.parametrize("factory", SET_KINDS)
def test_set_constraint_with_nested_tuple_uses_strict_inner_equality(
    factory: SetConstraintFactory,
) -> None:
    """Test type strictness applies to elements inside tuple members."""
    x = mock_identifier("x", 0)
    in_set = factory is InSetConstraint
    constraint = factory(x, [(True, 1)])

    assert constraint.is_satisfied_with_bindings({x: (True, 1)}) is in_set
    assert constraint.is_satisfied_with_bindings({x: (1, 1)}) is not in_set
    assert constraint.is_satisfied_with_bindings({x: (1, True)}) is not in_set


@pytest.mark.parametrize("factory", SET_KINDS)
def test_set_constraint_with_nested_frozenset_uses_strict_inner_equality(
    factory: SetConstraintFactory,
) -> None:
    """Test type strictness applies to elements inside frozenset members."""
    x = mock_identifier("x", 0)
    in_set = factory is InSetConstraint
    constraint = factory(x, [frozenset({True})])

    assert constraint.is_satisfied_with_bindings({x: frozenset({True})}) is in_set
    assert constraint.is_satisfied_with_bindings({x: frozenset({1})}) is not in_set


def test_in_set_constraint_with_nan_member_does_not_satisfy_distinct_nan_instance() -> (
    None
):
    """Test a distinct NaN instance is not detected as a member."""
    x = mock_identifier("x", 0)
    constraint = InSetConstraint(x, {float("nan")})

    assert not constraint.is_satisfied_with_bindings({x: float("nan")})


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
    x = mock_identifier("x", 0)
    in_set = factory is InSetConstraint
    constraint = factory(x, [empty_member])

    assert constraint.is_satisfied_with_bindings({x: empty_member}) is in_set


def test_in_set_constraint_isolates_from_post_construction_mutation() -> None:
    """Test mutating the source collection after construction does not leak in."""
    x = mock_identifier("x", 0)
    src = {1, 2}
    constraint = InSetConstraint(x, src)

    src.add(99)

    assert not constraint.is_satisfied_with_bindings({x: 99})


@pytest.mark.parametrize("factory", SET_KINDS)
def test_set_constraint_is_satisfied_with_bindings_unhashable_value_raises_type_error(
    factory: SetConstraintFactory,
) -> None:
    """Test an unhashable bound value propagates `TypeError`."""
    x = mock_identifier("x", 0)
    constraint = factory(x, {1, 2})

    with pytest.raises(TypeError):
        constraint.is_satisfied_with_bindings({x: {"a": 1}})


@pytest.mark.parametrize("factory", SET_KINDS)
def test_set_constraint_supports_negative_and_zero_numeric_members(
    factory: SetConstraintFactory,
) -> None:
    """Test set constraints accept negative and zero numeric members."""
    x = mock_identifier("x", 0)
    in_set = factory is InSetConstraint
    constraint = factory(x, {-1, 0, -2.5})

    for value in (-1, 0, -2.5):
        assert constraint.is_satisfied_with_bindings({x: value}) is in_set


@pytest.mark.parametrize("factory", SET_KINDS)
@pytest.mark.parametrize("members", ["abc", b"abc", bytearray(b"abc")])
def test_set_constraint_rejects_bare_string_like_members(
    factory: SetConstraintFactory, members: Any
) -> None:
    """Test a bare str/bytes/bytearray is rejected, not split into elements."""
    with pytest.raises(ConstraintError):
        factory(mock_identifier("x", 0), members)


# =============================================================================
# Public field encapsulation (`values`)
# =============================================================================

_SET_KINDS_WITH_FIELD = [
    pytest.param(InSetConstraint, "values", id="in_set"),
    pytest.param(NotInSetConstraint, "values", id="not_in_set"),
]


@pytest.mark.parametrize("factory, field_name", _SET_KINDS_WITH_FIELD)
def test_set_constraint_public_field_holds_the_raw_members(
    factory: SetConstraintFactory, field_name: str
) -> None:
    """Test the constructor-keyword field holds the raw member values."""
    constraint = factory(mock_identifier("x", 0), {1, 2})

    assert set(getattr(constraint, field_name)) == {1, 2}


@pytest.mark.parametrize("factory", SET_KINDS)
def test_set_constraint_accepts_the_unified_values_keyword(
    factory: SetConstraintFactory,
) -> None:
    """Test both kinds share one constructor field name: `values`.

    ``InSetConstraint`` and ``NotInSetConstraint`` are implemented by one
    shared base holding a single ``values`` field, so both accept the
    same keyword regardless of kind (the retired ``valid_values``/
    ``invalid_values`` split no longer exists).
    """
    x = mock_identifier("x", 0)

    constraint = factory(variable=x, values={1, 2})  # type: ignore[call-arg]

    assert set(constraint.values) == {1, 2}  # type: ignore[attr-defined]


# Regression guard for a field that used to store the internal type-strict
# wrapper directly: reading it gave a silently wrong membership answer
# (`1 in constraint.values` was `False` for an actual member `1`,
# because the wrapper's `__eq__`/`__hash__` never matched a raw `1`). Direct
# membership on the field reflects the constructed member set regardless of
# in-set/not-in-set polarity; `is_satisfied_with_bindings` (exercised
# elsewhere) is what differs by kind.
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


@pytest.mark.parametrize("factory, field_name", _SET_KINDS_WITH_FIELD)
def test_set_constraint_members_order_is_independent_of_construction_order(
    factory: SetConstraintFactory, field_name: str
) -> None:
    """Test `members` orders alike for two constraints built in opposite orders.

    Normalization stores members in `frozenset` iteration order, which
    depends on insertion order once members collide on hash (and, for
    `str` members, on the per-process hash seed). The members here
    collide, so the two constraints provably store them in different
    orders and the accessor has to impose the ordering itself rather
    than inherit one that happens to agree.
    """
    x = mock_identifier("x", 0)
    members = [HashCollidingMember(1), HashCollidingMember(2)]
    left = factory(x, list(members))
    right = factory(x, list(reversed(members)))
    assert isinstance(left, (InSetConstraint, NotInSetConstraint))
    assert isinstance(right, (InSetConstraint, NotInSetConstraint))

    assert getattr(left, field_name) != getattr(right, field_name), (
        "the two constraints must store their members in different orders "
        "for this test to say anything about the accessor's ordering"
    )
    assert left.members == right.members


# =============================================================================
# Type-strict member-set storage
# =============================================================================

_MEMBERS = (1, 2, 3)
"""Members shared by the member-set storage tests."""

_ABSENT_PROBE = 99
"""Value deliberately outside `_MEMBERS`, used to probe the negative outcome."""


def _evaluate_bound(constraint: Constraint, value: Any) -> ConstraintOutcome:
    """Return the outcome of binding a set constraint's own variable to ``value``."""
    assert isinstance(constraint, (InSetConstraint, NotInSetConstraint))
    return constraint.evaluate_with_bindings({constraint.variable: value})


_READERS: list[Any] = [
    pytest.param(lambda constraint: _evaluate_bound(constraint, 1), id="evaluate"),
    pytest.param(
        lambda constraint: constraint.is_satisfied_with_bindings(
            {constraint.variable: 1}
        ),
        id="is_satisfied_with_bindings",
    ),
    pytest.param(
        lambda constraint: constraint.convert_to_expression(),
        id="convert_to_expression",
    ),
    pytest.param(repr, id="repr"),
    pytest.param(str, id="str"),
]
"""Every reader of the type-strict member set, as a single-argument callable."""


class _IdentifierByReferencePickler(pickle.Pickler):
    """Pickler that emits identifiers as external references.

    A test constraint's variable is a ``Mock(spec=Identifier)``, which
    pickle refuses to serialize. Handing every identifier to the pickler
    as a persistent reference keeps the constraint itself -- including
    whatever derived state it stores alongside its fields -- on the real
    ``dumps``/``loads`` path.
    """

    referenced: dict[str, Identifier]

    def __init__(self, file: Any, referenced: dict[str, Identifier]) -> None:
        super().__init__(file)
        self.referenced = referenced

    @override
    def persistent_id(self, obj: Any) -> str | None:
        if isinstance(obj, Identifier):
            key = str(id(obj))
            self.referenced[key] = obj
            return key
        return None


class _IdentifierByReferenceUnpickler(pickle.Unpickler):
    """Unpickler resolving the external identifier references by key."""

    referenced: dict[str, Identifier]

    def __init__(self, file: Any, referenced: dict[str, Identifier]) -> None:
        super().__init__(file)
        self.referenced = referenced

    @override
    def persistent_load(self, pid: Any) -> Identifier:
        return self.referenced[pid]


def _round_trip_through_pickle(constraint: Constraint) -> Constraint:
    """Return the constraint after a ``pickle.dumps``/``loads`` round trip."""
    referenced: dict[str, Identifier] = {}
    buffer = io.BytesIO()
    _IdentifierByReferencePickler(buffer, referenced).dump(constraint)
    buffer.seek(0)
    restored = _IdentifierByReferenceUnpickler(buffer, referenced).load()
    assert isinstance(restored, Constraint)
    return restored


def _assert_membership_agrees_with_public_field(
    constraint: Constraint, field_name: str
) -> None:
    """Assert the constraint decides exactly as a fresh one over its public field.

    The type-strict member set is derived state; the public ``values``
    tuple is the source of truth. Any drift between the two shows up as
    a disagreement with a constraint built from that tuple alone.
    """
    assert isinstance(constraint, (InSetConstraint, NotInSetConstraint))
    public_members = tuple(getattr(constraint, field_name))
    reference = type(constraint)(constraint.variable, public_members)  # type: ignore[call-arg]

    for probe in (*public_members, _ABSENT_PROBE):
        assert _evaluate_bound(constraint, probe) is _evaluate_bound(
            reference, probe
        ), f"member set disagrees with {field_name} for probe {probe!r}"


@pytest.mark.parametrize("factory", SET_KINDS)
@pytest.mark.parametrize("read", _READERS)
def test_set_constraint_reader_does_not_rebuild_the_type_strict_member_set(
    monkeypatch: pytest.MonkeyPatch,
    factory: SetConstraintFactory,
    read: Callable[[Constraint], object],
) -> None:
    """Test no reader re-derives the type-strict member set from the raw field.

    The set is built once during construction. Re-deriving it on every
    read turns a constant-time membership check into a full rebuild --
    one wrapper allocation and one hash per stored member, per call --
    and ``__repr__`` additionally feeds the ``ConstraintSystem`` ordering
    key, so the cost multiplies across a system.
    """
    constraint = factory(mock_identifier("x", 0), _MEMBERS)
    rebuild_count = 0
    build_member_set = constraint_core_module._wrap_member_collection

    def counting_build_member_set(values: Any) -> Any:
        nonlocal rebuild_count
        rebuild_count += 1
        return build_member_set(values)

    monkeypatch.setattr(
        constraint_core_module, "_wrap_member_collection", counting_build_member_set
    )

    read(constraint)

    assert rebuild_count == 0


@pytest.mark.parametrize("factory, field_name", _SET_KINDS_WITH_FIELD)
def test_set_constraint_pickle_round_trip_preserves_evaluation(
    factory: SetConstraintFactory, field_name: str
) -> None:
    """Test a pickled-and-restored set constraint still evaluates correctly."""
    constraint = factory(mock_identifier("x", 0), _MEMBERS)

    restored = _round_trip_through_pickle(constraint)

    assert tuple(getattr(restored, field_name)) == tuple(
        getattr(constraint, field_name)
    )
    _assert_membership_agrees_with_public_field(restored, field_name)


@pytest.mark.parametrize("factory, field_name", _SET_KINDS_WITH_FIELD)
@pytest.mark.parametrize(
    "duplicate",
    [pytest.param(copy.copy, id="copy"), pytest.param(copy.deepcopy, id="deepcopy")],
)
def test_set_constraint_copy_preserves_evaluation(
    factory: SetConstraintFactory,
    field_name: str,
    duplicate: Callable[[Constraint], Constraint],
) -> None:
    """Test shallow and deep copies still evaluate against their own member set."""
    constraint = factory(mock_identifier("x", 0), _MEMBERS)

    duplicated = duplicate(constraint)

    assert tuple(getattr(duplicated, field_name)) == tuple(
        getattr(constraint, field_name)
    )
    _assert_membership_agrees_with_public_field(duplicated, field_name)


@pytest.mark.parametrize("factory, field_name", _SET_KINDS_WITH_FIELD)
def test_set_constraint_replace_rederives_the_member_set_from_the_new_values(
    factory: SetConstraintFactory, field_name: str
) -> None:
    """Test ``dataclasses.replace`` decides against the replacement members.

    The derived member set must not survive from the source instance; a
    stale set would keep answering for the members that were replaced.
    """
    constraint = factory(mock_identifier("x", 0), _MEMBERS)

    replaced = cast(
        Constraint, dataclasses.replace(cast(Any, constraint), **{field_name: (7, 8)})
    )

    assert set(getattr(replaced, field_name)) == {7, 8}
    _assert_membership_agrees_with_public_field(replaced, field_name)
    assert _evaluate_bound(replaced, 7) is not _evaluate_bound(replaced, _ABSENT_PROBE)
    assert _evaluate_bound(replaced, 1) is _evaluate_bound(replaced, _ABSENT_PROBE)


@pytest.mark.parametrize("factory, field_name", _SET_KINDS_WITH_FIELD)
def test_set_constraint_member_set_cannot_drift_from_the_public_field(
    factory: SetConstraintFactory, field_name: str
) -> None:
    """Test the public field stays the sole source of truth for membership.

    Neither the public field nor the derived member set is writable, so
    the two cannot be driven apart after construction.
    """
    constraint = factory(mock_identifier("x", 0), _MEMBERS)

    with pytest.raises(FrozenMutationError):
        setattr(constraint, field_name, (7, 8))
    with pytest.raises(FrozenMutationError):
        cast(Any, constraint)._members = frozenset()

    _assert_membership_agrees_with_public_field(constraint, field_name)


@pytest.mark.parametrize("factory", SET_KINDS)
@pytest.mark.parametrize(
    "duplicate",
    [
        pytest.param(_round_trip_through_pickle, id="pickle"),
        pytest.param(copy.copy, id="copy"),
        pytest.param(copy.deepcopy, id="deepcopy"),
    ],
)
def test_set_constraint_equivalence_survives_duplication(
    factory: SetConstraintFactory,
    duplicate: Callable[[Constraint], Constraint],
) -> None:
    """Test structural and alpha equivalence hold between a constraint and its copy."""
    constraint = factory(mock_identifier("x", 0), _MEMBERS)

    duplicated = duplicate(constraint)

    assert constraint.is_structurally_equivalent(duplicated)
    assert duplicated.is_structurally_equivalent(constraint)
    assert constraint.is_alpha_equivalent(duplicated)
