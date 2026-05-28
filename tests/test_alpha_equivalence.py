"""Tests the `AlphaEquivalence` trait, `AlphaRenaming`, and the
identifier-keyed mapping helper."""

from dataclasses import dataclass, field

import pytest

from fhy_core.identifier import Identifier
from fhy_core.trait import (
    AlphaEquivalence,
    AlphaEquivalenceMixin,
    AlphaRenaming,
    is_identifier_mapping_alpha_equivalent_under,
)

from .conftest import mock_identifier


@dataclass
class _AlphaLeaf(AlphaEquivalenceMixin):
    """Test-only leaf with an integer payload.

    alpha-equivalence is payload equality; the renaming is ignored. Used as
    the value type for `is_identifier_mapping_alpha_equivalent_under`
    tests and as a generic leaf in binder-recursion tests.

    """

    payload: int

    def is_alpha_equivalent_under(self, other: object, renaming: AlphaRenaming) -> bool:
        del renaming
        return isinstance(other, _AlphaLeaf) and self.payload == other.payload


@dataclass
class _IdRef(AlphaEquivalenceMixin):
    """Test-only identifier-reference node."""

    identifier: Identifier

    def is_alpha_equivalent_under(self, other: object, renaming: AlphaRenaming) -> bool:
        return isinstance(other, _IdRef) and renaming.are_identifiers_alpha_equivalent(
            self.identifier, other.identifier
        )


@dataclass
class _Pair(AlphaEquivalenceMixin):
    """Test-only two-child container that threads the renaming unchanged."""

    left: AlphaEquivalenceMixin
    right: AlphaEquivalenceMixin

    def is_alpha_equivalent_under(self, other: object, renaming: AlphaRenaming) -> bool:
        return (
            isinstance(other, _Pair)
            and self.left.is_alpha_equivalent_under(other.left, renaming)
            and self.right.is_alpha_equivalent_under(other.right, renaming)
        )


@dataclass
class _Binder(AlphaEquivalenceMixin):
    """Test-only single-parameter binder."""

    parameter: Identifier
    body: AlphaEquivalenceMixin

    def is_alpha_equivalent_under(self, other: object, renaming: AlphaRenaming) -> bool:
        if not isinstance(other, _Binder):
            return False
        extended = renaming.extend({self.parameter: other.parameter})
        return self.body.is_alpha_equivalent_under(other.body, extended)


@dataclass
class _MapBag(AlphaEquivalenceMixin):
    """Test-only node carrying an Identifier-keyed mapping of `_AlphaLeaf`s."""

    entries: dict[Identifier, _AlphaLeaf]

    def is_alpha_equivalent_under(self, other: object, renaming: AlphaRenaming) -> bool:
        return isinstance(
            other, _MapBag
        ) and is_identifier_mapping_alpha_equivalent_under(
            self.entries, other.entries, renaming
        )


@dataclass
class _RecordingMixinUser(AlphaEquivalenceMixin):
    """Records every (other, renaming) pair passed to `is_alpha_equivalent_under`."""

    received_others: list[object] = field(default_factory=list)
    received_renamings: list[AlphaRenaming] = field(default_factory=list)
    return_value: bool = True

    def is_alpha_equivalent_under(self, other: object, renaming: AlphaRenaming) -> bool:
        self.received_others.append(other)
        self.received_renamings.append(renaming)
        return self.return_value


class _MixinWithoutImplementation(AlphaEquivalenceMixin):
    """Subclass that fails to implement the abstract method."""


class _ProtocolOnlyAlpha:
    """Implements `is_alpha_equivalent` only --- not `is_alpha_equivalent_under`."""

    def is_alpha_equivalent(self, other: object) -> bool:
        del other
        return True


# ===========================================================================
# AlphaRenaming construction
# ===========================================================================


def test_alpha_renaming_empty_returns_renaming_instance() -> None:
    """Test `AlphaRenaming.empty()` returns an `AlphaRenaming`."""
    renaming = AlphaRenaming.empty()

    assert isinstance(renaming, AlphaRenaming)


def test_alpha_renaming_empty_resolves_to_identity() -> None:
    """Test the empty renaming resolves every identifier to itself."""
    renaming = AlphaRenaming.empty()
    identifier = mock_identifier("x", 1)

    resolved = renaming.resolve(identifier)

    assert resolved == identifier


def test_alpha_renaming_with_free_renaming_stores_mapping() -> None:
    """Test `with_free_renaming` exposes the mapping via `resolve`."""
    free_x = mock_identifier("x", 1)
    free_y = mock_identifier("y", 2)
    renaming = AlphaRenaming.with_free_renaming({free_x: free_y})

    resolved = renaming.resolve(free_x)

    assert resolved == free_y


def test_alpha_renaming_with_free_renaming_empty_matches_empty() -> None:
    """Test `with_free_renaming({})` behaves like `empty()`."""
    identifier = mock_identifier("x", 1)

    renaming = AlphaRenaming.with_free_renaming({})

    assert renaming.resolve(identifier) == identifier


def test_alpha_renaming_with_free_renaming_rejects_non_injective_mapping() -> None:
    """Test `with_free_renaming` raises `ValueError` on non-injective input."""
    a = mock_identifier("a", 1)
    b = mock_identifier("b", 2)
    target = mock_identifier("t", 3)

    with pytest.raises(ValueError, match="injective"):
        AlphaRenaming.with_free_renaming({a: target, b: target})


# ===========================================================================
# AlphaRenaming.extend
# ===========================================================================


def test_alpha_renaming_extend_returns_new_instance() -> None:
    """Test `extend` returns a new `AlphaRenaming` (no in-place mutation)."""
    renaming = AlphaRenaming.empty()
    x = mock_identifier("x", 1)
    y = mock_identifier("y", 2)

    extended = renaming.extend({x: y})

    assert extended is not renaming


def test_alpha_renaming_extend_does_not_mutate_receiver() -> None:
    """Test `extend` leaves the receiver's resolution semantics unchanged."""
    renaming = AlphaRenaming.empty()
    x = mock_identifier("x", 1)
    y = mock_identifier("y", 2)

    renaming.extend({x: y})

    assert renaming.resolve(x) == x


def test_alpha_renaming_extend_empty_frame_resolves_to_identity() -> None:
    """Test an empty `extend({})` frame leaves resolution at identity."""
    renaming = AlphaRenaming.empty().extend({})
    x = mock_identifier("x", 1)

    assert renaming.resolve(x) == x


def test_alpha_renaming_extend_resolves_bound_identifier() -> None:
    """Test resolution finds an identifier in the topmost binder frame."""
    x = mock_identifier("x", 1)
    y = mock_identifier("y", 2)
    renaming = AlphaRenaming.empty().extend({x: y})

    resolved = renaming.resolve(x)

    assert resolved == y


def test_alpha_renaming_extend_inner_frame_shadows_outer_frame() -> None:
    """Test the innermost frame's mapping wins over an outer frame's mapping."""
    x = mock_identifier("x", 1)
    a = mock_identifier("a", 2)
    b = mock_identifier("b", 3)

    renaming = AlphaRenaming.empty().extend({x: a}).extend({x: b})

    assert renaming.resolve(x) == b


def test_alpha_renaming_extend_permits_other_side_value_reuse_across_frames() -> None:
    """Test inner frames may reuse outer frames' other-side values."""
    x = mock_identifier("x", 1)
    y = mock_identifier("y", 2)
    shared_target = mock_identifier("a", 3)

    renaming = (
        AlphaRenaming.empty().extend({x: shared_target}).extend({y: shared_target})
    )

    assert renaming.resolve(x) == shared_target
    assert renaming.resolve(y) == shared_target


def test_alpha_renaming_extend_rejects_non_injective_frame() -> None:
    """Test `extend` raises `ValueError` on a non-injective frame."""
    a = mock_identifier("a", 1)
    b = mock_identifier("b", 2)
    shared_target = mock_identifier("t", 3)

    with pytest.raises(ValueError, match="injective"):
        AlphaRenaming.empty().extend({a: shared_target, b: shared_target})


def test_alpha_renaming_extend_preserves_free_renaming_for_unbound_keys() -> None:
    """Test `extend` does not clobber the free renaming for outside keys."""
    bound_x = mock_identifier("x", 1)
    bound_y = mock_identifier("y", 2)
    free_a = mock_identifier("a", 3)
    free_b = mock_identifier("b", 4)
    renaming = AlphaRenaming.with_free_renaming({free_a: free_b}).extend(
        {bound_x: bound_y}
    )

    assert renaming.resolve(free_a) == free_b


# ===========================================================================
# AlphaRenaming.resolve fallthrough order
# ===========================================================================


def test_alpha_renaming_resolve_falls_back_to_free_renaming_for_unframed_key() -> None:
    """Test `resolve` consults the free renaming when no frame matches."""
    free_a = mock_identifier("a", 1)
    free_b = mock_identifier("b", 2)
    bound_x = mock_identifier("x", 3)
    bound_y = mock_identifier("y", 4)
    renaming = AlphaRenaming.with_free_renaming({free_a: free_b}).extend(
        {bound_x: bound_y}
    )

    assert renaming.resolve(free_a) == free_b


def test_alpha_renaming_resolve_falls_back_to_identity_for_unmapped_key() -> None:
    """Test `resolve` returns the identifier itself when no frame or free
    renaming matches."""
    free_a = mock_identifier("a", 1)
    free_b = mock_identifier("b", 2)
    bound_x = mock_identifier("x", 3)
    bound_y = mock_identifier("y", 4)
    unrelated = mock_identifier("z", 5)
    renaming = AlphaRenaming.with_free_renaming({free_a: free_b}).extend(
        {bound_x: bound_y}
    )

    assert renaming.resolve(unrelated) == unrelated


def test_alpha_renaming_frame_overrides_free_renaming_for_same_key() -> None:
    """Test a frame's mapping wins over a free-renaming entry for the same key."""
    key = mock_identifier("k", 1)
    free_value = mock_identifier("free", 2)
    bound_value = mock_identifier("bound", 3)
    renaming = AlphaRenaming.with_free_renaming({key: free_value}).extend(
        {key: bound_value}
    )

    assert renaming.resolve(key) == bound_value


# ===========================================================================
# AlphaRenaming.are_identifiers_alpha_equivalent
# ===========================================================================


def test_are_identifiers_alpha_equivalent_returns_true_for_resolved_match() -> None:
    """Test `are_identifiers_alpha_equivalent` returns True when resolution
    matches."""
    x = mock_identifier("x", 1)
    y = mock_identifier("y", 2)
    renaming = AlphaRenaming.empty().extend({x: y})

    assert renaming.are_identifiers_alpha_equivalent(x, y)


def test_are_identifiers_alpha_equivalent_returns_false_for_resolution_mismatch() -> (
    None
):
    """Test `are_identifiers_alpha_equivalent` returns False when resolution
    yields a different identifier than `other`."""
    x = mock_identifier("x", 1)
    y = mock_identifier("y", 2)
    z = mock_identifier("z", 3)
    renaming = AlphaRenaming.empty().extend({x: y})

    assert not renaming.are_identifiers_alpha_equivalent(x, z)


def test_are_identifiers_alpha_equivalent_returns_true_for_identity_match() -> None:
    """Test `are_identifiers_alpha_equivalent` returns True under identity
    fallback."""
    x = mock_identifier("x", 1)
    renaming = AlphaRenaming.empty()

    assert renaming.are_identifiers_alpha_equivalent(x, x)


def test_are_identifiers_alpha_equivalent_returns_false_for_distinct_unmapped() -> None:
    """Test `are_identifiers_alpha_equivalent` returns False for two
    distinct, unbound identifiers."""
    x = mock_identifier("x", 1)
    y = mock_identifier("y", 2)
    renaming = AlphaRenaming.empty()

    assert not renaming.are_identifiers_alpha_equivalent(x, y)


def test_are_identifiers_alpha_equivalent_agrees_with_resolve() -> None:
    """Test `are_identifiers_alpha_equivalent(a, b)` iff `resolve(a) == b`
    across all three lookup paths."""
    x = mock_identifier("x", 1)
    y = mock_identifier("y", 2)
    free_a = mock_identifier("a", 3)
    free_b = mock_identifier("b", 4)
    unrelated = mock_identifier("z", 5)
    renaming = AlphaRenaming.with_free_renaming({free_a: free_b}).extend({x: y})

    for self_id in [x, free_a, unrelated]:
        for other_id in [x, y, free_a, free_b, unrelated]:
            assert renaming.are_identifiers_alpha_equivalent(self_id, other_id) == (
                renaming.resolve(self_id) == other_id
            )


# ===========================================================================
# AlphaRenaming value-object semantics
# ===========================================================================


def test_alpha_renaming_equal_by_structure() -> None:
    """Test two `AlphaRenaming`s with the same frames and free renaming are
    equal."""
    x = mock_identifier("x", 1)
    y = mock_identifier("y", 2)
    left = AlphaRenaming.empty().extend({x: y})
    right = AlphaRenaming.empty().extend({x: y})

    assert left == right


def test_alpha_renaming_hashable() -> None:
    """Test `AlphaRenaming` is hashable and equal instances hash alike."""
    x = mock_identifier("x", 1)
    y = mock_identifier("y", 2)
    left = AlphaRenaming.empty().extend({x: y})
    right = AlphaRenaming.empty().extend({x: y})

    assert hash(left) == hash(right)


def test_alpha_renaming_distinguishes_frame_order() -> None:
    """Test two stacks with different frame order are not equal."""
    x = mock_identifier("x", 1)
    y = mock_identifier("y", 2)
    a = mock_identifier("a", 3)
    b = mock_identifier("b", 4)
    left = AlphaRenaming.empty().extend({x: y}).extend({a: b})
    right = AlphaRenaming.empty().extend({a: b}).extend({x: y})

    assert left != right


def test_alpha_renaming_distinguishes_free_renaming() -> None:
    """Test two renamings with different free renamings are not equal."""
    free_a = mock_identifier("a", 1)
    free_b = mock_identifier("b", 2)
    free_c = mock_identifier("c", 3)
    left = AlphaRenaming.with_free_renaming({free_a: free_b})
    right = AlphaRenaming.with_free_renaming({free_a: free_c})

    assert left != right


def test_alpha_renaming_empty_equals_empty() -> None:
    """Test independently-constructed empty renamings compare equal."""
    assert AlphaRenaming.empty() == AlphaRenaming.empty()


# ===========================================================================
# is_identifier_mapping_alpha_equivalent_under
# ===========================================================================


def test_mapping_helper_returns_true_for_empty_inputs() -> None:
    """Test the helper returns True for two empty mappings."""
    assert is_identifier_mapping_alpha_equivalent_under({}, {}, AlphaRenaming.empty())


def test_mapping_helper_returns_true_for_reflexive_input() -> None:
    """Test the helper returns True when comparing a mapping to itself."""
    x = mock_identifier("x", 1)
    mapping = {x: _AlphaLeaf(7)}

    assert is_identifier_mapping_alpha_equivalent_under(
        mapping, mapping, AlphaRenaming.empty()
    )


def test_mapping_helper_returns_false_for_cardinality_mismatch() -> None:
    """Test the helper returns False when the two mappings have different
    sizes."""
    x = mock_identifier("x", 1)
    y = mock_identifier("y", 2)
    self_mapping = {x: _AlphaLeaf(1)}
    other_mapping = {x: _AlphaLeaf(1), y: _AlphaLeaf(2)}

    assert not is_identifier_mapping_alpha_equivalent_under(
        self_mapping, other_mapping, AlphaRenaming.empty()
    )


def test_mapping_helper_returns_true_for_matching_keys_and_values() -> None:
    """Test the helper returns True for matching keys (under identity) and
    matching values."""
    x = mock_identifier("x", 1)
    y = mock_identifier("y", 2)
    self_mapping = {x: _AlphaLeaf(1), y: _AlphaLeaf(2)}
    other_mapping = {x: _AlphaLeaf(1), y: _AlphaLeaf(2)}

    assert is_identifier_mapping_alpha_equivalent_under(
        self_mapping, other_mapping, AlphaRenaming.empty()
    )


def test_mapping_helper_returns_false_for_value_mismatch() -> None:
    """Test the helper returns False when one value pair fails alpha-equivalence."""
    x = mock_identifier("x", 1)
    y = mock_identifier("y", 2)
    self_mapping = {x: _AlphaLeaf(1), y: _AlphaLeaf(2)}
    other_mapping = {x: _AlphaLeaf(1), y: _AlphaLeaf(99)}

    assert not is_identifier_mapping_alpha_equivalent_under(
        self_mapping, other_mapping, AlphaRenaming.empty()
    )


def test_mapping_helper_resolves_keys_via_binder_frame() -> None:
    """Test the helper resolves keys via a binder frame and matches the
    corresponding other-side entry."""
    x = mock_identifier("x", 1)
    y = mock_identifier("y", 2)
    self_mapping = {x: _AlphaLeaf(7)}
    other_mapping = {y: _AlphaLeaf(7)}
    renaming = AlphaRenaming.empty().extend({x: y})

    assert is_identifier_mapping_alpha_equivalent_under(
        self_mapping, other_mapping, renaming
    )


def test_mapping_helper_resolves_keys_via_free_renaming() -> None:
    """Test the helper resolves keys via the free renaming."""
    x = mock_identifier("x", 1)
    xprime = mock_identifier("x", 2)
    self_mapping = {x: _AlphaLeaf(7)}
    other_mapping = {xprime: _AlphaLeaf(7)}
    renaming = AlphaRenaming.with_free_renaming({x: xprime})

    assert is_identifier_mapping_alpha_equivalent_under(
        self_mapping, other_mapping, renaming
    )


def test_mapping_helper_returns_false_for_key_set_mismatch() -> None:
    """Test the helper returns False when resolved keys do not match the
    other mapping's keys."""
    x = mock_identifier("x", 1)
    y = mock_identifier("y", 2)
    z = mock_identifier("z", 3)
    self_mapping = {x: _AlphaLeaf(1), y: _AlphaLeaf(2)}
    other_mapping = {x: _AlphaLeaf(1), z: _AlphaLeaf(2)}

    assert not is_identifier_mapping_alpha_equivalent_under(
        self_mapping, other_mapping, AlphaRenaming.empty()
    )


def test_mapping_helper_returns_false_when_two_keys_resolve_to_same_target() -> None:
    """Test the helper returns False when distinct self-keys resolve to the
    same other-side key (collision)."""
    x = mock_identifier("x", 1)
    y = mock_identifier("y", 2)
    self_mapping = {x: _AlphaLeaf(1), y: _AlphaLeaf(2)}
    other_mapping = {y: _AlphaLeaf(1)}
    renaming = AlphaRenaming.empty().extend({x: y})

    assert not is_identifier_mapping_alpha_equivalent_under(
        self_mapping, other_mapping, renaming
    )


def test_mapping_helper_threads_current_renaming_into_values() -> None:
    """Test the helper passes the same renaming when comparing values."""
    bound_x = mock_identifier("x", 1)
    bound_y = mock_identifier("y", 2)
    key = mock_identifier("k", 3)
    self_mapping = {key: _IdRef(bound_x)}
    other_mapping = {key: _IdRef(bound_y)}
    renaming = AlphaRenaming.empty().extend({bound_x: bound_y})

    assert is_identifier_mapping_alpha_equivalent_under(
        self_mapping, other_mapping, renaming
    )


def test_mapping_helper_does_not_mutate_inputs() -> None:
    """Test the helper does not mutate either input mapping."""
    x = mock_identifier("x", 1)
    y = mock_identifier("y", 2)
    self_mapping = {x: _AlphaLeaf(1), y: _AlphaLeaf(2)}
    other_mapping = {x: _AlphaLeaf(1), y: _AlphaLeaf(2)}
    self_snapshot = dict(self_mapping)
    other_snapshot = dict(other_mapping)

    is_identifier_mapping_alpha_equivalent_under(
        self_mapping, other_mapping, AlphaRenaming.empty()
    )

    assert self_mapping == self_snapshot
    assert other_mapping == other_snapshot


# ===========================================================================
# AlphaEquivalence protocol and AlphaEquivalenceMixin
# ===========================================================================


def test_alpha_equivalence_runtime_protocol_matches_full_implementor() -> None:
    """Test `isinstance(obj, AlphaEquivalence)` is True for a class that
    implements both protocol methods."""
    leaf = _AlphaLeaf(1)

    assert isinstance(leaf, AlphaEquivalence)


def test_alpha_equivalence_runtime_protocol_rejects_partial_implementor() -> None:
    """Test `isinstance(obj, AlphaEquivalence)` is False when only
    `is_alpha_equivalent` is implemented."""
    partial = _ProtocolOnlyAlpha()

    assert not isinstance(partial, AlphaEquivalence)


def test_alpha_equivalence_mixin_default_is_alpha_equivalent_passes_other() -> None:
    """Test `AlphaEquivalenceMixin.is_alpha_equivalent` forwards `other`
    to `is_alpha_equivalent_under`."""
    recorder = _RecordingMixinUser()
    sentinel = _AlphaLeaf(0)

    recorder.is_alpha_equivalent(sentinel)

    assert recorder.received_others == [sentinel]


def test_alpha_equivalence_mixin_default_passes_empty_renaming() -> None:
    """Test the default `is_alpha_equivalent` constructs `AlphaRenaming.empty()`
    when delegating."""
    recorder = _RecordingMixinUser()

    recorder.is_alpha_equivalent(_AlphaLeaf(0))

    assert recorder.received_renamings[0] == AlphaRenaming.empty()


def test_alpha_equivalence_mixin_default_returns_result_of_delegation() -> None:
    """Test the default `is_alpha_equivalent` returns whatever
    `is_alpha_equivalent_under` returned."""
    recorder = _RecordingMixinUser(return_value=False)

    assert recorder.is_alpha_equivalent(_AlphaLeaf(0)) is False


def test_alpha_equivalence_mixin_requires_under_method() -> None:
    """Test `AlphaEquivalenceMixin` cannot be instantiated when the abstract
    method is left unimplemented."""
    with pytest.raises(TypeError):
        _MixinWithoutImplementation()  # type: ignore[abstract]


# ===========================================================================
# Trait contract under binders, using the test-only `_Binder`.
# ===========================================================================


def test_binder_alpha_equivalence_handles_parameter_rename() -> None:
    """Test a single binder with renamed parameters is alpha-equivalent when
    bodies match under the extended renaming."""
    x = mock_identifier("x", 1)
    y = mock_identifier("y", 2)
    left = _Binder(parameter=x, body=_IdRef(x))
    right = _Binder(parameter=y, body=_IdRef(y))

    assert left.is_alpha_equivalent(right)


def test_binder_alpha_equivalence_distinguishes_diverging_bodies() -> None:
    """Test two binders with alpha-distinct bodies are not alpha-equivalent."""
    x = mock_identifier("x", 1)
    y = mock_identifier("y", 2)
    free = mock_identifier("z", 3)
    left = _Binder(parameter=x, body=_IdRef(x))
    right = _Binder(parameter=y, body=_IdRef(free))

    assert not left.is_alpha_equivalent(right)


def test_binder_alpha_equivalence_handles_nested_shadowing_match() -> None:
    """Test `lambda x. lambda x. x` alpha-equivalent to `lambda a. lambda b. b`."""
    x = mock_identifier("x", 1)
    a = mock_identifier("a", 2)
    b = mock_identifier("b", 3)
    left = _Binder(parameter=x, body=_Binder(parameter=x, body=_IdRef(x)))
    right = _Binder(parameter=a, body=_Binder(parameter=b, body=_IdRef(b)))

    assert left.is_alpha_equivalent(right)


def test_binder_alpha_equivalence_handles_nested_shadowing_mismatch() -> None:
    """Test the inner binder's shadow rejects equivalence with the outer body."""
    x = mock_identifier("x", 1)
    a = mock_identifier("a", 2)
    b = mock_identifier("b", 3)
    left = _Binder(parameter=x, body=_Binder(parameter=x, body=_IdRef(x)))
    right = _Binder(parameter=a, body=_Binder(parameter=b, body=_IdRef(a)))

    assert not left.is_alpha_equivalent(right)


def test_binder_alpha_equivalence_rejects_capture() -> None:
    """Test `lambda x. (x, y)` is not alpha-equivalent to `lambda y. (y, y)`."""
    x = mock_identifier("x", 1)
    y = mock_identifier("y", 2)
    left = _Binder(parameter=x, body=_Pair(_IdRef(x), _IdRef(y)))
    right = _Binder(parameter=y, body=_Pair(_IdRef(y), _IdRef(y)))

    assert not left.is_alpha_equivalent(right)


def test_binder_alpha_equivalence_swapped_arguments_under_swapped_binders() -> None:
    """Test swapping binder order and body operand order remains alpha-equivalent."""
    x = mock_identifier("x", 1)
    y = mock_identifier("y", 2)
    left = _Binder(
        parameter=x,
        body=_Binder(parameter=y, body=_Pair(_IdRef(x), _IdRef(y))),
    )
    right = _Binder(
        parameter=y,
        body=_Binder(parameter=x, body=_Pair(_IdRef(y), _IdRef(x))),
    )

    assert left.is_alpha_equivalent(right)


def test_binder_alpha_equivalence_handles_free_renaming_in_body() -> None:
    """Test free identifiers in a binder body use the caller-supplied free
    renaming."""
    x = mock_identifier("x", 1)
    y = mock_identifier("y", 2)
    free_a = mock_identifier("a", 3)
    free_b = mock_identifier("b", 4)
    left = _Binder(parameter=x, body=_Pair(_IdRef(x), _IdRef(free_a)))
    right = _Binder(parameter=y, body=_Pair(_IdRef(y), _IdRef(free_b)))
    renaming = AlphaRenaming.with_free_renaming({free_a: free_b})

    assert left.is_alpha_equivalent_under(right, renaming)


def test_binder_alpha_equivalence_self_bound_to_self_bound() -> None:
    """Test `lambda x. x` alpha-equivalent to `lambda x. x` when both sides use the same
    `Identifier` for the binder."""
    x = mock_identifier("x", 1)
    left = _Binder(parameter=x, body=_IdRef(x))
    right = _Binder(parameter=x, body=_IdRef(x))

    assert left.is_alpha_equivalent(right)


# ===========================================================================
# Cross-type behavior
# ===========================================================================


def test_alpha_equivalence_returns_false_for_unrelated_type() -> None:
    """Test alpha-equivalence returns False (not raises) when `other` is an
    unrelated type."""
    leaf = _AlphaLeaf(1)

    assert not leaf.is_alpha_equivalent("not a leaf")
    assert not leaf.is_alpha_equivalent(42)
    assert not leaf.is_alpha_equivalent(None)


def test_map_bag_alpha_equivalence_through_helper() -> None:
    """Test a node that delegates its mapping comparison to the helper picks
    up the renaming."""
    bound_x = mock_identifier("x", 1)
    bound_y = mock_identifier("y", 2)
    left = _MapBag({bound_x: _AlphaLeaf(5)})
    right = _MapBag({bound_y: _AlphaLeaf(5)})
    renaming = AlphaRenaming.empty().extend({bound_x: bound_y})

    assert left.is_alpha_equivalent_under(right, renaming)


def test_map_bag_alpha_equivalence_detects_value_mismatch() -> None:
    """Test a `_MapBag` reports alpha-inequivalence when a value differs."""
    bound_x = mock_identifier("x", 1)
    bound_y = mock_identifier("y", 2)
    left = _MapBag({bound_x: _AlphaLeaf(5)})
    right = _MapBag({bound_y: _AlphaLeaf(99)})
    renaming = AlphaRenaming.empty().extend({bound_x: bound_y})

    assert not left.is_alpha_equivalent_under(right, renaming)


# ===========================================================================
# Property-style tests over a hand-rolled term generator
# ===========================================================================


_p = mock_identifier("p", 1001)
_q = mock_identifier("q", 1002)
_r = mock_identifier("r", 1003)


@pytest.fixture()
def example_terms() -> list[AlphaEquivalenceMixin]:
    """Yield a small set of terms covering leaves, refs, pairs, and binders."""
    leaf1 = _AlphaLeaf(1)
    leaf2 = _AlphaLeaf(2)
    ref_p = _IdRef(_p)
    ref_q = _IdRef(_q)
    return [
        leaf1,
        leaf2,
        ref_p,
        ref_q,
        _Pair(leaf1, ref_p),
        _Pair(ref_q, leaf2),
        _Binder(parameter=_p, body=ref_p),
        _Binder(parameter=_q, body=ref_q),
        _Binder(parameter=_p, body=_Pair(ref_p, _IdRef(_r))),
        _Binder(
            parameter=_p,
            body=_Binder(parameter=_q, body=_Pair(ref_p, ref_q)),
        ),
    ]


def test_alpha_equivalence_is_reflexive(
    example_terms: list[AlphaEquivalenceMixin],
) -> None:
    """Test alpha-equivalence is reflexive on the example-term set."""
    for term in example_terms:
        assert term.is_alpha_equivalent(term)


def test_alpha_equivalence_is_symmetric(
    example_terms: list[AlphaEquivalenceMixin],
) -> None:
    """Test alpha-equivalence is symmetric on the example-term set."""
    for left in example_terms:
        for right in example_terms:
            assert left.is_alpha_equivalent(right) == right.is_alpha_equivalent(left)


def test_alpha_equivalence_is_transitive(
    example_terms: list[AlphaEquivalenceMixin],
) -> None:
    """Test alpha-equivalence is transitive on the example-term set."""
    triples_exercised = 0
    for left in example_terms:
        for middle in example_terms:
            for right in example_terms:
                if left.is_alpha_equivalent(middle) and middle.is_alpha_equivalent(
                    right
                ):
                    triples_exercised += 1
                    assert left.is_alpha_equivalent(right)

    assert triples_exercised > 0
