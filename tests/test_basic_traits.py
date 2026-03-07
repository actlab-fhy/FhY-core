"""Tests the basic compiler traits."""

from dataclasses import FrozenInstanceError, dataclass

import pytest
from fhy_core.identifier import Identifier
from fhy_core.provenance import Provenance
from fhy_core.trait import (
    Equal,
    EqualMixin,
    Frozen,
    FrozenMixin,
    FrozenMutationError,
    FrozenValidationError,
    HasIdentifier,
    HasIdentifierMixin,
    HasProvenance,
    HasProvenanceMixin,
    Orderable,
    OrderableMixin,
    PartialEqual,
    PartialEqualMixin,
    PartialOrderable,
    PartialOrderableMixin,
)
from frozendict import frozendict

from .conftest import mock_identifier


@dataclass
class _IdentifierCarrier(HasIdentifierMixin):
    _identifier: Identifier

    @property
    def identifier(self) -> Identifier:
        return self._identifier


@dataclass
class _ProvenanceCarrier(HasProvenanceMixin):
    _provenance: Provenance

    @property
    def provenance(self) -> Provenance:
        return self._provenance


@dataclass
class _FrozenNode(FrozenMixin):
    value: int
    items: list[int]

    def __post_init__(self) -> None:
        self.freeze(deep=True)


class _MutablePayload:
    def __init__(self) -> None:
        self.values = [1, 2, 3]


@dataclass(frozen=True)
class _PartialOrderableValue(PartialOrderableMixin):
    value: int

    def __lt__(self, other: object) -> bool:
        if not isinstance(other, _PartialOrderableValue):
            return NotImplemented
        return self.value < other.value


@dataclass(eq=True)
class _AutoPartialEqualValue(PartialEqualMixin):
    value: int


@dataclass(eq=False)
class _NoPartialEqualValue(PartialEqualMixin):
    value: int


@dataclass(eq=True, frozen=True)
class _AutoEqualValue(EqualMixin):
    value: int


class _NoHashEqualValue(EqualMixin):  # noqa: PLW1641
    pass


@dataclass(order=True)
class _AutoPartialOrderableValue(PartialOrderableMixin):
    value: int


@dataclass(order=True)
class _AutoOrderableValue(OrderableMixin):
    value: int


@dataclass
class _FrozenWrapper(FrozenMixin):
    payload: object


@dataclass
class _FrozenMapNode(FrozenMixin):
    payload: dict[str, int]

    def __post_init__(self) -> None:
        self.freeze(deep=True)


@dataclass
class _CyclicFrozenNode(FrozenMixin):
    peer: object | None = None


@dataclass(frozen=True)
class _AutoFrozenPoint(FrozenMixin):
    x: int
    y: int


@dataclass(frozen=True)
class _AutoFrozenPayload(FrozenMixin):
    payload: object


def test_has_identifier_runtime_protocol():
    """Test `HasIdentifier` runtime protocol."""
    carrier = _IdentifierCarrier(mock_identifier("x", 1))
    assert isinstance(carrier, HasIdentifier)


def test_has_provenance_runtime_protocol():
    """Test `HasProvenance` runtime protocol."""
    carrier = _ProvenanceCarrier(Provenance.unknown())
    assert isinstance(carrier, HasProvenance)


def test_partial_orderable_runtime_protocol() -> None:
    """Test `PartialOrderable` runtime protocol."""
    value = _PartialOrderableValue(3)
    assert isinstance(value, PartialOrderable)


def test_partial_orderable_supports_sorting() -> None:
    """Test `PartialOrderableMixin` implementations can be sorted."""
    values = [
        _PartialOrderableValue(3),
        _PartialOrderableValue(1),
        _PartialOrderableValue(2),
    ]
    sorted_values = sorted(values)
    assert [value.value for value in sorted_values] == [1, 2, 3]


def test_partial_equal_runtime_protocol() -> None:
    """Test `PartialEqual` runtime protocol."""
    value = _AutoPartialEqualValue(3)
    assert isinstance(value, PartialEqual)
    assert value.supports_partial_equality is True


def test_partial_equal_detects_eq_false_dataclass() -> None:
    """Test `PartialEqualMixin` detects `dataclass(eq=False)` as unsupported."""
    value = _NoPartialEqualValue(3)
    assert value.supports_partial_equality is False
    assert value.__eq__(_NoPartialEqualValue(3)) is NotImplemented


def test_equal_runtime_protocol() -> None:
    """Test `Equal` runtime protocol."""
    value = _AutoEqualValue(3)
    assert isinstance(value, Equal)
    assert value.supports_equality is True
    assert value.supports_partial_equality is True
    assert value == _AutoEqualValue(3)


def test_equal_mixin_requires_hash_implementation() -> None:
    """Test `EqualMixin` prompts subclasses to implement hash."""
    with pytest.raises(NotImplementedError, match="does not implement __hash__"):
        hash(_NoHashEqualValue())


def test_partial_orderable_detects_ordered_dataclass() -> None:
    """Test `PartialOrderableMixin` detects `dataclass(order=True)` support."""
    left = _AutoPartialOrderableValue(1)
    right = _AutoPartialOrderableValue(2)
    assert left.supports_partial_ordering is True
    assert left < right


def test_orderable_runtime_protocol() -> None:
    """Test `Orderable` runtime protocol."""
    value = _AutoOrderableValue(3)
    assert isinstance(value, Orderable)
    assert value.supports_ordering is True
    assert value.supports_partial_ordering is True


def test_orderable_mixin_defaults_to_total_order() -> None:
    """Test `OrderableMixin` defaults to total-order support."""
    value = _AutoOrderableValue(3)
    assert value.supports_partial_ordering is True
    assert value.supports_ordering is True


def test_identifier_mixin_contract():
    """Test `HasIdentifierMixin` contract."""
    carrier = _IdentifierCarrier(mock_identifier("field", 2))
    assert carrier.identifier.name_hint == "field"
    assert carrier.identifier.id == 2


def test_provenance_mixin_contract():
    """Test `HasProvenanceMixin` contract."""
    carrier = _ProvenanceCarrier(Provenance.unknown())
    assert carrier.provenance == Provenance.unknown()


def test_frozen_runtime_protocol() -> None:
    """Test `Frozen` runtime protocol."""
    node = _FrozenNode(1, [2, 3])
    assert isinstance(node, Frozen)


def test_frozen_blocks_attribute_updates_after_freeze() -> None:
    """Test `FrozenMixin` blocks direct attribute mutation after freezing."""
    node = _FrozenNode(1, [2, 3])
    with pytest.raises(FrozenMutationError):
        node.value = 42


def test_frozen_blocks_attribute_deletion_after_freeze() -> None:
    """Test `FrozenMixin` blocks attribute deletion after freezing."""
    node = _FrozenNode(1, [2, 3])
    with pytest.raises(FrozenMutationError):
        del node.value


def test_frozen_deep_freeze_converts_mutable_builtins() -> None:
    """Test `FrozenMixin.freeze(deep=True)` converts mutable containers."""
    node = _FrozenNode(1, [2, 3])
    assert isinstance(node.items, tuple)


def test_frozen_deep_freeze_converts_dict_to_frozendict() -> None:
    """Test `FrozenMixin.freeze(deep=True)` converts dict to frozendict."""
    node = _FrozenMapNode({"x": 1})
    assert isinstance(node.payload, frozendict)


def test_frozen_assert_frozen_passes_for_write_protected_instance() -> None:
    """Test `FrozenMixin.assert_frozen` passes for valid frozen instances."""
    node = _FrozenNode(1, [2, 3])
    node.assert_frozen(deep=True)


def test_frozen_assert_frozen_fails_when_not_frozen() -> None:
    """Test `FrozenMixin.assert_frozen` fails if instance was not frozen."""
    wrapper = _FrozenWrapper(_MutablePayload())
    with pytest.raises(FrozenValidationError):
        wrapper.assert_frozen()


def test_frozen_strict_check_detects_unknown_mutable_payload() -> None:
    """Test strict deep frozen checks reject unknown mutable nested objects."""
    wrapper = _FrozenWrapper(_MutablePayload())
    wrapper.freeze(deep=False)
    with pytest.raises(FrozenValidationError):
        wrapper.assert_frozen(deep=True, strict=True)


def test_frozen_deep_freeze_handles_self_reference() -> None:
    """Test `FrozenMixin.freeze(deep=True)` handles direct self-references."""
    node = _CyclicFrozenNode()
    node.peer = node
    node.freeze(deep=True)
    assert node.is_frozen
    assert node.peer is node


def test_frozen_deep_freeze_handles_mutually_recursive_nodes() -> None:
    """Test `FrozenMixin.freeze(deep=True)` handles mutual object cycles."""
    left = _CyclicFrozenNode()
    right = _CyclicFrozenNode()
    left.peer = right
    right.peer = left
    left.freeze(deep=True)
    assert left.is_frozen
    assert right.is_frozen
    assert left.peer is right
    assert right.peer is left


def test_native_frozen_dataclass_runtime_protocol() -> None:
    """Test native frozen dataclass instances satisfy the `Frozen` protocol."""
    point = _AutoFrozenPoint(1, 2)
    assert isinstance(point, Frozen)
    assert point.is_frozen is True


def test_native_frozen_dataclass_blocks_mutation() -> None:
    """Test native frozen dataclass blocks direct attribute mutation."""
    point = _AutoFrozenPoint(1, 2)
    with pytest.raises(FrozenInstanceError):
        point.x = 4  # type: ignore[misc]


def test_native_frozen_dataclass_assert_frozen_detects_deep_mutability() -> None:
    """Test native frozen dataclass deep checks reject mutable nested payloads."""
    payload = _AutoFrozenPayload(payload=[1, 2, 3])
    with pytest.raises(FrozenValidationError):
        payload.assert_frozen(deep=True)


def test_native_frozen_dataclass_with_frozen_mixin() -> None:
    """Test native `dataclass(frozen=True)` integrates with `FrozenMixin`."""

    @dataclass(frozen=True)
    class _NativeFrozenPoint(FrozenMixin):
        x: int
        y: int

    point = _NativeFrozenPoint(1, 2)

    assert isinstance(point, Frozen)
    assert point.is_frozen is True
    point.assert_frozen()
    with pytest.raises(FrozenInstanceError):
        setattr(point, "x", 4)
