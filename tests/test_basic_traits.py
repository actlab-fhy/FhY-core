"""Tests the basic compiler traits."""

from dataclasses import FrozenInstanceError, dataclass

import pytest
from fhy_core.identifier import Identifier
from fhy_core.provenance import Provenance
from fhy_core.trait import (
    Frozen,
    FrozenMixin,
    FrozenMutationError,
    FrozenValidationError,
    HasIdentifier,
    HasIdentifierMixin,
    HasProvenance,
    HasProvenanceMixin,
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
