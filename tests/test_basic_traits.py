"""Tests the basic compiler traits."""

from dataclasses import dataclass

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
