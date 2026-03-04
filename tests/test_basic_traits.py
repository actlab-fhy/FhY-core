"""Tests the basic compiler traits."""

from dataclasses import dataclass

from fhy_core.identifier import Identifier
from fhy_core.provenance import Provenance
from fhy_core.trait import (
    HasIdentifier,
    HasIdentifierMixin,
    HasProvenance,
    HasProvenanceMixin,
)

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
