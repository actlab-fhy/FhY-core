"""Tests core compiler traits."""

from dataclasses import dataclass

import pytest
from fhy_core.trait import (
    HasType,
    HasTypeMixin,
    Verifiable,
    VerifiableMixin,
    VerificationError,
)


@dataclass
class _TypedValue(HasTypeMixin[str]):
    _type: str

    def get_type(self) -> str:
        return self._type


@dataclass
class _VerifiableNode(VerifiableMixin):
    is_valid: bool

    def verify(self) -> None:
        if not self.is_valid:
            raise VerificationError("Node invariant violation.")


def test_has_type_runtime_protocol() -> None:
    """Test `HasType` runtime protocol."""
    value = _TypedValue("i32")
    assert isinstance(value, HasType)


def test_has_type_mixin_contract() -> None:
    """Test `HasTypeMixin` contract."""
    value = _TypedValue("index")
    assert value.get_type() == "index"


def test_verifiable_runtime_protocol() -> None:
    """Test `Verifiable` runtime protocol."""
    node = _VerifiableNode(True)
    assert isinstance(node, Verifiable)


def test_verifiable_mixin_contract() -> None:
    """Test `VerifiableMixin` contract."""
    node = _VerifiableNode(True)
    node.verify()


def test_verifiable_invariant_violation_raises() -> None:
    """Test `VerifiableMixin` raises on invariant violation."""
    node = _VerifiableNode(False)
    with pytest.raises(VerificationError):
        node.verify()
